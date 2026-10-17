"""Metrics + ablation scoring (plan Section 13.2). Offline only (§20)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from consilium.context.facts import F
from consilium.core.clock import FakeClock
from consilium.eval import ablation as ab
from consilium.eval import runners as r
from consilium.eval.datasets import SMOKE_MANIFEST_PATH, LoadedManifest, load_manifest
from consilium.eval.diversity import (
    HashedNgramEmbedder,
    role_specific_facts,
    s_inter,
    unique_fact_coverage,
)
from consilium.eval.fixtures import smoke_manifest, synthetic_draft, synthetic_subreport
from consilium.eval.metrics import cost_usd, percentile, score_outcome
from consilium.schemas.agents import Finding, Recommendation, SubReport
from consilium.schemas.enums import Confidence, Role

EMB = HashedNgramEmbedder()
RUNNER = r.default_runner()
PRICING = {"m": {"input": 1.0, "output": 2.0}}


def _report(role: Role, text: str, refs: tuple[str, ...] = (F.P_TIER,)) -> SubReport:
    return SubReport(
        role=role,
        findings=(Finding(text=text, evidence_refs=refs, confidence=Confidence.MODERATE),),
        recommendations=(Recommendation(text="See clinician.", triggered_by=(F.P_TIER,)),),
    )


@pytest.fixture(scope="module")
def scored() -> tuple[r.RunRecord, LoadedManifest]:
    plan = r.smoke_plan(samples=3)
    record = asyncio.run(
        r.run_ablation(plan, RUNNER, clock=FakeClock(current=datetime(2026, 1, 1, tzinfo=UTC)))
    )
    return (
        ab.score_run(record, plan.loaded, thresholds=RUNNER.thresholds, routing=RUNNER.routing),
        plan.loaded,
    )


# --- diversity --------------------------------------------------------------
def test_embedder_is_deterministic_and_normalised() -> None:
    a = EMB.embed(["Vertical cup-to-disc ratio measured.", "Medication safety lens."])
    b = EMB.embed(["Vertical cup-to-disc ratio measured.", "Medication safety lens."])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert EMB.embed([""])[0].sum() == 0.0  # empty text -> zero vector, no NaN


def test_s_inter_is_similarity_not_distance() -> None:
    same = [_report(Role.OPHTHALMOLOGIST, "tier frames attention")] * 2
    assert s_inter(same, EMB) == pytest.approx(1.0)
    apart = [
        _report(Role.OPHTHALMOLOGIST, "alpha beta gamma"),
        _report(Role.PHARMACIST, "delta epsilon zeta"),
    ]
    sim = s_inter(apart, EMB)
    assert sim is not None and sim < 0.5  # shared recommendation only
    assert s_inter(same[:1], EMB) is None


def test_role_specific_facts_follow_the_real_routing() -> None:
    ctx = r.global_context(smoke_manifest().cases[0])
    spec = role_specific_facts(ctx, RUNNER.routing)
    assert spec[Role.OPTOMETRIST] == frozenset()  # optometrist allowlist ⊂ ophthalmologist's
    assert {F.MEDS, F.INTOL, F.COMORB} <= spec[Role.PHARMACIST]
    assert F.CDR_A in spec[Role.OPHTHALMOLOGIST] and F.CDR_V not in spec[Role.OPHTHALMOLOGIST]
    cov = unique_fact_coverage([_report(Role.PHARMACIST, "x", (F.MEDS,))], ctx, RUNNER.routing)
    assert cov[Role.PHARMACIST] == pytest.approx(1 / len(spec[Role.PHARMACIST]))


# --- per-outcome metrics ----------------------------------------------------
def _outcome(stage: r.Stage, case_idx: int = 4, **draft_update: object) -> r.StageOutcome:
    case = smoke_manifest().cases[case_idx]
    ctx = r.global_context(case)
    contexts = RUNNER.role_contexts(stage, ctx)
    reports = (
        {role: synthetic_subreport(role, c, case, 1, 0) for role, c in contexts.items()}
        if contexts
        else {}
    )
    draft = synthetic_draft(ctx, reports, case, 1, 0, slip=False).model_copy(update=draft_update)
    return r.StageOutcome(
        case_id=case.case_id,
        stage=stage,
        sample=0,
        status="ok",
        sub_reports=tuple(reports.values()),
        draft=draft,
        wall_ms=1.0,
    )


def _score(outcome: r.StageOutcome, case_idx: int = 4) -> dict[str, float | None]:
    case = smoke_manifest().cases[case_idx]
    return score_outcome(
        outcome,
        case,
        r.global_context(case),
        thresholds=RUNNER.thresholds,
        routing=RUNNER.routing,
        embedder=EMB,
        pricing=PRICING,
    ).values


def test_clean_routed_draft_scores_clean() -> None:
    v = _score(_outcome(r.Stage.S3))
    assert v["unsupported_claim_rate"] == 0.0
    assert v["omission_rate"] == 0.0
    assert v["unsafe_language"] == 0.0
    assert v["out_of_scope_rate"] == 0.0
    assert v["minority_preservation"] == 1.0  # case index 4 is curated
    assert v["s_inter"] is not None and v["unverified"] is None


def test_faults_move_the_right_metric() -> None:
    clean = _outcome(r.Stage.S3)
    assert clean.draft is not None
    claims = clean.draft.claims
    slipped = claims[0].model_copy(update={"numeric_value": 0.99})
    v = _score(_outcome(r.Stage.S3, claims=(slipped, *claims[1:])))
    assert v["unsupported_claim_rate"] == pytest.approx(1 / len(claims))
    # drop the only claim citing laterality (vCDR is also cited by findings)
    v = _score(
        _outcome(r.Stage.S3, claims=tuple(c for c in claims if F.LAT not in c.evidence_refs))
    )
    assert v["omission_rate"] == pytest.approx(1 / 3)  # no caveats in case 4
    v = _score(_outcome(r.Stage.S3, disagreements=()))
    assert v["minority_preservation"] == 0.0


def test_shared_context_attribution_is_out_of_scope_not_unsupported() -> None:
    v = _score(_outcome(r.Stage.S2))
    assert v["out_of_scope_rate"] is not None and v["out_of_scope_rate"] > 0
    assert v["unsupported_claim_rate"] == 0.0


def test_s1_has_no_dissent_or_diversity_metric() -> None:
    v = _score(_outcome(r.Stage.S1))
    assert v["minority_preservation"] is None and v["s_inter"] is None


def test_failed_outcome_only_counts_as_failure() -> None:
    o = r.StageOutcome(
        case_id="smoke-005", stage=r.Stage.S3, sample=0, status="failed", wall_ms=0.0
    )
    case = smoke_manifest().cases[4]
    row = score_outcome(
        o,
        case,
        r.global_context(case),
        thresholds=RUNNER.thresholds,
        routing=RUNNER.routing,
        embedder=EMB,
        pricing=PRICING,
    )
    assert row.failed and row.values["unsupported_claim_rate"] is None


def test_cost_is_null_for_unpriced_models_never_guessed() -> None:
    call = r.LLMCallRecord(
        agent="director",
        model="m",
        prompt_version="v1",
        input_tokens=1_000_000,
        output_tokens=500_000,
        latency_ms=1.0,
        attempts=1,
    )
    o = r.StageOutcome(
        case_id="c", stage=r.Stage.S3, sample=0, status="ok", llm_calls=(call,), wall_ms=0
    )
    assert cost_usd(o, PRICING) == pytest.approx(2.0)
    assert cost_usd(o, {}) is None


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([5.0], 95) == 5.0


# --- ablation ---------------------------------------------------------------
def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    vals = [0.1, 0.4, 0.2, 0.9, 0.5]
    a = ab.bootstrap_ci(vals, resamples=500, confidence=0.95, rng=np.random.default_rng(1))
    b = ab.bootstrap_ci(vals, resamples=500, confidence=0.95, rng=np.random.default_rng(1))
    assert a == b and a is not None
    assert a[0] <= float(np.mean(vals)) <= a[1]
    assert ab.bootstrap_ci([], resamples=10, confidence=0.95, rng=np.random.default_rng()) is None
    one = ab.bootstrap_ci([0.3], resamples=10, confidence=0.95, rng=np.random.default_rng())
    assert one == [0.3, 0.3]


def test_eval_config_loads() -> None:
    cfg = ab.load_eval_config()
    assert cfg.resamples >= 1000 and cfg.confidence == 0.95
    assert (r.Stage.S3, r.Stage.S4) in cfg.paired


def test_score_run_produces_cis_and_paired_differences(
    scored: tuple[r.RunRecord, LoadedManifest],
) -> None:
    record, _ = scored
    m = record.metrics
    per_stage = m["per_stage"]
    assert isinstance(per_stage, dict) and set(per_stage) == {"S1", "S2", "S3", "S4"}
    s4 = per_stage["S4"]
    assert isinstance(s4, dict)
    unsupported = s4["unsupported_claim_rate"]
    assert isinstance(unsupported, dict) and unsupported["ci95"] is not None
    assert unsupported["n_cases"] == len(smoke_manifest().cases)
    paired = m["paired"]
    assert isinstance(paired, dict) and set(paired) == {"S2-S1", "S3-S2", "S4-S3", "S4-S1"}
    diff = paired["S4-S3"]
    assert isinstance(diff, dict)
    cell = diff["unsupported_claim_rate"]
    assert isinstance(cell, dict)
    per_case = cell["per_case"]
    assert isinstance(per_case, dict) and len(per_case) == len(smoke_manifest().cases)
    mean_diff = cell["mean_diff"]
    assert isinstance(mean_diff, float) and mean_diff < 0  # S4 repairs the seeded slips
    rows = m["rows"]
    assert isinstance(rows, list) and len(rows) == len(record.outcomes)
    instrument = m["instrument"]
    assert isinstance(instrument, dict)
    assert instrument["embedding_model"] == "hashed-ngram-tf@v1"
    assert "SIMILARITY" in str(instrument["s_inter"])


def test_score_run_refuses_a_different_case_set(
    scored: tuple[r.RunRecord, LoadedManifest], tmp_path: Path
) -> None:
    record, loaded = scored
    other = tmp_path / "m.json"
    other.write_bytes(SMOKE_MANIFEST_PATH.read_bytes() + b" ")
    with pytest.raises(ValueError, match="dataset_version"):
        ab.score_run(
            record, load_manifest(other), thresholds=RUNNER.thresholds, routing=RUNNER.routing
        )
    assert loaded.dataset_version == record.provenance.dataset_version


def test_cli_smoke_stores_a_scored_run(tmp_path: Path) -> None:
    run_id = ab.main(["--out", str(tmp_path), "smoke", "--samples", "1"])
    record = r.RunStore(tmp_path).load(run_id)
    assert record.provenance.mode == "smoke"
    assert len(record.outcomes) == len(smoke_manifest().cases) * len(r.ALL_STAGES)
    assert "per_stage" in record.metrics
