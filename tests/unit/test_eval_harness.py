"""Eval harness foundation (plan Section 13.1): frozen manifest, recorded
inputs, stage runner S1-S4, metering, run store. Offline only (§20)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from consilium.context.facts import F
from consilium.core.clock import FakeClock
from consilium.eval import runners as r
from consilium.eval.datasets import (
    SMOKE_MANIFEST_PATH,
    CaseManifest,
    load_manifest,
    manifest_bytes,
)
from consilium.eval.fixtures import (
    SYNTHETIC_LLM_BACKEND,
    RawEvalInput,
    record_cases,
    smoke_manifest,
    synthetic_draft,
    synthetic_subreport,
)
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.schemas.enums import Laterality, Role
from consilium.verify.deterministic import verify_draft
from consilium.vision import client_fake as vf
from consilium.vision import pipeline as vision_pipeline
from consilium.vision.config import load_models_config, load_thresholds

CLOCK = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _run(samples: int = 2) -> r.RunRecord:
    return asyncio.run(
        r.run_ablation(
            r.smoke_plan(samples=samples), r.default_runner(), clock=FakeClock(current=CLOCK)
        )
    )


@pytest.fixture(scope="module")
def smoke_run() -> r.RunRecord:
    return _run()


# --- manifest ---------------------------------------------------------------
def test_committed_smoke_manifest_matches_its_generator() -> None:
    assert SMOKE_MANIFEST_PATH.read_bytes() == manifest_bytes(smoke_manifest()), (
        "smoke_v1.json drifted from fixtures.smoke_manifest(); regenerate with "
        "`uv run python -m consilium.eval.fixtures` and review the new dataset version"
    )


def test_manifest_version_is_content_hash(tmp_path: Path) -> None:
    loaded = load_manifest(SMOKE_MANIFEST_PATH)
    assert len(loaded.dataset_version) == 64
    copy = tmp_path / "m.json"
    copy.write_bytes(SMOKE_MANIFEST_PATH.read_bytes().replace(b"smoke-001", b"smoke-00X"))
    assert load_manifest(copy).dataset_version != loaded.dataset_version


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    m = smoke_manifest()
    dup = CaseManifest(dataset_id="d", description="", cases=(m.cases[0], m.cases[0]))
    path = tmp_path / "dup.json"
    path.write_bytes(manifest_bytes(dup))
    with pytest.raises(ValueError, match="duplicate case IDs"):
        load_manifest(path)


def test_smoke_set_covers_the_paths_metrics_need() -> None:
    cases = smoke_manifest().cases
    ctx_ids = [{f.id for f in r.global_context(c).facts} for c in cases]
    assert any(c.note is None for c in cases)
    assert any(F.caveat("blurred") in ids for ids in ctx_ids)
    assert any(F.caveat("mask_qc") in ids for ids in ctx_ids)
    assert any(F.CAVEAT_EXTRACTION in ids for ids in ctx_ids)
    assert sum(bool(c.curated_disagreement_topics) for c in cases) >= 2
    assert {c.split for c in cases} == {"dev", "holdout"}
    assert all(c.cad.quality.gradable for c in cases)  # non-gradable never reaches LLMs


async def test_record_cases_freezes_vision_output_once() -> None:
    models = load_models_config()
    pin = models.endpoint.pinned_revision
    cases = await record_cases(
        [RawEvalInput("rec-1", vision_pipeline.synthetic_fundus_bytes(256), Laterality.OD)],
        client=vf.FakeVisionClient(revision=pin),
        thresholds=load_thresholds(),
        models=models,
    )
    assert cases[0].cad_provenance == f"recorded:{pin}"
    assert cases[0].cad.laterality is Laterality.OD


# --- responder + routing ------------------------------------------------------
def test_clean_synthetic_draft_passes_the_real_verifier_on_routed_context() -> None:
    runner = r.default_runner()
    for case in smoke_manifest().cases:
        ctx = r.global_context(case)
        contexts = runner.role_contexts(r.Stage.S3, ctx)
        assert contexts is not None
        reports = {
            role: synthetic_subreport(role, rc, case, 13, 0) for role, rc in contexts.items()
        }
        draft = synthetic_draft(ctx, reports, case, 13, 0, slip=False)
        issues = verify_draft(
            draft, ctx, tuple(reports.values()), runner.thresholds, runner.routing
        )
        assert issues == (), (case.case_id, issues)


def test_stage_contexts_shared_vs_routed() -> None:
    runner = r.default_runner()
    ctx = r.global_context(smoke_manifest().cases[0])
    assert runner.role_contexts(r.Stage.S1, ctx) is None
    shared = runner.role_contexts(r.Stage.S2, ctx)
    routed = runner.role_contexts(r.Stage.S3, ctx)
    assert shared is not None and routed is not None
    all_ids = ctx.fact_ids()
    assert all(c.allowed_ids() == all_ids for c in shared.values())
    assert F.CDR_V not in routed[Role.PHARMACIST].allowed_ids()
    assert F.MEDS not in routed[Role.OPHTHALMOLOGIST].allowed_ids()


# --- runner -------------------------------------------------------------------
def test_smoke_run_covers_every_case_stage_sample(smoke_run: r.RunRecord) -> None:
    n_cases = len(smoke_manifest().cases)
    assert len(smoke_run.outcomes) == n_cases * len(r.ALL_STAGES) * 2
    assert all(o.status == "ok" for o in smoke_run.outcomes)
    for o in smoke_run.outcomes:
        if o.stage is r.Stage.S4:
            assert o.audit_approved is True and o.audit_rounds in (1, 2)
            assert any(c.agent == "critic" for c in o.llm_calls)
        else:
            assert o.audit_approved is None and o.audit_rounds is None
        if o.stage is r.Stage.S1:
            assert o.sub_reports == () and [c.agent for c in o.llm_calls] == ["single_agent"]
        else:
            assert [s.role for s in o.sub_reports] == list(Role)


def test_stages_are_paired_and_s4_repairs_the_seeded_slip(smoke_run: r.RunRecord) -> None:
    by_key = {(o.case_id, o.sample, o.stage): o for o in smoke_run.outcomes}
    slips = 0
    for case in smoke_manifest().cases:
        truth = next(f.value for f in r.global_context(case).facts if f.id == F.CDR_V)
        for sample in range(2):

            def c1(stage: r.Stage, key: tuple[str, int] = (case.case_id, sample)) -> object:
                draft = by_key[(*key, stage)].draft
                assert draft is not None
                return draft.claims[0].numeric_value

            # Same slip decision in every stage for the same case+sample (paired).
            assert c1(r.Stage.S1) == c1(r.Stage.S2) == c1(r.Stage.S3)
            slipped = c1(r.Stage.S3) != truth
            slips += slipped
            s4 = by_key[(case.case_id, sample, r.Stage.S4)]
            assert c1(r.Stage.S4) == truth
            assert s4.audit_rounds == (2 if slipped else 1)
    assert slips > 0  # the smoke run must exercise reject -> revise


def test_provenance_is_complete(smoke_run: r.RunRecord) -> None:
    p = smoke_run.provenance
    loaded = load_manifest(SMOKE_MANIFEST_PATH)
    assert p.run_id.startswith("eval-smoke-20260102T030405Z-")
    assert p.dataset_version == loaded.dataset_version
    assert p.llm_backend == SYNTHETIC_LLM_BACKEND
    assert set(p.config_hashes) == {"routing.yaml", "thresholds.yaml", "models.yaml", "eval.yaml"}
    assert p.routing_version == p.config_hashes["routing.yaml"]
    assert p.shared_routing_version != p.routing_version
    assert set(p.prompt_versions) == {
        "ophthalmologist",
        "optometrist",
        "pharmacist",
        "director",
        "critic",
        "single_agent",
    }
    assert p.samples_per_case == 2 and p.seed == r.DEFAULT_SEED
    assert len(p.git_sha) >= 7


def test_runs_are_deterministic_given_seed_and_clock(smoke_run: r.RunRecord) -> None:
    again = _run()
    assert again.provenance.run_id == smoke_run.provenance.run_id
    assert [o.model_copy(update={"wall_ms": 0.0}) for o in again.outcomes] == [
        o.model_copy(update={"wall_ms": 0.0}) for o in smoke_run.outcomes
    ]


async def test_failed_stage_is_stored_not_dropped() -> None:
    runner = r.default_runner()
    case = smoke_manifest().cases[0]
    ctx = r.global_context(case)
    llm = FakeLLMClient(
        {
            (role.value, "v1", "default"): CannedResponse((CannedCall(FakeMode.REFUSAL),))
            for role in Role
        }
    )
    outcome = await runner.run(r.Stage.S3, case, ctx, llm, 0)
    assert outcome.status == "failed"
    assert outcome.failure_reason == "SPECIALIST_FAILED"
    assert outcome.draft is None


# --- run store ----------------------------------------------------------------
def test_run_store_round_trip_and_immutability(tmp_path: Path, smoke_run: r.RunRecord) -> None:
    store = r.RunStore(tmp_path)
    store.save(smoke_run)
    assert store.load(smoke_run.provenance.run_id) == smoke_run
    assert store.run_ids() == (smoke_run.provenance.run_id,)
    with pytest.raises(FileExistsError):
        store.save(smoke_run)
    for bad in ("", "../x", "a/b", "a.b"):
        with pytest.raises(ValueError, match="invalid run id"):
            store.path_for(bad)
