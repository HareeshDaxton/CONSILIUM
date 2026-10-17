"""Critic evaluation stage + clinician rubric tooling (plan Section 13.3).
Offline only (§20): scripted critic, canned judge, synthetic raters."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from consilium.core.clock import FakeClock
from consilium.eval import ablation as ab
from consilium.eval import fault_injection as fi
from consilium.eval import judge as jd
from consilium.eval import runners as r
from consilium.eval.datasets import LoadedManifest
from consilium.eval.fixtures import smoke_manifest
from consilium.schemas.enums import Laterality

RUNNER = r.default_runner()


@pytest.fixture(scope="module")
def run_and_manifest() -> tuple[r.RunRecord, LoadedManifest]:
    plan = r.smoke_plan(samples=1)
    record = asyncio.run(
        r.run_ablation(plan, RUNNER, clock=FakeClock(current=datetime(2026, 2, 1, tzinfo=UTC)))
    )
    return record, plan.loaded


@pytest.fixture(scope="module")
def fault_report() -> dict[str, object]:
    corpus = fi.eval_corpus(smoke_manifest().cases, RUNNER.thresholds, RUNNER.routing)
    return asyncio.run(
        fi.fault_injection_stage(
            corpus,
            RUNNER.thresholds,
            RUNNER.routing,
            critic=fi.ScriptedCritic(),
            critic_label="scripted",
        )
    )


def _cls(report: dict[str, object], layer: str, fault: fi.FaultClass) -> dict[str, object]:
    layers = report["layers"]
    assert isinstance(layers, dict)
    per_class = layers[layer]["per_class"]
    assert isinstance(per_class, dict)
    row = per_class[fault.value]
    assert isinstance(row, dict)
    return row


# --- fault injection as an eval stage --------------------------------------
def test_eval_corpus_is_known_good_and_not_critic_built() -> None:
    corpus = fi.eval_corpus(smoke_manifest().cases, RUNNER.thresholds, RUNNER.routing)
    assert [e.name for e in corpus] == ["clean_v1", *(c.case_id for c in smoke_manifest().cases)]


def test_inapplicable_mutations_are_excluded_from_totals(
    fault_report: dict[str, object],
) -> None:
    curated = sum(bool(c.curated_disagreement_topics) for c in smoke_manifest().cases)
    row = _cls(fault_report, "deterministic", fi.FaultClass.SUPPRESSED_DISAGREEMENT)
    assert row["total"] == curated + 1  # + clean_v1, which carries a disagreement
    assert _cls(fault_report, "combined", fi.FaultClass.DOSE_INSERTION)["total"] == 7


def test_combined_layer_and_misses_are_reported_by_name(
    fault_report: dict[str, object],
) -> None:
    for fault in fi.FaultClass:
        row = _cls(fault_report, "combined", fault)
        assert row["caught"] == row["total"] - len(row["misses"])  # type: ignore[arg-type]
    # Honest miss: the laterality check exempts UNKNOWN laterality, so a flip
    # to "left eye (OS)" on an unknown-eye case passes the deterministic layer.
    unknown = [c.case_id for c in smoke_manifest().cases if c.cad.laterality is Laterality.UNKNOWN]
    flip = _cls(fault_report, "combined", fi.FaultClass.LATERALITY_FLIP)
    assert flip["misses"] == unknown
    assert _cls(fault_report, "deterministic", fi.FaultClass.FABRICATED_WITH_REF)["caught"] == 0


def test_clean_corpus_false_rejection_is_reported(fault_report: dict[str, object]) -> None:
    layers = fault_report["layers"]
    assert isinstance(layers, dict)
    for name in ("deterministic", "llm_critic", "combined"):
        assert layers[name]["clean_rejected"] == []
        assert layers[name]["clean_false_rejection_rate"] == 0.0


def test_v1_committed_reports_unchanged_by_generalised_mutations() -> None:
    report = asyncio.run(
        fi.evaluate_layers(fi.known_good_corpus(), RUNNER.thresholds, RUNNER.routing)
    )
    assert report == json.loads(fi.LAYERS_REPORT_PATH.read_text(encoding="utf-8"))


# --- rubric tooling --------------------------------------------------------
def test_packet_is_blinded_stratified_and_deterministic(
    run_and_manifest: tuple[r.RunRecord, LoadedManifest],
) -> None:
    record, loaded = run_and_manifest
    packet, key = jd.build_packet(record, loaded, per_stage=2, seed=7)
    again, _ = jd.build_packet(record, loaded, per_stage=2, seed=7)
    assert packet == again
    assert len(packet.items) == 2 * len(r.ALL_STAGES)
    assert sorted(e.stage for e in key.entries) == sorted(list(r.ALL_STAGES) * 2)
    assert [i.item_id for i in packet.items] == sorted(i.item_id for i in packet.items)
    blob = packet.model_dump_json()
    for leak in ('"S1"', '"S4"', "ophthalmologist", "pharmacist", "smoke-00"):
        assert leak not in blob
    assert "[stated value:" in packet.items[0].report_block  # numbers are reviewable


def test_scores_must_match_the_runs_packet(
    run_and_manifest: tuple[r.RunRecord, LoadedManifest],
) -> None:
    record, loaded = run_and_manifest
    packet, key = jd.build_packet(record, loaded, per_stage=1, seed=7)
    good = jd.synthetic_rater_scores(packet, seed=1)
    jd.validate_scores(jd.RubricScores(run_id=key.run_id, scores=good), key)
    stray = good[0].model_copy(update={"item_id": "item-nope"})
    with pytest.raises(ValueError, match="unknown items"):
        jd.validate_scores(jd.RubricScores(run_id=key.run_id, scores=(stray,)), key)
    with pytest.raises(ValueError, match="different run"):
        jd.validate_scores(jd.RubricScores(run_id="other", scores=good), key)
    with pytest.raises(ValueError):
        jd.RubricScore.model_validate({**good[0].model_dump(), "safety": 6})


def test_judge_agreement_metrics() -> None:
    def score(item: str, v: int, kind: str = "clinician") -> jd.RubricScore:
        return jd.RubricScore.model_validate(
            {
                "item_id": item,
                "rater_id": "x",
                "rater_kind": kind,
                **dict.fromkeys(jd.RUBRIC_DIMENSIONS, v),
            }
        )

    ref = [score("a", 1), score("b", 3), score("c", 5)]
    same = jd.judge_agreement(ref, [score(s.item_id, s.accuracy, "llm_judge") for s in ref])
    acc = same["accuracy"]
    assert isinstance(acc, dict)
    assert acc["mae"] == 0.0 and acc["exact_agreement"] == 1.0 and acc["spearman"] == 1.0
    off = jd.judge_agreement(ref, [score("a", 2), score("b", 3), score("zz", 1)])
    assert off["n_items"] == 2


async def test_llm_judge_scores_via_the_llm_layer() -> None:
    llm = jd.canned_judge_llm()
    judge = jd.LLMJudge(llm=llm, model="judge-snap", temperature=0.0)
    item = jd.RubricItem(item_id="item-1", facts_block="f", report_block="r")
    out = await judge.score(item)
    assert out.rater_kind == "llm_judge" and out.rater_id == "judge-snap@T=0.0"
    assert llm.received[0].key == ("judge", "v1", "default")


async def test_dry_run_is_labelled_synthetic(
    run_and_manifest: tuple[r.RunRecord, LoadedManifest],
) -> None:
    record, loaded = run_and_manifest
    judge = jd.LLMJudge(llm=jd.canned_judge_llm(), model="canned", temperature=0.0)
    out = await jd.dry_run(
        record, loaded, judge=judge, per_stage=1, seed=3, resamples=200, confidence=0.95
    )
    assert out["label"] == jd.DRY_RUN_LABEL and out["items"] == len(r.ALL_STAGES)
    assert set(out["rater_summary"]) == {s.value for s in r.ALL_STAGES}  # type: ignore[arg-type]


# --- run store integration -------------------------------------------------
def test_smoke_cli_stores_13_3_stages_and_rubric_round_trip(tmp_path: Path) -> None:
    run_id = ab.main(["--out", str(tmp_path), "smoke", "--samples", "1"])
    store = r.RunStore(tmp_path)
    record = store.load(run_id)
    assert set(record.extras) == {"fault_injection", "rubric_dry_run"}
    faults = record.extras["fault_injection"]
    assert isinstance(faults, dict) and "scripted_stand_in" in str(faults["critic"])

    ab.main(["--out", str(tmp_path), "rubric-export", "--run-id", run_id, "--per-stage", "1"])
    packet = store.read_attachment(run_id, "rubric_packet", jd.RubricPacket)
    assert packet is not None
    scores = jd.RubricScores(run_id=run_id, scores=jd.synthetic_rater_scores(packet, seed=1))
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(scores.model_dump_json(), encoding="utf-8")
    ab.main(
        ["--out", str(tmp_path), "rubric-import", "--run-id", run_id, "--scores", str(scores_path)]
    )
    assert store.read_attachment(run_id, "rubric_scores", jd.RubricScores) == scores
    with pytest.raises(FileExistsError):  # attachments are write-once
        store.attach(run_id, "rubric_scores", scores)
    with pytest.raises(ValueError, match="invalid attachment"):
        store.attach(run_id, "../x", scores)
