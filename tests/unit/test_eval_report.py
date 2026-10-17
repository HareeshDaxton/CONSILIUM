"""Report generation from stored runs only (plan Section 13.4; I-11)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from consilium.eval import ablation as ab
from consilium.eval import judge as jd
from consilium.eval import report as rp
from consilium.eval import runners as r


@pytest.fixture(scope="module")
def record() -> r.RunRecord:
    return asyncio.run(ab.run_smoke(samples=2, seed=13))


def _with_ci(record: r.RunRecord, lo: float, hi: float) -> r.RunRecord:
    metrics = dict(record.metrics)
    metrics["paired"] = {"S3-S2": {"s_inter": {"mean_diff": (lo + hi) / 2, "ci95": [lo, hi]}}}
    return record.model_copy(update={"metrics": metrics})


def test_report_is_generated_from_the_run_and_labels_synthetic(record: r.RunRecord) -> None:
    text = rp.render(record)
    assert text == rp.render(record)  # deterministic
    assert record.provenance.run_id in text
    assert "SYNTHETIC templates" in text and "SYNTHETIC smoke set" in text
    assert "| Metric | S1 | S2 | S3 | S4 |" in text
    assert "## Hypotheses (negative and null results are reported)" in text
    assert "laterality_flip" in text and "missed smoke-003" in text  # misses by name
    assert jd.DRY_RUN_LABEL in text  # no clinician scores -> dry-run label only
    assert "Threats to validity" in text


@pytest.mark.parametrize(
    ("lo", "hi", "status"),
    [
        (-0.2, -0.1, "supported"),
        (0.1, 0.2, "contradicted"),
        (-0.1, 0.1, "not shown (CI includes 0)"),
    ],
)
def test_hypothesis_status_reports_negative_and_null(
    record: r.RunRecord, lo: float, hi: float, status: str
) -> None:
    h = rp.HYPOTHESES[0]  # S3-S2 s_inter lower
    assert rp.hypothesis_status(_with_ci(record, lo, hi), h) == status
    missing = record.model_copy(update={"metrics": {}})
    assert rp.hypothesis_status(missing, h) == "n/a"


def test_readme_block_rewrites_only_the_marked_block(record: r.RunRecord, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{rp.BEGIN}\nold\n{rp.END}\noutro\n", encoding="utf-8")
    assert rp.update_readme(record, readme) is True
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert record.provenance.run_id in text and "synthetic LLM: not a result" in text
    assert rp.update_readme(record, readme) is False  # idempotent
    bare = tmp_path / "bare.md"
    bare.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError, match="block"):
        rp.update_readme(record, bare)


def test_clinician_scores_replace_the_dry_run_section(record: r.RunRecord, tmp_path: Path) -> None:
    store = r.RunStore(tmp_path)
    store.save(record)
    loaded = ab.manifest_for(record)
    packet, key = jd.build_packet(record, loaded, per_stage=1, seed=5)
    scores = jd.RubricScores(
        run_id=record.provenance.run_id,
        scores=tuple(
            s.model_copy(update={"rater_kind": "clinician"})
            for s in jd.synthetic_rater_scores(packet, seed=5)
        ),
    )
    text = rp.render(record, rubric_scores=scores, rubric_key=key)
    assert f"{len(packet.items)} clinician scores attached." in text
    assert "| Stage | accuracy | completeness | safety | usefulness |" in text


def test_cli_write_attaches_report_once(record: r.RunRecord, tmp_path: Path) -> None:
    store = r.RunStore(tmp_path)
    store.save(record)
    rid = record.provenance.run_id
    rp.main(["--out", str(tmp_path), "--run-id", rid, "--write"])
    assert (tmp_path / rid / "report.md").read_text(encoding="utf-8") == rp.render(record)
    with pytest.raises(FileExistsError):
        rp.main(["--out", str(tmp_path), "--run-id", rid, "--write"])
    with pytest.raises(ValueError, match="invalid attachment"):
        store.attach_text(rid, "../x.md", "x")
