"""Layer-by-layer fault-injection report tests (plan Section 8.4; §13.4).

Proves the measured three-configuration report: deterministic only / scripted
LLM critic only / combined in production order. The combined layer must catch
every seeded fault class; the deterministic-only layer must honestly miss
``fabricated_with_ref`` (entailment is the critic's job); the critic-only
layer must miss the deterministic classes — together this is the evidence for
the two-layer design (§13.3). The committed JSON fixture is drift-checked so
stored numbers cannot silently change (I-11).
"""

from __future__ import annotations

import json

import pytest

from consilium.context.router import RoutingConfig, load_routing_config
from consilium.eval import fault_injection as fi
from consilium.vision.config import Thresholds, load_thresholds

_REPO_ROOT = fi.REPORT_PATH.parents[2]


@pytest.fixture(scope="module")
def thresholds() -> Thresholds:
    return load_thresholds()


@pytest.fixture(scope="module")
def routing() -> RoutingConfig:
    return load_routing_config(_REPO_ROOT / "configs" / "routing.yaml")


@pytest.fixture(scope="module")
def report(thresholds: Thresholds, routing: RoutingConfig) -> dict[str, object]:
    import asyncio

    return asyncio.run(fi.evaluate_layers(fi.known_good_corpus(), thresholds, routing))


def _layer(report: dict[str, object], name: str) -> dict[str, object]:
    layers = report["layers"]
    assert isinstance(layers, dict)
    layer = layers[name]
    assert isinstance(layer, dict)
    return layer


def _recall(report: dict[str, object], layer: str, fault: fi.FaultClass) -> float:
    per_class = _layer(report, layer)["per_class"]
    assert isinstance(per_class, dict)
    row = per_class[fault.value]
    assert isinstance(row, dict)
    recall = row["recall"]
    assert isinstance(recall, float)
    return recall


def test_combined_layer_catches_every_fault_class(report: dict[str, object]) -> None:
    for fault in fi.FaultClass:
        assert _recall(report, "combined", fault) == 1.0, fault.value


def test_deterministic_only_misses_entailment_fabrication(
    report: dict[str, object],
) -> None:
    """Honesty check (§13.4): a valid-ref fabricated finding is invisible to
    the deterministic layer — the report must record the miss, not hide it."""
    assert _recall(report, "deterministic", fi.FaultClass.FABRICATED_WITH_REF) == 0.0
    for fault in fi.FaultClass:
        if fault is not fi.FaultClass.FABRICATED_WITH_REF:
            assert _recall(report, "deterministic", fault) == 1.0, fault.value


def test_critic_only_layer_catches_entailment_fabrication_only(
    report: dict[str, object],
) -> None:
    """The scripted critic catches the entailment fault and none of the
    deterministic classes — the two layers are complementary, never redundant."""
    assert _recall(report, "llm_critic", fi.FaultClass.FABRICATED_WITH_REF) == 1.0
    assert _recall(report, "llm_critic", fi.FaultClass.NUMERIC_GROSS) == 0.0
    assert _recall(report, "llm_critic", fi.FaultClass.DOSE_INSERTION) == 0.0


def test_clean_drafts_never_false_rejected(report: dict[str, object]) -> None:
    for name in ("deterministic", "llm_critic", "combined"):
        layer = _layer(report, name)
        assert layer["clean_rejected"] == 0
        assert layer["clean_false_rejection_rate"] == 0.0


def test_report_labels_scripted_critic_honestly(report: dict[str, object]) -> None:
    label = report["llm_critic"]
    assert isinstance(label, str) and "scripted_stand_in" in label


def test_committed_layers_fixture_has_no_drift(report: dict[str, object]) -> None:
    stored = json.loads(fi.LAYERS_REPORT_PATH.read_text(encoding="utf-8"))
    assert report == stored, (
        "layered fault-injection numbers drifted — regenerate with "
        "`uv run python -m consilium.eval.fault_injection` and review the diff"
    )
