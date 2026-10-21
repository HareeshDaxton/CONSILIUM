"""Live / full / record eval paths exercised OFFLINE (plan Section 13.4).

The live client is replaced by a FakeLLMClient (§20: no network in the
default suite); the vision endpoint by FakeVisionClient. What is proven here
is the wiring and the fail-closed behaviour — real live numbers come only
from ``python -m consilium.eval.ablation live|full``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from consilium.core.settings import DEFAULT_CRITIC_PROMPT_VERSION as CRITIC_V
from consilium.core.settings import DEFAULT_DIRECTOR_PROMPT_VERSION as DIRECTOR_V
from consilium.core.settings import Settings
from consilium.eval import ablation as ab
from consilium.eval import fault_injection as fi
from consilium.eval.datasets import SMOKE_MANIFEST_PATH, load_manifest
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.schemas.enums import Role
from consilium.vision import client_fake as vf
from consilium.vision import client_http
from consilium.vision import pipeline as vision_pipeline
from consilium.vision.config import load_models_config

PIN = load_models_config().endpoint.pinned_revision


def _refusing() -> FakeLLMClient:
    agents = [r.value for r in Role] + ["director", "critic", "single_agent", "judge"]
    return FakeLLMClient(
        {
            (
                a,
                {"critic": CRITIC_V, "director": DIRECTOR_V}.get(a, "v1"),
                "default",
            ): CannedResponse((CannedCall(FakeMode.REFUSAL),))
            for a in agents
        }
    )


async def test_live_run_fails_closed_and_is_still_stored_and_scored(
    settings: Settings, tmp_path: Path
) -> None:
    """Every LLM call refuses: every outcome is kept FAILED with its reason,
    the metrics report failure_rate 1.0, critic errors in the fault stage are
    fail-closed rejections counted separately, and the run is storable."""
    record = await ab.run_live(
        settings,
        mode="live",
        manifest_path=SMOKE_MANIFEST_PATH,
        samples=1,
        seed=1,
        llm=_refusing(),
    )
    p = record.provenance
    assert (p.mode, p.llm_backend) == ("live", "openai")
    assert all(o.status == "failed" for o in record.outcomes)
    assert {o.failure_reason for o in record.outcomes} <= {"SPECIALIST_FAILED", "LLM_REFUSAL"}
    per_stage = record.metrics["per_stage"]
    assert isinstance(per_stage, dict)
    assert all(block["failure_rate"] == 1.0 for block in per_stage.values())  # type: ignore[index]
    faults = record.extras["fault_injection"]
    assert isinstance(faults, dict) and str(faults["critic"]).startswith("live Critic")
    critic_layer = faults["layers"]["llm_critic"]  # type: ignore[index]
    assert critic_layer["clean_critic_errors"] == critic_layer["clean_total"]  # type: ignore[index]
    rubric = record.extras["rubric_dry_run"]
    assert isinstance(rubric, dict) and rubric["items"] == 0  # no successful drafts to rate
    ab.RunStore(tmp_path).save(record)


async def test_fault_stage_counts_critic_errors_separately(settings: Settings) -> None:
    runner = ab.default_runner()
    corpus = fi.known_good_corpus()

    async def broken_critic(*_: object) -> object:
        from consilium.core.errors import ValidationFailure

        raise ValidationFailure("LLM_OUTPUT_INVALID")

    report = await fi.fault_injection_stage(
        corpus,
        runner.thresholds,
        runner.routing,
        critic=broken_critic,  # type: ignore[arg-type]
        critic_label="broken",
    )
    layers = report["layers"]
    assert isinstance(layers, dict)
    critic_only = layers["llm_critic"]
    assert isinstance(critic_only, dict)
    row = critic_only["per_class"]["numeric_gross"]  # type: ignore[index]
    assert row["critic_errors"] == row["total"] == 1  # type: ignore[index]
    assert critic_only["clean_critic_errors"] == 1
    # Deterministic layer never calls the critic.
    assert layers["deterministic"]["per_class"]["numeric_gross"]["critic_errors"] == 0  # type: ignore[index]


def test_eval_settings_never_depend_on_vision_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_CLIENT_MODE", "http")  # http without a pin fails closed...
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-synthetic")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-synthetic-32-bytes-min")
    monkeypatch.delenv("VISION_PINNED_REVISION", raising=False)
    s = ab.eval_settings()  # ...but the eval reads recorded CAD only
    assert s.vision_client_mode == "fake"


def test_live_cli_refuses_a_missing_frozen_set(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="record it first"):
        ab.main(["--out", str(tmp_path), "full", "--manifest", str(tmp_path / "none.json")])


def _record_settings(settings: Settings, **extra: object) -> Settings:
    return settings.model_copy(
        update={
            "vision_endpoint_url": "https://vision.invalid",
            "vision_endpoint_token": SecretStr("t"),
            "vision_pinned_revision": PIN,
            **extra,
        }
    )


def test_record_full_set_freezes_once(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        client_http, "HttpVisionClient", lambda **_: vf.FakeVisionClient(revision=PIN)
    )
    image = tmp_path / "img.png"
    image.write_bytes(vision_pipeline.synthetic_fundus_bytes(256))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps([{"case_id": "full-001", "image": str(image), "laterality": "OS"}]),
        encoding="utf-8",
    )
    out = tmp_path / "full_v1.json"
    s = _record_settings(settings)
    ab.record_full_set(s, spec, out)
    loaded = load_manifest(out)
    assert loaded.manifest.cases[0].cad_provenance == f"recorded:{PIN}"
    assert str(image) not in out.read_text(encoding="utf-8")  # no file paths stored (I-6)
    with pytest.raises(FileExistsError):
        ab.record_full_set(s, spec, out)


def test_record_full_set_fails_closed_on_config(settings: Settings, tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="VISION_ENDPOINT"):
        ab.record_full_set(settings, spec, tmp_path / "a.json")
    with pytest.raises(ValueError, match="pin"):
        ab.record_full_set(
            _record_settings(settings, vision_pinned_revision="other"), spec, tmp_path / "b.json"
        )
