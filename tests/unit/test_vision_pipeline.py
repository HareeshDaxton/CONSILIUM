"""Section 2.5 — vision pipeline end-to-end with fakes (offline, §20)."""

import io

import pytest
from PIL import Image

from consilium.core.errors import InputRejected, VisionFailure
from consilium.schemas.enums import Laterality, ScreeningTier
from consilium.vision.client import RetryPolicy
from consilium.vision.client_fake import FakeVisionClient, build_output
from consilium.vision.config import load_models_config, load_thresholds
from consilium.vision.pipeline import run, synthetic_fundus_bytes

THRESHOLDS = load_thresholds()
MODELS = load_models_config()
REV = MODELS.endpoint.pinned_revision
FAST = RetryPolicy(
    total_budget_s=5.0, initial_backoff_s=0.001, max_backoff_s=0.005, jitter_fraction=0.0
)


def _client(**kw: object) -> FakeVisionClient:
    return FakeVisionClient(revision=REV, **kw)  # type: ignore[arg-type]


async def test_happy_path_returns_cadresult() -> None:
    result = await run(
        synthetic_fundus_bytes(),
        client=_client(),
        thresholds=THRESHOLDS,
        models=MODELS,
        laterality=Laterality.OD,
        retry=FAST,
    )
    assert result.laterality == Laterality.OD
    assert result.classifier.p_raw == 0.5
    assert result.classifier.p_calibrated == 0.5  # temperature 1.0 placeholder = identity
    assert result.classifier.tier == ScreeningTier.INTERMEDIATE  # CLINICAL-REVIEW thresholds
    assert result.cdr.vertical == pytest.approx(0.5, abs=0.02)  # concentric r_cup = r_disc/2
    assert result.mask_qc.ok
    assert result.quality.gradable
    assert MODELS.classifier.model_id in result.classifier.model_version
    assert MODELS.segmenter.model_id in result.segmenter_version


async def test_cold_start_recovers_within_budget() -> None:
    client = _client(cold_start_calls=3)
    result = await run(
        synthetic_fundus_bytes(),
        client=client,
        thresholds=THRESHOLDS,
        models=MODELS,
        laterality=Laterality.OS,
        retry=FAST,
    )
    assert client.calls == 4
    assert result.laterality == Laterality.OS


async def test_cold_start_budget_exceeded_fails_closed() -> None:
    with pytest.raises(VisionFailure, match="VISION_UNAVAILABLE"):
        await run(
            synthetic_fundus_bytes(),
            client=_client(always_transient=True),
            thresholds=THRESHOLDS,
            models=MODELS,
            laterality=Laterality.UNKNOWN,
            retry=RetryPolicy(
                total_budget_s=0.01, initial_backoff_s=0.001, jitter_fraction=0.0, max_attempts=3
            ),
        )


async def test_revision_mismatch_propagates_fail_closed() -> None:
    with pytest.raises(VisionFailure, match="VISION_REVISION_MISMATCH"):
        # The pipeline's HTTP client enforces the pin; the fake can bypass it,
        # so the pipeline re-checks before interpreting anything (defense in depth).
        await run(
            synthetic_fundus_bytes(),
            client=_client(force_revision="consilium-vision-WRONG"),
            thresholds=THRESHOLDS,
            models=MODELS,
            laterality=Laterality.OD,
            retry=FAST,
        )


async def test_garbage_rejected_before_endpoint_call() -> None:
    client = _client()
    with pytest.raises(InputRejected):
        await run(
            b"not an image at all",
            client=client,
            thresholds=THRESHOLDS,
            models=MODELS,
            laterality=Laterality.OD,
            retry=FAST,
        )
    assert client.calls == 0  # precheck kept garbage off the endpoint (§8.4)


async def test_oversize_rejected() -> None:
    with pytest.raises(InputRejected, match="IMAGE_TOO_LARGE"):
        await run(
            b"\x00" * (THRESHOLDS.image_precheck.max_bytes + 1),
            client=_client(),
            thresholds=THRESHOLDS,
            models=MODELS,
            laterality=Laterality.OD,
            retry=FAST,
        )


async def test_empty_mask_fails_closed() -> None:
    client = _client(default=build_output(p_raw=0.5, revision=REV, empty_mask=True))
    with pytest.raises(VisionFailure, match="DEGENERATE_MASK"):
        await run(
            synthetic_fundus_bytes(),
            client=client,
            thresholds=THRESHOLDS,
            models=MODELS,
            laterality=Laterality.OD,
            retry=FAST,
        )


async def test_non_gradable_image_fails_with_actionable_reason() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (5, 5, 5)).save(buf, format="PNG")  # near-black
    with pytest.raises(VisionFailure, match="UNDEREXPOSED"):
        await run(
            buf.getvalue(),
            client=_client(),
            thresholds=THRESHOLDS,
            models=MODELS,
            laterality=Laterality.OD,
            retry=FAST,
        )


async def test_keyed_fixture_lookup_by_sha256() -> None:
    from consilium.vision.client_fake import default_fixtures

    image = synthetic_fundus_bytes()
    client = _client()
    client.register(image, default_fixtures(REV)["suspect"])
    result = await run(
        image,
        client=client,
        thresholds=THRESHOLDS,
        models=MODELS,
        laterality=Laterality.OD,
        retry=FAST,
    )
    assert result.classifier.p_raw == 0.8
    assert result.classifier.tier == ScreeningTier.HIGH
