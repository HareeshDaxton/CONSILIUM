"""Vision pipeline: raw image bytes -> CADResult (AGENTS.md §8.0, §8.4).

All interpretation happens HERE, in-repo, from raw endpoint outputs:
calibration -> tier, masks -> post-processing -> measurements -> CDR, mask QC,
quality gate. The endpoint returns raw logits + masks only (I-13).

Fail-closed rules (I-5):
  - precheck failure            -> InputRejected (REJECTED_INPUT)
  - endpoint budget exhausted   -> VisionFailure(VISION_UNAVAILABLE)
  - revision pin mismatch       -> VisionFailure(VISION_REVISION_MISMATCH)
  - RLE/schema malformed        -> VisionFailure(VISION_MALFORMED_PAYLOAD)
  - degenerate (empty) disc     -> VisionFailure(DEGENERATE_MASK)
  - non-gradable image          -> VisionFailure(<actionable quality reason>)
  - mask_qc.ok == False         -> CONTINUES; QC result is carried on CADResult
    and surfaces as caveat facts downstream (context builder, P4).
"""

from __future__ import annotations

import numpy as np

from consilium.core.errors import VisionFailure
from consilium.schemas.enums import Laterality, QualityFlag, ScreeningTier
from consilium.schemas.vision import CADResult, ClassifierOutput
from consilium.vision import cdr as cdr_mod
from consilium.vision import postprocess, quality, rle
from consilium.vision.calibration import calibrate
from consilium.vision.client import RetryPolicy, VisionClient, predict_with_retry
from consilium.vision.config import ModelsConfig, Thresholds

# QualityFlag -> actionable reason strings surfaced to the user (AGENTS.md §4).
_QUALITY_REASONS: dict[QualityFlag, str] = {
    QualityFlag.UNDEREXPOSED: "UNDEREXPOSED",
    QualityFlag.OVEREXPOSED: "OVEREXPOSED",
    QualityFlag.BLURRED: "BLURRED",
    QualityFlag.NOT_FUNDUS: "NOT_A_FUNDUS_IMAGE",
    QualityFlag.DISC_NOT_VISIBLE: "OPTIC_DISC_NOT_VISIBLE",
    QualityFlag.LOW_RESOLUTION: "LOW_RESOLUTION",
}


def _version_string(model_id: str, revision: str, mlflow_run_id: str) -> str:
    return f"{model_id}@{revision}+mlflow:{mlflow_run_id}"


async def run(
    image_bytes: bytes,
    *,
    client: VisionClient,
    thresholds: Thresholds,
    models: ModelsConfig,
    laterality: Laterality,
    retry: RetryPolicy | None = None,
) -> CADResult:
    # 1. Client-side precheck (cheap; keeps garbage off the endpoint).
    quality.precheck_image_bytes(image_bytes, thresholds.image_precheck)

    # 2. Remote inference through the client (cold-start tolerant).
    raw = await predict_with_retry(client, image_bytes, retry or RetryPolicy())
    # Defense in depth: the pin is enforced here even if a client impl skips it.
    if raw.endpoint_revision != models.endpoint.pinned_revision:
        raise VisionFailure(
            "VISION_REVISION_MISMATCH",
            detail="endpoint revision != pinned revision in configs/models.yaml",
        )

    # 3. Decode + post-process masks (schema-validated wire format; RLE is not).
    try:
        disc = rle.decode_rle(raw.disc_mask_rle, raw.mask_height, raw.mask_width)
        cup = rle.decode_rle(raw.cup_mask_rle, raw.mask_height, raw.mask_width)
    except ValueError as exc:
        raise VisionFailure("VISION_MALFORMED_PAYLOAD", detail=f"RLE decode failed: {exc}") from exc
    disc, cup = postprocess.postprocess_masks(disc, cup)

    # 4. Measurements + CDR (degenerate disc = fail closed).
    try:
        measurements = cdr_mod.compute_measurements(disc, cup)
    except ValueError as exc:
        raise VisionFailure("DEGENERATE_MASK", detail=str(exc)) from exc
    cdr_metrics = cdr_mod.compute_cdr(measurements)

    # 5. Mask QC — recorded on the result; does NOT stop the pipeline.
    qc = quality.mask_qc(disc, cup, thresholds.post_inference_quality)

    # 6. Post-inference quality gate — non-gradable never reaches the LLMs.
    image = quality.decode_image(image_bytes)
    image_quality = quality.assess_quality(image, disc, thresholds.post_inference_quality)
    if not image_quality.gradable:
        first = image_quality.flags[0] if image_quality.flags else QualityFlag.NOT_FUNDUS
        raise VisionFailure(
            _QUALITY_REASONS[first],
            detail="non-gradable image: " + ",".join(f.value for f in image_quality.flags),
        )

    # 7. Calibration (in-repo, versioned params) -> screening tier.
    p_calibrated = calibrate(raw.p_raw, models.calibration.temperature)
    tier = ScreeningTier(thresholds.screening_tiers.tier_for(p_calibrated))

    return CADResult(
        quality=image_quality,
        classifier=ClassifierOutput(
            p_raw=raw.p_raw,
            p_calibrated=p_calibrated,
            tier=tier,
            model_version=_version_string(
                models.classifier.model_id,
                models.classifier.revision,
                models.classifier.mlflow_run_id,
            ),
        ),
        measurements=measurements,
        cdr=cdr_metrics,
        mask_qc=qc,
        segmenter_version=_version_string(
            models.segmenter.model_id,
            models.segmenter.revision,
            models.segmenter.mlflow_run_id,
        ),
        laterality=laterality,
    )


def synthetic_fundus_bytes(size: int = 256) -> bytes:
    """Deterministic synthetic fundus-like PNG (gradability-friendly).

    Exists so tests, the fake server and local dev share ONE canonical image
    whose sha256 can key canned fixtures. Not a real fundus; no PHI.
    """
    import io

    from PIL import Image

    yy, xx = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (size / 2)
    red = np.clip(0.75 - 0.35 * r, 0.15, 0.95)
    green = np.clip(0.45 - 0.30 * r, 0.08, 0.8)
    blue = np.clip(0.30 - 0.22 * r, 0.03, 0.6)
    # Fixed-seed fine texture: keeps the Laplacian variance in the gradable
    # band (a smooth gradient reads as BLURRED to the quality gate).
    rng = np.random.default_rng(7)
    texture = rng.normal(0.0, 0.04, (size, size))
    # Dark vessel-like radial streaks for mid-frequency structure.
    theta = np.arctan2(yy - cy, xx - cx)
    vessels = -0.06 * (np.sin(6 * theta) > 0.7).astype(float)
    rgb = np.stack([red + texture + vessels, green + texture + vessels, blue + texture], axis=-1)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
