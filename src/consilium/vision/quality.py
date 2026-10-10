"""Quality gates (AGENTS.md §8.4).

Two layers:
1. ``precheck_image_bytes`` — client-side, before the endpoint is called:
   decode check, format allowlist, size cap, resolution floor. Raises
   ``InputRejected`` with an actionable reason; cheap garbage never reaches
   the endpoint.
2. ``assess_quality`` / ``mask_qc`` — post-inference, in-repo: exposure,
   blur, fundus heuristic, disc visibility, mask QC. Thresholds come from
   configs/thresholds.yaml (versioned). A non-gradable image never reaches
   the LLMs — the pipeline fails closed with the flag as the reason.

All thresholds are heuristics marked CLINICAL-REVIEW in thresholds.yaml —
tune them on real data, not to make tests pass (§26.4).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped]  # scipy ships no stubs

from consilium.core.errors import InputRejected
from consilium.schemas.enums import QualityFlag
from consilium.schemas.vision import ImageQuality, MaskQC
from consilium.vision.config import ImagePrecheckConfig, PostInferenceQualityConfig


def precheck_image_bytes(data: bytes, cfg: ImagePrecheckConfig) -> None:
    """Cheap client-side gate. Raises InputRejected with an actionable reason."""
    if len(data) > cfg.max_bytes:
        raise InputRejected("IMAGE_TOO_LARGE", detail=f"{len(data)} > {cfg.max_bytes} bytes")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            width, height = img.size
            img.load()  # actually decode — truncated files fail here
    except Exception as exc:
        raise InputRejected("IMAGE_UNDECODABLE") from exc
    if fmt not in cfg.allowed_formats:
        raise InputRejected("UNSUPPORTED_IMAGE_FORMAT", detail=f"format={fmt}")
    if min(width, height) < cfg.min_resolution_px:
        raise InputRejected(
            "LOW_RESOLUTION", detail=f"{width}x{height} < {cfg.min_resolution_px}px"
        )


def decode_image(data: bytes) -> np.ndarray:
    """Decode to float32 HxWx3 in [0, 1]. Call only after precheck passes."""
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _laplacian_variance(gray: np.ndarray) -> float:
    """4-neighbour Laplacian variance — higher means sharper."""
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(lap.var())


def assess_quality(
    image: np.ndarray, disc_mask: np.ndarray, cfg: PostInferenceQualityConfig
) -> ImageQuality:
    """Post-inference quality assessment. ``image`` is HxWx3 float in [0,1]."""
    flags: list[QualityFlag] = []
    mean_intensity = float(image.mean())

    exposure_score = 1.0
    if mean_intensity < cfg.underexposed_below:
        flags.append(QualityFlag.UNDEREXPOSED)
        exposure_score = mean_intensity / cfg.underexposed_below
    elif mean_intensity > cfg.overexposed_above:
        flags.append(QualityFlag.OVEREXPOSED)
        exposure_score = (1.0 - mean_intensity) / (1.0 - cfg.overexposed_above)

    gray = image.mean(axis=2)
    blur_var = _laplacian_variance(gray)
    blur_score = min(1.0, blur_var / cfg.blur_laplacian_var_below)
    if blur_var < cfg.blur_laplacian_var_below:
        flags.append(QualityFlag.BLURRED)

    red_blue_margin = float(image[..., 0].mean() - image[..., 2].mean())
    fundus_score = min(1.0, max(0.0, red_blue_margin / cfg.fundus_red_blue_margin))
    if red_blue_margin < cfg.fundus_red_blue_margin:
        flags.append(QualityFlag.NOT_FUNDUS)

    disc_fraction = float(np.asarray(disc_mask, dtype=bool).mean())
    disc_score = 1.0
    if not (cfg.min_disc_area_fraction <= disc_fraction <= cfg.max_disc_area_fraction):
        flags.append(QualityFlag.DISC_NOT_VISIBLE)
        disc_score = 0.0

    return ImageQuality(
        score=min(exposure_score, blur_score, fundus_score, disc_score),
        gradable=not flags,
        flags=tuple(flags),
    )


def mask_qc(disc_mask: np.ndarray, cup_mask: np.ndarray, cfg: PostInferenceQualityConfig) -> MaskQC:
    """QC over the POST-processed masks that measurements describe.

    ``ok`` is the AND of the three checks (AGENTS.md §7).
    """
    disc = np.asarray(disc_mask, dtype=bool)
    cup = np.asarray(cup_mask, dtype=bool)
    cup_within_disc = not (cup & ~disc).any()
    disc_components = int(ndimage.label(disc, structure=np.ones((3, 3)))[1])
    cup_components = int(ndimage.label(cup, structure=np.ones((3, 3)))[1])
    single_component_each = disc_components <= 1 and cup_components <= 1
    disc_fraction = float(disc.mean())
    disc_area_plausible = cfg.min_disc_area_fraction <= disc_fraction <= cfg.max_disc_area_fraction
    ok = cup_within_disc and single_component_each and disc_area_plausible
    return MaskQC(
        cup_within_disc=cup_within_disc,
        single_component_each=single_component_each,
        disc_area_plausible=disc_area_plausible,
        ok=ok,
    )
