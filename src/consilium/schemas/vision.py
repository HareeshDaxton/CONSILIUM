"""Vision contracts (AGENTS.md §7, §8).

Two families:

- ``RawVisionOutput`` — the endpoint wire format. What the HF Inference
  Endpoint's custom handler returns and what ``VisionClient`` parses: raw
  probability + RLE masks + revision pin, nothing interpreted (I-13).
  Persisted as ``cad_results.raw_vision_json`` for auditability.
- ``CADResult`` and its parts — the deterministic, versioned, in-repo
  interpretation (calibration → tier, masks → measurements → CDR, mask QC).
  This is the only source of CAD numbers the rest of the pipeline may use
  (I-1). Persisted as ``cad_results.cad_json``.
"""

from pydantic import Field

from consilium.schemas.base import Strict
from consilium.schemas.enums import Laterality, QualityFlag, ScreeningTier


class ImageQuality(Strict):
    score: float = Field(ge=0, le=1)
    gradable: bool
    flags: tuple[QualityFlag, ...] = ()


class ClassifierOutput(Strict):
    p_raw: float = Field(ge=0, le=1)  # raw logit→prob from endpoint
    p_calibrated: float = Field(ge=0, le=1)  # temperature-scaled HERE, in-repo
    tier: ScreeningTier
    model_version: str  # endpoint model revision + MLflow run ID


class MaskQC(Strict):
    cup_within_disc: bool
    single_component_each: bool
    disc_area_plausible: bool  # within thresholds.yaml range
    ok: bool  # AND of the above


class Measurements(Strict):
    disc_area_px: int = Field(gt=0)
    cup_area_px: int = Field(ge=0)
    disc_vertical_diameter_px: int = Field(gt=0)
    cup_vertical_diameter_px: int = Field(ge=0)


class CDRMetrics(Strict):
    """Cup-to-disc ratios (AGENTS.md §8.2).

    ``vertical`` is PRIMARY (clinically used height ratio). ``area_based`` is
    secondary, for comparison with the paper/MedChat. The disc mask INCLUDES
    the cup (nested convention) — do not apply the paper's rim-only formula.
    """

    vertical: float = Field(ge=0, le=1)
    area_based: float = Field(ge=0, le=1)
    mask_convention: str = "disc_includes_cup"


class CADResult(Strict):
    """Deterministic backend interpretation of endpoint output.

    Computed only in ``vision/``; LLMs reference these numbers by fact ID and
    can never create or alter them (I-1).
    """

    quality: ImageQuality
    classifier: ClassifierOutput
    measurements: Measurements
    cdr: CDRMetrics
    mask_qc: MaskQC
    segmenter_version: str  # endpoint model revision + MLflow run ID
    laterality: Laterality


class RawVisionOutput(Strict):
    """Endpoint wire format — raw logits and masks only, nothing else (I-13).

    ``endpoint_revision`` must equal the pinned revision in
    ``configs/models.yaml``; the VisionClient rejects mismatches
    (fail closed, VISION_UNAVAILABLE). The check lives in the client, not
    here, so recorded fixtures stay loadable in tests.
    """

    p_raw: float = Field(ge=0, le=1)
    disc_mask_rle: str  # run-length-encoded, resolution declared in header
    cup_mask_rle: str
    mask_width: int = Field(gt=0)
    mask_height: int = Field(gt=0)
    endpoint_revision: str
