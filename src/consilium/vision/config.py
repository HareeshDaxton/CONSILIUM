"""Config file loaders for vision (configs/thresholds.yaml, configs/models.yaml).

Config files are hand-written and versioned in git — parsed with extra=forbid
so a typo'd key fails loudly instead of being silently ignored. These are NOT
LLM-facing contracts, so they live here rather than in schemas/.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_THRESHOLDS_PATH = _REPO_ROOT / "configs" / "thresholds.yaml"
DEFAULT_MODELS_PATH = _REPO_ROOT / "configs" / "models.yaml"


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScreeningTierThresholds(_ConfigModel):
    intermediate_at: float = Field(ge=0, le=1)
    high_at: float = Field(ge=0, le=1)

    def tier_for(self, p_calibrated: float) -> str:
        if p_calibrated >= self.high_at:
            return "high"
        if p_calibrated >= self.intermediate_at:
            return "intermediate"
        return "low"


class ImagePrecheckConfig(_ConfigModel):
    max_bytes: int = Field(gt=0)
    allowed_formats: tuple[str, ...]
    min_resolution_px: int = Field(gt=0)


class PostInferenceQualityConfig(_ConfigModel):
    underexposed_below: float = Field(ge=0, le=1)
    overexposed_above: float = Field(ge=0, le=1)
    blur_laplacian_var_below: float = Field(gt=0)
    fundus_red_blue_margin: float
    min_disc_area_fraction: float = Field(gt=0, le=1)
    max_disc_area_fraction: float = Field(gt=0, le=1)


class NumericTolerances(_ConfigModel):
    default: float = Field(ge=0)
    per_fact: dict[str, float] = {}

    def tolerance_for(self, fact_id: str) -> float:
        return self.per_fact.get(fact_id, self.default)


class StrayNumberWhitelist(_ConfigModel):
    """Regex classes whose numeric matches are NOT stray numbers (verifier P7)."""

    date: str
    follow_up_interval: str
    age: str


class DisagreementTopic(_ConfigModel):
    """A configured polar topic for the DISAGREEMENT_SUPPRESSED check (§13.1).

    If sub-reports from different roles mention the topic with opposing stance
    markers, the draft must carry a visible ``Disagreement`` for it.
    Conservative: flags for review, never auto-resolves.
    """

    topic: str = Field(min_length=1)
    keywords: tuple[str, ...] = Field(min_length=1)
    stance_a: tuple[str, ...] = Field(min_length=1)
    stance_b: tuple[str, ...] = Field(min_length=1)


class Thresholds(_ConfigModel):
    version: str
    screening_tiers: ScreeningTierThresholds
    image_precheck: ImagePrecheckConfig
    post_inference_quality: PostInferenceQualityConfig
    numeric_tolerances: NumericTolerances
    stray_number_whitelist: StrayNumberWhitelist
    disagreement_topics: tuple[DisagreementTopic, ...] = ()


class EndpointConfig(_ConfigModel):
    pinned_revision: str = Field(min_length=1)


class ModelEntry(_ConfigModel):
    model_id: str
    revision: str
    mlflow_run_id: str


class CalibrationConfig(_ConfigModel):
    method: str
    temperature: float = Field(gt=0)
    fitted_on: str


class ModelsConfig(_ConfigModel):
    version: str
    endpoint: EndpointConfig
    classifier: ModelEntry
    segmenter: ModelEntry
    calibration: CalibrationConfig


def load_thresholds(path: Path = DEFAULT_THRESHOLDS_PATH) -> Thresholds:
    return Thresholds.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_models_config(path: Path = DEFAULT_MODELS_PATH) -> ModelsConfig:
    return ModelsConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
