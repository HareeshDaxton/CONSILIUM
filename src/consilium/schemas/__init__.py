"""All cross-component Pydantic models (AGENTS.md §7).

This package imports nothing from the rest of ``consilium`` — everything else
may import it (§5 hard boundary, enforced by import-linter).
"""

from consilium.schemas.base import Strict
from consilium.schemas.enums import (
    AgentName,
    CaseStatus,
    ClaimKind,
    Confidence,
    FactSource,
    IssueType,
    Laterality,
    QualityFlag,
    Role,
    ScreeningTier,
)
from consilium.schemas.vision import (
    CADResult,
    CDRMetrics,
    ClassifierOutput,
    ImageQuality,
    MaskQC,
    Measurements,
    RawVisionOutput,
)

__all__ = [
    "AgentName",
    "CADResult",
    "CDRMetrics",
    "CaseStatus",
    "ClaimKind",
    "ClassifierOutput",
    "Confidence",
    "FactSource",
    "ImageQuality",
    "IssueType",
    "Laterality",
    "MaskQC",
    "Measurements",
    "QualityFlag",
    "RawVisionOutput",
    "Role",
    "ScreeningTier",
    "Strict",
]
