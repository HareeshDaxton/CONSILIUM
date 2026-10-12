"""All cross-component Pydantic models (AGENTS.md §7).

This package imports nothing from the rest of ``consilium`` — everything else
may import it (§5 hard boundary, enforced by import-linter).
"""

from consilium.schemas.agents import (
    Claim,
    DirectorDraft,
    Disagreement,
    Finding,
    Recommendation,
    SubReport,
)
from consilium.schemas.audit import (
    UNVERIFIED_BANNER,
    AuditResult,
    CriticVerdict,
    Issue,
    MergedVerdict,
)
from consilium.schemas.base import Strict
from consilium.schemas.context import Fact, GlobalContext, RoleContext
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
from consilium.schemas.note import ClinicalNote, NoteExtraction
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
    "UNVERIFIED_BANNER",
    "AgentName",
    "AuditResult",
    "CADResult",
    "CDRMetrics",
    "CaseStatus",
    "Claim",
    "ClaimKind",
    "ClassifierOutput",
    "ClinicalNote",
    "Confidence",
    "CriticVerdict",
    "DirectorDraft",
    "Disagreement",
    "Fact",
    "FactSource",
    "Finding",
    "GlobalContext",
    "ImageQuality",
    "Issue",
    "IssueType",
    "Laterality",
    "MaskQC",
    "Measurements",
    "MergedVerdict",
    "NoteExtraction",
    "QualityFlag",
    "RawVisionOutput",
    "Recommendation",
    "Role",
    "RoleContext",
    "ScreeningTier",
    "Strict",
    "SubReport",
]
