"""HTTP view models (ARCHITECTURE.md §7). API-only shapes that compose the
contract models from ``schemas/``; they never carry note text, filenames,
image bytes or prompts (I-6/I-14)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import Issue
from consilium.schemas.enums import CaseStatus, Laterality

__all__ = [
    "CadSummary",
    "CaseView",
    "DraftView",
    "EvidenceChip",
    "FailureView",
    "SubmitResponse",
]


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SubmitResponse(_View):
    case_id: str
    status: CaseStatus
    status_url: str


class FailureView(_View):
    type: str
    reason: str


class CadSummary(_View):
    tier: str
    p_calibrated: float
    cdr_vertical: float
    quality_flags: tuple[str, ...]
    mask_qc_ok: bool


class CaseView(_View):
    case_id: str
    status: CaseStatus
    laterality: Laterality
    created_at: datetime | None
    failure: FailureView | None = None
    cad_summary: CadSummary | None = None
    unverified: bool = False


class EvidenceChip(_View):
    label: str
    value: str | float | int | bool | tuple[str, ...]
    unit: str | None = None


class DraftView(_View):
    case_id: str
    case_status: CaseStatus
    revision: int
    draft: DirectorDraft
    open_issues: tuple[Issue, ...]
    unverified: bool
    unverified_banner: str | None
    overlay_url: str
    evidence: dict[str, EvidenceChip]
