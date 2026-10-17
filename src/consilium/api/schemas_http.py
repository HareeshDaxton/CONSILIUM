"""HTTP view models (ARCHITECTURE.md §7). API-only shapes that compose the
contract models from ``schemas/``; they never carry note text, filenames,
image bytes or prompts (I-6/I-14)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import Issue
from consilium.schemas.enums import CaseStatus, FactSource, Laterality

__all__ = [
    "AuthConfigView",
    "CadSummary",
    "CaseListItem",
    "CaseListView",
    "CaseView",
    "DraftView",
    "EvidenceChip",
    "FactView",
    "FailureView",
    "GoogleLoginRequest",
    "MeView",
    "SessionUser",
    "SessionView",
    "SubmitResponse",
    "SystemStatusView",
    "TimelineEvent",
    "VisionStatusView",
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
    # True while the pipeline still has nodes to run (decided by the graph's
    # edges, server-side). The UI polls while true; it never reads statuses.
    in_progress: bool = False


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


class MeView(_View):
    """The verified caller (from the token; the UI never trusts its own decode)."""

    sub: str
    roles: tuple[str, ...]


class CaseListItem(_View):
    case_id: str
    status: CaseStatus
    laterality: Laterality
    created_at: datetime | None
    tier: str | None = None
    unverified: bool = False


class CaseListView(_View):
    items: tuple[CaseListItem, ...]


class TimelineEvent(_View):
    """Audit event name + opaque actor + time; audit ``detail_json`` is never
    exposed (it is for investigators, not the review UI)."""

    event: str
    actor: str
    at: datetime


class FactView(_View):
    id: str
    label: str
    value: str | float | int | bool | tuple[str, ...]
    unit: str | None = None
    source: FactSource


class AuthConfigView(_View):
    """Public sign-in configuration for the UI (the Google client ID is public)."""

    google_client_id: str | None


class GoogleLoginRequest(_View):
    credential: str = Field(min_length=20, max_length=8192)  # Google ID token (JWT)


class SessionUser(_View):
    sub: str
    roles: tuple[str, ...]
    email: str | None = None
    name: str | None = None
    picture: str | None = None


class SessionView(_View):
    access_token: str
    token_type: str = "Bearer"  # noqa: S105  # OAuth token-type label, not a secret
    expires_in: int
    user: SessionUser


class VisionStatusView(_View):
    mode: str  # "http" (Deva's GPU server / HF endpoint) or "fake" (offline dev)
    reachable: bool
    revision_ok: bool
    device: str | None = None
    detail: str | None = None


class SystemStatusView(_View):
    """What the upload page needs to know before a submission (no case data)."""

    vision: VisionStatusView
    llm_configured: bool
