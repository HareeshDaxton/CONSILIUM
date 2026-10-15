"""CaseState — the single object flowing through the pipeline (§4, §5.3).

Plain Pydantic, no LangGraph import (thin-wiring rule: all logic lives in
ordinary modules). Nodes return the NEXT immutable snapshot via
``model_copy(update=...)``; persistence and LangGraph adaptation live in
``graph/build.py`` (Section 9.3).

PHI discipline (I-6/I-14): ``raw_note_input`` and ``raw_note_redacted`` are
``exclude=True`` — they NEVER land in the persisted ``state_json`` snapshot.
The redacted note is stored only in ``clinical_notes.raw_redacted_text_enc``
(encrypted) and rehydrated from there on crash-resume. Image bytes are never
in state at all — only ``image_uri``/``image_sha256``; bytes are re-loaded
from the object store by the node that needs them.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from consilium.schemas.agents import DirectorDraft, SubReport
from consilium.schemas.audit import Issue, MergedVerdict
from consilium.schemas.base import Strict
from consilium.schemas.context import GlobalContext, RoleContext
from consilium.schemas.enums import CaseStatus, Laterality, Role
from consilium.schemas.note import ClinicalNote, NoteExtraction
from consilium.schemas.vision import CADResult

__all__ = ["AgentRunRecord", "CaseState", "FailureRecord"]


class FailureRecord(Strict):
    """Why a case stopped (§4 fail-closed). ``reason`` is an internal detail
    string — it must never carry note text, image data, or prompts (I-6)."""

    type: str  # error code, e.g. "VISION_UNAVAILABLE"
    node: str  # node that raised
    reason: str
    retryable: bool = False


class AgentRunRecord(Strict):
    """Telemetry for one LLM call, drained by the runner into ``agent_runs``.

    Transient: nodes append to ``state.agent_runs_pending``; the persistence
    wrapper writes the rows in the same transaction as the node transition
    and clears the list. Never persisted inside ``state_json``.
    """

    agent: str
    prompt_version: str | None
    model_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    attempt: int = 1
    status: str = "ok"
    error: str | None = None
    output_json: dict[str, Any] | None = None


class CaseState(Strict):
    case_id: str
    status: CaseStatus = CaseStatus.RECEIVED
    idempotency_key: str | None = None
    created_by: str = ""
    laterality: Laterality = Laterality.UNKNOWN
    image_sha256: str | None = None
    image_uri: str | None = None
    # --- transient note text (never persisted — see module docstring) ---
    raw_note_input: str | None = Field(default=None, exclude=True)
    raw_note_redacted: str | None = Field(default=None, exclude=True)
    note_redactions: tuple[str, ...] = ()  # pattern NAMES only (I-6)
    # persisted flag: lets resume tell "no note submitted" from "note text lost
    # in a crash before the input rail persisted it" (the latter fails closed)
    note_submitted: bool = False
    # --- pipeline artifacts ---
    note: ClinicalNote | None = None
    extraction: NoteExtraction | None = None
    cad: CADResult | None = None
    global_ctx: GlobalContext | None = None
    role_contexts: dict[Role, RoleContext] = Field(default_factory=dict)
    sub_reports: dict[Role, SubReport] = Field(default_factory=dict)
    draft: DirectorDraft | None = None
    verdict: MergedVerdict | None = None
    audit_round: int = 0
    draft_revision: int = 0
    open_issues: tuple[Issue, ...] = ()
    unverified: bool = False
    failure: FailureRecord | None = None
    last_node: str = ""
    # --- transient telemetry (drained by the runner, never persisted) ---
    agent_runs_pending: tuple[AgentRunRecord, ...] = Field(default=(), exclude=True)

    def to_snapshot(self) -> dict[str, Any]:
        """JSON-safe snapshot for ``cases.state_json`` (PHI fields excluded
        by the Field declarations above)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> CaseState:
        # JSON round-trip coercion (str→enum, list→tuple) is intentional here.
        return cls.model_validate(snapshot, strict=False)
