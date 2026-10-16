"""Clinician review service — the ONLY writer of APPROVED* statuses (I-4,
AGENTS.md §15, ARCHITECTURE.md §5.11, clinician-review skill).

Rules enforced here (server-side, never trusted from the client):
  - reviewer identity comes from the auth context; a case's submitter may not
    approve it (no self-approval);
  - reason codes are mandatory for APPROVE_WITH_EDITS / REJECT; OTHER needs text;
  - edits are stored as the full edited draft AND a field-level diff against
    the exact AI draft revision (``base_revision`` must be the latest);
  - UNVERIFIED drafts (critic loop exhausted) cannot be approved unless the edit
    set resolves every issue of the latest stored verdict: each claim-scoped
    issue's claim must be edited or removed, and the edited draft must pass
    the deterministic verifier against the case's GlobalContext;
  - approved reports are immutable: the only action on an APPROVED* case is an
    amendment (APPROVE_WITH_EDITS) that creates a new linked version;
  - the review row + case transition + audit event are ONE transaction, under
    a row lock on the case (two clinicians cannot both finalise it).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from consilium.context.router import RoutingConfig
from consilium.core.errors import Forbidden, NotFound, RequestInvalid, StatusConflict
from consilium.persistence.repositories import PersistedReview, Repositories
from consilium.review.diff import Change, diff_drafts
from consilium.review.feedback import Correction, FeedbackStore, PostgresFeedbackStore
from consilium.review.reason_codes import ReasonCode
from consilium.schemas.agents import DirectorDraft, SubReport
from consilium.schemas.context import GlobalContext
from consilium.schemas.enums import CaseStatus, Laterality
from consilium.verify import deterministic
from consilium.vision.config import Thresholds

__all__ = [
    "SCREENING_NOTICE",
    "ReportView",
    "ReviewAction",
    "ReviewCommand",
    "ReviewPolicy",
    "ReviewResult",
    "load_report",
    "review",
]

SCREENING_NOTICE = (
    "Screening support only — not a diagnosis. Content reviewed and approved by "
    "the attributed clinician."
)


class ReviewAction(StrEnum):
    APPROVE = "APPROVE"
    APPROVE_WITH_EDITS = "APPROVE_WITH_EDITS"
    REJECT = "REJECT"


_RESULT_STATUS: dict[ReviewAction, CaseStatus] = {
    ReviewAction.APPROVE: CaseStatus.APPROVED,
    ReviewAction.APPROVE_WITH_EDITS: CaseStatus.APPROVED_WITH_EDITS,
    ReviewAction.REJECT: CaseStatus.REJECTED,
}
_APPROVED = frozenset({CaseStatus.APPROVED, CaseStatus.APPROVED_WITH_EDITS})
_APPROVING = frozenset({ReviewAction.APPROVE.value, ReviewAction.APPROVE_WITH_EDITS.value})


class ReviewCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ReviewAction
    base_revision: int = Field(ge=0)
    edited_draft: DirectorDraft | None = None
    reason_codes: tuple[ReasonCode, ...] = ()
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("edited_draft", mode="before")
    @classmethod
    def _draft_from_json(cls, value: object) -> object:
        # Validate in JSON mode: keeps DirectorDraft's strictness (I-8) while
        # accepting JSON arrays for its tuple fields.
        if isinstance(value, dict):
            return DirectorDraft.model_validate_json(json.dumps(value))
        return value


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str
    case_id: str
    action: ReviewAction
    status: CaseStatus
    version: int | None  # report version for approvals; None for REJECT
    amends_review_id: str | None


class ReportView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    status: CaseStatus
    laterality: Laterality
    version: int
    review_id: str
    amends_review_id: str | None
    approved_by: str
    approved_at: datetime
    action: ReviewAction
    draft_revision: int
    reason_codes: tuple[str, ...]
    report: DirectorDraft
    screening_notice: str = SCREENING_NOTICE


@dataclass(frozen=True)
class ReviewPolicy:
    repos: Repositories
    thresholds: Thresholds
    routing: RoutingConfig
    feedback: FeedbackStore = field(default_factory=PostgresFeedbackStore)


def _check_command(cmd: ReviewCommand) -> None:
    if cmd.action == ReviewAction.APPROVE:
        if cmd.edited_draft is not None:
            raise RequestInvalid("APPROVE_CANNOT_CARRY_EDITS")
        if cmd.reason_codes:
            raise RequestInvalid("REASON_CODES_ONLY_FOR_EDITS_OR_REJECT")
        return
    if not cmd.reason_codes:
        raise RequestInvalid("REASON_CODES_REQUIRED")
    if ReasonCode.OTHER in cmd.reason_codes and not (cmd.comment or "").strip():
        raise RequestInvalid("OTHER_REQUIRES_TEXT")
    if cmd.action == ReviewAction.APPROVE_WITH_EDITS and cmd.edited_draft is None:
        raise RequestInvalid("EDITS_REQUIRED")
    if cmd.action == ReviewAction.REJECT and cmd.edited_draft is not None:
        raise RequestInvalid("REJECT_CANNOT_CARRY_EDITS")


def _edited_claim_ids(changes: list[Change]) -> set[str]:
    ids: set[str] = set()
    for ch in changes:
        if ch.path.startswith("claims[id="):
            ids.add(ch.path[len("claims[id=") : ch.path.index("]")])
    return ids


def _unresolved_issues(
    stored: list[Any],
    changes: list[Change],
    edited: DirectorDraft,
    state_json: dict[str, Any],
    policy: ReviewPolicy,
) -> int:
    """Count stored issues the edit set does not resolve (server-side check)."""
    edited_ids = _edited_claim_ids(changes)
    open_count = sum(
        1
        for issue in stored
        if isinstance(issue, dict)
        and issue.get("claim_id") is not None
        and issue["claim_id"] not in edited_ids
    )
    ctx_json = state_json.get("global_ctx")
    if ctx_json is None:
        return open_count + 1  # cannot re-verify → fail closed
    global_ctx = GlobalContext.model_validate(ctx_json, strict=False)
    sub_reports = tuple(
        SubReport.model_validate(r, strict=False)
        for r in (state_json.get("sub_reports") or {}).values()
    )
    remaining = deterministic.verify_draft(
        edited, global_ctx, sub_reports, policy.thresholds, policy.routing
    )
    return open_count + len(remaining)


async def review(
    case_id: str, cmd: ReviewCommand, *, reviewer_id: str, policy: ReviewPolicy
) -> ReviewResult:
    _check_command(cmd)
    try:
        cid = uuid.UUID(case_id)
    except ValueError:
        raise NotFound("CASE_NOT_FOUND") from None

    async with policy.repos.transaction() as tx:
        case = await tx.case_for_update(cid)
        if case is None:
            raise NotFound("CASE_NOT_FOUND")
        status = CaseStatus(case.status)
        latest = await tx.draft_latest_revision(cid)
        if latest is None:
            raise StatusConflict("DRAFT_NOT_AVAILABLE")
        if cmd.base_revision != latest:
            raise StatusConflict("STALE_REVISION")
        base = await tx.draft_get(cid, latest)
        assert base is not None
        approvals = [r for r in await tx.reviews_for_case(cid) if r.action in _APPROVING]

        amends: PersistedReview | None = None
        base_json = base.draft_json
        if status in _APPROVED:
            if cmd.action != ReviewAction.APPROVE_WITH_EDITS:
                raise StatusConflict("REPORT_IMMUTABLE")
            amends = approvals[-1]
            base_json = amends.edited_draft_json or base.draft_json  # amend the live report
        elif status == CaseStatus.NEEDS_ATTENTION:
            if cmd.action != ReviewAction.REJECT and not base.unverified:
                raise StatusConflict("CASE_NOT_REVIEWABLE")
        elif status != CaseStatus.PENDING_REVIEW:
            raise StatusConflict("CASE_NOT_REVIEWABLE")

        # CLINICAL-REVIEW: only APPROVALS by the case's submitter are refused;
        # whether self-REJECT should also be refused is a team decision.
        if cmd.action != ReviewAction.REJECT and reviewer_id == case.created_by:
            raise Forbidden("SELF_APPROVAL")

        changes: list[Change] = []
        edited_json: dict[str, Any] | None = None
        if cmd.action == ReviewAction.APPROVE_WITH_EDITS:
            assert cmd.edited_draft is not None
            edited_json = cmd.edited_draft.model_dump(mode="json")
            changes = diff_drafts(base_json, edited_json)
            if not changes:
                raise RequestInvalid("NO_EDITS")

        if cmd.action != ReviewAction.REJECT and base.unverified and amends is None:
            if cmd.edited_draft is None:
                raise StatusConflict("UNVERIFIED_REQUIRES_EDITS")
            stored = await tx.verdict_latest_issues(cid) or base.open_issues_json or []
            if _unresolved_issues(stored, changes, cmd.edited_draft, case.state_json, policy):
                raise StatusConflict("UNVERIFIED_ISSUES_OPEN")

        version = len(approvals) + 1 if cmd.action != ReviewAction.REJECT else None
        review_id = uuid.uuid4()
        meta: dict[str, Any] = {
            "base_revision": latest,
            "version": version,
            "amends_review_id": None if amends is None else str(amends.id),
            "changes": [c.to_json() for c in changes],
        }
        await policy.feedback.record(
            tx,
            Correction(
                review_id=review_id,
                case_id=cid,
                draft_revision=latest,
                reviewer_id=reviewer_id,
                action=cmd.action.value,
                reason_codes=tuple(c.value for c in cmd.reason_codes),
                comment=cmd.comment,
                edits_diff_json=meta,
                edited_draft_json=edited_json,
            ),
        )
        new_status = _RESULT_STATUS[cmd.action]
        await tx.transition(
            case_id=cid,
            status=new_status.value,
            state_json={**case.state_json, "status": new_status.value},
            last_node=case.last_node or "persist_pending",
            failure_json=case.failure_json,
            event=f"review.{cmd.action.value.lower()}",
            actor=f"clinician:{reviewer_id}",
            # codes + ids only — the free-text comment never enters the audit log
            detail={
                "review_id": str(review_id),
                "action": cmd.action.value,
                "draft_revision": latest,
                "version": version,
                "amends_review_id": meta["amends_review_id"],
                "reason_codes": [c.value for c in cmd.reason_codes],
                "changed_fields": len(changes),
            },
        )
    return ReviewResult(
        review_id=str(review_id),
        case_id=str(cid),
        action=cmd.action,
        status=new_status,
        version=version,
        amends_review_id=meta["amends_review_id"],
    )


async def load_report(case_id: str, repos: Repositories) -> ReportView:
    """The approved report (latest version). Gated: APPROVED* cases only."""
    try:
        cid = uuid.UUID(case_id)
    except ValueError:
        raise NotFound("CASE_NOT_FOUND") from None
    case = await repos.load_case(cid)
    if case is None:
        raise NotFound("CASE_NOT_FOUND")
    if CaseStatus(case.status) not in _APPROVED:
        raise StatusConflict("REPORT_NOT_APPROVED")
    async with repos.transaction() as tx:
        approvals = [r for r in await tx.reviews_for_case(cid) if r.action in _APPROVING]
        if not approvals:  # pragma: no cover - status/review invariant
            raise StatusConflict("REPORT_NOT_APPROVED")
        latest = approvals[-1]
        assert latest.draft_revision is not None
        if latest.edited_draft_json is not None:
            content = latest.edited_draft_json
        else:
            row = await tx.draft_get(cid, latest.draft_revision)
            assert row is not None
            content = row.draft_json
    meta = latest.edits_diff_json or {}
    return ReportView(
        case_id=str(cid),
        status=CaseStatus(case.status),
        laterality=Laterality(case.state_json.get("laterality", "UNKNOWN")),
        version=int(meta.get("version") or len(approvals)),
        review_id=str(latest.id),
        amends_review_id=meta.get("amends_review_id"),
        approved_by=latest.reviewer_id,
        approved_at=latest.decided_at,
        action=ReviewAction(latest.action),
        draft_revision=latest.draft_revision,
        reason_codes=tuple(str(c) for c in latest.reason_codes),
        report=DirectorDraft.model_validate(content, strict=False),
    )
