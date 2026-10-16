"""Correction capture (AGENTS.md §15) — the V2 dataset, stored relationally.

``FeedbackStore.record`` is the write seam: V1 binds it to the Postgres
``reviews`` table (append-only) inside the review transaction. The retrieval
seam (``FeedbackRetriever``) exists so a V2 implementation could be injected
into the Director's prompt builder — V1 binds it to a no-op. No embeddings,
no vector search, no retrieval code (I-10).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from consilium.persistence.repositories import RepoTx

__all__ = [
    "Correction",
    "FeedbackRetriever",
    "FeedbackStore",
    "NoopFeedbackRetriever",
    "PostgresFeedbackStore",
]


@dataclass(frozen=True, slots=True)
class Correction:
    review_id: uuid.UUID
    case_id: uuid.UUID
    draft_revision: int
    reviewer_id: str
    action: str
    reason_codes: tuple[str, ...]
    comment: str | None
    edits_diff_json: dict[str, Any] | None
    edited_draft_json: dict[str, Any] | None


class FeedbackStore(Protocol):
    async def record(self, tx: RepoTx, correction: Correction) -> None: ...


class PostgresFeedbackStore:
    """V1: the ``reviews`` row IS the correction record."""

    async def record(self, tx: RepoTx, correction: Correction) -> None:
        await tx.review_add(
            review_id=correction.review_id,
            case_id=correction.case_id,
            draft_revision=correction.draft_revision,
            reviewer_id=correction.reviewer_id,
            action=correction.action,
            edits_diff_json=correction.edits_diff_json,
            edited_draft_json=correction.edited_draft_json,
            reason_codes=list(correction.reason_codes),
            free_text_reason=correction.comment,
        )


class FeedbackRetriever(Protocol):
    """V2 seam: advisory past corrections for a case. Never overrides CAD
    facts or verifier output."""

    async def relevant(self, case_id: str) -> tuple[Correction, ...]: ...


class NoopFeedbackRetriever:
    async def relevant(self, case_id: str) -> tuple[Correction, ...]:
        return ()
