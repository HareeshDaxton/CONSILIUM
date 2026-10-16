"""Reason-code analytics feeding the V2 RAG gate (AGENTS.md §23,
rag-readiness-gate skill). Counts come from the append-only ``reviews`` table
(SQL in ``persistence/``); this module only derives shares. Reports numbers,
never a decision — the threshold is set with the clinical co-authors."""

from __future__ import annotations

from dataclasses import dataclass

from consilium.persistence.repositories import Repositories
from consilium.review.reason_codes import KNOWLEDGE_CODES, ReasonCode

__all__ = ["ReasonCodeReport", "reason_code_report"]


@dataclass(frozen=True, slots=True)
class ReasonCodeReport:
    reviews_with_codes: int  # APPROVE_WITH_EDITS + REJECT reviews
    code_counts: dict[str, int]
    code_shares: dict[str, float]  # per code / reviews_with_codes
    knowledge_share: float  # reviews citing KNOWLEDGE_GAP or CLINICAL_INTERPRETATION


async def reason_code_report(repos: Repositories) -> ReasonCodeReport:
    counts, total, knowledge = await repos.reason_code_counts(
        knowledge_codes=tuple(sorted(KNOWLEDGE_CODES))
    )
    full = {code.value: counts.get(code.value, 0) for code in ReasonCode}
    return ReasonCodeReport(
        reviews_with_codes=total,
        code_counts=full,
        code_shares={k: (v / total if total else 0.0) for k, v in full.items()},
        knowledge_share=knowledge / total if total else 0.0,
    )
