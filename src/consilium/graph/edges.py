"""Graph edges — plain functions of state (ARCHITECTURE.md §5.3).

No LangGraph import: these are unit-tested without the graph. The compiled
StateGraph (Section 9.3) is a thin adapter over ``next_node``.

Shape (§4):

    precheck_image → input_rail → extract_note → call_vision → build_context
      → route_context → run_specialists → draft → audit
    audit → output_rail        [approved]
    audit → revise → audit     [rejected, round < CRITIC_MAX_ROUNDS]
    audit → NEEDS_ATTENTION    [rejected at the cap — set by the audit node
                                itself as a terminal status with the
                                UNVERIFIED contract on the state]
    output_rail → persist_pending → PENDING_REVIEW (terminal)

Any terminal status ends the walk. Crash-resume re-enters at the successor
of the last persisted node — ``next_node`` on the rehydrated state IS the
resume function.
"""

from __future__ import annotations

from consilium.graph.state import CaseState
from consilium.schemas.enums import CaseStatus

__all__ = ["LINEAR_CHAIN", "TERMINAL_STATUSES", "next_node"]

LINEAR_CHAIN: tuple[str, ...] = (
    "precheck_image",
    "input_rail",
    "extract_note",
    "call_vision",
    "build_context",
    "route_context",
    "run_specialists",
    "draft",
    "audit",
)

TERMINAL_STATUSES: frozenset[CaseStatus] = frozenset(
    {
        CaseStatus.REJECTED_INPUT,
        CaseStatus.FAILED,
        CaseStatus.NEEDS_ATTENTION,
        CaseStatus.PENDING_REVIEW,
        CaseStatus.APPROVED,
        CaseStatus.APPROVED_WITH_EDITS,
        CaseStatus.REJECTED,
    }
)


def next_node(state: CaseState) -> str | None:
    """Successor of ``state.last_node``, or None when the case is terminal."""
    if state.status in TERMINAL_STATUSES:
        return None
    if state.last_node == "":
        return LINEAR_CHAIN[0]
    if state.last_node in LINEAR_CHAIN[:-1]:
        return LINEAR_CHAIN[LINEAR_CHAIN.index(state.last_node) + 1]
    if state.last_node == "audit":
        if state.verdict is None:
            raise ValueError("audit completed without a verdict (graph bug)")
        return "output_rail" if state.verdict.approved else "revise"
    if state.last_node == "revise":
        return "audit"
    if state.last_node == "output_rail":
        return "persist_pending"
    if state.last_node == "persist_pending":
        return None
    raise ValueError(f"unknown last_node {state.last_node!r} (graph bug)")
