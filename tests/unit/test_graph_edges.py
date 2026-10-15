"""Section 9.2 exit: edge unit tests without the graph (plan 9.2)."""

from __future__ import annotations

import pytest

from consilium.graph.edges import LINEAR_CHAIN, TERMINAL_STATUSES, next_node
from consilium.graph.state import CaseState
from consilium.schemas.audit import Issue, MergedVerdict
from consilium.schemas.enums import CaseStatus, IssueType


def _state(last_node: str, status: CaseStatus, **kw: object) -> CaseState:
    return CaseState(
        case_id="00000000-0000-0000-0000-000000000902",
        last_node=last_node,
        status=status,
        **kw,  # type: ignore[arg-type]
    )


def test_linear_chain_walks_in_order() -> None:
    expected = ["precheck_image", "input_rail", "extract_note", "call_vision"]
    node = next_node(_state("", CaseStatus.RECEIVED))
    assert node == "precheck_image"
    for current, nxt in zip(LINEAR_CHAIN, expected[1:], strict=False):
        assert next_node(_state(current, CaseStatus.IMAGE_PRECHECKED)) == nxt


def test_audit_edges() -> None:
    approved = MergedVerdict(approved=True, issues=(), round=1, critic_ran=True)
    assert next_node(_state("audit", CaseStatus.AUDITING, verdict=approved)) == "output_rail"
    rejected = MergedVerdict(
        approved=False,
        issues=(
            Issue(
                type=IssueType.UNSUPPORTED,
                claim_id="c1",
                detail="x",
                raised_by="llm_critic",
            ),
        ),
        round=1,
        critic_ran=True,
    )
    assert next_node(_state("audit", CaseStatus.AUDITING, verdict=rejected)) == "revise"
    assert next_node(_state("revise", CaseStatus.REVISING)) == "audit"


def test_terminal_statuses_stop_the_walk() -> None:
    for status in TERMINAL_STATUSES:
        assert next_node(_state("audit", status)) is None
    assert next_node(_state("persist_pending", CaseStatus.PENDING_REVIEW)) is None


def test_output_rail_to_persist_pending() -> None:
    assert next_node(_state("output_rail", CaseStatus.OUTPUT_RAIL_PASSED)) == "persist_pending"


def test_unknown_last_node_is_a_bug_not_a_fallback() -> None:
    with pytest.raises(ValueError, match="unknown last_node"):
        next_node(_state("nope", CaseStatus.RECEIVED))
