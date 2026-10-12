"""Unit + contract tests for verify/merge.py (plan Sections 8.2/8.3; §13.3).

8.2 exit: merge unit tests — det-reject, critic-reject, both-pass, and the
fixed order (a det-failing draft never reaches the critic) are all proven.

8.3 exit: UNVERIFIED draft contract — loop exhaustion keeps the last draft
revision with open issues attached and the banner set (never auto-approved,
never dropped), and a draft whose only blemish is preserved Disagreement
entries is approved normally (disagreements alone never flag NEEDS_ATTENTION).
"""

from __future__ import annotations

import pytest

from consilium.context.router import RoutingConfig, load_routing_config
from consilium.eval import fault_injection as fi
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import (
    UNVERIFIED_BANNER,
    AuditResult,
    CriticVerdict,
    Issue,
    MergedVerdict,
)
from consilium.schemas.enums import IssueType
from consilium.verify.merge import merge_verdicts, run_audit_loop
from consilium.vision.config import Thresholds, load_thresholds

CORPUS = fi.known_good_corpus()
ENTRY = CORPUS[0]


@pytest.fixture(scope="module")
def thresholds() -> Thresholds:
    return load_thresholds()


@pytest.fixture(scope="module")
def routing() -> RoutingConfig:
    return load_routing_config(fi.REPORT_PATH.parents[2] / "configs" / "routing.yaml")


async def _approve(draft: DirectorDraft, ctx: object, round_: int) -> CriticVerdict:
    return CriticVerdict(approved=True, round=round_)


def _det_issue() -> Issue:
    return Issue(
        type=IssueType.NUMERIC_MISMATCH,
        claim_id="c1",
        detail="numeric mismatch",
        raised_by="deterministic",
    )


def _critic_issue() -> Issue:
    return Issue(
        type=IssueType.UNSUPPORTED,
        claim_id="c9",
        detail="claim not entailed by the cited fact",
        raised_by="llm_critic",
    )


# ---------- 8.2: merge_verdicts ----------


def test_merge_both_pass_approves() -> None:
    merged = merge_verdicts((), CriticVerdict(approved=True, round=1), 1)
    assert merged.approved and merged.critic_ran and merged.issues == ()


def test_merge_det_reject_blocks_critic_and_approval() -> None:
    merged = merge_verdicts((_det_issue(),), None, 1)
    assert not merged.approved
    assert not merged.critic_ran
    assert [i.raised_by for i in merged.issues] == ["deterministic"]


def test_merge_critic_reject_keeps_raised_by() -> None:
    verdict = CriticVerdict(approved=False, issues=(_critic_issue(),), round=1)
    merged = merge_verdicts((), verdict, 1)
    assert not merged.approved
    assert merged.critic_ran
    assert [i.raised_by for i in merged.issues] == ["llm_critic"]


def test_merge_rejection_without_issues_is_incoherent() -> None:
    with pytest.raises(ValueError, match="rejection must state"):
        MergedVerdict(approved=False, issues=(), round=1, critic_ran=True)


def test_merge_approval_requires_critic_to_have_run() -> None:
    with pytest.raises(ValueError, match="BOTH layers"):
        MergedVerdict(approved=True, issues=(), round=1, critic_ran=False)


# ---------- 8.2: run_audit_loop ----------


async def test_loop_both_pass_approves_in_round_one(
    thresholds: Thresholds, routing: RoutingConfig
) -> None:
    critic_calls: list[int] = []

    async def critic(draft: DirectorDraft, ctx: object, round_: int) -> CriticVerdict:
        critic_calls.append(round_)
        return CriticVerdict(approved=True, round=round_)

    async def revise(draft: DirectorDraft, issues: tuple[Issue, ...], round_: int) -> DirectorDraft:
        raise AssertionError("revise must not run when the draft passes")

    result = await run_audit_loop(
        draft=ENTRY.draft,
        global_ctx=ENTRY.global_ctx,
        sub_reports=ENTRY.sub_reports,
        thresholds=thresholds,
        routing=routing,
        run_critic=critic,
        revise=revise,
        max_rounds=3,
    )
    assert result.approved and result.rounds == 1 and not result.unverified
    assert result.review_banner is None
    assert critic_calls == [1]


async def test_loop_det_reject_never_calls_critic(
    thresholds: Thresholds, routing: RoutingConfig
) -> None:
    """Order proven: a draft failing the deterministic layer never reaches
    the critic — cheap failures never burn critic tokens (§13.3)."""
    bad = fi.apply_fault(ENTRY, fi.FaultClass.NUMERIC_GROSS)
    critic_calls: list[int] = []

    async def critic(draft: DirectorDraft, ctx: object, round_: int) -> CriticVerdict:
        critic_calls.append(round_)
        return CriticVerdict(approved=True, round=round_)

    revisions: list[int] = []

    async def revise(draft: DirectorDraft, issues: tuple[Issue, ...], round_: int) -> DirectorDraft:
        revisions.append(draft.revision + 1)
        return draft.model_copy(update={"revision": draft.revision + 1})

    result = await run_audit_loop(
        draft=bad.draft,
        global_ctx=bad.global_ctx,
        sub_reports=bad.sub_reports,
        thresholds=thresholds,
        routing=routing,
        run_critic=critic,
        revise=revise,
        max_rounds=3,
    )
    assert critic_calls == []
    assert revisions == [1, 2]
    assert not result.approved
    assert all(i.raised_by == "deterministic" for i in result.open_issues)


async def test_loop_critic_reject_then_revision_approves(
    thresholds: Thresholds, routing: RoutingConfig
) -> None:
    rounds_seen: list[int] = []

    async def critic(draft: DirectorDraft, ctx: object, round_: int) -> CriticVerdict:
        rounds_seen.append(round_)
        if draft.revision == 0:
            return CriticVerdict(approved=False, issues=(_critic_issue(),), round=round_)
        return CriticVerdict(approved=True, round=round_)

    async def revise(draft: DirectorDraft, issues: tuple[Issue, ...], round_: int) -> DirectorDraft:
        assert issues and issues[0].raised_by == "llm_critic"
        return draft.model_copy(update={"revision": draft.revision + 1})

    result = await run_audit_loop(
        draft=ENTRY.draft,
        global_ctx=ENTRY.global_ctx,
        sub_reports=ENTRY.sub_reports,
        thresholds=thresholds,
        routing=routing,
        run_critic=critic,
        revise=revise,
        max_rounds=3,
    )
    assert result.approved and result.rounds == 2
    assert result.draft.revision == 1
    assert rounds_seen == [1, 2]


# ---------- 8.3: UNVERIFIED draft contract ----------


async def test_loop_exhaustion_yields_unverified_draft_with_open_issues(
    thresholds: Thresholds, routing: RoutingConfig
) -> None:
    async def critic(draft: DirectorDraft, ctx: object, round_: int) -> CriticVerdict:
        return CriticVerdict(approved=False, issues=(_critic_issue(),), round=round_)

    async def revise(draft: DirectorDraft, issues: tuple[Issue, ...], round_: int) -> DirectorDraft:
        return draft.model_copy(update={"revision": draft.revision + 1})

    result = await run_audit_loop(
        draft=ENTRY.draft,
        global_ctx=ENTRY.global_ctx,
        sub_reports=ENTRY.sub_reports,
        thresholds=thresholds,
        routing=routing,
        run_critic=critic,
        revise=revise,
        max_rounds=3,
    )
    # Never auto-approved, never silently dropped, issues attached (I-4/I-5).
    assert not result.approved
    assert result.unverified
    assert result.rounds == 3
    assert result.draft.revision == 2  # last revision kept
    assert result.open_issues and all(i.raised_by == "llm_critic" for i in result.open_issues)
    assert result.review_banner == UNVERIFIED_BANNER


async def test_disagreement_only_draft_is_approved(
    thresholds: Thresholds, routing: RoutingConfig
) -> None:
    """Real role disagreement ≠ failed audit (§4): the corpus draft carries a
    preserved Disagreement; with clean layers it must approve normally."""
    assert ENTRY.draft.disagreements  # precondition: the fixture has one

    async def revise(draft: DirectorDraft, issues: tuple[Issue, ...], round_: int) -> DirectorDraft:
        raise AssertionError("no revision expected")

    result = await run_audit_loop(
        draft=ENTRY.draft,
        global_ctx=ENTRY.global_ctx,
        sub_reports=ENTRY.sub_reports,
        thresholds=thresholds,
        routing=routing,
        run_critic=_approve,
        revise=revise,
        max_rounds=3,
    )
    assert result.approved and not result.unverified


def test_audit_result_consistency_validation() -> None:
    with pytest.raises(ValueError, match="open issues"):
        AuditResult(draft=ENTRY.draft, approved=True, rounds=1, open_issues=(_critic_issue(),))
    with pytest.raises(ValueError, match="attach its open issues"):
        AuditResult(draft=ENTRY.draft, approved=False, rounds=3)
