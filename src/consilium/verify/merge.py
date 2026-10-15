"""Verdict merge + bounded audit loop (plan Sections 8.2/8.3; AGENTS.md §13.3).

Merge rule: a draft is approved ⇔ the deterministic verifier produced zero
issues AND the LLM critic approved. The order is fixed — deterministic FIRST;
a draft that fails it never reaches the critic, so cheap exact failures never
burn critic tokens. Issues from both layers keep their original
``raised_by`` (``"deterministic"`` | ``"llm_critic"``).

Loop rule (§4): rejection → Director revises with the issue list only, up to
``CRITIC_MAX_ROUNDS`` (default 3, config). On exhaustion the result is
UNVERIFIED: the last draft revision is kept with its open issues attached —
never auto-approved, never silently dropped (I-4/I-5). Disagreement entries
in the draft are never issues by themselves and never trigger NEEDS_ATTENTION.

The loop depends on two injected callables, not on concrete agents, so the
graph layer (P9) wires ``run_critic``/``Director.revise`` and tests wire
fakes. Round/revision bookkeeping lives here and in the agents — never in
the LLM's hands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from consilium.context.router import RoutingConfig
from consilium.core.logging import get_logger
from consilium.schemas.agents import DirectorDraft, SubReport
from consilium.schemas.audit import AuditResult, CriticVerdict, Issue, MergedVerdict
from consilium.schemas.context import GlobalContext
from consilium.verify.deterministic import verify_draft
from consilium.vision.config import Thresholds

logger = get_logger(__name__)

CriticFn = Callable[[DirectorDraft, GlobalContext, int], Awaitable[CriticVerdict]]
ReviseFn = Callable[[DirectorDraft, tuple[Issue, ...], int], Awaitable[DirectorDraft]]


def merge_verdicts(
    deterministic_issues: tuple[Issue, ...],
    critic_verdict: CriticVerdict | None,
    round_: int,
) -> MergedVerdict:
    """§13.3: approved ⇔ no deterministic issues AND critic approved.

    ``critic_verdict`` is None exactly when the deterministic layer rejected
    first — the critic never ran, so the round is a rejection regardless.
    """
    critic_issues = critic_verdict.issues if critic_verdict is not None else ()
    issues = (*deterministic_issues, *critic_issues)
    approved = not deterministic_issues and critic_verdict is not None and critic_verdict.approved
    # Rejection without any issue object is unreachable: verify_draft returns
    # its reasons, a rejecting CriticVerdict must carry issues (schema), and
    # the loop passes critic_verdict=None only when det_issues is non-empty.
    # The MergedVerdict validator hard-fails if that ever changes.
    return MergedVerdict(
        approved=approved,
        issues=issues,
        round=round_,
        critic_ran=critic_verdict is not None,
    )


async def run_audit_loop(
    *,
    draft: DirectorDraft,
    global_ctx: GlobalContext,
    sub_reports: tuple[SubReport, ...],
    thresholds: Thresholds,
    routing: RoutingConfig,
    run_critic: CriticFn,
    revise: ReviseFn,
    max_rounds: int,
) -> AuditResult:
    """Bounded AUDITING ⇄ REVISING loop (§4). Fail closed on exhaustion."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    current = draft
    last: MergedVerdict | None = None
    for round_ in range(1, max_rounds + 1):
        det_issues = verify_draft(current, global_ctx, sub_reports, thresholds, routing)
        verdict = (
            merge_verdicts(det_issues, None, round_)
            if det_issues
            else merge_verdicts((), await run_critic(current, global_ctx, round_), round_)
        )
        logger.info(
            "audit.round",
            extra={
                "round": round_,
                "critic_ran": verdict.critic_ran,
                "approved": verdict.approved,
                "issue_count": len(verdict.issues),
            },
        )
        if verdict.approved:
            return AuditResult(draft=current, approved=True, rounds=round_)
        last = verdict
        if round_ < max_rounds:
            current = await revise(current, verdict.issues, round_)
    # Loop exhausted (§4): NEEDS_ATTENTION surface — last revision kept, open
    # issues attached, never approved, never dropped.
    assert last is not None
    logger.info(
        "audit.exhausted",
        extra={"rounds": max_rounds, "open_issue_count": len(last.issues)},
    )
    return AuditResult(
        draft=current,
        approved=False,
        rounds=max_rounds,
        open_issues=last.issues,
    )
