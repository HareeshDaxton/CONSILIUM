"""LLM Critic façade (plan Section 8.1; AGENTS.md §13.2).

Thin wrapper around ``agents.critic_agent.Critic``: one call per audit round,
logging issue COUNTS only — never issue content, claim text, or fact values
(I-6/I-7).

Design recorded from §13.2:
- the Critic runs on a different model or at minimum a different
  temperature/setting than the Director — enforced at startup by
  ``Settings._check_critic_differs_from_director`` (self-critique by the same
  configuration is weak evidence);
- it sees the draft's claims plus the GlobalContext fact sheet ONLY — never
  the specialists' reasoning;
- known limit: an LLM critic can approve a wrong statement. That is why the
  deterministic verifier (§13.1) runs FIRST, and why a human is final (I-4).

The bounded AUDITING ⇄ REVISING loop and the merge of deterministic + critic
verdicts (§13.3) live in the graph layer (P9/Section 8.2), not here.
"""

from consilium.agents.critic_agent import Critic
from consilium.core.logging import get_logger
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import CriticVerdict
from consilium.schemas.context import GlobalContext

logger = get_logger(__name__)


async def run_critic(
    draft: DirectorDraft,
    global_ctx: GlobalContext,
    round_: int,
    critic: Critic,
) -> CriticVerdict:
    """One audit round: delegate to the Critic agent, log counts, return the
    verdict (round already stamped by the agent)."""
    verdict = (await critic.run(draft, global_ctx, round_)).data
    logger.info(
        "critic.verdict",
        extra={
            "round": verdict.round,
            "approved": verdict.approved,
            "issue_count": len(verdict.issues),
        },
    )
    return verdict
