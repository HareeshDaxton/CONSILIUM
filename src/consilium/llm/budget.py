"""Per-case token budget enforcement (AGENTS.md §11; plan Section 5.1).

``LLM_CASE_TOKEN_BUDGET`` bounds the total tokens one case may consume across
ALL LLM calls (note extraction, specialists, director, critic, repair rounds).
Exceeding the budget RAISES ``AgentFailure(LLM_TOKEN_BUDGET_EXCEEDED)`` — the
case fails closed (I-5). A budget is never enforced by truncating output:
silent truncation of a clinical report is worse than a failed case.

Budgets are charged from the usage metadata of SUCCESSFUL responses only
(input + output tokens as reported by the provider / fake). Failed attempts
have no trustworthy usage to charge; the wall-clock retry budget bounds their
cost in time.
"""

from __future__ import annotations

from dataclasses import dataclass

from consilium.core.errors import AgentFailure
from consilium.core.logging import get_logger
from consilium.core.settings import Settings

logger = get_logger(__name__)


@dataclass(slots=True)
class TokenBudget:
    """Running token ledger for one case. Created per case, shared by every
    LLM call site in the pipeline run."""

    limit_tokens: int
    used_tokens: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> TokenBudget:
        return cls(limit_tokens=settings.llm_case_token_budget)

    @property
    def remaining(self) -> int:
        return self.limit_tokens - self.used_tokens

    def check(self) -> None:
        """Pre-call guard: never place another call once the budget is gone."""
        if self.used_tokens >= self.limit_tokens:
            logger.warning("llm.budget_exhausted", extra={"used_tokens": self.used_tokens})
            raise AgentFailure(
                "LLM_TOKEN_BUDGET_EXCEEDED",
                detail=f"case token budget of {self.limit_tokens} already consumed",
            )

    def charge(self, input_tokens: int, output_tokens: int) -> None:
        """Charge a successful response; exceeding the limit raises (I-5)."""
        self.used_tokens += input_tokens + output_tokens
        if self.used_tokens > self.limit_tokens:
            logger.warning("llm.budget_exceeded", extra={"used_tokens": self.used_tokens})
            raise AgentFailure(
                "LLM_TOKEN_BUDGET_EXCEEDED",
                detail=f"case token budget exceeded: {self.used_tokens} > {self.limit_tokens}",
            )
