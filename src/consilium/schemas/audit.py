"""Audit contracts (AGENTS.md §7, §13).

The deterministic verifier runs FIRST (exact checks), then the LLM Critic
(entailment). ``Issue`` is the common currency of both layers;
``CriticVerdict`` merges per round. A draft passes only if both layers
produce zero issues (§13.3).
"""

from typing import Self

from pydantic import Field, model_validator

from consilium.schemas.base import Strict
from consilium.schemas.enums import IssueType


class Issue(Strict):
    type: IssueType
    claim_id: str | None  # None for draft-level issues (e.g. omitted finding)
    detail: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    raised_by: str = Field(min_length=1)  # "deterministic" | "llm_critic"


class CriticVerdict(Strict):
    """Consistency rule (§7): approved ⇔ zero issues. Enforced here so an
    incoherent verdict can never be constructed, stored, or merged."""

    approved: bool
    issues: tuple[Issue, ...] = ()
    round: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.approved and self.issues:
            raise ValueError("approved verdict cannot carry issues")
        if not self.approved and not self.issues:
            raise ValueError("rejection must state at least one issue")
        return self
