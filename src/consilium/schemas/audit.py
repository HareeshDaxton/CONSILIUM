"""Audit contracts (AGENTS.md §7, §13).

The deterministic verifier runs FIRST (exact checks), then the LLM Critic
(entailment). ``Issue`` is the common currency of both layers;
``CriticVerdict`` merges per round. A draft passes only if both layers
produce zero issues (§13.3).
"""

from typing import Self

from pydantic import Field, model_validator

from consilium.schemas.agents import DirectorDraft
from consilium.schemas.base import Strict
from consilium.schemas.enums import IssueType

# §4/§17: banner the review UI must show when the audit loop exhausted
# without approval. Defined here (P8) so the P11 API surface reuses the
# exact wording instead of re-inventing it.
UNVERIFIED_BANNER = "UNVERIFIED — critic did not approve"


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


class MergedVerdict(Strict):
    """One audit round after merging both layers (§13.3).

    ``critic_ran`` records the fixed order: the deterministic verifier runs
    FIRST and a draft that fails it never reaches the LLM critic (cheap
    failures never burn critic tokens). Issues keep their original
    ``raised_by`` so the Director and the audit trail can tell layers apart.
    """

    approved: bool
    issues: tuple[Issue, ...] = ()
    round: int = Field(ge=1)
    critic_ran: bool

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.approved and self.issues:
            raise ValueError("approved verdict cannot carry issues")
        if not self.approved and not self.issues:
            raise ValueError("rejection must state at least one issue")
        if self.approved and not self.critic_ran:
            raise ValueError("a draft is approved only when BOTH layers passed")
        return self


class AuditResult(Strict):
    """Terminal state of the bounded AUDITING ⇄ REVISING loop (§4, §13.3).

    UNVERIFIED draft contract (I-4/I-5): on loop exhaustion the LAST draft
    revision is kept (never silently dropped, never auto-approved), the open
    issues stay attached, and ``review_banner`` carries the exact banner text
    the review UI must show. A draft whose only blemish is preserved
    ``Disagreement`` entries is approved normally — disagreements are never
    issues by themselves.
    """

    draft: DirectorDraft
    approved: bool
    rounds: int = Field(ge=1)
    open_issues: tuple[Issue, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.approved and self.open_issues:
            raise ValueError("an approved audit cannot carry open issues")
        if not self.approved and not self.open_issues:
            raise ValueError("an unapproved audit must attach its open issues")
        return self

    @property
    def unverified(self) -> bool:
        return not self.approved

    @property
    def review_banner(self) -> str | None:
        return None if self.approved else UNVERIFIED_BANNER
