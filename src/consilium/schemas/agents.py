"""Agent output contracts (AGENTS.md §7, §11-12).

``SubReport`` is what each isolated specialist returns; ``DirectorDraft`` is
what the Director synthesizes from the three sub-reports plus the read-only
C_global fact sheet. Both are LLM-facing: strict, bounded, and every factual
statement carries evidence refs resolvable against the author's context
(I-3) — enforcement of ref validity lives in the output validators and the
deterministic verifier, not here.
"""

from typing import Self

from pydantic import Field, model_validator

from consilium.schemas.base import Strict
from consilium.schemas.enums import ClaimKind, Confidence, Role


class Finding(Strict):
    text: str = Field(min_length=1, max_length=600)
    evidence_refs: tuple[str, ...] = Field(min_length=1)  # fact IDs (I-3)
    confidence: Confidence


class Recommendation(Strict):
    text: str = Field(min_length=1, max_length=500)
    triggered_by: tuple[str, ...] = Field(min_length=1)  # fact IDs that motivate it
    basis: str = "model_knowledge"  # V1: always; flagged for clinician review


class SubReport(Strict):
    """One specialist's structured output (never free text, I-8)."""

    role: Role
    findings: tuple[Finding, ...]
    recommendations: tuple[Recommendation, ...]
    uncertainties: tuple[str, ...] = ()
    declined_out_of_scope: tuple[str, ...] = ()  # noticed but not allowed to judge


class Claim(Strict):
    """One attributable statement in a DirectorDraft.

    Attribute, don't launder (§12): ``source_roles`` records who asserted
    it. NUMERIC claims must pin their number to a fact ID so the
    deterministic verifier can compare within tolerance.
    """

    id: str = Field(min_length=1)  # "c1", "c2", ... unique within the draft
    kind: ClaimKind
    text: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)  # resolve against GlobalContext
    source_roles: tuple[Role, ...] = Field(min_length=1)
    numeric_value: float | None = None  # REQUIRED when kind == NUMERIC
    numeric_ref: str | None = None  # fact ID the number must equal (REQUIRED for NUMERIC)

    @model_validator(mode="after")
    def _numeric_requirements(self) -> Self:
        if self.kind is ClaimKind.NUMERIC and (
            self.numeric_value is None or self.numeric_ref is None
        ):
            raise ValueError("NUMERIC claim requires numeric_value and numeric_ref")
        return self


class Disagreement(Strict):
    """A preserved, visible role disagreement — a safety feature, not an
    audit failure (§4, §12)."""

    topic: str = Field(min_length=1)
    positions: tuple[tuple[Role, str], ...] = Field(min_length=2)
    resolution: str = Field(min_length=1)  # how the draft handles it, or "left to clinician"

    @model_validator(mode="after")
    def _distinct_roles(self) -> Self:
        roles = [role for role, _ in self.positions]
        if len(set(roles)) != len(roles):
            raise ValueError("disagreement positions must come from distinct roles")
        return self


class DirectorDraft(Strict):
    """The synthesized draft. Never final without clinician action (I-4).

    ``limitations`` is mandatory (no default): it must carry every QC caveat
    from context and screening-support wording; the deterministic verifier
    checks coverage.
    """

    impression: str = Field(min_length=1)  # screening-support wording only (I-12)
    claims: tuple[Claim, ...]
    disagreements: tuple[Disagreement, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    limitations: tuple[str, ...]
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _unique_claim_ids(self) -> Self:
        ids = [c.id for c in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError("claim IDs must be unique within a draft")
        return self
