"""Fact and context contracts (AGENTS.md §7, §9).

The Context Builder turns ``CADResult`` + ``ClinicalNote`` into a
``GlobalContext`` (C_global) of ``Fact``s with stable, dotted IDs. The
Context Router projects it deterministically into per-role ``RoleContext``s
(C_k). Fact IDs are an API: they are defined ONLY in ``context/facts.py``
and never inlined elsewhere (§7 contract rules).
"""

from typing import Self

from pydantic import Field, model_validator

from consilium.schemas.base import Strict
from consilium.schemas.enums import FactSource, Laterality, Role


class Fact(Strict):
    """One typed, citable fact. Every claim in the pipeline cites these by ID."""

    id: str = Field(min_length=1)  # stable, dotted: "cad.cdr.vertical"
    label: str = Field(min_length=1)
    value: str | float | int | bool | tuple[str, ...]
    unit: str | None = None
    source: FactSource


def _ensure_unique_ids(facts: tuple[Fact, ...]) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for fact in facts:
        (dupes if fact.id in seen else seen).add(fact.id)
    if dupes:
        raise ValueError(f"duplicate fact IDs: {sorted(dupes)}")


class GlobalContext(Strict):
    """C_global — every fact for a case, before routing.

    Builder guarantees (enforced here at the schema boundary): fact IDs are
    unique. "Every CAD field represented" and caveat completeness are the
    builder's job (P4) and are verified by the deterministic verifier.
    """

    case_id: str = Field(min_length=1)
    laterality: Laterality
    facts: tuple[Fact, ...]

    @model_validator(mode="after")
    def _unique_fact_ids(self) -> Self:
        _ensure_unique_ids(self.facts)
        return self

    def fact_ids(self) -> frozenset[str]:
        return frozenset(f.id for f in self.facts)


class RoleContext(Strict):
    """C_k — the allowlist projection a single role is allowed to see (I-2).

    The router emits only allowlisted facts, so ``allowed_ids()`` — the set
    a specialist's ``evidence_refs``/``triggered_by`` must stay within — is
    exactly the IDs present here. This is the second, independent isolation
    enforcement (the router restricts what the agent sees; output validators
    reject citations outside this set).
    """

    role: Role
    case_id: str = Field(min_length=1)
    laterality: Laterality
    facts: tuple[Fact, ...]
    routing_version: str = Field(min_length=1)  # sha256 of routing.yaml

    @model_validator(mode="after")
    def _unique_fact_ids(self) -> Self:
        _ensure_unique_ids(self.facts)
        return self

    def allowed_ids(self) -> frozenset[str]:
        return frozenset(f.id for f in self.facts)
