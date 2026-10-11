"""Context Router (AGENTS.md §9.2, I-2; ARCHITECTURE.md §5.6).

Deterministic allowlist projection of ``GlobalContext`` into per-role
``RoleContext``s. Pure function: no LLM, no heuristics, no fallback —
deny-by-default: a fact not explicitly granted to a role is not delivered.
``routing_version`` = sha256 of the raw routing.yaml bytes, stamped on every
RoleContext so the exact policy is auditable per run.
"""

import fnmatch
import hashlib
from pathlib import Path

import yaml
from pydantic import field_validator

from consilium.schemas.base import Strict
from consilium.schemas.context import GlobalContext, RoleContext
from consilium.schemas.enums import Role


class RoleRouting(Strict):
    allow: tuple[str, ...]

    @field_validator("allow", mode="before")
    @classmethod
    def _list_to_tuple(cls, v: object) -> object:
        return tuple(v) if isinstance(v, list) else v


class RoutingConfig(Strict):
    version: str  # sha256 hex of the raw YAML bytes
    roles: dict[Role, RoleRouting]

    @field_validator("roles", mode="before")
    @classmethod
    def _coerce_role_keys(cls, v: object) -> object:
        if isinstance(v, dict):
            return {Role(k): val for k, val in v.items()}
        return v


def load_routing_config(path: Path) -> RoutingConfig:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or "roles" not in data:
        raise ValueError(f"routing config {path} must define 'roles'")
    return RoutingConfig(version=hashlib.sha256(raw).hexdigest(), roles=data["roles"])


def _allowed(patterns: tuple[str, ...], fact_id: str) -> bool:
    return any(fnmatch.fnmatchcase(fact_id, pat) for pat in patterns)


def route(global_ctx: GlobalContext, cfg: RoutingConfig) -> dict[Role, RoleContext]:
    """Project C_global into C_k per role. Allowlist only; fact order preserved."""
    out: dict[Role, RoleContext] = {}
    for role, policy in cfg.roles.items():
        facts = tuple(f for f in global_ctx.facts if _allowed(policy.allow, f.id))
        out[role] = RoleContext(
            role=role,
            case_id=global_ctx.case_id,
            laterality=global_ctx.laterality,
            facts=facts,
            routing_version=cfg.version,
        )
    return out
