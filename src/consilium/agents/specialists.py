"""Specialist fan-out (plan Section 5.4; AGENTS.md §11).

The three specialists run concurrently (``asyncio.gather``) and never see each
other's output. Isolation rule (I-5, fail closed):

  - a missing ``RoleContext`` for any specialist is a wiring error → the case
    fails, it is never silently skipped;
  - if any specialist fails, the case fails (``SpecialistFanOutFailure``) —
    but the SIBLING outputs are attached to the exception so the graph node
    (P9) can persist them before marking the case FAILED.

No LLM/vision imports beyond the llm layer; agents receive facts, never CAD
machinery (§5 hard boundary).
"""

from __future__ import annotations

import asyncio

from consilium.agents.base import BaseSpecialist
from consilium.core.errors import AgentFailure, AppError
from consilium.core.settings import Settings
from consilium.llm.budget import TokenBudget
from consilium.llm.client import LLMClient, ParsedResponse
from consilium.llm.prompt_loader import PromptLoader
from consilium.llm.retry import LLMRetryPolicy
from consilium.schemas.agents import SubReport
from consilium.schemas.context import RoleContext
from consilium.schemas.enums import Role

SPECIALIST_PROMPT_VERSION = "v1"  # bump via prompt-change skill, never in place

# Role → prompts/<component>/ (names deliberately identical to the role values)
SPECIALIST_ROLES: tuple[Role, ...] = (
    Role.OPHTHALMOLOGIST,
    Role.OPTOMETRIST,
    Role.PHARMACIST,
)


class SpecialistFanOutFailure(AgentFailure):
    """One or more specialists failed (I-5: one failure fails the case).

    Carries the successful sibling reports so the caller (graph node, P9) can
    persist them before the case goes FAILED. Contains IDs/reasons only —
    never note text or message bodies (I-6).
    """

    def __init__(
        self,
        *,
        succeeded: dict[Role, ParsedResponse[SubReport]],
        failed: dict[Role, AppError],
    ) -> None:
        failed_roles = sorted(r.value for r in failed)
        reasons = sorted({e.reason for e in failed.values()})
        super().__init__(
            "SPECIALIST_FAILED",
            detail=f"specialists failed: {failed_roles} (reasons: {reasons})",
        )
        self.succeeded = succeeded
        self.failed = failed


def build_specialists(
    *,
    settings: Settings,
    llm: LLMClient,
    prompt_loader: PromptLoader | None = None,
    retry_policy: LLMRetryPolicy | None = None,
    budget: TokenBudget | None = None,
    fixture: str | None = None,
) -> dict[Role, BaseSpecialist]:
    """Construct the three specialists from settings (model snapshot +
    temperature are config, never hard-coded). One shared per-case ``budget``
    may be injected so all specialists charge the same ledger (§11)."""
    loader = prompt_loader or PromptLoader()
    return {
        role: BaseSpecialist(
            role=role,
            prompt=loader.load(role.value, SPECIALIST_PROMPT_VERSION),
            llm=llm,
            model=settings.llm_model_specialist,
            temperature=settings.llm_temperature_specialist,
            retry_policy=retry_policy,
            budget=budget,
            fixture=fixture,
        )
        for role in SPECIALIST_ROLES
    }


async def run_all(
    contexts: dict[Role, RoleContext],
    specialists: dict[Role, BaseSpecialist],
) -> dict[Role, ParsedResponse[SubReport]]:
    """Run every specialist concurrently against its routed context.

    Returns one ParsedResponse[SubReport] per role on full success. Fail
    closed otherwise: missing context or any specialist failure raises before
    any partial result is returned (sibling outputs ride on the exception).
    """
    missing = [role for role in specialists if role not in contexts]
    if missing:
        raise AgentFailure(
            "SPECIALIST_CONTEXT_MISSING",
            detail=f"no RoleContext routed for: {sorted(r.value for r in missing)}",
        )

    roles = list(specialists)
    results = await asyncio.gather(
        *(specialists[role].run(contexts[role]) for role in roles),
        return_exceptions=True,
    )

    reports: dict[Role, ParsedResponse[SubReport]] = {}
    failures: dict[Role, AppError] = {}
    for role, result in zip(roles, results, strict=True):
        if isinstance(result, AppError):
            failures[role] = result
        elif isinstance(result, BaseException):
            # Unknown error: converted at the graph-node boundary (§21) — but
            # never lose sibling outputs, so fail the fan-out here too.
            failures[role] = AgentFailure(
                "SPECIALIST_FAILED",
                detail=f"{role.value} raised {type(result).__name__}",
            )
        else:
            reports[role] = result

    if failures:
        raise SpecialistFanOutFailure(succeeded=reports, failed=failures)
    return reports
