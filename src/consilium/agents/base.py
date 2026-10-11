"""BaseSpecialist — the shared specialist agent (plan Section 5.4; AGENTS.md §11).

All three specialists (ophthalmologist, optometrist, pharmacist) are the same
class with a different prompt and a different ``RoleContext``. There is NO
role-specific parsing anywhere — role differences live in the prompt files and
in ``configs/routing.yaml`` (SKILLS.md specialist-agent, step 4).

One run is:

  1. render the versioned prompt with the routed fact block
     (``agents/templates.render_facts`` — trusted/untrusted split, I-7);
  2. structured-output call via ``generate_structured`` (transient retry,
     bounded repair, per-case token budget — llm/retry.py);
  3. semantic scope validation as an ``output_validator`` in the SAME repair
     loop: every ``evidence_refs`` / ``triggered_by`` must be ⊆
     ``RoleContext.allowed_ids()`` and the report's ``role`` must match the
     agent. This is the second, independent enforcement of isolation (the
     router restricts what the agent SEES; this rejects output that cites
     something it should not know — AGENTS.md §7 contract rules).

Scope violations exhaust the repair loop and surface as ``ValidationFailure``
(LLM_OUTPUT_INVALID → case FAILED), exactly like schema-invalid output —
fail closed, never silently degraded (I-5).

Logging rule: token counts, model ID, prompt version only — never message
bodies or note content (I-6/I-7).
"""

from __future__ import annotations

from collections.abc import Callable

from consilium.agents.templates import render_facts
from consilium.core.errors import AgentFailure
from consilium.llm.budget import TokenBudget
from consilium.llm.client import ChatMessage, LLMClient, ParsedResponse
from consilium.llm.prompt_loader import Prompt
from consilium.llm.retry import LLMRetryPolicy, generate_structured
from consilium.schemas.agents import SubReport
from consilium.schemas.context import RoleContext
from consilium.schemas.enums import Role

_MAX_OUTPUT_TOKENS = 2000  # SubReport is small; caps runaway generations

_USER_INSTRUCTION = (
    "Produce your SubReport for the case data in the system message. "
    "Return ONLY the JSON object described there."
)


class BaseSpecialist:
    """One isolated specialist role. Immutable after construction."""

    def __init__(
        self,
        *,
        role: Role,
        prompt: Prompt,
        llm: LLMClient,
        model: str,
        temperature: float,
        max_output_tokens: int = _MAX_OUTPUT_TOKENS,
        timeout_s: float = 60.0,
        retry_policy: LLMRetryPolicy | None = None,
        budget: TokenBudget | None = None,
        fixture: str | None = None,
    ) -> None:
        self.role = role
        self._prompt = prompt
        self._llm = llm
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s
        self._retry_policy = retry_policy
        self._budget = budget
        self._fixture = fixture

    def _scope_validator(self, ctx: RoleContext) -> Callable[[SubReport], str | None]:
        """Post-schema semantic check for the repair loop (see module docstring).

        Returns None when the report is in scope, else an error summary fed
        back to the model as a repair hint. The summary contains fact IDs and
        the role name only — never fact values or note content (I-6).
        """
        allowed = ctx.allowed_ids()

        def validate(report: SubReport) -> str | None:
            problems: list[str] = []
            if report.role is not self.role:
                problems.append(f"role must be {self.role.value!r}, got {report.role.value!r}")
            for i, finding in enumerate(report.findings):
                bad = sorted(set(finding.evidence_refs) - allowed)
                if bad:
                    problems.append(
                        f"findings[{i}].evidence_refs cite facts not in your block: {bad}"
                    )
            for i, rec in enumerate(report.recommendations):
                bad = sorted(set(rec.triggered_by) - allowed)
                if bad:
                    problems.append(
                        f"recommendations[{i}].triggered_by cite facts not in your block: {bad}"
                    )
            if not problems:
                return None
            return (
                "Scope violation: every evidence_refs/triggered_by entry must be one of the "
                "fact IDs shown in your case-data block. Problems: " + "; ".join(problems)
            )

        return validate

    async def run(self, ctx: RoleContext) -> ParsedResponse[SubReport]:
        """Render → generate → schema+scope validate (≤2 repairs) → SubReport."""
        if ctx.role is not self.role:
            raise AgentFailure(
                "SPECIALIST_CONTEXT_MISMATCH",
                detail=f"{self.role.value} agent received a {ctx.role.value} context",
            )
        system = self._prompt.render({"facts_block": render_facts(ctx)})
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=_USER_INSTRUCTION),
        ]
        tags = {
            "agent": self.role.value,
            "prompt_version": self._prompt.version,
        }
        if self._fixture is not None:
            tags["fixture"] = self._fixture
        return await generate_structured(
            self._llm,
            messages,
            SubReport,
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            timeout_s=self._timeout_s,
            tags=tags,
            retry_policy=self._retry_policy,
            budget=self._budget,
            output_validator=self._scope_validator(ctx),
        )
