"""Director agent — draft + revise (plan Section 6.2; AGENTS.md §12).

The Director synthesizes the three isolated SubReports plus the read-only
GlobalContext fact sheet into a ``DirectorDraft``. Design decisions (§12):

- attribute, don't launder — claims carry ``source_roles`` and disagreements
  are preserved, never smoothed (prompt-enforced, tested in Section 6.3);
- revision mode receives the previous draft + the issue list ONLY — never
  the critic's reasoning — and must address each issue or justify non-change;
- ``revision`` is bookkeeping, stamped by this module (draft = 0, revise =
  previous + 1), never left to the model.

Schema validation, transient retry, bounded repair, and the per-case token
budget come from ``generate_structured`` — same hardening as the specialists.
Reference resolution / numeric checks against the fact sheet are the
deterministic verifier's job (P7); they are deliberately NOT duplicated here.
"""

from __future__ import annotations

from consilium.agents.templates import render_facts, render_subreports
from consilium.llm.budget import TokenBudget
from consilium.llm.client import ChatMessage, LLMClient, ParsedResponse
from consilium.llm.prompt_loader import Prompt
from consilium.llm.retry import LLMRetryPolicy, generate_structured
from consilium.schemas.agents import DirectorDraft, SubReport
from consilium.schemas.audit import Issue
from consilium.schemas.context import GlobalContext
from consilium.schemas.enums import Role

_MAX_OUTPUT_TOKENS = 4000  # DirectorDraft is larger than a SubReport

_USER_INSTRUCTION = (
    "Synthesize the DirectorDraft for the case data in the system message. "
    "Return ONLY the JSON object described there."
)


def _revision_block(draft: DirectorDraft, issues: tuple[Issue, ...], round_: int) -> str:
    """Previous draft + issue list ONLY (§12: never the critic's reasoning)."""
    lines = [
        f"You are revising draft revision {draft.revision} (audit round {round_}). "
        "Address EVERY issue below, or explicitly justify non-change in the "
        "revised draft. Previous draft:",
        draft.model_dump_json(),
        "Issues to address:",
    ]
    for i, issue in enumerate(issues, start=1):
        claim = f" claim={issue.claim_id}" if issue.claim_id else ""
        lines.append(f"- [i{i}] {issue.type.value}{claim}: {issue.detail}")
    lines.append(f'Return the full revised DirectorDraft with "revision": {draft.revision + 1}.')
    return "\n".join(lines)


class Director:
    """The synthesis agent. Immutable after construction."""

    def __init__(
        self,
        *,
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
        self._prompt = prompt
        self._llm = llm
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s
        self._retry_policy = retry_policy
        self._budget = budget
        self._fixture = fixture

    async def _call(
        self,
        revision_block: str,
        ctx: GlobalContext,
        reports: dict[Role, SubReport],
    ) -> ParsedResponse[DirectorDraft]:
        system = self._prompt.render(
            {
                "facts_block": render_facts(ctx),
                "subreports_block": render_subreports(reports),
                "revision_block": revision_block,
            }
        )
        tags = {"agent": "director", "prompt_version": self._prompt.version}
        if self._fixture is not None:
            tags["fixture"] = self._fixture
        return await generate_structured(
            self._llm,
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=_USER_INSTRUCTION),
            ],
            DirectorDraft,
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            timeout_s=self._timeout_s,
            tags=tags,
            retry_policy=self._retry_policy,
            budget=self._budget,
        )

    async def draft(
        self,
        sub_reports: dict[Role, SubReport],
        global_ctx: GlobalContext,
    ) -> ParsedResponse[DirectorDraft]:
        """First draft: revision stamped 0 regardless of model output."""
        resp = await self._call("This is a first draft.", global_ctx, sub_reports)
        return resp.model_copy(update={"data": resp.data.model_copy(update={"revision": 0})})

    async def revise(
        self,
        draft: DirectorDraft,
        issues: tuple[Issue, ...],
        round_: int,
        sub_reports: dict[Role, SubReport],
        global_ctx: GlobalContext,
    ) -> ParsedResponse[DirectorDraft]:
        """Revision round: previous draft + issue list only; revision += 1."""
        resp = await self._call(_revision_block(draft, issues, round_), global_ctx, sub_reports)
        return resp.model_copy(
            update={"data": resp.data.model_copy(update={"revision": draft.revision + 1})}
        )
