"""Critic agent — adversarial semantic audit (plan Section 8.1; AGENTS.md §13.2).

The Critic audits a ``DirectorDraft`` against the read-only ``GlobalContext``
fact sheet. Design decisions (§13.2):

- the Critic sees ONLY the draft's claims (+ disagreements, impression,
  limitations) and the fact sheet — never specialist prose, never the
  Director's reasoning — so its judgement stays independent of the story the
  draft tells itself;
- its job is semantic entailment, not exactness: numeric tolerance math and
  the forbidden-language lexicon belong to the deterministic verifier, which
  runs FIRST (§13.1) — the Critic never duplicates them;
- it runs on a different model or at minimum a different generation setting
  than the Director — enforced by ``Settings._check_critic_differs_from_director``;
- ``round`` is bookkeeping, stamped by this module, never left to the model
  (same discipline as the Director's ``revision``).

Semantic validation as an ``output_validator`` in the same bounded repair
loop as the specialists (I-8): every issue's ``claim_id`` must be null or
resolve to an existing claim ID of the draft under audit, and every
``evidence`` entry must be a fact ID present in the GlobalContext. Repair
hints name IDs only — never values (I-6). Exhaustion fails closed with
``ValidationFailure`` (I-5).

Known limit (§13.2): an LLM critic can approve a wrong statement — that is
why the deterministic layer runs first and a human is final (I-4).

Logging rule: token counts, model ID, prompt version only — never message
bodies or note content (I-6/I-7).
"""

from __future__ import annotations

from collections.abc import Callable

from consilium.agents.templates import render_claims, render_facts
from consilium.llm.budget import TokenBudget
from consilium.llm.client import ChatMessage, LLMClient, ParsedResponse
from consilium.llm.prompt_loader import Prompt
from consilium.llm.retry import LLMRetryPolicy, generate_structured
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import CriticVerdict
from consilium.schemas.context import GlobalContext

_MAX_OUTPUT_TOKENS = 3000  # verdict is small; caps runaway generations

_USER_INSTRUCTION = (
    "Audit the draft in the system message against the fact sheet. "
    "Return ONLY the CriticVerdict JSON object described there."
)


class Critic:
    """The adversarial audit agent. Immutable after construction."""

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

    @staticmethod
    def _reference_validator(
        draft: DirectorDraft, global_ctx: GlobalContext
    ) -> Callable[[CriticVerdict], str | None]:
        """Post-schema semantic check for the repair loop (see module docstring).

        Returns None when the verdict's references resolve, else an error
        summary fed back to the model as a repair hint. The summary contains
        claim IDs and fact IDs only — never values or note content (I-6).
        """
        claim_ids = {claim.id for claim in draft.claims}
        fact_ids = global_ctx.fact_ids()

        def validate(verdict: CriticVerdict) -> str | None:
            problems: list[str] = []
            for i, issue in enumerate(verdict.issues):
                if issue.claim_id is not None and issue.claim_id not in claim_ids:
                    problems.append(
                        f"issues[{i}].claim_id {issue.claim_id!r} is not a claim "
                        "ID in the draft under audit"
                    )
                bad = sorted(set(issue.evidence) - fact_ids)
                if bad:
                    problems.append(f"issues[{i}].evidence cite facts not on the fact sheet: {bad}")
            if not problems:
                return None
            return (
                "Reference violation: every issue's claim_id must be null or one of the "
                "draft's claim IDs, and every evidence entry must be a fact ID from the "
                "fact sheet. Problems: " + "; ".join(problems)
            )

        return validate

    async def run(
        self,
        draft: DirectorDraft,
        global_ctx: GlobalContext,
        round_: int,
    ) -> ParsedResponse[CriticVerdict]:
        """Audit one draft revision. ``round`` is stamped by code, overriding
        whatever the model returned (same discipline as Director.revision)."""
        system = self._prompt.render(
            {
                "facts_block": render_facts(global_ctx),
                "claims_block": render_claims(draft),
            }
        )
        tags = {"agent": "critic", "prompt_version": self._prompt.version}
        if self._fixture is not None:
            tags["fixture"] = self._fixture
        resp = await generate_structured(
            self._llm,
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=_USER_INSTRUCTION),
            ],
            CriticVerdict,
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            timeout_s=self._timeout_s,
            tags=tags,
            retry_policy=self._retry_policy,
            budget=self._budget,
            output_validator=self._reference_validator(draft, global_ctx),
        )
        return resp.model_copy(update={"data": resp.data.model_copy(update={"round": round_})})
