"""LLM retry / repair / budget policy (plan Section 5.1; AGENTS.md §6/§11).

Hardened call wrappers every LLM call site shares:

  - ``generate_with_retry`` — retries *transient* failures (timeout, 429, 5xx,
    connection) with exponential backoff + jitter, bounded by a wall-clock
    budget that beats the attempt count, then fails closed with
    ``AgentFailure(LLM_UNAVAILABLE)`` (I-5). Refusals surface as
    ``AgentFailure`` from the client and are NEVER retried blindly or
    swallowed (SKILLS.md llm-client-and-fakes).
  - ``generate_structured`` — bounded repair retry (default ≤2) for output
    that fails schema validation: the validation summary is fed back to the
    model as a repair hint. Exhaustion → ``ValidationFailure`` (I-8).
  - Optional ``TokenBudget`` — per-case token ledger; checked before each
    attempt and charged after each successful response. Exceeding raises
    ``AgentFailure(LLM_TOKEN_BUDGET_EXCEEDED)``; output is never truncated.

Logging rule (SKILLS.md llm-client-and-fakes): token counts, model ID,
latency, attempt only — NEVER message bodies (I-6/I-7).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from consilium.core.errors import AgentFailure, ValidationFailure
from consilium.core.logging import get_logger
from consilium.core.settings import Settings
from consilium.llm.budget import TokenBudget
from consilium.llm.client import (
    ChatMessage,
    LLMClient,
    LLMTransient,
    ModelOutputInvalid,
    ParsedResponse,
)

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LLMRetryPolicy:
    """Bounded retry for transient failures. Total wall-clock budget wins over
    attempt count — a slow attempt consumes budget like any other."""

    total_budget_s: float = 60.0  # == LLM_TIMEOUT_S default
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.25  # +/- uniform jitter on each sleep
    max_attempts: int = 3  # == LLM_MAX_RETRIES default

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMRetryPolicy:
        """Map configured LLM_TIMEOUT_S / LLM_MAX_RETRIES onto the policy."""
        return cls(
            total_budget_s=settings.llm_timeout_s,
            max_attempts=settings.llm_max_retries,
        )


async def generate_with_retry(
    client: LLMClient,
    messages: list[ChatMessage],
    response_model: type[T],
    *,
    policy: LLMRetryPolicy,
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: float,
    tags: dict[str, str],
    budget: TokenBudget | None = None,
) -> ParsedResponse[T]:
    """Call ``client.generate`` retrying LLMTransient until the budget or
    attempt cap is exhausted, then fail closed with LLM_UNAVAILABLE (I-5)."""
    started = time.monotonic()
    backoff = policy.initial_backoff_s
    attempt = 0
    while True:
        if budget is not None:
            budget.check()
        attempt += 1
        try:
            resp = await client.generate(
                messages,
                response_model,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
                tags=tags,
            )
        except LLMTransient:
            elapsed = time.monotonic() - started
            if attempt >= policy.max_attempts or elapsed + backoff > policy.total_budget_s:
                logger.warning(
                    "llm.unavailable",
                    extra={"attempts": attempt, "elapsed_s": round(elapsed, 2)},
                )
                raise AgentFailure(
                    "LLM_UNAVAILABLE",
                    detail=f"model not reachable after {attempt} attempt(s) "
                    f"in {elapsed:.1f}s (budget {policy.total_budget_s:.1f}s)",
                ) from None
            sleep_s = backoff * (
                1.0
                + random.uniform(  # noqa: S311 — jitter, not crypto
                    -policy.jitter_fraction, policy.jitter_fraction
                )
            )
            logger.info("llm.retry", extra={"attempt": attempt, "sleep_s": round(sleep_s, 2)})
            await asyncio.sleep(sleep_s)
            backoff = min(backoff * policy.backoff_multiplier, policy.max_backoff_s)
            continue
        if budget is not None:
            budget.charge(resp.input_tokens, resp.output_tokens)
        return resp


_REPAIR_HINT = (
    "Your previous reply failed validation against the required schema. "
    "Validation error:\n{error}\n\nReturn ONLY a JSON object that conforms to "
    "the schema. Do not add prose. Do not invent values the note does not state."
)


async def generate_structured(
    client: LLMClient,
    messages: list[ChatMessage],
    response_model: type[T],
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: float,
    tags: dict[str, str],
    max_repairs: int = 2,
    retry_policy: LLMRetryPolicy | None = None,
    budget: TokenBudget | None = None,
) -> ParsedResponse[T]:
    """Structured call with bounded repair retries (AGENTS.md §10 step 3, I-8).

    Transient transport failures are retried first (``generate_with_retry``).
    Output that fails validation gets up to ``max_repairs`` repair attempts
    with the validation error fed back as a user message; exhaustion fails
    closed with ``ValidationFailure`` → case FAILED (I-5). When ``budget`` is
    given, every successful response (initial or repair) is charged against
    the shared per-case ledger.
    """
    policy = retry_policy or LLMRetryPolicy()
    conversation = list(messages)
    for attempt in range(1, max_repairs + 2):  # 1 initial + max_repairs repairs
        try:
            resp = await generate_with_retry(
                client,
                conversation,
                response_model,
                policy=policy,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
                tags=tags,
                budget=budget,
            )
        except ModelOutputInvalid as exc:
            if attempt > max_repairs:
                logger.warning("llm.output_invalid", extra={"attempts": attempt})
                raise ValidationFailure(
                    "LLM_OUTPUT_INVALID",
                    detail=f"schema validation failed after {attempt} attempt(s)",
                ) from None
            conversation = [
                *conversation,
                ChatMessage(role="user", content=_REPAIR_HINT.format(error=exc.error_summary)),
            ]
            continue
        return resp.model_copy(update={"attempts": attempt})
    raise AssertionError("unreachable")  # pragma: no cover
