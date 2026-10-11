"""LLMClient protocol + shared call helpers (AGENTS.md §6/§11; ARCHITECTURE.md §5.7).

Every LLM call in the system goes through the ``LLMClient`` protocol. Two
implementations exist:

  - ``client_openai.OpenAILLMClient`` — the ONLY OpenAI SDK importer in src/
    (import-linter enforces this).
  - ``client_fake.FakeLLMClient`` — deterministic canned outputs (dev/tests).

Shared helpers here give every call site the same behavior:

  - ``generate_with_retry`` — retries *transient* failures (timeout, 429, 5xx,
    connection) with exponential backoff + jitter, bounded by a wall-clock
    budget, then fails closed with ``AgentFailure(LLM_UNAVAILABLE)`` (I-5).
  - ``generate_structured`` — bounded repair retry (default ≤2) for output
    that fails schema validation: the validation summary is fed back to the
    model as a repair hint. Exhaustion → ``ValidationFailure`` (I-8).

Logging rule (SKILLS.md llm-client-and-fakes): token counts, model ID,
latency, attempt only — NEVER message bodies (I-6/I-7).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from consilium.core.errors import AgentFailure, ValidationFailure
from consilium.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ChatMessage(BaseModel):
    """One chat message. LLM-layer internal type (not a cross-component contract)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str

    def to_param(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ParsedResponse(BaseModel, Generic[T]):
    """Validated model output plus usage metadata for agent_runs/observability."""

    model_config = ConfigDict(frozen=True)

    data: T
    model: str
    prompt_version: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    attempts: int


class LLMTransient(Exception):
    """Retryable LLM call failure (timeout, 429, 5xx, connection error).

    Carries no payloads — detail strings must never contain note text or
    message bodies (I-6/I-7).
    """


class ModelOutputInvalid(Exception):
    """Model output failed schema validation (malformed JSON or schema-violating).

    The summary is fed back to the model as a repair hint ONLY — it may echo
    note-derived content, so it is never logged or stored (I-6).
    """

    def __init__(self, error_summary: str) -> None:
        super().__init__("model output failed schema validation")
        self.error_summary = error_summary


class LLMClient(Protocol):
    """The only interface any component uses for LLM calls."""

    async def generate(
        self,
        messages: list[ChatMessage],
        response_model: type[T],
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
        tags: dict[str, str],
    ) -> ParsedResponse[T]:
        """One structured-output call. Raises LLMTransient (retryable),
        ModelOutputInvalid (repairable), or AgentFailure (permanent: refusal,
        rejected request)."""
        ...


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
) -> ParsedResponse[T]:
    """Call ``client.generate`` retrying LLMTransient until the budget or
    attempt cap is exhausted, then fail closed with LLM_UNAVAILABLE (I-5)."""
    started = time.monotonic()
    backoff = policy.initial_backoff_s
    attempt = 0
    while True:
        attempt += 1
        try:
            return await client.generate(
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
) -> ParsedResponse[T]:
    """Structured call with bounded repair retries (AGENTS.md §10 step 3, I-8).

    Transient transport failures are retried first (``generate_with_retry``).
    Output that fails validation gets up to ``max_repairs`` repair attempts
    with the validation error fed back as a user message; exhaustion fails
    closed with ``ValidationFailure`` → case FAILED (I-5).
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
