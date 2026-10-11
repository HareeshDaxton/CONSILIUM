"""OpenAI implementation of LLMClient (ARCHITECTURE.md §5.7).

The ONLY module in src/ that imports the OpenAI SDK (import-linter contract
``openai-only-in-llm``). Structured output uses the SDK's
``chat.completions.parse`` with ``response_format`` bound to the Pydantic
response model (verified against the pinned SDK, openai 3.27.0 — do not rely
on memory when upgrading).

Error mapping (SKILLS.md llm-client-and-fakes):
  - timeout / 429 / 5xx / connection        → LLMTransient (retryable)
  - refusal / content-filter finish         → AgentFailure(LLM_REFUSAL) — never retried
  - validation failure / truncated output   → ModelOutputInvalid (bounded repair)
  - other 4xx APIStatusError                → AgentFailure(LLM_REQUEST_REJECTED)

Model IDs come from config as dated snapshots (AGENTS.md §6) — never floating
aliases. Message bodies are never logged (I-6/I-7).
"""

from __future__ import annotations

import time
from typing import TypeVar, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from consilium.core.errors import AgentFailure
from consilium.llm.client import (
    ChatMessage,
    LLMTransient,
    ModelOutputInvalid,
    ParsedResponse,
)

T = TypeVar("T", bound=BaseModel)

_TRANSIENT_ERRORS = (
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)


class OpenAILLMClient:
    """LLMClient backed by the OpenAI Chat Completions structured-output API."""

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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
        started = time.monotonic()
        try:
            completion = await self._client.chat.completions.parse(
                model=model,
                messages=cast(
                    "list[ChatCompletionMessageParam]",
                    [m.to_param() for m in messages],
                ),
                response_format=response_model,
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
                timeout=timeout_s,
            )
        except _TRANSIENT_ERRORS as exc:
            raise LLMTransient(type(exc).__name__) from exc
        except openai.ContentFilterFinishReasonError as exc:
            raise AgentFailure("LLM_REFUSAL", detail="content-filter finish") from exc
        except openai.LengthFinishReasonError as exc:
            raise ModelOutputInvalid("output truncated at the token limit") from exc
        except ValidationError as exc:
            raise ModelOutputInvalid(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise AgentFailure(
                "LLM_REQUEST_REJECTED", detail=f"api status {exc.status_code}"
            ) from exc
        latency_ms = (time.monotonic() - started) * 1000.0

        message = completion.choices[0].message
        if message.refusal is not None:
            raise AgentFailure("LLM_REFUSAL", detail="model refusal")
        if message.parsed is None:
            raise ModelOutputInvalid("no parsed object returned for the response schema")

        usage = completion.usage
        return ParsedResponse(
            data=message.parsed,
            model=model,
            prompt_version=tags.get("prompt_version"),
            input_tokens=usage.prompt_tokens if usage is not None else 0,
            output_tokens=usage.completion_tokens if usage is not None else 0,
            latency_ms=latency_ms,
            attempts=1,
        )
