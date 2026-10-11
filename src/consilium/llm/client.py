"""LLMClient protocol + shared types (AGENTS.md §6/§11; ARCHITECTURE.md §5.7).

Every LLM call in the system goes through the ``LLMClient`` protocol. Two
implementations exist:

  - ``client_openai.OpenAILLMClient`` — the ONLY OpenAI SDK importer in src/
    (import-linter enforces this).
  - ``client_fake.FakeLLMClient`` — deterministic canned outputs (dev/tests).

Retry / repair / budget policy lives in ``llm/retry.py`` (plan Section 5.1);
per-case token accounting in ``llm/budget.py``.

Logging rule (SKILLS.md llm-client-and-fakes): token counts, model ID,
latency, attempt only — NEVER message bodies (I-6/I-7).
"""

from __future__ import annotations

from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

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
