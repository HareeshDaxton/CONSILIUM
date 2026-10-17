"""Langfuse LLM observability sink (AGENTS.md §18; ARCHITECTURE.md §5.13).

Metadata ONLY by default: agent, model snapshot, prompt version, token usage,
latency, attempt, status. Prompt and response bodies are never sent (I-6/I-7)
— self-hosted Langfuse is required before real note text could ever be
enabled, and that is a separate, explicit decision.

Telemetry is best-effort: a Langfuse outage must never fail a case, so the
SDK sink converts any SDK error into a logged warning (exception TYPE only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from consilium.core.logging import get_logger
from consilium.core.settings import Settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class Generation:
    agent: str
    model: str
    prompt_version: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str  # "ok" | error class name


class LangfuseSink(Protocol):
    def generation(self, gen: Generation) -> None: ...


class NoopSink:
    def generation(self, gen: Generation) -> None:
        return None


class LangfuseSdkSink:
    """Wraps the Langfuse SDK client (v4 ``start_observation`` API)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def generation(self, gen: Generation) -> None:
        try:
            obs = self._client.start_observation(
                name=gen.agent,
                as_type="generation",
                model=gen.model,
                metadata={
                    "prompt_version": gen.prompt_version,
                    "latency_ms": gen.latency_ms,
                    "status": gen.status,
                },
                usage_details={"input": gen.input_tokens, "output": gen.output_tokens},
            )
            obs.end()
        except Exception as exc:  # best-effort telemetry boundary: never fail a case
            logger.warning("langfuse.emit_failed", extra={"error": type(exc).__name__})


def sink_from_settings(settings: Settings) -> LangfuseSink:
    """SDK sink when host + keys are configured, else no-op."""
    if not (
        settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key
    ):
        return NoopSink()
    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    return LangfuseSdkSink(client)
