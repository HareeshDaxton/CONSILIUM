"""Instrumented LLM + vision clients (AGENTS.md §18; ARCHITECTURE.md §5.13).

Decorators over the ``LLMClient`` / ``VisionClient`` protocols, applied once
at the composition root (``graph/deps.py``). Each call gets an OTel span and
Prometheus metrics; LLM calls are also reported to the Langfuse sink.

Span attributes are identities and counts only: agent, model, prompt
version, token counts, status, endpoint revision, cold-start flag. Never
message bodies, note text, images or model output (I-6/I-14).
"""

from __future__ import annotations

import time
from typing import TypeVar

from opentelemetry import trace
from pydantic import BaseModel

from consilium.core.errors import VisionFailure
from consilium.llm.client import ChatMessage, LLMClient, ParsedResponse
from consilium.observability import metrics as m
from consilium.observability.langfuse_sink import Generation, LangfuseSink, NoopSink
from consilium.schemas.vision import RawVisionOutput
from consilium.vision.client import VisionClient, VisionTransient

T = TypeVar("T", bound=BaseModel)

_tracer = trace.get_tracer("consilium.pipeline")


class InstrumentedLLM:
    def __init__(self, inner: LLMClient, sink: LangfuseSink | None = None) -> None:
        self._inner = inner
        self._sink = sink or NoopSink()

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
        agent = tags.get("agent", "unknown")
        started = time.monotonic()
        with _tracer.start_as_current_span(f"llm {agent}") as span:
            span.set_attribute("llm.agent", agent)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_version", tags.get("prompt_version", ""))
            try:
                resp = await self._inner.generate(
                    messages,
                    response_model,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    timeout_s=timeout_s,
                    tags=tags,
                )
            except Exception as exc:
                status = type(exc).__name__
                span.set_attribute("llm.status", status)
                m.LLM_CALLS.labels(agent=agent, status=status).inc()
                self._sink.generation(
                    Generation(agent, model, tags.get("prompt_version"), 0, 0, 0.0, status)
                )
                raise
            seconds = time.monotonic() - started
            span.set_attribute("llm.status", "ok")
            span.set_attribute("llm.input_tokens", resp.input_tokens)
            span.set_attribute("llm.output_tokens", resp.output_tokens)
        m.LLM_CALLS.labels(agent=agent, status="ok").inc()
        m.LLM_TOKENS.labels(agent=agent, model=model, kind="input").inc(resp.input_tokens)
        m.LLM_TOKENS.labels(agent=agent, model=model, kind="output").inc(resp.output_tokens)
        m.LLM_LATENCY.labels(agent=agent).observe(seconds)
        self._sink.generation(
            Generation(
                agent,
                model,
                resp.prompt_version,
                resp.input_tokens,
                resp.output_tokens,
                seconds * 1000,
                "ok",
            )
        )
        return resp


class InstrumentedVision:
    def __init__(self, inner: VisionClient) -> None:
        self._inner = inner

    async def predict(self, image_bytes: bytes) -> RawVisionOutput:
        started = time.monotonic()
        with _tracer.start_as_current_span("vision predict") as span:
            try:
                out = await self._inner.predict(image_bytes)
            except VisionTransient:
                span.set_attribute("vision.cold_start", True)
                m.VISION_COLD_START.inc()
                m.VISION_CALLS.labels(status="transient").inc()
                raise
            except VisionFailure as exc:
                span.set_attribute("vision.status", exc.reason)
                m.VISION_CALLS.labels(status="failed").inc()
                if "REVISION" in exc.reason:
                    m.VISION_REVISION_MISMATCH.inc()
                raise
            span.set_attribute("vision.endpoint_revision", out.endpoint_revision)
            span.set_attribute("vision.cold_start", False)
        m.VISION_CALLS.labels(status="ok").inc()
        m.VISION_LATENCY.observe(time.monotonic() - started)
        return out
