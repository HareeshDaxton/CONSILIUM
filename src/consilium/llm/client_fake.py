"""Fake LLMClient for dev/tests (ARCHITECTURE.md §5.7; AGENTS.md §20).

Deterministic canned responses keyed by ``(agent, prompt_version, fixture)``
taken from the call ``tags`` (fixture defaults to ``"default"`` so production-
style call sites need no test-only tag). Modes per SKILLS.md
llm-client-and-fakes: valid / malformed / schema_violating / timeout / refusal.

A canned response is a SEQUENCE of calls (e.g. timeout → valid) so retry and
repair paths are exercisable; the last call repeats when the sequence runs out.

The fake records every message list it receives (``received``) so tests can
assert what the LLM would have seen — e.g. that PHI was redacted BEFORE the
call (I-6). The default test suite never touches the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel

from consilium.core.errors import AgentFailure
from consilium.llm.client import (
    ChatMessage,
    LLMTransient,
    ModelOutputInvalid,
    ParsedResponse,
)

T = TypeVar("T", bound=BaseModel)

FakeKey = tuple[str, str, str]  # (agent, prompt_version, fixture)


class FakeMode(StrEnum):
    VALID = "valid"
    MALFORMED = "malformed"  # unparseable output
    SCHEMA_VIOLATING = "schema_violating"  # parseable JSON, wrong shape
    TIMEOUT = "timeout"  # transient → retryable
    REFUSAL = "refusal"  # permanent → AgentFailure, never retried


@dataclass(frozen=True, slots=True)
class CannedCall:
    mode: FakeMode
    payload: dict[str, Any] | None = None  # required for VALID


@dataclass(frozen=True, slots=True)
class CannedResponse:
    """Ordered calls; the last one repeats when exhausted."""

    calls: tuple[CannedCall, ...]

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError("CannedResponse needs at least one call")


@dataclass(slots=True)
class ReceivedCall:
    key: FakeKey
    messages: list[ChatMessage]


@dataclass(slots=True)
class FakeLLMClient:
    responses: dict[FakeKey, CannedResponse]
    received: list[ReceivedCall] = field(default_factory=list)
    _cursors: dict[FakeKey, int] = field(default_factory=dict)

    @classmethod
    def valid(
        cls,
        payload: dict[str, Any],
        *,
        agent: str,
        prompt_version: str,
        fixture: str = "default",
    ) -> FakeLLMClient:
        """Shortcut: one canned VALID response at one key."""
        key = (agent, prompt_version, fixture)
        return cls({key: CannedResponse((CannedCall(FakeMode.VALID, payload),))})

    def _key(self, tags: dict[str, str]) -> FakeKey:
        return (
            tags.get("agent", ""),
            tags.get("prompt_version", ""),
            tags.get("fixture", "default"),
        )

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
        key = self._key(tags)
        if key not in self.responses:
            raise ValueError(f"FakeLLMClient: no canned response for key {key!r}")
        self.received.append(ReceivedCall(key=key, messages=list(messages)))

        canned = self.responses[key]
        cursor = self._cursors.get(key, 0)
        call = canned.calls[min(cursor, len(canned.calls) - 1)]
        self._cursors[key] = cursor + 1

        if call.mode is FakeMode.TIMEOUT:
            raise LLMTransient("canned timeout")
        if call.mode is FakeMode.REFUSAL:
            raise AgentFailure("LLM_REFUSAL", detail="canned refusal")
        if call.mode is FakeMode.MALFORMED:
            raise ModelOutputInvalid("canned malformed output: not valid JSON")
        if call.mode is FakeMode.SCHEMA_VIOLATING:
            raise ModelOutputInvalid("canned schema violation: output does not match the schema")

        if call.payload is None:
            raise ValueError("FakeLLMClient: VALID call needs a payload")
        # Validate in JSON mode, exactly like the SDK's parse path
        # (openai.lib._parsing: TypeAdapter(response_format).validate_json) —
        # JSON arrays for tuple fields, no Python-mode surprises.
        data = response_model.model_validate_json(json.dumps(call.payload))
        return ParsedResponse(
            data=data,
            model=model,
            prompt_version=key[1],
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(str(call.payload)) // 4,
            latency_ms=1.0,
            attempts=1,
        )
