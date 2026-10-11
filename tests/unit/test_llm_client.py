"""LLMClient protocol/fake/retry/repair tests (Section 3.2).

Offline only: the OpenAI impl is compile-checked and exercised by a
live-tagged contract stub (never run in the default suite — AGENTS.md §20).
"""

from typing import Any

import pytest

from consilium.core.errors import AgentFailure, ValidationFailure
from consilium.llm.client import (
    ChatMessage,
    LLMRetryPolicy,
    generate_structured,
    generate_with_retry,
)
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.llm.client_openai import OpenAILLMClient
from consilium.schemas.enums import AgentName
from consilium.schemas.note import ClinicalNote

AGENT = AgentName.NOTE_EXTRACTOR.value
PV = "v1"
KEY = (AGENT, PV, "default")

NOTE_PAYLOAD: dict[str, Any] = {
    "age_years": 62,
    "iop_mmhg_od": 24.0,
    "iop_mmhg_os": 22.0,
    "current_medications": ["lisinopril"],
    "drug_intolerances_or_allergies": ["sulfa"],
    "comorbidities": ["hypertension"],
    "prior_glaucoma_dx": None,
    "exam_findings": "Optic discs appear suspicious on funduscopy.",
    "free_text": None,
}

MESSAGES = [
    ChatMessage(role="system", content="extract"),
    ChatMessage(role="user", content="<untrusted-clinical-note>note</untrusted-clinical-note>"),
]
TAGS = {"agent": AGENT, "prompt_version": PV}

GEN_KW: dict[str, Any] = {
    "model": "gpt-4o-mini-2024-07-18",
    "temperature": 0.0,
    "max_output_tokens": 2000,
    "timeout_s": 30.0,
}
FAST = LLMRetryPolicy(total_budget_s=5.0, initial_backoff_s=0.0, max_backoff_s=0.0)


def _client(*calls: CannedCall) -> FakeLLMClient:
    return FakeLLMClient({KEY: CannedResponse(tuple(calls))})


async def test_fake_valid_round_trip() -> None:
    client = _client(CannedCall(FakeMode.VALID, NOTE_PAYLOAD))
    resp = await client.generate(MESSAGES, ClinicalNote, tags=TAGS, **GEN_KW)
    assert resp.data.iop_mmhg_od == 24.0
    assert resp.data.current_medications == ("lisinopril",)
    assert resp.prompt_version == PV
    assert resp.input_tokens > 0 and resp.attempts == 1
    # The fake records what the LLM would have seen (leak assertions use this).
    assert client.received[0].key == KEY
    assert len(client.received[0].messages) == 2


async def test_fake_unknown_key_fails_loudly() -> None:
    client = _client(CannedCall(FakeMode.VALID, NOTE_PAYLOAD))
    with pytest.raises(ValueError, match="no canned response"):
        await client.generate(MESSAGES, ClinicalNote, tags={"agent": "nobody"}, **GEN_KW)


async def test_fake_refusal_is_agent_failure_not_retried() -> None:
    client = _client(CannedCall(FakeMode.REFUSAL), CannedCall(FakeMode.VALID, NOTE_PAYLOAD))
    with pytest.raises(AgentFailure) as exc_info:
        await generate_with_retry(client, MESSAGES, ClinicalNote, policy=FAST, tags=TAGS, **GEN_KW)
    assert exc_info.value.reason == "LLM_REFUSAL"
    assert len(client.received) == 1  # refusals are never retried blindly


async def test_retry_recovers_from_transient_then_succeeds() -> None:
    client = _client(
        CannedCall(FakeMode.TIMEOUT),
        CannedCall(FakeMode.TIMEOUT),
        CannedCall(FakeMode.VALID, NOTE_PAYLOAD),
    )
    resp = await generate_with_retry(
        client, MESSAGES, ClinicalNote, policy=FAST, tags=TAGS, **GEN_KW
    )
    assert resp.data.age_years == 62
    assert len(client.received) == 3


async def test_retry_exhaustion_fails_closed() -> None:
    policy = LLMRetryPolicy(
        total_budget_s=60.0, initial_backoff_s=0.0, max_backoff_s=0.0, max_attempts=3
    )
    client = _client(CannedCall(FakeMode.TIMEOUT))
    with pytest.raises(AgentFailure) as exc_info:
        await generate_with_retry(
            client, MESSAGES, ClinicalNote, policy=policy, tags=TAGS, **GEN_KW
        )
    assert exc_info.value.reason == "LLM_UNAVAILABLE"
    assert exc_info.value.case_status == "FAILED"
    assert len(client.received) == 3


async def test_repair_retry_feeds_validation_error_back() -> None:
    client = _client(
        CannedCall(FakeMode.SCHEMA_VIOLATING),
        CannedCall(FakeMode.VALID, NOTE_PAYLOAD),
    )
    resp = await generate_structured(client, MESSAGES, ClinicalNote, tags=TAGS, **GEN_KW)
    assert resp.data.iop_mmhg_os == 22.0
    assert resp.attempts == 2
    # The repair attempt carried the original messages + a validation-error hint.
    repair_messages = client.received[1].messages
    assert len(repair_messages) == 3
    assert "failed validation" in repair_messages[-1].content


async def test_repair_exhaustion_raises_validation_failure() -> None:
    client = _client(CannedCall(FakeMode.MALFORMED))
    with pytest.raises(ValidationFailure) as exc_info:
        await generate_structured(
            client, MESSAGES, ClinicalNote, max_repairs=2, tags=TAGS, **GEN_KW
        )
    assert exc_info.value.reason == "LLM_OUTPUT_INVALID"
    assert exc_info.value.case_status == "FAILED"
    assert len(client.received) == 3  # 1 initial + 2 repairs, no more


async def test_transient_inside_repair_loop_is_retried_first() -> None:
    client = _client(
        CannedCall(FakeMode.TIMEOUT),
        CannedCall(FakeMode.SCHEMA_VIOLATING),
        CannedCall(FakeMode.VALID, NOTE_PAYLOAD),
    )
    resp = await generate_structured(
        client, MESSAGES, ClinicalNote, retry_policy=FAST, tags=TAGS, **GEN_KW
    )
    assert resp.attempts == 2
    assert len(client.received) == 3


def test_openai_client_constructs_without_network() -> None:
    """Compile/smoke check for the real impl — construction only, no call."""
    client = OpenAILLMClient(api_key="sk-test-synthetic")
    assert client is not None


@pytest.mark.live
async def test_openai_live_contract() -> None:  # pragma: no cover — manual/nightly only
    """Live contract stub (AGENTS.md §20): real call, schema conformance.

    Run explicitly with: uv run pytest -m live (needs OPENAI_API_KEY).
    """
    import os

    client = OpenAILLMClient(api_key=os.environ["OPENAI_API_KEY"])
    resp = await generate_structured(
        client,
        MESSAGES,
        ClinicalNote,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.0,
        max_output_tokens=500,
        timeout_s=30.0,
        tags=TAGS,
    )
    assert isinstance(resp.data, ClinicalNote)
