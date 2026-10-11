"""Per-case token budget tests (Section 5.1; AGENTS.md §11).

Budget exceeding RAISES (AgentFailure LLM_TOKEN_BUDGET_EXCEEDED) — output is
never truncated. Offline only, via FakeLLMClient.
"""

from typing import Any

import pytest
from tests.unit.test_llm_client import GEN_KW, KEY, MESSAGES, NOTE_PAYLOAD, TAGS

from consilium.core.errors import AgentFailure
from consilium.core.settings import Settings
from consilium.llm.budget import TokenBudget
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.llm.retry import LLMRetryPolicy, generate_structured, generate_with_retry
from consilium.schemas.note import ClinicalNote

FAST = LLMRetryPolicy(total_budget_s=5.0, initial_backoff_s=0.0, max_backoff_s=0.0)


def _client(*calls: CannedCall) -> FakeLLMClient:
    return FakeLLMClient({KEY: CannedResponse(tuple(calls))})


def test_from_settings_uses_case_budget(settings: Settings) -> None:
    budget = TokenBudget.from_settings(settings)
    assert budget.limit_tokens == settings.llm_case_token_budget
    assert budget.remaining == settings.llm_case_token_budget


def test_check_raises_when_exhausted() -> None:
    budget = TokenBudget(limit_tokens=100, used_tokens=100)
    with pytest.raises(AgentFailure) as exc:
        budget.check()
    assert exc.value.reason == "LLM_TOKEN_BUDGET_EXCEEDED"


def test_charge_raises_on_overflow() -> None:
    budget = TokenBudget(limit_tokens=10)
    with pytest.raises(AgentFailure) as exc:
        budget.charge(6, 6)
    assert exc.value.reason == "LLM_TOKEN_BUDGET_EXCEEDED"
    assert budget.used_tokens == 12  # charge is recorded before raising


async def test_budget_charged_through_retry_wrapper() -> None:
    client = _client(CannedCall(FakeMode.VALID, NOTE_PAYLOAD))
    budget = TokenBudget(limit_tokens=100_000)
    resp = await generate_with_retry(
        client, MESSAGES, ClinicalNote, policy=FAST, tags=TAGS, budget=budget, **GEN_KW
    )
    assert budget.used_tokens == resp.input_tokens + resp.output_tokens
    assert budget.used_tokens > 0


async def test_budget_overflow_fails_closed_in_structured_call() -> None:
    client = _client(CannedCall(FakeMode.VALID, NOTE_PAYLOAD))
    budget = TokenBudget(limit_tokens=1)
    with pytest.raises(AgentFailure) as exc:
        await generate_structured(
            client, MESSAGES, ClinicalNote, tags=TAGS, budget=budget, **GEN_KW
        )
    assert exc.value.reason == "LLM_TOKEN_BUDGET_EXCEEDED"


async def test_exhausted_budget_blocks_next_call() -> None:
    client = _client(CannedCall(FakeMode.VALID, NOTE_PAYLOAD))
    budget = TokenBudget(limit_tokens=1, used_tokens=1)
    with pytest.raises(AgentFailure):
        await generate_with_retry(
            client, MESSAGES, ClinicalNote, policy=FAST, tags=TAGS, budget=budget, **GEN_KW
        )
    assert client.received == []  # no LLM call was ever placed


async def test_budget_shared_across_repairs() -> None:
    payload_bad: dict[str, Any] = {**NOTE_PAYLOAD, "age_years": 999}
    client = _client(
        CannedCall(FakeMode.SCHEMA_VIOLATING, payload_bad),
        CannedCall(FakeMode.VALID, NOTE_PAYLOAD),
    )
    budget = TokenBudget(limit_tokens=100_000)
    await generate_structured(client, MESSAGES, ClinicalNote, tags=TAGS, budget=budget, **GEN_KW)
    assert len(client.received) == 2  # repair consumed budget too
    assert budget.used_tokens > 0
