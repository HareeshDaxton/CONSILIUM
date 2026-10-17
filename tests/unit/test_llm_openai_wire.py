"""OpenAI client wire format + error mapping, fully offline: the SDK's
``chat.completions.create`` is replaced by a stub (the pinned SDK ships its own
HTTP stack, so HTTP-level mocks would not intercept it — no network, §20).
Regression for the Phase 13 live-eval finding: strict structured outputs
reject JSON-Schema tuple arrays (``prefixItems``), which made every live
Director call fail with HTTP 400."""

from __future__ import annotations

import json
from typing import Any

import openai
import pytest
from openai.types.chat import ChatCompletion

from consilium.core.errors import AgentFailure
from consilium.llm.client import ChatMessage, ModelOutputInvalid
from consilium.llm.client_openai import OpenAILLMClient, response_format_for
from consilium.schemas.agents import DirectorDraft, SubReport
from consilium.schemas.audit import CriticVerdict

DRAFT: dict[str, Any] = {
    "impression": "Findings warrant evaluation for glaucoma.",
    "claims": [],
    "disagreements": [
        {
            "topic": "escalation urgency",
            "positions": [["ophthalmologist", "urgent"], ["optometrist", "routine"]],
            "resolution": "left to clinician",
        }
    ],
    "recommendations": [],
    "limitations": ["Screening support only."],
    "revision": 0,
}


def _walk(node: object) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(node, dict):
        out.append(node)
        for v in node.values():
            out += _walk(v)
    elif isinstance(node, list):
        for v in node:
            out += _walk(v)
    return out


def _completion(
    content: str | None, *, finish: str = "stop", refusal: str | None = None
) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class _Stub:
    """Stands in for ``AsyncOpenAI.chat.completions.create``; records kwargs."""

    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> ChatCompletion:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return ChatCompletion.model_validate(self.result)


async def _call(result: dict[str, Any] | Exception, model: type = DirectorDraft) -> Any:
    client = OpenAILLMClient("sk-test-synthetic", base_url="https://api.openai.invalid/v1")
    stub = _Stub(result)
    client._client.chat.completions.create = stub  # type: ignore[method-assign]
    resp = await client.generate(
        [ChatMessage(role="user", content="x")],
        model,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.2,
        max_output_tokens=100,
        timeout_s=5.0,
        tags={"agent": "director", "prompt_version": "v1"},
    )
    return resp, stub.kwargs


def _status_error(code: int) -> openai.APIStatusError:
    err = openai.BadRequestError.__new__(openai.BadRequestError)
    err.status_code = code
    return err


def test_tuple_arrays_are_adapted_and_nothing_else_changes() -> None:
    fmt = response_format_for(DirectorDraft)
    nodes = _walk(fmt)
    assert not any("prefixItems" in n for n in nodes)
    assert fmt["json_schema"]["strict"] is True
    # Schemas without tuples are passed through exactly as the SDK builds them.
    from openai.lib._parsing._completions import type_to_response_format_param

    for model in (SubReport, CriticVerdict):
        assert response_format_for(model) == dict(type_to_response_format_param(model))  # type: ignore[arg-type]


async def test_valid_reply_is_validated_by_the_strict_model() -> None:
    resp, body = await _call(_completion(json.dumps(DRAFT)))
    assert isinstance(resp.data, DirectorDraft)
    assert resp.data.disagreements[0].positions[0][1] == "urgent"
    assert (resp.input_tokens, resp.output_tokens) == (11, 7)
    assert body["response_format"]["type"] == "json_schema" and body["temperature"] == 0.2
    assert not any("prefixItems" in n for n in _walk(body["response_format"]))


async def test_wrong_tuple_arity_goes_to_repair_not_through() -> None:
    bad = {
        **DRAFT,
        "disagreements": [
            {**DRAFT["disagreements"][0], "positions": [["ophthalmologist"], ["optometrist", "x"]]}
        ],
    }
    with pytest.raises(ModelOutputInvalid):
        await _call(_completion(json.dumps(bad)))


@pytest.mark.parametrize(
    ("content", "finish", "refusal", "expected"),
    [
        ("{", "stop", None, ModelOutputInvalid),
        (None, "stop", None, ModelOutputInvalid),
        ('{"impression": "x"', "length", None, ModelOutputInvalid),
        (None, "stop", "I can't help with that.", AgentFailure),
        (None, "content_filter", None, AgentFailure),
    ],
)
async def test_failure_modes_map_to_the_contract(
    content: str | None, finish: str, refusal: str | None, expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        await _call(_completion(content, finish=finish, refusal=refusal))


async def test_bad_request_is_rejected_not_retried() -> None:
    with pytest.raises(AgentFailure) as err:
        await _call(_status_error(400))
    assert err.value.reason == "LLM_REQUEST_REJECTED"
