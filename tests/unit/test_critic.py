"""Critic prompt + agent tests (plan Section 8.1; AGENTS.md §13.2).

Unit tests with FakeLLMClient (no network, §20): the prompt renders against a
fixture GlobalContext + DirectorDraft; the verdict's ``round`` is stamped by
code (never the model); issues pass through with ``raised_by`` preserved; an
issue citing an unknown claim_id is repaired via the output_validator loop;
a persistent bad claim_id fails closed; refusals are never retried. The
Settings critic≠director guard (AGENTS.md §6) is covered here too.
"""

import logging
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from consilium.agents.critic_agent import Critic
from consilium.agents.templates import render_claims, render_facts
from consilium.context.facts import F
from consilium.core.errors import AgentFailure, ValidationFailure
from consilium.core.settings import DEFAULT_CRITIC_PROMPT_VERSION as CRITIC_V
from consilium.core.settings import Settings
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.llm.prompt_loader import PromptLoader
from consilium.schemas.agents import Claim, DirectorDraft, Disagreement, Recommendation
from consilium.schemas.context import Fact, GlobalContext
from consilium.schemas.enums import ClaimKind, FactSource, Laterality, Role
from consilium.verify.critic import run_critic

REQUIRED_SECTION_MARKERS = (
    "Role & scope",
    "exclusions",
    "never see specialist prose",
    "Audit task",
    "entailment",
    "Evidence vocabulary",
    "deterministic verifier",
    "Output contract",
    "CriticVerdict",
    "unsupported",
    "omitted_finding",
    "disagreement_suppressed",
    "untrusted-data",
    "screening support",
    "llm_critic",
)


def _global_ctx() -> GlobalContext:
    return GlobalContext(
        case_id="00000000-0000-0000-0000-0000000000d3",
        laterality=Laterality.OD,
        facts=(
            Fact(id=F.P_TIER, label="Screening tier", value="intermediate", source=FactSource.CAD),
            Fact(id=F.CDR_V, label="Vertical CDR", value=0.62, source=FactSource.CAD),
            Fact(
                id=F.FREE,
                label="Note free text",
                value="Patient mentions occasional headaches.",
                source=FactSource.NOTE,
            ),
        ),
    )


def _draft() -> DirectorDraft:
    return DirectorDraft(
        impression="Findings raise suspicion of glaucoma and warrant evaluation.",
        claims=(
            Claim(
                id="c1",
                kind=ClaimKind.NUMERIC,
                text="Vertical CDR is within the measured range.",
                evidence_refs=(F.CDR_V,),
                source_roles=(Role.OPHTHALMOLOGIST,),
                numeric_value=0.62,
                numeric_ref=F.CDR_V,
            ),
            Claim(
                id="c2",
                kind=ClaimKind.CATEGORICAL,
                text="Screening tier is intermediate.",
                evidence_refs=(F.P_TIER,),
                source_roles=(Role.OPHTHALMOLOGIST, Role.OPTOMETRIST),
            ),
        ),
        disagreements=(
            Disagreement(
                topic="escalation urgency",
                positions=(
                    (Role.OPHTHALMOLOGIST, "Urgent evaluation warranted."),
                    (Role.OPTOMETRIST, "Routine follow-up is sufficient."),
                ),
                resolution="left to clinician",
            ),
        ),
        recommendations=(
            Recommendation(text="Consider visual field testing.", triggered_by=(F.P_TIER,)),
        ),
        limitations=("Screening support only; does not diagnose.",),
        revision=0,
    )


def _approve_payload(*, round_: int = 0) -> dict[str, Any]:
    return {"approved": True, "issues": [], "round": round_}


def _reject_payload(*, claim_id: str | None = "c1", round_: int = 0) -> dict[str, Any]:
    return {
        "approved": False,
        "issues": [
            {
                "type": "unsupported",
                "claim_id": claim_id,
                "detail": "The claim's interpretation goes beyond the cited evidence.",
                "evidence": [F.CDR_V],
                "raised_by": "llm_critic",
            },
            {
                "type": "omitted_finding",
                "claim_id": None,
                "detail": "A material caveat fact is absent from claims and limitations.",
                "evidence": [F.P_TIER],
                "raised_by": "llm_critic",
            },
        ],
        "round": round_,
    }


def _critic(settings: Settings, llm: FakeLLMClient) -> Critic:
    return Critic(
        prompt=PromptLoader().load("critic", CRITIC_V),
        llm=llm,
        model=settings.llm_model_critic,
        temperature=settings.llm_temperature_critic,
    )


def _fake(payload: dict[str, Any]) -> FakeLLMClient:
    return FakeLLMClient.valid(payload, agent="critic", prompt_version=CRITIC_V)


# --- (a) prompt rendering ---------------------------------------------------


def test_critic_prompt_renders_with_fixture_inputs() -> None:
    rendered = (
        PromptLoader()
        .load("critic", CRITIC_V)
        .render(
            {
                "facts_block": render_facts(_global_ctx()),
                "claims_block": render_claims(_draft()),
            }
        )
    )
    assert "{{" not in rendered
    for marker in REQUIRED_SECTION_MARKERS:
        assert marker in rendered, f"missing required section marker: {marker}"
    # fact sheet content is embedded
    assert f"{F.CDR_V} = 0.62" in rendered
    assert '"Patient mentions occasional headaches."' in rendered
    # draft content is embedded: claims, disagreement, impression, limitations
    assert "c1 (kind: numeric)" in rendered
    assert "numeric_value: 0.62; numeric_ref: cad.cdr.vertical" in rendered
    assert "escalation urgency" in rendered
    assert "position[optometrist]: Routine follow-up is sufficient." in rendered
    assert "Findings raise suspicion of glaucoma" in rendered
    assert "Screening support only; does not diagnose." in rendered
    # recommendations are shown, so the critic cannot call recommended workup "omitted"
    assert "[draft-recommendations]" in rendered
    assert "- Consider visual field testing." in rendered
    assert f"  triggered_by: {F.P_TIER}" in rendered


# --- (b) approve path; round stamped by code --------------------------------


async def test_approve_path_stamps_round_from_code_not_model(
    settings: Settings,
) -> None:
    llm = _fake(_approve_payload(round_=7))
    resp = await _critic(settings, llm).run(_draft(), _global_ctx(), 1)
    verdict = resp.data
    assert resp.prompt_version == CRITIC_V
    assert verdict.approved is True
    assert verdict.issues == ()
    assert verdict.round == 1  # model said 7; code wins


async def test_run_critic_facade_returns_verdict_and_logs_counts(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    llm = _fake(_reject_payload())
    with caplog.at_level(logging.INFO):
        verdict = await run_critic(_draft(), _global_ctx(), 2, _critic(settings, llm))
    assert verdict.approved is False
    assert verdict.round == 2
    record = next(r for r in caplog.records if r.message == "critic.verdict")
    assert record.issue_count == 2  # type: ignore[attr-defined]
    # counts only — no issue detail text in the log record (I-6)
    assert not hasattr(record, "detail")


# --- (c) reject path: issues pass through -----------------------------------


async def test_reject_path_issues_pass_through_with_raised_by(
    settings: Settings,
) -> None:
    llm = _fake(_reject_payload())
    verdict = (await _critic(settings, llm).run(_draft(), _global_ctx(), 0)).data
    assert verdict.approved is False
    assert len(verdict.issues) == 2
    unsupported, omitted = verdict.issues
    assert unsupported.type.value == "unsupported"
    assert unsupported.claim_id == "c1"
    assert unsupported.evidence == (F.CDR_V,)
    assert omitted.type.value == "omitted_finding"
    assert omitted.claim_id is None
    for issue in verdict.issues:
        assert issue.raised_by == "llm_critic"


# --- (d) unknown claim_id repaired on round 2 -------------------------------


async def test_unknown_claim_id_is_repaired_and_hint_names_the_id(
    settings: Settings,
) -> None:
    key = ("critic", CRITIC_V, "default")
    llm = FakeLLMClient(
        {
            key: CannedResponse(
                (
                    CannedCall(FakeMode.VALID, _reject_payload(claim_id="c99")),
                    CannedCall(FakeMode.VALID, _approve_payload()),
                )
            )
        }
    )
    resp = await _critic(settings, llm).run(_draft(), _global_ctx(), 0)
    assert resp.data.approved is True
    assert resp.attempts == 2
    assert len(llm.received) == 2
    repair_hint = llm.received[1].messages[-1].content
    assert "'c99'" in repair_hint  # hint names the ID, never any value (I-6)


# --- (e) persistent bad claim_id fails closed -------------------------------


async def test_persistent_bad_claim_id_fails_closed_after_three_calls(
    settings: Settings,
) -> None:
    # single canned call repeats when the sequence runs out (fake semantics)
    llm = _fake(_reject_payload(claim_id="c99"))
    with pytest.raises(ValidationFailure):
        await _critic(settings, llm).run(_draft(), _global_ctx(), 0)
    assert len(llm.received) == 3  # 1 initial + 2 repairs, then fail closed


async def test_unknown_evidence_fact_id_fails_closed(settings: Settings) -> None:
    payload = _reject_payload()
    payload["issues"][0]["evidence"] = ["cad.cdr.vertical.typo"]
    llm = _fake(payload)
    with pytest.raises(ValidationFailure):
        await _critic(settings, llm).run(_draft(), _global_ctx(), 0)
    assert len(llm.received) == 3


# --- (f) refusal never retried ----------------------------------------------


async def test_refusal_fails_closed_never_retried(settings: Settings) -> None:
    llm = FakeLLMClient(
        {("critic", CRITIC_V, "default"): CannedResponse((CannedCall(FakeMode.REFUSAL),))}
    )
    with pytest.raises(AgentFailure):
        await _critic(settings, llm).run(_draft(), _global_ctx(), 0)
    assert len(llm.received) == 1


# --- (g) Settings critic≠director guard (AGENTS.md §6) -----------------------


def test_settings_rejects_identical_critic_and_director_configuration() -> None:
    with pytest.raises(PydanticValidationError, match="critic must differ"):
        Settings(
            _env_file=None,
            OPENAI_API_KEY="sk-test-synthetic",
            DATABASE_URL="postgresql+asyncpg://consilium:consilium@localhost:5432/consilium_test",
            JWT_SECRET="test-jwt-secret",
            LLM_MODEL_CRITIC="gpt-4o-mini-2024-07-18",
            LLM_MODEL_DIRECTOR="gpt-4o-mini-2024-07-18",
            LLM_TEMPERATURE_CRITIC=0.2,
            LLM_TEMPERATURE_DIRECTOR=0.2,
        )


def test_default_settings_critic_differs_from_director(settings: Settings) -> None:
    assert settings.llm_model_critic != settings.llm_model_director
    assert settings.llm_temperature_critic == 1.0  # gpt-6-luna accepts only T=1
    assert settings.llm_temperature_director == 0.2
