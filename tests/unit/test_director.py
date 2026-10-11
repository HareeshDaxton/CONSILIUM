"""Director draft + revise tests (plan Section 6.2).

Unit tests with FakeLLMClient (no network, §20): refs resolve against the
GlobalContext fact sheet, limitations are populated, revise bumps revision
and receives the previous draft + issue list ONLY (never critic reasoning).
Disagreement-preservation cases live in Section 6.3 below.
"""

from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from consilium.agents.director import Director
from consilium.context.facts import F
from consilium.core.errors import AgentFailure, ValidationFailure
from consilium.core.settings import Settings
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.llm.prompt_loader import PromptLoader
from consilium.schemas.agents import DirectorDraft, Finding, Recommendation, SubReport
from consilium.schemas.audit import Issue
from consilium.schemas.context import Fact, GlobalContext
from consilium.schemas.enums import (
    Confidence,
    FactSource,
    IssueType,
    Laterality,
    Role,
)

ROLES = (Role.OPHTHALMOLOGIST, Role.OPTOMETRIST, Role.PHARMACIST)


def _global_ctx() -> GlobalContext:
    return GlobalContext(
        case_id="00000000-0000-0000-0000-0000000000d2",
        laterality=Laterality.OD,
        facts=(
            Fact(id=F.P_TIER, label="Screening tier", value="intermediate", source=FactSource.CAD),
            Fact(id=F.CDR_V, label="Vertical CDR", value=0.62, source=FactSource.CAD),
        ),
    )


def _subreports() -> dict[Role, SubReport]:
    return {
        role: SubReport(
            role=role,
            findings=(
                Finding(
                    text=f"{role.value} finding",
                    evidence_refs=(F.P_TIER,),
                    confidence=Confidence.MODERATE,
                ),
            ),
            recommendations=(
                Recommendation(text=f"{role.value} recommendation", triggered_by=(F.P_TIER,)),
            ),
        )
        for role in ROLES
    }


def _draft_payload(*, revision: int = 0) -> dict[str, Any]:
    return {
        "impression": "Findings raise suspicion of glaucoma and warrant evaluation.",
        "claims": [
            {
                "id": "c1",
                "kind": "numeric",
                "text": "Vertical CDR is within the measured range.",
                "evidence_refs": [F.CDR_V],
                "source_roles": ["ophthalmologist"],
                "numeric_value": 0.62,
                "numeric_ref": F.CDR_V,
            },
            {
                "id": "c2",
                "kind": "categorical",
                "text": "Screening tier is intermediate.",
                "evidence_refs": [F.P_TIER],
                "source_roles": ["ophthalmologist", "optometrist"],
                "numeric_value": None,
                "numeric_ref": None,
            },
        ],
        "disagreements": [],
        "recommendations": [{"text": "Consider visual field testing.", "triggered_by": [F.P_TIER]}],
        "limitations": [
            "Screening support only; does not diagnose.",
            "No visual field or OCT data available.",
        ],
        "revision": revision,
    }


def _director(settings: Settings, llm: FakeLLMClient) -> Director:
    return Director(
        prompt=PromptLoader().load("director", "v1"),
        llm=llm,
        model=settings.llm_model_director,
        temperature=settings.llm_temperature_director,
    )


def _fake(payload: dict[str, Any]) -> FakeLLMClient:
    return FakeLLMClient.valid(payload, agent="director", prompt_version="v1")


async def test_draft_refs_resolve_and_limitations_present(settings: Settings) -> None:
    ctx = _global_ctx()
    llm = _fake(_draft_payload())
    resp = await _director(settings, llm).draft(_subreports(), ctx)
    draft = resp.data
    assert resp.prompt_version == "v1"
    fact_ids = {f.id for f in ctx.facts}
    for claim in draft.claims:
        assert set(claim.evidence_refs) <= fact_ids
        if claim.kind.value == "numeric":
            assert claim.numeric_ref in fact_ids
    assert len(draft.limitations) >= 2
    assert draft.revision == 0


async def test_draft_stamps_revision_zero_even_if_model_says_otherwise(
    settings: Settings,
) -> None:
    resp = await _director(settings, _fake(_draft_payload(revision=7))).draft(
        _subreports(), _global_ctx()
    )
    assert resp.data.revision == 0


async def test_revise_bumps_revision_and_carries_issue_list_only(
    settings: Settings,
) -> None:
    ctx = _global_ctx()
    first = _fake(_draft_payload())
    draft = (await _director(settings, first).draft(_subreports(), ctx)).data

    issues = (
        Issue(
            type=IssueType.NUMERIC_MISMATCH,
            claim_id="c1",
            detail="numeric_value 0.62 does not match fact cad.cdr.vertical within tolerance",
            raised_by="deterministic",
        ),
        Issue(
            type=IssueType.OMITTED_FINDING,
            claim_id=None,
            detail="laterality is not mentioned in any claim or limitation",
            raised_by="llm_critic",
        ),
    )
    revised_payload = _draft_payload(revision=1)
    revised_payload["limitations"].append("Laterality: OD (right eye).")
    llm = _fake(revised_payload)
    resp = await _director(settings, llm).revise(draft, issues, 1, _subreports(), ctx)
    assert resp.data.revision == 1
    system = llm.received[0].messages[0].content
    # the revision block carries the previous draft and the issue list...
    assert "revising draft revision 0" in system
    assert draft.impression in system
    assert "[i1]" in system and "[i2]" in system
    assert issues[0].detail in system and issues[1].detail in system
    # ...and nothing beyond the issue list (no critic reasoning field exists
    # on Issue; the block is built from type/claim_id/detail only — §12)
    assert "raised_by" not in system


async def test_refusal_fails_closed_never_retried(settings: Settings) -> None:
    llm = FakeLLMClient(
        {("director", "v1", "default"): CannedResponse((CannedCall(FakeMode.REFUSAL),))}
    )
    with pytest.raises(AgentFailure):
        await _director(settings, llm).draft(_subreports(), _global_ctx())
    assert len(llm.received) == 1


# ---------------------------------------------------------------------------
# Section 6.3 — Disagreement preservation on curated cases
#
# Curated fixture cases where roles genuinely hold polar positions. Under
# fakes the draft payload is canned, so what is proven deterministically:
#   1. the conflict is VISIBLE to the Director (both positions render in the
#      prompt's sub-reports block — the prerequisite for preservation);
#   2. a draft carrying the Disagreement preserves every role's position
#      verbatim, with a resolution, never merged into one confident claim;
#   3. schema rejects a "disagreement" smoothed down to a single position
#      (positions min_length=2) — the malformed-output path fails closed;
# Detecting a draft that OMITS a real disagreement is the deterministic
# verifier's DISAGREEMENT_SUPPRESSED check (P7), not the Director's.
# ---------------------------------------------------------------------------


def _polar_subreports(topic: str, position_a: str, position_b: str) -> dict[Role, SubReport]:
    reports = _subreports()
    reports[Role.OPHTHALMOLOGIST] = reports[Role.OPHTHALMOLOGIST].model_copy(
        update={"recommendations": (Recommendation(text=position_a, triggered_by=(F.P_TIER,)),)}
    )
    reports[Role.OPTOMETRIST] = reports[Role.OPTOMETRIST].model_copy(
        update={"recommendations": (Recommendation(text=position_b, triggered_by=(F.P_TIER,)),)}
    )
    return reports


# Curated case 1: escalation urgency (intermediate tier, CDR 0.62)
CASE_ESCALATION = (
    "escalation urgency",
    "Findings warrant urgent ophthalmology evaluation.",
    "Routine follow-up within six months is sufficient at this tier.",
)

# Curated case 2: interpretation of the same structural finding
CASE_INTERPRETATION = (
    "interpretation of vertical CDR",
    "The vertical CDR is consistent with glaucomatous change.",
    "The vertical CDR alone is within screening norms; low urgency.",
)


def _disagreeing_payload(topic: str, pos_a: str, pos_b: str) -> dict[str, Any]:
    payload = _draft_payload()
    payload["disagreements"] = [
        {
            "topic": topic,
            "positions": [["ophthalmologist", pos_a], ["optometrist", pos_b]],
            "resolution": "left to clinician",
        }
    ]
    return payload


@pytest.mark.parametrize("case", [CASE_ESCALATION, CASE_INTERPRETATION])
async def test_disagreement_preserved_with_per_role_positions(
    settings: Settings, case: tuple[str, str, str]
) -> None:
    topic, pos_a, pos_b = case
    reports = _polar_subreports(topic, pos_a, pos_b)
    llm = _fake(_disagreeing_payload(topic, pos_a, pos_b))
    resp = await _director(settings, llm).draft(reports, _global_ctx())

    # 1. both polar positions were visible in the Director's prompt
    system = llm.received[0].messages[0].content
    assert pos_a in system and pos_b in system

    # 2. the draft preserves the disagreement verbatim, one position per role
    assert len(resp.data.disagreements) == 1
    disagreement = resp.data.disagreements[0]
    assert disagreement.topic == topic
    assert dict(disagreement.positions) == {
        Role.OPHTHALMOLOGIST: pos_a,
        Role.OPTOMETRIST: pos_b,
    }
    assert disagreement.resolution == "left to clinician"
    # the disagreeing roles are NOT merged into a shared source_roles claim
    for claim in resp.data.claims:
        if claim.kind.value == "recommendation":
            assert not ({Role.OPHTHALMOLOGIST, Role.OPTOMETRIST} <= set(claim.source_roles)), (
                "polar positions laundered into a single consensus claim"
            )


async def test_single_position_disagreement_is_schema_rejected(
    settings: Settings,
) -> None:
    """A 'disagreement' smoothed to one position violates positions>=2:
    the schema rejects it outright, and a schema-violating model output
    exhausts the repair loop → ValidationFailure (fail closed, I-5)."""
    payload = _draft_payload()
    payload["disagreements"] = [
        {
            "topic": CASE_ESCALATION[0],
            "positions": [["ophthalmologist", CASE_ESCALATION[1]]],
            "resolution": "left to clinician",
        }
    ]
    with pytest.raises(PydanticValidationError):
        DirectorDraft.model_validate(payload)

    llm = FakeLLMClient(
        {("director", "v1", "default"): CannedResponse((CannedCall(FakeMode.SCHEMA_VIOLATING),))}
    )
    with pytest.raises(ValidationFailure):
        await _director(settings, llm).draft(_polar_subreports(*CASE_ESCALATION), _global_ctx())
    assert len(llm.received) == 3  # 1 initial + 2 repairs, then fail closed
