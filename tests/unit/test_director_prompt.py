"""Director prompt v1 tests (Section 6.1; plan: prompt renders against fixture
sub-reports + fact sheet).

The Director prompt must carry the §12 design decisions: attribute-don't-
launder (`source_roles` on every claim), disagreement preservation, the
NUMERIC-claim rule (numeric_value + numeric_ref), mandatory limitations, and
the I-12 wording contract.
"""

import pytest
from jinja2.exceptions import UndefinedError

from consilium.agents.templates import render_facts, render_subreports
from consilium.context.facts import F
from consilium.core.settings import DEFAULT_DIRECTOR_PROMPT_VERSION as DIRECTOR_V
from consilium.llm.prompt_loader import PromptLoader
from consilium.schemas.agents import Finding, Recommendation, SubReport
from consilium.schemas.context import Fact, GlobalContext
from consilium.schemas.enums import Confidence, FactSource, Laterality, Role

REQUIRED_SECTION_MARKERS = (
    "Role & scope",
    "exclusions",
    "Attribution rule",
    "source_roles",
    "NUMERIC-claim rule",
    "numeric_value",
    "numeric_ref",
    "Disagreement-preservation rule",
    "Limitations rule",
    "Untrusted-data rule",
    "Wording rules",
    "Revision mode",
    "Output contract",
    "DirectorDraft",
    "screening",
    "evidence_refs",
    "untrusted-data",
    "left to clinician",
)


def _global_ctx() -> GlobalContext:
    return GlobalContext(
        case_id="00000000-0000-0000-0000-0000000000d1",
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
            uncertainties=(f"{role.value} uncertainty",),
            declined_out_of_scope=(f"{role.value} declined item",),
        )
        for role in (Role.OPHTHALMOLOGIST, Role.OPTOMETRIST, Role.PHARMACIST)
    }


def _render(**overrides: str) -> str:
    context = {
        "facts_block": render_facts(_global_ctx()),
        "subreports_block": render_subreports(_subreports()),
        "revision_block": "",
        **overrides,
    }
    return PromptLoader().load("director", DIRECTOR_V).render(context)


def test_director_prompt_renders_with_fixture_inputs() -> None:
    rendered = _render()
    assert "{{" not in rendered
    # fact sheet content is embedded
    assert f"{F.CDR_V} = 0.62" in rendered
    assert '"Patient mentions occasional headaches."' in rendered
    # all three sub-reports are embedded, in fixed role order
    oph = rendered.index('sub-report role="ophthalmologist"')
    opt = rendered.index('sub-report role="optometrist"')
    pharm = rendered.index('sub-report role="pharmacist"')
    assert oph < opt < pharm
    assert "ophthalmologist finding" in rendered
    assert "pharmacist declined item" in rendered


def test_director_prompt_carries_all_required_rule_sections() -> None:
    rendered = _render()
    for marker in REQUIRED_SECTION_MARKERS:
        assert marker in rendered, f"missing required section marker: {marker}"


def test_director_prompt_forbids_diagnosis_doses_and_laundering() -> None:
    rendered = _render().replace("\n", " ")
    assert "not diagnose" in rendered
    assert "drug doses, regimens, or brand names" in rendered
    assert "Never merge opposing positions" in rendered


def test_director_prompt_missing_variable_fails_loudly() -> None:
    with pytest.raises(UndefinedError):
        PromptLoader().load("director", DIRECTOR_V).render({})


def test_revision_block_is_rendered_when_supplied() -> None:
    rendered = _render(revision_block="Revision round 1: address issues i1, i2.")
    assert "Revision round 1: address issues i1, i2." in rendered
