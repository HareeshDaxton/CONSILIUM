"""Specialist prompt v1 tests (Section 5.3; SKILLS.md specialist-agent).

Each prompt must render against a fixture RoleContext and carry the required
rule sections: role & scope, exclusions, output contract, evidence citation,
uncertainty, out-of-scope, untrusted-data, wording (I-12; pharmacist: no
doses/regimens/brands). Prompt-level leak control (no role seeing forbidden
fact content) is proven separately by test_router_leak.py — re-run at section
end.
"""

import pytest

from consilium.agents.templates import render_facts
from consilium.context.facts import F
from consilium.llm.prompt_loader import PromptLoader
from consilium.schemas.context import Fact, RoleContext
from consilium.schemas.enums import FactSource, Laterality, Role

ROLES = {
    Role.OPHTHALMOLOGIST: "ophthalmologist",
    Role.OPTOMETRIST: "optometrist",
    Role.PHARMACIST: "pharmacist",
}

REQUIRED_SECTION_MARKERS = (
    "Role & scope",
    "exclusions",
    "Evidence-citation rule",
    "Uncertainty rule",
    "Out-of-scope rule",
    "Untrusted-data rule",
    "Wording rules",
    "Output contract",
    "SubReport",
    "screening",
    "evidence_refs",
    "uncertainties",
    "declined_out_of_scope",
    "untrusted-data",
)


def _ctx(role: Role) -> RoleContext:
    return RoleContext(
        role=role,
        case_id="00000000-0000-0000-0000-0000000000c1",
        laterality=Laterality.OD,
        facts=(
            Fact(
                id=F.P_TIER,
                label="Screening tier",
                value="intermediate",
                source=FactSource.CAD,
            ),
            Fact(
                id=F.CDR_V,
                label="Vertical CDR",
                value=0.62,
                source=FactSource.CAD,
            ),
            Fact(
                id=F.FREE,
                label="Note free text",
                value="Patient mentions occasional headaches.",
                source=FactSource.NOTE,
            ),
        ),
        routing_version="testhash",
    )


@pytest.mark.parametrize("role", list(ROLES))
def test_prompt_renders_with_facts_block(role: Role) -> None:
    component = ROLES[role]
    prompt = PromptLoader().load(component, "v1")
    rendered = prompt.render({"facts_block": render_facts(_ctx(role))})
    assert "0.62" in rendered  # the fixture facts block is embedded
    assert rendered.count("{{") == 0  # nothing left unrendered


@pytest.mark.parametrize("role", list(ROLES))
def test_required_rule_sections_present(role: Role) -> None:
    content = PromptLoader().load(ROLES[role], "v1").content
    for marker in REQUIRED_SECTION_MARKERS:
        assert marker.lower() in content.lower(), f"{ROLES[role]} missing {marker!r}"


def test_pharmacist_prompt_forbids_doses_and_brands() -> None:
    content = PromptLoader().load("pharmacist", "v1").content.lower()
    assert "doses" in content and "regimens" in content and "brand" in content
    assert "prescriber judgement" in content


def test_prompts_declare_no_diagnosis_and_no_invented_numbers() -> None:
    for component in ROLES.values():
        content = PromptLoader().load(component, "v1").content.lower()
        assert "definitive diagnosis" in content
        assert "never" in content  # exclusion rules are stated as prohibitions


def test_missing_facts_block_variable_fails_loudly() -> None:
    from jinja2.exceptions import UndefinedError

    with pytest.raises(UndefinedError):
        PromptLoader().load("ophthalmologist", "v1").render({})
