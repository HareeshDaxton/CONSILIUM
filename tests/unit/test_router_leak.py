"""Leak property tests (§4.4; I-2; AGENTS.md §9.2 enforcement #2).

For randomly generated ``GlobalContext``s — built through the REAL builder so
labels/IDs are the production ones, with free text that deliberately quotes
CDR-like numbers — route with the REAL ``configs/routing.yaml``, render each
role's prompt fact block, and assert that no FORBIDDEN fact ID, label, or
value string appears in it. A seeded deliberate leak proves the check has
teeth (a test that cannot fail proves nothing).
"""

import fnmatch
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings

from consilium.agents.templates import render_facts
from consilium.context.builder import build
from consilium.context.facts import F
from consilium.context.router import load_routing_config, route
from consilium.schemas.context import Fact, GlobalContext, RoleContext
from consilium.schemas.enums import (
    Confidence,
    Laterality,
    QualityFlag,
    Role,
    ScreeningTier,
)
from consilium.schemas.note import ClinicalNote, NoteExtraction
from consilium.schemas.vision import (
    CADResult,
    CDRMetrics,
    ClassifierOutput,
    ImageQuality,
    MaskQC,
    Measurements,
)

ROUTING = load_routing_config(Path("configs/routing.yaml"))

CASE = "case-leak"

# Distinctive float in (0, 1): 12 decimal places, so str() is long and unique
# enough that an accidental substring collision is practically impossible.
_distinctive_float = st.integers(min_value=10_000_000, max_value=999_999_999_999).map(
    lambda n: n / 1_000_000_000_000
)
# Distinctive string token (mirrors meds/comorbidities/free-text fragments).
_token = st.uuids().map(lambda u: f"tok{u.hex[:12]}")


def _value_strings(fact: Fact) -> list[str]:
    """Rendered substrings that would prove THIS fact's value leaked.

    Only distinctive values are checked: booleans/short strings/ints are too
    collision-prone to be evidence of a leak. Floats render via str() in
    templates._format_value; tuples render as ", ".join of their items.
    """
    v = fact.value
    if isinstance(v, bool):
        return []
    if isinstance(v, str):
        return [v] if len(v) >= 8 else []
    if isinstance(v, float):
        s = str(v)
        return [s] if len(s) >= 8 else []
    if isinstance(v, tuple):
        return [s for s in v if len(s) >= 8]
    return []


def leaks(gctx: GlobalContext, rc: RoleContext) -> list[str]:
    """Forbidden fact IDs / labels / value strings found in the rendered prompt.

    "Forbidden" is decided by the ROUTING POLICY (the allowlist patterns),
    not by what the router happened to deliver — so a deliberately injected
    fact is caught too.
    """
    allowed_ids = {
        f.id
        for f in gctx.facts
        if any(fnmatch.fnmatchcase(f.id, pat) for pat in ROUTING.roles[rc.role].allow)
    }
    rendered = render_facts(rc)
    found: list[str] = []
    for fact in gctx.facts:
        if fact.id in allowed_ids:
            continue
        if fact.id in rendered:
            found.append(f"id:{fact.id}")
        if len(fact.label) >= 8 and fact.label in rendered:
            found.append(f"label:{fact.id}")
        for s in _value_strings(fact):
            if s in rendered:
                found.append(f"value:{fact.id}")
    return found


@st.composite
def global_contexts(draw: st.DrawFn) -> GlobalContext:
    flags = draw(st.frozensets(st.sampled_from(QualityFlag), max_size=3))
    mask_ok = draw(st.booleans())
    p_cal = draw(_distinctive_float)
    cdr_v = draw(_distinctive_float)
    cdr_a = draw(_distinctive_float)
    q_score = draw(_distinctive_float)
    # str() of one float must never be a substring of another's (trailing-zero
    # trimming makes e.g. "0.12345678901" ⊂ "0.123456789012") — such pairs are
    # not leaks, just undetectable coincidences.
    float_strs = [str(p_cal), str(cdr_v), str(cdr_a), str(q_score)]
    assume(
        all(b not in a for i, a in enumerate(float_strs) for b in float_strs[i + 1 :])
        and all(a not in b for i, a in enumerate(float_strs) for b in float_strs[i + 1 :])
    )
    laterality = draw(st.sampled_from(Laterality))
    cad = CADResult(
        quality=ImageQuality(score=q_score, gradable=True, flags=tuple(sorted(flags))),
        classifier=ClassifierOutput(
            p_raw=0.7,
            p_calibrated=p_cal,
            tier=draw(st.sampled_from(ScreeningTier)),
            model_version="m@rev+mlflow:r1",
        ),
        measurements=Measurements(
            disc_area_px=draw(st.integers(min_value=5000, max_value=40000)),
            cup_area_px=draw(st.integers(min_value=1, max_value=4999)),
            disc_vertical_diameter_px=draw(st.integers(min_value=80, max_value=240)),
            cup_vertical_diameter_px=draw(st.integers(min_value=1, max_value=79)),
        ),
        cdr=CDRMetrics(vertical=cdr_v, area_based=cdr_a),
        mask_qc=MaskQC(
            cup_within_disc=True,
            single_component_each=True,
            disc_area_plausible=True,
            ok=mask_ok,
        ),
        segmenter_version="s@rev+mlflow:r2",
        laterality=laterality,
    )

    note: ClinicalNote | None = None
    extraction: NoteExtraction | None = None
    iop = st.integers(min_value=5_000, max_value=60_000).map(lambda n: n / 1000)
    if draw(st.booleans()):
        # Free text deliberately quotes CAD numbers (the classic derived-text
        # leak: §9.2 pitfall) plus random distinctive tokens.
        free_text = (
            f"{draw(_token)} referring clinician thinks CDR {cdr_v} and "
            f"probability {p_cal} {draw(_token)}"
        )
        note = ClinicalNote(
            age_years=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=120))),
            iop_mmhg_od=draw(st.one_of(st.none(), iop)),
            iop_mmhg_os=draw(st.one_of(st.none(), iop)),
            current_medications=draw(st.tuples(_token, _token)),
            drug_intolerances_or_allergies=draw(st.tuples(_token)),
            comorbidities=draw(st.tuples(_token)),
            prior_glaucoma_dx=draw(st.one_of(st.none(), _token)),
            # Untrusted text fields are deliberately token-based (no '.' or
            # fact-ID-shaped fragments): the routing guarantee is about which
            # FACTS are delivered, not substring censorship of untrusted text
            # that a role IS allowed to read.
            exam_findings=draw(
                st.one_of(st.none(), _token, st.tuples(_token, _token).map(" ".join))
            ),
            free_text=free_text,
        )
        extraction = NoteExtraction(
            note=note,
            extractor_model="gpt-4o-mini-2024-07-18",
            extractor_prompt_version="v1",
            extraction_confidence=draw(st.sampled_from(Confidence)),
        )
    return build(cad, note, extraction, CASE)


@given(ctx=global_contexts())
@settings(max_examples=1000, deadline=None, derandomize=True)
@pytest.mark.parametrize("role", list(Role))
def test_no_forbidden_fact_leaks_into_role_prompt(ctx: GlobalContext, role: Role) -> None:
    """Property (≥1000 examples): routed+rendered prompt for a role contains
    no fact ID, label, or distinctive value string of any fact the role was
    not granted — including CAD numbers quoted inside denied free text."""
    rc = route(ctx, ROUTING)[role]
    assert leaks(ctx, rc) == []


def test_seeded_deliberate_leak_is_caught() -> None:
    """The check has teeth: a RoleContext hand-injected with a forbidden fact
    (pharmacist given note.free_text quoting a CDR) is detected."""
    ctx = build(
        CADResult(
            quality=ImageQuality(score=0.9, gradable=True),
            classifier=ClassifierOutput(
                p_raw=0.7,
                p_calibrated=0.62,
                tier=ScreeningTier.HIGH,
                model_version="m@r",
            ),
            measurements=Measurements(
                disc_area_px=10000,
                cup_area_px=2500,
                disc_vertical_diameter_px=120,
                cup_vertical_diameter_px=60,
            ),
            cdr=CDRMetrics(vertical=0.5, area_based=0.5),
            mask_qc=MaskQC(
                cup_within_disc=True,
                single_component_each=True,
                disc_area_plausible=True,
                ok=True,
            ),
            segmenter_version="s@r",
            laterality=Laterality.OD,
        ),
        ClinicalNote(free_text="secret marker: CDR 0.83 noted by referrer"),
        None,
        CASE,
    )
    rc = route(ctx, ROUTING)[Role.PHARMACIST]
    assert F.FREE not in {f.id for f in rc.facts}  # router itself is clean
    injected = RoleContext(
        role=rc.role,
        case_id=rc.case_id,
        laterality=rc.laterality,
        facts=(*rc.facts, *tuple(f for f in ctx.facts if f.id == F.FREE)),
        routing_version=rc.routing_version,
    )
    found = leaks(ctx, injected)
    assert f"id:{F.FREE}" in found
    assert f"value:{F.FREE}" in found
