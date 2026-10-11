"""Builder tests (§4.1): full CAD coverage, caveat completeness, guarantees."""

import pytest

from consilium.context.builder import build
from consilium.context.facts import F
from consilium.schemas.enums import (
    Confidence,
    FactSource,
    Laterality,
    QualityFlag,
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

CASE = "case-0001"


def make_cad(*, flags: tuple[QualityFlag, ...] = (), mask_ok: bool = True) -> CADResult:
    return CADResult(
        quality=ImageQuality(score=0.9, gradable=True, flags=flags),
        classifier=ClassifierOutput(
            p_raw=0.7, p_calibrated=0.62, tier=ScreeningTier.HIGH, model_version="m@rev+mlflow:r1"
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
            ok=mask_ok,
        ),
        segmenter_version="s@rev+mlflow:r2",
        laterality=Laterality.OD,
    )


def make_note() -> ClinicalNote:
    return ClinicalNote(
        age_years=58,
        iop_mmhg_od=24.0,
        current_medications=("metformin",),
        drug_intolerances_or_allergies=("sulfa",),
        comorbidities=("diabetes",),
        prior_glaucoma_dx="none",
        exam_findings="disc looks suspicious",
        free_text="patient worried",
    )


def make_extraction(confidence: Confidence = Confidence.HIGH) -> NoteExtraction:
    return NoteExtraction(
        note=make_note(),
        extractor_model="gpt-4o-mini-2024-07-18",
        extractor_prompt_version="v1",
        extraction_confidence=confidence,
    )


def test_all_cad_fields_represented() -> None:
    ctx = build(make_cad(), None, None, CASE)
    required = {
        F.P_CAL,
        F.P_TIER,
        F.CDR_V,
        F.CDR_A,
        F.DISC_AREA,
        F.CUP_AREA,
        F.DISC_VD,
        F.CUP_VD,
        F.Q_SCORE,
        F.Q_FLAGS,
        F.MASKQC,
        F.LAT,
    }
    assert required <= ctx.fact_ids()
    assert ctx.laterality is Laterality.OD
    values = {f.id: f.value for f in ctx.facts}
    assert values[F.P_CAL] == 0.62
    assert values[F.CDR_V] == 0.5
    assert values[F.P_TIER] == "high"


def test_units_attached() -> None:
    ctx = build(make_cad(), None, None, CASE)
    units = {f.id: f.unit for f in ctx.facts}
    assert units[F.DISC_AREA] == "px"
    assert units[F.CUP_VD] == "px"
    assert units[F.CDR_V] is None


def test_caveat_for_every_quality_flag() -> None:
    flags = (QualityFlag.UNDEREXPOSED, QualityFlag.BLURRED)
    ctx = build(make_cad(flags=flags), None, None, CASE)
    assert F.caveat("underexposed") in ctx.fact_ids()
    assert F.caveat("blurred") in ctx.fact_ids()
    assert F.caveat("mask_qc") not in ctx.fact_ids()


def test_caveat_for_mask_qc_failure() -> None:
    ctx = build(make_cad(mask_ok=False), None, None, CASE)
    assert F.caveat("mask_qc") in ctx.fact_ids()
    assert {f.value for f in ctx.facts if f.id == F.MASKQC} == {False}


def test_extraction_caveat_only_on_low_confidence() -> None:
    low = build(make_cad(), make_note(), make_extraction(Confidence.LOW), CASE)
    assert F.CAVEAT_EXTRACTION in low.fact_ids()
    high = build(make_cad(), make_note(), make_extraction(Confidence.HIGH), CASE)
    assert F.CAVEAT_EXTRACTION not in high.fact_ids()


def test_note_facts_and_empty_optionals_omitted() -> None:
    ctx = build(make_cad(), make_note(), make_extraction(), CASE)
    values = {f.id: f.value for f in ctx.facts}
    assert values[F.AGE] == 58
    assert values[F.IOP_OD] == 24.0
    assert values[F.MEDS] == ("metformin",)
    assert F.IOP_OS not in ctx.fact_ids()  # None -> omitted

    sparse = build(make_cad(), ClinicalNote(), None, CASE)
    for fid in (F.AGE, F.MEDS, F.EXAM, F.FREE):
        assert fid not in sparse.fact_ids()


def test_no_note_means_no_note_facts() -> None:
    ctx = build(make_cad(), None, None, CASE)
    assert not any(f.source is FactSource.NOTE for f in ctx.facts)


def test_fact_ids_unique() -> None:
    ctx = build(
        make_cad(flags=(QualityFlag.BLURRED,), mask_ok=False),
        make_note(),
        make_extraction(Confidence.LOW),
        CASE,
    )
    ids = [f.id for f in ctx.facts]
    assert len(ids) == len(set(ids))
    with pytest.raises(ValueError, match="duplicate"):
        from consilium.schemas.context import Fact, GlobalContext

        GlobalContext(
            case_id=CASE,
            laterality=Laterality.OD,
            facts=(
                Fact(id=F.LAT, label="a", value="OD", source=FactSource.CAD),
                Fact(id=F.LAT, label="b", value="OD", source=FactSource.CAD),
            ),
        )
