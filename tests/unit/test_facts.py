"""Section 1.5 — fact-ID constants (ARCHITECTURE.md §5.6, AGENTS.md §9.1)."""

from consilium.context.facts import F
from consilium.schemas.enums import QualityFlag

EXPECTED_IDS = {
    "P_CAL": "cad.p_glaucoma.calibrated",
    "P_TIER": "cad.p_glaucoma.tier",
    "CDR_V": "cad.cdr.vertical",
    "CDR_A": "cad.cdr.area_based",
    "DISC_AREA": "cad.disc.area_px",
    "CUP_AREA": "cad.cup.area_px",
    "DISC_VD": "cad.disc.vdiam_px",
    "CUP_VD": "cad.cup.vdiam_px",
    "Q_SCORE": "cad.quality.score",
    "Q_FLAGS": "cad.quality.flags",
    "MASKQC": "cad.maskqc.ok",
    "LAT": "cad.laterality",
    "AGE": "note.age",
    "IOP_OD": "note.iop.od",
    "IOP_OS": "note.iop.os",
    "MEDS": "note.meds",
    "INTOL": "note.intolerances",
    "COMORB": "note.comorbidities",
    "PRIOR_DX": "note.prior_dx",
    "EXAM": "note.exam_findings",
    "FREE": "note.free_text",
    "CAVEAT_EXTRACTION": "note.caveat.extraction",
}


def test_fact_ids_match_canonical_spec() -> None:
    for attr, expected in EXPECTED_IDS.items():
        assert getattr(F, attr) == expected, f"F.{attr} drifted from the §5.6 spec"


def test_no_unexpected_or_duplicate_ids() -> None:
    values = [v for k, v in vars(F).items() if not k.startswith("_") and isinstance(v, str)]
    assert set(values) == set(EXPECTED_IDS.values())
    assert len(values) == len(set(values)), "duplicate fact ID values"


def test_caveat_helper() -> None:
    assert F.caveat("underexposed") == "cad.caveat.underexposed"
    assert F.caveat(QualityFlag.NOT_FUNDUS.value) == "cad.caveat.not_fundus"


def test_caveat_covers_every_quality_flag() -> None:
    for flag in QualityFlag:
        assert F.caveat(flag.value) == f"cad.caveat.{flag.value}"
