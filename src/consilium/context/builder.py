"""Context Builder (AGENTS.md §9.1, ARCHITECTURE.md §5.6).

Turns ``CADResult`` + ``ClinicalNote`` + ``NoteExtraction`` into a
``GlobalContext`` (C_global) of ``Fact``s with stable IDs from
``context/facts.py``. Guarantees: unique IDs (also enforced by the schema),
every CAD field represented, caveat facts for every quality flag / mask-QC
failure / low note-extraction confidence.

Optional note fields that are None/empty are omitted (a fact that does not
exist cannot be cited); required CAD fields are always emitted.
"""

from consilium.context.facts import F, caveat_for_extraction
from consilium.schemas.context import Fact, GlobalContext
from consilium.schemas.enums import FactSource
from consilium.schemas.note import ClinicalNote, NoteExtraction
from consilium.schemas.vision import CADResult

_CAD_REQUIRED: frozenset[str] = frozenset(
    {
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
)


def _cad_facts(cad: CADResult) -> list[Fact]:
    m, c, q = cad.measurements, cad.classifier, cad.quality
    facts = [
        Fact(
            id=F.P_CAL,
            label="Calibrated glaucoma probability",
            value=c.p_calibrated,
            source=FactSource.CAD,
        ),
        Fact(id=F.P_TIER, label="Screening tier", value=c.tier.value, source=FactSource.CAD),
        Fact(
            id=F.CDR_V,
            label="Vertical cup-to-disc ratio",
            value=cad.cdr.vertical,
            source=FactSource.CAD,
        ),
        Fact(
            id=F.CDR_A,
            label="Area-based cup-to-disc ratio",
            value=cad.cdr.area_based,
            source=FactSource.CAD,
        ),
        Fact(
            id=F.DISC_AREA,
            label="Disc area",
            value=m.disc_area_px,
            unit="px",
            source=FactSource.CAD,
        ),
        Fact(
            id=F.CUP_AREA, label="Cup area", value=m.cup_area_px, unit="px", source=FactSource.CAD
        ),
        Fact(
            id=F.DISC_VD,
            label="Disc vertical diameter",
            value=m.disc_vertical_diameter_px,
            unit="px",
            source=FactSource.CAD,
        ),
        Fact(
            id=F.CUP_VD,
            label="Cup vertical diameter",
            value=m.cup_vertical_diameter_px,
            unit="px",
            source=FactSource.CAD,
        ),
        Fact(id=F.Q_SCORE, label="Image quality score", value=q.score, source=FactSource.CAD),
        Fact(
            id=F.Q_FLAGS,
            label="Image quality flags",
            value=tuple(f.value for f in q.flags),
            source=FactSource.CAD,
        ),
        Fact(id=F.MASKQC, label="Mask QC passed", value=cad.mask_qc.ok, source=FactSource.CAD),
        Fact(id=F.LAT, label="Laterality", value=cad.laterality.value, source=FactSource.CAD),
    ]
    for flag in q.flags:
        facts.append(
            Fact(
                id=F.caveat(flag.value),
                label=f"Quality caveat: {flag.value}",
                value=True,
                source=FactSource.CAD,
            )
        )
    if not cad.mask_qc.ok:
        facts.append(
            Fact(id=F.caveat("mask_qc"), label="Mask QC failed", value=True, source=FactSource.CAD)
        )
    return facts


def _note_facts(note: ClinicalNote) -> list[Fact]:
    facts: list[Fact] = []
    if note.age_years is not None:
        facts.append(
            Fact(id=F.AGE, label="Age", value=note.age_years, unit="years", source=FactSource.NOTE)
        )
    if note.iop_mmhg_od is not None:
        facts.append(
            Fact(
                id=F.IOP_OD,
                label="IOP (OD)",
                value=note.iop_mmhg_od,
                unit="mmHg",
                source=FactSource.NOTE,
            )
        )
    if note.iop_mmhg_os is not None:
        facts.append(
            Fact(
                id=F.IOP_OS,
                label="IOP (OS)",
                value=note.iop_mmhg_os,
                unit="mmHg",
                source=FactSource.NOTE,
            )
        )
    if note.current_medications:
        facts.append(
            Fact(
                id=F.MEDS,
                label="Current medications",
                value=note.current_medications,
                source=FactSource.NOTE,
            )
        )
    if note.drug_intolerances_or_allergies:
        facts.append(
            Fact(
                id=F.INTOL,
                label="Drug intolerances/allergies",
                value=note.drug_intolerances_or_allergies,
                source=FactSource.NOTE,
            )
        )
    if note.comorbidities:
        facts.append(
            Fact(
                id=F.COMORB, label="Comorbidities", value=note.comorbidities, source=FactSource.NOTE
            )
        )
    if note.prior_glaucoma_dx is not None:
        facts.append(
            Fact(
                id=F.PRIOR_DX,
                label="Prior glaucoma diagnosis",
                value=note.prior_glaucoma_dx,
                source=FactSource.NOTE,
            )
        )
    if note.exam_findings is not None:
        facts.append(
            Fact(
                id=F.EXAM,
                label="Exam findings (untrusted text)",
                value=note.exam_findings,
                source=FactSource.NOTE,
            )
        )
    if note.free_text is not None:
        facts.append(
            Fact(
                id=F.FREE,
                label="Note free text (untrusted)",
                value=note.free_text,
                source=FactSource.NOTE,
            )
        )
    return facts


def build(
    cad: CADResult,
    note: ClinicalNote | None,
    extraction: NoteExtraction | None,
    case_id: str,
) -> GlobalContext:
    """Build C_global. ``note``/``extraction`` are None when no note was submitted."""
    facts = _cad_facts(cad)
    if note is not None:
        facts.extend(_note_facts(note))
    if extraction is not None:
        caveat_id = caveat_for_extraction(extraction.extraction_confidence)
        if caveat_id is not None:
            facts.append(
                Fact(
                    id=caveat_id,
                    label="Low note-extraction confidence",
                    value=True,
                    source=FactSource.NOTE,
                )
            )
    missing = _CAD_REQUIRED - {f.id for f in facts}
    if missing:  # defensive: schema guarantees uniqueness, builder guarantees coverage
        raise ValueError(f"missing required CAD facts: {sorted(missing)}")
    return GlobalContext(case_id=case_id, laterality=cad.laterality, facts=tuple(facts))
