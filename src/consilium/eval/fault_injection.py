"""Seeded fault injection for verifier/critic evaluation (AGENTS.md §13.4).

Takes known-good drafts and applies deterministic mutations, one fault class
at a time, then measures which layer catches them. Phase 7 wires the
**deterministic layer only** (``verify_draft``); Phase 8 adds the LLM critic
and the combined layer-by-layer report.

Fault classes (§13.4): numeric gross + subtle, fabricated finding with/without
ref, deleted mandatory finding, dose insertion, laterality flip, suppressed
disagreement. Each class carries the ``IssueType`` the deterministic layer is
EXPECTED to catch it with — ``fabricated_with_ref`` expects ``None``: an
entailed-looking claim with a valid ref is the LLM critic's job, and the
report records that honestly instead of pretending the deterministic layer
sees it.

The corpus is synthetic and versioned in code. The recall report is stored as
committed JSON (``tests/fixtures/fault_injection_deterministic_v1.json``) and
drift-checked by ``tests/unit/test_verifier.py`` (I-11: measured numbers only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from consilium.context.builder import build
from consilium.context.facts import F
from consilium.context.router import RoutingConfig, load_routing_config
from consilium.schemas.agents import (
    Claim,
    DirectorDraft,
    Disagreement,
    Finding,
    Recommendation,
    SubReport,
)
from consilium.schemas.audit import Issue
from consilium.schemas.context import GlobalContext
from consilium.schemas.enums import (
    ClaimKind,
    Confidence,
    IssueType,
    Laterality,
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
from consilium.verify.deterministic import verify_draft
from consilium.vision.config import Thresholds, load_thresholds

REPORT_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "fault_injection_deterministic_v1.json"
)


class FaultClass(StrEnum):
    NUMERIC_GROSS = "numeric_gross"
    NUMERIC_SUBTLE = "numeric_subtle"
    FABRICATED_WITH_REF = "fabricated_with_ref"
    FABRICATED_NO_REF = "fabricated_no_ref"
    DELETED_MANDATORY = "deleted_mandatory"
    DOSE_INSERTION = "dose_insertion"
    LATERALITY_FLIP = "laterality_flip"
    SUPPRESSED_DISAGREEMENT = "suppressed_disagreement"


# IssueType the deterministic layer is expected to catch each class with.
# None = expected miss at this layer (entailment fault → LLM critic, P8).
EXPECTED_DETECTION: dict[FaultClass, IssueType | None] = {
    FaultClass.NUMERIC_GROSS: IssueType.NUMERIC_MISMATCH,
    FaultClass.NUMERIC_SUBTLE: IssueType.NUMERIC_MISMATCH,
    FaultClass.FABRICATED_WITH_REF: None,
    FaultClass.FABRICATED_NO_REF: IssueType.BAD_REFERENCE,
    FaultClass.DELETED_MANDATORY: IssueType.OMITTED_FINDING,
    FaultClass.DOSE_INSERTION: IssueType.UNSAFE_LANGUAGE,
    FaultClass.LATERALITY_FLIP: IssueType.UNSUPPORTED,
    FaultClass.SUPPRESSED_DISAGREEMENT: IssueType.DISAGREEMENT_SUPPRESSED,
}

CDR_FACT_VALUE = 0.5


@dataclass(frozen=True)
class CorpusEntry:
    """One known-good case: draft + the evidence it was verified against."""

    name: str
    draft: DirectorDraft
    global_ctx: GlobalContext
    sub_reports: tuple[SubReport, ...]


def _clean_cad() -> CADResult:
    return CADResult(
        quality=ImageQuality(score=0.9, gradable=True, flags=()),
        classifier=ClassifierOutput(
            p_raw=0.7, p_calibrated=0.62, tier=ScreeningTier.HIGH, model_version="m@rev+mlflow:r1"
        ),
        measurements=Measurements(
            disc_area_px=10000,
            cup_area_px=2500,
            disc_vertical_diameter_px=120,
            cup_vertical_diameter_px=60,
        ),
        cdr=CDRMetrics(vertical=CDR_FACT_VALUE, area_based=0.5),
        mask_qc=MaskQC(
            cup_within_disc=True,
            single_component_each=True,
            disc_area_plausible=True,
            ok=True,
        ),
        segmenter_version="s@rev+mlflow:r2",
        laterality=Laterality.OD,
    )


def _clean_note() -> ClinicalNote:
    return ClinicalNote(
        age_years=58,
        iop_mmhg_od=24.0,
        current_medications=("metformin",),
        drug_intolerances_or_allergies=("sulfa",),
        comorbidities=("diabetes",),
        prior_glaucoma_dx="none",
        exam_findings="disc appearance noted as suspicious on referral",
        free_text="patient worried about family history",
    )


def _sub_reports() -> tuple[SubReport, ...]:
    ophth = SubReport(
        role=Role.OPHTHALMOLOGIST,
        findings=(
            Finding(
                text=(
                    "Vertical CDR is enlarged for the tier and warrants urgent"
                    " referral for glaucoma evaluation."
                ),
                evidence_refs=(F.CDR_V, F.P_TIER),
                confidence=Confidence.MODERATE,
            ),
        ),
        recommendations=(),
    )
    optom = SubReport(
        role=Role.OPTOMETRIST,
        findings=(
            Finding(
                text="Findings suitable for routine monitoring with annual review.",
                evidence_refs=(F.P_TIER, F.CDR_V),
                confidence=Confidence.LOW,
            ),
        ),
        recommendations=(),
    )
    pharm = SubReport(
        role=Role.PHARMACIST,
        findings=(
            Finding(
                text=(
                    "Sulfa intolerance noted; consider non-sulfonamide drug"
                    " classes subject to prescriber judgement."
                ),
                evidence_refs=(F.INTOL,),
                confidence=Confidence.MODERATE,
            ),
        ),
        recommendations=(),
    )
    return (ophth, optom, pharm)


def _clean_draft() -> DirectorDraft:
    return DirectorDraft(
        impression=(
            "Findings raise suspicion of glaucomatous optic neuropathy and"
            " warrant evaluation for glaucoma."
        ),
        claims=(
            Claim(
                id="c1",
                kind=ClaimKind.NUMERIC,
                text="Vertical cup-to-disc ratio is 0.5.",
                evidence_refs=(F.CDR_V,),
                source_roles=(Role.OPHTHALMOLOGIST,),
                numeric_value=CDR_FACT_VALUE,
                numeric_ref=F.CDR_V,
            ),
            Claim(
                id="c2",
                kind=ClaimKind.CATEGORICAL,
                text="Screening tier is high.",
                evidence_refs=(F.P_TIER, F.P_CAL),
                source_roles=(Role.OPHTHALMOLOGIST, Role.OPTOMETRIST),
            ),
            Claim(
                id="c3",
                kind=ClaimKind.CATEGORICAL,
                text="Findings pertain to the right eye (OD).",
                evidence_refs=(F.LAT,),
                source_roles=(Role.OPHTHALMOLOGIST,),
            ),
            Claim(
                id="c4",
                kind=ClaimKind.NUMERIC,
                text="Calibrated probability is 0.62.",
                evidence_refs=(F.P_CAL,),
                source_roles=(Role.OPHTHALMOLOGIST, Role.OPTOMETRIST),
                numeric_value=0.62,
                numeric_ref=F.P_CAL,
            ),
        ),
        disagreements=(
            Disagreement(
                topic="escalation urgency",
                positions=(
                    (Role.OPHTHALMOLOGIST, "urgent referral warranted"),
                    (Role.OPTOMETRIST, "routine monitoring sufficient"),
                ),
                resolution="left to clinician",
            ),
        ),
        recommendations=(
            Recommendation(
                text="Repeat imaging in 6 months if not referred sooner.",
                triggered_by=(F.P_TIER,),
            ),
        ),
        limitations=(
            "Screening support only; not a diagnosis.",
            "No visual field or OCT data available.",
        ),
    )


def known_good_corpus() -> tuple[CorpusEntry, ...]:
    """The v1 known-good corpus (synthetic; every draft must verify clean)."""
    note = _clean_note()
    ctx = build(
        _clean_cad(),
        note,
        NoteExtraction(
            note=note,
            extractor_model="gpt-4o-mini-2024-07-18",
            extractor_prompt_version="v1",
            extraction_confidence=Confidence.HIGH,
        ),
        "case-fault-corpus-v1",
    )
    return (
        CorpusEntry(
            name="clean_v1",
            draft=_clean_draft(),
            global_ctx=ctx,
            sub_reports=_sub_reports(),
        ),
    )


def apply_fault(entry: CorpusEntry, fault: FaultClass) -> CorpusEntry:
    """Apply one deterministic mutation; result stays schema-valid (I-8)."""
    draft = entry.draft
    if fault is FaultClass.NUMERIC_GROSS:
        draft = _mutate_claim(draft, "c1", numeric_value=0.8)
    elif fault is FaultClass.NUMERIC_SUBTLE:
        draft = _mutate_claim(draft, "c1", numeric_value=0.53)  # tol is 0.02
    elif fault is FaultClass.FABRICATED_WITH_REF:
        draft = _add_claim(
            draft,
            Claim(
                id="c9",
                kind=ClaimKind.INTERPRETIVE,
                text="The optic disc shows a clear inferior notch.",
                evidence_refs=(F.EXAM,),
                source_roles=(Role.OPHTHALMOLOGIST,),
            ),
        )
    elif fault is FaultClass.FABRICATED_NO_REF:
        draft = _add_claim(
            draft,
            Claim(
                id="c9",
                kind=ClaimKind.INTERPRETIVE,
                text="The optic disc shows a clear inferior notch.",
                evidence_refs=("cad.cdr.made_up",),
                source_roles=(Role.OPHTHALMOLOGIST,),
            ),
        )
    elif fault is FaultClass.DELETED_MANDATORY:
        draft = draft.model_copy(update={"claims": tuple(c for c in draft.claims if c.id != "c1")})
    elif fault is FaultClass.DOSE_INSERTION:
        draft = draft.model_copy(
            update={
                "recommendations": (
                    *draft.recommendations,
                    Recommendation(
                        text="Consider latanoprost 0.005% once daily.",
                        triggered_by=(F.P_TIER,),
                    ),
                )
            }
        )
    elif fault is FaultClass.LATERALITY_FLIP:
        draft = _mutate_claim(draft, "c3", text="Findings pertain to the left eye (OS).")
    elif fault is FaultClass.SUPPRESSED_DISAGREEMENT:
        draft = draft.model_copy(update={"disagreements": ()})
    else:  # pragma: no cover - exhaustiveness guard
        raise ValueError(f"unknown fault class {fault}")
    return CorpusEntry(
        name=f"{entry.name}__{fault.value}",
        draft=draft,
        global_ctx=entry.global_ctx,
        sub_reports=entry.sub_reports,
    )


def _mutate_claim(draft: DirectorDraft, claim_id: str, **updates: object) -> DirectorDraft:
    claims = tuple(c.model_copy(update=updates) if c.id == claim_id else c for c in draft.claims)
    return draft.model_copy(update={"claims": claims})


def _add_claim(draft: DirectorDraft, claim: Claim) -> DirectorDraft:
    return draft.model_copy(update={"claims": (*draft.claims, claim)})


@dataclass(frozen=True)
class FaultReport:
    """Measured recall per fault class + clean-draft false-rejection rate."""

    per_class: dict[str, dict[str, object]]
    clean_total: int
    clean_rejected: int

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": "deterministic",
            "per_class": self.per_class,
            "clean_total": self.clean_total,
            "clean_rejected": self.clean_rejected,
            "clean_false_rejection_rate": (
                self.clean_rejected / self.clean_total if self.clean_total else 0.0
            ),
        }


def evaluate_deterministic(
    corpus: tuple[CorpusEntry, ...], thresholds: Thresholds, routing: RoutingConfig
) -> FaultReport:
    """Run every fault class against ``verify_draft`` and measure recall."""
    per_class: dict[str, dict[str, object]] = {}
    clean_rejected = 0
    for entry in corpus:
        issues: tuple[Issue, ...] = verify_draft(
            entry.draft, entry.global_ctx, entry.sub_reports, thresholds, routing
        )
        if issues:
            clean_rejected += 1
    for fault in FaultClass:
        expected = EXPECTED_DETECTION[fault]
        caught = 0
        for entry in corpus:
            issues = verify_draft(
                apply_fault(entry, fault).draft,
                entry.global_ctx,
                entry.sub_reports,
                thresholds,
                routing,
            )
            if expected is not None and any(i.type is expected for i in issues):
                caught += 1
        total = len(corpus)
        per_class[fault.value] = {
            "expected_issue": expected.value if expected is not None else None,
            "caught": caught,
            "total": total,
            "recall": caught / total if total else 0.0,
        }
    return FaultReport(per_class=per_class, clean_total=len(corpus), clean_rejected=clean_rejected)


def main() -> None:
    """Regenerate the committed recall report (deterministic layer, v1 corpus)."""
    repo = Path(__file__).resolve().parents[3]
    report = evaluate_deterministic(
        known_good_corpus(),
        load_thresholds(repo / "configs" / "thresholds.yaml"),
        load_routing_config(repo / "configs" / "routing.yaml"),
    )
    REPORT_PATH.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
