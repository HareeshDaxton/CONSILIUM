"""Eval fixtures: recorded CAD inputs + the synthetic recorded-LLM responder
(plan Section 13.1; AGENTS.md §19, §20; SKILLS.md ablation-run).

Three things live here:

1. ``smoke_manifest()`` — the generator of the committed smoke case set
   (``eval/data/smoke_v1.json``). Synthetic CADResults are internally
   consistent (vCDR = cup/disc vertical diameter, area CDR = sqrt of the area
   ratio, nested masks) and cover the paths the metrics must exercise:
   clean, no note, quality flag, failed mask QC, low-confidence extraction,
   and two curated disagreement cases.
2. ``record_cases()`` — freezes a FULL case set: runs the vision pipeline
   ONCE per image through a ``VisionClient`` and stores the resulting
   ``CADResult`` in the manifest, so every later stage reads the same
   recorded output (vision never varies between stages).
3. ``synthetic_llm()`` — the offline LLM for ``eval-smoke``: a
   ``FakeLLMClient`` loaded with deterministic, schema-valid responses
   derived from the context each agent will ACTUALLY be shown. It is an
   honest stand-in that exercises the harness wiring (routing, fan-out,
   director, audit loop, metering, run store); its outputs are templated, so
   smoke-run metrics measure the pipeline mechanics, NEVER model behaviour,
   and are never quoted as results (I-11). Every stored smoke run says so in
   its provenance (``llm_backend``).

Responder determinism: every choice is a sha256-keyed pick over
``(seed, case_id, sample, …)`` — no global RNG. The Director's first draft
carries a seeded numeric slip at ``SMOKE_DRAFT_SLIP_RATE`` (identical across
S1-S4 for the same case+sample, so stages stay paired); its revision is
clean. That lets the smoke run exercise the S4 reject→revise path end to end.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from consilium.context.facts import F
from consilium.core.settings import DEFAULT_CRITIC_PROMPT_VERSION, DEFAULT_DIRECTOR_PROMPT_VERSION
from consilium.eval.datasets import SMOKE_MANIFEST_PATH, CaseManifest, EvalCase, manifest_bytes
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.schemas.agents import (
    Claim,
    DirectorDraft,
    Disagreement,
    Finding,
    Recommendation,
    SubReport,
)
from consilium.schemas.context import GlobalContext, RoleContext
from consilium.schemas.enums import (
    ClaimKind,
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
from consilium.vision import pipeline as vision_pipeline
from consilium.vision.client import VisionClient
from consilium.vision.config import ModelsConfig, Thresholds

SMOKE_DATASET_ID = "consilium-smoke-v1"
SYNTHETIC_LLM_BACKEND = "synthetic-recorded-v1"
SMOKE_DRAFT_SLIP_RATE = 0.34  # share of first drafts carrying a seeded numeric slip
_EXTRACTOR_MODEL = "synthetic-structured-note"  # smoke notes are authored structured

# Configured disagreement topics (configs/thresholds.yaml) the curated cases use.
TOPIC_ESCALATION = "escalation urgency"
TOPIC_CDR = "cdr interpretation"


# ---------------------------------------------------------------------------
# 1. Smoke case set (synthetic CAD + structured notes)
# ---------------------------------------------------------------------------
def _cad(
    *,
    p_cal: float,
    tier: ScreeningTier,
    cdr_v: float,
    laterality: Laterality,
    flags: tuple[QualityFlag, ...] = (),
    mask_ok: bool = True,
) -> CADResult:
    """Internally consistent synthetic CAD output (nested masks, §8.2)."""
    disc_vd = 120
    cup_vd = round(cdr_v * disc_vd)
    disc_area = round(math.pi * (disc_vd / 2) ** 2)
    cup_area = round(math.pi * (cup_vd / 2) ** 2)
    return CADResult(
        quality=ImageQuality(score=0.62 if flags else 0.91, gradable=True, flags=flags),
        classifier=ClassifierOutput(
            p_raw=min(1.0, p_cal + 0.05),
            p_calibrated=p_cal,
            tier=tier,
            model_version="synthetic@smoke+mlflow:none",
        ),
        measurements=Measurements(
            disc_area_px=disc_area,
            cup_area_px=cup_area,
            disc_vertical_diameter_px=disc_vd,
            cup_vertical_diameter_px=cup_vd,
        ),
        cdr=CDRMetrics(
            vertical=round(cup_vd / disc_vd, 4),
            area_based=round(math.sqrt(cup_area / disc_area), 4),
        ),
        mask_qc=MaskQC(
            cup_within_disc=True,
            single_component_each=mask_ok,
            disc_area_plausible=True,
            ok=mask_ok,
        ),
        segmenter_version="synthetic@smoke+mlflow:none",
        laterality=laterality,
    )


def _extraction(note: ClinicalNote, confidence: Confidence) -> NoteExtraction:
    return NoteExtraction(
        note=note,
        extractor_model=_EXTRACTOR_MODEL,
        extractor_prompt_version="v1",
        extraction_confidence=confidence,
    )


def _case(
    case_id: str,
    cad: CADResult,
    note: ClinicalNote | None = None,
    *,
    confidence: Confidence = Confidence.HIGH,
    topics: tuple[str, ...] = (),
    split: str = "dev",
) -> EvalCase:
    return EvalCase.model_validate(
        {
            "case_id": case_id,
            "split": split,
            "cad": cad,
            "note": note,
            "extraction": _extraction(note, confidence) if note is not None else None,
            "cad_provenance": "synthetic",
            "curated_disagreement_topics": topics,
        }
    )


def smoke_manifest() -> CaseManifest:
    """Generator of the committed smoke set (synthetic; no patient data)."""
    note_a = ClinicalNote(
        age_years=64,
        iop_mmhg_od=25.0,
        current_medications=("metformin",),
        drug_intolerances_or_allergies=("sulfonamides",),
        comorbidities=("type 2 diabetes",),
        prior_glaucoma_dx="none recorded",
    )
    note_b = ClinicalNote(age_years=51, iop_mmhg_os=18.0, comorbidities=("asthma",))
    note_c = ClinicalNote(
        age_years=70,
        iop_mmhg_od=27.0,
        current_medications=("amlodipine",),
        comorbidities=("hypertension",),
    )
    cases = (
        _case(
            "smoke-001",
            _cad(p_cal=0.82, tier=ScreeningTier.HIGH, cdr_v=0.72, laterality=Laterality.OD),
            note_a,
        ),
        _case(
            "smoke-002",
            _cad(p_cal=0.12, tier=ScreeningTier.LOW, cdr_v=0.35, laterality=Laterality.OS),
        ),
        _case(
            "smoke-003",
            _cad(
                p_cal=0.45,
                tier=ScreeningTier.INTERMEDIATE,
                cdr_v=0.55,
                laterality=Laterality.UNKNOWN,
                flags=(QualityFlag.BLURRED,),
            ),
            note_b,
            confidence=Confidence.LOW,
        ),
        _case(
            "smoke-004",
            _cad(
                p_cal=0.77,
                tier=ScreeningTier.HIGH,
                cdr_v=0.68,
                laterality=Laterality.OD,
                mask_ok=False,
            ),
            note_c,
        ),
        _case(
            "smoke-005",
            _cad(p_cal=0.51, tier=ScreeningTier.INTERMEDIATE, cdr_v=0.6, laterality=Laterality.OS),
            note_b,
            topics=(TOPIC_ESCALATION, TOPIC_CDR),
        ),
        _case(
            "smoke-006",
            _cad(p_cal=0.69, tier=ScreeningTier.HIGH, cdr_v=0.7, laterality=Laterality.OD),
            note_a,
            topics=(TOPIC_ESCALATION, TOPIC_CDR),
            split="holdout",
        ),
    )
    return CaseManifest(
        dataset_id=SMOKE_DATASET_ID,
        description=(
            "Synthetic smoke set for eval-smoke (CI-safe). Wiring only: never a "
            "source of reported numbers (I-11)."
        ),
        cases=cases,
    )


# ---------------------------------------------------------------------------
# 2. Recording a full set (vision called ONCE, then frozen)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RawEvalInput:
    """One image + synthetic structured note to freeze into an EvalCase."""

    case_id: str
    image: bytes
    laterality: Laterality
    note: ClinicalNote | None = None
    split: str = "dev"
    curated_disagreement_topics: tuple[str, ...] = ()


async def record_cases(
    inputs: Sequence[RawEvalInput],
    *,
    client: VisionClient,
    thresholds: Thresholds,
    models: ModelsConfig,
) -> tuple[EvalCase, ...]:
    """Run vision once per input and freeze the CADResult (SKILLS.md
    ablation-run step 1: recorded endpoint responses held constant)."""
    revision = models.endpoint.pinned_revision
    cases: list[EvalCase] = []
    for item in inputs:
        cad = await vision_pipeline.run(
            item.image,
            client=client,
            thresholds=thresholds,
            models=models,
            laterality=item.laterality,
        )
        cases.append(
            EvalCase.model_validate(
                {
                    "case_id": item.case_id,
                    "split": item.split,
                    "cad": cad,
                    "note": item.note,
                    "extraction": (
                        _extraction(item.note, Confidence.HIGH) if item.note is not None else None
                    ),
                    "cad_provenance": f"recorded:{revision}",
                    "curated_disagreement_topics": item.curated_disagreement_topics,
                }
            )
        )
    return tuple(cases)


# ---------------------------------------------------------------------------
# 3. Synthetic recorded-LLM responder (offline smoke only)
# ---------------------------------------------------------------------------
def _digest(*key: object) -> int:
    raw = "|".join(str(k) for k in key).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def pick(options: Sequence[str], *key: object) -> str:
    """Deterministic choice keyed by ``key`` (no RNG state)."""
    return options[_digest(*key) % len(options)]


def chance(rate: float, *key: object) -> bool:
    """Deterministic Bernoulli(rate) keyed by ``key``."""
    return (_digest(*key) % 10_000) < round(rate * 10_000)


# Neutral templates: no numerals (verifier stray-number scan) and no stance
# markers from configs/thresholds.yaml disagreement_topics (only the curated
# cases below take positions on purpose).
_FACT_PHRASES: dict[str, tuple[str, ...]] = {
    F.CDR_V: (
        "The vertical cup-to-disc measurement is available for structural interpretation.",
        "Vertical cup-to-disc ratio has been measured for this eye.",
    ),
    F.CDR_A: ("Area-based cup-to-disc ratio is available as a secondary structural measure.",),
    F.P_TIER: (
        "The screening tier frames the level of clinical attention indicated.",
        "The calibrated screening tier is the starting point for the care pathway.",
    ),
    F.P_CAL: ("The calibrated glaucoma probability accompanies the screening tier.",),
    F.DISC_VD: ("Disc vertical extent is available from segmentation.",),
    F.CUP_VD: ("Cup vertical extent is available from segmentation.",),
    F.Q_SCORE: ("Image quality was scored before interpretation.",),
    F.MASKQC: ("Segmentation quality control status is recorded.",),
    F.IOP_OD: ("Intraocular pressure for the right eye is recorded in the note.",),
    F.IOP_OS: ("Intraocular pressure for the left eye is recorded in the note.",),
    F.AGE: ("Patient age is recorded in the note.",),
    F.MEDS: (
        "Current medications should be checked against glaucoma drug classes, subject to"
        " prescriber judgement.",
    ),
    F.INTOL: (
        "Listed intolerances are relevant to drug-class selection, subject to prescriber"
        " judgement.",
    ),
    F.COMORB: (
        "Listed comorbidities may bear on drug-class choice, subject to prescriber judgement.",
    ),
    F.PRIOR_DX: ("A prior glaucoma history entry is present in the note.",),
}

# Role focus order: which visible facts a role comments on first (capped).
_ROLE_FOCUS: dict[Role, tuple[str, ...]] = {
    Role.OPHTHALMOLOGIST: (
        F.CDR_V,
        F.CDR_A,
        F.P_CAL,
        F.CUP_VD,
        F.DISC_VD,
        F.IOP_OD,
        F.IOP_OS,
        F.P_TIER,
    ),
    Role.OPTOMETRIST: (F.P_TIER, F.Q_SCORE, F.MASKQC, F.CDR_V, F.IOP_OD, F.IOP_OS, F.AGE),
    Role.PHARMACIST: (F.INTOL, F.MEDS, F.COMORB, F.P_TIER, F.PRIOR_DX, F.AGE, F.CDR_V),
}
_MAX_FINDINGS = 4

_ROLE_DECLINE: dict[Role, str] = {
    Role.OPHTHALMOLOGIST: "Medication selection is outside this role's scope.",
    Role.OPTOMETRIST: "Medication selection is outside this role's scope.",
    Role.PHARMACIST: "Optic disc morphology is outside this role's scope.",
}

# Curated-disagreement positions: polar on BOTH configured topics.
_CURATED_POSITION: dict[Role, str] = {
    Role.OPHTHALMOLOGIST: ("Cup-to-disc appearance is concerning and urgent referral is advised."),
    Role.OPTOMETRIST: (
        "Cup-to-disc appearance looks physiological; routine monitoring with follow-up"
        " is sufficient."
    ),
}


def synthetic_subreport(
    role: Role, ctx: RoleContext, case: EvalCase, seed: int, sample: int
) -> SubReport:
    """Templated SubReport citing ONLY facts in ``ctx`` (scope-valid by
    construction — the real scope validator still checks it)."""
    visible = ctx.allowed_ids()
    findings: list[Finding] = []
    if case.curated_disagreement_topics and role in _CURATED_POSITION:
        refs = tuple(r for r in (F.CDR_V, F.P_TIER) if r in visible)
        if refs:
            findings.append(
                Finding(text=_CURATED_POSITION[role], evidence_refs=refs, confidence=Confidence.LOW)
            )
    for fact_id in _ROLE_FOCUS[role]:
        if len(findings) >= _MAX_FINDINGS:
            break
        if fact_id not in visible or fact_id not in _FACT_PHRASES:
            continue
        findings.append(
            Finding(
                text=pick(_FACT_PHRASES[fact_id], seed, case.case_id, sample, role, fact_id),
                evidence_refs=(fact_id,),
                confidence=Confidence.MODERATE,
            )
        )
    recs: tuple[Recommendation, ...] = ()
    if F.P_TIER in visible:
        recs = (
            Recommendation(
                text="Interpret the screening tier together with the clinical examination.",
                triggered_by=(F.P_TIER,),
            ),
        )
    return SubReport(
        role=role,
        findings=tuple(findings),
        recommendations=recs,
        uncertainties=("No visual field or OCT data were provided.",),
        declined_out_of_scope=(_ROLE_DECLINE[role],),
    )


def _laterality_text(laterality: str) -> str:
    if laterality == Laterality.OD:
        return "Findings pertain to the right eye (OD)."
    if laterality == Laterality.OS:
        return "Findings pertain to the left eye (OS)."
    return "Laterality is recorded as unknown."


def synthetic_draft(
    global_ctx: GlobalContext,
    reports: dict[Role, SubReport],
    case: EvalCase,
    seed: int,
    sample: int,
    *,
    slip: bool,
) -> DirectorDraft:
    """Templated DirectorDraft (verifier-clean unless ``slip``: then the vCDR
    claim carries a numeric slip the deterministic verifier must catch)."""
    facts = {f.id: f for f in global_ctx.facts}
    cdr = facts[F.CDR_V].value
    assert isinstance(cdr, float)
    lead = (Role.OPHTHALMOLOGIST,)
    value = round(min(1.0, cdr + 0.1), 4) if slip else cdr
    claims: list[Claim] = [
        Claim(
            id="c1",
            kind=ClaimKind.NUMERIC,
            text="Vertical cup-to-disc ratio is as measured.",
            evidence_refs=(F.CDR_V,),
            source_roles=lead,
            numeric_value=value,
            numeric_ref=F.CDR_V,
        ),
        Claim(
            id="c2",
            kind=ClaimKind.CATEGORICAL,
            text="Screening tier is as computed.",
            evidence_refs=(F.P_TIER,),
            source_roles=lead,
        ),
        Claim(
            id="c3",
            kind=ClaimKind.CATEGORICAL,
            text=_laterality_text(global_ctx.laterality),
            evidence_refs=(F.LAT,),
            source_roles=lead,
        ),
    ]
    for role, report in reports.items():
        for finding in report.findings:
            claims.append(
                Claim(
                    id=f"c{len(claims) + 1}",
                    kind=ClaimKind.INTERPRETIVE,
                    text=finding.text,
                    evidence_refs=finding.evidence_refs,
                    source_roles=(role,),
                )
            )
    disagreements: tuple[Disagreement, ...] = ()
    if case.curated_disagreement_topics and all(r in reports for r in _CURATED_POSITION):
        positions = tuple((r, _CURATED_POSITION[r]) for r in _CURATED_POSITION)
        disagreements = tuple(
            Disagreement(topic=topic, positions=positions, resolution="left to clinician")
            for topic in case.curated_disagreement_topics
        )
    caveats = [
        "Caveat carried: " + f.id.rsplit(".", 1)[-1].replace("_", " ") + "."
        for f in global_ctx.facts
        if f.id.startswith("cad.caveat.") or f.id == F.CAVEAT_EXTRACTION
    ]
    impression = pick(
        (
            "Findings warrant evaluation for glaucoma by a clinician.",
            "Findings raise suspicion that warrants clinician evaluation for glaucoma.",
        ),
        seed,
        case.case_id,
        sample,
        "impression",
    )
    return DirectorDraft(
        impression=impression,
        claims=tuple(claims),
        disagreements=disagreements,
        recommendations=(
            Recommendation(
                text="Clinician evaluation of the screening findings is advised.",
                triggered_by=(F.P_TIER,),
            ),
        ),
        limitations=(
            "Screening support only; not a diagnosis.",
            "No visual field or OCT data available.",
            *caveats,
        ),
    )


_APPROVE: dict[str, Any] = {"approved": True, "issues": [], "round": 1}


def _valid(payload: dict[str, Any]) -> CannedCall:
    return CannedCall(FakeMode.VALID, payload)


def synthetic_llm(
    *,
    case: EvalCase,
    global_ctx: GlobalContext,
    role_contexts: dict[Role, RoleContext] | None,
    seed: int,
    sample: int,
    prompt_version: str = "v1",
) -> FakeLLMClient:
    """Offline LLM for one (case, stage, sample) run.

    ``role_contexts`` None ⇒ single-agent stage (S1): only the
    ``single_agent`` key is loaded. Otherwise the specialist responses are
    derived from each role's routed context and the director/critic keys are
    loaded; the director answers draft → clean revision.
    """
    slip = chance(SMOKE_DRAFT_SLIP_RATE, seed, case.case_id, sample, "draft-slip")
    if role_contexts is None:
        # S1: one agent wearing all three hats over the FULL context. It holds
        # no dissent with itself, so curated positions are not taken.
        solo = case.model_copy(update={"curated_disagreement_topics": ()})
        hats = {
            role: synthetic_subreport(
                role,
                RoleContext(
                    role=role,
                    case_id=global_ctx.case_id,
                    laterality=global_ctx.laterality,
                    facts=global_ctx.facts,
                    routing_version="single-agent-full-context",
                ),
                solo,
                seed,
                sample,
            )
            for role in _ROLE_FOCUS
        }
        first = synthetic_draft(global_ctx, hats, solo, seed, sample, slip=slip)
        clean = synthetic_draft(global_ctx, hats, solo, seed, sample, slip=False)
        return FakeLLMClient(
            {
                ("single_agent", prompt_version, "default"): CannedResponse(
                    (_valid(first.model_dump(mode="json")), _valid(clean.model_dump(mode="json")))
                )
            }
        )
    reports = {
        role: synthetic_subreport(role, ctx, case, seed, sample)
        for role, ctx in role_contexts.items()
    }
    first = synthetic_draft(global_ctx, reports, case, seed, sample, slip=slip)
    clean = synthetic_draft(global_ctx, reports, case, seed, sample, slip=False)
    keys: dict[tuple[str, str, str], CannedResponse] = {
        (role.value, prompt_version, "default"): CannedResponse(
            (_valid(report.model_dump(mode="json")),)
        )
        for role, report in reports.items()
    }
    keys[("director", DEFAULT_DIRECTOR_PROMPT_VERSION, "default")] = CannedResponse(
        (_valid(first.model_dump(mode="json")), _valid(clean.model_dump(mode="json")))
    )
    keys[("critic", DEFAULT_CRITIC_PROMPT_VERSION, "default")] = CannedResponse((_valid(_APPROVE),))
    return FakeLLMClient(keys)


def main() -> None:
    """Regenerate the committed smoke manifest (review the diff — it is a new
    dataset version for every stored run that follows)."""
    SMOKE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_MANIFEST_PATH.write_bytes(manifest_bytes(smoke_manifest()))
    print(f"wrote {SMOKE_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
