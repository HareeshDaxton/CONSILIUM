"""Unit tests for verify/deterministic.py (P7.3) + seeded-fault recall (§13.4).

Every check gets positive / negative / boundary coverage. The numeric
boundary cases are mandatory per the implementation plan: an exactly-at-
tolerance claim must PASS and a just-beyond one must FAIL — the verifier is
exact, and tolerances are never loosened to make a draft pass (§26.4).

The drift-check tests recompute ``evaluate_deterministic`` and compare against
the committed JSON fixture so the stored §13.4 numbers cannot silently drift
(I-11). Corpus builders from ``consilium.eval.fault_injection`` are reused
(private helpers imported deliberately) so the tests and the measured recall
report exercise the exact same drafts.
"""

from __future__ import annotations

import json

import pytest

from consilium.context.builder import build
from consilium.context.facts import F
from consilium.context.router import RoutingConfig, load_routing_config
from consilium.eval import fault_injection as fi
from consilium.schemas.agents import Claim, DirectorDraft, Disagreement, SubReport
from consilium.schemas.audit import Issue
from consilium.schemas.context import GlobalContext
from consilium.schemas.enums import (
    ClaimKind,
    Confidence,
    IssueType,
    Laterality,
    QualityFlag,
    Role,
)
from consilium.schemas.note import NoteExtraction
from consilium.schemas.vision import CADResult
from consilium.verify.deterministic import (
    check_coverage,
    check_disagreement_preservation,
    check_language,
    check_laterality_consistency,
    check_numerics,
    check_references,
    check_role_scope,
    check_stray_numbers,
    verify_draft,
)
from consilium.vision.config import Thresholds, load_thresholds

_REPO_ROOT = fi.REPORT_PATH.parents[2]

# cad.cdr.vertical fact value in the corpus; tolerance from thresholds.yaml is 0.02.
CDR_VALUE = fi.CDR_FACT_VALUE
CDR_TOL = 0.02


@pytest.fixture(scope="module")
def thresholds() -> Thresholds:
    return load_thresholds()


@pytest.fixture(scope="module")
def routing() -> RoutingConfig:
    return load_routing_config(_REPO_ROOT / "configs" / "routing.yaml")


@pytest.fixture(scope="module")
def entry() -> fi.CorpusEntry:
    return fi.known_good_corpus()[0]


def _types(issues: tuple[Issue, ...]) -> set[IssueType]:
    return {i.type for i in issues}


def _mutate(draft: DirectorDraft, claim_id: str, **updates: object) -> DirectorDraft:
    return fi._mutate_claim(draft, claim_id, **updates)


def _ctx_with_cad(cad: CADResult) -> GlobalContext:
    note = fi._clean_note()
    return build(
        cad,
        note,
        NoteExtraction(
            note=note,
            extractor_model="gpt-4o-mini-2024-07-18",
            extractor_prompt_version="v1",
            extraction_confidence=Confidence.HIGH,
        ),
        "case-verifier-test",
    )


def _ctx_with_note_prior_dx(prior_dx: str) -> GlobalContext:
    note = fi._clean_note().model_copy(update={"prior_glaucoma_dx": prior_dx})
    return build(
        fi._clean_cad(),
        note,
        NoteExtraction(
            note=note,
            extractor_model="gpt-4o-mini-2024-07-18",
            extractor_prompt_version="v1",
            extraction_confidence=Confidence.HIGH,
        ),
        "case-verifier-test",
    )


class TestCheckReferences:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry) -> None:
        assert check_references(entry.draft, entry.global_ctx) == ()

    def test_unknown_evidence_ref_flagged(self, entry: fi.CorpusEntry) -> None:
        faulted = fi.apply_fault(entry, fi.FaultClass.FABRICATED_NO_REF)
        issues = check_references(faulted.draft, entry.global_ctx)
        assert _types(issues) == {IssueType.BAD_REFERENCE}
        assert any(i.claim_id == "c9" and "cad.cdr.made_up" in i.evidence for i in issues)

    def test_unknown_numeric_ref_flagged(self, entry: fi.CorpusEntry) -> None:
        draft = _mutate(entry.draft, "c1", numeric_ref="cad.cdr.made_up")
        issues = check_references(draft, entry.global_ctx)
        assert _types(issues) == {IssueType.BAD_REFERENCE}
        assert any("numeric ref" in i.detail for i in issues)

    def test_unknown_recommendation_trigger_flagged(self, entry: fi.CorpusEntry) -> None:
        rec = entry.draft.recommendations[0].model_copy(
            update={"triggered_by": ("note.meds.made_up",)}
        )
        draft = entry.draft.model_copy(update={"recommendations": (rec,)})
        issues = check_references(draft, entry.global_ctx)
        assert _types(issues) == {IssueType.BAD_REFERENCE}
        assert any("recommendation" in i.detail for i in issues)


class TestCheckNumerics:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry, thresholds: Thresholds) -> None:
        assert check_numerics(entry.draft, entry.global_ctx, thresholds) == ()

    def test_gross_mismatch_flagged(self, entry: fi.CorpusEntry, thresholds: Thresholds) -> None:
        faulted = fi.apply_fault(entry, fi.FaultClass.NUMERIC_GROSS)
        issues = check_numerics(faulted.draft, entry.global_ctx, thresholds)
        assert _types(issues) == {IssueType.NUMERIC_MISMATCH}
        assert [i.claim_id for i in issues] == ["c1"]

    @pytest.mark.parametrize(
        "value",
        [
            CDR_VALUE,  # exact
            CDR_VALUE + CDR_TOL,  # exactly at tolerance, upper — MUST PASS
            CDR_VALUE - CDR_TOL,  # exactly at tolerance, lower — MUST PASS
            CDR_VALUE + CDR_TOL - 1e-6,  # just inside
        ],
    )
    def test_within_tolerance_passes(
        self, entry: fi.CorpusEntry, thresholds: Thresholds, value: float
    ) -> None:
        draft = _mutate(entry.draft, "c1", numeric_value=value)
        assert check_numerics(draft, entry.global_ctx, thresholds) == ()

    @pytest.mark.parametrize(
        "value",
        [
            CDR_VALUE + CDR_TOL + 1e-6,  # just beyond, upper — MUST FAIL
            CDR_VALUE - CDR_TOL - 1e-6,  # just beyond, lower — MUST FAIL
            CDR_VALUE + 0.03,  # the seeded NUMERIC_SUBTLE fault
        ],
    )
    def test_beyond_tolerance_flagged(
        self, entry: fi.CorpusEntry, thresholds: Thresholds, value: float
    ) -> None:
        draft = _mutate(entry.draft, "c1", numeric_value=value)
        issues = check_numerics(draft, entry.global_ctx, thresholds)
        assert _types(issues) == {IssueType.NUMERIC_MISMATCH}

    def test_subtle_seeded_fault_caught(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        # 0.53 vs 0.5: a subtle corruption a human reviewer could miss.
        faulted = fi.apply_fault(entry, fi.FaultClass.NUMERIC_SUBTLE)
        issues = check_numerics(faulted.draft, entry.global_ctx, thresholds)
        assert _types(issues) == {IssueType.NUMERIC_MISMATCH}

    def test_numeric_ref_to_non_numeric_fact_flagged(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        draft = _mutate(entry.draft, "c1", numeric_ref=F.P_TIER)
        issues = check_numerics(draft, entry.global_ctx, thresholds)
        assert _types(issues) == {IssueType.NUMERIC_MISMATCH}
        assert any("non-numeric fact" in i.detail for i in issues)

    def test_unresolved_numeric_ref_not_double_flagged(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        # Unknown refs are BAD_REFERENCE only; numerics must stay silent.
        draft = _mutate(entry.draft, "c1", numeric_ref="cad.cdr.made_up")
        assert check_numerics(draft, entry.global_ctx, thresholds) == ()


class TestCheckStrayNumbers:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry, thresholds: Thresholds) -> None:
        assert check_stray_numbers(entry.draft, entry.global_ctx, thresholds) == ()

    def test_stray_numeral_in_impression_flagged(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        draft = entry.draft.model_copy(
            update={"impression": entry.draft.impression + " IOP measured 30."}
        )
        issues = check_stray_numbers(draft, entry.global_ctx, thresholds)
        assert _types(issues) == {IssueType.UNSUPPORTED}
        assert any("30" in i.detail for i in issues)

    def test_whitelisted_numerals_pass(self, entry: fi.CorpusEntry, thresholds: Thresholds) -> None:
        draft = entry.draft.model_copy(
            update={
                "impression": (
                    "Seen on 2024-01-15; 58 years old; findings warrant evaluation"
                    " for glaucoma. Review in 6 months."
                )
            }
        )
        assert check_stray_numbers(draft, entry.global_ctx, thresholds) == ()

    @pytest.mark.parametrize(
        "text", ["Comorbid type 2 diabetes is noted.", "History of Type 1 DM.", "type2 diabetes"]
    )
    def test_disease_type_numeral_passes(
        self, entry: fi.CorpusEntry, thresholds: Thresholds, text: str
    ) -> None:
        draft = _mutate(entry.draft, "c1", text=text)
        assert check_stray_numbers(draft, entry.global_ctx, thresholds) == ()

    def test_other_numbers_next_to_type_still_flagged(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        # only "type 1/2 diabetes|DM" is whitelisted; "type 3" or a bare "2" is not
        draft = _mutate(entry.draft, "c1", text="Type 3 pattern seen, grade 2.")
        assert _types(check_stray_numbers(draft, entry.global_ctx, thresholds)) == {
            IssueType.UNSUPPORTED
        }

    def test_numeral_within_tolerance_of_cited_fact_passes(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        # 0.52 in prose is within tolerance of the cited cad.cdr.vertical fact.
        draft = _mutate(entry.draft, "c1", text="Vertical cup-to-disc ratio is 0.52.")
        assert check_stray_numbers(draft, entry.global_ctx, thresholds) == ()

    def test_numeral_attributable_to_numeric_claim_passes(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        # 0.62 is the value of numeric claim c4; the impression may repeat it.
        draft = entry.draft.model_copy(
            update={"impression": "Calibrated probability 0.62 warrants evaluation."}
        )
        assert check_stray_numbers(draft, entry.global_ctx, thresholds) == ()


class TestCheckCoverage:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry) -> None:
        assert check_coverage(entry.draft, entry.global_ctx) == ()

    @pytest.mark.parametrize(
        ("claim_id", "missing_fact"),
        [("c1", F.CDR_V), ("c2", F.P_TIER), ("c3", F.LAT)],
    )
    def test_mandatory_fact_uncited_flagged(
        self, entry: fi.CorpusEntry, claim_id: str, missing_fact: str
    ) -> None:
        draft = entry.draft.model_copy(
            update={"claims": tuple(c for c in entry.draft.claims if c.id != claim_id)}
        )
        issues = check_coverage(draft, entry.global_ctx)
        assert IssueType.OMITTED_FINDING in _types(issues)
        assert any(missing_fact in i.evidence for i in issues)

    def test_deleted_mandatory_seeded_fault_caught(self, entry: fi.CorpusEntry) -> None:
        faulted = fi.apply_fault(entry, fi.FaultClass.DELETED_MANDATORY)
        issues = check_coverage(faulted.draft, entry.global_ctx)
        assert IssueType.OMITTED_FINDING in _types(issues)
        assert any(F.CDR_V in i.evidence for i in issues)

    def test_uncovered_caveat_flagged(self, entry: fi.CorpusEntry) -> None:
        cad = fi._clean_cad().model_copy(
            update={
                "quality": fi._clean_cad().quality.model_copy(
                    update={"gradable": False, "flags": (QualityFlag.UNDEREXPOSED,)}
                )
            }
        )
        ctx = _ctx_with_cad(cad)
        issues = check_coverage(entry.draft, ctx)
        assert IssueType.OMITTED_FINDING in _types(issues)
        assert any(F.caveat("underexposed") in i.evidence for i in issues)

    def test_caveat_covered_by_limitation_passes(self, entry: fi.CorpusEntry) -> None:
        cad = fi._clean_cad().model_copy(
            update={
                "quality": fi._clean_cad().quality.model_copy(
                    update={"gradable": False, "flags": (QualityFlag.UNDEREXPOSED,)}
                )
            }
        )
        ctx = _ctx_with_cad(cad)
        draft = entry.draft.model_copy(
            update={
                "limitations": (
                    *entry.draft.limitations,
                    "Image underexposed; interpret with caution.",
                )
            }
        )
        assert check_coverage(draft, ctx) == ()

    def test_caveat_covered_by_claim_passes(self, entry: fi.CorpusEntry) -> None:
        cad = fi._clean_cad().model_copy(
            update={
                "quality": fi._clean_cad().quality.model_copy(
                    update={"gradable": False, "flags": (QualityFlag.UNDEREXPOSED,)}
                )
            }
        )
        ctx = _ctx_with_cad(cad)
        caveat_claim = Claim(
            id="c8",
            kind=ClaimKind.CATEGORICAL,
            text="Image quality is degraded by underexposure.",
            evidence_refs=(F.caveat("underexposed"), F.Q_FLAGS),
            source_roles=(Role.OPHTHALMOLOGIST,),
        )
        draft = fi._add_claim(entry.draft, caveat_claim)
        assert check_coverage(draft, ctx) == ()


class TestCheckLanguage:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry) -> None:
        assert check_language(entry.draft, entry.global_ctx) == ()

    def test_dose_insertion_seeded_fault_caught(self, entry: fi.CorpusEntry) -> None:
        faulted = fi.apply_fault(entry, fi.FaultClass.DOSE_INSERTION)
        issues = check_language(faulted.draft, entry.global_ctx)
        assert IssueType.UNSAFE_LANGUAGE in _types(issues)
        assert any("dose_or_regimen" in i.detail for i in issues)

    def test_definitive_diagnosis_flagged(self, entry: fi.CorpusEntry) -> None:
        draft = entry.draft.model_copy(update={"impression": "The patient has glaucoma."})
        issues = check_language(draft, entry.global_ctx)
        assert IssueType.UNSAFE_LANGUAGE in _types(issues)
        assert any("definitive_diagnosis" in i.detail for i in issues)

    def test_stage_severity_without_note_support_flagged(self, entry: fi.CorpusEntry) -> None:
        draft = _mutate(entry.draft, "c2", text="Findings suggest moderate-stage glaucoma.")
        issues = check_language(draft, entry.global_ctx)
        assert IssueType.UNSAFE_LANGUAGE in _types(issues)
        assert any("stage_severity" in i.detail for i in issues)

    def test_stage_severity_with_note_support_passes(self, entry: fi.CorpusEntry) -> None:
        # I-12: stage/severity wording is allowed when the clinical note itself
        # supplies it.
        ctx = _ctx_with_note_prior_dx("moderate-stage glaucoma")
        draft = _mutate(entry.draft, "c2", text="Note reports moderate-stage glaucoma.")
        assert check_language(draft, ctx) == ()


class TestCheckDisagreementPreservation:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry, thresholds: Thresholds) -> None:
        assert check_disagreement_preservation(entry.draft, entry.sub_reports, thresholds) == ()

    def test_suppressed_disagreement_seeded_fault_caught(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        faulted = fi.apply_fault(entry, fi.FaultClass.SUPPRESSED_DISAGREEMENT)
        issues = check_disagreement_preservation(faulted.draft, entry.sub_reports, thresholds)
        assert _types(issues) == {IssueType.DISAGREEMENT_SUPPRESSED}
        assert any("escalation urgency" in i.detail for i in issues)

    def test_no_polar_stances_no_issue(self, entry: fi.CorpusEntry, thresholds: Thresholds) -> None:
        # Neutral sub-reports + empty disagreements must not flag.
        neutral = SubReport(
            role=Role.OPHTHALMOLOGIST,
            findings=(
                entry.sub_reports[0]
                .findings[0]
                .model_copy(update={"text": "Measurements recorded for the chart."}),
            ),
            recommendations=(),
        )
        draft = entry.draft.model_copy(update={"disagreements": ()})
        assert check_disagreement_preservation(draft, (neutral,), thresholds) == ()

    def test_unrelated_disagreement_does_not_cover(
        self, entry: fi.CorpusEntry, thresholds: Thresholds
    ) -> None:
        # A Disagreement that never names the polar topic must not count.
        draft = entry.draft.model_copy(
            update={
                "disagreements": (
                    Disagreement(
                        topic="wording tone",
                        positions=(
                            (Role.OPHTHALMOLOGIST, "prefers concise wording"),
                            (Role.OPTOMETRIST, "prefers detailed wording"),
                        ),
                        resolution="left to clinician",
                    ),
                )
            }
        )
        issues = check_disagreement_preservation(draft, entry.sub_reports, thresholds)
        assert _types(issues) == {IssueType.DISAGREEMENT_SUPPRESSED}


class TestCheckLateralityConsistency:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry) -> None:
        assert check_laterality_consistency(entry.draft, entry.global_ctx) == ()

    def test_laterality_flip_seeded_fault_caught(self, entry: fi.CorpusEntry) -> None:
        faulted = fi.apply_fault(entry, fi.FaultClass.LATERALITY_FLIP)
        issues = check_laterality_consistency(faulted.draft, entry.global_ctx)
        assert _types(issues) == {IssueType.UNSUPPORTED}
        assert any(i.claim_id == "c3" for i in issues)

    def test_unknown_laterality_exempt(self, entry: fi.CorpusEntry) -> None:
        cad = fi._clean_cad().model_copy(update={"laterality": Laterality.UNKNOWN})
        ctx = _ctx_with_cad(cad)
        draft = _mutate(entry.draft, "c3", text="Laterality is not recorded for this image.")
        assert check_laterality_consistency(draft, ctx) == ()

    def test_claim_not_citing_laterality_ignored(self, entry: fi.CorpusEntry) -> None:
        # A claim that does not cite cad.laterality may mention anatomy freely.
        draft = _mutate(
            entry.draft, "c1", text="Vertical cup-to-disc ratio is 0.5 in the left eye."
        )
        assert check_laterality_consistency(draft, entry.global_ctx) == ()


class TestCheckRoleScope:
    def test_clean_draft_passes(self, entry: fi.CorpusEntry, routing: RoutingConfig) -> None:
        assert check_role_scope(entry.draft, entry.global_ctx, routing) == ()

    def test_pharmacist_citing_cdr_flagged(
        self, entry: fi.CorpusEntry, routing: RoutingConfig
    ) -> None:
        # routing.yaml denies the pharmacist all pixel-level segmentation facts.
        claim = Claim(
            id="c8",
            kind=ClaimKind.CATEGORICAL,
            text="Cup-to-disc ratio is enlarged.",
            evidence_refs=(F.CDR_V,),
            source_roles=(Role.PHARMACIST,),
        )
        draft = fi._add_claim(entry.draft, claim)
        issues = check_role_scope(draft, entry.global_ctx, routing)
        assert _types(issues) == {IssueType.OUT_OF_SCOPE}
        assert any(i.claim_id == "c8" and F.CDR_V in i.evidence for i in issues)

    def test_pharmacist_citing_allowed_fact_passes(
        self, entry: fi.CorpusEntry, routing: RoutingConfig
    ) -> None:
        claim = Claim(
            id="c8",
            kind=ClaimKind.CATEGORICAL,
            text="Screening tier is relevant to medication-safety review.",
            evidence_refs=(F.P_TIER,),
            source_roles=(Role.PHARMACIST,),
        )
        draft = fi._add_claim(entry.draft, claim)
        assert check_role_scope(draft, entry.global_ctx, routing) == ()

    def test_multi_role_claim_flagged_for_the_out_of_scope_role(
        self, entry: fi.CorpusEntry, routing: RoutingConfig
    ) -> None:
        # Optometrist may cite cad.cdr.vertical; pharmacist may not — a claim
        # sourced from both must flag the pharmacist only.
        claim = Claim(
            id="c8",
            kind=ClaimKind.CATEGORICAL,
            text="Cup-to-disc ratio informs follow-up timing.",
            evidence_refs=(F.CDR_V,),
            source_roles=(Role.OPTOMETRIST, Role.PHARMACIST),
        )
        draft = fi._add_claim(entry.draft, claim)
        issues = check_role_scope(draft, entry.global_ctx, routing)
        assert _types(issues) == {IssueType.OUT_OF_SCOPE}
        assert len(issues) == 1
        assert "pharmacist" in issues[0].detail
        assert "optometrist" not in issues[0].detail


class TestVerifyDraftCleanCorpus:
    def test_clean_corpus_zero_issues(
        self, entry: fi.CorpusEntry, thresholds: Thresholds, routing: RoutingConfig
    ) -> None:
        # False-rejection rate on known-good drafts must be 0 (§13.4).
        for corpus_entry in fi.known_good_corpus():
            issues = verify_draft(
                corpus_entry.draft,
                corpus_entry.global_ctx,
                corpus_entry.sub_reports,
                thresholds,
                routing,
            )
            assert issues == (), f"clean draft {corpus_entry.name} rejected: {issues}"

    def test_fabricated_with_ref_is_expected_deterministic_miss(
        self, entry: fi.CorpusEntry, thresholds: Thresholds, routing: RoutingConfig
    ) -> None:
        # An entailed-looking claim with a valid ref is the LLM critic's job
        # (P8); the deterministic layer records an honest expected miss.
        faulted = fi.apply_fault(entry, fi.FaultClass.FABRICATED_WITH_REF)
        assert (
            verify_draft(faulted.draft, entry.global_ctx, entry.sub_reports, thresholds, routing)
            == ()
        )


class TestFaultInjectionReport:
    """Drift check: the committed §13.4 JSON must match a fresh recomputation."""

    def test_expected_detection_covers_every_fault_class(self) -> None:
        assert set(fi.EXPECTED_DETECTION) == set(fi.FaultClass)

    def test_report_matches_committed_fixture(
        self, thresholds: Thresholds, routing: RoutingConfig
    ) -> None:
        report = fi.evaluate_deterministic(fi.known_good_corpus(), thresholds, routing)
        committed = json.loads(fi.REPORT_PATH.read_text(encoding="utf-8"))
        assert report.to_dict() == committed, (
            "fault-injection report drifted from the committed fixture — "
            "regenerate with `uv run python -m consilium.eval.fault_injection` "
            "and commit the diff deliberately"
        )

    def test_full_recall_on_every_class_the_layer_must_catch(
        self, thresholds: Thresholds, routing: RoutingConfig
    ) -> None:
        report = fi.evaluate_deterministic(fi.known_good_corpus(), thresholds, routing)
        for fault, expected in fi.EXPECTED_DETECTION.items():
            row = report.per_class[fault.value]
            if expected is None:
                # Documented expected miss (LLM critic's job, P8).
                assert row["recall"] == 0.0
            else:
                assert row["recall"] == 1.0, f"{fault.value}: recall {row['recall']}"
                assert row["expected_issue"] == expected.value

    def test_clean_false_rejection_rate_zero(
        self, thresholds: Thresholds, routing: RoutingConfig
    ) -> None:
        report = fi.evaluate_deterministic(fi.known_good_corpus(), thresholds, routing)
        assert report.clean_rejected == 0
        assert report.to_dict()["clean_false_rejection_rate"] == 0.0
