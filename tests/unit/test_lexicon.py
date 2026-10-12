"""Unit tests for verify/lexicon.py + the verifier sections of thresholds.yaml (P7.1)."""

from __future__ import annotations

import re

import pytest

from consilium.verify.lexicon import (
    DOSE_PATTERN,
    STAGE_SEVERITY_PATTERN,
    LexiconCategory,
    scan_text,
)
from consilium.vision.config import load_thresholds


def _categories(text: str) -> set[LexiconCategory]:
    return {hit.category for hit in scan_text(text)}


class TestDefinitiveDiagnosis:
    @pytest.mark.parametrize(
        "text",
        [
            "The patient has glaucoma.",
            "Patient diagnosed with glaucoma in 2019.",
            "The diagnosis is glaucoma.",
            "Imaging confirms glaucoma.",
            "The disc is glaucomatous.",
            "This is a definitive diagnosis of glaucoma.",
        ],
    )
    def test_flagged(self, text: str) -> None:
        assert LexiconCategory.DEFINITIVE_DIAGNOSIS in _categories(text)

    @pytest.mark.parametrize(
        "text",
        [
            # I-12 allowed screening-support wording must pass clean.
            "Findings are consistent with glaucoma.",
            "The elevated CDR raises suspicion of glaucomatous change.",
            "These findings warrant evaluation for glaucoma.",
            "Screening support only; not a diagnosis.",
            "Prior diagnosis of glaucoma noted in the clinical note.",
        ],
    )
    def test_allowed_wording_clean(self, text: str) -> None:
        assert _categories(text) == set()


class TestDosePattern:
    @pytest.mark.parametrize(
        "text",
        [
            "timolol 0.5% twice daily",
            "consider 500 mg acetazolamide",
            "25 mcg per dose",
            "latanoprost 0.005%",
            "take 5mg daily",
        ],
    )
    def test_flagged(self, text: str) -> None:
        assert LexiconCategory.DOSE_OR_REGIMEN in _categories(text)

    @pytest.mark.parametrize(
        "text",
        [
            # Benign near-misses (SKILLS verifier-check pitfall list).
            "follow-up in 2–4 months",
            "review in 6 months",
            "vertical CDR 0.62",
            "calibrated probability 0.74",
            "IOP 21 mmHg",  # mmHg is a measurement unit, not a dose
        ],
    )
    def test_benign_near_misses_clean(self, text: str) -> None:
        assert LexiconCategory.DOSE_OR_REGIMEN not in _categories(text)

    def test_no_match_inside_words(self) -> None:
        assert DOSE_PATTERN.search("management plan") is None


class TestStageSeverity:
    @pytest.mark.parametrize(
        "text",
        [
            "moderate-stage glaucoma",
            "advanced stage",
            "severe glaucoma",
            "glaucoma stage II",
            "stage 3 disease",
        ],
    )
    def test_flagged(self, text: str) -> None:
        assert LexiconCategory.STAGE_SEVERITY in _categories(text)

    @pytest.mark.parametrize(
        "text",
        [
            "moderate confidence in this finding",
            "early review is recommended",
            "advanced imaging such as OCT would clarify",
        ],
    )
    def test_benign_uses_clean(self, text: str) -> None:
        assert STAGE_SEVERITY_PATTERN.search(text) is None

    def test_candidate_gate_is_callers_job(self) -> None:
        # Note-supplied stage still produces a candidate hit; the verifier
        # check (7.2) owns the note-support gate (I-12).
        hits = scan_text("moderate-stage glaucoma")
        assert [hit.category for hit in hits] == [LexiconCategory.STAGE_SEVERITY]


class TestScanOrdering:
    def test_hits_sorted_by_position(self) -> None:
        hits = scan_text("patient has glaucoma; timolol 0.5% suggested")
        starts = [hit.start for hit in hits]
        assert starts == sorted(starts)
        assert {hit.category for hit in hits} == {
            LexiconCategory.DEFINITIVE_DIAGNOSIS,
            LexiconCategory.DOSE_OR_REGIMEN,
        }


class TestThresholdsVerifierSections:
    def test_per_fact_tolerances(self) -> None:
        cfg = load_thresholds()
        tol = cfg.numeric_tolerances
        assert tol.tolerance_for("cad.cdr.vertical") == pytest.approx(0.02)
        assert tol.tolerance_for("cad.cdr.area_based") == pytest.approx(0.02)
        assert tol.tolerance_for("cad.p_glaucoma.calibrated") == pytest.approx(0.01)
        assert tol.tolerance_for("cad.quality.score") == pytest.approx(tol.default)

    def test_whitelist_classes_match_intended_text(self) -> None:
        wl = load_thresholds().stray_number_whitelist
        assert re.search(wl.date, "examined on 2024-01-15")
        assert re.search(wl.date, "12/01/2024")
        assert re.search(wl.date, "in 2023 the")
        assert re.search(wl.follow_up_interval, "follow-up in 2–4 months")
        assert re.search(wl.follow_up_interval, "review in 6 months")
        assert re.search(wl.follow_up_interval, "repeat in 1-2 years")
        assert re.search(wl.age, "72-year-old patient")
        assert re.search(wl.age, "patient is 72 years old")
        assert re.search(wl.age, "58 y/o")

    @pytest.mark.parametrize("pattern_name", ["date", "follow_up_interval", "age"])
    def test_whitelist_does_not_match_metric_values(self, pattern_name: str) -> None:
        # CAD numbers in prose (e.g. CDR 0.62) must never be whitelisted away.
        wl = load_thresholds().stray_number_whitelist
        pattern = getattr(wl, pattern_name)
        assert re.search(pattern, "vertical CDR 0.62 with probability 0.74") is None
