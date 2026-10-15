"""Phase 10.3 — deterministic output checks (guardrails/output_checks.py).

Built on the known-good fault-injection corpus draft (verifier-clean by
construction), mutated one policy breach at a time. Obfuscation sweeps are
in tests/adversarial/test_rails_output.py (10.4).
"""

from __future__ import annotations

import pytest

from consilium.core.errors import PolicyViolation
from consilium.eval import fault_injection as fi
from consilium.guardrails import output_checks
from consilium.guardrails.output_checks import check_draft, draft_texts, load_output_rail
from consilium.intake.deidentify import load_phi_patterns
from consilium.schemas.agents import DirectorDraft
from consilium.verify import lexicon

CFG = load_output_rail()
PHI = load_phi_patterns()
ENTRY = fi.known_good_corpus()[0]


def _check(draft: DirectorDraft) -> None:
    check_draft(draft, ENTRY.global_ctx, CFG, PHI)


def _reason(draft: DirectorDraft) -> str:
    with pytest.raises(PolicyViolation) as exc:
        _check(draft)
    assert exc.value.case_status == "NEEDS_ATTENTION"
    return exc.value.reason


def test_known_good_draft_passes() -> None:
    _check(ENTRY.draft)


def test_lexicon_is_imported_not_redeclared() -> None:
    assert output_checks.scan_text is lexicon.scan_text
    names = set(vars(output_checks))
    assert not names & {"DOSE_PATTERN", "STAGE_SEVERITY_PATTERN", "DEFINITIVE_DIAGNOSIS_PHRASES"}


@pytest.mark.parametrize(
    ("impression", "reason"),
    [
        ("The patient has glaucoma.", "DRAFT_DEFINITIVE_DIAGNOSIS"),
        ("Consider timolol 0.5% twice daily.", "DRAFT_DOSE_OR_REGIMEN"),
        ("Findings suggest advanced glaucoma.", "DRAFT_STAGE_SEVERITY"),
    ],
)
def test_lexicon_breach_in_impression(impression: str, reason: str) -> None:
    assert _reason(ENTRY.draft.model_copy(update={"impression": impression})) == reason


def test_breach_in_limitations_is_scanned_too() -> None:
    lims = (*ENTRY.draft.limitations, "Use 20 mg acetazolamide.")
    assert _reason(ENTRY.draft.model_copy(update={"limitations": lims})) == "DRAFT_DOSE_OR_REGIMEN"


def test_phi_echo_blocked_with_rule_name_only() -> None:
    lims = (*ENTRY.draft.limitations, "Contact 555-123-4567 for follow-up.")
    with pytest.raises(PolicyViolation) as exc:
        _check(ENTRY.draft.model_copy(update={"limitations": lims}))
    assert exc.value.reason == "DRAFT_PHI_ECHO"
    assert exc.value.detail == "output_rail.phi:phone_number"


def test_missing_disclaimer_blocked() -> None:
    lims = tuple(t for t in ENTRY.draft.limitations if "screening" not in t.lower())
    draft = ENTRY.draft.model_copy(update={"limitations": lims})
    assert _reason(draft) == "DRAFT_MISSING_DISCLAIMER"


def test_draft_texts_covers_every_visible_field() -> None:
    texts = draft_texts(ENTRY.draft)
    assert ENTRY.draft.impression in texts
    assert all(c.text in texts for c in ENTRY.draft.claims)
    assert all(lim in texts for lim in ENTRY.draft.limitations)


def test_check_never_mutates_draft() -> None:
    before = ENTRY.draft.model_dump()
    _check(ENTRY.draft)
    assert ENTRY.draft.model_dump() == before
