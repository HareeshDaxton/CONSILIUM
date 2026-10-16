"""Section 11.3 — field-level diff round-trip + review command validation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from consilium.core.errors import RequestInvalid
from consilium.review.diff import Change, apply_diff, diff_drafts
from consilium.review.reason_codes import ReasonCode
from consilium.review.service import ReviewAction, ReviewCommand, _check_command

BASE: dict[str, Any] = {
    "impression": "Findings warrant evaluation for glaucoma.",
    "claims": [
        {
            "id": "c1",
            "kind": "numeric",
            "text": "CDR as measured.",
            "evidence_refs": ["cad.cdr.vertical"],
        },
        {
            "id": "c2",
            "kind": "categorical",
            "text": "Tier as computed.",
            "evidence_refs": ["cad.p_glaucoma.tier"],
        },
    ],
    "disagreements": [],
    "recommendations": [],
    "limitations": ["Screening support only; not a diagnosis.", "Caveat carried: blurred."],
    "revision": 0,
}


def _edited() -> dict[str, Any]:
    e = copy.deepcopy(BASE)
    e["impression"] = "Findings raise suspicion of glaucoma; clinician evaluation advised."
    e["claims"][1]["text"] = "Tier as computed by the model."
    e["claims"].append(
        {
            "id": "c3",
            "kind": "categorical",
            "text": "Laterality unknown.",
            "evidence_refs": ["cad.laterality"],
        }
    )
    e["limitations"].pop()
    return e


def test_diff_paths_and_ops() -> None:
    changes = {c.path: c for c in diff_drafts(BASE, _edited())}
    assert changes["impression"].op == "changed"
    assert changes["claims[id=c2].text"].after == "Tier as computed by the model."
    assert changes["claims[id=c3].text"].op == "added"
    assert changes["limitations[1]"].op == "removed"
    assert "claims[id=c1].text" not in changes


def test_round_trip_and_json_serialisation() -> None:
    edited = _edited()
    changes = [Change.from_json(c.to_json()) for c in diff_drafts(BASE, edited)]
    assert apply_diff(BASE, changes) == edited


def test_claim_paths_stable_under_reorder() -> None:
    reordered = copy.deepcopy(BASE)
    reordered["claims"].reverse()
    assert diff_drafts(BASE, reordered) == []


def test_identical_documents_have_no_diff() -> None:
    assert diff_drafts(BASE, copy.deepcopy(BASE)) == []


@pytest.mark.parametrize(
    ("cmd", "reason"),
    [
        (ReviewCommand(action=ReviewAction.REJECT, base_revision=0), "REASON_CODES_REQUIRED"),
        (
            ReviewCommand(
                action=ReviewAction.REJECT, base_revision=0, reason_codes=(ReasonCode.OTHER,)
            ),
            "OTHER_REQUIRES_TEXT",
        ),
        (
            ReviewCommand(
                action=ReviewAction.REJECT,
                base_revision=0,
                reason_codes=(ReasonCode.OTHER,),
                comment="   ",
            ),
            "OTHER_REQUIRES_TEXT",
        ),
        (
            ReviewCommand(
                action=ReviewAction.APPROVE_WITH_EDITS,
                base_revision=0,
                reason_codes=(ReasonCode.TONE_OR_WORDING,),
            ),
            "EDITS_REQUIRED",
        ),
        (
            ReviewCommand(
                action=ReviewAction.APPROVE, base_revision=0, reason_codes=(ReasonCode.OMISSION,)
            ),
            "REASON_CODES_ONLY_FOR_EDITS_OR_REJECT",
        ),
    ],
)
def test_command_validation(cmd: ReviewCommand, reason: str) -> None:
    with pytest.raises(RequestInvalid) as exc:
        _check_command(cmd)
    assert exc.value.reason == reason


def test_valid_commands_pass() -> None:
    _check_command(ReviewCommand(action=ReviewAction.APPROVE, base_revision=0))
    _check_command(
        ReviewCommand(
            action=ReviewAction.REJECT,
            base_revision=0,
            reason_codes=(ReasonCode.OTHER,),
            comment="image not usable",
        )
    )


def test_unknown_reason_code_rejected_by_schema() -> None:
    with pytest.raises(ValueError):
        ReviewCommand.model_validate(
            {"action": "REJECT", "base_revision": 0, "reason_codes": ["MADE_UP"]}
        )
