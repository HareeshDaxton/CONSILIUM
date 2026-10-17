"""Phase 12 — the review UI's test fixtures are REAL API shapes.

Every JSON file under ``frontend/src/test/fixtures/`` must validate against
the Pydantic view model the API serves for it. The TypeScript types in
``frontend/src/api/types.ts`` are hand-kept; this test is what makes drift
between the UI's fixtures and the server contract fail CI. Fixtures are
synthetic (generated from the offline graph kit) — no patient data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from consilium.api.schemas_http import (
    CaseListView,
    CaseView,
    DraftView,
    FactView,
    MeView,
    SystemStatusView,
    TimelineEvent,
)
from consilium.reporting.document import ReportDocument
from consilium.review.reason_codes import ReasonCode
from consilium.review.service import ReportView, ReviewResult

FIXTURES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "test" / "fixtures"

# fixture stem -> the response model the API serves for it
MODELS: dict[str, Any] = {
    "me_clinician": MeView,
    "case_list": CaseListView,
    "case_pending": CaseView,
    "case_failed": CaseView,
    "timeline": tuple[TimelineEvent, ...],
    "facts": tuple[FactView, ...],
    "draft_verified": DraftView,
    "draft_unverified": DraftView,
    "reason_codes": list[dict[str, str]],
    "report": ReportView,
    "review_result": ReviewResult,
    "report_document": ReportDocument,
    "report_document_unverified": ReportDocument,
    "system_status": SystemStatusView,
}


def _files() -> list[Path]:
    return sorted(FIXTURES.glob("*.json"))


def test_every_fixture_is_mapped_to_a_model() -> None:
    stems = {p.stem for p in _files()}
    assert stems, "no UI fixtures found"
    assert stems == set(MODELS), f"unmapped={stems - set(MODELS)} missing={set(MODELS) - stems}"


@pytest.mark.parametrize("stem", sorted(MODELS))
def test_fixture_validates_against_api_model(stem: str) -> None:
    data = json.loads((FIXTURES / f"{stem}.json").read_text(encoding="utf-8"))
    TypeAdapter(MODELS[stem]).validate_json(json.dumps(data))


def test_reason_code_fixture_is_the_closed_taxonomy() -> None:
    data = json.loads((FIXTURES / "reason_codes.json").read_text(encoding="utf-8"))
    assert [d["code"] for d in data] == [c.value for c in ReasonCode]
