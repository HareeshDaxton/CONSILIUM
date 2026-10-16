"""Section 11.3 exit: per-action review tests, self-approval refused,
unverified-approval blocked unless the edit set resolves the stored issues,
immutability + linked amendments, diff round-trip from the stored row, report
gating, reason-code analytics on fixture data. Real local Postgres; fakes only."""

from __future__ import annotations

from typing import Any

import pytest
from graph_kit import IMAGE, NOTE_TEXT, REJECT, clean_draft, count_rows, expected_ctx, fake_llm
from sqlalchemy import select
from tests.api_kit import asgi_client, bearer, make_app_deps

from consilium.api.auth import ApiRole
from consilium.api.main import create_app
from consilium.core.settings import Settings
from consilium.jobs.worker import Worker
from consilium.persistence.models import AuditLogRow, ReviewRow
from consilium.persistence.repositories import Repositories
from consilium.review.analytics import reason_code_report
from consilium.review.diff import Change, apply_diff

SUBMITTER = "submitter-1"


async def _client_for(settings: Settings, repos: Repositories, *, critic: tuple[Any, ...]):  # type: ignore[no-untyped-def]
    llm = fake_llm(clean_draft(await expected_ctx()), critic=critic)
    deps = make_app_deps(settings, repos=repos, llm=llm)
    return asgi_client(create_app(deps)), deps


async def _processed_case(client, deps, settings, repos) -> str:  # type: ignore[no-untyped-def]
    r = await client.post(
        "/v1/cases",
        headers=bearer(settings, SUBMITTER, ApiRole.SUBMITTER),
        files={"image": ("scan.png", IMAGE, "image/png")},
        data={"note": NOTE_TEXT},
    )
    case_id: str = r.json()["case_id"]
    await Worker(deps.runner, repos).run_once()
    return case_id


@pytest.fixture
async def ok(settings: Settings, repos: Repositories):  # type: ignore[no-untyped-def]
    client, deps = await _client_for(
        settings, repos, critic=({"approved": True, "issues": [], "round": 1},)
    )
    async with client:
        case_id = await _processed_case(client, deps, settings, repos)
        yield client, case_id


def _clin(settings: Settings, sub: str = "clin-1") -> dict[str, str]:
    return bearer(settings, sub, ApiRole.CLINICIAN)


async def _draft(client, settings, case_id) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    r = await client.get(f"/v1/cases/{case_id}/draft", headers=_clin(settings))
    return dict(r.json())


async def _review(client, settings, case_id, body, sub="clin-1"):  # type: ignore[no-untyped-def]
    return await client.post(f"/v1/cases/{case_id}/review", headers=_clin(settings, sub), json=body)


async def test_approve_then_report_with_attribution(ok, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, case_id = ok
    gated = await client.get(f"/v1/cases/{case_id}/report", headers=_clin(settings))
    assert gated.status_code == 409 and gated.json()["reason"] == "REPORT_NOT_APPROVED"

    r = await _review(client, settings, case_id, {"action": "APPROVE", "base_revision": 0})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "APPROVED" and r.json()["version"] == 1

    rep = (await client.get(f"/v1/cases/{case_id}/report", headers=_clin(settings))).json()
    assert rep["approved_by"] == "clin-1" and rep["version"] == 1
    assert rep["screening_notice"].startswith("Screening support only")
    assert rep["report"]["impression"].startswith("Findings warrant evaluation")
    async with repos._session_factory() as s:
        events = (await s.execute(select(AuditLogRow.event))).scalars().all()
    assert "review.approve" in events
    # immutable: a second plain approval or a rejection is refused
    for body in (
        {"action": "APPROVE", "base_revision": 0},
        {"action": "REJECT", "base_revision": 0, "reason_codes": ["OMISSION"]},
    ):
        again = await _review(client, settings, case_id, body)
        assert again.status_code == 409 and again.json()["reason"] == "REPORT_IMMUTABLE"


async def test_self_approval_refused(ok, settings) -> None:  # type: ignore[no-untyped-def]
    client, case_id = ok
    r = await _review(
        client, settings, case_id, {"action": "APPROVE", "base_revision": 0}, sub=SUBMITTER
    )
    assert r.status_code == 403 and r.json()["reason"] == "SELF_APPROVAL"


async def test_submitter_cannot_review(ok, settings) -> None:  # type: ignore[no-untyped-def]
    client, case_id = ok
    r = await client.post(
        f"/v1/cases/{case_id}/review",
        headers=bearer(settings, "someone", ApiRole.SUBMITTER),
        json={"action": "APPROVE", "base_revision": 0},
    )
    assert r.status_code == 403


async def test_reason_codes_and_stale_revision(ok, settings) -> None:  # type: ignore[no-untyped-def]
    client, case_id = ok
    no_codes = await _review(client, settings, case_id, {"action": "REJECT", "base_revision": 0})
    assert no_codes.status_code == 422 and no_codes.json()["reason"] == "REASON_CODES_REQUIRED"
    stale = await _review(
        client,
        settings,
        case_id,
        {"action": "REJECT", "base_revision": 5, "reason_codes": ["OMISSION"]},
    )
    assert stale.status_code == 409 and stale.json()["reason"] == "STALE_REVISION"
    r = await _review(
        client,
        settings,
        case_id,
        {"action": "REJECT", "base_revision": 0, "reason_codes": ["OTHER"], "comment": "unusable"},
    )
    assert r.status_code == 200 and r.json()["status"] == "REJECTED" and r.json()["version"] is None
    again = await _review(
        client,
        settings,
        case_id,
        {"action": "REJECT", "base_revision": 0, "reason_codes": ["OMISSION"]},
    )
    assert again.status_code == 409 and again.json()["reason"] == "CASE_NOT_REVIEWABLE"


async def test_edits_diff_round_trip_and_amendment(ok, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, case_id = ok
    base = (await _draft(client, settings, case_id))["draft"]
    edited = {**base, "impression": "Findings raise suspicion of glaucoma; evaluation advised."}
    r = await _review(
        client,
        settings,
        case_id,
        {
            "action": "APPROVE_WITH_EDITS",
            "base_revision": 0,
            "edited_draft": edited,
            "reason_codes": ["TONE_OR_WORDING"],
        },
    )
    assert r.status_code == 200, r.text
    first_id = r.json()["review_id"]
    async with repos._session_factory() as s:
        row = (await s.execute(select(ReviewRow))).scalar_one()
    changes = [Change.from_json(c) for c in row.edits_diff_json["changes"]]
    assert [c.path for c in changes] == ["impression"]
    assert apply_diff(base, changes) == row.edited_draft_json  # both sides stored, diff exact

    # amendment: new linked version, previous review untouched (append-only)
    amended = {**edited, "limitations": [*edited["limitations"], "Repeat imaging advised."]}
    r2 = await _review(
        client,
        settings,
        case_id,
        {
            "action": "APPROVE_WITH_EDITS",
            "base_revision": 0,
            "edited_draft": amended,
            "reason_codes": ["OMISSION"],
        },
        sub="clin-2",
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["version"] == 2 and r2.json()["amends_review_id"] == first_id
    rep = (await client.get(f"/v1/cases/{case_id}/report", headers=_clin(settings))).json()
    assert rep["version"] == 2 and rep["approved_by"] == "clin-2"
    assert rep["report"]["limitations"][-1] == "Repeat imaging advised."
    assert await count_rows(repos, ReviewRow, case_id) == 2


async def test_unverified_approval_blocked_until_issues_resolved(settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = await _client_for(settings, repos, critic=(REJECT,))
    async with client:
        case_id = await _processed_case(client, deps, settings, repos)
        view = await _draft(client, settings, case_id)
        assert view["case_status"] == "NEEDS_ATTENTION" and view["unverified"] is True
        assert view["unverified_banner"] == "UNVERIFIED — critic did not approve"
        assert [i["claim_id"] for i in view["open_issues"]] == ["c2"]
        rev = view["revision"]

        plain = await _review(
            client, settings, case_id, {"action": "APPROVE", "base_revision": rev}
        )
        assert plain.status_code == 409 and plain.json()["reason"] == "UNVERIFIED_REQUIRES_EDITS"

        def body(draft: dict[str, Any]) -> dict[str, Any]:
            return {
                "action": "APPROVE_WITH_EDITS",
                "base_revision": rev,
                "edited_draft": draft,
                "reason_codes": ["CLINICAL_INTERPRETATION"],
            }

        base = view["draft"]
        wrong_field = {**base, "impression": "Findings warrant clinician evaluation for glaucoma."}
        r = await _review(client, settings, case_id, body(wrong_field))
        assert r.status_code == 409 and r.json()["reason"] == "UNVERIFIED_ISSUES_OPEN"

        bad_fix = {
            **base,
            "claims": [
                {**c, "text": "Tier is high at 0.93."} if c["id"] == "c2" else c
                for c in base["claims"]
            ],
        }  # edits c2 but introduces a stray number → deterministic verifier still fails
        r = await _review(client, settings, case_id, body(bad_fix))
        assert r.status_code == 409 and r.json()["reason"] == "UNVERIFIED_ISSUES_OPEN"

        fixed = {
            **base,
            "claims": [
                {**c, "text": "Screening tier is as computed by the model."}
                if c["id"] == "c2"
                else c
                for c in base["claims"]
            ],
        }
        r = await _review(client, settings, case_id, body(fixed))
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "APPROVED_WITH_EDITS"


async def test_reason_code_analytics(ok, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, case_id = ok
    await _review(
        client,
        settings,
        case_id,
        {"action": "REJECT", "base_revision": 0, "reason_codes": ["KNOWLEDGE_GAP", "OMISSION"]},
    )
    report = await reason_code_report(repos)
    assert report.reviews_with_codes == 1
    assert report.code_counts["KNOWLEDGE_GAP"] == 1 and report.code_counts["OMISSION"] == 1
    assert report.code_counts["OTHER"] == 0
    assert report.knowledge_share == 1.0


async def test_reason_codes_meta(ok, settings) -> None:  # type: ignore[no-untyped-def]
    client, _ = ok
    codes = (await client.get("/v1/meta/reason-codes", headers=_clin(settings))).json()
    assert {c["code"] for c in codes} >= {"FACTUAL_ERROR_CAD", "KNOWLEDGE_GAP", "OTHER"}
