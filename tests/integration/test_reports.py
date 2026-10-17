"""Screening report: server-decided state, on-screen document, PDF/Word export,
in-progress flag, and the vision/LLM status probe. Offline: fake LLM + fake
vision + local Postgres (skips if down)."""

from __future__ import annotations

import io

import pytest
from docx import Document
from graph_kit import IMAGE, NOTE_TEXT, REJECT, clean_draft, expected_ctx, fake_llm
from tests.api_kit import PIN, asgi_client, bearer, make_app_deps

from consilium.api.auth import ApiRole
from consilium.api.main import create_app
from consilium.core.settings import Settings
from consilium.jobs.worker import Worker
from consilium.persistence.object_store import MemoryObjectStore
from consilium.persistence.repositories import Repositories
from consilium.vision.client_fake import FakeVisionClient

SUB = ("sub-r", ApiRole.SUBMITTER)
CLIN = ("clin-r", ApiRole.CLINICIAN)


async def _stack(settings: Settings, repos: Repositories, *, critic=None, vision=None):  # type: ignore[no-untyped-def]
    draft = clean_draft(await expected_ctx())
    llm = fake_llm(draft, critic=critic) if critic else fake_llm(draft)
    deps = make_app_deps(
        settings,
        repos=repos,
        llm=llm,
        objects=MemoryObjectStore(),
        **({"vision": vision} if vision else {}),
    )
    return asgi_client(create_app(deps)), deps


async def _submit(client, settings) -> str:  # type: ignore[no-untyped-def]
    r = await client.post(
        "/v1/cases",
        headers=bearer(settings, *SUB),
        files={"image": ("fundus.png", IMAGE, "image/png")},
        data={"note": NOTE_TEXT, "laterality": "OD"},
    )
    assert r.status_code == 202, r.text
    case_id: str = r.json()["case_id"]
    return case_id


async def test_in_progress_then_draft_document(settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = await _stack(settings, repos)
    clin = bearer(settings, *CLIN)
    async with client:
        case_id = await _submit(client, settings)
        view = (await client.get(f"/v1/cases/{case_id}", headers=clin)).json()
        assert view["in_progress"] is True
        early = await client.get(f"/v1/cases/{case_id}/report-document", headers=clin)
        assert early.status_code == 409 and early.json()["reason"] == "REPORT_IN_PROGRESS"

        await Worker(deps.runner, repos).run_once()
        assert (await client.get(f"/v1/cases/{case_id}", headers=clin)).json()[
            "in_progress"
        ] is False

        doc = (await client.get(f"/v1/cases/{case_id}/report-document", headers=clin)).json()
        assert doc["state"] == "draft"
        assert doc["state_label"].startswith("DRAFT")
        assert doc["approved_by"] is None
        assert doc["eye"] == "Right eye (OD)"
        labels = [r["label"] for r in doc["image_findings"]]
        assert "Vertical cup-to-disc ratio" in labels and "Screening tier" in labels
        assert doc["findings"] and all(f["evidence"] for f in doc["findings"])
        assert any(n["label"] for n in doc["note_facts"])
        assert "not a diagnosis" in doc["footer"]
        assert NOTE_TEXT not in str(doc)  # raw note text never in the report
        sub = await client.get(
            f"/v1/cases/{case_id}/report-document", headers=bearer(settings, *SUB)
        )
        assert sub.status_code == 403


async def test_unverified_document_lists_open_issues(settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = await _stack(settings, repos, critic=(REJECT,))
    clin = bearer(settings, *CLIN)
    async with client:
        case_id = await _submit(client, settings)
        await Worker(deps.runner, repos).run_once()
        doc = (await client.get(f"/v1/cases/{case_id}/report-document", headers=clin)).json()
    assert doc["state"] == "unverified"
    flagged = {f["claim_id"]: f["flagged"] for f in doc["findings"]}
    assert flagged["c2"] and "calibrated probability" in flagged["c2"][0]


async def test_approved_document_carries_attribution(settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = await _stack(settings, repos)
    clin = bearer(settings, *CLIN)
    async with client:
        case_id = await _submit(client, settings)
        await Worker(deps.runner, repos).run_once()
        rev = (await client.get(f"/v1/cases/{case_id}/draft", headers=clin)).json()["revision"]
        r = await client.post(
            f"/v1/cases/{case_id}/review",
            headers=clin,
            json={"action": "APPROVE", "base_revision": rev},
        )
        assert r.status_code == 200, r.text
        doc = (await client.get(f"/v1/cases/{case_id}/report-document", headers=clin)).json()
    assert doc["state"] == "approved"
    assert doc["approved_by"] == "clin-r" and doc["version"] == 1


@pytest.mark.parametrize("fmt", ["pdf", "docx"])
async def test_export_files(settings, repos, fmt: str) -> None:  # type: ignore[no-untyped-def]
    client, deps = await _stack(settings, repos)
    clin = bearer(settings, *CLIN)
    async with client:
        case_id = await _submit(client, settings)
        await Worker(deps.runner, repos).run_once()
        r = await client.get(f"/v1/cases/{case_id}/export", headers=clin, params={"format": fmt})
    assert r.status_code == 200
    disposition = r.headers["content-disposition"]
    assert f"consilium-report-{case_id[:8]}-draft.{fmt}" in disposition
    if fmt == "pdf":
        assert r.headers["content-type"] == "application/pdf"
        assert r.content.startswith(b"%PDF") and len(r.content) > 5000  # includes the overlay
    else:
        text = "\n".join(p.text for p in Document(io.BytesIO(r.content)).paragraphs)
        assert "DRAFT — awaiting clinician review" in text
        assert "Impression" in text and "Limitations" in text


async def test_failed_case_has_no_report(settings, repos) -> None:  # type: ignore[no-untyped-def]
    vision = FakeVisionClient(revision=PIN, force_revision="sha256:someone-else")
    client, deps = await _stack(settings, repos, vision=vision)
    clin = bearer(settings, *CLIN)
    async with client:
        case_id = await _submit(client, settings)
        await Worker(deps.runner, repos).run_once()
        view = (await client.get(f"/v1/cases/{case_id}", headers=clin)).json()
        assert view["status"] == "FAILED" and view["in_progress"] is False
        r = await client.get(f"/v1/cases/{case_id}/export", headers=clin)
    assert r.status_code == 409 and r.json()["reason"] == "REPORT_NOT_AVAILABLE"


async def test_system_status_reports_vision_and_llm(settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, _ = await _stack(settings, repos)
    async with client:
        ok = (await client.get("/v1/meta/status", headers=bearer(settings, *SUB))).json()
    assert ok == {
        "vision": {
            "mode": "fake",
            "reachable": True,
            "revision_ok": True,
            "device": "fake",
            "detail": None,
        },
        "llm_configured": True,
    }
    stale, _ = await _stack(
        settings, repos, vision=FakeVisionClient(revision=PIN, force_revision="other")
    )
    async with stale:
        body = (await stale.get("/v1/meta/status", headers=bearer(settings, *SUB))).json()
    assert body["vision"]["reachable"] is True and body["vision"]["revision_ok"] is False
