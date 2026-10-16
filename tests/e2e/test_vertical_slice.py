"""Section 11.4 / PHASE 11 exit — the vertical slice, offline:

synthetic image + synthetic note → POST /v1/cases → worker → PENDING_REVIEW →
overlay fetch → clinician review → approved report with a full audit trail;
and a loop-exhausted case surfaces the UNVERIFIED banner data.

Real pieces: FastAPI app + auth + RFC 7807, Postgres (compose), Alembic
schema, job queue + worker, file object store, ``HttpVisionClient`` talking
HTTP to the actual fake-vision server app (scripts/fake_vision_server.py,
in-process ASGI transport — no sockets, no network), revision-pin check.
LLM: canned structured responses (FakeLLMClient) — never a real model (§20).
"""

from __future__ import annotations

import importlib.util
import io
import uuid
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from sqlalchemy import select
from tests.api_kit import PIN, asgi_client, bearer, make_app_deps
from tests.integration.graph_kit import (
    IMAGE,
    NOTE_TEXT,
    REJECT,
    clean_draft,
    expected_ctx,
    fake_llm,
)

from consilium.api.auth import ApiRole
from consilium.api.main import create_app
from consilium.core.settings import Settings
from consilium.jobs.worker import Worker
from consilium.persistence.models import AuditLogRow
from consilium.persistence.object_store import FileObjectStore
from consilium.persistence.repositories import Repositories
from consilium.vision.client_http import HttpVisionClient

_SERVER = Path(__file__).resolve().parents[2] / "scripts" / "fake_vision_server.py"
SUBMITTER, CLINICIAN = "submitter-e2e", "clinician-e2e"

# Node audit events of a clean run, in order (§4 state machine).
PIPELINE_EVENTS = [
    "case.created",
    "node.precheck_image",
    "node.input_rail",
    "node.extract_note",
    "node.call_vision",
    "node.build_context",
    "node.route_context",
    "node.run_specialists",
    "node.draft",
    "node.audit",
    "node.output_rail",
    "node.persist_pending",
]


def _fake_vision_app() -> Any:
    spec = importlib.util.spec_from_file_location("fake_vision_server_e2e", _SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def _http_vision(pinned: str = PIN) -> HttpVisionClient:
    return HttpVisionClient(
        endpoint_url="http://fake-vision",
        token="dev-fake-token",
        pinned_revision=pinned,
        transport=httpx.ASGITransport(app=_fake_vision_app()),
    )


async def _stack(  # type: ignore[no-untyped-def]
    settings: Settings,
    repos: Repositories,
    tmp_path: Path,
    critic: tuple[Any, ...],
    pinned: str = PIN,
):
    vision = _http_vision(pinned)
    deps = make_app_deps(
        settings,
        repos=repos,
        llm=fake_llm(clean_draft(await expected_ctx()), critic=critic),
        vision=vision,
        objects=FileObjectStore(tmp_path / "objects"),
    )
    return asgi_client(create_app(deps)), deps, vision


async def _submit_and_process(client, deps, settings, repos) -> str:  # type: ignore[no-untyped-def]
    r = await client.post(
        "/v1/cases",
        headers={**bearer(settings, SUBMITTER, ApiRole.SUBMITTER), "Idempotency-Key": "e2e-1"},
        files={"image": ("synthetic_fundus.png", IMAGE, "image/png")},
        data={"note": NOTE_TEXT},
    )
    assert r.status_code == 202, r.text
    case_id: str = r.json()["case_id"]
    assert await Worker(deps.runner, repos, worker_id="e2e").run_once() == case_id
    return case_id


async def _events(repos: Repositories, case_id: str) -> list[str]:
    async with repos._session_factory() as s:
        rows = (
            await s.execute(
                select(AuditLogRow)
                .where(AuditLogRow.case_id == uuid.UUID(case_id))
                .order_by(AuditLogRow.id)
            )
        ).scalars()
        return [r.event for r in rows]


async def test_submit_process_review_report(
    settings: Settings, repos: Repositories, tmp_path: Path
) -> None:
    client, deps, vision = await _stack(
        settings, repos, tmp_path, critic=({"approved": True, "issues": [], "round": 1},)
    )
    clin = bearer(settings, CLINICIAN, ApiRole.CLINICIAN)
    async with client:
        case_id = await _submit_and_process(client, deps, settings, repos)

        status = (await client.get(f"/v1/cases/{case_id}", headers=clin)).json()
        assert status["status"] == "PENDING_REVIEW" and status["failure"] is None

        draft = (await client.get(f"/v1/cases/{case_id}/draft", headers=clin)).json()
        assert draft["unverified"] is False and draft["evidence"]

        overlay = await client.get(draft["overlay_url"], headers=clin)
        assert overlay.status_code == 200 and overlay.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(overlay.content)).size == (256, 256)
        assert list((tmp_path / "objects" / "overlays").iterdir())  # cached on disk

        edited = {
            **draft["draft"],
            "impression": "Findings raise suspicion of glaucoma; clinician evaluation advised.",
        }
        rv = await client.post(
            f"/v1/cases/{case_id}/review",
            headers=clin,
            json={
                "action": "APPROVE_WITH_EDITS",
                "base_revision": draft["revision"],
                "edited_draft": edited,
                "reason_codes": ["TONE_OR_WORDING"],
            },
        )
        assert rv.status_code == 200, rv.text

        report = (await client.get(f"/v1/cases/{case_id}/report", headers=clin)).json()
    await vision.aclose()

    assert report["status"] == "APPROVED_WITH_EDITS"
    assert report["approved_by"] == CLINICIAN and report["version"] == 1
    assert report["report"]["impression"] == edited["impression"]
    assert "Screening support only" in report["screening_notice"]

    events = await _events(repos, case_id)
    assert [e for e in events if e in PIPELINE_EVENTS] == PIPELINE_EVENTS
    assert events[-1] == "review.approve_with_edits"
    assert "guardrail.triggered" not in events


async def test_loop_exhausted_case_shows_unverified_banner(
    settings: Settings, repos: Repositories, tmp_path: Path
) -> None:
    client, deps, vision = await _stack(settings, repos, tmp_path, critic=(REJECT,))
    clin = bearer(settings, CLINICIAN, ApiRole.CLINICIAN)
    async with client:
        case_id = await _submit_and_process(client, deps, settings, repos)
        status = (await client.get(f"/v1/cases/{case_id}", headers=clin)).json()
        draft = (await client.get(f"/v1/cases/{case_id}/draft", headers=clin)).json()
        approve = await client.post(
            f"/v1/cases/{case_id}/review",
            headers=clin,
            json={"action": "APPROVE", "base_revision": draft["revision"]},
        )
    await vision.aclose()

    assert status["status"] == "NEEDS_ATTENTION" and status["unverified"] is True
    assert draft["unverified"] is True
    assert draft["unverified_banner"] == "UNVERIFIED — critic did not approve"
    assert draft["open_issues"] and draft["open_issues"][0]["claim_id"] == "c2"
    assert approve.status_code == 409  # never final without a clinician fix (I-4)
    assert (await _events(repos, case_id)).count("node.audit") == settings.critic_max_rounds


async def test_revision_pin_mismatch_fails_closed(
    settings: Settings, repos: Repositories, tmp_path: Path
) -> None:
    # the client pins a different revision than the server reports → fail closed
    client, deps, vision = await _stack(
        settings, repos, tmp_path, critic=(REJECT,), pinned="sha256:rogue/other-999"
    )
    clin = bearer(settings, CLINICIAN, ApiRole.CLINICIAN)
    async with client:
        case_id = await _submit_and_process(client, deps, settings, repos)
        status = (await client.get(f"/v1/cases/{case_id}", headers=clin)).json()
    await vision.aclose()
    assert status["status"] == "FAILED"
    assert status["failure"]["type"].startswith("VISION_")
