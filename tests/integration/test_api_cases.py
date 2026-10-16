"""Section 11.2 exit: idempotent submit, worker pickup → PENDING_REVIEW, draft
view, overlay render + cache, input rejection, PHI-safe filename handling.
Offline: fake LLM + fake vision + real local Postgres (skips if down)."""

from __future__ import annotations

import io
import uuid

import numpy as np
import pytest
from graph_kit import IMAGE, NOTE_TEXT, clean_draft, count_rows, expected_ctx, fake_llm
from PIL import Image
from sqlalchemy import select
from tests.api_kit import asgi_client, bearer, make_app_deps

from consilium.api.auth import ApiRole
from consilium.api.main import create_app
from consilium.core.settings import Settings
from consilium.jobs.worker import Worker
from consilium.persistence.models import AuditLogRow, Case, JobRow
from consilium.persistence.object_store import MemoryObjectStore
from consilium.persistence.repositories import Repositories
from consilium.vision.overlay import CUP_RGB, DISC_RGB

PHI_FILENAME = "Jane_Roe_MRN12345678_left.png"


@pytest.fixture
async def env(settings: Settings, repos: Repositories):  # type: ignore[no-untyped-def]
    objects = MemoryObjectStore()
    llm = fake_llm(clean_draft(await expected_ctx()))
    deps = make_app_deps(settings, repos=repos, llm=llm, objects=objects)
    app = create_app(deps)
    async with asgi_client(app) as client:
        yield client, deps, objects


async def _submit(client, settings, *, key=None, data=IMAGE, note=NOTE_TEXT):  # type: ignore[no-untyped-def]
    headers = bearer(settings, "submitter-1", ApiRole.SUBMITTER)
    if key:
        headers["Idempotency-Key"] = key
    return await client.post(
        "/v1/cases",
        headers=headers,
        files={"image": (PHI_FILENAME, data, "image/png")},
        data={"note": note, "laterality": "OD"},
    )


async def test_submit_is_idempotent_and_enqueues_once(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = env
    first = await _submit(client, settings, key="idem-1")
    second = await _submit(client, settings, key="idem-1")
    assert first.status_code == second.status_code == 202
    case_id = first.json()["case_id"]
    assert second.json()["case_id"] == case_id
    assert first.json()["status"] == "INPUT_RAIL_PASSED"
    assert await count_rows(repos, JobRow, case_id) == 1
    async with repos._session_factory() as s:
        assert len((await s.execute(select(Case))).scalars().all()) == 1


async def test_worker_completes_case_and_draft_view(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps, _ = env
    case_id = (await _submit(client, settings)).json()["case_id"]
    worker = Worker(deps.runner, repos, worker_id="w-test")
    assert await worker.run_once() == case_id
    assert await worker.run_once() is None  # queue drained
    job = await repos.job_get(uuid.UUID(case_id))
    assert job is not None and job.status == "done" and job.attempts == 1

    view = await client.get(
        f"/v1/cases/{case_id}", headers=bearer(settings, "submitter-1", ApiRole.SUBMITTER)
    )
    assert view.json()["status"] == "PENDING_REVIEW"
    assert view.json()["laterality"] == "OD"
    assert view.json()["cad_summary"]["tier"] in ("low", "intermediate", "high")

    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    draft = (await client.get(f"/v1/cases/{case_id}/draft", headers=clin)).json()
    assert draft["unverified"] is False and draft["unverified_banner"] is None
    assert draft["open_issues"] == []
    assert draft["evidence"] and all("label" in c for c in draft["evidence"].values())
    assert draft["overlay_url"] == f"/v1/cases/{case_id}/overlay"
    # submitters cannot read drafts
    sub = bearer(settings, "submitter-1", ApiRole.SUBMITTER)
    assert (await client.get(f"/v1/cases/{case_id}/draft", headers=sub)).status_code == 403


async def test_overlay_render_and_cache(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps, objects = env
    case_id = (await _submit(client, settings)).json()["case_id"]
    await Worker(deps.runner, repos).run_once()
    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    r1 = await client.get(f"/v1/cases/{case_id}/overlay", headers=clin)
    assert r1.status_code == 200 and r1.headers["content-type"] == "image/png"
    pixels = np.array(Image.open(io.BytesIO(r1.content)).convert("RGB")).reshape(-1, 3)
    colours = {tuple(int(v) for v in p) for p in pixels}
    assert DISC_RGB in colours and CUP_RGB in colours
    cached = [u for u in objects.objects if u.startswith("mem://overlays/")]
    assert len(cached) == 1
    r2 = await client.get(f"/v1/cases/{case_id}/overlay", headers=clin)
    assert r2.content == r1.content and r2.headers["etag"] == r1.headers["etag"]
    assert len([u for u in objects.objects if u.startswith("mem://overlays/")]) == 1


async def test_draft_and_overlay_before_pipeline_conflict(env, settings) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = env
    case_id = (await _submit(client, settings)).json()["case_id"]
    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    for path in ("draft", "overlay"):
        r = await client.get(f"/v1/cases/{case_id}/{path}", headers=clin)
        assert r.status_code == 409
        assert r.json()["type"] == "urn:consilium:problem:conflict-status"


async def test_bad_image_rejected_without_job(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = env
    r = await _submit(client, settings, data=b"not an image at all")
    assert r.status_code == 422
    body = r.json()
    assert body["type"] == "urn:consilium:problem:input-rejected"
    assert await count_rows(repos, JobRow, body["case_id"]) == 0


async def test_filename_and_note_never_stored(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps, objects = env
    case_id = (await _submit(client, settings)).json()["case_id"]
    await Worker(deps.runner, repos).run_once()
    async with repos._session_factory() as s:
        case = (await s.execute(select(Case))).scalar_one()
        audit = (await s.execute(select(AuditLogRow))).scalars().all()
    blob = repr(case.state_json) + repr(case.image_uri) + repr([a.detail_json for a in audit])
    assert "Jane" not in blob and "MRN" not in blob and NOTE_TEXT not in blob
    assert all("Jane" not in uri for uri in objects.objects)
    assert case.image_uri is not None and case.image_uri.startswith("mem://images/")
    assert str(case.id) == case_id


async def test_unknown_case_and_auth(env, settings) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = env
    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    assert (await client.get("/v1/cases/not-a-uuid", headers=clin)).status_code == 404
    assert (await client.get(f"/v1/cases/{uuid.uuid4()}", headers=clin)).status_code == 404
    assert (await client.get(f"/v1/cases/{uuid.uuid4()}")).status_code == 401
    # clinicians cannot submit
    r = await client.post("/v1/cases", headers=clin, files={"image": ("x.png", IMAGE, "image/png")})
    assert r.status_code == 403


async def test_worker_infra_failure_retries_then_dead(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = env
    case_id = (await _submit(client, settings)).json()["case_id"]

    class _Boom:
        async def resume(self, case_id: str) -> None:
            raise ConnectionError("db down: Jane Roe")  # content must never be logged

    worker = Worker(_Boom(), repos, worker_id="w-x", max_attempts=2)  # type: ignore[arg-type]
    assert await worker.run_once() == case_id
    job = await repos.job_get(uuid.UUID(case_id))
    assert job is not None and job.status == "queued" and job.attempts == 1
    async with repos._session_factory() as s:  # make the backoff elapse
        await s.execute(JobRow.__table__.update().values(run_at=None))
        await s.commit()
    assert await worker.run_once() == case_id
    job = await repos.job_get(uuid.UUID(case_id))
    assert job is not None and job.status == "dead"
    async with repos._session_factory() as s:
        dead = (
            await s.execute(select(AuditLogRow).where(AuditLogRow.event == "job.dead"))
        ).scalar_one()
    assert dead.detail_json == {"error_type": "ConnectionError", "attempts": 2}
