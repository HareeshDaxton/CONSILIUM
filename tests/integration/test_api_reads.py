"""Sections 12.1/12.3 — read endpoints the review UI needs: case list (queue
filters, submitter scoping), timeline (no audit detail), fact sheet
(clinician only), ``/v1/me``, and the overlay ``layer`` parameter. Offline:
fake LLM + fake vision + local Postgres (skips if down)."""

from __future__ import annotations

import io

import pytest
from graph_kit import IMAGE, NOTE_TEXT, clean_draft, expected_ctx, fake_llm
from PIL import Image
from tests.api_kit import asgi_client, bearer, make_app_deps

from consilium.api.auth import ApiRole
from consilium.api.main import create_app
from consilium.context.facts import F
from consilium.core.settings import Settings
from consilium.jobs.worker import Worker
from consilium.persistence.object_store import MemoryObjectStore
from consilium.persistence.repositories import Repositories


@pytest.fixture
async def env(settings: Settings, repos: Repositories):  # type: ignore[no-untyped-def]
    deps = make_app_deps(
        settings,
        repos=repos,
        llm=fake_llm(clean_draft(await expected_ctx())),
        objects=MemoryObjectStore(),
    )
    async with asgi_client(create_app(deps)) as client:
        yield client, deps


async def _submit(client, settings, sub: str) -> str:  # type: ignore[no-untyped-def]
    r = await client.post(
        "/v1/cases",
        headers=bearer(settings, sub, ApiRole.SUBMITTER),
        files={"image": ("x.png", IMAGE, "image/png")},
        data={"note": NOTE_TEXT},
    )
    assert r.status_code == 202, r.text
    case_id: str = r.json()["case_id"]
    return case_id


async def test_list_newest_first_status_filter_and_submitter_scope(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = env
    a = await _submit(client, settings, "sub-a")
    b = await _submit(client, settings, "sub-b")
    assert await Worker(deps.runner, repos).run_once() == a  # only `a` processed

    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    items = (await client.get("/v1/cases", headers=clin)).json()["items"]
    assert [i["case_id"] for i in items] == [b, a]
    by_id = {i["case_id"]: i for i in items}
    assert by_id[a]["status"] == "PENDING_REVIEW" and by_id[a]["tier"] is not None
    assert by_id[a]["unverified"] is False
    assert by_id[b]["status"] == "INPUT_RAIL_PASSED" and by_id[b]["tier"] is None

    queue = await client.get(
        "/v1/cases",
        headers=clin,
        params=[("status", "PENDING_REVIEW"), ("status", "NEEDS_ATTENTION")],
    )
    assert [i["case_id"] for i in queue.json()["items"]] == [a]

    mine = await client.get("/v1/cases", headers=bearer(settings, "sub-b", ApiRole.SUBMITTER))
    assert [i["case_id"] for i in mine.json()["items"]] == [b]

    limited = await client.get("/v1/cases", headers=clin, params={"limit": 1})
    assert [i["case_id"] for i in limited.json()["items"]] == [b]


async def test_list_rejects_bad_params_and_anonymous(env, settings) -> None:  # type: ignore[no-untyped-def]
    client, _ = env
    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    for params in ({"status": "NOT_A_STATUS"}, {"limit": 0}, {"limit": 201}):
        r = await client.get("/v1/cases", headers=clin, params=params)
        assert r.status_code == 422
        assert r.headers["content-type"] == "application/problem+json"
    assert (await client.get("/v1/cases")).status_code == 401


async def test_timeline_in_order_without_detail(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = env
    case_id = await _submit(client, settings, "sub-a")
    await Worker(deps.runner, repos).run_once()
    r = await client.get(
        f"/v1/cases/{case_id}/timeline", headers=bearer(settings, "sub-a", ApiRole.SUBMITTER)
    )
    assert r.status_code == 200
    events = r.json()
    assert events[0]["event"] == "case.created"
    assert events[-1]["event"] == "node.persist_pending"
    assert all(set(e) == {"event", "actor", "at"} for e in events)
    assert NOTE_TEXT not in r.text
    missing = await client.get(
        "/v1/cases/00000000-0000-0000-0000-000000000000/timeline",
        headers=bearer(settings, "sub-a", ApiRole.SUBMITTER),
    )
    assert missing.status_code == 404


async def test_facts_clinician_only_and_conflict_before_context(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = env
    case_id = await _submit(client, settings, "sub-a")
    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    early = await client.get(f"/v1/cases/{case_id}/facts", headers=clin)
    assert early.status_code == 409 and early.json()["reason"] == "FACTS_NOT_AVAILABLE"

    await Worker(deps.runner, repos).run_once()
    sub = bearer(settings, "sub-a", ApiRole.SUBMITTER)
    assert (await client.get(f"/v1/cases/{case_id}/facts", headers=sub)).status_code == 403
    facts = {
        f["id"]: f for f in (await client.get(f"/v1/cases/{case_id}/facts", headers=clin)).json()
    }
    assert facts[F.CDR_V]["source"] == "cad"
    assert facts[F.AGE]["source"] == "note" and facts[F.AGE]["value"] == 65


async def test_me_echoes_verified_identity(env, settings) -> None:  # type: ignore[no-untyped-def]
    client, _ = env
    r = await client.get(
        "/v1/me", headers=bearer(settings, "clin-9", ApiRole.SUBMITTER, ApiRole.CLINICIAN)
    )
    assert r.json() == {"sub": "clin-9", "roles": ["clinician", "submitter"]}
    assert (await client.get("/v1/me")).status_code == 401


async def test_overlay_layers_are_cached_separately_and_validated(env, settings, repos) -> None:  # type: ignore[no-untyped-def]
    client, deps = env
    case_id = await _submit(client, settings, "sub-a")
    await Worker(deps.runner, repos).run_once()
    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    url = f"/v1/cases/{case_id}/overlay"
    composite = await client.get(url, headers=clin)
    assert composite.status_code == 200
    etags = {composite.headers["etag"]}
    modes = {}
    for layer in ("image", "disc", "cup"):
        r = await client.get(url, headers=clin, params={"layer": layer})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        etags.add(r.headers["etag"])
        modes[layer] = Image.open(io.BytesIO(r.content)).mode
    explicit = await client.get(url, headers=clin, params={"layer": "composite"})
    assert explicit.content == composite.content
    assert len(etags) == 4  # one cache entry per layer
    assert modes == {"image": "RGB", "disc": "RGBA", "cup": "RGBA"}
    objects = deps.objects
    assert len([u for u in objects.objects if u.startswith("mem://overlays/")]) == 4
    bad = await client.get(url, headers=clin, params={"layer": "vessels"})
    assert bad.status_code == 422
