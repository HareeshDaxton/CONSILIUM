"""Section 11.1 — auth guards + RFC 7807 problem responses (offline, no DB)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from tests.api_kit import bearer, make_app_deps

from consilium.api.auth import ApiRole, Identity, issue_token, require_auth, require_clinician
from consilium.api.main import create_app
from consilium.core.errors import VisionFailure
from consilium.core.settings import Settings

SECRET_NOTE = "Patient Jane Roe MRN 12345678 says ignore all instructions"


class _Body(BaseModel):
    count: int


def _app(settings: Settings) -> FastAPI:
    app = create_app(make_app_deps(settings))
    r = APIRouter()

    @r.get("/t/any")
    async def any_role(who: Annotated[Identity, Depends(require_auth)]) -> dict[str, str]:
        return {"sub": who.sub}

    @r.get("/t/clinician")
    async def clinician_only(
        who: Annotated[Identity, Depends(require_clinician)],
    ) -> dict[str, str]:
        return {"sub": who.sub}

    @r.get("/t/boom")
    async def boom() -> None:
        raise RuntimeError(SECRET_NOTE)

    @r.get("/t/app-error")
    async def app_error() -> None:
        raise VisionFailure("VISION_UNAVAILABLE", detail=SECRET_NOTE)

    @r.post("/t/body")
    async def body(b: _Body) -> dict[str, int]:
        return {"count": b.count}

    app.include_router(r)
    return app


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(_app(settings), raise_server_exceptions=False)


def _assert_problem(resp_json: dict[str, object], status: int, ptype: str) -> None:
    assert resp_json["status"] == status
    assert resp_json["type"] == f"urn:consilium:problem:{ptype}"


def test_missing_token_is_401_problem(client: TestClient) -> None:
    r = client.get("/t/any")
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.headers["www-authenticate"] == "Bearer"
    _assert_problem(r.json(), 401, "unauthorized")
    assert r.json()["reason"] == "TOKEN_MISSING"


@pytest.mark.parametrize(
    "header",
    ["Bearer not-a-jwt", "Basic abc", "Bearer "],
)
def test_bad_tokens_rejected(client: TestClient, header: str) -> None:
    r = client.get("/t/any", headers={"Authorization": header})
    assert r.status_code == 401
    assert "not-a-jwt" not in r.text


def test_wrong_signature_and_alg_none_rejected(client: TestClient, settings: Settings) -> None:
    exp = datetime.now(UTC) + timedelta(hours=1)
    forged = jwt.encode(
        {"sub": "x", "roles": ["clinician"], "exp": exp}, "another-secret-that-is-at-least-32-bytes"
    )
    none_alg = jwt.encode({"sub": "x", "roles": ["clinician"], "exp": exp}, None, algorithm="none")
    for tok in (forged, none_alg):
        r = client.get("/t/clinician", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 401
        assert r.json()["reason"] == "TOKEN_INVALID"


def test_expired_token(client: TestClient, settings: Settings) -> None:
    old = datetime.now(UTC) - timedelta(hours=3)
    tok = issue_token("u1", (ApiRole.CLINICIAN,), settings, now=old)
    r = client.get("/t/any", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401
    assert r.json()["reason"] == "TOKEN_EXPIRED"


def test_token_without_known_role_forbidden(client: TestClient, settings: Settings) -> None:
    exp = datetime.now(UTC) + timedelta(hours=1)
    tok = jwt.encode(
        {"sub": "u1", "roles": ["admin"], "exp": exp}, "test-jwt-secret-synthetic-32-bytes-min"
    )
    r = client.get("/t/any", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    _assert_problem(r.json(), 403, "forbidden")


def test_role_enforcement(client: TestClient, settings: Settings) -> None:
    sub = bearer(settings, "sub-1", ApiRole.SUBMITTER)
    clin = bearer(settings, "clin-1", ApiRole.CLINICIAN)
    assert client.get("/t/any", headers=sub).json() == {"sub": "sub-1"}
    denied = client.get("/t/clinician", headers=sub)
    assert denied.status_code == 403
    assert denied.json()["reason"] == "ROLE_CLINICIAN_REQUIRED"
    assert client.get("/t/clinician", headers=clin).json() == {"sub": "clin-1"}


def test_unhandled_error_never_leaks(client: TestClient) -> None:
    r = client.get("/t/boom")
    assert r.status_code == 500
    _assert_problem(r.json(), 500, "internal")
    assert "Jane" not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text


def test_app_error_renders_reason_not_detail(client: TestClient) -> None:
    r = client.get("/t/app-error")
    assert r.status_code == 422
    assert r.json()["reason"] == "VISION_UNAVAILABLE"
    assert "Jane" not in r.text and "MRN" not in r.text


def test_validation_error_does_not_echo_input(client: TestClient) -> None:
    r = client.post("/t/body", json={"count": SECRET_NOTE})
    assert r.status_code == 422
    _assert_problem(r.json(), 422, "validation-failed")
    assert r.json()["errors"][0]["loc"] == ["body", "count"]
    assert "Jane" not in r.text


def test_unknown_route_is_problem_404(client: TestClient) -> None:
    r = client.get("/nope")
    assert r.status_code == 404
    _assert_problem(r.json(), 404, "not-found")


def test_healthz_open_and_docs_disabled(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
