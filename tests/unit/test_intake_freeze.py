"""Rollback switch (plan Section 14.4): INTAKE_FROZEN refuses new cases with a
503 problem and never touches storage; reads keep working."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.api_kit import bearer, make_app_deps

from consilium.api.auth import ApiRole
from consilium.api.main import create_app
from consilium.core.settings import Settings


def test_frozen_intake_returns_503_problem(settings: Settings) -> None:
    frozen = settings.model_copy(update={"intake_frozen": True})
    client = TestClient(create_app(make_app_deps(frozen)))
    response = client.post(
        "/v1/cases",
        headers=bearer(frozen, "sub-1", ApiRole.SUBMITTER),
        files={"image": ("x.png", b"\x89PNG", "image/png")},
    )
    assert response.status_code == 503
    assert response.json()["reason"] == "INTAKE_FROZEN"
    assert client.get("/healthz").status_code == 200
