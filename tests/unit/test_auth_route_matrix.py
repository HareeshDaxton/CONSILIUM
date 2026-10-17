"""Security review (plan Section 14.2): no /v1 route is reachable anonymously
except the two sign-in endpoints. Guards against a future route forgetting
``require_auth`` — it fails here instead of shipping open."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient
from tests.api_kit import make_app_deps

from consilium.api.main import create_app
from consilium.core.settings import Settings

PUBLIC = {"/v1/auth/config", "/v1/auth/google"}  # sign-in bootstrap only


def test_every_v1_route_requires_authentication(settings: Settings) -> None:
    app = create_app(make_app_deps(settings))
    client = TestClient(app)
    checked = 0
    # Enumerate from the generated schema (included routers are nested objects).
    for template, ops in app.openapi()["paths"].items():
        if not template.startswith("/v1") or template in PUBLIC:
            continue
        path = re.sub(r"\{[^}]+\}", "00000000-0000-0000-0000-000000000000", template)
        for method in ops:
            response = client.request(method.upper(), path)
            assert response.status_code == 401, (method, template, response.status_code)
            checked += 1
    assert checked >= 8
