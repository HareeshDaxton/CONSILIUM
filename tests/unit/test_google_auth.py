"""Google sign-in (UI redesign): ID-token verification, allowlist role mapping,
and the /v1/auth endpoints. Offline: tokens are signed with a locally
generated RSA key standing in for Google's (no network, AGENTS.md §20)."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from tests.api_kit import make_app_deps

from consilium.api.auth import ApiRole, decode_token
from consilium.api.google_auth import GoogleIdTokenVerifier, roles_for, session_subject
from consilium.api.main import create_app
from consilium.core.errors import Forbidden, Unauthorized
from consilium.core.settings import Settings

CLIENT_ID = "1234567890-synthetic.apps.googleusercontent.com"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _claims(**over: Any) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": "https://accounts.google.com",
        "aud": CLIENT_ID,
        "sub": "109876543210",
        "email": "Clinician.One@Example.org",
        "email_verified": True,
        "name": "Synthetic Clinician",
        "picture": "https://example.org/a.png",
        "iat": now,
        "exp": now + 600,
    }
    claims.update(over)
    return {k: v for k, v in claims.items() if v is not None}


def _token(key: Any = KEY, alg: str = "RS256", **over: Any) -> str:
    return jwt.encode(_claims(**over), key, algorithm=alg, headers={"kid": "k1"})


def _verifier() -> GoogleIdTokenVerifier:
    return GoogleIdTokenVerifier(CLIENT_ID, lambda _t: KEY.public_key())


# ------------------------------------------------------------ verification


def test_valid_token_yields_normalised_identity() -> None:
    who = _verifier().verify(_token())
    assert who.sub == "109876543210"
    assert who.email == "clinician.one@example.org"
    assert who.name == "Synthetic Clinician"
    assert session_subject(who) == "google:109876543210"


@pytest.mark.parametrize(
    ("token", "reason"),
    [
        (_token(aud="someone-else.apps.googleusercontent.com"), "GOOGLE_TOKEN_INVALID"),
        (_token(iss="https://evil.example"), "GOOGLE_TOKEN_INVALID"),
        (_token(exp=int(time.time()) - 3600), "GOOGLE_TOKEN_EXPIRED"),
        (_token(key=OTHER_KEY), "GOOGLE_TOKEN_INVALID"),
        (_token(email=None), "GOOGLE_TOKEN_INVALID"),
        (
            _token(key="a-shared-secret-that-is-long-enough-32b", alg="HS256"),
            "GOOGLE_TOKEN_INVALID",
        ),
        ("not-a-jwt", "GOOGLE_TOKEN_INVALID"),
    ],
    ids=["wrong-aud", "wrong-iss", "expired", "wrong-key", "no-email", "hs256", "garbage"],
)
def test_bad_tokens_are_unauthorized(token: str, reason: str) -> None:
    with pytest.raises(Unauthorized) as exc:
        _verifier().verify(token)
    assert exc.value.reason == reason


def test_unverified_email_is_forbidden() -> None:
    with pytest.raises(Forbidden) as exc:
        _verifier().verify(_token(email_verified=False))
    assert exc.value.reason == "GOOGLE_EMAIL_UNVERIFIED"


def test_key_resolution_failure_is_unauthorized() -> None:
    def boom(_t: str) -> Any:
        raise jwt.PyJWKClientError("no matching kid")

    with pytest.raises(Unauthorized):
        GoogleIdTokenVerifier(CLIENT_ID, boom).verify(_token())


# ------------------------------------------------------------ roles


def _settings(base: Settings, **over: Any) -> Settings:
    return base.model_copy(update={"google_client_id": CLIENT_ID, **over})


def test_roles_come_only_from_allowlists(settings: Settings) -> None:
    s = _settings(
        settings,
        google_clinicians=" clinician.one@example.org , both@example.org",
        google_submitters="both@example.org,submitter@example.org",
    )
    assert roles_for("CLINICIAN.ONE@example.org", s) == (ApiRole.CLINICIAN,)
    assert roles_for("submitter@example.org", s) == (ApiRole.SUBMITTER,)
    assert set(roles_for("both@example.org", s)) == {ApiRole.CLINICIAN, ApiRole.SUBMITTER}
    with pytest.raises(Forbidden) as exc:
        roles_for("stranger@example.org", s)
    assert exc.value.reason == "GOOGLE_ACCOUNT_NOT_AUTHORIZED"


def test_google_sign_in_requires_hs_session_signing() -> None:
    with pytest.raises(ValueError, match="HS"):
        Settings(
            _env_file=None,
            OPENAI_API_KEY="x",
            DATABASE_URL="x",
            JWT_SECRET="x" * 32,
            JWT_ALGORITHM="RS256",
            GOOGLE_CLIENT_ID=CLIENT_ID,
        )


# ------------------------------------------------------------ endpoints


def _client(settings: Settings, *, enabled: bool = True) -> tuple[TestClient, Settings]:
    s = _settings(settings, google_clinicians="clinician.one@example.org") if enabled else settings
    deps = make_app_deps(s)
    if enabled:
        deps = dataclasses.replace(deps, google=_verifier())
    return TestClient(create_app(deps)), s


def test_config_exposes_client_id_only_when_enabled(settings: Settings) -> None:
    on, _ = _client(settings)
    assert on.get("/v1/auth/config").json() == {"google_client_id": CLIENT_ID}
    off, _ = _client(settings, enabled=False)
    assert off.get("/v1/auth/config").json() == {"google_client_id": None}
    r = off.post("/v1/auth/google", json={"credential": _token()})
    assert r.status_code == 404 and r.json()["reason"] == "GOOGLE_AUTH_DISABLED"


def test_sign_in_returns_a_session_token_the_api_accepts(settings: Settings) -> None:
    client, s = _client(settings)
    credential = _token()
    r = client.post("/v1/auth/google", json={"credential": credential})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_type"] == "Bearer" and body["expires_in"] == s.session_ttl_minutes * 60
    assert body["user"] == {
        "sub": "google:109876543210",
        "roles": ["clinician"],
        "email": "clinician.one@example.org",
        "name": "Synthetic Clinician",
        "picture": "https://example.org/a.png",
    }
    assert credential not in r.text  # the Google token is never echoed
    identity = decode_token(body["access_token"], s)
    assert identity.sub == "google:109876543210"
    me = client.get("/v1/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.json() == {"sub": "google:109876543210", "roles": ["clinician"]}


def test_sign_in_refusals_are_problems_without_echo(settings: Settings) -> None:
    client, _ = _client(settings)
    stranger = _token(email="stranger@example.org")
    r = client.post("/v1/auth/google", json={"credential": stranger})
    assert r.status_code == 403 and r.json()["reason"] == "GOOGLE_ACCOUNT_NOT_AUTHORIZED"
    assert "stranger@example.org" not in r.text and stranger not in r.text
    bad = client.post("/v1/auth/google", json={"credential": _token(key=OTHER_KEY)})
    assert bad.status_code == 401 and bad.json()["reason"] == "GOOGLE_TOKEN_INVALID"
    assert client.post("/v1/auth/google", json={"credential": "short"}).status_code == 422
    # a Google ID token is not a CONSILIUM bearer token
    direct = client.get("/v1/me", headers={"Authorization": f"Bearer {_token()}"})
    assert direct.status_code == 401
