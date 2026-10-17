"""Google sign-in for the review UI (AGENTS.md §17).

Flow: the browser gets a Google ID token (Google Identity Services) and POSTs
it to ``/v1/auth/google``. Here we verify it — RS256 signature against
Google's published keys, ``aud`` = our client ID, ``iss`` = Google, unexpired,
``email_verified`` — then map the account to CONSILIUM roles from explicit
allowlists and mint our own short-lived bearer token (``auth.issue_token``).
Every other endpoint keeps verifying only CONSILIUM tokens.

Identity: the session ``sub`` is ``google:<Google sub>`` — opaque and stable
(reviews/audit store it); the email is used only for the allowlist check and
returned to the UI for display. Nothing here is logged (no emails, no tokens).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jwt

from consilium.api.auth import ApiRole
from consilium.core.errors import Forbidden, Unauthorized
from consilium.core.settings import Settings

__all__ = [
    "GOOGLE_CERTS_URL",
    "GOOGLE_ISSUERS",
    "GoogleIdTokenVerifier",
    "GoogleIdentity",
    "GoogleTokenVerifier",
    "google_jwks_resolver",
    "roles_for",
    "session_subject",
]

GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")


@dataclass(frozen=True, slots=True)
class GoogleIdentity:
    sub: str
    email: str
    name: str | None = None
    picture: str | None = None


class GoogleTokenVerifier(Protocol):
    def verify(self, credential: str) -> GoogleIdentity: ...


KeyResolver = Callable[[str], Any]  # ID token -> public key that signed it


def google_jwks_resolver(url: str = GOOGLE_CERTS_URL) -> KeyResolver:
    """Fetches (and caches) Google's signing keys. Network: call from a thread."""
    client = jwt.PyJWKClient(url, cache_keys=True, lifespan=3600, timeout=10)
    return lambda token: client.get_signing_key_from_jwt(token).key


class GoogleIdTokenVerifier:
    """Verifies Google ID tokens for ONE client ID. ``resolve_key`` is
    injectable so tests sign tokens with a local key (no network, §20)."""

    def __init__(self, client_id: str, resolve_key: KeyResolver, *, leeway_s: int = 30) -> None:
        if not client_id:
            raise ValueError("Google client ID required")
        self._client_id = client_id
        self._resolve_key = resolve_key
        self._leeway_s = leeway_s

    def verify(self, credential: str) -> GoogleIdentity:
        try:
            claims = jwt.decode(
                credential,
                self._resolve_key(credential),
                algorithms=["RS256"],  # pinned: never HS*/none for a third-party token
                audience=self._client_id,
                issuer=GOOGLE_ISSUERS,
                leeway=self._leeway_s,
                options={"require": ["sub", "iss", "aud", "exp", "iat", "email"]},
            )
        except jwt.ExpiredSignatureError:
            raise Unauthorized("GOOGLE_TOKEN_EXPIRED") from None
        except (jwt.PyJWTError, ValueError):
            raise Unauthorized("GOOGLE_TOKEN_INVALID") from None
        sub, email = claims.get("sub"), claims.get("email")
        if not isinstance(sub, str) or not sub or not isinstance(email, str) or not email:
            raise Unauthorized("GOOGLE_TOKEN_INVALID")
        if claims.get("email_verified") is not True:
            raise Forbidden("GOOGLE_EMAIL_UNVERIFIED")
        name, picture = claims.get("name"), claims.get("picture")
        return GoogleIdentity(
            sub=sub,
            email=email.strip().lower(),
            name=name if isinstance(name, str) else None,
            picture=picture if isinstance(picture, str) else None,
        )


def _emails(raw: str) -> frozenset[str]:
    return frozenset(e.strip().lower() for e in raw.split(",") if e.strip())


def roles_for(email: str, settings: Settings) -> tuple[ApiRole, ...]:
    """Roles from the allowlists only. Not listed → 403 (fail closed)."""
    address = email.strip().lower()
    roles = [
        role
        for role, raw in (
            (ApiRole.CLINICIAN, settings.google_clinicians),
            (ApiRole.SUBMITTER, settings.google_submitters),
        )
        if address in _emails(raw)
    ]
    if not roles:
        raise Forbidden("GOOGLE_ACCOUNT_NOT_AUTHORIZED")
    return tuple(roles)


def session_subject(identity: GoogleIdentity) -> str:
    return f"google:{identity.sub}"
