"""Bearer-token authentication + role guards (AGENTS.md §17, ARCHITECTURE.md §5.2).

JWT bearer tokens: HS256 in dev (shared ``JWT_SECRET``), OIDC-ready for prod
(RS256/ES256 with the IdP public key, issuer/audience checked when set).
Claims: ``sub`` (opaque user id) and ``roles`` (subset of submitter/clinician);
``exp`` is mandatory. Identity always comes from the verified token — never
from a request body (review service relies on this for the no-self-approval
rule).

Failures raise ``Unauthorized``/``Forbidden``; the RFC 7807 handler renders
them without echoing the token or any claim values.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import jwt
from fastapi import Request

from consilium.core.errors import Forbidden, Unauthorized
from consilium.core.settings import Settings

__all__ = [
    "ApiRole",
    "Identity",
    "decode_token",
    "issue_token",
    "require_auth",
    "require_clinician",
    "require_submitter",
]


class ApiRole(StrEnum):
    SUBMITTER = "submitter"
    CLINICIAN = "clinician"


class Identity:
    """Verified caller identity (immutable)."""

    __slots__ = ("roles", "sub")

    def __init__(self, sub: str, roles: frozenset[ApiRole]) -> None:
        self.sub = sub
        self.roles = roles

    def has(self, role: ApiRole) -> bool:
        return role in self.roles

    def __repr__(self) -> str:
        return f"Identity(sub={self.sub!r}, roles={sorted(self.roles)!r})"


def _verification_key(settings: Settings) -> str:
    if settings.jwt_algorithm.startswith("HS"):
        return settings.jwt_secret.get_secret_value()
    if settings.jwt_public_key is None:
        raise Unauthorized("AUTH_MISCONFIGURED", detail="asymmetric JWT without public key")
    return settings.jwt_public_key.get_secret_value()


def decode_token(token: str, settings: Settings) -> Identity:
    """Verify signature, expiry, issuer/audience; map claims to an Identity."""
    try:
        claims = jwt.decode(
            token,
            _verification_key(settings),
            algorithms=[settings.jwt_algorithm],  # pinned: no alg confusion / "none"
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["sub", "exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise Unauthorized("TOKEN_EXPIRED") from None
    except jwt.PyJWTError:
        raise Unauthorized("TOKEN_INVALID") from None
    sub = claims.get("sub")
    raw_roles = claims.get("roles", [])
    if not isinstance(sub, str) or not sub or not isinstance(raw_roles, list):
        raise Unauthorized("TOKEN_INVALID")
    roles = frozenset(ApiRole(r) for r in raw_roles if r in ApiRole.__members__.values())
    if not roles:
        raise Forbidden("NO_ROLE")
    return Identity(sub, roles)


def issue_token(
    sub: str,
    roles: tuple[ApiRole, ...],
    settings: Settings,
    *,
    ttl: timedelta = timedelta(hours=1),
    now: datetime | None = None,
) -> str:
    """Dev/test token minting (HS* only). Prod tokens come from the IdP."""
    if not settings.jwt_algorithm.startswith("HS"):
        raise ValueError("issue_token only mints HS* dev tokens")
    issued = now or datetime.now(UTC)
    payload: dict[str, object] = {
        "sub": sub,
        "roles": [r.value for r in roles],
        "iat": issued,
        "exp": issued + ttl,
    }
    if settings.jwt_issuer is not None:
        payload["iss"] = settings.jwt_issuer
    if settings.jwt_audience is not None:
        payload["aud"] = settings.jwt_audience
    return jwt.encode(
        payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise Unauthorized("TOKEN_MISSING")
    return token.strip()


async def require_auth(request: Request) -> Identity:
    """Any authenticated caller; the identity is cached on the request."""
    cached = getattr(request.state, "identity", None)
    if isinstance(cached, Identity):
        return cached
    settings: Settings = request.app.state.deps.settings
    identity = decode_token(_bearer(request), settings)
    request.state.identity = identity
    return identity


def _require(role: ApiRole) -> Callable[[Request], Awaitable[Identity]]:
    async def guard(request: Request) -> Identity:
        identity = await require_auth(request)
        if not identity.has(role):
            raise Forbidden(f"ROLE_{role.value.upper()}_REQUIRED")
        return identity

    guard.__name__ = f"require_{role.value}"
    return guard


require_submitter = _require(ApiRole.SUBMITTER)
require_clinician = _require(ApiRole.CLINICIAN)
