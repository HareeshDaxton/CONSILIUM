"""Sign-in endpoints (AGENTS.md §17). Public by necessity: they are how a
browser obtains a bearer token. ``/v1/auth/google`` verifies a Google ID token
(``api/google_auth.py``), maps the account to roles from the allowlists, and
returns a CONSILIUM session token. Neither the Google credential nor the
email is logged or persisted; the audit trail records the opaque subject
when that user later acts (reviews)."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from fastapi import APIRouter, Request

from consilium.api.auth import issue_token
from consilium.api.deps import AppDeps
from consilium.api.google_auth import roles_for, session_subject
from consilium.api.schemas_http import (
    AuthConfigView,
    GoogleLoginRequest,
    SessionUser,
    SessionView,
)
from consilium.core.errors import NotFound

router = APIRouter(tags=["auth"])


def _deps(request: Request) -> AppDeps:
    deps: AppDeps = request.app.state.deps
    return deps


@router.get("/auth/config", response_model=AuthConfigView)
async def auth_config(request: Request) -> AuthConfigView:
    deps = _deps(request)
    enabled = deps.google is not None and bool(deps.settings.google_client_id)
    return AuthConfigView(google_client_id=deps.settings.google_client_id if enabled else None)


@router.post("/auth/google", response_model=SessionView)
async def google_sign_in(body: GoogleLoginRequest, request: Request) -> SessionView:
    deps = _deps(request)
    if deps.google is None:
        raise NotFound("GOOGLE_AUTH_DISABLED")
    # key fetch may hit the network (cached afterwards): keep it off the loop
    identity = await asyncio.to_thread(deps.google.verify, body.credential)
    roles = roles_for(identity.email, deps.settings)
    ttl = timedelta(minutes=deps.settings.session_ttl_minutes)
    sub = session_subject(identity)
    token = issue_token(sub, roles, deps.settings, ttl=ttl)
    return SessionView(
        access_token=token,
        expires_in=int(ttl.total_seconds()),
        user=SessionUser(
            sub=sub,
            roles=tuple(sorted(r.value for r in roles)),
            email=identity.email,
            name=identity.name,
            picture=identity.picture,
        ),
    )
