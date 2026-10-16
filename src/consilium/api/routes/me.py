"""``GET /v1/me`` — the verified caller identity. The UI uses it to validate a
token at login; it never derives roles from its own decode of the token."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from consilium.api.auth import Identity, require_auth
from consilium.api.schemas_http import MeView

router = APIRouter(tags=["session"])


@router.get("/me", response_model=MeView)
async def me(identity: Annotated[Identity, Depends(require_auth)]) -> MeView:
    return MeView(sub=identity.sub, roles=tuple(sorted(r.value for r in identity.roles)))
