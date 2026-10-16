"""Health, readiness and metrics endpoints (unauthenticated; no case data)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from consilium.api.deps import AppDeps

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "consilium-api"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    deps: AppDeps = request.app.state.deps
    db_ok = await deps.repos.ping()
    return JSONResponse(
        {"status": "ready" if db_ok else "not-ready", "database": db_ok},
        status_code=200 if db_ok else 503,
    )


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
