"""FastAPI application factory (ARCHITECTURE.md §5.2).

Run with ``uvicorn consilium.api.main:create_app --factory``. No module-level
app: importing this module must never construct real clients. Tests call
``create_app(deps)`` with fake-backed ``AppDeps``; production uses
``build_app_deps()`` → ``graph.deps.build_prod_deps`` (the ONLY real-client
construction site).
"""

from __future__ import annotations

from fastapi import FastAPI

from consilium.api.deps import AppDeps, build_app_deps
from consilium.api.errors import install_problem_handlers
from consilium.api.routes import cases, health, me, review
from consilium.observability.otel import RequestTracingMiddleware

__all__ = ["create_app"]


def create_app(deps: AppDeps | None = None) -> FastAPI:
    deps = deps or build_app_deps()
    docs = "/docs" if deps.settings.api_docs_enabled else None
    app = FastAPI(
        title="consilium",
        version="1.0.0",
        docs_url=docs,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.deps = deps
    app.include_router(health.router)
    app.include_router(cases.router, prefix="/v1")
    app.include_router(review.router, prefix="/v1")
    app.include_router(me.router, prefix="/v1")
    install_problem_handlers(app)
    app.add_middleware(RequestTracingMiddleware)
    return app
