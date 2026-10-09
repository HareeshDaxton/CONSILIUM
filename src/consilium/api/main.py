"""Minimal API placeholder (Phase 0, Section 0.4).

Exists only so the compose stack's `api` service answers /healthz. The real
application — create_app(deps) with Deps injection, auth, RFC 7807 errors and
all routes per ARCHITECTURE.md §5.2 — is built in Phase 11 and replaces this.
"""

from fastapi import FastAPI

app = FastAPI(title="consilium", version="0.1.0", docs_url=None, redoc_url=None)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "consilium-api", "phase": "p0-placeholder"}
