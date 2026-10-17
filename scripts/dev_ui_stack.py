"""Offline dev stack for the review UI (Phase 12.4 walkthrough).

Runs the REAL FastAPI app (auth, RFC 7807, review service), the REAL job
queue + worker and ``HttpVisionClient`` speaking HTTP to the fake-vision app
(in-process ASGI transport) against the compose Postgres, in one process on
127.0.0.1. The LLM is the canned ``FakeLLMClient`` from the offline graph kit
— no network, no real model (AGENTS.md §20), synthetic data only.

Seeding goes through the HTTP API exactly as a submitter would (multipart
POST /v1/cases, Idempotency-Key) and the worker; nothing is written to the
database by hand. Scenarios (each with its own canned LLM/vision):

    approve     → PENDING_REVIEW               (approve-with-edits path)
    reject      → PENDING_REVIEW               (reject path)
    unverified  → NEEDS_ATTENTION, UNVERIFIED  (critic always rejects c2)
    failed      → FAILED (vision revision-pin mismatch, fail closed)

Usage (Postgres from ``docker compose up -d postgres``):

    uv run python scripts/dev_ui_stack.py --reset --out frontend/e2e/.stack.json
    cd frontend && npm run dev       # http://localhost:5173, proxies /v1 → :8000

``--out`` writes the dev tokens + seeded case IDs as JSON (gitignored path);
tokens are HS256 dev tokens signed with the compose dev secret.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "tests" / "integration", REPO / "scripts", REPO):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from graph_kit import IMAGE, NOTE_TEXT, REJECT, clean_draft, expected_ctx, fake_llm  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402
from tests.api_kit import PIN, make_app_deps  # noqa: E402

from consilium.api.auth import ApiRole, issue_token  # noqa: E402
from consilium.api.deps import AppDeps, build_google_verifier  # noqa: E402
from consilium.api.main import create_app  # noqa: E402
from consilium.core.settings import Settings  # noqa: E402
from consilium.jobs.worker import Worker  # noqa: E402
from consilium.persistence.db import make_engine, make_session_factory  # noqa: E402
from consilium.persistence.encryption import AesGcmFieldEncryptor, parse_key  # noqa: E402
from consilium.persistence.object_store import FileObjectStore  # noqa: E402
from consilium.persistence.repositories import Repositories  # noqa: E402
from consilium.vision.client_http import HttpVisionClient  # noqa: E402
from dev_token import COMPOSE_DEV_SECRET  # noqa: E402

ADMIN_URL = os.environ.get(
    "CONSILIUM_DEV_ADMIN_URL", "postgresql+asyncpg://consilium:consilium@localhost:5432/postgres"
)
DEV_DB = "consilium_dev_ui"
# Public, dev-only AES key for SYNTHETIC notes in the throwaway dev database.
DEV_FIELD_KEY = "dev-only-field-key-synthetic-32b"
OBJECTS = REPO / "data" / "dev_ui_objects"  # repo-root data/ is gitignored
SUBMITTER = "dev-submitter"
CLINICIAN = "dev-clinician"
CLINICIAN_2 = "dev-clinician-2"
SCENARIOS = ("approve", "reject", "unverified", "failed")
APPROVE_VERDICT: dict[str, Any] = {"approved": True, "issues": [], "round": 1}


def _env(name: str) -> str | None:
    """Process env first, then the repo ``.env`` (only these named keys are read)."""
    dotenv = REPO / ".env"
    value = os.environ.get(name) or (dotenv_values(dotenv).get(name) if dotenv.exists() else None)
    return value or None


def dev_settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        OPENAI_API_KEY="unused-offline-fake-llm",
        DATABASE_URL=database_url,
        JWT_SECRET=_env("JWT_SECRET") or COMPOSE_DEV_SECRET,
        FIELD_ENCRYPTION_KEY=_env("FIELD_ENCRYPTION_KEY") or DEV_FIELD_KEY,
        RAILS_ENGINE="passthrough",
        # Google sign-in for the UI: live Google token verification when set.
        GOOGLE_CLIENT_ID=_env("GOOGLE_CLIENT_ID"),
        GOOGLE_CLINICIANS=_env("GOOGLE_CLINICIANS") or "",
        GOOGLE_SUBMITTERS=_env("GOOGLE_SUBMITTERS") or "",
    )


async def prepare_database(admin_url: str, *, reset: bool) -> str:
    """Create the dev database if missing; ``reset`` wipes it. Then migrate."""
    url = admin_url.rsplit("/", 1)[0] + f"/{DEV_DB}"
    admin = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        exists = (
            await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": DEV_DB})
        ).scalar()
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{DEV_DB}"'))
    await admin.dispose()
    if reset:
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
        await engine.dispose()
    await asyncio.to_thread(_upgrade, url)
    return url


def _upgrade(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "src/consilium/persistence/alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def _fake_vision_app() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fake_vision_server_dev", REPO / "scripts" / "fake_vision_server.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def http_vision(pinned: str = PIN) -> HttpVisionClient:
    """Real HTTP client + pin check against the fake-vision app (no sockets)."""
    return HttpVisionClient(
        endpoint_url="http://fake-vision",
        token="dev-fake-token",  # noqa: S106  # fake-vision dev token (compose default)
        pinned_revision=pinned,
        transport=httpx.ASGITransport(app=_fake_vision_app()),
    )


@dataclass(frozen=True)
class Stack:
    deps: AppDeps  # serves the API + the live worker (clean scenario)
    scenario_deps: dict[str, AppDeps]


async def build_stack(settings: Settings, repos: Repositories, objects: Any) -> Stack:
    draft = clean_draft(await expected_ctx())

    def deps(*, critic: tuple[dict[str, Any], ...] = (APPROVE_VERDICT,), pin: str = PIN) -> AppDeps:
        return make_app_deps(
            settings,
            repos=repos,
            llm=fake_llm(draft, critic=critic),
            vision=http_vision(pin),
            objects=objects,
        )

    return Stack(
        deps=deps(),
        scenario_deps={
            "approve": deps(),
            "reject": deps(),
            "unverified": deps(critic=(REJECT,)),
            "failed": deps(pin="sha256:dev/pin-mismatch"),
        },
    )


def dev_tokens(settings: Settings, ttl: timedelta = timedelta(hours=8)) -> dict[str, str]:
    def mint(sub: str, *roles: ApiRole) -> str:
        return issue_token(sub, roles, settings, ttl=ttl)

    return {
        "clinician": mint(CLINICIAN, ApiRole.CLINICIAN),
        "clinician_2": mint(CLINICIAN_2, ApiRole.CLINICIAN),
        "submitter": mint(SUBMITTER, ApiRole.SUBMITTER),
        # same identity as the seeded cases' submitter → self-approval is refused
        "submitter_clinician": mint(SUBMITTER, ApiRole.SUBMITTER, ApiRole.CLINICIAN),
    }


async def seed(stack: Stack, settings: Settings, repos: Repositories) -> dict[str, str]:
    """Submit one synthetic case per scenario through the HTTP API, then let
    that scenario's worker process it. Idempotent per scenario key."""
    token = dev_tokens(settings)["submitter"]
    app = create_app(stack.deps)
    cases: dict[str, str] = {}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://dev") as client:
        for name in SCENARIOS:
            r = await client.post(
                "/v1/cases",
                headers={"Authorization": f"Bearer {token}", "Idempotency-Key": f"dev-ui-{name}"},
                files={"image": ("synthetic_fundus.png", IMAGE, "image/png")},
                data={"note": NOTE_TEXT},
            )
            r.raise_for_status()
            cases[name] = r.json()["case_id"]
            await Worker(
                stack.scenario_deps[name].runner, repos, worker_id=f"seed-{name}"
            ).run_once()
    return cases


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def serve(args: argparse.Namespace) -> None:
    url = await prepare_database(args.admin_url, reset=args.reset)
    settings = dev_settings(url)
    assert settings.field_encryption_key is not None
    repos = Repositories(
        make_session_factory(make_engine(url)),
        AesGcmFieldEncryptor(parse_key(settings.field_encryption_key.get_secret_value())),
    )
    await asyncio.to_thread(OBJECTS.mkdir, parents=True, exist_ok=True)
    stack = await build_stack(settings, repos, FileObjectStore(OBJECTS))
    cases = await seed(stack, settings, repos)
    info = {
        "api": f"http://{args.host}:{args.port}",
        "tokens": dev_tokens(settings),
        "cases": cases,
    }
    if args.out:
        await asyncio.to_thread(_write_json, Path(args.out), info)
    print(f"dev UI stack: API {info['api']}  db={DEV_DB}")
    for name, case_id in cases.items():
        print(f"  seeded {name:<10} {case_id}")
    print(f"  clinician token: {info['tokens']['clinician']}")

    google = None if args.no_google else build_google_verifier(settings)
    served = dataclasses.replace(stack.deps, google=google)
    print(
        "  Google sign-in: "
        + ("ON (allowlists from GOOGLE_CLINICIANS / GOOGLE_SUBMITTERS)" if served.google else "off")
    )

    stop = asyncio.Event()
    worker = Worker(stack.deps.runner, repos, worker_id="dev-ui-worker", poll_interval_s=0.5)
    worker_task = asyncio.create_task(worker.run_forever(stop))
    server = uvicorn.Server(
        uvicorn.Config(create_app(served), host=args.host, port=args.port, log_level="warning")
    )
    try:
        await server.serve()
    finally:
        stop.set()
        await worker_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline dev stack for the review UI")
    parser.add_argument("--reset", action="store_true", help=f"wipe and re-seed {DEV_DB}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--admin-url", default=ADMIN_URL)
    parser.add_argument("--out", default=None, help="write tokens + seeded case IDs as JSON")
    parser.add_argument(
        "--no-google", action="store_true", help="disable Google sign-in (deterministic e2e runs)"
    )
    asyncio.run(serve(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
