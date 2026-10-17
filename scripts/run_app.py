"""Run CONSILIUM for real: API + pipeline worker in one process.

    docker compose up -d postgres
    uv run python scripts/run_app.py            # API on http://127.0.0.1:8000
    cd frontend && npm run dev                  # UI on http://localhost:5173

Unlike scripts/dev_ui_stack.py (canned LLM, fake vision), this uses:
  - Deva's GPU vision server (Deva/server.py) over HTTP for SwinV2 + SegFormer;
  - OpenAI for note extraction, the three specialists, the Director and the critic.

Settings come from the repo-root .env. Needed:
  OPENAI_API_KEY, VISION_ENDPOINT_URL, VISION_ENDPOINT_TOKEN,
  JWT_SECRET (>= 32 bytes), FIELD_ENCRYPTION_KEY, and for Google sign-in
  GOOGLE_CLIENT_ID + GOOGLE_CLINICIANS / GOOGLE_SUBMITTERS.
Sensible defaults are filled in for the rest (local Postgres, http vision,
the pinned model revision, passthrough model-rail when NeMo is not installed —
the deterministic input/output checks always run).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _fill_defaults() -> list[str]:
    """Load .env into the environment and fill safe defaults. Returns problems."""
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env", override=False)
    from consilium.vision.config import load_models_config

    defaults = {
        "DATABASE_URL": "postgresql+asyncpg://consilium:consilium@localhost:5432/consilium",
        "VISION_CLIENT_MODE": "http",
        "VISION_PINNED_REVISION": load_models_config().endpoint.pinned_revision,
        "OBJECT_STORE_URI": f"file://{(REPO / 'data' / 'objects').as_posix()}",
    }
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
    if not os.environ.get("RAILS_ENGINE") and importlib.util.find_spec("nemoguardrails") is None:
        os.environ["RAILS_ENGINE"] = "passthrough"

    problems = []
    required = {
        "OPENAI_API_KEY": "the AI agents that write the report (OpenAI API key)",
        "VISION_ENDPOINT_URL": "Deva's vision server URL (see Deva/README.md)",
        "VISION_ENDPOINT_TOKEN": "the shared token from Deva's Deva/.env",
        "JWT_SECRET": "signing sessions (any random string, >= 32 characters)",
        "FIELD_ENCRYPTION_KEY": "encrypting notes at rest (see .env.example)",
    }
    for key, why in required.items():
        if not os.environ.get(key, "").strip():
            problems.append(f"  {key} is not set in .env - needed for {why}")
    return problems


def _migrate(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "src/consilium/persistence/alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


async def serve(host: str, port: int) -> None:
    import uvicorn

    from consilium.api.deps import build_app_deps
    from consilium.api.main import create_app
    from consilium.core.settings import Settings
    from consilium.jobs.worker import Worker
    from consilium.vision.client import SupportsHealth

    settings = Settings()
    await asyncio.to_thread(_migrate, settings.database_url)
    deps = build_app_deps(settings)

    print(f"CONSILIUM: API http://{host}:{port}   (UI: cd frontend && npm run dev)")
    print(f"  vision server : {settings.vision_endpoint_url}")
    if isinstance(deps.graph.vision, SupportsHealth):
        health = await deps.graph.vision.health()
        pin = deps.graph.models.endpoint.pinned_revision
        if not health.reachable:
            print(
                f"  vision status : NOT REACHABLE ({health.detail}) - start Deva's server;"
                " cases fail closed until then"
            )
        elif health.revision != pin:
            print(
                "  vision status : REACHABLE but model revision differs -"
                " give Deva the Deva/ folder from this project version"
            )
        else:
            print(f"  vision status : OK on {health.device}")
    print(
        f"  Google sign-in: {'ON' if deps.google else 'off'}   model rail: {settings.rails_engine}"
    )

    stop = asyncio.Event()
    worker = Worker(
        deps.runner,
        deps.repos,
        max_attempts=settings.worker_max_attempts,
        poll_interval_s=settings.worker_poll_interval_s,
    )
    worker_task = asyncio.create_task(worker.run_forever(stop))
    server = uvicorn.Server(
        uvicorn.Config(create_app(deps), host=host, port=port, log_level="warning")
    )
    try:
        await server.serve()
    finally:
        stop.set()
        await worker_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CONSILIUM (API + worker) for real")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]  # show status lines at once
    problems = _fill_defaults()
    if problems:
        print("CONSILIUM cannot start yet:\n" + "\n".join(problems), file=sys.stderr)
        return 2
    asyncio.run(serve(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
