"""Retention / right-to-erasure CLI (plan Section 14.2; AGENTS.md §16).

    uv run python scripts/retention.py erase --case-id <uuid>
    uv run python scripts/retention.py sweep --days <N> [--dry-run]

Runs with ``RETENTION_DATABASE_URL`` (privileged role: the serving role has
DELETE revoked on the append-only tables) and the configured
``OBJECT_STORE_URI``. Prints counts and opaque case IDs only.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from datetime import UTC, datetime

from consilium.core.settings import Settings
from consilium.jobs.retention import erase_everywhere, sweep
from consilium.persistence.db import make_engine, make_session_factory
from consilium.persistence.object_store import make_object_store


async def _main(args: argparse.Namespace) -> int:
    settings = Settings(vision_client_mode="fake")
    url = os.environ.get("RETENTION_DATABASE_URL", settings.database_url)
    engine = make_engine(url)
    factory = make_session_factory(engine)
    objects = make_object_store(settings.object_store_uri)
    try:
        if args.command == "erase":
            result = await erase_everywhere(factory, objects, uuid.UUID(args.case_id))
            if result is None:
                print("case not found")
                return 1
            print(
                f"erased {result.db.case_id}: {sum(result.db.rows.values())} rows, "
                f"{result.objects_deleted} objects, image shared={result.image_kept_shared}"
            )
            return 0
        due = await sweep(
            factory, objects, retention_days=args.days, now=datetime.now(UTC), dry_run=args.dry_run
        )
        print(f"{'would erase' if args.dry_run else 'erased'} {len(due)} case(s)")
        return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(prog="retention")
    sub = parser.add_subparsers(dest="command", required=True)
    erase = sub.add_parser("erase", help="right-to-erasure for one case")
    erase.add_argument("--case-id", required=True)
    sw = sub.add_parser("sweep", help="erase terminal cases older than --days")
    sw.add_argument("--days", type=int, required=True)
    sw.add_argument("--dry-run", action="store_true")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
