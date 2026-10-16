"""Mint a DEV bearer token for the review UI / API (HS* only).

Production tokens come from the IdP (RS256/ES256, ``api/auth.py``); this is
for the local stack only. The signing secret is ``JWT_SECRET`` from the
environment, defaulting to the docker-compose dev value — never use that
default anywhere real.

    uv run python scripts/dev_token.py --sub clin-1 --role clinician
    uv run python scripts/dev_token.py --sub sub-1 --role submitter --ttl-min 30
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

from consilium.api.auth import ApiRole, issue_token
from consilium.core.settings import Settings

# Mirrors the docker-compose.yml default; a public dev-only value, not a credential.
COMPOSE_DEV_SECRET = "dev-only-insecure-secret-change-me"  # noqa: S105


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sub", required=True, help="opaque user id (never a real name)")
    parser.add_argument(
        "--role", action="append", choices=[r.value for r in ApiRole], required=True
    )
    parser.add_argument("--ttl-min", type=int, default=60)
    args = parser.parse_args(argv)

    secret = os.environ.get("JWT_SECRET")
    if not secret:
        print("JWT_SECRET unset: using the docker-compose DEV secret", file=sys.stderr)
        secret = COMPOSE_DEV_SECRET
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="unused-by-token-minting",
        DATABASE_URL="unused-by-token-minting",
        JWT_SECRET=secret,
        JWT_ALGORITHM=os.environ.get("JWT_ALGORITHM", "HS256"),
        JWT_ISSUER=os.environ.get("JWT_ISSUER") or None,
        JWT_AUDIENCE=os.environ.get("JWT_AUDIENCE") or None,
    )
    roles = tuple(ApiRole(r) for r in dict.fromkeys(args.role))
    print(issue_token(args.sub, roles, settings, ttl=timedelta(minutes=args.ttl_min)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
