"""E2E fixtures: reuse the real-Postgres fixtures from the integration suite
(skip loudly when the compose Postgres is down — never fall back to SQLite)."""

from tests.integration.conftest import pg_engine, repos  # noqa: F401
