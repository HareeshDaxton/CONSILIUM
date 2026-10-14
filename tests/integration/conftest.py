"""Integration-test fixtures needing the local Postgres (docker compose).

Offline rule still holds: no network, no LLM — but these tests use a REAL
local Postgres (plan 9.1/9.4: "real Postgres via test container/local").
If Postgres is unreachable the tests SKIP with an explicit reason instead of
failing; start it with ``rtk docker compose up -d postgres``.
"""

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from consilium.persistence.db import make_engine, make_session_factory
from consilium.persistence.encryption import AesGcmFieldEncryptor
from consilium.persistence.repositories import Repositories

ADMIN_URL = os.environ.get(
    "CONSILIUM_TEST_ADMIN_URL",
    "postgresql+asyncpg://consilium:consilium@localhost:5432/postgres",
)
TEST_DB = "consilium_test"
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"

TEST_KEY = base64.urlsafe_b64encode(b"k" * 32).decode()


async def _probe(url: str) -> bool:
    try:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False


async def _ensure_test_db() -> None:
    engine = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        exists = (
            await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": TEST_DB})
        ).scalar()
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    await engine.dispose()


async def _reset_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))


def _upgrade(url: str, revision: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, revision)


@pytest.fixture
async def pg_engine() -> AsyncIterator[AsyncEngine]:
    """Fresh-migrated test database per test; skips when Postgres is down."""
    if not await _probe(ADMIN_URL):
        pytest.skip("local Postgres unavailable — start with `rtk docker compose up -d postgres`")
    await _ensure_test_db()
    engine = make_engine(TEST_URL)
    await _reset_schema(engine)
    await asyncio.to_thread(_upgrade, TEST_URL, "head")
    yield engine
    await engine.dispose()


@pytest.fixture
async def repos(pg_engine: AsyncEngine) -> Repositories:
    return Repositories(make_session_factory(pg_engine), AesGcmFieldEncryptor(b"k" * 32))
