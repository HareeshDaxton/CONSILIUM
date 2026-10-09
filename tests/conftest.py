"""Shared pytest fixtures (ARCHITECTURE.md §12).

Rules:
  - The default suite is fully offline: no network, no real LLM (fakes only).
  - Fixtures contain NO real patient data — synthetic only, produced by the
    committed generator scripts/make_fixtures.py (see tests/fixtures/PROVENANCE.md).
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from consilium.core.clock import FakeClock
from consilium.core.ids import FakeIdGen
from consilium.core.settings import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory of committed synthetic fixtures (generated, never real data)."""
    return FIXTURES_DIR


@pytest.fixture
def fake_clock() -> FakeClock:
    """Deterministic clock pinned at a fixed instant."""
    return FakeClock(current=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC))


@pytest.fixture
def fake_id_gen() -> FakeIdGen:
    """Deterministic ID sequence: test-...-000000000001, ...2, ..."""
    return FakeIdGen()


@pytest.fixture
def object_store_dir(tmp_path: Path) -> Path:
    """Temporary content-addressed object store stand-in (dev uses local FS)."""
    store = tmp_path / "objects"
    store.mkdir()
    return store


@pytest.fixture
def settings() -> Settings:
    """Settings with synthetic required values; no .env, no real secrets."""
    return Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test-synthetic",
        DATABASE_URL="postgresql+asyncpg://consilium:consilium@localhost:5432/consilium_test",
        JWT_SECRET="test-jwt-secret",
    )
