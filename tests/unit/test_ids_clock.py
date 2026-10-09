"""Unit tests for core/ids.py and core/clock.py."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from consilium.core.clock import FakeClock, SystemClock
from consilium.core.ids import FakeIdGen, UuidIdGen


class TestUuidIdGen:
    def test_returns_valid_unique_uuids(self) -> None:
        gen = UuidIdGen()
        ids = {gen.new_id() for _ in range(100)}
        assert len(ids) == 100
        for i in ids:
            assert uuid.UUID(i).version == 4


class TestFakeIdGen:
    def test_deterministic_sequence(self) -> None:
        gen = FakeIdGen(prefix="case")
        assert gen.new_id() == "case-00000000-0000-0000-0000-000000000001"
        assert gen.new_id() == "case-00000000-0000-0000-0000-000000000002"

    def test_independent_counters(self) -> None:
        a, b = FakeIdGen(), FakeIdGen()
        assert a.new_id() == b.new_id()


class TestSystemClock:
    def test_now_is_tz_aware_utc(self) -> None:
        now = SystemClock().now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)
        assert abs((datetime.now(UTC) - now).total_seconds()) < 5


class TestFakeClock:
    def test_now_returns_current(self) -> None:
        t = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
        assert FakeClock(current=t).now() == t

    def test_advance_default_step_and_custom_delta(self) -> None:
        clock = FakeClock(current=datetime(2025, 1, 1, tzinfo=UTC), step=timedelta(seconds=5))
        assert clock.advance() == datetime(2025, 1, 1, 0, 0, 5, tzinfo=UTC)
        assert clock.advance(timedelta(minutes=2)) == datetime(2025, 1, 1, 0, 2, 5, tzinfo=UTC)

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            FakeClock(current=datetime(2025, 1, 1))
