"""Injectable clock (ARCHITECTURE.md §4.3).

All timestamps in the system are timezone-aware UTC. Tests use ``FakeClock``
so latency/retry logic is deterministic.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime."""
        ...


@dataclass(frozen=True)
class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class FakeClock:
    """Manually controlled clock for tests."""

    current: datetime
    step: timedelta = timedelta(seconds=1)

    def __post_init__(self) -> None:
        if self.current.tzinfo is None:
            raise ValueError("FakeClock requires a timezone-aware datetime (use UTC)")

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta | None = None) -> datetime:
        self.current += delta if delta is not None else self.step
        return self.current
