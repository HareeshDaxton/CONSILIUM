"""Opaque ID generation (I-6: case IDs are opaque, never derived from PHI).

``IdGen`` is a protocol so tests can inject deterministic IDs
(ARCHITECTURE.md §4.3 dependency injection).
"""

import uuid
from dataclasses import dataclass, field
from typing import Protocol


class IdGen(Protocol):
    def new_id(self) -> str:
        """Return a new opaque identifier."""
        ...


@dataclass(frozen=True)
class UuidIdGen:
    """Production generator: random UUID4."""

    def new_id(self) -> str:
        return str(uuid.uuid4())


@dataclass
class FakeIdGen:
    """Deterministic generator for tests: ``<prefix>-00000000-0000-0000-0000-NNNNNNNNNNNN``."""

    prefix: str = "test"
    _counter: int = field(default=0, init=False, repr=False)

    def new_id(self) -> str:
        self._counter += 1
        return f"{self.prefix}-00000000-0000-0000-0000-{self._counter:012d}"
