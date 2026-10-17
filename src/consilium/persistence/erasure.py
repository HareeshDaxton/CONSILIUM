"""Right-to-erasure + retention queries (AGENTS.md §16; plan Section 14.2).

``erase_case`` deletes EVERY row of one case from EVERY table, children
first, in one transaction, and returns what the object-store side still has
to remove (image URI/hash, mask RLEs for the overlay cache keys). It is the
only sanctioned delete path for the append-only tables (``audit_log``,
``verdicts``, ``reviews``): the app role has UPDATE/DELETE revoked there, so
this job must run with the privileged retention role
(``RETENTION_DATABASE_URL``) — never with the serving credentials.

``ERASE_ORDER`` must name every table that references ``cases``; a test
asserts it covers ``Base.metadata`` so a new table cannot be forgotten.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select

from consilium.persistence.db import SessionFactory
from consilium.persistence.models import (
    AgentRunRow,
    AuditLogRow,
    Base,
    CadResultRow,
    Case,
    ClinicalNoteRow,
    DraftRow,
    JobRow,
    ReviewRow,
    RoutedContextRow,
    VerdictRow,
    VisionCallRow,
)

# Children first; ``cases`` last.
ERASE_ORDER: tuple[type[Base], ...] = (
    AgentRunRow,
    RoutedContextRow,
    VisionCallRow,
    CadResultRow,
    ClinicalNoteRow,
    DraftRow,
    VerdictRow,
    ReviewRow,
    AuditLogRow,
    JobRow,
    Case,
)


@dataclass(frozen=True)
class CaseErasure:
    """What was removed from the DB + what the object store must drop."""

    case_id: str
    rows: dict[str, int] = field(default_factory=dict)
    image_uri: str | None = None
    image_sha256: str | None = None
    disc_mask_rle: str | None = None
    cup_mask_rle: str | None = None


async def erase_case(factory: SessionFactory, case_id: uuid.UUID) -> CaseErasure | None:
    """Delete one case everywhere in the DB (single transaction). None if unknown."""
    async with factory() as session, session.begin():
        case = await session.get(Case, case_id)
        if case is None:
            return None
        cad = await session.get(CadResultRow, case_id)
        raw: dict[str, Any] = cad.raw_vision_json if cad is not None else {}
        image_uri, image_sha = case.image_uri, case.image_sha256
        rows: dict[str, int] = {}
        for model in ERASE_ORDER:
            column = model.id if model is Case else model.case_id  # type: ignore[attr-defined]
            result = await session.execute(delete(model).where(column == case_id))
            rows[model.__tablename__] = int(result.rowcount or 0)  # type: ignore[attr-defined]
    return CaseErasure(
        case_id=str(case_id),
        rows=rows,
        image_uri=image_uri,
        image_sha256=image_sha,
        disc_mask_rle=raw.get("disc_mask_rle"),
        cup_mask_rle=raw.get("cup_mask_rle"),
    )


async def image_referenced(factory: SessionFactory, image_sha256: str) -> bool:
    """Content-addressed images may be shared: True if any case still uses it."""
    async with factory() as session:
        count = await session.scalar(
            select(func.count()).select_from(Case).where(Case.image_sha256 == image_sha256)
        )
    return bool(count)


async def cases_older_than(
    factory: SessionFactory, cutoff: datetime, statuses: tuple[str, ...]
) -> tuple[uuid.UUID, ...]:
    """Retention candidates: cases created before ``cutoff`` in ``statuses``."""
    async with factory() as session:
        result = await session.scalars(
            select(Case.id).where(Case.created_at < cutoff, Case.status.in_(statuses))
        )
        return tuple(result)
