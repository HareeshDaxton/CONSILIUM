"""Retention / right-to-erasure job (AGENTS.md §16; plan Section 14.2).

``erase_everywhere`` removes one case from Postgres (every table, one
transaction) and then from the object store: every cached overlay layer and
the uploaded image — the image only when no other case references the same
content hash (content addressing, §16). ``sweep`` applies the retention
period to terminal cases.

Order is DB first, objects second: if object deletion fails after the DB
commit, the objects are orphans with no remaining reference (re-run the sweep
with the reported URIs); the reverse order could leave live rows pointing at
deleted images.

Logs carry the opaque case ID and row/object COUNTS only (I-6).

# CLINICAL-REVIEW: the retention period (RETENTION_DAYS) and which statuses
# are retained/erased are legal + clinical decisions; unset = sweep disabled.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from consilium.core.logging import get_logger
from consilium.persistence.db import SessionFactory
from consilium.persistence.erasure import (
    CaseErasure,
    cases_older_than,
    erase_case,
    image_referenced,
)
from consilium.persistence.object_store import ObjectStore
from consilium.schemas.enums import CaseStatus
from consilium.vision.overlay import OverlayLayer, overlay_cache_key

logger = get_logger(__name__)

OVERLAY_PREFIX = "overlays"  # == api.routes.cases.OVERLAY_PREFIX
TERMINAL_STATUSES: tuple[str, ...] = tuple(
    s.value
    for s in (
        CaseStatus.APPROVED,
        CaseStatus.APPROVED_WITH_EDITS,
        CaseStatus.REJECTED,
        CaseStatus.REJECTED_INPUT,
        CaseStatus.FAILED,
    )
)


@dataclass(frozen=True)
class ErasureResult:
    db: CaseErasure
    objects_deleted: int
    image_kept_shared: bool


async def erase_everywhere(
    factory: SessionFactory, objects: ObjectStore, case_id: uuid.UUID
) -> ErasureResult | None:
    db = await erase_case(factory, case_id)
    if db is None:
        return None
    deleted = 0
    if db.disc_mask_rle is not None and db.cup_mask_rle is not None:
        for layer in OverlayLayer:
            key = overlay_cache_key(db.case_id, db.disc_mask_rle, db.cup_mask_rle, layer)
            deleted += await objects.delete(objects.uri_for(key, prefix=OVERLAY_PREFIX))
    shared = False
    if db.image_uri is not None:
        shared = db.image_sha256 is not None and await image_referenced(factory, db.image_sha256)
        if not shared:
            deleted += await objects.delete(db.image_uri)
    logger.info(
        "retention.erased",
        extra={
            "case_id": db.case_id,
            "rows": sum(db.rows.values()),
            "objects": deleted,
            "image_shared": shared,
        },
    )
    return ErasureResult(db=db, objects_deleted=deleted, image_kept_shared=shared)


async def sweep(
    factory: SessionFactory,
    objects: ObjectStore,
    *,
    retention_days: int,
    now: datetime,
    dry_run: bool = False,
) -> tuple[uuid.UUID, ...]:
    """Erase terminal cases older than ``retention_days``. Returns their IDs."""
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    due = await cases_older_than(factory, now - timedelta(days=retention_days), TERMINAL_STATUSES)
    if not dry_run:
        for case_id in due:
            await erase_everywhere(factory, objects, case_id)
    return due
