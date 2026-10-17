"""Right-to-erasure job against real Postgres (plan Section 14.2 exit check):
a fully processed fixture case is removed from EVERY table and the object
store; a content-addressed image still used by another case is kept."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from graph_kit import (
    IMAGE,
    IMAGE_URI,
    clean_draft,
    count_rows,
    expected_ctx,
    fake_llm,
    make_runner,
    submit,
)
from sqlalchemy import inspect as sa_inspect

from consilium.core.settings import Settings
from consilium.jobs.retention import OVERLAY_PREFIX, erase_everywhere, sweep
from consilium.persistence.erasure import ERASE_ORDER
from consilium.persistence.models import Base, Case
from consilium.persistence.object_store import MemoryObjectStore
from consilium.persistence.repositories import Repositories
from consilium.vision.overlay import OverlayLayer, overlay_cache_key


def test_erase_order_covers_every_table() -> None:
    tables = set(Base.metadata.tables)
    assert {m.__tablename__ for m in ERASE_ORDER} == tables  # type: ignore[attr-defined]


async def test_erasure_removes_a_case_everywhere(settings: Settings, repos: Repositories) -> None:
    runner = make_runner(settings, repos, llm=fake_llm(clean_draft(await expected_ctx())))
    keep = await runner.run(await submit(runner, key="other"))  # same image hash
    gone = await runner.run(await submit(runner, key="erase-me"))
    factory = repos._session_factory
    objects = MemoryObjectStore({IMAGE_URI: IMAGE})
    async with factory() as s:
        from consilium.persistence.models import CadResultRow

        raw = (await s.get(CadResultRow, uuid.UUID(gone.case_id))).raw_vision_json  # type: ignore[union-attr]
    overlay = objects.uri_for(
        overlay_cache_key(
            gone.case_id, raw["disc_mask_rle"], raw["cup_mask_rle"], OverlayLayer.DISC
        ),
        prefix=OVERLAY_PREFIX,
    )
    objects.objects[overlay] = b"png"

    result = await erase_everywhere(factory, objects, uuid.UUID(gone.case_id))
    assert result is not None and result.objects_deleted == 1  # overlay; image is shared
    assert result.image_kept_shared and IMAGE_URI in objects.objects
    for model in ERASE_ORDER:
        if model is Case:
            async with factory() as s:
                assert await s.get(Case, uuid.UUID(gone.case_id)) is None
            continue
        assert await count_rows(repos, model, gone.case_id) == 0, model.__tablename__  # type: ignore[attr-defined]
    assert await count_rows(repos, ERASE_ORDER[0], keep.case_id) > 0  # neighbour untouched

    # Last reference gone -> the image itself is erased too.
    last = await erase_everywhere(factory, objects, uuid.UUID(keep.case_id))
    assert last is not None and not last.image_kept_shared and IMAGE_URI not in objects.objects
    assert await erase_everywhere(factory, objects, uuid.uuid4()) is None


async def test_sweep_only_takes_old_terminal_cases(settings: Settings, repos: Repositories) -> None:
    runner = make_runner(settings, repos, llm=fake_llm(clean_draft(await expected_ctx())))
    pending = await runner.run(await submit(runner, key="pending"))  # PENDING_REVIEW: not terminal
    objects = MemoryObjectStore()
    future = datetime.now(UTC) + timedelta(days=400)
    due = await sweep(repos._session_factory, objects, retention_days=30, now=future, dry_run=True)
    assert uuid.UUID(pending.case_id) not in due
    assert sa_inspect(Case).primary_key  # sanity: mapped model
