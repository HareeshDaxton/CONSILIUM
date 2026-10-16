"""Postgres-backed job queue facade (ARCHITECTURE.md §5.2).

The ``jobs`` table IS the queue: ``enqueue`` inserts ``(case_id, queued)``
(idempotent — one job per case); workers claim with ``SELECT … FOR UPDATE
SKIP LOCKED``. No broker in V1; the SQL lives in ``persistence/`` (only
persistence touches the DB). Job rows carry case IDs and statuses only —
never note text or image data (I-6/I-14).
"""

from __future__ import annotations

import uuid
from datetime import timedelta

from consilium.persistence.repositories import PersistedJob, Repositories

__all__ = ["JobQueue"]


class JobQueue:
    def __init__(self, repos: Repositories) -> None:
        self._repos = repos

    async def enqueue(self, case_id: str) -> None:
        await self._repos.jobs_enqueue(uuid.UUID(case_id))

    async def claim(self, worker_id: str) -> str | None:
        claimed = await self._repos.jobs_claim(worker_id)
        return None if claimed is None else str(claimed)

    async def done(self, case_id: str) -> None:
        await self._repos.jobs_finish(uuid.UUID(case_id), status="done")

    async def dead(self, case_id: str) -> None:
        await self._repos.jobs_finish(uuid.UUID(case_id), status="dead")

    async def retry(self, case_id: str, *, delay: timedelta) -> None:
        await self._repos.jobs_retry(uuid.UUID(case_id), delay=delay)

    async def requeue_stale(self, *, older_than: timedelta) -> int:
        return await self._repos.jobs_requeue_stale(older_than=older_than)

    async def get(self, case_id: str) -> PersistedJob | None:
        return await self._repos.job_get(uuid.UUID(case_id))
