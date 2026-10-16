"""Pipeline worker: claim a job → ``GraphRunner.resume`` → mark done.

Run with ``python -m consilium.jobs.worker``. The graph itself converts every
pipeline error into a persisted terminal status (FAILED / NEEDS_ATTENTION /
REJECTED_INPUT, I-5), so ``resume`` normally returns. An exception escaping it
is infrastructure (DB/object store down): the job is retried with backoff up
to ``WORKER_MAX_ATTEMPTS``, then marked ``dead`` with an audit event naming
the exception TYPE only. Resume is idempotent (§4), so a retried job never
recomputes vision or extraction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import uuid
from datetime import timedelta

from consilium.graph.build import GraphRunner
from consilium.jobs.queue import JobQueue
from consilium.persistence.repositories import Repositories

__all__ = ["Worker", "main"]

logger = logging.getLogger("consilium.worker")


class Worker:
    def __init__(
        self,
        runner: GraphRunner,
        repos: Repositories,
        *,
        worker_id: str | None = None,
        max_attempts: int = 3,
        poll_interval_s: float = 1.0,
        stale_after: timedelta = timedelta(minutes=15),
    ) -> None:
        self._runner = runner
        self._repos = repos
        self._queue = JobQueue(repos)
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._max_attempts = max_attempts
        self._poll_interval_s = poll_interval_s
        self._stale_after = stale_after

    async def run_once(self) -> str | None:
        """Process at most one job; returns its case ID, or None if idle."""
        case_id = await self._queue.claim(self.worker_id)
        if case_id is None:
            return None
        try:
            final = await self._runner.resume(case_id)
        except Exception as exc:  # worker boundary: infra failure, not pipeline logic
            await self._on_error(case_id, exc)
            return case_id
        await self._queue.done(case_id)
        logger.info("job done", extra={"case_id": case_id, "status": final.status.value})
        return case_id

    async def _on_error(self, case_id: str, exc: Exception) -> None:
        job = await self._queue.get(case_id)
        attempts = job.attempts if job is not None else self._max_attempts
        error_type = type(exc).__name__  # never str(exc): may carry content (I-6)
        if attempts >= self._max_attempts:
            await self._queue.dead(case_id)
            async with self._repos.transaction() as tx:
                await tx.audit(
                    case_id=uuid.UUID(case_id),
                    event="job.dead",
                    actor="worker",
                    detail={"error_type": error_type, "attempts": attempts},
                )
            logger.error("job dead", extra={"case_id": case_id, "error_type": error_type})
        else:
            await self._queue.retry(case_id, delay=timedelta(seconds=2**attempts))
            logger.warning("job retry", extra={"case_id": case_id, "error_type": error_type})

    async def run_forever(self, stop: asyncio.Event) -> None:
        await self._queue.requeue_stale(older_than=self._stale_after)
        while not stop.is_set():
            if await self.run_once() is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_interval_s)


def main() -> None:  # pragma: no cover - process entry point
    from consilium.api.deps import build_app_deps
    from consilium.core.logging import setup_logging

    setup_logging("INFO")
    deps = build_app_deps()
    worker = Worker(
        deps.runner,
        deps.repos,
        max_attempts=deps.settings.worker_max_attempts,
        poll_interval_s=deps.settings.worker_poll_interval_s,
    )
    asyncio.run(worker.run_forever(asyncio.Event()))


if __name__ == "__main__":  # pragma: no cover
    main()
