"""Worker placeholder (Phase 0, Section 0.4).

Keeps the compose `worker` service alive so the stack topology is real. The
actual worker — the Postgres-backed job loop (SELECT ... FOR UPDATE SKIP
LOCKED) driving the LangGraph state machine per ARCHITECTURE.md §5.2/§5.3 —
lands in Phase 9/11 as consilium.jobs.worker and replaces this script.
"""

import time

from consilium.core.logging import get_logger, setup_logging

logger = get_logger("consilium.worker")


def main() -> None:
    setup_logging("INFO")
    logger.info("worker placeholder started", extra={"phase": "p0-placeholder"})
    try:
        while True:
            time.sleep(30)
            logger.info("worker heartbeat", extra={"phase": "p0-placeholder"})
    except KeyboardInterrupt:
        logger.info("worker placeholder stopped")


if __name__ == "__main__":
    main()
