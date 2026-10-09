"""Sample integration test (Section 0.5 exit).

Offline, fakes only: proves core components wire together across module
boundaries (settings → logging → errors). Real graph/DB integration tests
land in Phase 9.
"""

import io
import json
import logging

from consilium.core.errors import VisionFailure
from consilium.core.logging import JsonFormatter
from consilium.core.settings import Settings


def test_settings_logging_errors_wire_together(settings: Settings) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("consilium.itest")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel("INFO")
    logger.propagate = False

    err = VisionFailure("VISION_UNAVAILABLE", detail="cold start budget exceeded")
    logger.error(
        "node failed",
        extra={
            "case_id": "synthetic-case-001",
            "node": "call_vision",
            "status": err.case_status,
            "vision_mode": settings.vision_client_mode,
        },
    )

    stream.seek(0)
    (record,) = [json.loads(line) for line in stream.read().strip().splitlines()]
    assert record["status"] == "FAILED"  # fail-closed mapping survives the boundary
    assert record["vision_mode"] == "fake"
    assert record["node"] == "call_vision"
