"""Structured JSON logging (AGENTS.md §18, §21).

PHI discipline (I-6/I-7/I-14, SKILLS.md phi-review): note text, report text,
upload filenames, prompts, and image data must NEVER appear in logs. Any
``extra`` field whose key is in :data:`DENY_KEYS` is replaced with
``"[REDACTED]"`` at emission time so misuse is visible instead of silent.

Usage::

    logger = get_logger(__name__)
    logger.info("node complete", extra={"case_id": case_id, "node": "call_vision"})
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Keys that must never be emitted (I-6/I-7/I-14). Matching is exact, lowercase.
DENY_KEYS: frozenset[str] = frozenset(
    {
        "note",
        "note_text",
        "raw_note",
        "raw_text",
        "filename",
        "upload_filename",
        "image",
        "image_bytes",
        "prompt",
        "messages",
        "draft_text",
        "patient",
        "mrn",
        "dob",
    }
)

REDACTED = "[REDACTED]"

# Standard LogRecord attributes — everything else in __dict__ came from `extra`.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields included, deny-listed keys redacted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = REDACTED if key.lower() in DENY_KEYS else _safe(value)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "unknown"
        return json.dumps(payload, default=str)


def _safe(value: object) -> object:
    """Keep log values JSON-simple; stringify anything exotic."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger for JSON output on stdout. Idempotent."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
