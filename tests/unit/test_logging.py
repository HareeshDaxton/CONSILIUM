"""Unit tests for core/logging.py — JSON output + PHI key redaction (I-6/I-7)."""

import io
import json
import logging

import pytest

from consilium.core.logging import DENY_KEYS, REDACTED, JsonFormatter, get_logger, setup_logging


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def logger(stream: io.StringIO) -> logging.Logger:
    lg = get_logger("consilium.test")
    lg.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    lg.addHandler(handler)
    lg.setLevel("DEBUG")
    lg.propagate = False
    return lg


def read_lines(stream: io.StringIO) -> list[dict[str, object]]:
    stream.seek(0)
    return [json.loads(line) for line in stream.read().strip().splitlines()]


class TestJsonOutput:
    def test_emits_valid_json_with_core_fields(
        self, logger: logging.Logger, stream: io.StringIO
    ) -> None:
        logger.info("node complete", extra={"case_id": "abc-123", "node": "call_vision"})
        (record,) = read_lines(stream)
        assert record["level"] == "INFO"
        assert record["logger"] == "consilium.test"
        assert record["msg"] == "node complete"
        assert record["case_id"] == "abc-123"
        assert record["node"] == "call_vision"
        assert "ts" in record

    def test_exception_type_included(self, logger: logging.Logger, stream: io.StringIO) -> None:
        try:
            raise ValueError("synthetic")
        except ValueError:
            logger.exception("failed")
        (record,) = read_lines(stream)
        assert record["exc_type"] == "ValueError"

    def test_exotic_values_stringified(self, logger: logging.Logger, stream: io.StringIO) -> None:
        logger.info("x", extra={"attempts": (1, 2), "meta": {"k": object()}})
        (record,) = read_lines(stream)  # must not raise on json.dumps
        assert record["attempts"] == [1, 2]


class TestPhiRedaction:
    """SKILLS.md phi-review: denied keys are redacted, never emitted raw."""

    @pytest.mark.parametrize("key", sorted(DENY_KEYS - {"filename"}))
    def test_deny_keys_redacted(
        self, logger: logging.Logger, stream: io.StringIO, key: str
    ) -> None:
        logger.info("x", extra={key: "SENSITIVE-VALUE"})
        (record,) = read_lines(stream)
        assert record[key] == REDACTED
        assert "SENSITIVE-VALUE" not in stream.getvalue()

    def test_filename_extra_blocked_by_stdlib(self, logger: logging.Logger) -> None:
        # "filename" is a reserved LogRecord attribute: stdlib logging refuses
        # extra={"filename": ...} with KeyError. Fail-loud is acceptable — the
        # PHI-safe path is upload_filename (redacted above).
        with pytest.raises(KeyError):
            logger.info("x", extra={"filename": "smith_mrn123.jpg"})

    def test_case_insensitive_key_match(self, logger: logging.Logger, stream: io.StringIO) -> None:
        logger.info("x", extra={"FileName": "smith_john_mrn123.jpg"})
        (record,) = read_lines(stream)
        assert record["FileName"] == REDACTED

    def test_allowed_keys_pass_through(self, logger: logging.Logger, stream: io.StringIO) -> None:
        logger.info("x", extra={"latency_ms": 812, "endpoint_revision": "sha256:abc/run-1"})
        (record,) = read_lines(stream)
        assert record["latency_ms"] == 812
        assert record["endpoint_revision"] == "sha256:abc/run-1"


class TestSetup:
    def test_setup_logging_is_idempotent(self) -> None:
        setup_logging("DEBUG")
        setup_logging("INFO")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.level == logging.INFO
