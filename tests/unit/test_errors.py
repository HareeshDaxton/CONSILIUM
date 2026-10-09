"""Unit tests for core/errors.py — the error → case-status / problem-type map."""

import pytest

from consilium.core.errors import (
    AgentFailure,
    AppError,
    InputRejected,
    PolicyViolation,
    ValidationFailure,
    VisionFailure,
)


class TestMapping:
    """AGENTS.md §21: each error maps to a case status and a problem type."""

    @pytest.mark.parametrize(
        ("exc", "status", "problem"),
        [
            (InputRejected("NOT_A_FUNDUS_IMAGE"), "REJECTED_INPUT", "input-rejected"),
            (VisionFailure("VISION_UNAVAILABLE"), "FAILED", "unprocessable"),
            (AgentFailure("specialist_retries_exhausted"), "FAILED", "unprocessable"),
            (ValidationFailure("subreport_schema"), "FAILED", "validation-failed"),
            (PolicyViolation("prompt_injection"), "REJECTED_INPUT", "input-rejected"),
            (AppError("boom"), "FAILED", "internal"),
        ],
    )
    def test_mapping(self, exc: AppError, status: str, problem: str) -> None:
        assert exc.case_status == status
        assert exc.problem_type == problem

    def test_reason_and_detail(self) -> None:
        err = VisionFailure("OPTIC_DISC_NOT_VISIBLE", detail="qc: disc area 0px")
        assert err.reason == "OPTIC_DISC_NOT_VISIBLE"
        assert err.detail == "qc: disc area 0px"
        assert str(err) == "OPTIC_DISC_NOT_VISIBLE"


class TestPolicyViolationStatus:
    def test_default_is_rejected_input(self) -> None:
        assert PolicyViolation("phi_detected").case_status == "REJECTED_INPUT"

    def test_post_extraction_is_needs_attention(self) -> None:
        err = PolicyViolation("output_rail_dose_pattern", case_status="NEEDS_ATTENTION")
        assert err.case_status == "NEEDS_ATTENTION"

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid case_status"):
            PolicyViolation("x", case_status="APPROVED")  # I-4 can never come from a rail
