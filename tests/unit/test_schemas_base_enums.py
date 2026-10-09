"""Section 1.1 — Strict base + enums (AGENTS.md §7, §4)."""

import pytest
from pydantic import ValidationError

from consilium.schemas.base import Strict
from consilium.schemas.enums import (
    AgentName,
    CaseStatus,
    ClaimKind,
    Confidence,
    FactSource,
    IssueType,
    Laterality,
    QualityFlag,
    Role,
    ScreeningTier,
)


class _Probe(Strict):
    count: int
    label: str


class TestStrictBase:
    def test_valid_construction(self) -> None:
        probe = _Probe(count=1, label="a")
        assert probe.count == 1

    def test_no_coercion_in_strict_mode(self) -> None:
        with pytest.raises(ValidationError):
            _Probe(count="1", label="a")  # type: ignore[arg-type]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _Probe(count=1, label="a", bogus=True)  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        probe = _Probe(count=1, label="a")
        with pytest.raises(ValidationError):
            probe.count = 2  # type: ignore[misc]

    def test_round_trip(self) -> None:
        probe = _Probe(count=1, label="a")
        assert _Probe.model_validate_json(probe.model_dump_json()) == probe


class TestEnums:
    def test_laterality_values(self) -> None:
        assert {m.value for m in Laterality} == {"OD", "OS", "UNKNOWN"}

    def test_roles(self) -> None:
        assert {m.value for m in Role} == {"ophthalmologist", "optometrist", "pharmacist"}

    def test_agent_names_cover_all_agents(self) -> None:
        assert {m.value for m in AgentName} == {
            "note_extractor",
            "ophthalmologist",
            "optometrist",
            "pharmacist",
            "director",
            "critic",
        }

    def test_screening_tiers(self) -> None:
        assert [m.value for m in ScreeningTier] == ["low", "intermediate", "high"]

    def test_quality_flags(self) -> None:
        assert {m.value for m in QualityFlag} == {
            "underexposed",
            "overexposed",
            "blurred",
            "disc_not_visible",
            "not_fundus",
            "low_resolution",
        }

    def test_fact_sources(self) -> None:
        assert {m.value for m in FactSource} == {"cad", "note", "derived"}

    def test_confidence(self) -> None:
        assert [m.value for m in Confidence] == ["low", "moderate", "high"]

    def test_claim_kinds(self) -> None:
        assert {m.value for m in ClaimKind} == {
            "numeric",
            "categorical",
            "interpretive",
            "recommendation",
        }

    def test_issue_types(self) -> None:
        assert {m.value for m in IssueType} == {
            "unsupported",
            "numeric_mismatch",
            "omitted_finding",
            "out_of_scope",
            "unsafe_language",
            "disagreement_suppressed",
            "bad_reference",
        }

    def test_case_status_covers_state_machine(self) -> None:
        """Every status name in AGENTS.md §4 / core/errors.py exists."""
        assert {m.value for m in CaseStatus} == {
            "RECEIVED",
            "IMAGE_PRECHECKED",
            "REJECTED_INPUT",
            "INPUT_RAIL_PASSED",
            "NOTE_EXTRACTED",
            "VISION_CALLED",
            "CAD_COMPLETE",
            "CONTEXT_BUILT",
            "ROUTED",
            "SPECIALISTS_DONE",
            "DRAFTED",
            "AUDITING",
            "REVISING",
            "OUTPUT_RAIL_PASSED",
            "PENDING_REVIEW",
            "APPROVED",
            "APPROVED_WITH_EDITS",
            "REJECTED",
            "FAILED",
            "NEEDS_ATTENTION",
        }

    def test_case_status_matches_error_mapping(self) -> None:
        """core/errors.py case_status strings must be valid CaseStatus values."""
        from consilium.core.errors import (
            AgentFailure,
            InputRejected,
            ValidationFailure,
            VisionFailure,
        )

        for cls in (InputRejected, VisionFailure, AgentFailure, ValidationFailure):
            CaseStatus(cls.case_status)  # raises if not a member

    def test_enums_serialize_as_plain_strings(self) -> None:
        assert Laterality.OD == "OD"
        assert Role.PHARMACIST.value == "pharmacist"
