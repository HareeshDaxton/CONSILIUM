"""Shared enumerations (AGENTS.md §7 + §4).

Enum VALUES are persisted API (database rows, stored JSON, snapshots):
renaming a value silently breaks stored data — add, never rename
(schema-change skill, doc/SKILLS.md §2).
"""

from enum import StrEnum


class Laterality(StrEnum):
    OD = "OD"
    OS = "OS"
    UNKNOWN = "UNKNOWN"


class Role(StrEnum):
    OPHTHALMOLOGIST = "ophthalmologist"
    OPTOMETRIST = "optometrist"
    PHARMACIST = "pharmacist"


class AgentName(StrEnum):
    """Identity for agent_runs / prompt versioning."""

    NOTE_EXTRACTOR = "note_extractor"
    OPHTHALMOLOGIST = "ophthalmologist"
    OPTOMETRIST = "optometrist"
    PHARMACIST = "pharmacist"
    DIRECTOR = "director"
    CRITIC = "critic"


class ScreeningTier(StrEnum):
    """Derived from CALIBRATED probability via configs/thresholds.yaml."""

    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"


class QualityFlag(StrEnum):
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    BLURRED = "blurred"
    DISC_NOT_VISIBLE = "disc_not_visible"
    NOT_FUNDUS = "not_fundus"
    LOW_RESOLUTION = "low_resolution"


class FactSource(StrEnum):
    CAD = "cad"
    NOTE = "note"
    DERIVED = "derived"


class Confidence(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ClaimKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    INTERPRETIVE = "interpretive"
    RECOMMENDATION = "recommendation"


class IssueType(StrEnum):
    UNSUPPORTED = "unsupported"
    NUMERIC_MISMATCH = "numeric_mismatch"
    OMITTED_FINDING = "omitted_finding"
    OUT_OF_SCOPE = "out_of_scope"
    UNSAFE_LANGUAGE = "unsafe_language"
    DISAGREEMENT_SUPPRESSED = "disagreement_suppressed"
    BAD_REFERENCE = "bad_reference"


class CaseStatus(StrEnum):
    """Pipeline state machine (AGENTS.md §4).

    Values match the status strings used by core/errors.py. Transition rules
    and persistence live in the graph (P9); this enum is the shared vocabulary.
    """

    RECEIVED = "RECEIVED"
    IMAGE_PRECHECKED = "IMAGE_PRECHECKED"
    REJECTED_INPUT = "REJECTED_INPUT"
    INPUT_RAIL_PASSED = "INPUT_RAIL_PASSED"
    NOTE_EXTRACTED = "NOTE_EXTRACTED"
    VISION_CALLED = "VISION_CALLED"
    CAD_COMPLETE = "CAD_COMPLETE"
    CONTEXT_BUILT = "CONTEXT_BUILT"
    ROUTED = "ROUTED"
    SPECIALISTS_DONE = "SPECIALISTS_DONE"
    DRAFTED = "DRAFTED"
    AUDITING = "AUDITING"
    REVISING = "REVISING"
    OUTPUT_RAIL_PASSED = "OUTPUT_RAIL_PASSED"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    APPROVED_WITH_EDITS = "APPROVED_WITH_EDITS"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
