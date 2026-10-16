"""Closed reason-code taxonomy for clinician edits/rejections (AGENTS.md §15).

Extendable ONLY via the ``schema-change`` skill: these codes are the V2
dataset (the RAG decision gate, §23, is evaluated on them). Making them
optional "to reduce friction" destroys that dataset — they are mandatory for
APPROVE_WITH_EDITS and REJECT, and ``OTHER`` requires free text.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["KNOWLEDGE_CODES", "REASON_CODE_DESCRIPTIONS", "ReasonCode"]


class ReasonCode(StrEnum):
    FACTUAL_ERROR_CAD = "FACTUAL_ERROR_CAD"
    CLINICAL_INTERPRETATION = "CLINICAL_INTERPRETATION"
    KNOWLEDGE_GAP = "KNOWLEDGE_GAP"
    UNSAFE_RECOMMENDATION = "UNSAFE_RECOMMENDATION"
    OMISSION = "OMISSION"
    TONE_OR_WORDING = "TONE_OR_WORDING"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    OTHER = "OTHER"


REASON_CODE_DESCRIPTIONS: dict[ReasonCode, str] = {
    ReasonCode.FACTUAL_ERROR_CAD: "Misread or misstated a CAD value",
    ReasonCode.CLINICAL_INTERPRETATION: "Wrong clinical reading of correct facts",
    ReasonCode.KNOWLEDGE_GAP: "Missing or wrong domain knowledge",
    ReasonCode.UNSAFE_RECOMMENDATION: "Unsafe recommendation",
    ReasonCode.OMISSION: "Something material was omitted",
    ReasonCode.TONE_OR_WORDING: "Tone or wording",
    ReasonCode.SCOPE_VIOLATION: "A role went beyond its scope",
    ReasonCode.OTHER: "Other (free-text reason required)",
}

# The V2 gate (§23) watches the share of these among edits/rejections.
KNOWLEDGE_CODES: frozenset[ReasonCode] = frozenset(
    {ReasonCode.KNOWLEDGE_GAP, ReasonCode.CLINICAL_INTERPRETATION}
)
