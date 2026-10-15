"""Compatibility re-export — the deterministic input rail now lives in
``consilium.guardrails.input_checks`` (Phase 10.1, the single source).

Kept so intake-side callers (``note_pipeline``) and existing tests keep one
import path; do NOT add logic here.
"""

from consilium.guardrails.input_checks import (
    DEFAULT_RAIL_PATH,
    CharsetPolicy,
    InputCheckResult,
    InputRailConfig,
    PhiRailPolicy,
    RailLimits,
    RailPattern,
    check_note,
    load_input_rail,
)

__all__ = [
    "DEFAULT_RAIL_PATH",
    "CharsetPolicy",
    "InputCheckResult",
    "InputRailConfig",
    "PhiRailPolicy",
    "RailLimits",
    "RailPattern",
    "check_note",
    "load_input_rail",
]
