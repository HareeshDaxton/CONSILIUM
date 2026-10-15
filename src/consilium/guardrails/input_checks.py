"""Deterministic input checks on the RAW clinical note (AGENTS.md §10 step 1, §14; I-6/I-7).

The canonical first layer of the input rail. Runs FIRST in note intake —
before de-identification, before any LLM call (and before the model-based
NeMo rail, which only ever sees the REDACTED note; see ``nemo_wiring``).
A block raises ``PolicyViolation(case_status=NEEDS_ATTENTION)`` carrying the
rule NAME only, never the matched content (I-6/I-7, fail closed I-5). The
status follows the AGENTS.md §4 state machine (``INPUT_RAIL_PASSED`` fail →
``NEEDS_ATTENTION``).

Checks (config-driven, ``configs/guardrails/policies/input_rail.yaml``):
  1. length cap — reject, never truncate (no silent data loss),
  2. control-character fraction — corrupt/adversarial byte soup,
  3. charset allowlist — private-use / surrogate / unassigned code points and
     bidi-override controls (Trojan-Source style reordering) are refused,
  4. PHI pattern detection — SHARED config with de-identification
     (``phi_patterns.yaml``); per-config action ``redact`` (report the
     pattern names, de-identification redacts next) or ``block``,
  5. instruction-like patterns — matched on ``normalize.normalize_for_matching``
     text (NFKC, invisible chars stripped, Cyrillic/Greek lookalikes folded,
     whitespace collapsed) so obfuscation cannot smuggle instructions past.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator

from consilium.core.errors import PolicyViolation
from consilium.guardrails.normalize import normalize_for_matching
from consilium.intake.deidentify import PhiPolicy
from consilium.schemas.base import Strict

DEFAULT_RAIL_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "guardrails" / "policies" / "input_rail.yaml"
)

_NEEDS_ATTENTION = "NEEDS_ATTENTION"


def _list_to_tuple(v: object) -> object:
    return tuple(v) if isinstance(v, list) else v


class RailLimits(Strict):
    max_chars: int = Field(gt=0)
    max_control_char_fraction: float = Field(ge=0.0, le=1.0)


class CharsetPolicy(Strict):
    disallowed_categories: tuple[str, ...]  # Unicode general categories, e.g. "Co"
    disallowed_codepoints: tuple[str, ...]  # hex, e.g. "202E"

    @field_validator("disallowed_categories", "disallowed_codepoints", mode="before")
    @classmethod
    def _to_tuple(cls, v: object) -> object:
        return _list_to_tuple(v)

    def codepoints(self) -> frozenset[str]:
        return frozenset(chr(int(h, 16)) for h in self.disallowed_codepoints)


class PhiRailPolicy(Strict):
    on_match: Literal["redact", "block"]


class RailPattern(Strict):
    name: str
    pattern: str


class InputRailConfig(Strict):
    version: int
    limits: RailLimits
    charset: CharsetPolicy
    phi: PhiRailPolicy
    patterns: tuple[RailPattern, ...]

    @field_validator("patterns", mode="before")
    @classmethod
    def _patterns_to_tuple(cls, v: object) -> object:
        return _list_to_tuple(v)

    def compiled(self) -> tuple[tuple[RailPattern, re.Pattern[str]], ...]:
        return tuple((p, re.compile(p.pattern, re.IGNORECASE)) for p in self.patterns)


@dataclass(frozen=True)
class InputCheckResult:
    """Outcome of a PASSING check: PHI pattern NAMES detected (never values)."""

    phi_patterns: tuple[str, ...] = ()


def load_input_rail(path: Path = DEFAULT_RAIL_PATH) -> InputRailConfig:
    """Load and validate the rail config. Typos fail loudly (extra=forbid)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return InputRailConfig.model_validate(raw)


def _control_char_fraction(text: str) -> float:
    if not text:
        return 0.0
    controls = sum(1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r")
    return controls / len(text)


def _charset_violation(text: str, charset: CharsetPolicy) -> bool:
    blocked = charset.codepoints()
    cats = frozenset(charset.disallowed_categories)
    return any(ch in blocked or unicodedata.category(ch) in cats for ch in text)


def _block(reason: str, detail: str) -> PolicyViolation:
    return PolicyViolation(reason, case_status=_NEEDS_ATTENTION, detail=detail)


def check_note(
    text: str, cfg: InputRailConfig, phi_policy: PhiPolicy | None = None
) -> InputCheckResult:
    """Apply every rail rule; raise PolicyViolation on the first block.

    Reasons are machine strings; details carry rule/pattern NAMES only.
    PHI detection runs only when ``phi_policy`` is supplied (the graph always
    supplies it; it is the same policy de-identification uses).
    """
    if len(text) > cfg.limits.max_chars:
        raise _block("NOTE_TOO_LONG", f"input_rail.length: {len(text)} chars > max_chars")
    if _control_char_fraction(text) > cfg.limits.max_control_char_fraction:
        raise _block("NOTE_CONTROL_CHARS", "input_rail.control_chars")
    if _charset_violation(text, cfg.charset):
        raise _block("NOTE_CHARSET", "input_rail.charset")
    normalized = normalize_for_matching(text)
    for pattern, rx in cfg.compiled():
        if rx.search(normalized) is not None:
            raise _block("NOTE_INSTRUCTION_LIKE", f"input_rail.pattern:{pattern.name}")
    phi_hits: tuple[str, ...] = ()
    if phi_policy is not None:
        phi_hits = tuple(p.name for p, rx in phi_policy.compiled() if rx.search(text) is not None)
        if phi_hits and cfg.phi.on_match == "block":
            raise _block("NOTE_PHI", "input_rail.phi:" + ",".join(phi_hits))
    return InputCheckResult(phi_patterns=phi_hits)
