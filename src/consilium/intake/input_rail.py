"""Deterministic input rail on the raw clinical note (AGENTS.md §10 step 1, §14).

Runs FIRST in note intake — before de-identification, before any LLM call.
A block raises ``PolicyViolation(case_status=NEEDS_ATTENTION)`` carrying the
rule NAME only, never the matched content (I-6/I-7, fail closed I-5).

Checks (config-driven, ``configs/guardrails/policies/input_rail.yaml``):
  1. length cap — reject, never truncate (no silent data loss),
  2. control-character fraction — corrupt/adversarial charsets,
  3. instruction-like patterns — matched on the NFKC-normalized,
     zero-width-stripped text so homoglyph/invisible-char tricks cannot
     smuggle instructions past the rail.

NeMo Guardrails (model-based detection) layers on top in P10; this module
stays the cheap, exact, offline-testable first layer.
"""

import re
import unicodedata
from pathlib import Path

import yaml
from pydantic import Field, field_validator

from consilium.core.errors import PolicyViolation
from consilium.schemas.base import Strict

DEFAULT_RAIL_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "guardrails" / "policies" / "input_rail.yaml"
)

# zero-width space / non-joiner / joiner / word joiner / BOM
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


class RailLimits(Strict):
    max_chars: int = Field(gt=0)
    max_control_char_fraction: float = Field(ge=0.0, le=1.0)


class RailPattern(Strict):
    name: str
    pattern: str


class InputRailConfig(Strict):
    version: int
    limits: RailLimits
    patterns: tuple[RailPattern, ...]

    @field_validator("patterns", mode="before")
    @classmethod
    def _list_to_tuple(cls, v: object) -> object:
        return tuple(v) if isinstance(v, list) else v

    def compiled(self) -> tuple[tuple[RailPattern, re.Pattern[str]], ...]:
        return tuple((p, re.compile(p.pattern, re.IGNORECASE)) for p in self.patterns)


def load_input_rail(path: Path = DEFAULT_RAIL_PATH) -> InputRailConfig:
    """Load and validate the rail config. Typos fail loudly (extra=forbid)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return InputRailConfig.model_validate(raw)


def _normalize_for_matching(text: str) -> str:
    """NFKC-normalize (full-width homoglyphs → ASCII) and strip zero-width
    characters. Used for MATCHING only — the stored/extracted text is the
    original."""
    return _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))


def _control_char_fraction(text: str) -> float:
    if not text:
        return 0.0
    controls = sum(1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r")
    return controls / len(text)


def check_note(text: str, cfg: InputRailConfig) -> None:
    """Apply every rail rule; raise PolicyViolation on the first block.

    Reasons are machine strings; details carry rule/pattern NAMES only.
    """
    if len(text) > cfg.limits.max_chars:
        raise PolicyViolation(
            "NOTE_TOO_LONG",
            case_status="NEEDS_ATTENTION",
            detail=f"input_rail.length: {len(text)} chars > max_chars",
        )
    if _control_char_fraction(text) > cfg.limits.max_control_char_fraction:
        raise PolicyViolation(
            "NOTE_CONTROL_CHARS",
            case_status="NEEDS_ATTENTION",
            detail="input_rail.control_chars",
        )
    normalized = _normalize_for_matching(text)
    for pattern, rx in cfg.compiled():
        if rx.search(normalized) is not None:
            raise PolicyViolation(
                "NOTE_INSTRUCTION_LIKE",
                case_status="NEEDS_ATTENTION",
                detail=f"input_rail.pattern:{pattern.name}",
            )
