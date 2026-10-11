"""PHI de-identification (AGENTS.md §10 step 2, I-6; SKILLS.md note-extraction).

Runs on the raw note BEFORE any LLM call, after the deterministic input rail.
Patterns live in ``configs/guardrails/policies/phi_patterns.yaml`` — never in
code — and the ONLY thing recorded about a match is the pattern NAME
(``NoteExtraction.redactions``). Matched values are replaced with
``[REDACTED:<name>]`` and never stored, logged, or returned (I-6).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import field_validator

from consilium.schemas.base import Strict

DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "guardrails"
    / "policies"
    / "phi_patterns.yaml"
)

_FLAG_MAP = {"IGNORECASE": re.IGNORECASE, "MULTILINE": re.MULTILINE}


class PhiPattern(Strict):
    name: str
    pattern: str
    action: Literal["redact"] = "redact"
    flags: tuple[Literal["IGNORECASE", "MULTILINE"], ...] = ()

    @field_validator("flags", mode="before")
    @classmethod
    def _list_to_tuple(cls, v: object) -> object:
        return tuple(v) if isinstance(v, list) else v


class PhiPolicy(Strict):
    version: int
    patterns: tuple[PhiPattern, ...]

    @field_validator("patterns", mode="before")
    @classmethod
    def _list_to_tuple(cls, v: object) -> object:
        return tuple(v) if isinstance(v, list) else v

    def compiled(self) -> tuple[tuple[PhiPattern, re.Pattern[str]], ...]:
        out: list[tuple[PhiPattern, re.Pattern[str]]] = []
        for p in self.patterns:
            re_flags = 0
            for f in p.flags:
                re_flags |= _FLAG_MAP[f]
            out.append((p, re.compile(p.pattern, re_flags)))
        return tuple(out)


def load_phi_patterns(path: Path = DEFAULT_POLICY_PATH) -> PhiPolicy:
    """Load and validate the PHI pattern policy. Typos fail loudly (extra=forbid)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PhiPolicy.model_validate(raw)


@dataclass(frozen=True)
class RedactionResult:
    """Redacted text + the NAMES of the patterns that fired (never values)."""

    text: str
    pattern_names: tuple[str, ...]


def redact(text: str, policy: PhiPolicy) -> RedactionResult:
    """Apply every pattern in order; return redacted text and fired pattern names.

    Pattern names appear at most once, in first-firing order. The redaction
    placeholder embeds only the pattern name — never any part of the match.
    """
    fired: list[str] = []
    redacted = text
    for pattern, rx in policy.compiled():
        if rx.search(redacted) is None:
            continue
        fired.append(pattern.name)
        redacted = rx.sub(f"[REDACTED:{pattern.name}]", redacted)
    return RedactionResult(text=redacted, pattern_names=tuple(fired))
