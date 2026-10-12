"""Forbidden-language lexicon — the single source of truth for unsafe wording.

AGENTS.md §13.1 + I-12: screening-support reports never state a definitive
diagnosis, never give drug doses or regimens, and never state glaucoma
stage/severity unless a clinician-provided note supplies it. The patterns
live here and ONLY here: the deterministic verifier (P7) and the NeMo output
rail (P10) must both import from this module instead of re-declaring rules.

`scan_text` is deliberately context-free: it reports candidate hits. The
note-support gate for STAGE_SEVERITY ("allowed when the clinical note states
it") is applied by the verifier check, which owns the GlobalContext.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class LexiconCategory(StrEnum):
    DEFINITIVE_DIAGNOSIS = "definitive_diagnosis"
    DOSE_OR_REGIMEN = "dose_or_regimen"
    STAGE_SEVERITY = "stage_severity"


@dataclass(frozen=True)
class LexiconHit:
    category: LexiconCategory
    matched: str
    start: int
    end: int


# I-12 wording contract: the impression uses "findings are consistent with /
# raise suspicion of / warrant evaluation for" — never "patient has". Phrases
# are matched case-insensitively as substrings; keep them specific enough that
# allowed screening wording cannot trip them.
DEFINITIVE_DIAGNOSIS_PHRASES: tuple[str, ...] = (
    "has glaucoma",
    "diagnosed with",
    "diagnosis is",
    "confirms glaucoma",
    "confirms the diagnosis",
    "is glaucomatous",
    "definitive diagnosis",
)

# Dose/regimen content (AGENTS.md §13.1: `\d+\s?(mg|mcg|%)`). Eye-drop
# concentrations are doses, hence "%". No trailing word boundary: "%" is a
# non-word char, so "\b" after it would never match. Benign numerals such as
# "2-4 months" (en-dash) or "0.62" do not match — they carry no dose unit.
DOSE_PATTERN: re.Pattern[str] = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|%)", re.IGNORECASE)

# Glaucoma stage/severity claims. Requires glaucoma/stage context so that
# benign uses like "moderate confidence" never match. These are CANDIDATE
# hits: the verifier suppresses them only when the clinical note itself
# supplies the stage/severity (I-12).
STAGE_SEVERITY_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:mild|moderate|severe|advanced|early|end)[- ]stage\b"
    r"|\b(?:mild|moderate|severe|advanced)\s+glaucoma\b"
    r"|\bglaucoma\s+(?:stage|severity)\b"
    r"|\bstage\s+(?:[ivx]+|\d+)\b",
    re.IGNORECASE,
)


def scan_text(text: str) -> tuple[LexiconHit, ...]:
    """Return all forbidden-language candidate hits in `text`, ordered by position."""
    hits: list[LexiconHit] = []
    lowered = text.lower()
    for phrase in DEFINITIVE_DIAGNOSIS_PHRASES:
        idx = 0
        while (found := lowered.find(phrase, idx)) != -1:
            end = found + len(phrase)
            hits.append(
                LexiconHit(LexiconCategory.DEFINITIVE_DIAGNOSIS, text[found:end], found, end)
            )
            idx = end
    for match in DOSE_PATTERN.finditer(text):
        hits.append(
            LexiconHit(LexiconCategory.DOSE_OR_REGIMEN, match.group(), match.start(), match.end())
        )
    for match in STAGE_SEVERITY_PATTERN.finditer(text):
        hits.append(
            LexiconHit(LexiconCategory.STAGE_SEVERITY, match.group(), match.start(), match.end())
        )
    hits.sort(key=lambda hit: hit.start)
    return tuple(hits)
