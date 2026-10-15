"""Deterministic output checks on the DirectorDraft (AGENTS.md §14; I-6/I-12).

Runs in the ``output_rail`` node BEFORE the model-based NeMo output rail and
before any clinician sees the draft. A trigger raises
``PolicyViolation(case_status=NEEDS_ATTENTION)`` with the rule NAME only (the
runner writes a ``guardrail.triggered`` audit event).

Checks, over EVERY text field of the draft (impression, claims,
recommendations, limitations, disagreement topics/positions/resolutions):
  1. forbidden language + dose patterns — ``verify.lexicon.scan_text`` (the
     single source; nothing re-declared here). Stage/severity wording passes
     only when the note supplies it (``verify.deterministic.note_supplies_stage``),
  2. PHI echo — the shared ``phi_patterns.yaml`` de-identification policy,
  3. mandatory screening-support disclaimer in ``limitations``.

Matching runs on ``normalize.normalize_for_matching`` text so homoglyph,
zero-width, full-width and spacing obfuscations cannot slip past. This is a
POLICY layer: it never changes the draft's shape or values.
"""

import re
from pathlib import Path

import yaml
from pydantic import field_validator

from consilium.core.errors import PolicyViolation
from consilium.guardrails.normalize import normalize_for_matching
from consilium.intake.deidentify import PhiPolicy
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.base import Strict
from consilium.schemas.context import GlobalContext
from consilium.verify.deterministic import note_supplies_stage
from consilium.verify.lexicon import LexiconCategory, scan_text

DEFAULT_OUTPUT_RAIL_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "guardrails" / "policies" / "output_rail.yaml"
)

_REASONS: dict[LexiconCategory, str] = {
    LexiconCategory.DEFINITIVE_DIAGNOSIS: "DRAFT_DEFINITIVE_DIAGNOSIS",
    LexiconCategory.DOSE_OR_REGIMEN: "DRAFT_DOSE_OR_REGIMEN",
    LexiconCategory.STAGE_SEVERITY: "DRAFT_STAGE_SEVERITY",
}


class DisclaimerPolicy(Strict):
    any_of: tuple[str, ...]

    @field_validator("any_of", mode="before")
    @classmethod
    def _to_tuple(cls, v: object) -> object:
        return tuple(v) if isinstance(v, list) else v


class OutputRailConfig(Strict):
    version: int
    disclaimer: DisclaimerPolicy

    def disclaimer_patterns(self) -> tuple[re.Pattern[str], ...]:
        return tuple(re.compile(p, re.IGNORECASE) for p in self.disclaimer.any_of)


def load_output_rail(path: Path = DEFAULT_OUTPUT_RAIL_PATH) -> OutputRailConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OutputRailConfig.model_validate(raw)


def draft_texts(draft: DirectorDraft) -> tuple[str, ...]:
    """Every human-visible text field of the draft, in display order."""
    texts: list[str] = [draft.impression]
    texts.extend(c.text for c in draft.claims)
    texts.extend(r.text for r in draft.recommendations)
    texts.extend(draft.limitations)
    for d in draft.disagreements:
        texts.append(d.topic)
        texts.extend(pos for _, pos in d.positions)
        texts.append(d.resolution)
    return tuple(texts)


def _block(reason: str, rule: str) -> PolicyViolation:
    return PolicyViolation(reason, case_status="NEEDS_ATTENTION", detail=f"output_rail.{rule}")


def check_draft(
    draft: DirectorDraft,
    global_ctx: GlobalContext | None,
    cfg: OutputRailConfig,
    phi_policy: PhiPolicy,
) -> None:
    """Raise PolicyViolation on the first policy breach; return None if clean."""
    stage_ok = global_ctx is not None and note_supplies_stage(global_ctx)
    normalized = tuple(normalize_for_matching(t) for t in draft_texts(draft))
    for text in normalized:
        for hit in scan_text(text):
            if hit.category is LexiconCategory.STAGE_SEVERITY and stage_ok:
                continue
            raise _block(_REASONS[hit.category], f"lexicon:{hit.category.value}")
    for text in normalized:
        for pattern, rx in phi_policy.compiled():
            if rx.search(text) is not None:
                raise _block("DRAFT_PHI_ECHO", f"phi:{pattern.name}")
    patterns = cfg.disclaimer_patterns()
    limitations = (normalize_for_matching(t) for t in draft.limitations)
    if not any(rx.search(t) for t in limitations for rx in patterns):
        raise _block("DRAFT_MISSING_DISCLAIMER", "disclaimer")
