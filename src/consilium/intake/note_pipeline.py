"""Note intake pipeline (AGENTS.md §10) — the FIXED order, never reordered:

    raw note → input rail → de-identify → LLM extract → validate → provenance

The LLM never sees unredacted text (I-6); the raw note never touches an LLM
before the rail passes (I-7). Every failure fails closed:

  - rail block            → PolicyViolation  → NEEDS_ATTENTION (§4)
  - LLM unreachable       → AgentFailure     → FAILED
  - invalid output (1+2 tries) → ValidationFailure → FAILED

Provenance (§10.5): model snapshot, prompt version, redaction pattern NAMES
(never values), and deterministic extraction confidence are assembled into
``NoteExtraction`` on every run. LOW confidence surfaces the
``note.caveat.extraction`` fact downstream in the Context Builder (P4) —
see ``context.facts.caveat_for_extraction``.
"""

from consilium.core.settings import DEFAULT_MODEL_SNAPSHOT
from consilium.intake.deidentify import PhiPolicy, load_phi_patterns, redact
from consilium.intake.extractor import (
    PROMPT_VERSION,
    estimate_confidence,
    extract_note,
    load_extractor_prompt,
)
from consilium.intake.input_rail import InputRailConfig, check_note, load_input_rail
from consilium.llm.client import LLMClient, LLMRetryPolicy
from consilium.schemas.note import NoteExtraction


async def run(
    raw_text: str,
    *,
    llm: LLMClient,
    rail: InputRailConfig | None = None,
    phi_policy: PhiPolicy | None = None,
    prompt: str | None = None,
    model: str = DEFAULT_MODEL_SNAPSHOT,
    temperature: float = 0.0,
    timeout_s: float = 60.0,
    max_repairs: int = 2,
    retry_policy: LLMRetryPolicy | None = None,
    fixture: str | None = None,
) -> NoteExtraction:
    """Full intake for one raw note. Configs default-load from configs/."""
    rail_cfg = rail if rail is not None else load_input_rail()
    policy = phi_policy if phi_policy is not None else load_phi_patterns()
    system_prompt = prompt if prompt is not None else load_extractor_prompt()

    check_note(raw_text, rail_cfg)  # 1. rail BEFORE anything else
    redaction = redact(raw_text, policy)  # 2. de-identify BEFORE the LLM
    resp = await extract_note(  # 3. extract + validate (+ ≤2 repairs)
        redaction.text,
        llm=llm,
        prompt=system_prompt,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_repairs=max_repairs,
        retry_policy=retry_policy,
        fixture=fixture,
    )
    return NoteExtraction(  # 4. provenance on every run (§10.5)
        note=resp.data,
        extractor_model=model,
        extractor_prompt_version=PROMPT_VERSION,
        redactions=redaction.pattern_names,
        extraction_confidence=estimate_confidence(resp.data),
    )
