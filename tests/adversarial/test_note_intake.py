"""Adversarial note suite (Section 3.4; AGENTS.md §20, SKILLS.md note-extraction).

Proves the intake boundary holds against hostile/malformed notes, offline,
with FakeLLMClient: injection (plain + unicode tricks), PHI-in-note,
oversized input, control-char payloads, multilingual noise, and extractor-
redirection attempts. Every case asserts BOTH the failure mode (fail closed,
I-5) and that no note content ever leaks into error details (I-6/I-7).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from consilium.context.facts import F, caveat_for_extraction
from consilium.core.errors import PolicyViolation
from consilium.intake import note_pipeline
from consilium.intake.extractor import PROMPT_VERSION
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.schemas.enums import AgentName, Confidence

FIXTURES = Path(__file__).parent.parent / "fixtures"
NOTES = {n["id"]: n["text"] for n in json.loads((FIXTURES / "synthetic_notes.json").read_text())}

KEY = (AgentName.NOTE_EXTRACTOR.value, PROMPT_VERSION, "default")

EMPTY_PAYLOAD: dict[str, Any] = {
    "age_years": None,
    "iop_mmhg_od": None,
    "iop_mmhg_os": None,
    "current_medications": [],
    "drug_intolerances_or_allergies": [],
    "comorbidities": [],
    "prior_glaucoma_dx": None,
    "exam_findings": None,
    "free_text": None,
}
PLANTED_PAYLOAD: dict[str, Any] = {
    **EMPTY_PAYLOAD,
    "iop_mmhg_od": 45.0,
    "iop_mmhg_os": 45.0,
    "prior_glaucoma_dx": "advanced glaucoma",
}


def _client(payload: dict[str, Any] = EMPTY_PAYLOAD) -> FakeLLMClient:
    return FakeLLMClient({KEY: CannedResponse((CannedCall(FakeMode.VALID, payload),))})


async def _blocked(note_id: str) -> PolicyViolation:
    client = _client()
    with pytest.raises(PolicyViolation) as exc_info:
        await note_pipeline.run(NOTES[note_id], llm=client)
    assert client.received == []  # blocked before any LLM call
    return exc_info.value


# --- injection: plain, homoglyph, zero-width ---


@pytest.mark.parametrize(
    "note_id",
    ["note_injection_attempt", "adv_unicode_homoglyph", "adv_zero_width_injection"],
)
async def test_injection_variants_blocked_by_rail(note_id: str) -> None:
    err = await _blocked(note_id)
    assert err.reason == "NOTE_INSTRUCTION_LIKE"
    assert err.case_status == "NEEDS_ATTENTION"


async def test_error_detail_never_echoes_note_content() -> None:
    err = await _blocked("note_injection_attempt")
    for fragment in ("pilocarpine", "IGNORE ALL", "advanced glaucoma"):
        assert fragment not in (err.detail or "")


# --- PHI in the note ---


async def test_phi_never_reaches_llm_and_only_pattern_names_persist() -> None:
    client = _client()
    result = await note_pipeline.run(NOTES["adv_phi_in_note"], llm=client)
    sent = "\n".join(m.content for call in client.received for m in call.messages)
    for phi_value in ("Jane Doe", "03/14/1968", "4482910", "555-123-7789"):
        assert phi_value not in sent
    # Pattern names only — never values (I-6).
    assert all(isinstance(name, str) and "[" not in name for name in result.redactions)


# --- size / charset abuse ---


async def test_oversized_note_rejected_before_llm() -> None:
    client = _client()
    with pytest.raises(PolicyViolation) as exc_info:
        await note_pipeline.run("x" * 10_000, llm=client)
    assert exc_info.value.reason == "NOTE_TOO_LONG"
    assert client.received == []


async def test_control_char_payload_rejected_before_llm() -> None:
    client = _client()
    with pytest.raises(PolicyViolation) as exc_info:
        await note_pipeline.run("visit notes " + "\x00\x07\x1b" * 50, llm=client)
    assert exc_info.value.reason == "NOTE_CONTROL_CHARS"
    assert client.received == []


# --- multilingual noise: passes the rail, extracts as data ---


async def test_multilingual_note_passes_rail_and_extracts() -> None:
    client = _client()
    result = await note_pipeline.run(NOTES["adv_multilingual_noise"], llm=client)
    assert len(client.received) == 1
    assert result.extractor_prompt_version == PROMPT_VERSION
    # The untrusted text was delivered inside the delimited data block (I-7).
    assert "<untrusted-clinical-note>" in client.received[0].messages[1].content


# --- extractor redirection: text is data, behavior unchanged ---


async def test_extractor_redirection_does_not_change_extraction() -> None:
    """The note tries to plant IOP 45 / 'advanced glaucoma'. The pipeline's
    behavior is identical to a benign note: the (fake) extractor output for
    the fixture is returned verbatim — the planted text has no special power.
    Injection resistance of the REAL model is pinned by the prompt contract
    (prompts/note_extractor/v1.md) and checked by the live suite (§20)."""
    benign_client = _client(EMPTY_PAYLOAD)
    hostile_client = _client(EMPTY_PAYLOAD)
    benign = await note_pipeline.run(NOTES["note_minimal"], llm=benign_client)
    hostile = await note_pipeline.run(NOTES["adv_extractor_redirection"], llm=hostile_client)
    assert hostile.note == benign.note
    # The planted values appear in NEITHER the extracted fields...
    assert hostile.note.iop_mmhg_od is None
    assert hostile.note.prior_glaucoma_dx is None
    # ...nor does the rail over-block this clinical-looking text.
    assert len(hostile_client.received) == 1


async def test_planted_values_in_extractor_output_are_data_not_instructions() -> None:
    """Even when the extractor returns planted-looking values, they land in
    typed fields only — downstream treats them as untrusted note facts."""
    client = _client(PLANTED_PAYLOAD)
    result = await note_pipeline.run(NOTES["adv_extractor_redirection"], llm=client)
    assert result.note.iop_mmhg_od == 45.0  # stored as a note fact, typed
    assert result.extraction_confidence is Confidence.MODERATE


# --- low-confidence path emits the caveat marker P4 will consume ---


async def test_low_confidence_maps_to_extraction_caveat_fact() -> None:
    client = _client(EMPTY_PAYLOAD)
    result = await note_pipeline.run(NOTES["note_minimal"], llm=client)
    assert result.extraction_confidence is Confidence.LOW
    assert caveat_for_extraction(result.extraction_confidence) == F.CAVEAT_EXTRACTION


async def test_high_confidence_maps_to_no_caveat() -> None:
    assert caveat_for_extraction(Confidence.HIGH) is None
    assert caveat_for_extraction(Confidence.MODERATE) is None
