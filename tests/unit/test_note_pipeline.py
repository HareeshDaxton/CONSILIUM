"""Note intake pipeline tests (Section 3.3): fixed order, provenance,
anti-over-extraction mapping, and the redaction-before-LLM guarantee (I-6)."""

import json
from pathlib import Path
from typing import Any

import pytest

from consilium.core.errors import PolicyViolation, ValidationFailure
from consilium.intake import note_pipeline
from consilium.intake.extractor import PROMPT_VERSION, estimate_confidence
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.schemas.enums import AgentName, Confidence
from consilium.schemas.note import ClinicalNote

FIXTURES = Path(__file__).parent.parent / "fixtures"
NOTES = {n["id"]: n["text"] for n in json.loads((FIXTURES / "synthetic_notes.json").read_text())}

AGENT = AgentName.NOTE_EXTRACTOR.value
KEY = (AGENT, PROMPT_VERSION, "default")

FULL_PAYLOAD: dict[str, Any] = {
    "age_years": 62,
    "iop_mmhg_od": 24.0,
    "iop_mmhg_os": 22.0,
    "current_medications": ["lisinopril"],
    "drug_intolerances_or_allergies": ["sulfa"],
    "comorbidities": ["hypertension"],
    "prior_glaucoma_dx": None,
    "exam_findings": "Optic discs appear suspicious on funduscopy.",
    "free_text": None,
}
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


def _client(*calls: CannedCall) -> FakeLLMClient:
    return FakeLLMClient({KEY: CannedResponse(tuple(calls))})


async def test_happy_path_extracts_and_records_provenance() -> None:
    client = _client(CannedCall(FakeMode.VALID, FULL_PAYLOAD))
    result = await note_pipeline.run(NOTES["note_routine_referral"], llm=client)

    note = result.note
    assert note.age_years == 62
    assert note.iop_mmhg_od == 24.0 and note.iop_mmhg_os == 22.0
    assert note.current_medications == ("lisinopril",)
    assert result.extractor_model  # pinned snapshot recorded
    assert result.extractor_prompt_version == PROMPT_VERSION
    assert result.redactions == ()
    assert result.extraction_confidence is Confidence.HIGH

    # The LLM saw the prompt + a delimited untrusted-data block (I-7).
    messages = client.received[0].messages
    assert messages[0].role == "system"
    assert "<untrusted-clinical-note>" in messages[1].content
    assert NOTES["note_routine_referral"] in messages[1].content


async def test_vague_note_maps_to_empty_extraction_no_invented_numbers() -> None:
    """Anti-over-extraction (SKILLS.md note-extraction step 4): the vague-note
    fixture yields an all-empty ClinicalNote — no fabricated IOP/age."""
    client = _client(CannedCall(FakeMode.VALID, EMPTY_PAYLOAD))
    result = await note_pipeline.run(NOTES["note_vague_pressure"], llm=client)
    assert result.note.iop_mmhg_od is None
    assert result.note.iop_mmhg_os is None
    assert result.note.age_years is None
    assert result.extraction_confidence is Confidence.LOW


async def test_minimal_note_low_confidence() -> None:
    client = _client(CannedCall(FakeMode.VALID, EMPTY_PAYLOAD))
    result = await note_pipeline.run(NOTES["note_minimal"], llm=client)
    assert result.extraction_confidence is Confidence.LOW


async def test_phi_redacted_before_llm_and_names_recorded() -> None:
    """I-6 proof: PHI values never reach the LLM; only pattern NAMES are kept."""
    phi_note = NOTES["adv_phi_in_note"]
    client = _client(CannedCall(FakeMode.VALID, FULL_PAYLOAD))
    result = await note_pipeline.run(phi_note, llm=client)

    assert set(result.redactions) >= {
        "named_person",
        "date_of_birth",
        "medical_record_number",
        "phone_number",
    }
    sent = "\n".join(m.content for m in client.received[0].messages)
    for phi_value in ("Jane Doe", "03/14/1968", "4482910", "555-123-7789"):
        assert phi_value in phi_note  # sanity: fixture really contains them
        assert phi_value not in sent  # and none of them reached the LLM
    assert "[REDACTED:" in sent


async def test_injection_note_blocked_before_any_llm_call() -> None:
    client = _client(CannedCall(FakeMode.VALID, FULL_PAYLOAD))
    with pytest.raises(PolicyViolation) as exc_info:
        await note_pipeline.run(NOTES["note_injection_attempt"], llm=client)
    assert exc_info.value.reason == "NOTE_INSTRUCTION_LIKE"
    assert exc_info.value.case_status == "NEEDS_ATTENTION"
    assert client.received == []  # rail runs first: the LLM was never called


async def test_malformed_output_repaired_then_valid() -> None:
    client = _client(
        CannedCall(FakeMode.SCHEMA_VIOLATING),
        CannedCall(FakeMode.VALID, FULL_PAYLOAD),
    )
    result = await note_pipeline.run(NOTES["note_routine_referral"], llm=client)
    assert result.note.age_years == 62
    assert len(client.received) == 2


async def test_persistent_invalid_output_fails_closed() -> None:
    client = _client(CannedCall(FakeMode.MALFORMED))
    with pytest.raises(ValidationFailure) as exc_info:
        await note_pipeline.run(NOTES["note_routine_referral"], llm=client)
    assert exc_info.value.reason == "LLM_OUTPUT_INVALID"
    assert exc_info.value.case_status == "FAILED"
    assert len(client.received) == 3  # 1 + 2 repairs, bounded


def test_estimate_confidence_thresholds() -> None:
    full = ClinicalNote.model_validate_json(json.dumps(FULL_PAYLOAD))
    empty = ClinicalNote.model_validate_json("{}")
    one_field = ClinicalNote.model_validate_json('{"age_years": 62}')
    assert estimate_confidence(full) is Confidence.HIGH
    assert estimate_confidence(one_field) is Confidence.MODERATE
    assert estimate_confidence(empty) is Confidence.LOW
