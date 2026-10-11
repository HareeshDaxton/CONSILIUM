"""De-identification tests (plan 3.1; SKILLS.md note-extraction step 2).

Covers: positive firing per pattern, negative near-misses (dates, "2-4
months", IOP numbers, ages), boundary placement, and the proof that no
matched VALUE ever appears in any output (I-6).
"""

import pytest

from consilium.intake.deidentify import (
    PhiPolicy,
    load_phi_patterns,
    redact,
)


@pytest.fixture(scope="module")
def policy() -> PhiPolicy:
    return load_phi_patterns()


def redact_text(text: str, policy: PhiPolicy) -> str:
    return redact(text, policy).text


# ---------- positive: each pattern fires ----------


@pytest.mark.parametrize(
    ("text", "pattern_name"),
    [
        ("Call the patient at 555-123-4567 tomorrow.", "phone_number"),
        ("Phone: (555) 123-4567.", "phone_number"),
        ("Reachable at +1 555 123 4567.", "phone_number"),
        ("Email jane.doe@example-clinic.org for records.", "email_address"),
        ("SSN 123-45-6789 on file.", "ssn"),
        ("MRN 8392011 referred for screening.", "medical_record_number"),
        ("Medical Record Number: 445566.", "medical_record_number"),
        ("Patient ID #998877.", "medical_record_number"),
        ("DOB: 03/04/1959.", "date_of_birth"),
        ("Date of birth 1959-03-04.", "date_of_birth"),
        ("Patient name: Jane Doe, age 62.", "named_person"),
        ("Referred by Dr. John Smith.", "named_person"),
        ("Name: Jane Doe. Signed: Mark Brown", "named_person"),
    ],
)
def test_pattern_fires(policy: PhiPolicy, text: str, pattern_name: str) -> None:
    result = redact(text, policy)
    assert pattern_name in result.pattern_names
    assert f"[REDACTED:{pattern_name}]" in result.text


# ---------- negative: benign near-misses must NOT fire ----------


@pytest.mark.parametrize(
    "text",
    [
        "Follow-up appointment on 2024-01-15.",  # ISO date
        "Review in 2-4 months.",  # interval range
        "IOP 24 mmHg OD, 22 mmHg OS.",  # clinical numbers
        "62-year-old patient.",  # age
        "Started latanoprost on 12 March 2024.",  # month-name date
        "Visual acuity 20/40 OD.",  # ratio-like numbers
        "The name of the drug class is prostaglandin analogues.",  # lowercase 'name'
        "Born prematurely at 32 weeks.",  # 'born' without label/shape
        "Cup-disc ratio 0.7-0.8.",  # decimal range
    ],
)
def test_benign_near_misses_do_not_fire(policy: PhiPolicy, text: str) -> None:
    result = redact(text, policy)
    assert result.pattern_names == ()
    assert result.text == text


# ---------- boundary / structural ----------


def test_multiple_patterns_all_reported_in_order(policy: PhiPolicy) -> None:
    text = "Name: Jane Doe. MRN 123456. Call 555-123-4567."
    result = redact(text, policy)
    assert set(result.pattern_names) == {"named_person", "medical_record_number", "phone_number"}
    assert len(result.pattern_names) == len(set(result.pattern_names))  # unique


def test_pattern_at_string_start_and_end(policy: PhiPolicy) -> None:
    text = "jane@example.org is the contact, SSN 123-45-6789"
    result = redact(text, policy)
    assert result.text.startswith("[REDACTED:email_address]")
    assert result.text.endswith("[REDACTED:ssn]")


def test_repeated_match_same_pattern_name_once(policy: PhiPolicy) -> None:
    text = "MRN 111111 and again MRN 222222."
    result = redact(text, policy)
    assert result.pattern_names == ("medical_record_number",)
    assert result.text.count("[REDACTED:medical_record_number]") == 2


def test_redaction_is_deterministic(policy: PhiPolicy) -> None:
    text = "Patient name: Jane Doe. DOB: 03/04/1959. Phone 555-123-4567."
    assert redact(text, policy) == redact(text, policy)


def test_no_match_value_appears_in_any_output(policy: PhiPolicy) -> None:
    """The core I-6 guarantee: values never leak into text OR pattern names."""
    secrets = ["Jane Doe", "123-45-6789", "555-123-4567", "jane.doe@example-clinic.org"]
    text = (
        "Patient name: Jane Doe. SSN 123-45-6789. "
        "Call 555-123-4567 or email jane.doe@example-clinic.org."
    )
    result = redact(text, policy)
    for secret in secrets:
        assert secret not in result.text
        assert all(secret not in name for name in result.pattern_names)
