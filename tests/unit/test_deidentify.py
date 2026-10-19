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
        # Indian clinic formats (synthetic values only)
        ("Contact: 98765 43210", "indian_mobile_number"),
        ("Call patient at +91 98765 43210", "indian_mobile_number"),
        ("Contact: +91-98765-43210", "indian_mobile_number"),
        ("Mob 9876543210.", "indian_mobile_number"),
        ("Ph 098765 43210", "indian_mobile_number"),
        ("Aadhaar: 1234 5678 9012", "aadhaar_number"),
        ("Aadhaar No. 123456789012", "aadhaar_number"),
        ("1234-5678-9012 on file", "aadhaar_number"),
        ("Patient ID: 123456", "medical_record_number"),
        ("MRN: TN-2026-12345", "medical_record_number"),
        ("UHID: 2026001234", "medical_record_number"),
        ("OP No: 45678", "medical_record_number"),
        ("Reg. No 2026/123", "medical_record_number"),
        ("D.O.B: 03-04-1959", "date_of_birth"),
        ("Patient: Mr Ramesh Kumar came for an eye checkup.", "named_person"),
        ("Mr Ramesh Kumar came for a glaucoma screening.", "named_person"),
        ("Mr. R. Kumar reviewed.", "named_person"),
        ("Shri K Ramesh, 62Y", "named_person"),
        ("Smt Lakshmi Devi attended.", "named_person"),
        ("Ramesh Kumar S/o Suresh Raman", "named_person"),
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
        # Indian clinic near-misses: labels/titles that are really clinical prose
        "Review on 2026-01-15.",  # ISO date vs 4-4-4 Aadhaar shape
        "The reading was 1234567890 mmHg.",  # 10 digits not starting 6-9
        "Platelets 250000, lab ref 9876543210123",  # mobile shape inside a longer number
        "Hospital not specified.",  # 'hospital no' label prefix
        "patient identified as glaucoma suspect",  # 'patient ID' label prefix
        "medical record review pending",  # label without a digit-bearing identifier
        "OP notes reviewed.",  # 'OP no' label prefix
        "History of MS noted.",  # MS = multiple sclerosis, not a title
        "vision w/o glasses 6/12",  # w/o = without
        "c/o blurred vision",  # c/o = complains of
        "Dr. advised review in 3 months.",  # title without a capitalised name
        "Drops: Timolol 0.5% BD",  # 'Dr' inside a word
        "Patient: Known case of POAG",  # 'Patient:' is not a name label
        "Visual acuity 6/9 OD, 6/12 OS.",  # Indian Snellen notation
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


def test_indian_note_values_never_appear_in_output(policy: PhiPolicy) -> None:
    """I-6 on an Indian-format note: every identifier value is gone, clinical text stays."""
    secrets = [
        "Ramesh Kumar",
        "Suresh Raman",
        "98765 43210",
        "1234 5678 9012",
        "TN-2026-12345",
        "03/04/1959",
        "ramesh.test@example.com",
    ]
    text = (
        "Patient: Mr Ramesh Kumar, DOB: 03/04/1959, MRN: TN-2026-12345, "
        "Aadhaar: 1234 5678 9012, Mob +91 98765 43210, email ramesh.test@example.com. "
        "Referred by Dr. Suresh Raman. IOP 24 mmHg OD, 22 mmHg OS. Review in 2-4 months."
    )
    result = redact(text, policy)
    for secret in secrets:
        assert secret not in result.text
    assert "IOP 24 mmHg OD, 22 mmHg OS. Review in 2-4 months." in result.text
