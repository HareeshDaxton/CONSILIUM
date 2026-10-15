"""Phase 10.1 — deterministic input checks (guardrails/input_checks.py).

Pattern/length/control-char behaviour is pinned in test_input_rail.py (same
functions via the intake re-export); this file covers what 10.1 adds:
charset allowlist, shared-config PHI detection (redact vs block), the
NEEDS_ATTENTION mapping at the node boundary, and the guardrail audit event.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from consilium.core.errors import PolicyViolation
from consilium.graph.build import GraphRunner
from consilium.graph.state import CaseState, FailureRecord
from consilium.guardrails.input_checks import check_note, load_input_rail
from consilium.intake import input_rail as intake_shim
from consilium.intake.deidentify import load_phi_patterns
from consilium.schemas.enums import CaseStatus

CFG = load_input_rail()
PHI = load_phi_patterns()


def test_intake_shim_is_the_same_object() -> None:
    assert intake_shim.check_note is check_note  # single source, no duplicate rail


@pytest.mark.parametrize(
    "text",
    [
        "IOP 18 mmHg OD ‮gnissim‬",  # RTL override
        "notes ⁦hidden⁩ here",  # bidi isolate
        "private use  char",  # Co
    ],
)
def test_charset_blocks(text: str) -> None:
    with pytest.raises(PolicyViolation) as exc:
        check_note(text, CFG)
    assert exc.value.reason == "NOTE_CHARSET"
    assert exc.value.case_status == "NEEDS_ATTENTION"


@pytest.mark.parametrize(
    "text",
    ["IOP 18 mmHg OD, 21 OS. Ätiologie unklar; café-au-lait; 眼压正常", "CDR​ noted"],
)
def test_charset_allows_normal_unicode(text: str) -> None:
    check_note(text, CFG)


def test_phi_redact_mode_reports_names_only() -> None:
    res = check_note("MRN 1234567, phone 555-123-4567. IOP 18.", CFG, PHI)
    assert set(res.phi_patterns) == {"medical_record_number", "phone_number"}
    assert all("1234567" not in n for n in res.phi_patterns)


def test_phi_block_mode_blocks_without_values() -> None:
    cfg = CFG.model_copy(update={"phi": CFG.phi.model_copy(update={"on_match": "block"})})
    with pytest.raises(PolicyViolation) as exc:
        check_note("MRN 1234567 seen today", cfg, PHI)
    assert exc.value.reason == "NOTE_PHI"
    assert "1234567" not in (exc.value.detail or "")


def test_clean_note_no_phi() -> None:
    assert check_note("IOP 18 mmHg OD on 2024-01-15.", CFG, PHI).phi_patterns == ()


class _RecTx:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def transition(self, **kw: Any) -> None:
        self.events.append((kw["event"], kw["detail"]))

    async def audit(self, **kw: Any) -> None:
        self.events.append((kw["event"], kw["detail"]))


def _failed(node: str) -> CaseState:
    return CaseState(
        case_id=str(uuid.uuid4()),
        status=CaseStatus.NEEDS_ATTENTION,
        failure=FailureRecord(
            type="NOTE_INSTRUCTION_LIKE", node=node, reason="input_rail.pattern:role_hijack"
        ),
        last_node=node,
    )


async def test_rail_failure_writes_guardrail_audit_event() -> None:
    tx = _RecTx()
    await GraphRunner._write_transition(tx, "input_rail", _failed("input_rail"))  # type: ignore[arg-type]
    events = dict(tx.events)
    assert events["guardrail.triggered"] == {
        "rail": "input",
        "reason": "NOTE_INSTRUCTION_LIKE",
        "rule": "input_rail.pattern:role_hijack",
    }


async def test_non_rail_failure_has_no_guardrail_event() -> None:
    tx = _RecTx()
    await GraphRunner._write_transition(tx, "call_vision", _failed("call_vision"))  # type: ignore[arg-type]
    assert [e for e, _ in tx.events] == ["node.call_vision"]
