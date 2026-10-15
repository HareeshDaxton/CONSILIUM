"""Section 9.3 exit: crash-resume (kill mid-pipeline, restart, completes
without recomputation). Real local Postgres; fake LLM + fake vision."""

from __future__ import annotations

import uuid

import pytest
from graph_kit import PIN, clean_draft, expected_ctx, fake_llm, make_runner, submit
from sqlalchemy import func, select

from consilium.core.settings import Settings
from consilium.graph.state import CaseState
from consilium.persistence.models import AuditLogRow, CadResultRow, ClinicalNoteRow
from consilium.persistence.repositories import Repositories
from consilium.schemas.enums import CaseStatus
from consilium.vision import client_fake as vf


class SimulatedCrash(RuntimeError):
    """Raised by the hook AFTER a node's transaction committed — the process
    'dies' with the DB holding exactly that node's persisted state."""


def _crash_after(node: str):  # type: ignore[no-untyped-def]
    def hook(name: str, _state: CaseState) -> None:
        if name == node:
            raise SimulatedCrash(name)

    return hook


async def _count(repos: Repositories, model: type, case_id: str) -> int:
    async with repos._session_factory() as s:  # test-only read
        return int(
            (
                await s.execute(
                    select(func.count())
                    .select_from(model)
                    .where(
                        model.case_id == uuid.UUID(case_id)  # type: ignore[attr-defined]
                    )
                )
            ).scalar_one()
        )


async def test_crash_after_vision_resumes_without_recomputation(
    settings: Settings, repos: Repositories
) -> None:
    draft = clean_draft(await expected_ctx())
    crashing = make_runner(settings, repos, llm=fake_llm(draft), hook=_crash_after("call_vision"))
    state = await submit(crashing)
    with pytest.raises(SimulatedCrash):
        await crashing.run(state)

    vision2 = vf.FakeVisionClient(revision=PIN)
    llm2 = fake_llm(draft)
    final = await make_runner(settings, repos, llm=llm2, vision=vision2).resume(state.case_id)

    assert final.status == CaseStatus.PENDING_REVIEW, final.failure
    assert vision2.calls == 0  # CAD never recomputed (§4 idempotency)
    assert all(c.key[0] != "note_extractor" for c in llm2.received)  # nor re-extracted
    assert await _count(repos, CadResultRow, state.case_id) == 1
    persisted = await repos.load_case(uuid.UUID(state.case_id))
    assert persisted is not None and persisted.status == CaseStatus.PENDING_REVIEW.value
    assert "raw_note" not in str(persisted.state_json)  # I-6: no note text in snapshot


async def test_crash_after_input_rail_rehydrates_redacted_note(
    settings: Settings, repos: Repositories
) -> None:
    draft = clean_draft(await expected_ctx())
    crashing = make_runner(settings, repos, llm=fake_llm(draft), hook=_crash_after("input_rail"))
    state = await submit(crashing)
    with pytest.raises(SimulatedCrash):
        await crashing.run(state)

    llm2 = fake_llm(draft)
    final = await make_runner(settings, repos, llm=llm2).resume(state.case_id)
    assert final.status == CaseStatus.PENDING_REVIEW, final.failure
    assert final.note is not None and final.note.age_years == 65
    assert sum(c.key[0] == "note_extractor" for c in llm2.received) == 1
    assert await _count(repos, ClinicalNoteRow, state.case_id) == 1


async def test_note_lost_before_input_rail_fails_closed(
    settings: Settings, repos: Repositories
) -> None:
    draft = clean_draft(await expected_ctx())
    crashing = make_runner(
        settings, repos, llm=fake_llm(draft), hook=_crash_after("precheck_image")
    )
    state = await submit(crashing)
    with pytest.raises(SimulatedCrash):
        await crashing.run(state)

    final = await make_runner(settings, repos, llm=fake_llm(draft)).resume(state.case_id)
    assert final.status == CaseStatus.FAILED
    assert final.failure is not None and final.failure.type == "NOTE_NOT_RECOVERABLE"
    assert await _count(repos, AuditLogRow, state.case_id) >= 3  # created, precheck, resume
