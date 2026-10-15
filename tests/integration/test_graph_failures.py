"""Section 9.4 exit: the failure-edge matrix (ARCHITECTURE.md §10), offline —
fake LLM + fake vision + real local Postgres. Every edge fails closed (I-5)
and every pipeline exit is a persisted, audited status; nothing is APPROVED
by the graph (I-4)."""

from __future__ import annotations

import uuid

import pytest
from graph_kit import (
    APPROVE,
    IMAGE_URI,
    PIN,
    REJECT,
    clean_draft,
    count_rows,
    expected_ctx,
    fake_llm,
    make_runner,
    submit,
)

from consilium.core.settings import Settings
from consilium.persistence.models import (
    AgentRunRow,
    AuditLogRow,
    DraftRow,
    VerdictRow,
    VisionCallRow,
)
from consilium.persistence.repositories import Repositories
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.enums import CaseStatus
from consilium.vision import client_fake as vf


@pytest.fixture
async def draft() -> DirectorDraft:
    return clean_draft(await expected_ctx())


async def _persisted_status(repos: Repositories, case_id: str) -> str:
    row = await repos.load_case(uuid.UUID(case_id))
    assert row is not None
    return row.status


async def test_happy_path_reaches_pending_review(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    runner = make_runner(settings, repos, llm=fake_llm(draft))
    final = await runner.run(await submit(runner))
    cid = final.case_id
    assert final.status == CaseStatus.PENDING_REVIEW, final.failure
    assert await _persisted_status(repos, cid) == "PENDING_REVIEW"
    assert await count_rows(repos, DraftRow, cid, unverified=False) == 1
    assert await count_rows(repos, VerdictRow, cid, approved=True) == 1
    # extractor + 3 specialists + director + critic
    assert await count_rows(repos, AgentRunRow, cid) == 6
    assert await count_rows(repos, AuditLogRow, cid, event="node.persist_pending") == 1


async def test_specialist_failure_after_retries_fails_case(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    runner = make_runner(settings, repos, llm=fake_llm(draft, refuse_role="pharmacist"))
    final = await runner.run(await submit(runner))
    assert final.status == CaseStatus.FAILED
    assert final.failure is not None and final.failure.node == "run_specialists"
    assert await _persisted_status(repos, final.case_id) == "FAILED"
    assert await count_rows(repos, DraftRow, final.case_id) == 0  # no degraded report


async def test_critic_reject_then_revise_then_approve(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    runner = make_runner(settings, repos, llm=fake_llm(draft, critic=(REJECT, APPROVE)))
    final = await runner.run(await submit(runner))
    assert final.status == CaseStatus.PENDING_REVIEW, final.failure
    assert final.audit_round == 2 and final.draft_revision == 1
    assert await count_rows(repos, VerdictRow, final.case_id) == 2
    assert await count_rows(repos, DraftRow, final.case_id) == 2  # rev 0 + rev 1


async def test_loop_exhaustion_needs_attention_with_unverified_contract(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    runner = make_runner(settings, repos, llm=fake_llm(draft, critic=(REJECT,)))
    final = await runner.run(await submit(runner))
    cid = final.case_id
    assert final.status == CaseStatus.NEEDS_ATTENTION
    assert final.audit_round == settings.critic_max_rounds
    assert final.unverified and final.open_issues  # issues attached, never dropped
    assert await count_rows(repos, VerdictRow, cid, approved=False) == settings.critic_max_rounds
    # the last draft revision is kept and flagged UNVERIFIED (§4 / I-4)
    assert (
        await count_rows(repos, DraftRow, cid, revision=final.draft_revision, unverified=True) == 1
    )
    assert await _persisted_status(repos, cid) == "NEEDS_ATTENTION"


async def test_vision_cold_start_is_retried(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    vision = vf.FakeVisionClient(revision=PIN, cold_start_calls=1)
    runner = make_runner(settings, repos, llm=fake_llm(draft), vision=vision)
    final = await runner.run(await submit(runner))
    assert final.status == CaseStatus.PENDING_REVIEW, final.failure
    assert vision.calls == 2  # one 503, one success


async def test_revision_pin_mismatch_fails_closed(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    vision = vf.FakeVisionClient(revision=PIN, force_revision="rogue-revision")
    runner = make_runner(settings, repos, llm=fake_llm(draft), vision=vision)
    final = await runner.run(await submit(runner))
    assert final.status == CaseStatus.FAILED
    assert final.failure is not None and final.failure.node == "call_vision"
    assert "REVISION" in final.failure.type
    assert await count_rows(repos, VisionCallRow, final.case_id, status="failed") == 1


async def test_bad_image_is_rejected_input(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    runner = make_runner(
        settings, repos, llm=fake_llm(draft), images={IMAGE_URI: b"definitely not an image"}
    )
    final = await runner.run(await submit(runner))
    assert final.status == CaseStatus.REJECTED_INPUT
    assert final.failure is not None and final.failure.node == "precheck_image"
    assert await _persisted_status(repos, final.case_id) == "REJECTED_INPUT"


async def test_idempotent_reentry(
    settings: Settings, repos: Repositories, draft: DirectorDraft
) -> None:
    llm = fake_llm(draft)
    runner = make_runner(settings, repos, llm=llm)
    first = await submit(runner, key="idem-key-1")
    final = await runner.run(first)
    calls = len(llm.received)

    again = await submit(runner, key="idem-key-1")  # re-submission → same case
    assert again.case_id == first.case_id and again.status == CaseStatus.PENDING_REVIEW
    resumed = await runner.resume(final.case_id)  # terminal → unchanged, no work
    assert resumed.status == CaseStatus.PENDING_REVIEW
    assert len(llm.received) == calls
