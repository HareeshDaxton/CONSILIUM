"""Repositories — one per aggregate, plus the graph's atomic transition.

Persistence-per-node rule (ARCHITECTURE.md §5.3): every graph node writes
``(status, state_json, audit event)`` in ONE transaction via
``RepoTx.transition``. Domain rows (clinical_notes, cad_results, drafts,
verdicts, agent_runs, routed_contexts, vision_calls) written by the same node
join the SAME transaction — a node either persists completely or not at all
(crash-resume depends on this).

The redacted note text is encrypted before it touches the DB
(``FieldEncryptor``); it is never written to ``state_json``, logs, or
``audit_log.detail_json`` (I-6).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.dialects.postgresql import array as pg_array
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from consilium.persistence.db import SessionFactory
from consilium.persistence.encryption import FieldEncryptor
from consilium.persistence.models import (
    APPEND_ONLY_TABLES,
    AgentRunRow,
    AuditLogRow,
    CadResultRow,
    Case,
    ClinicalNoteRow,
    DraftRow,
    JobRow,
    ReviewRow,
    RoutedContextRow,
    VerdictRow,
    VisionCallRow,
)

__all__ = [
    "APPEND_ONLY_TABLES",
    "PersistedAuditEvent",
    "PersistedCad",
    "PersistedCase",
    "PersistedDraft",
    "PersistedJob",
    "PersistedReview",
    "RepoTx",
    "Repositories",
]


@dataclass(frozen=True, slots=True)
class PersistedCase:
    """Read view used by the graph runner for crash-resume."""

    id: uuid.UUID
    status: str
    state_json: dict[str, Any]
    last_node: str | None
    failure_json: dict[str, Any] | None
    created_at: datetime | None = None
    created_by: str | None = None


@dataclass(frozen=True, slots=True)
class PersistedDraft:
    revision: int
    draft_json: dict[str, Any]
    unverified: bool
    open_issues_json: list[Any] | None


@dataclass(frozen=True, slots=True)
class PersistedCad:
    cad_json: dict[str, Any]
    raw_vision_json: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PersistedAuditEvent:
    """Timeline view: event name, opaque actor, time — never ``detail_json``."""

    event: str
    actor: str
    at: datetime


@dataclass(frozen=True, slots=True)
class PersistedJob:
    case_id: uuid.UUID
    status: str
    attempts: int
    claimed_by: str | None


@dataclass(frozen=True, slots=True)
class PersistedReview:
    id: uuid.UUID
    draft_revision: int | None
    reviewer_id: str
    action: str
    edits_diff_json: dict[str, Any] | None
    edited_draft_json: dict[str, Any] | None
    reason_codes: list[Any]
    free_text_reason: str | None
    decided_at: datetime


def _review_view(row: ReviewRow) -> PersistedReview:
    return PersistedReview(
        id=row.id,
        draft_revision=row.draft_revision,
        reviewer_id=row.reviewer_id,
        action=row.action,
        edits_diff_json=row.edits_diff_json,
        edited_draft_json=row.edited_draft_json,
        reason_codes=row.reason_codes,
        free_text_reason=row.free_text_reason,
        decided_at=row.decided_at,
    )


def _case_view(row: Case) -> PersistedCase:
    return PersistedCase(
        id=row.id,
        status=row.status,
        state_json=row.state_json,
        last_node=row.last_node,
        failure_json=row.failure_json,
        created_at=row.created_at,
        created_by=row.created_by,
    )


class RepoTx:
    """One database transaction scoped to a single graph node (or test)."""

    def __init__(self, session: AsyncSession, encryptor: FieldEncryptor) -> None:
        self._session = session
        self._encryptor = encryptor

    # ----- cases + audit (the graph transition) -----

    async def create_case(
        self,
        *,
        case_id: uuid.UUID,
        status: str,
        laterality: str,
        created_by: str,
        idempotency_key: str | None,
        image_sha256: str | None,
        image_uri: str | None,
        state_json: dict[str, Any],
        last_node: str | None,
    ) -> None:
        self._session.add(
            Case(
                id=case_id,
                status=status,
                laterality=laterality,
                created_by=created_by,
                idempotency_key=idempotency_key,
                image_sha256=image_sha256,
                image_uri=image_uri,
                state_json=state_json,
                last_node=last_node,
            )
        )
        await self._session.flush()  # case row must exist before its audit event (FK)
        await self.audit(case_id=case_id, event="case.created", actor="graph", detail={})

    async def transition(
        self,
        *,
        case_id: uuid.UUID,
        status: str,
        state_json: dict[str, Any],
        last_node: str,
        failure_json: dict[str, Any] | None,
        event: str,
        actor: str = "graph",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """The ONE-transaction node write: cases row + audit event (§5.3)."""
        await self._session.execute(
            update(Case)
            .where(Case.id == case_id)
            .values(
                status=status,
                state_json=state_json,
                last_node=last_node,
                failure_json=failure_json,
                updated_at=datetime.now(UTC),
            )
        )
        await self.audit(case_id=case_id, event=event, actor=actor, detail=detail or {})

    async def audit(
        self,
        *,
        case_id: uuid.UUID | None,
        event: str,
        actor: str,
        detail: dict[str, Any],
    ) -> None:
        """Append-only audit event. ``detail`` must never contain note text,
        image data, prompts, or upload filenames (I-6/I-14)."""
        self._session.add(
            AuditLogRow(case_id=case_id, event=event, actor=actor, detail_json=detail)
        )

    async def case_for_update(self, case_id: uuid.UUID) -> PersistedCase | None:
        """Row-locked read (``FOR UPDATE``): serialises concurrent reviews of
        one case so two clinicians cannot both finalise it."""
        row = (
            await self._session.execute(select(Case).where(Case.id == case_id).with_for_update())
        ).scalar_one_or_none()
        return None if row is None else _case_view(row)

    # ----- reviews (append-only; the review service is the only writer) -----

    async def review_add(
        self,
        *,
        review_id: uuid.UUID,
        case_id: uuid.UUID,
        draft_revision: int,
        reviewer_id: str,
        action: str,
        edits_diff_json: dict[str, Any] | None,
        edited_draft_json: dict[str, Any] | None,
        reason_codes: list[str],
        free_text_reason: str | None,
    ) -> None:
        self._session.add(
            ReviewRow(
                id=review_id,
                case_id=case_id,
                draft_revision=draft_revision,
                reviewer_id=reviewer_id,
                action=action,
                edits_diff_json=edits_diff_json,
                edited_draft_json=edited_draft_json,
                reason_codes=reason_codes,
                free_text_reason=free_text_reason,
            )
        )

    async def reviews_for_case(self, case_id: uuid.UUID) -> list[PersistedReview]:
        rows = (
            await self._session.execute(
                select(ReviewRow)
                .where(ReviewRow.case_id == case_id)
                .order_by(ReviewRow.decided_at, ReviewRow.id)
            )
        ).scalars()
        return [_review_view(r) for r in rows]

    async def draft_get(self, case_id: uuid.UUID, revision: int) -> PersistedDraft | None:
        row = await self._session.get(DraftRow, (case_id, revision))
        if row is None:
            return None
        return PersistedDraft(
            revision=row.revision,
            draft_json=row.draft_json,
            unverified=row.unverified,
            open_issues_json=row.open_issues_json,
        )

    async def draft_latest_revision(self, case_id: uuid.UUID) -> int | None:
        value = (
            await self._session.execute(
                select(func.max(DraftRow.revision)).where(DraftRow.case_id == case_id)
            )
        ).scalar_one_or_none()
        return None if value is None else int(value)

    async def verdict_latest_issues(self, case_id: uuid.UUID) -> list[Any] | None:
        row = (
            await self._session.execute(
                select(VerdictRow)
                .where(VerdictRow.case_id == case_id)
                .order_by(VerdictRow.at.desc(), VerdictRow.round.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return None if row is None else row.issues_json

    # ----- clinical notes -----

    async def notes_save_redacted(self, *, case_id: uuid.UUID, redacted_text: str) -> None:
        """Persist the redacted note at input-rail time (crash before
        extraction stays resumable); extraction completes the row later."""
        enc = self._encryptor.encrypt(redacted_text)
        existing = await self._session.get(ClinicalNoteRow, case_id)
        if existing is None:
            self._session.add(
                ClinicalNoteRow(
                    case_id=case_id,
                    note_json=None,
                    extraction_json=None,
                    raw_redacted_text_enc=enc,
                )
            )
        else:
            existing.raw_redacted_text_enc = enc

    async def notes_complete(
        self,
        *,
        case_id: uuid.UUID,
        note_json: dict[str, Any],
        extraction_json: dict[str, Any],
    ) -> None:
        row = await self._session.get(ClinicalNoteRow, case_id)
        if row is None:
            raise LookupError(f"clinical_notes row missing for case {case_id}")
        row.note_json = note_json
        row.extraction_json = extraction_json

    async def notes_get_redacted(self, *, case_id: uuid.UUID) -> str | None:
        row = await self._session.get(ClinicalNoteRow, case_id)
        if row is None:
            return None
        return self._encryptor.decrypt(row.raw_redacted_text_enc)

    async def notes_completed(self, *, case_id: uuid.UUID) -> bool:
        row = await self._session.get(ClinicalNoteRow, case_id)
        return row is not None and row.note_json is not None

    # ----- CAD -----

    async def cad_save(
        self,
        *,
        case_id: uuid.UUID,
        cad_json: dict[str, Any],
        raw_vision_json: dict[str, Any],
        classifier_version: str,
        segmenter_version: str,
        endpoint_revision: str,
    ) -> None:
        self._session.add(
            CadResultRow(
                case_id=case_id,
                cad_json=cad_json,
                raw_vision_json=raw_vision_json,
                classifier_version=classifier_version,
                segmenter_version=segmenter_version,
                endpoint_revision=endpoint_revision,
            )
        )

    async def cad_exists(self, *, case_id: uuid.UUID) -> bool:
        return await self._session.get(CadResultRow, case_id) is not None

    # ----- vision call telemetry -----

    async def vision_call_add(
        self,
        *,
        case_id: uuid.UUID,
        endpoint_revision: str | None,
        latency_ms: int | None,
        attempts: int | None,
        status: str,
        error: str | None,
    ) -> None:
        self._session.add(
            VisionCallRow(
                case_id=case_id,
                endpoint_revision=endpoint_revision,
                latency_ms=latency_ms,
                attempts=attempts,
                status=status,
                error=error,
            )
        )

    # ----- routed contexts -----

    async def routed_save(
        self,
        *,
        case_id: uuid.UUID,
        role: str,
        routing_version: str,
        facts_json: dict[str, Any],
        prompt_hash: str,
    ) -> None:
        self._session.add(
            RoutedContextRow(
                case_id=case_id,
                role=role,
                routing_version=routing_version,
                facts_json=facts_json,
                prompt_hash=prompt_hash,
            )
        )

    # ----- agent runs -----

    async def agent_run_add(
        self,
        *,
        case_id: uuid.UUID,
        agent: str,
        prompt_version: str | None,
        model_id: str,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: int | None,
        attempt: int,
        status: str,
        output_json: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        self._session.add(
            AgentRunRow(
                case_id=case_id,
                agent=agent,
                prompt_version=prompt_version,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                attempt=attempt,
                status=status,
                output_json=output_json,
                error=error,
            )
        )

    # ----- drafts / verdicts -----

    async def draft_save(
        self,
        *,
        case_id: uuid.UUID,
        revision: int,
        draft_json: dict[str, Any],
        producer_run_id: str | None,
        unverified: bool,
        open_issues_json: list[Any] | None,
    ) -> None:
        self._session.add(
            DraftRow(
                case_id=case_id,
                revision=revision,
                draft_json=draft_json,
                producer_run_id=producer_run_id,
                unverified=unverified,
                open_issues_json=open_issues_json,
            )
        )

    async def draft_mark_unverified(
        self, *, case_id: uuid.UUID, revision: int, open_issues_json: list[Any]
    ) -> None:
        """Loop exhaustion (§4): flag the existing draft revision UNVERIFIED
        and attach its open issues — the draft is kept, never dropped."""
        row = await self._session.get(DraftRow, (case_id, revision))
        if row is None:
            raise LookupError(f"draft revision {revision} missing for case {case_id}")
        row.unverified = True
        row.open_issues_json = open_issues_json

    async def verdict_add(
        self,
        *,
        case_id: uuid.UUID,
        revision: int,
        round_: int,
        approved: bool,
        issues_json: list[Any],
        layer: str,
    ) -> None:
        self._session.add(
            VerdictRow(
                case_id=case_id,
                revision=revision,
                round=round_,
                approved=approved,
                issues_json=issues_json,
                layer=layer,
            )
        )


class Repositories:
    """One repository bundle per aggregate; transactions via ``transaction()``."""

    def __init__(self, session_factory: SessionFactory, encryptor: FieldEncryptor) -> None:
        self._session_factory = session_factory
        self._encryptor = encryptor

    def transaction(self) -> _TxContext:
        return _TxContext(self)

    async def ping(self) -> bool:
        """Readiness probe: True iff the database answers ``SELECT 1``."""
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:  # readiness boundary: any DB error means "not ready"
            return False

    async def load_case(self, case_id: uuid.UUID) -> PersistedCase | None:
        """Read-only load for crash-resume (own short session)."""
        async with self._session_factory() as session:
            row = await session.get(Case, case_id)
            return None if row is None else _case_view(row)

    async def load_redacted_note(self, case_id: uuid.UUID) -> str | None:
        """Read-only load of the redacted note for state rehydration."""
        async with self._session_factory() as session:
            row = await session.get(ClinicalNoteRow, case_id)
            if row is None:
                return None
            return self._encryptor.decrypt(row.raw_redacted_text_enc)

    async def find_case_by_idempotency_key(self, key: str) -> PersistedCase | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(select(Case).where(Case.idempotency_key == key))
            ).scalar_one_or_none()
            return None if row is None else _case_view(row)

    # ----- read views for the API -----

    async def cases_list(
        self,
        *,
        statuses: tuple[str, ...] = (),
        created_by: str | None = None,
        limit: int = 50,
    ) -> list[PersistedCase]:
        """Newest first; optional status filter (queue views) and owner scope."""
        stmt = select(Case).order_by(Case.created_at.desc(), Case.id).limit(limit)
        if statuses:
            stmt = stmt.where(Case.status.in_(statuses))
        if created_by is not None:
            stmt = stmt.where(Case.created_by == created_by)
        async with self._session_factory() as session:
            return [_case_view(row) for row in (await session.execute(stmt)).scalars()]

    async def audit_events(self, case_id: uuid.UUID) -> list[PersistedAuditEvent]:
        """Append order. ``detail_json`` is deliberately not projected."""
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(AuditLogRow.event, AuditLogRow.actor, AuditLogRow.at)
                    .where(AuditLogRow.case_id == case_id)
                    .order_by(AuditLogRow.id)
                )
            ).all()
        return [PersistedAuditEvent(event=e, actor=a, at=at) for e, a, at in rows]

    async def draft_latest(self, case_id: uuid.UUID) -> PersistedDraft | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(DraftRow)
                    .where(DraftRow.case_id == case_id)
                    .order_by(DraftRow.revision.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return PersistedDraft(
                revision=row.revision,
                draft_json=row.draft_json,
                unverified=row.unverified,
                open_issues_json=row.open_issues_json,
            )

    async def cad_load(self, case_id: uuid.UUID) -> PersistedCad | None:
        async with self._session_factory() as session:
            row = await session.get(CadResultRow, case_id)
            if row is None:
                return None
            return PersistedCad(cad_json=row.cad_json, raw_vision_json=row.raw_vision_json)

    async def reviews_for_case(self, case_id: uuid.UUID) -> list[PersistedReview]:
        async with self.transaction() as tx:
            return await tx.reviews_for_case(case_id)

    async def reason_code_counts(
        self, *, knowledge_codes: tuple[str, ...]
    ) -> tuple[dict[str, int], int, int]:
        """(per-code counts, #reviews carrying codes, #reviews citing any
        knowledge code) over the append-only reviews table."""
        async with self._session_factory() as session:
            # constant SQL (no user input): explode the JSONB code arrays laterally
            per_code = (
                await session.execute(
                    text(
                        "SELECT code, count(*) FROM reviews, "
                        "LATERAL jsonb_array_elements_text(reviews.reason_codes) AS code "
                        "GROUP BY code"
                    )
                )
            ).all()
            has_codes = func.jsonb_array_length(ReviewRow.reason_codes) > 0
            total = (
                await session.execute(select(func.count()).select_from(ReviewRow).where(has_codes))
            ).scalar_one()
            knowledge = (
                await session.execute(
                    select(func.count())
                    .select_from(ReviewRow)
                    .where(ReviewRow.reason_codes.op("?|")(pg_array(list(knowledge_codes))))
                )
            ).scalar_one()
        return {str(c): int(n) for c, n in per_code}, int(total), int(knowledge)

    # ----- job queue (ARCHITECTURE.md §5.2: the jobs table IS the queue) -----

    async def jobs_enqueue(self, case_id: uuid.UUID) -> None:
        """Idempotent: a case has at most one job row."""
        async with self._session_factory() as session, session.begin():
            await session.execute(
                pg_insert(JobRow)
                .values(case_id=case_id, status="queued", attempts=0)
                .on_conflict_do_nothing(index_elements=[JobRow.case_id])
            )

    async def jobs_claim(self, worker_id: str) -> uuid.UUID | None:
        """Claim the oldest runnable job (``FOR UPDATE SKIP LOCKED``): two
        workers never claim the same job; a crashed claim is reclaimed by
        ``jobs_requeue_stale``."""
        now = datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            row = (
                await session.execute(
                    select(JobRow)
                    .where(
                        JobRow.status == "queued",
                        or_(JobRow.run_at.is_(None), JobRow.run_at <= now),
                    )
                    .order_by(JobRow.at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            row.status = "running"
            row.claimed_by = worker_id
            row.attempts += 1
            row.at = now  # claim time doubles as the staleness heartbeat
            return row.case_id

    async def jobs_finish(self, case_id: uuid.UUID, *, status: str) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(JobRow).where(JobRow.case_id == case_id).values(status=status)
            )

    async def jobs_retry(self, case_id: uuid.UUID, *, delay: timedelta) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(JobRow)
                .where(JobRow.case_id == case_id)
                .values(status="queued", claimed_by=None, run_at=datetime.now(UTC) + delay)
            )

    async def jobs_requeue_stale(self, *, older_than: timedelta) -> int:
        cutoff = datetime.now(UTC) - older_than
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                update(JobRow)
                .where(JobRow.status == "running", JobRow.at < cutoff)
                .values(status="queued", claimed_by=None)
            )
            return int(getattr(result, "rowcount", 0) or 0)

    async def job_get(self, case_id: uuid.UUID) -> PersistedJob | None:
        async with self._session_factory() as session:
            row = await session.get(JobRow, case_id)
            if row is None:
                return None
            return PersistedJob(
                case_id=row.case_id,
                status=row.status,
                attempts=row.attempts,
                claimed_by=row.claimed_by,
            )


class _TxContext:
    def __init__(self, repos: Repositories) -> None:
        self._repos = repos
        self.tx: RepoTx | None = None

    async def __aenter__(self) -> RepoTx:
        self._session = self._repos._session_factory()
        self.tx = RepoTx(self._session, self._repos._encryptor)
        return self.tx

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.close()
