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
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from sqlalchemy import select, update
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
    RoutedContextRow,
    VerdictRow,
    VisionCallRow,
)

__all__ = ["APPEND_ONLY_TABLES", "PersistedCase", "RepoTx", "Repositories"]


@dataclass(frozen=True, slots=True)
class PersistedCase:
    """Read view used by the graph runner for crash-resume."""

    id: uuid.UUID
    status: str
    state_json: dict[str, Any]
    last_node: str | None
    failure_json: dict[str, Any] | None


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

    async def load_case(self, case_id: uuid.UUID) -> PersistedCase | None:
        """Read-only load for crash-resume (own short session)."""
        async with self._session_factory() as session:
            row = await session.get(Case, case_id)
            if row is None:
                return None
            return PersistedCase(
                id=row.id,
                status=row.status,
                state_json=row.state_json,
                last_node=row.last_node,
                failure_json=row.failure_json,
            )

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
            if row is None:
                return None
            return PersistedCase(
                id=row.id,
                status=row.status,
                state_json=row.state_json,
                last_node=row.last_node,
                failure_json=row.failure_json,
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
