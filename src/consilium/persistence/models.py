"""SQLAlchemy models — DDL per ARCHITECTURE.md §5.12 (Alembic-managed).

Rules (§5.12/§16):
  - JSONB columns mirror ``schemas/`` exactly (persisted-JSON contract —
    schema-change governs their evolution).
  - ``audit_log``, ``verdicts``, ``reviews`` are append-only: the migration
    revokes UPDATE/DELETE for the app DB role (enforcement test).
  - ``clinical_notes.raw_redacted_text_enc`` is encrypted at the column level
    by the repositories (``persistence/encryption.py``); the raw redacted
    note text lives ONLY there, never in ``cases.state_json`` (I-6).
  - ``clinical_notes.note_json`` / ``extraction_json`` are nullable: the graph
    persists the redacted text at input-rail time (so a crash before
    extraction is resumable) and completes the row at extraction time.

Only ``persistence/`` may import sqlalchemy/alembic (import-linter).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


APPEND_ONLY_TABLES: tuple[str, ...] = ("audit_log", "verdicts", "reviews")


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    laterality: Mapped[str] = mapped_column(Text, nullable=False, default="UNKNOWN")
    image_sha256: Mapped[str | None] = mapped_column(Text)
    image_uri: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(Text, unique=True)
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    last_node: Mapped[str | None] = mapped_column(Text)
    failure_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ClinicalNoteRow(Base):
    __tablename__ = "clinical_notes"

    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), primary_key=True)
    note_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    extraction_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    raw_redacted_text_enc: Mapped[bytes] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CadResultRow(Base):
    __tablename__ = "cad_results"

    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), primary_key=True)
    cad_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    raw_vision_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    classifier_version: Mapped[str] = mapped_column(Text, nullable=False)
    segmenter_version: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint_revision: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class VisionCallRow(Base):
    __tablename__ = "vision_calls"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), nullable=False)
    endpoint_revision: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    attempts: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RoutedContextRow(Base):
    __tablename__ = "routed_contexts"

    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), primary_key=True)
    role: Mapped[str] = mapped_column(Text, primary_key=True)
    routing_version: Mapped[str] = mapped_column(Text, nullable=False)
    facts_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AgentRunRow(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), nullable=False, index=True)
    agent: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DraftRow(Base):
    __tablename__ = "drafts"

    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    draft_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    producer_run_id: Mapped[str | None] = mapped_column(Text)
    unverified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    open_issues_json: Mapped[list[Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class VerdictRow(Base):  # append-only — migration revokes UPDATE/DELETE
    __tablename__ = "verdicts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), nullable=False)
    revision: Mapped[int | None] = mapped_column(Integer)
    round: Mapped[int | None] = mapped_column(Integer)
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    issues_json: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    layer: Mapped[str] = mapped_column(Text, nullable=False)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ReviewRow(Base):  # append-only — migration revokes UPDATE/DELETE
    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), nullable=False)
    draft_revision: Mapped[int | None] = mapped_column(Integer)
    reviewer_id: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    edits_diff_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    edited_draft_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    reason_codes: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    free_text_reason: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AuditLogRow(Base):  # append-only — migration revokes UPDATE/DELETE
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    case_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("cases.id"))
    event: Mapped[str] = mapped_column(Text, nullable=False)
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


Index("ix_audit_log_case_at", AuditLogRow.case_id, AuditLogRow.at)


class JobRow(Base):
    __tablename__ = "jobs"

    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"), primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claimed_by: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


Index("ix_jobs_status_run_at", JobRow.status, JobRow.run_at)
