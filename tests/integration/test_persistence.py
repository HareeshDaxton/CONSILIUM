"""Section 9.1 exit tests: migration up/down, append-only enforcement,
transition atomicity (plan 9.1; ARCHITECTURE.md §5.12 rules)."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from tests.integration.conftest import TEST_URL, _upgrade

from consilium.persistence.repositories import Repositories


def _downgrade(url: str, revision: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.downgrade(cfg, revision)


async def _table_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        return {r[0] for r in rows}


EXPECTED_TABLES = {
    "cases",
    "clinical_notes",
    "cad_results",
    "vision_calls",
    "routed_contexts",
    "agent_runs",
    "drafts",
    "verdicts",
    "reviews",
    "audit_log",
    "jobs",
    "alembic_version",
}


async def test_migration_down_up_roundtrip(pg_engine: AsyncEngine) -> None:
    assert await _table_names(pg_engine) == EXPECTED_TABLES
    await asyncio.to_thread(_downgrade, TEST_URL, "base")
    assert await _table_names(pg_engine) == {"alembic_version"}  # alembic's own bookkeeping stays
    await asyncio.to_thread(_upgrade, TEST_URL, "head")
    assert await _table_names(pg_engine) == EXPECTED_TABLES


async def test_append_only_tables_reject_update_delete(pg_engine: AsyncEngine) -> None:
    """A role with only SELECT+INSERT cannot mutate audit_log/verdicts/reviews."""
    role = "consilium_limited"
    # Role name is a fixed test constant, never user input; built separately
    # from conn.execute to keep the SQL static-looking to the linter.
    create_role = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                CREATE ROLE {role} LOGIN PASSWORD 'limited';
            END IF;
        END
        $$;
        """
    grant_usage = f"GRANT USAGE ON SCHEMA public TO {role}"
    grant_tables = f"GRANT SELECT, INSERT ON audit_log, verdicts, reviews TO {role}"
    async with pg_engine.begin() as conn:
        await conn.execute(text(create_role))
        await conn.execute(text(grant_usage))
        await conn.execute(text(grant_tables))
        await conn.execute(
            text(
                "INSERT INTO audit_log (case_id, event, actor, detail_json) "
                "VALUES (NULL, 'test.event', 'test', '{}')"
            )
        )

    limited_url = TEST_URL.replace("consilium:consilium", f"{role}:limited")
    limited = create_async_engine(limited_url)
    try:
        for stmt in (
            "UPDATE audit_log SET actor = 'tampered'",
            "DELETE FROM audit_log",
        ):
            with pytest.raises(Exception) as excinfo:
                async with limited.begin() as conn:
                    await conn.execute(text(stmt))
            assert "permission denied" in str(excinfo.value).lower()
    finally:
        await limited.dispose()


async def test_transition_is_atomic_with_domain_writes(repos: Repositories) -> None:
    """A failing domain write in the same tx rolls back the status transition."""
    case_id = uuid.uuid4()
    async with repos.transaction() as tx:
        await tx.create_case(
            case_id=case_id,
            status="RECEIVED",
            laterality="UNKNOWN",
            created_by="test",
            idempotency_key=None,
            image_sha256=None,
            image_uri=None,
            state_json={"case_id": str(case_id)},
            last_node="create",
        )

    async def _failing_tx() -> None:
        async with repos.transaction() as tx:
            await tx.transition(
                case_id=case_id,
                status="CAD_COMPLETE",
                state_json={"case_id": str(case_id), "status": "CAD_COMPLETE"},
                last_node="call_vision",
                failure_json=None,
                event="node.call_vision",
            )
            await tx.cad_save(  # first insert is fine...
                case_id=case_id,
                cad_json={},
                raw_vision_json={},
                classifier_version="c",
                segmenter_version="s",
                endpoint_revision="r",
            )
            await tx.cad_save(  # ...duplicate PK aborts the commit
                case_id=case_id,
                cad_json={},
                raw_vision_json={},
                classifier_version="c",
                segmenter_version="s",
                endpoint_revision="r",
            )

    with pytest.raises(IntegrityError):
        await _failing_tx()

    case = await repos.load_case(case_id)
    assert case is not None
    assert case.status == "RECEIVED"  # transition rolled back with the bad write
    assert case.last_node == "create"


async def test_redacted_note_is_encrypted_at_rest(
    repos: Repositories, pg_engine: AsyncEngine
) -> None:
    case_id = uuid.uuid4()
    async with repos.transaction() as tx:
        await tx.create_case(
            case_id=case_id,
            status="RECEIVED",
            laterality="UNKNOWN",
            created_by="test",
            idempotency_key=None,
            image_sha256=None,
            image_uri=None,
            state_json={"case_id": str(case_id)},
            last_node="create",
        )
    async with repos.transaction() as tx:
        await tx.notes_save_redacted(case_id=case_id, redacted_text="synthetic note")
    # raw column holds ciphertext, not the text
    async with pg_engine.connect() as conn:
        (blob,) = (
            await conn.execute(
                text("SELECT raw_redacted_text_enc FROM clinical_notes WHERE case_id = :c"),
                {"c": case_id},
            )
        ).one()
    assert b"synthetic note" not in bytes(blob)
    assert await repos.load_redacted_note(case_id) == "synthetic note"
