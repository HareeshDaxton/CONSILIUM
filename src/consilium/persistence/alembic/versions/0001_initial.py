"""initial schema — all §5.12 tables + append-only privilege revocations

Revision ID: 0001_initial
Revises:
Create Date: 2026-01-01

Tables are created from ``models.Base.metadata`` (single DDL source).
Append-only enforcement (AGENTS.md §16): UPDATE/DELETE on
``audit_log``/``verdicts``/``reviews`` are revoked from the app DB role
(``APP_DB_ROLE`` env var, default ``consilium_app``). The revoke is skipped
when the role does not exist (fresh dev databases create it later); the
enforcement test proves the mechanism with a dedicated limited role.
"""

import os
from collections.abc import Sequence

from alembic import op

from consilium.persistence.models import APPEND_ONLY_TABLES, Base

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APP_ROLE = os.environ.get("APP_DB_ROLE", "consilium_app")


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)
    tables = ", ".join(APPEND_ONLY_TABLES)
    # Built separately from op.execute: role/table names are server-side
    # constants (env-controlled app role), never user input.
    revoke_sql = f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN
                EXECUTE 'REVOKE UPDATE, DELETE ON {tables} FROM {_APP_ROLE}';
            END IF;
        END
        $$;
        """
    op.execute(revoke_sql)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
