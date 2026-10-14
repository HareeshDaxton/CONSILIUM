"""Alembic environment (async, asyncpg).

URL resolution order: ``-x db_url=...`` option, then ``DATABASE_URL`` env var,
then ``alembic.ini``. Migrations are the ONLY DDL path outside tests
(AGENTS.md §16).
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from consilium.persistence.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    if "db_url" in x_args:
        return x_args["db_url"]
    return os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url", ""))


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _url()
    connectable = async_engine_from_config(cfg, prefix="sqlalchemy.")
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


asyncio.run(run_migrations_online())
