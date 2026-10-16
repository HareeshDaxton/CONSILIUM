"""Section 12.4 — the offline dev stack seeds every walkthrough scenario
through the HTTP API + worker (no hand-written DB rows), idempotently.
Offline: canned LLM, fake-vision app over in-process HTTP, local Postgres."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType

from sqlalchemy import func, select

from consilium.api.auth import decode_token
from consilium.core.settings import Settings
from consilium.graph.state import CaseState
from consilium.persistence.models import Case
from consilium.persistence.object_store import MemoryObjectStore
from consilium.persistence.repositories import Repositories

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev_ui_stack.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dev_ui_stack", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(module)
    return module


async def test_seed_produces_every_walkthrough_state(
    settings: Settings, repos: Repositories
) -> None:
    stack_mod = _module()
    stack = await stack_mod.build_stack(settings, repos, MemoryObjectStore())
    cases = await stack_mod.seed(stack, settings, repos)
    assert set(cases) == set(stack_mod.SCENARIOS)

    async def state(name: str) -> tuple[str, CaseState]:
        persisted = await repos.load_case(uuid.UUID(cases[name]))
        assert persisted is not None
        assert persisted.created_by == stack_mod.SUBMITTER
        return persisted.status, CaseState.from_snapshot(persisted.state_json)

    for name in ("approve", "reject"):
        status, st = await state(name)
        assert status == "PENDING_REVIEW" and not st.unverified
    status, st = await state("unverified")
    assert status == "NEEDS_ATTENTION" and st.unverified
    status, st = await state("failed")
    assert status == "FAILED"
    assert st.failure is not None and st.failure.type == "VISION_REVISION_MISMATCH"

    # idempotent: re-seeding returns the same cases and creates nothing new
    assert await stack_mod.seed(stack, settings, repos) == cases
    async with repos._session_factory() as s:
        assert (await s.execute(select(func.count()).select_from(Case))).scalar_one() == 4


def test_dev_tokens_verify_and_encode_the_self_approval_identity(settings: Settings) -> None:
    stack_mod = _module()
    tokens = stack_mod.dev_tokens(settings)
    clin = decode_token(tokens["clinician"], settings)
    both = decode_token(tokens["submitter_clinician"], settings)
    assert clin.sub == stack_mod.CLINICIAN
    assert both.sub == stack_mod.SUBMITTER and len(both.roles) == 2
