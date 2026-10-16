"""Shared offline kit for API tests (Phase 11).

Builds ``AppDeps`` from fakes: FakeLLMClient + FakeVisionClient + in-memory
object store. ``Repositories`` points at the given database URL; the engine
is lazy, so pure auth/problem tests never open a connection. Tokens are
minted with the synthetic test secret — never a real credential.
"""

from __future__ import annotations

import httpx

from consilium.api.auth import ApiRole, issue_token
from consilium.api.deps import AppDeps
from consilium.core.clock import SystemClock
from consilium.core.settings import Settings
from consilium.graph.build import GraphRunner
from consilium.graph.deps import build_fake_deps
from consilium.llm.client import LLMClient
from consilium.llm.client_fake import FakeLLMClient
from consilium.persistence.db import make_engine, make_session_factory
from consilium.persistence.encryption import AesGcmFieldEncryptor
from consilium.persistence.object_store import MemoryObjectStore, ObjectStore
from consilium.persistence.repositories import Repositories
from consilium.vision.client import VisionClient
from consilium.vision.client_fake import FakeVisionClient
from consilium.vision.config import load_models_config

PIN = load_models_config().endpoint.pinned_revision


def make_app_deps(
    settings: Settings,
    *,
    repos: Repositories | None = None,
    llm: LLMClient | None = None,
    vision: VisionClient | None = None,
    objects: ObjectStore | None = None,
) -> AppDeps:
    objects = objects if objects is not None else MemoryObjectStore()
    graph = build_fake_deps(
        settings,
        llm=llm or FakeLLMClient({}),
        vision=vision or FakeVisionClient(revision=PIN),
        image_store=objects,
    )
    if repos is None:
        repos = Repositories(
            make_session_factory(make_engine(settings.database_url)),
            AesGcmFieldEncryptor(b"k" * 32),
        )
    return AppDeps(
        settings=settings,
        graph=graph,
        repos=repos,
        objects=objects,
        runner=GraphRunner(graph, repos),
        clock=SystemClock(),
    )


def bearer(settings: Settings, sub: str, *roles: ApiRole) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(sub, roles, settings)}"}


def asgi_client(app: object) -> httpx.AsyncClient:
    """Async client on the SAME event loop as the asyncpg engine (TestClient
    would run the app on a different loop)."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://testserver",
    )
