"""API composition root (ARCHITECTURE.md §4.3, §5.2).

``AppDeps`` bundles everything the HTTP layer needs. Tests build it from fakes
(``FakeLLMClient`` + ``FakeVisionClient`` + memory object store + a local
Postgres); ``build_app_deps`` builds the real thing, delegating every real
client construction to ``graph.deps.build_prod_deps`` — the single site.
"""

from __future__ import annotations

from dataclasses import dataclass

from consilium.api.google_auth import (
    GoogleIdTokenVerifier,
    GoogleTokenVerifier,
    google_jwks_resolver,
)
from consilium.core.clock import Clock, SystemClock
from consilium.core.settings import Settings
from consilium.graph.build import GraphRunner
from consilium.graph.deps import Deps, build_prod_deps
from consilium.persistence.db import make_engine, make_session_factory
from consilium.persistence.encryption import AesGcmFieldEncryptor, parse_key
from consilium.persistence.object_store import ObjectStore, make_object_store
from consilium.persistence.repositories import Repositories

__all__ = ["AppDeps", "build_app_deps", "build_google_verifier", "build_repositories"]


@dataclass(frozen=True)
class AppDeps:
    settings: Settings
    graph: Deps
    repos: Repositories
    objects: ObjectStore
    runner: GraphRunner
    clock: Clock
    google: GoogleTokenVerifier | None = None  # None = Google sign-in disabled


def build_repositories(settings: Settings) -> Repositories:
    if settings.field_encryption_key is None:
        raise ValueError("FIELD_ENCRYPTION_KEY is required to serve (I-6: note at rest)")
    key = parse_key(settings.field_encryption_key.get_secret_value())
    factory = make_session_factory(make_engine(settings.database_url))
    return Repositories(factory, AesGcmFieldEncryptor(key))


def build_app_deps(settings: Settings | None = None) -> AppDeps:
    settings = settings or Settings()  # env-populated
    objects = make_object_store(settings.object_store_uri)
    graph = build_prod_deps(settings, image_store=objects)
    repos = build_repositories(settings)
    return AppDeps(
        settings=settings,
        graph=graph,
        repos=repos,
        objects=objects,
        runner=GraphRunner(graph, repos),
        clock=SystemClock(),
        google=build_google_verifier(settings),
    )


def build_google_verifier(settings: Settings) -> GoogleTokenVerifier | None:
    """Real verifier (Google's published keys) when GOOGLE_CLIENT_ID is set."""
    if not settings.google_client_id:
        return None
    return GoogleIdTokenVerifier(settings.google_client_id, google_jwks_resolver())
