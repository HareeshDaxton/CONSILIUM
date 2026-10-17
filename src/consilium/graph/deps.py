"""Deps — everything a graph node needs, injected (§21: no global state).

``build_prod_deps`` is the ONLY place real clients are constructed
(ARCHITECTURE.md §5.2); tests and local dev use ``build_fake_deps``
(FakeLLMClient + FakeVisionClient + in-memory image store, §20 offline rule).

The output rail is a callable seam: Phase 10 wires the real NeMo/deterministic
output checks behind it; until then the default is a pass-through that only
asserts the draft exists. The node still exists so the state machine matches
§4 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from consilium.agents.base import BaseSpecialist
from consilium.agents.critic_agent import Critic
from consilium.agents.director import Director
from consilium.agents.specialists import build_specialists
from consilium.context.router import RoutingConfig, load_routing_config
from consilium.core.settings import Settings
from consilium.guardrails.input_checks import InputRailConfig, load_input_rail
from consilium.guardrails.nemo_wiring import FakeRailEngine, GuardedRails
from consilium.guardrails.output_checks import OutputRailConfig, load_output_rail
from consilium.intake.deidentify import PhiPolicy, load_phi_patterns
from consilium.intake.extractor import load_extractor_prompt
from consilium.llm.budget import TokenBudget
from consilium.llm.client import LLMClient
from consilium.llm.prompt_loader import PromptLoader
from consilium.observability.instrument import InstrumentedLLM, InstrumentedVision
from consilium.observability.langfuse_sink import LangfuseSink, sink_from_settings
from consilium.observability.tracing import configure_tracing
from consilium.schemas.enums import Role
from consilium.vision.client import VisionClient
from consilium.vision.config import (
    ModelsConfig,
    Thresholds,
    load_models_config,
    load_thresholds,
)

__all__ = ["Deps", "ImageStore", "MemoryImageStore", "build_fake_deps", "build_prod_deps"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTING_PATH = _REPO_ROOT / "configs" / "routing.yaml"


class ImageStore(Protocol):
    """Object-store seam (§25: local FS in dev ↔ S3-compatible in prod).

    Image bytes are loaded by the node that needs them, never carried in
    CaseState (I-14; state_json must stay free of image data).
    """

    async def load(self, uri: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class MemoryImageStore:
    """Test/dev stand-in: uri → bytes mapping kept in memory."""

    images: dict[str, bytes] = field(default_factory=dict)

    async def load(self, uri: str) -> bytes:
        try:
            return self.images[uri]
        except KeyError:
            from consilium.core.errors import InputRejected

            raise InputRejected("IMAGE_NOT_FOUND", detail=f"no object at {uri}") from None


@dataclass(frozen=True)
class Deps:
    settings: Settings
    thresholds: Thresholds
    models: ModelsConfig
    routing: RoutingConfig
    input_rail: InputRailConfig
    phi_policy: PhiPolicy
    extractor_prompt: str
    vision: VisionClient
    image_store: ImageStore
    llm: LLMClient
    specialists: dict[Role, BaseSpecialist]
    director: Director
    critic: Critic
    output_policy: OutputRailConfig  # deterministic output checks (§14)
    rails: GuardedRails  # model-based (NeMo) layer; deterministic checks run first
    budget: TokenBudget | None = None


def _assemble(
    settings: Settings,
    *,
    llm: LLMClient,
    vision: VisionClient,
    image_store: ImageStore,
    rails: GuardedRails,
    fixture: str | None,
    sink: LangfuseSink | None = None,
    instrument: bool = False,
) -> Deps:
    if instrument:
        # Observability decorators (§18): metrics + spans + Langfuse metadata
        # on every LLM/vision call, counts only. Real deps always; test kits
        # keep the bare fakes so they can inspect them.
        llm = InstrumentedLLM(llm, sink)
        vision = InstrumentedVision(vision)
    loader = PromptLoader()
    return Deps(
        settings=settings,
        thresholds=load_thresholds(),
        models=load_models_config(),
        routing=load_routing_config(DEFAULT_ROUTING_PATH),
        input_rail=load_input_rail(),
        phi_policy=load_phi_patterns(),
        extractor_prompt=load_extractor_prompt(),
        vision=vision,
        image_store=image_store,
        llm=llm,
        specialists=build_specialists(settings=settings, llm=llm, fixture=fixture),
        director=Director(
            prompt=loader.load("director", "v1"),
            llm=llm,
            model=settings.llm_model_director,
            temperature=settings.llm_temperature_director,
            fixture=fixture,
        ),
        critic=Critic(
            prompt=loader.load("critic", "v1"),
            llm=llm,
            model=settings.llm_model_critic,
            temperature=settings.llm_temperature_critic,
            fixture=fixture,
        ),
        output_policy=load_output_rail(),
        rails=rails,
    )


def build_fake_deps(
    settings: Settings,
    *,
    llm: LLMClient,
    vision: VisionClient,
    image_store: ImageStore | None = None,
    rails: GuardedRails | None = None,
) -> Deps:
    """Fully offline deps: fake clients, committed configs, no network (§20)."""
    return _assemble(
        settings,
        llm=llm,
        vision=vision,
        image_store=image_store or MemoryImageStore(),
        # offline: deterministic fake engine that never triggers unless told to
        rails=rails or GuardedRails(FakeRailEngine()),
        fixture="default",
    )


def build_prod_deps(settings: Settings, *, image_store: ImageStore) -> Deps:
    """Real clients (OpenAI + HTTP vision endpoint + NeMo). Constructed HERE only
    (ARCHITECTURE.md §5.2). Fails closed on any missing configuration."""
    from consilium.guardrails.nemo_wiring import NemoRailEngine
    from consilium.llm.client_openai import OpenAILLMClient
    from consilium.vision.client_http import HttpVisionClient

    if settings.vision_client_mode != "http":
        raise ValueError("prod deps require VISION_CLIENT_MODE=http (fake is tests/dev only)")
    configure_tracing(settings)
    assert settings.vision_endpoint_url is not None  # enforced by Settings validator
    assert settings.vision_endpoint_token is not None
    assert settings.vision_pinned_revision is not None
    if settings.vision_pinned_revision != load_models_config().endpoint.pinned_revision:
        raise ValueError("VISION_PINNED_REVISION != configs/models.yaml pin (fail closed)")
    vision = HttpVisionClient(
        endpoint_url=settings.vision_endpoint_url,
        token=settings.vision_endpoint_token.get_secret_value(),
        pinned_revision=settings.vision_pinned_revision,
    )
    # Model rail: real NeMo unless explicitly set to the dev pass-through
    # engine (deterministic input/output checks run regardless — §14).
    engine = (
        NemoRailEngine() if settings.rails_engine == "nemo" else FakeRailEngine()
    )  # passthrough
    return _assemble(
        settings,
        llm=OpenAILLMClient(settings.openai_api_key.get_secret_value()),
        vision=vision,
        image_store=image_store,
        rails=GuardedRails(engine),
        fixture=None,
        sink=sink_from_settings(settings),
        instrument=True,
    )
