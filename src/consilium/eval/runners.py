"""Ablation stage runner + run store (plan Section 13.1; AGENTS.md §19).

Stages (§19.1) — only the architecture varies; cases, recorded CADResults,
model snapshots and prompt versions are identical across stages:

  S1  single agent, full context         (``prompts/single_agent``, eval-only)
  S2  multi-agent, SHARED context        (every role sees every fact)
  S3  multi-agent, ROUTED context        (configs/routing.yaml)
  S4  routed + verification              (S3 + deterministic verifier FIRST,
                                          then LLM critic, bounded loop)

The stages reuse the production agents (``BaseSpecialist``, ``Director``,
``Critic``) and the production audit loop (``verify.merge.run_audit_loop``)
unchanged — the harness never re-implements pipeline logic, so what is
measured is what ships. Every LLM call goes through ``MeteredLLM``, which
records model, prompt version, tokens, latency and attempts per call
(counts only, never bodies — I-6).

Fail closed (I-5): a stage run that fails (specialist failure, validation
exhaustion, token budget) is STORED with its failure reason — never dropped,
never retried silently. Only ``AppError`` is converted here; anything else is
a harness bug and propagates loudly.

Run store: one immutable JSON document per run under ``<root>/<run_id>/
run.json`` (default root: gitignored ``data/eval_runs``), carrying git SHA,
config hashes, routing versions, model/prompt versions, dataset id+version,
seed, samples, LLM backend, and every per-(case, stage, sample) outcome. All
metrics and tables are computed FROM a stored run (I-11). The CLI lives in
``eval/ablation.py`` (run → score → store).
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from consilium.agents.base import BaseSpecialist
from consilium.agents.critic_agent import Critic
from consilium.agents.director import Director
from consilium.agents.specialists import SPECIALIST_PROMPT_VERSION, SPECIALIST_ROLES, run_all
from consilium.agents.templates import render_facts
from consilium.context.builder import build
from consilium.context.router import RoleRouting, RoutingConfig, load_routing_config, route
from consilium.core.clock import Clock, SystemClock
from consilium.core.errors import AppError
from consilium.core.settings import DEFAULT_MODEL_SNAPSHOT, Settings
from consilium.eval.datasets import (
    SMOKE_MANIFEST_PATH,
    EvalCase,
    LoadedManifest,
    Split,
    load_manifest,
)
from consilium.eval.fixtures import SYNTHETIC_LLM_BACKEND, synthetic_llm
from consilium.llm.budget import TokenBudget
from consilium.llm.client import ChatMessage, LLMClient, ParsedResponse
from consilium.llm.prompt_loader import Prompt, PromptLoader
from consilium.llm.retry import generate_structured
from consilium.schemas.agents import DirectorDraft, SubReport
from consilium.schemas.audit import CriticVerdict, Issue
from consilium.schemas.context import GlobalContext, RoleContext
from consilium.schemas.enums import Role
from consilium.verify.critic import run_critic
from consilium.verify.merge import run_audit_loop
from consilium.vision.config import Thresholds, load_thresholds

T = TypeVar("T", bound=BaseModel)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "eval_runs"
DIRECTOR_PROMPT_VERSION = "v1"
CRITIC_PROMPT_VERSION = "v1"
SINGLE_AGENT_PROMPT_VERSION = "v1"
DEFAULT_SAMPLES = 3  # §19.3: ≥3 samples per case for stochastic LLM paths
DEFAULT_SEED = 13

# smoke: synthetic set + synthetic LLM | live: synthetic set + live LLM
# full: recorded full set + live LLM
Mode = Literal["smoke", "live", "full"]


class Stage(StrEnum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


STAGE_DESCRIPTIONS: dict[Stage, str] = {
    Stage.S1: "single agent, full context",
    Stage.S2: "multi-agent, shared context",
    Stage.S3: "multi-agent, routed context",
    Stage.S4: "routed + verification (full CONSILIUM)",
}
ALL_STAGES: tuple[Stage, ...] = tuple(Stage)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentSetting:
    model: str
    temperature: float


@dataclass(frozen=True)
class EvalModels:
    """Model snapshots + generation settings, identical across stages."""

    specialist: AgentSetting
    director: AgentSetting
    critic: AgentSetting
    single_agent: AgentSetting  # S1 baseline: same snapshot/setting as the Director
    critic_max_rounds: int = 3
    timeout_s: float = 60.0
    token_budget: int = 100_000

    @classmethod
    def defaults(cls) -> EvalModels:
        """Settings-class defaults (smoke runs need no secrets)."""
        return cls(
            specialist=AgentSetting(DEFAULT_MODEL_SNAPSHOT, 0.2),
            director=AgentSetting(DEFAULT_MODEL_SNAPSHOT, 0.2),
            critic=AgentSetting(DEFAULT_MODEL_SNAPSHOT, 0.0),
            single_agent=AgentSetting(DEFAULT_MODEL_SNAPSHOT, 0.2),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> EvalModels:
        director = AgentSetting(settings.llm_model_director, settings.llm_temperature_director)
        return cls(
            specialist=AgentSetting(
                settings.llm_model_specialist, settings.llm_temperature_specialist
            ),
            director=director,
            critic=AgentSetting(settings.llm_model_critic, settings.llm_temperature_critic),
            single_agent=director,
            critic_max_rounds=settings.critic_max_rounds,
            timeout_s=settings.llm_timeout_s,
            token_budget=settings.llm_case_token_budget,
        )


def shared_routing() -> RoutingConfig:
    """S2 routing: every role is granted every fact (MedChat-like baseline)."""
    return RoutingConfig(
        version=hashlib.sha256(b"consilium-eval-shared-context:*").hexdigest(),
        roles={role: RoleRouting(allow=("*",)) for role in SPECIALIST_ROLES},
    )


@dataclass(frozen=True)
class PromptSet:
    specialists: dict[Role, Prompt]
    director: Prompt
    critic: Prompt
    single_agent: Prompt

    @classmethod
    def load(cls, loader: PromptLoader | None = None) -> PromptSet:
        loader = loader or PromptLoader()
        return cls(
            specialists={
                role: loader.load(role.value, SPECIALIST_PROMPT_VERSION)
                for role in SPECIALIST_ROLES
            },
            director=loader.load("director", DIRECTOR_PROMPT_VERSION),
            critic=loader.load("critic", CRITIC_PROMPT_VERSION),
            single_agent=loader.load("single_agent", SINGLE_AGENT_PROMPT_VERSION),
        )

    def versions(self) -> dict[str, str]:
        """``component -> vN@sha12`` (I-9: exact prompt text is traceable)."""
        prompts = [*self.specialists.values(), self.director, self.critic, self.single_agent]
        return {
            p.component: f"{p.version}@{hashlib.sha256(p.content.encode()).hexdigest()[:12]}"
            for p in prompts
        }


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------
class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMCallRecord(_Record):
    """One LLM call: identity + usage counts only (never bodies, I-6)."""

    agent: str
    model: str
    prompt_version: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    attempts: int


@dataclass
class MeteredLLM:
    """``LLMClient`` wrapper recording usage per call for the run store."""

    inner: LLMClient
    calls: list[LLMCallRecord] = field(default_factory=list)

    async def generate(
        self,
        messages: list[ChatMessage],
        response_model: type[T],
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
        tags: dict[str, str],
    ) -> ParsedResponse[T]:
        resp = await self.inner.generate(
            messages,
            response_model,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            tags=tags,
        )
        self.calls.append(
            LLMCallRecord(
                agent=tags.get("agent", ""),
                model=resp.model,
                prompt_version=resp.prompt_version,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_ms=resp.latency_ms,
                attempts=resp.attempts,
            )
        )
        return resp


# ---------------------------------------------------------------------------
# Outcomes + run record
# ---------------------------------------------------------------------------
class StageOutcome(_Record):
    """One (case, stage, sample) run, stored whether it succeeded or not."""

    case_id: str
    stage: Stage
    sample: int
    status: Literal["ok", "failed"]
    failure_reason: str | None = None
    sub_reports: tuple[SubReport, ...] = ()
    draft: DirectorDraft | None = None
    # S4 only (None elsewhere): audit-loop verdict and rounds used.
    audit_approved: bool | None = None
    audit_rounds: int | None = None
    open_issues: tuple[Issue, ...] = ()
    llm_calls: tuple[LLMCallRecord, ...] = ()
    wall_ms: float = Field(ge=0)


class RunProvenance(_Record):
    run_id: str
    mode: Mode
    created_at: str  # ISO-8601 UTC
    git_sha: str
    git_dirty: bool
    dataset_id: str
    dataset_version: str
    split: str
    case_ids: tuple[str, ...]
    stages: tuple[Stage, ...]
    samples_per_case: int
    seed: int
    llm_backend: str  # "synthetic-recorded-v1" (smoke) | "openai" (live/full)
    config_hashes: dict[str, str]
    routing_version: str
    shared_routing_version: str
    models: dict[str, str]  # agent -> "<snapshot>@T=<temperature>"
    prompt_versions: dict[str, str]
    critic_max_rounds: int


class RunRecord(_Record):
    """The stored run. ``metrics`` is filled from ``outcomes`` (Section 13.2)
    and ``extras`` carries additional eval stages (e.g. fault injection)."""

    provenance: RunProvenance
    outcomes: tuple[StageOutcome, ...]
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    extras: dict[str, JsonValue] = Field(default_factory=dict)


class RunStore:
    """Immutable, file-backed run store: ``<root>/<run_id>/run.json``."""

    def __init__(self, root: Path = DEFAULT_RUN_ROOT) -> None:
        self.root = root

    def path_for(self, run_id: str) -> Path:
        if not run_id or any(c in run_id for c in "/\\.") or run_id != run_id.strip():
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.root / run_id / "run.json"

    def save(self, record: RunRecord) -> Path:
        """Write once. A stored run is never overwritten (I-11 traceability)."""
        path = self.path_for(record.provenance.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as fh:
            fh.write(record.model_dump_json(indent=2) + "\n")
        return path

    def attach(self, run_id: str, name: str, payload: BaseModel) -> Path:
        """Write a write-once companion file next to a stored run (rubric
        packet/key/scores). The run itself is never modified."""
        if not name.replace("_", "").isalnum():
            raise ValueError(f"invalid attachment name: {name!r}")
        run_json = self.path_for(run_id)
        if not run_json.is_file():
            raise FileNotFoundError(f"no stored run {run_id}")
        path = run_json.parent / f"{name}.json"
        with path.open("x", encoding="utf-8") as fh:
            fh.write(payload.model_dump_json(indent=2) + "\n")
        return path

    def attach_text(self, run_id: str, filename: str, text: str) -> Path:
        """Write-once generated text next to a run (e.g. ``report.md``)."""
        stem, _, ext = filename.partition(".")
        if not stem.replace("_", "").isalnum() or ext not in ("md", "txt"):
            raise ValueError(f"invalid attachment name: {filename!r}")
        run_json = self.path_for(run_id)
        if not run_json.is_file():
            raise FileNotFoundError(f"no stored run {run_id}")
        path = run_json.parent / filename
        with path.open("x", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def read_attachment(self, run_id: str, name: str, model: type[T]) -> T | None:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"invalid attachment name: {name!r}")
        path = self.path_for(run_id).parent / f"{name}.json"
        return model.model_validate_json(path.read_bytes()) if path.is_file() else None

    def load(self, run_id: str) -> RunRecord:
        return RunRecord.model_validate_json(self.path_for(run_id).read_bytes())

    def run_ids(self) -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(sorted(p.parent.name for p in self.root.glob("*/run.json")))


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------
# Called once per (case, stage, sample). ``role_contexts`` is None for S1.
LLMFactory = Callable[[EvalCase, GlobalContext, dict[Role, RoleContext] | None, int], LLMClient]

_SINGLE_AGENT_INSTRUCTION = (
    "Write the DirectorDraft for the case data in the system message. "
    "Return ONLY the JSON object described there."
)
_DRAFT_MAX_OUTPUT_TOKENS = 4000


def global_context(case: EvalCase) -> GlobalContext:
    """C_global from the RECORDED CAD output — identical for every stage."""
    return build(case.cad, case.note, case.extraction, case.case_id)


@dataclass(frozen=True)
class StageRunner:
    thresholds: Thresholds
    routing: RoutingConfig
    shared: RoutingConfig
    models: EvalModels
    prompts: PromptSet

    def role_contexts(self, stage: Stage, ctx: GlobalContext) -> dict[Role, RoleContext] | None:
        if stage is Stage.S1:
            return None
        return route(ctx, self.shared if stage is Stage.S2 else self.routing)

    async def _single_agent(
        self, ctx: GlobalContext, llm: LLMClient, budget: TokenBudget
    ) -> DirectorDraft:
        prompt = self.prompts.single_agent
        resp = await generate_structured(
            llm,
            [
                ChatMessage(
                    role="system", content=prompt.render({"facts_block": render_facts(ctx)})
                ),
                ChatMessage(role="user", content=_SINGLE_AGENT_INSTRUCTION),
            ],
            DirectorDraft,
            model=self.models.single_agent.model,
            temperature=self.models.single_agent.temperature,
            max_output_tokens=_DRAFT_MAX_OUTPUT_TOKENS,
            timeout_s=self.models.timeout_s,
            tags={"agent": "single_agent", "prompt_version": prompt.version},
            budget=budget,
        )
        return resp.data.model_copy(update={"revision": 0})

    async def run(
        self,
        stage: Stage,
        case: EvalCase,
        ctx: GlobalContext,
        llm: LLMClient,
        sample: int,
    ) -> StageOutcome:
        metered = MeteredLLM(llm)
        budget = TokenBudget(limit_tokens=self.models.token_budget)
        started = time.perf_counter()
        reports: tuple[SubReport, ...] = ()
        draft: DirectorDraft | None = None
        approved: bool | None = None
        rounds: int | None = None
        open_issues: tuple[Issue, ...] = ()
        try:
            contexts = self.role_contexts(stage, ctx)
            if contexts is None:
                draft = await self._single_agent(ctx, metered, budget)
            else:
                reports, draft, approved, rounds, open_issues = await self._multi_agent(
                    stage, ctx, contexts, metered, budget
                )
        except AppError as exc:
            return StageOutcome(
                case_id=case.case_id,
                stage=stage,
                sample=sample,
                status="failed",
                failure_reason=exc.reason,
                sub_reports=reports,
                llm_calls=tuple(metered.calls),
                wall_ms=(time.perf_counter() - started) * 1000,
            )
        return StageOutcome(
            case_id=case.case_id,
            stage=stage,
            sample=sample,
            status="ok",
            sub_reports=reports,
            draft=draft,
            audit_approved=approved,
            audit_rounds=rounds,
            open_issues=open_issues,
            llm_calls=tuple(metered.calls),
            wall_ms=(time.perf_counter() - started) * 1000,
        )

    async def _multi_agent(
        self,
        stage: Stage,
        ctx: GlobalContext,
        contexts: dict[Role, RoleContext],
        llm: LLMClient,
        budget: TokenBudget,
    ) -> tuple[tuple[SubReport, ...], DirectorDraft, bool | None, int | None, tuple[Issue, ...]]:
        m = self.models
        specialists = {
            role: BaseSpecialist(
                role=role,
                prompt=self.prompts.specialists[role],
                llm=llm,
                model=m.specialist.model,
                temperature=m.specialist.temperature,
                timeout_s=m.timeout_s,
                budget=budget,
            )
            for role in SPECIALIST_ROLES
        }
        by_role = {role: r.data for role, r in (await run_all(contexts, specialists)).items()}
        reports = tuple(by_role[role] for role in SPECIALIST_ROLES)
        director = Director(
            prompt=self.prompts.director,
            llm=llm,
            model=m.director.model,
            temperature=m.director.temperature,
            timeout_s=m.timeout_s,
            budget=budget,
        )
        draft = (await director.draft(by_role, ctx)).data
        if stage is not Stage.S4:
            return reports, draft, None, None, ()

        critic = Critic(
            prompt=self.prompts.critic,
            llm=llm,
            model=m.critic.model,
            temperature=m.critic.temperature,
            timeout_s=m.timeout_s,
            budget=budget,
        )

        async def critic_fn(d: DirectorDraft, g: GlobalContext, r: int) -> CriticVerdict:
            return await run_critic(d, g, r, critic)

        async def revise_fn(d: DirectorDraft, issues: tuple[Issue, ...], r: int) -> DirectorDraft:
            return (await director.revise(d, issues, r, by_role, ctx)).data

        result = await run_audit_loop(
            draft=draft,
            global_ctx=ctx,
            sub_reports=reports,
            thresholds=self.thresholds,
            routing=self.routing,
            run_critic=critic_fn,
            revise=revise_fn,
            max_rounds=m.critic_max_rounds,
        )
        return reports, result.draft, result.approved, result.rounds, result.open_issues


# ---------------------------------------------------------------------------
# Provenance + orchestration
# ---------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["git", *args],  # noqa: S607 - git resolved from PATH by design
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> tuple[str, bool]:
    """(HEAD sha, working tree dirty). ``unknown`` when git is unavailable."""
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return (sha or "unknown", bool(status) if status is not None else True)


def config_hashes(configs_dir: Path = CONFIGS_DIR) -> dict[str, str]:
    """sha256 of every config that can change an eval outcome."""
    return {
        name: hashlib.sha256((configs_dir / name).read_bytes()).hexdigest()
        for name in ("routing.yaml", "thresholds.yaml", "models.yaml", "eval.yaml")
    }


def _setting(s: AgentSetting) -> str:
    return f"{s.model}@T={s.temperature}"


@dataclass(frozen=True)
class AblationPlan:
    loaded: LoadedManifest
    mode: Mode
    llm_backend: str
    llm_factory: LLMFactory
    stages: tuple[Stage, ...] = ALL_STAGES
    samples: int = DEFAULT_SAMPLES
    seed: int = DEFAULT_SEED
    split: Split | None = None


async def run_ablation(
    plan: AblationPlan,
    runner: StageRunner,
    *,
    clock: Clock | None = None,
    on_outcome: Callable[[StageOutcome], Awaitable[None]] | None = None,
) -> RunRecord:
    """Run every stage on every case x sample. Cases and their GlobalContext
    are built once and shared across stages (paired design, §19.3)."""
    if plan.samples < 1:
        raise ValueError("samples must be >= 1")
    clock = clock or SystemClock()
    created = clock.now()
    cases = plan.loaded.manifest.select(plan.split)
    outcomes: list[StageOutcome] = []
    for case in cases:
        ctx = global_context(case)
        for stage in plan.stages:
            contexts = runner.role_contexts(stage, ctx)
            for sample in range(plan.samples):
                llm = plan.llm_factory(case, ctx, contexts, sample)
                outcome = await runner.run(stage, case, ctx, llm, sample)
                outcomes.append(outcome)
                if on_outcome is not None:
                    await on_outcome(outcome)

    sha, dirty = git_state()
    m = runner.models
    body = {
        "mode": plan.mode,
        "created_at": created.isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "dataset_id": plan.loaded.manifest.dataset_id,
        "dataset_version": plan.loaded.dataset_version,
        "split": plan.split or "all",
        "case_ids": tuple(c.case_id for c in cases),
        "stages": plan.stages,
        "samples_per_case": plan.samples,
        "seed": plan.seed,
        "llm_backend": plan.llm_backend,
        "config_hashes": config_hashes(),
        "routing_version": runner.routing.version,
        "shared_routing_version": runner.shared.version,
        "models": {
            "specialist": _setting(m.specialist),
            "director": _setting(m.director),
            "critic": _setting(m.critic),
            "single_agent": _setting(m.single_agent),
        },
        "prompt_versions": runner.prompts.versions(),
        "critic_max_rounds": m.critic_max_rounds,
    }
    digest = hashlib.sha256(
        RunProvenance.model_validate({"run_id": "pending", **body}).model_dump_json().encode()
        + b"".join(o.model_dump_json(exclude={"wall_ms"}).encode() for o in outcomes)
    ).hexdigest()[:8]
    run_id = f"eval-{plan.mode}-{created.strftime('%Y%m%dT%H%M%SZ')}-{digest}"
    return RunRecord(
        provenance=RunProvenance.model_validate({"run_id": run_id, **body}),
        outcomes=tuple(outcomes),
    )


def default_runner(models: EvalModels | None = None) -> StageRunner:
    return StageRunner(
        thresholds=load_thresholds(),
        routing=load_routing_config(CONFIGS_DIR / "routing.yaml"),
        shared=shared_routing(),
        models=models or EvalModels.defaults(),
        prompts=PromptSet.load(),
    )


def synthetic_factory(seed: int) -> LLMFactory:
    """Smoke LLM factory: fresh synthetic recorded client per run (offline)."""

    def factory(
        case: EvalCase,
        ctx: GlobalContext,
        contexts: dict[Role, RoleContext] | None,
        sample: int,
    ) -> LLMClient:
        return synthetic_llm(
            case=case, global_ctx=ctx, role_contexts=contexts, seed=seed, sample=sample
        )

    return factory


def smoke_plan(
    *, samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED, path: Path = SMOKE_MANIFEST_PATH
) -> AblationPlan:
    return AblationPlan(
        loaded=load_manifest(path),
        mode="smoke",
        llm_backend=SYNTHETIC_LLM_BACKEND,
        llm_factory=synthetic_factory(seed),
        samples=samples,
        seed=seed,
    )
