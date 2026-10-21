"""S1-S4 ablation scoring + CLI (plan Section 13.2; AGENTS.md §19.1/§19.3;
SKILLS.md ablation-run).

``score_run`` turns a run's stored outcomes into the ``metrics`` block of the
run record — nothing is computed from anything but the stored outcomes and
the frozen manifest they were produced from (the manifest's sha256 must equal
the run's ``dataset_version``, otherwise scoring refuses).

Statistics (§19.3):
- the unit of analysis is the CASE: samples (≥3 per case) are averaged within
  a case first, and their spread is reported as ``within_case_sd`` (LLM
  stochasticity is visible, never hidden in a single sample);
- 95% CIs are percentile bootstrap intervals resampling CASES (cluster
  bootstrap), seeded from ``configs/eval.yaml``;
- stage comparisons are PAIRED: per-case differences (later - earlier) on the
  identical case set, reported per case and as a mean with its own CI — not
  just two means side by side.

Small-N honesty: with the 6-case smoke set the intervals are wide and the
smoke responder is templated; smoke metrics validate the machinery and are
never quoted as findings (I-11).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, field_validator

from consilium.agents.critic_agent import Critic
from consilium.context.router import RoutingConfig
from consilium.core.settings import Settings
from consilium.eval.datasets import (
    FULL_MANIFEST_PATH,
    SMOKE_MANIFEST_PATH,
    CaseManifest,
    EvalCase,
    LoadedManifest,
    load_manifest,
    manifest_bytes,
)
from consilium.eval.diversity import S_INTER_DEFINITION, Embedder, HashedNgramEmbedder
from consilium.eval.fault_injection import ScriptedCritic, eval_corpus, fault_injection_stage
from consilium.eval.fixtures import RawEvalInput, record_cases
from consilium.eval.judge import (
    LLMJudge,
    RubricKey,
    RubricScores,
    build_packet,
    canned_judge_llm,
    validate_scores,
)
from consilium.eval.judge import dry_run as rubric_dry_run
from consilium.eval.metrics import (
    ALL_METRICS,
    UNSUPPORTED_TYPES,
    MetricRow,
    bootstrap_ci,
    percentile,
    score_outcome,
)
from consilium.eval.runners import (
    CONFIGS_DIR,
    DEFAULT_RUN_ROOT,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    AblationPlan,
    EvalModels,
    LLMFactory,
    RunRecord,
    RunStore,
    Stage,
    StageOutcome,
    StageRunner,
    default_runner,
    global_context,
    run_ablation,
    smoke_plan,
)
from consilium.llm.client import LLMClient
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import CriticVerdict
from consilium.schemas.context import GlobalContext, RoleContext
from consilium.schemas.enums import Laterality, Role
from consilium.schemas.note import ClinicalNote
from consilium.verify.critic import run_critic
from consilium.verify.merge import CriticFn
from consilium.vision.config import Thresholds, load_models_config, load_thresholds

EVAL_CONFIG_PATH = CONFIGS_DIR / "eval.yaml"


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resamples: int
    confidence: float
    bootstrap_seed: int
    paired: tuple[tuple[Stage, Stage], ...]
    pricing: dict[str, dict[str, float]]

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("confidence must be in (0, 1)")
        return v


def load_eval_config(path: Path = EVAL_CONFIG_PATH) -> EvalConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    boot = data["bootstrap"]
    return EvalConfig(
        resamples=boot["resamples"],
        confidence=boot["confidence"],
        bootstrap_seed=boot["seed"],
        paired=tuple((Stage(a), Stage(b)) for a, b in data["paired_comparisons"]),
        pricing=data.get("pricing_usd_per_1m_tokens") or {},
    )


def _case_means(
    rows: Sequence[MetricRow],
) -> dict[tuple[Stage, str], dict[str, tuple[float, float | None]]]:
    """(stage, metric) -> case -> (mean over defined samples, sample sd or None)."""
    samples: dict[tuple[Stage, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.failed:
            continue
        for metric, value in row.values.items():
            if value is not None:
                samples[(row.stage, metric)][row.case_id].append(value)
    out: dict[tuple[Stage, str], dict[str, tuple[float, float | None]]] = {}
    for key, per_case in samples.items():
        out[key] = {
            case: (float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else None)
            for case, vals in per_case.items()
        }
    return out


def score_run(
    record: RunRecord,
    loaded: LoadedManifest,
    *,
    thresholds: Thresholds,
    routing: RoutingConfig,
    embedder: Embedder | None = None,
    config: EvalConfig | None = None,
) -> RunRecord:
    """Return ``record`` with its ``metrics`` block computed from its outcomes."""
    if loaded.dataset_version != record.provenance.dataset_version:
        raise ValueError("manifest version does not match the run's dataset_version")
    emb: Embedder = embedder if embedder is not None else HashedNgramEmbedder()
    config = config or load_eval_config()
    cases = {c.case_id: c for c in loaded.manifest.cases}
    contexts = {cid: global_context(c) for cid, c in cases.items()}
    rows = [
        score_outcome(
            o,
            cases[o.case_id],
            contexts[o.case_id],
            thresholds=thresholds,
            routing=routing,
            embedder=emb,
            pricing=config.pricing,
        )
        for o in record.outcomes
    ]
    rng = np.random.default_rng(config.bootstrap_seed)

    def ci(values: Sequence[float]) -> list[JsonValue] | None:
        bounds = bootstrap_ci(
            values, resamples=config.resamples, confidence=config.confidence, rng=rng
        )
        return None if bounds is None else list(bounds)

    case_means = _case_means(rows)
    stages = record.provenance.stages

    per_stage: dict[str, JsonValue] = {}
    for stage in stages:
        stage_rows = [r for r in rows if r.stage is stage]
        block: dict[str, JsonValue] = {
            "outcomes": len(stage_rows),
            "failure_rate": (
                sum(r.failed for r in stage_rows) / len(stage_rows) if stage_rows else None
            ),
        }
        for metric in ALL_METRICS:
            per_case = case_means.get((stage, metric), {})
            means = [m for m, _ in per_case.values()]
            sds = [sd for _, sd in per_case.values() if sd is not None]
            block[metric] = {
                "mean": float(np.mean(means)) if means else None,
                "ci95": ci(means),
                "n_cases": len(means),
                "within_case_sd": float(np.mean(sds)) if sds else None,
            }
        ok = [r for r in stage_rows if not r.failed]
        for name in ("wall_ms", "llm_latency_ms"):
            vals = [v for r in ok if (v := r.values.get(name)) is not None]
            block[f"{name}_p50"] = percentile(vals, 50)
            block[f"{name}_p95"] = percentile(vals, 95)
        per_stage[stage.value] = block

    paired: dict[str, JsonValue] = {}
    for earlier, later in config.paired:
        if earlier not in stages or later not in stages:
            continue
        comparison: dict[str, JsonValue] = {}
        for metric in ALL_METRICS:
            a = case_means.get((earlier, metric), {})
            b = case_means.get((later, metric), {})
            shared = sorted(set(a) & set(b))
            diffs = {case: b[case][0] - a[case][0] for case in shared}
            comparison[metric] = {
                "mean_diff": float(np.mean(list(diffs.values()))) if diffs else None,
                "ci95": ci(list(diffs.values())),
                "n_cases": len(diffs),
                "per_case": dict(diffs),
            }
        paired[f"{later.value}-{earlier.value}"] = comparison

    unsupported_types: list[JsonValue] = []
    unsupported_types.extend(sorted(t.value for t in UNSUPPORTED_TYPES))
    metrics: dict[str, JsonValue] = {
        "instrument": {
            "claim_checks": "deterministic verifier (verify_draft) on every stage's final draft",
            "unsupported_issue_types": list(unsupported_types),
            "s_inter": S_INTER_DEFINITION,
            "embedding_model": f"{emb.name}@{emb.version}",
            "unit": "case (samples averaged within case)",
        },
        "bootstrap": {
            "method": "percentile, cluster (case) resampling",
            "resamples": config.resamples,
            "confidence": config.confidence,
            "seed": config.bootstrap_seed,
        },
        "per_stage": per_stage,
        "paired": paired,
        "rows": [row.model_dump(mode="json") for row in rows],
    }
    return record.model_copy(update={"metrics": metrics})


SCRIPTED_CRITIC_LABEL = (
    "scripted_stand_in: offline wiring measurement only (§20); real critic recall "
    "comes from a live run and is never claimed from smoke numbers (I-11)"
)
RUBRIC_DRY_RUN_PER_STAGE = 2


async def add_eval_stages(
    record: RunRecord,
    loaded: LoadedManifest,
    runner: StageRunner,
    *,
    critic: CriticFn,
    critic_label: str,
    judge: LLMJudge,
    config: EvalConfig | None = None,
) -> RunRecord:
    """Section 13.3 stages stored in ``extras``: fault injection per layer on
    a known-good corpus built from the SAME frozen cases, and the rubric
    pipeline dry run (synthetic raters + judge)."""
    config = config or load_eval_config()
    corpus = eval_corpus(loaded.manifest.cases, runner.thresholds, runner.routing)
    faults = await fault_injection_stage(
        corpus, runner.thresholds, runner.routing, critic=critic, critic_label=critic_label
    )
    rubric = await rubric_dry_run(
        record,
        loaded,
        judge=judge,
        per_stage=RUBRIC_DRY_RUN_PER_STAGE,
        seed=record.provenance.seed,
        resamples=config.resamples,
        confidence=config.confidence,
    )
    return record.model_copy(
        update={"extras": {**record.extras, "fault_injection": faults, "rubric_dry_run": rubric}}
    )


async def run_smoke(*, samples: int, seed: int) -> RunRecord:
    """Run S1-S4 offline on the smoke set, score it, add the 13.3 stages
    (scripted critic, canned judge). Not yet stored."""
    plan = smoke_plan(samples=samples, seed=seed)
    runner = default_runner()
    record = await run_ablation(plan, runner)
    record = score_run(record, plan.loaded, thresholds=runner.thresholds, routing=runner.routing)
    judge = LLMJudge(llm=canned_judge_llm(), model="canned-dry-run-judge", temperature=0.0)
    return await add_eval_stages(
        record,
        plan.loaded,
        runner,
        critic=ScriptedCritic(),
        critic_label=SCRIPTED_CRITIC_LABEL,
        judge=judge,
    )


LIVE_LLM_BACKEND = "openai"


def live_factory(llm: LLMClient) -> LLMFactory:
    """Live LLM factory: one shared client; every call is metered per run."""

    def factory(
        case: EvalCase,
        ctx: GlobalContext,
        contexts: dict[Role, RoleContext] | None,
        sample: int,
    ) -> LLMClient:
        return llm

    return factory


async def run_live(
    settings: Settings,
    *,
    mode: Literal["live", "full"],
    manifest_path: Path,
    samples: int,
    seed: int,
    llm: LLMClient | None = None,
    on_outcome: Callable[[StageOutcome], Awaitable[None]] | None = None,
) -> RunRecord:
    """Live-LLM ablation on a frozen manifest (recorded CAD; vision is NOT
    called). ``live`` = committed synthetic set; ``full`` = recorded full set.
    The fault-injection stage uses the REAL Critic (critic model/setting from
    settings) and the rubric dry run a real LLM judge (critic snapshot, T=0)."""
    from consilium.llm.client_openai import OpenAILLMClient

    client: LLMClient = llm or OpenAILLMClient(
        settings.openai_api_key.get_secret_value(), model_options=settings.llm_model_options()
    )
    models = EvalModels.from_settings(settings)
    runner = default_runner(models)
    plan = AblationPlan(
        loaded=load_manifest(manifest_path),
        mode=mode,
        llm_backend=LIVE_LLM_BACKEND,
        llm_factory=live_factory(client),
        samples=samples,
        seed=seed,
    )
    record = await run_ablation(plan, runner, on_outcome=on_outcome)
    record = score_run(record, plan.loaded, thresholds=runner.thresholds, routing=runner.routing)
    critic = Critic(
        prompt=runner.prompts.critic,
        llm=client,
        model=models.critic.model,
        temperature=models.critic.temperature,
        max_output_tokens=models.critic_max_output_tokens,
        timeout_s=models.timeout_s,
    )

    async def critic_fn(d: DirectorDraft, g: GlobalContext, r: int) -> CriticVerdict:
        return await run_critic(d, g, r, critic)

    judge = LLMJudge(llm=client, model=models.critic.model, temperature=models.critic.temperature)
    return await add_eval_stages(
        record,
        plan.loaded,
        runner,
        critic=critic_fn,
        critic_label=f"live Critic {models.critic.label()}",
        judge=judge,
    )


class _RecordSpec(BaseModel):
    """One entry of the ``record`` input spec (JSON list)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    image: Path
    laterality: Laterality
    split: Literal["dev", "holdout"] = "dev"
    note: ClinicalNote | None = None
    curated_disagreement_topics: tuple[str, ...] = ()


def record_full_set(settings: Settings, spec_path: Path, out: Path) -> Path:
    """Freeze the full set: call the pinned vision endpoint ONCE per image and
    write the manifest (never overwritten). Image paths are not logged (I-6)."""
    from consilium.vision.client_http import HttpVisionClient

    if out.exists():
        raise FileExistsError(f"{out} exists; a frozen set is never overwritten")
    if (
        settings.vision_endpoint_url is None
        or settings.vision_endpoint_token is None
        or settings.vision_pinned_revision is None
    ):
        raise ValueError("recording needs VISION_ENDPOINT_URL/TOKEN and VISION_PINNED_REVISION")
    models_cfg = load_models_config()
    if settings.vision_pinned_revision != models_cfg.endpoint.pinned_revision:
        raise ValueError("VISION_PINNED_REVISION != configs/models.yaml pin (fail closed)")
    specs = TypeAdapter(list[_RecordSpec]).validate_json(spec_path.read_bytes())
    client = HttpVisionClient(
        endpoint_url=settings.vision_endpoint_url,
        token=settings.vision_endpoint_token.get_secret_value(),
        pinned_revision=settings.vision_pinned_revision,
    )
    cases = asyncio.run(
        record_cases(
            [
                RawEvalInput(
                    case_id=s.case_id,
                    image=s.image.read_bytes(),
                    laterality=s.laterality,
                    note=s.note,
                    split=s.split,
                    curated_disagreement_topics=s.curated_disagreement_topics,
                )
                for s in specs
            ],
            client=client,
            thresholds=load_thresholds(),
            models=models_cfg,
        )
    )
    manifest = CaseManifest(
        dataset_id="consilium-full-v1",
        description="Recorded full eval set (vision called once, then frozen).",
        cases=cases,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        fh.write(manifest_bytes(manifest))
    return out


def eval_settings() -> Settings:
    """Settings for live/full ablations: LLM config from env/.env, vision
    client forced to ``fake`` — these runs read RECORDED CAD and never call
    the vision endpoint, so they must not depend on its configuration."""
    return Settings(vision_client_mode="fake")


def manifest_for(record: RunRecord) -> LoadedManifest:
    """The frozen manifest a stored run was produced from (version-checked)."""
    path = FULL_MANIFEST_PATH if record.provenance.mode == "full" else SMOKE_MANIFEST_PATH
    loaded = load_manifest(path)
    if loaded.dataset_version != record.provenance.dataset_version:
        raise ValueError(f"{path.name} no longer matches run {record.provenance.run_id}")
    return loaded


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="consilium.eval.ablation")
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN_ROOT, help="run store root")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="S1-S4 on the synthetic smoke set, offline")
    smoke.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    smoke.add_argument("--seed", type=int, default=DEFAULT_SEED)
    export = sub.add_parser("rubric-export", help="blinded clinician rubric packet + key")
    export.add_argument("--run-id", required=True)
    export.add_argument("--per-stage", type=int, default=10)
    export.add_argument("--seed", type=int, default=DEFAULT_SEED)
    live = sub.add_parser("live", help="S1-S4 on the frozen synthetic set with the LIVE LLM")
    full = sub.add_parser("full", help="S1-S4 on the recorded full set with the LIVE LLM")
    for p in (live, full):
        p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    full.add_argument("--manifest", type=Path, default=FULL_MANIFEST_PATH)
    rec = sub.add_parser("record", help="freeze the full set via the vision endpoint (once)")
    rec.add_argument("--spec", type=Path, required=True, help="JSON list of cases to record")
    rec.add_argument("--manifest", type=Path, default=FULL_MANIFEST_PATH)
    imp = sub.add_parser("rubric-import", help="validate + attach clinician rubric scores")
    imp.add_argument("--run-id", required=True)
    imp.add_argument("--scores", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    """CLI. ``smoke`` returns the stored run ID; rubric commands the file path."""
    args = _parse(argv)
    store = RunStore(args.out)
    if args.command == "smoke":
        record = asyncio.run(run_smoke(samples=args.samples, seed=args.seed))
        path = store.save(record)
        print(f"stored run {record.provenance.run_id} -> {path}")
        return record.provenance.run_id
    if args.command in ("live", "full"):
        path_in = SMOKE_MANIFEST_PATH if args.command == "live" else args.manifest
        if not path_in.is_file():
            raise FileNotFoundError(
                f"frozen manifest {path_in} not found; record it first (ablation record)"
            )
        done = 0

        async def progress(outcome: StageOutcome) -> None:
            nonlocal done
            done += 1
            print(
                f"[{done}] {outcome.case_id} {outcome.stage.value} s{outcome.sample}: "
                f"{outcome.status}",
                flush=True,
            )

        record = asyncio.run(
            run_live(
                eval_settings(),
                mode=args.command,
                manifest_path=path_in,
                samples=args.samples,
                seed=args.seed,
                on_outcome=progress,
            )
        )
        path = store.save(record)
        print(f"stored run {record.provenance.run_id} -> {path}")
        return record.provenance.run_id
    if args.command == "record":
        settings = Settings()  # from env / .env
        path = record_full_set(settings, args.spec, args.manifest)
        print(f"recorded frozen set -> {path}")
        return str(path)
    record = store.load(args.run_id)
    if args.command == "rubric-export":
        packet, key = build_packet(
            record, manifest_for(record), per_stage=args.per_stage, seed=args.seed
        )
        store.attach(args.run_id, "rubric_key", key)
        path = store.attach(args.run_id, "rubric_packet", packet)
        print(f"rubric packet ({len(packet.items)} items) -> {path}")
        return str(path)
    key_model = store.read_attachment(args.run_id, "rubric_key", RubricKey)
    if key_model is None:
        raise FileNotFoundError(f"run {args.run_id} has no rubric_key; export a packet first")
    scores = RubricScores.model_validate_json(args.scores.read_bytes())
    validate_scores(scores, key_model)
    path = store.attach(args.run_id, "rubric_scores", scores)
    print(f"{len(scores.scores)} rubric scores attached -> {path}")
    return str(path)


if __name__ == "__main__":
    main()
