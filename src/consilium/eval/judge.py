"""Clinician rubric tooling + optional LLM judge (plan Section 13.3;
AGENTS.md §19.2 "Clinical quality"; SKILLS.md ablation-run step 5).

Flow for a stored run:

1. ``build_packet`` — deterministic, stage-stratified sample of successful
   drafts, rendered WITHOUT stage, roles or claim IDs and shuffled by opaque
   item ID, so raters are blind to the architecture that produced each
   report. The un-blinding ``RubricKey`` is written separately.
2. Clinicians score each item on the rubric (1-5 per dimension) →
   ``RubricScore`` rows (validated strictly on import).
3. ``summarize`` — un-blind via the key and aggregate per stage with
   case-agnostic bootstrap CIs over items.
4. Optional ``LLMJudge`` (eval-only prompt ``prompts/judge/v1.md``) scores
   the SAME items; ``judge_agreement`` compares it with the clinician scores
   item by item (MAE, exact and within-one agreement, Spearman). The judge
   may supplement, never replace, clinicians; its model + setting are stored
   with every judge score.

``dry_run`` exercises the whole pipeline offline with SYNTHETIC rater scores
and a canned judge — it validates the machinery and is labelled as such in
the run record; it is never a clinical-quality result (I-11).

# CLINICAL-REVIEW: rubric dimensions, the 1-5 scale and the anchor wording
# in prompts/judge/v1.md are a first proposal; the clinical co-authors must
# confirm them (and the per-stage sample size) before real scoring starts.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, JsonValue

from consilium.agents.templates import render_facts
from consilium.core.errors import AppError
from consilium.eval.datasets import LoadedManifest
from consilium.eval.fixtures import pick
from consilium.eval.metrics import bootstrap_ci
from consilium.eval.runners import RunRecord, Stage, global_context
from consilium.llm.client import ChatMessage, LLMClient
from consilium.llm.client_fake import FakeLLMClient
from consilium.llm.prompt_loader import Prompt, PromptLoader
from consilium.llm.retry import generate_structured
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.base import Strict

RUBRIC_DIMENSIONS: tuple[str, ...] = ("accuracy", "completeness", "safety", "usefulness")
JUDGE_PROMPT_VERSION = "v1"
DRY_RUN_LABEL = "SYNTHETIC rater scores + canned judge: pipeline dry run, NOT clinical data"

Score = int  # 1..5


class RubricItem(Strict):
    """One blinded item: fact sheet + report, no stage/roles/claim IDs."""

    item_id: str
    facts_block: str
    report_block: str


class RubricPacket(Strict):
    run_id: str
    instructions: str
    dimensions: tuple[str, ...]
    items: tuple[RubricItem, ...]


class RubricKeyEntry(Strict):
    item_id: str
    case_id: str
    stage: Stage
    sample: int


class RubricKey(Strict):
    """Un-blinding key — kept away from raters."""

    run_id: str
    entries: tuple[RubricKeyEntry, ...]


class RubricScore(Strict):
    item_id: str
    rater_id: str = Field(min_length=1, max_length=64)  # opaque, never a name
    rater_kind: Literal["clinician", "llm_judge", "synthetic"]
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)

    def value(self, dimension: str) -> int:
        score = getattr(self, dimension)
        assert isinstance(score, int)
        return score


class RubricScores(Strict):
    run_id: str
    scores: tuple[RubricScore, ...]


class JudgeOutput(Strict):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    rationale: str = Field(max_length=400)


def render_report(draft: DirectorDraft) -> str:
    """Blinded rendering: content only — no stage, role attribution or IDs."""
    lines = ["[impression]", draft.impression, "[findings]"]
    for c in draft.claims:  # numeric claims show the value they assert
        stated = f" [stated value: {c.numeric_value}]" if c.numeric_value is not None else ""
        lines.append(f"- {c.text}{stated}")
    if draft.disagreements:
        lines.append("[unresolved differences of opinion]")
        for d in draft.disagreements:
            lines.append(f"- {d.topic}: " + " | ".join(pos for _, pos in d.positions))
    lines.append("[recommendations]")
    lines += [f"- {r.text}" for r in draft.recommendations]
    lines.append("[limitations]")
    lines += [f"- {item}" for item in draft.limitations]
    return "\n".join(lines)


def _item_id(run_id: str, case_id: str, stage: Stage, sample: int, seed: int) -> str:
    raw = f"{run_id}|{case_id}|{stage.value}|{sample}|{seed}".encode()
    return "item-" + hashlib.sha256(raw).hexdigest()[:12]


def build_packet(
    record: RunRecord, loaded: LoadedManifest, *, per_stage: int, seed: int
) -> tuple[RubricPacket, RubricKey]:
    """Stage-stratified, deterministic, blinded rubric packet + its key."""
    if loaded.dataset_version != record.provenance.dataset_version:
        raise ValueError("manifest version does not match the run's dataset_version")
    cases = {c.case_id: c for c in loaded.manifest.cases}
    run_id = record.provenance.run_id
    chosen: list[tuple[str, RubricItem, RubricKeyEntry]] = []
    for stage in record.provenance.stages:
        pool = [o for o in record.outcomes if o.stage is stage and o.draft is not None]
        pool.sort(key=lambda o: _item_id(run_id, o.case_id, o.stage, o.sample, seed))
        for o in pool[:per_stage]:
            assert o.draft is not None
            iid = _item_id(run_id, o.case_id, o.stage, o.sample, seed)
            item = RubricItem(
                item_id=iid,
                facts_block=render_facts(global_context(cases[o.case_id])),
                report_block=render_report(o.draft),
            )
            entry = RubricKeyEntry(item_id=iid, case_id=o.case_id, stage=o.stage, sample=o.sample)
            chosen.append((iid, item, entry))
    chosen.sort(key=lambda t: t[0])  # opaque order hides the stage
    packet = RubricPacket(
        run_id=run_id,
        instructions=(
            "Score each report 1 (poor) to 5 (excellent) on every dimension against its "
            "fact sheet. Reports are anonymised; do not try to infer how they were made."
        ),
        dimensions=RUBRIC_DIMENSIONS,
        items=tuple(item for _, item, _ in chosen),
    )
    return packet, RubricKey(run_id=run_id, entries=tuple(e for _, _, e in chosen))


def validate_scores(scores: RubricScores, key: RubricKey) -> None:
    """Every score must reference an item of THIS run's packet."""
    if scores.run_id != key.run_id:
        raise ValueError("scores belong to a different run")
    known = {e.item_id for e in key.entries}
    unknown = sorted({s.item_id for s in scores.scores} - known)
    if unknown:
        raise ValueError(f"scores reference unknown items: {unknown}")


def summarize(
    scores: Sequence[RubricScore],
    key: RubricKey,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, JsonValue]:
    """Per stage x dimension: mean, bootstrap CI over items, n."""
    stage_of = {e.item_id: e.stage for e in key.entries}
    per: dict[tuple[Stage, str], list[float]] = defaultdict(list)
    for s in scores:
        for dim in RUBRIC_DIMENSIONS:
            per[(stage_of[s.item_id], dim)].append(float(s.value(dim)))
    rng = np.random.default_rng(seed)
    out: dict[str, JsonValue] = {}
    for stage in sorted({e.stage for e in key.entries}):
        block: dict[str, JsonValue] = {}
        for dim in RUBRIC_DIMENSIONS:
            vals = per.get((stage, dim), [])
            bounds = bootstrap_ci(vals, resamples=resamples, confidence=confidence, rng=rng)
            block[dim] = {
                "mean": float(np.mean(vals)) if vals else None,
                "ci95": None if bounds is None else list(bounds),
                "n_items": len(vals),
            }
        out[stage.value] = block
    return out


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return None

    def ranks(x: Sequence[float]) -> npt.NDArray[np.float64]:  # average ranks for ties
        arr = np.asarray(x, dtype=np.float64)
        order = arr.argsort()
        r = np.empty(len(arr), dtype=np.float64)
        r[order] = np.arange(len(arr), dtype=np.float64)
        for v in np.unique(arr):
            r[arr == v] = r[arr == v].mean()
        return r

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def judge_agreement(
    reference: Sequence[RubricScore], judge: Sequence[RubricScore]
) -> dict[str, JsonValue]:
    """Item-level agreement of judge vs reference (clinician) scores."""
    ref = {s.item_id: s for s in reference}
    paired = [(ref[j.item_id], j) for j in judge if j.item_id in ref]
    out: dict[str, JsonValue] = {"n_items": len(paired)}
    for dim in RUBRIC_DIMENSIONS:
        a = [float(r.value(dim)) for r, _ in paired]
        b = [float(j.value(dim)) for _, j in paired]
        diffs = [abs(x - y) for x, y in zip(a, b, strict=True)]
        out[dim] = {
            "mae": float(np.mean(diffs)) if diffs else None,
            "exact_agreement": float(np.mean([d == 0 for d in diffs])) if diffs else None,
            "within_one": float(np.mean([d <= 1 for d in diffs])) if diffs else None,
            "spearman": _spearman(a, b),
        }
    return out


class LLMJudge:
    """Optional rubric judge (eval-only prompt). Immutable after construction."""

    def __init__(
        self, *, llm: LLMClient, model: str, temperature: float, prompt: Prompt | None = None
    ) -> None:
        self._llm = llm
        self._model = model
        self._temperature = temperature
        self._prompt = prompt or PromptLoader().load("judge", JUDGE_PROMPT_VERSION)

    @property
    def identity(self) -> str:
        return f"{self._model}@T={self._temperature}"

    async def score(self, item: RubricItem) -> RubricScore:
        system = self._prompt.render(
            {"facts_block": item.facts_block, "report_block": item.report_block}
        )
        resp = await generate_structured(
            self._llm,
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content="Score the report. Return ONLY the JSON."),
            ],
            JudgeOutput,
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=600,
            timeout_s=60.0,
            tags={"agent": "judge", "prompt_version": self._prompt.version},
        )
        out = resp.data
        return RubricScore(
            item_id=item.item_id,
            rater_id=self.identity[:64],
            rater_kind="llm_judge",
            accuracy=out.accuracy,
            completeness=out.completeness,
            safety=out.safety,
            usefulness=out.usefulness,
            comment=out.rationale,
        )


def canned_judge_llm() -> FakeLLMClient:
    """Offline judge for the dry run (fixed mid-scale verdict)."""
    payload: dict[str, JsonValue] = {
        "accuracy": 4,
        "completeness": 4,
        "safety": 5,
        "usefulness": 3,
        "rationale": "canned dry-run verdict",
    }
    return FakeLLMClient.valid(payload, agent="judge", prompt_version=JUDGE_PROMPT_VERSION)


def synthetic_rater_scores(packet: RubricPacket, *, seed: int) -> tuple[RubricScore, ...]:
    """Deterministic SYNTHETIC rater (dry run only — never clinical data)."""
    scale = ("3", "4", "4", "5")
    return tuple(
        RubricScore.model_validate(
            {
                "item_id": item.item_id,
                "rater_id": "synthetic-rater-1",
                "rater_kind": "synthetic",
                **{dim: int(pick(scale, seed, item.item_id, dim)) for dim in RUBRIC_DIMENSIONS},
            }
        )
        for item in packet.items
    )


async def dry_run(
    record: RunRecord,
    loaded: LoadedManifest,
    *,
    judge: LLMJudge,
    per_stage: int,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, JsonValue]:
    """End-to-end rubric pipeline on synthetic scores (packet → score →
    validate → un-blind → summarize → judge → agreement)."""
    packet, key = build_packet(record, loaded, per_stage=per_stage, seed=seed)
    raters = synthetic_rater_scores(packet, seed=seed)
    validate_scores(RubricScores(run_id=key.run_id, scores=raters), key)
    judged_list: list[RubricScore] = []
    judge_errors = 0
    for item in packet.items:
        try:
            judged_list.append(await judge.score(item))
        except AppError:
            judge_errors += 1  # recorded, never silently dropped (I-5)
    judged = tuple(judged_list)
    return {
        "label": DRY_RUN_LABEL,
        "items": len(packet.items),
        "per_stage_items": per_stage,
        "judge": judge.identity,
        "judge_prompt": f"judge/{JUDGE_PROMPT_VERSION}",
        "judge_errors": judge_errors,
        "rater_summary": summarize(
            raters, key, resamples=resamples, confidence=confidence, seed=seed
        ),
        "judge_summary": summarize(
            judged, key, resamples=resamples, confidence=confidence, seed=seed
        ),
        "judge_vs_rater": judge_agreement(raters, judged),
    }
