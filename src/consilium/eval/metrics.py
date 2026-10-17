"""Per-outcome evaluation metrics (plan Section 13.2; AGENTS.md §19.2).

Each stored ``StageOutcome`` is scored into one ``MetricRow``. The instrument
for claim-level metrics is the production deterministic verifier
(``verify_draft``) applied to the FINAL draft of every stage — the same exact
checks for S1-S4, so S1-S3 (which never run verification) are measured by the
identical ruler S4 is audited with:

- ``unsupported_claim_rate`` — claims carrying an UNSUPPORTED, BAD_REFERENCE
  or NUMERIC_MISMATCH issue / all claims. Draft-level unsupported numerals
  (impression/recommendations, no claim ID) count in
  ``draft_level_unsupported``.
- ``omission_rate`` — OMITTED_FINDING issues / mandatory items (laterality,
  tier, vertical CDR + every caveat fact in the case).
- ``unsafe_language`` — 1.0 when the draft has any UNSAFE_LANGUAGE issue.
- ``out_of_scope_rate`` — claims flagged OUT_OF_SCOPE against the REAL
  routing policy / all claims. Reported separately and never folded into
  "unsupported": S1/S2 agents legitimately saw everything, so this measures
  attribution drift away from the routed design, not hallucination.
- ``minority_preservation`` — curated disagreement cases only, multi-agent
  stages only: 1.0 iff the draft carries a ``Disagreement`` (≥2 roles) for
  every curated topic (keyword match identical to the verifier's).
- ``s_inter`` / ``unique_fact_coverage`` — see ``eval/diversity.py``.
- operational: ``rounds_to_approval`` (S4, approved only), ``unverified``
  (S4: 1.0 when the loop exhausted), ``tokens``, ``cost_usd`` (null when the
  model has no configured price — never guessed), ``wall_ms``,
  ``llm_latency_ms``.

A metric that does not apply is ``None`` and is excluded from aggregates
(the aggregate reports its ``n``). Failed outcomes contribute only to the
failure rate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from consilium.context.facts import F
from consilium.context.router import RoutingConfig
from consilium.eval.datasets import EvalCase
from consilium.eval.diversity import Embedder, mean_defined, s_inter, unique_fact_coverage
from consilium.eval.runners import Stage, StageOutcome
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.context import GlobalContext
from consilium.schemas.enums import IssueType
from consilium.verify.deterministic import verify_draft
from consilium.vision.config import Thresholds

UNSUPPORTED_TYPES = frozenset(
    {IssueType.UNSUPPORTED, IssueType.BAD_REFERENCE, IssueType.NUMERIC_MISMATCH}
)

# Metric names, in report order.
CLAIM_METRICS = (
    "unsupported_claim_rate",
    "draft_level_unsupported",
    "omission_rate",
    "unsafe_language",
    "out_of_scope_rate",
    "minority_preservation",
)
DIVERSITY_METRICS = ("s_inter", "unique_fact_coverage")
OPERATIONAL_METRICS = (
    "rounds_to_approval",
    "unverified",
    "tokens",
    "cost_usd",
    "wall_ms",
    "llm_latency_ms",
)
ALL_METRICS = CLAIM_METRICS + DIVERSITY_METRICS + OPERATIONAL_METRICS

Pricing = Mapping[str, Mapping[str, float]]  # model -> {"input", "output"} USD / 1M


class MetricRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    stage: Stage
    sample: int
    failed: bool
    values: dict[str, float | None]


def _mandatory_count(ctx: GlobalContext) -> int:
    caveats = sum(
        1 for f in ctx.facts if f.id.startswith("cad.caveat.") or f.id == F.CAVEAT_EXTRACTION
    )
    return 3 + caveats  # laterality, tier, vertical CDR + caveats


def minority_preserved(
    draft: DirectorDraft, topics: tuple[str, ...], thresholds: Thresholds
) -> bool:
    """Every curated topic has a visible ≥2-role Disagreement in the draft."""
    keywords = {
        t.topic: tuple(k.lower() for k in t.keywords) for t in thresholds.disagreement_topics
    }
    for topic in topics:
        kws = keywords.get(topic, (topic.lower(),))
        hit = any(
            len({role for role, _ in d.positions}) >= 2
            and (
                any(k in d.topic.lower() for k in kws)
                or any(k in pos.lower() for _, pos in d.positions for k in kws)
            )
            for d in draft.disagreements
        )
        if not hit:
            return False
    return True


def cost_usd(outcome: StageOutcome, pricing: Pricing) -> float | None:
    total = 0.0
    for call in outcome.llm_calls:
        price = pricing.get(call.model)
        if price is None:
            return None
        total += (call.input_tokens * price["input"] + call.output_tokens * price["output"]) / 1e6
    return total


def score_outcome(
    outcome: StageOutcome,
    case: EvalCase,
    ctx: GlobalContext,
    *,
    thresholds: Thresholds,
    routing: RoutingConfig,
    embedder: Embedder,
    pricing: Pricing,
) -> MetricRow:
    values: dict[str, float | None] = dict.fromkeys(ALL_METRICS)
    values["tokens"] = float(sum(c.input_tokens + c.output_tokens for c in outcome.llm_calls))
    values["cost_usd"] = cost_usd(outcome, pricing)
    values["wall_ms"] = outcome.wall_ms
    values["llm_latency_ms"] = float(sum(c.latency_ms for c in outcome.llm_calls))
    if outcome.status == "failed" or outcome.draft is None:
        return MetricRow(
            case_id=outcome.case_id,
            stage=outcome.stage,
            sample=outcome.sample,
            failed=True,
            values=values,
        )

    draft = outcome.draft
    issues = verify_draft(draft, ctx, outcome.sub_reports, thresholds, routing)
    n_claims = len(draft.claims)
    unsupported_ids = {i.claim_id for i in issues if i.type in UNSUPPORTED_TYPES and i.claim_id}
    scope_ids = {i.claim_id for i in issues if i.type is IssueType.OUT_OF_SCOPE and i.claim_id}
    values["unsupported_claim_rate"] = len(unsupported_ids) / n_claims if n_claims else None
    values["draft_level_unsupported"] = float(
        sum(1 for i in issues if i.type in UNSUPPORTED_TYPES and i.claim_id is None)
    )
    values["omission_rate"] = sum(
        1 for i in issues if i.type is IssueType.OMITTED_FINDING
    ) / _mandatory_count(ctx)
    values["unsafe_language"] = float(any(i.type is IssueType.UNSAFE_LANGUAGE for i in issues))
    values["out_of_scope_rate"] = len(scope_ids) / n_claims if n_claims else None
    if case.curated_disagreement_topics and outcome.stage is not Stage.S1:
        values["minority_preservation"] = float(
            minority_preserved(draft, case.curated_disagreement_topics, thresholds)
        )
    if outcome.sub_reports:
        values["s_inter"] = s_inter(outcome.sub_reports, embedder)
        values["unique_fact_coverage"] = mean_defined(
            list(unique_fact_coverage(outcome.sub_reports, ctx, routing).values())
        )
    if outcome.stage is Stage.S4:
        values["unverified"] = 0.0 if outcome.audit_approved else 1.0
        if outcome.audit_approved and outcome.audit_rounds is not None:
            values["rounds_to_approval"] = float(outcome.audit_rounds)
    return MetricRow(
        case_id=outcome.case_id,
        stage=outcome.stage,
        sample=outcome.sample,
        failed=False,
        values=values,
    )


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100]); None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float], *, resamples: int, confidence: float, rng: np.random.Generator
) -> list[float] | None:
    """Percentile bootstrap CI of the mean over ``values`` (one per case)."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return [float(arr[0]), float(arr[0])]
    means = arr[rng.integers(0, len(arr), size=(resamples, len(arr)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))]
