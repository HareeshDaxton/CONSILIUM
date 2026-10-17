"""Prometheus metrics (ARCHITECTURE.md §5.13; AGENTS.md §18).

Collectors live in the default registry served by ``GET /metrics``. Labels are
closed vocabularies only — node names, agent names, statuses, issue types,
reason codes — never case IDs, note text, filenames or model output (I-6).

Cost per case is NOT computed here (prices are eval config, not serving
config): dashboards derive it from ``consilium_llm_tokens_total`` x the list
price, labelled as an estimate.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from consilium.schemas.audit import MergedVerdict
from consilium.schemas.enums import CaseStatus

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60, 120)

NODE_LATENCY = Histogram(
    "consilium_node_latency_seconds", "Graph node latency", ["node"], buckets=_LATENCY_BUCKETS
)
CASES = Counter("consilium_cases_total", "Case status transitions", ["status"])
NEEDS_ATTENTION = Counter("consilium_needs_attention_total", "Cases entering NEEDS_ATTENTION")
LLM_CALLS = Counter("consilium_llm_calls_total", "LLM calls", ["agent", "status"])
LLM_TOKENS = Counter("consilium_llm_tokens_total", "LLM tokens", ["agent", "model", "kind"])
LLM_LATENCY = Histogram(
    "consilium_llm_latency_seconds", "LLM call latency", ["agent"], buckets=_LATENCY_BUCKETS
)
VISION_LATENCY = Histogram(
    "consilium_vision_latency_seconds", "Vision endpoint call latency", buckets=_LATENCY_BUCKETS
)
VISION_CALLS = Counter("consilium_vision_calls_total", "Vision endpoint calls", ["status"])
VISION_COLD_START = Counter(
    "consilium_vision_cold_start_total", "Retryable vision attempts (503/timeout: cold start)"
)
VISION_REVISION_MISMATCH = Counter(
    "consilium_vision_revision_mismatch_total", "Endpoint revision != pinned revision"
)
CRITIC_ISSUES = Counter("consilium_critic_issues_total", "Audit issues raised", ["type", "layer"])
ROUNDS_TO_APPROVAL = Histogram(
    "consilium_rounds_to_approval", "Audit rounds until approval", buckets=(1, 2, 3, 4, 5)
)
REVIEW_ACTIONS = Counter(
    "consilium_review_actions_total", "Clinician review actions", ["action", "reason_code"]
)

REVIEW_DECISIONS = Counter(
    "consilium_review_decisions_total", "Clinician review decisions (one per review)", ["action"]
)


def observe_node(
    node: str,
    seconds: float,
    prev_status: CaseStatus,
    status: CaseStatus,
    new_verdict: MergedVerdict | None,
) -> None:
    """Node latency + status/audit counters for one transition. ``new_verdict``
    is the verdict this node produced (None when the node made none)."""
    NODE_LATENCY.labels(node=node).observe(seconds)
    if status != prev_status:
        CASES.labels(status=status.value).inc()
        if status is CaseStatus.NEEDS_ATTENTION:
            NEEDS_ATTENTION.inc()
    if new_verdict is not None:
        for issue in new_verdict.issues:
            CRITIC_ISSUES.labels(type=issue.type.value, layer=issue.raised_by).inc()
        if new_verdict.approved:
            ROUNDS_TO_APPROVAL.observe(new_verdict.round)


def record_review(action: str, reason_codes: tuple[str, ...]) -> None:
    REVIEW_DECISIONS.labels(action=action).inc()
    for code in reason_codes or ("none",):
        REVIEW_ACTIONS.labels(action=action, reason_code=code).inc()
