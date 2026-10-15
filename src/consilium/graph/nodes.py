"""Graph nodes — thin wrappers over the logic modules (ARCHITECTURE.md §5.3).

Every node is a plain async function ``(CaseState, Deps) -> CaseState``; no
LangGraph import here. Persistence-per-node happens in the runner
(Section 9.3, ``graph/build.py``), which wraps ``invoke`` and writes
``(status, state_json, audit event, drained telemetry)`` in ONE transaction.

Failure discipline (§4/§21, I-5): nodes raise the ``core.errors`` hierarchy;
``invoke`` is the single node-boundary catch that converts errors into a
terminal status + ``FailureRecord`` (fail closed, never a degraded report).
``AppError.case_status`` carries the mapping (InputRejected → REJECTED_INPUT,
PolicyViolation → REJECTED_INPUT or NEEDS_ATTENTION per its own flag, …);
unknown exceptions become FAILED with the exception TYPE NAME only (the
message may carry content; I-6).

Idempotency (§4): nodes that produce persisted artifacts check for the
artifact first and short-circuit — crash-resume never recomputes vision or
re-extracts a note.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from consilium.agents import specialists as specialist_mod
from consilium.context import builder as context_builder
from consilium.context import router as context_router
from consilium.core.errors import AppError
from consilium.graph.deps import Deps
from consilium.graph.state import AgentRunRecord, CaseState, FailureRecord
from consilium.guardrails import input_checks, output_checks
from consilium.intake import deidentify
from consilium.intake import extractor as extractor_mod
from consilium.intake.extractor import PROMPT_VERSION as EXTRACTOR_PROMPT_VERSION
from consilium.schemas.audit import MergedVerdict
from consilium.schemas.enums import CaseStatus
from consilium.schemas.note import NoteExtraction
from consilium.verify import deterministic
from consilium.verify import merge as merge_mod
from consilium.vision import pipeline as vision_pipeline
from consilium.vision import quality
from consilium.vision.client import RetryPolicy

__all__ = ["NODES", "NodeSpec", "invoke"]

NodeFn = Callable[[CaseState, Deps], Awaitable[CaseState]]


def _telemetry(state: CaseState, record: AgentRunRecord) -> tuple[AgentRunRecord, ...]:
    return (*state.agent_runs_pending, record)


# ---------------------------------------------------------------- precheck


async def precheck_image(state: CaseState, deps: Deps) -> CaseState:
    """Client-side image gate: decode, format, size, resolution (§8.4).
    Non-gradable → InputRejected → REJECTED_INPUT with an actionable reason."""
    if state.image_uri is None:
        from consilium.core.errors import InputRejected

        raise InputRejected("IMAGE_MISSING", detail="case has no image_uri")
    image = await deps.image_store.load(state.image_uri)
    quality.precheck_image_bytes(image, deps.thresholds.image_precheck)
    return state.model_copy(update={"status": CaseStatus.IMAGE_PRECHECKED})


async def input_rail(state: CaseState, deps: Deps) -> CaseState:
    """Deterministic input checks (guardrails/input_checks, §14) +
    de-identification on the raw note (§10). Runs BEFORE any LLM sees any
    note text (I-6/I-7) and therefore before extract_note. A trigger raises
    PolicyViolation → NEEDS_ATTENTION (§4); the runner audits it as a
    ``guardrail.triggered`` event. Note is optional."""
    raw = state.raw_note_input if state.raw_note_input is not None else state.raw_note_redacted
    if raw is None:
        return state.model_copy(update={"status": CaseStatus.INPUT_RAIL_PASSED})
    if state.raw_note_input is not None:
        input_checks.check_note(raw, deps.input_rail, deps.phi_policy)  # raises PolicyViolation
    if state.raw_note_redacted is None:
        redacted = deidentify.redact(raw, deps.phi_policy)
        # model-based (NeMo) injection rail sees the REDACTED note only (I-6/I-7)
        await deps.rails.check_input(redacted.text)
        return state.model_copy(
            update={
                "status": CaseStatus.INPUT_RAIL_PASSED,
                "raw_note_input": None,
                "raw_note_redacted": redacted.text,
                "note_redactions": redacted.pattern_names,
            }
        )
    return state.model_copy(update={"status": CaseStatus.INPUT_RAIL_PASSED, "raw_note_input": None})


async def extract_note(state: CaseState, deps: Deps) -> CaseState:
    """Structured extraction from the REDACTED note (I-6); ≤2 repair retries
    inside extractor.extract_note, then ValidationFailure → FAILED."""
    if state.extraction is not None or state.raw_note_redacted is None:
        # idempotent re-entry, or no note was submitted at all
        return state.model_copy(update={"status": CaseStatus.NOTE_EXTRACTED})
    resp = await extractor_mod.extract_note(
        state.raw_note_redacted,
        llm=deps.llm,
        prompt=deps.extractor_prompt,
        timeout_s=deps.settings.llm_timeout_s,
        budget=deps.budget,
    )
    extraction = NoteExtraction(
        note=resp.data,
        extractor_model=resp.model,
        extractor_prompt_version=EXTRACTOR_PROMPT_VERSION,
        redactions=state.note_redactions,
        extraction_confidence=extractor_mod.estimate_confidence(resp.data),
    )
    return state.model_copy(
        update={
            "status": CaseStatus.NOTE_EXTRACTED,
            "note": resp.data,
            "extraction": extraction,
            "agent_runs_pending": _telemetry(
                state,
                AgentRunRecord(
                    agent="note_extractor",
                    prompt_version=EXTRACTOR_PROMPT_VERSION,
                    model_id=resp.model,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    latency_ms=int(resp.latency_ms),
                    attempt=resp.attempts,
                    output_json=extraction.model_dump(mode="json"),
                ),
            ),
        }
    )


# ---------------------------------------------------------------- vision


async def call_vision(state: CaseState, deps: Deps) -> CaseState:
    """Remote inference + in-repo interpretation (I-13). Idempotent: an
    existing CADResult is never recomputed (§4)."""
    if state.cad is not None:
        return state.model_copy(update={"status": CaseStatus.CAD_COMPLETE})
    if state.image_uri is None:
        from consilium.core.errors import VisionFailure

        raise VisionFailure("IMAGE_MISSING", detail="case has no image_uri")
    image = await deps.image_store.load(state.image_uri)
    cad = await vision_pipeline.run(
        image,
        client=deps.vision,
        thresholds=deps.thresholds,
        models=deps.models,
        laterality=state.laterality,
        # §4: total cold-start wait bounded by VISION_TIMEOUT_S, then fail closed
        retry=RetryPolicy(total_budget_s=deps.settings.vision_timeout_s),
    )
    return state.model_copy(update={"status": CaseStatus.CAD_COMPLETE, "cad": cad})


# ---------------------------------------------------------------- context


async def build_context(state: CaseState, deps: Deps) -> CaseState:
    if state.cad is None:
        raise ValueError("build_context requires cad (graph ordering bug)")
    ctx = context_builder.build(state.cad, state.note, state.extraction, state.case_id)
    return state.model_copy(update={"status": CaseStatus.CONTEXT_BUILT, "global_ctx": ctx})


async def route_context(state: CaseState, deps: Deps) -> CaseState:
    if state.global_ctx is None:
        raise ValueError("route_context requires global_ctx (graph ordering bug)")
    role_contexts = context_router.route(state.global_ctx, deps.routing)
    return state.model_copy(update={"status": CaseStatus.ROUTED, "role_contexts": role_contexts})


# ---------------------------------------------------------------- agents


async def run_specialists(state: CaseState, deps: Deps) -> CaseState:
    """Fan-out to the three isolated specialists (§11). Fail closed: any
    specialist failure after its retries fails the whole case (I-5) —
    run_all raises SpecialistFanOutFailure carrying sibling outputs."""
    if len(state.sub_reports) == len(deps.specialists):
        return state.model_copy(update={"status": CaseStatus.SPECIALISTS_DONE})
    results = await specialist_mod.run_all(state.role_contexts, deps.specialists)
    pending = state.agent_runs_pending
    reports = {}
    for role, resp in results.items():
        reports[role] = resp.data
        pending = (
            *pending,
            AgentRunRecord(
                agent=role.value,
                prompt_version=resp.prompt_version,
                model_id=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_ms=int(resp.latency_ms),
                attempt=resp.attempts,
                output_json=resp.data.model_dump(mode="json"),
            ),
        )
    return state.model_copy(
        update={
            "status": CaseStatus.SPECIALISTS_DONE,
            "sub_reports": reports,
            "agent_runs_pending": pending,
        }
    )


async def draft(state: CaseState, deps: Deps) -> CaseState:
    if state.global_ctx is None:
        raise ValueError("draft requires global_ctx (graph ordering bug)")
    resp = await deps.director.draft(state.sub_reports, state.global_ctx)
    return state.model_copy(
        update={
            "status": CaseStatus.DRAFTED,
            "draft": resp.data,
            "draft_revision": resp.data.revision,
            "agent_runs_pending": _telemetry(
                state,
                AgentRunRecord(
                    agent="director",
                    prompt_version=resp.prompt_version,
                    model_id=resp.model,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    latency_ms=int(resp.latency_ms),
                    attempt=resp.attempts,
                    output_json=resp.data.model_dump(mode="json"),
                ),
            ),
        }
    )


# ---------------------------------------------------------------- audit


async def audit(state: CaseState, deps: Deps) -> CaseState:
    """One audit round: deterministic verifier FIRST, LLM critic only on a
    clean deterministic pass, then merge (§13, fixed order). Loop control:
    rejection below the round cap routes to ``revise`` (edges.py); rejection
    AT the cap converts to NEEDS_ATTENTION with the UNVERIFIED contract
    (open issues attached, never auto-approved, never dropped — §4/I-4)."""
    if state.draft is None or state.global_ctx is None:
        raise ValueError("audit requires draft + global_ctx (graph ordering bug)")
    round_ = state.audit_round + 1
    det_issues = deterministic.verify_draft(
        state.draft,
        state.global_ctx,
        tuple(state.sub_reports.values()),
        deps.thresholds,
        deps.routing,
    )
    pending = state.agent_runs_pending
    if det_issues:
        verdict: MergedVerdict = merge_mod.merge_verdicts(det_issues, None, round_)
    else:
        resp = await deps.critic.run(state.draft, state.global_ctx, round_)
        pending = (
            *pending,
            AgentRunRecord(
                agent="critic",
                prompt_version=resp.prompt_version,
                model_id=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_ms=int(resp.latency_ms),
                attempt=resp.attempts,
                output_json=resp.data.model_dump(mode="json"),
            ),
        )
        verdict = merge_mod.merge_verdicts((), resp.data, round_)

    update: dict[str, object] = {
        "verdict": verdict,
        "audit_round": round_,
        "open_issues": verdict.issues,
        "agent_runs_pending": pending,
    }
    if verdict.approved:
        update["status"] = CaseStatus.AUDITING  # approved: edge → output_rail
    elif round_ >= deps.settings.critic_max_rounds:
        update["status"] = CaseStatus.NEEDS_ATTENTION
        update["unverified"] = True
    else:
        update["status"] = CaseStatus.AUDITING  # rejected: edge → revise
    return state.model_copy(update=update)


async def revise(state: CaseState, deps: Deps) -> CaseState:
    """Director revision round: previous draft + issue list ONLY (§12)."""
    if state.draft is None or state.global_ctx is None:
        raise ValueError("revise requires draft + global_ctx (graph ordering bug)")
    resp = await deps.director.revise(
        state.draft, state.open_issues, state.audit_round, state.sub_reports, state.global_ctx
    )
    return state.model_copy(
        update={
            "status": CaseStatus.REVISING,
            "draft": resp.data,
            "draft_revision": resp.data.revision,
            "agent_runs_pending": _telemetry(
                state,
                AgentRunRecord(
                    agent="director",
                    prompt_version=resp.prompt_version,
                    model_id=resp.model,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    latency_ms=int(resp.latency_ms),
                    attempt=resp.attempts,
                    output_json=resp.data.model_dump(mode="json"),
                ),
            ),
        }
    )


# ---------------------------------------------------------------- rails


async def output_rail(state: CaseState, deps: Deps) -> CaseState:
    """Output rail on the draft before clinician view (§14): deterministic
    output_checks FIRST (lexicon, PHI echo, disclaimer), then the model-based
    NeMo rail. Any trigger → PolicyViolation → NEEDS_ATTENTION + a
    ``guardrail.triggered`` audit event. Never alters the draft."""
    if state.draft is None:
        raise ValueError("output_rail requires draft (graph ordering bug)")
    output_checks.check_draft(state.draft, state.global_ctx, deps.output_policy, deps.phi_policy)
    await deps.rails.check_output("\n".join(output_checks.draft_texts(state.draft)))
    return state.model_copy(update={"status": CaseStatus.OUTPUT_RAIL_PASSED})


async def persist_pending(state: CaseState, deps: Deps) -> CaseState:
    """Terminal pipeline node: the draft is ready for mandatory clinician
    review (I-4). APPROVED is only ever set by the review service (P11)."""
    return state.model_copy(update={"status": CaseStatus.PENDING_REVIEW})


# ---------------------------------------------------------------- registry


@dataclass(frozen=True, slots=True)
class NodeSpec:
    name: str
    fn: NodeFn


_NODE_FNS: tuple[tuple[str, NodeFn], ...] = (
    ("precheck_image", precheck_image),
    ("input_rail", input_rail),
    ("extract_note", extract_note),
    ("call_vision", call_vision),
    ("build_context", build_context),
    ("route_context", route_context),
    ("run_specialists", run_specialists),
    ("draft", draft),
    ("audit", audit),
    ("revise", revise),
    ("output_rail", output_rail),
    ("persist_pending", persist_pending),
)

NODES: dict[str, NodeSpec] = {name: NodeSpec(name, fn) for name, fn in _NODE_FNS}


async def invoke(spec: NodeSpec, state: CaseState, deps: Deps) -> CaseState:
    """THE node boundary (§21): run one node, convert any error to a
    fail-closed terminal status + FailureRecord, stamp last_node."""
    try:
        out = await spec.fn(state, deps)
    except AppError as exc:
        status = CaseStatus(exc.case_status)
        return state.model_copy(
            update={
                "status": status,
                "failure": FailureRecord(
                    type=exc.reason, node=spec.name, reason=exc.detail or exc.reason
                ),
                "last_node": spec.name,
            }
        )
    except Exception as exc:  # the sanctioned node-boundary catch (§21)
        return state.model_copy(
            update={
                "status": CaseStatus.FAILED,
                "failure": FailureRecord(
                    type="INTERNAL", node=spec.name, reason=type(exc).__name__
                ),
                "last_node": spec.name,
            }
        )
    return out.model_copy(update={"last_node": spec.name})
