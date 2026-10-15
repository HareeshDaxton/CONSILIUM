"""Graph build + persistence-per-node + crash-resume (§4, ARCHITECTURE.md §5.3).

LangGraph is thin wiring only: every LangGraph node is the same wrapper —
``nodes.invoke`` (the fail-closed node boundary) followed by ONE database
transaction that writes ``(status, state_json, last_node, failure, audit
event)`` plus the node's domain rows and drained agent-run telemetry. The
routing function from ``START`` and after every node is ``edges.next_node``,
so a fresh case and a resumed case walk the SAME graph: resume simply enters
at the successor of the last persisted node. The LangGraph checkpointer is
deliberately unused — Postgres is the only source of truth.

Idempotency (§4): state snapshots are written in the same transaction as the
artifact rows, so a persisted ``state.cad`` ⇔ a ``cad_results`` row and a
persisted ``state.extraction`` ⇔ a completed ``clinical_notes`` row. The nodes
short-circuit on those fields; resume never recomputes vision or extraction.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Hashable
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from consilium.graph import edges
from consilium.graph.deps import Deps
from consilium.graph.nodes import NODES, NodeSpec, invoke
from consilium.graph.state import CaseState, FailureRecord
from consilium.persistence.repositories import Repositories, RepoTx
from consilium.schemas.enums import CaseStatus, Laterality

__all__ = ["GraphRunner", "NodeHook"]

# Called after a node's transaction COMMITS — tests use it to simulate a crash.
NodeHook = Callable[[str, CaseState], None]

_RECURSION_LIMIT = 64  # ≥ linear chain + 2·CRITIC_MAX_ROUNDS + rails, with margin


class _GState(TypedDict):
    state: CaseState


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class GraphRunner:
    def __init__(
        self, deps: Deps, repos: Repositories, *, on_node_persisted: NodeHook | None = None
    ) -> None:
        self._deps = deps
        self._repos = repos
        self._hook = on_node_persisted
        self._graph = self._compile()

    # ------------------------------------------------------------ build

    def _compile(self) -> Any:
        g: StateGraph[_GState] = StateGraph(_GState)
        targets: dict[Hashable, str] = {name: name for name in NODES}
        targets["__end__"] = END
        for name, spec in NODES.items():
            g.add_node(name, cast(Any, self._wrap(spec)))
            g.add_conditional_edges(name, self._route, targets)
        g.add_conditional_edges(START, self._route, targets)
        return g.compile()  # no checkpointer: Postgres is the source of truth

    @staticmethod
    def _route(gs: _GState) -> str:
        nxt = edges.next_node(gs["state"])
        return "__end__" if nxt is None else nxt

    def _wrap(self, spec: NodeSpec) -> Callable[[_GState], Any]:
        async def node(gs: _GState) -> _GState:
            prev = gs["state"]
            out = await invoke(spec, prev, self._deps)
            out = await self._persist(spec.name, prev, out)
            if self._hook is not None:
                self._hook(spec.name, out)
            return {"state": out}

        return node

    # ------------------------------------------------------------ public API

    async def submit(
        self,
        *,
        image_uri: str,
        image_sha256: str,
        created_by: str,
        raw_note: str | None = None,
        laterality: Laterality = Laterality.UNKNOWN,
        idempotency_key: str | None = None,
    ) -> CaseState:
        """Create the case row (status RECEIVED) and return the initial state.

        Idempotent: a known ``idempotency_key`` returns the existing case's
        persisted state instead of creating a second case (§4).
        """
        if idempotency_key is not None:
            existing = await self._repos.find_case_by_idempotency_key(idempotency_key)
            if existing is not None:
                return CaseState.from_snapshot(existing.state_json)
        state = CaseState(
            case_id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
            created_by=created_by,
            laterality=laterality,
            image_sha256=image_sha256,
            image_uri=image_uri,
            raw_note_input=raw_note,
            note_submitted=raw_note is not None,
        )
        async with self._repos.transaction() as tx:
            await tx.create_case(
                case_id=uuid.UUID(state.case_id),
                status=state.status.value,
                laterality=state.laterality.value,
                created_by=created_by,
                idempotency_key=idempotency_key,
                image_sha256=image_sha256,
                image_uri=image_uri,
                state_json=state.to_snapshot(),
                last_node=None,
            )
        return state

    async def run(self, state: CaseState) -> CaseState:
        """Walk the graph from ``state`` until a terminal status."""
        result = await self._graph.ainvoke(
            {"state": state}, config={"recursion_limit": _RECURSION_LIMIT}
        )
        final: CaseState = result["state"]
        return final

    async def resume(self, case_id: str) -> CaseState:
        """Crash-resume: rehydrate from Postgres and continue at the successor
        of the last persisted node. Terminal cases are returned unchanged."""
        persisted = await self._repos.load_case(uuid.UUID(case_id))
        if persisted is None:
            raise LookupError(f"unknown case {case_id}")
        state = CaseState.from_snapshot(persisted.state_json)
        if edges.next_node(state) is None:
            return state
        redacted = await self._repos.load_redacted_note(uuid.UUID(case_id))
        if redacted is not None:
            state = state.model_copy(update={"raw_note_redacted": redacted})
        elif state.note_submitted and state.status in (
            CaseStatus.RECEIVED,
            CaseStatus.IMAGE_PRECHECKED,
        ):
            # The raw note is never persisted (I-6); it was lost before the
            # input rail could redact + store it. Fail closed: proceeding
            # would silently produce a report without the submitted note.
            failed = state.model_copy(
                update={
                    "status": CaseStatus.FAILED,
                    "failure": FailureRecord(
                        type="NOTE_NOT_RECOVERABLE",
                        node="resume",
                        reason="note lost before input rail; resubmit the case",
                    ),
                }
            )
            async with self._repos.transaction() as tx:
                await self._write_transition(tx, "resume", failed)
            return failed
        return await self.run(state)

    # ------------------------------------------------------------ persistence

    async def _persist(self, name: str, prev: CaseState, out: CaseState) -> CaseState:
        case_id = uuid.UUID(out.case_id)
        async with self._repos.transaction() as tx:
            await self._write_domain_rows(tx, case_id, name, prev, out)
            for run in out.agent_runs_pending:
                await tx.agent_run_add(
                    case_id=case_id,
                    agent=run.agent,
                    prompt_version=run.prompt_version,
                    model_id=run.model_id,
                    input_tokens=run.input_tokens,
                    output_tokens=run.output_tokens,
                    latency_ms=run.latency_ms,
                    attempt=run.attempt,
                    status=run.status,
                    output_json=run.output_json,
                    error=run.error,
                )
            await self._write_transition(tx, name, out)
        return out.model_copy(update={"agent_runs_pending": ()})

    @staticmethod
    async def _write_transition(tx: RepoTx, name: str, out: CaseState) -> None:
        detail: dict[str, Any] = {"node": name, "status": out.status.value}
        if out.failure is not None:
            detail["failure_type"] = out.failure.type  # code only, never content
        await tx.transition(
            case_id=uuid.UUID(out.case_id),
            status=out.status.value,
            state_json=out.to_snapshot(),
            last_node=out.last_node or name,
            failure_json=None if out.failure is None else out.failure.model_dump(mode="json"),
            event=f"node.{name}",
            detail=detail,
        )

    async def _write_domain_rows(
        self, tx: RepoTx, case_id: uuid.UUID, name: str, prev: CaseState, out: CaseState
    ) -> None:
        if (
            name == "input_rail"
            and out.raw_note_redacted is not None
            and (prev.raw_note_redacted is None)
        ):
            await tx.notes_save_redacted(case_id=case_id, redacted_text=out.raw_note_redacted)
        elif name == "extract_note" and out.extraction is not None and prev.extraction is None:
            await tx.notes_complete(
                case_id=case_id,
                note_json=out.extraction.note.model_dump(mode="json"),
                extraction_json=out.extraction.model_dump(mode="json"),
            )
        elif name == "call_vision":
            await self._write_vision(tx, case_id, prev, out)
        elif name == "route_context" and out.status == CaseStatus.ROUTED:
            for role, ctx in out.role_contexts.items():
                facts = ctx.model_dump(mode="json")
                await tx.routed_save(
                    case_id=case_id,
                    role=role.value,
                    routing_version=ctx.routing_version,
                    facts_json=facts,
                    prompt_hash=_sha256_json(facts),
                )
        elif name in ("draft", "revise") and out.draft is not None and out.failure is None:
            await tx.draft_save(
                case_id=case_id,
                revision=out.draft_revision,
                draft_json=out.draft.model_dump(mode="json"),
                producer_run_id=None,
                unverified=False,
                open_issues_json=None,
            )
        elif name == "audit" and out.verdict is not None and out.failure is None:
            verdict = out.verdict
            issues = [i.model_dump(mode="json") for i in verdict.issues]
            await tx.verdict_add(
                case_id=case_id,
                revision=out.draft_revision,
                round_=verdict.round,
                approved=verdict.approved,
                issues_json=issues,
                layer="deterministic+llm_critic" if verdict.critic_ran else "deterministic",
            )
            if out.unverified:
                await tx.draft_mark_unverified(
                    case_id=case_id, revision=out.draft_revision, open_issues_json=issues
                )

    async def _write_vision(
        self, tx: RepoTx, case_id: uuid.UUID, prev: CaseState, out: CaseState
    ) -> None:
        pinned = self._deps.models.endpoint.pinned_revision
        if out.cad is not None and prev.cad is None:
            # Only reachable when the pipeline verified response revision == pin.
            await tx.cad_save(
                case_id=case_id,
                cad_json=out.cad.model_dump(mode="json"),
                raw_vision_json={"p_raw": out.cad.classifier.p_raw, "endpoint_revision": pinned},
                classifier_version=out.cad.classifier.model_version,
                segmenter_version=out.cad.segmenter_version,
                endpoint_revision=pinned,
            )
            await tx.vision_call_add(
                case_id=case_id,
                endpoint_revision=pinned,
                latency_ms=None,
                attempts=None,
                status="ok",
                error=None,
            )
        elif out.failure is not None:
            await tx.vision_call_add(
                case_id=case_id,
                endpoint_revision=None,
                latency_ms=None,
                attempts=None,
                status="failed",
                error=out.failure.type,
            )
