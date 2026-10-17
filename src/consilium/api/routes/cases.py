"""Case endpoints (AGENTS.md §17, ARCHITECTURE.md §7).

Submit flow (``POST /v1/cases``): the image is stored content-addressed (the
upload FILENAME is never read, logged or stored — I-6), the case row is
created (Idempotency-Key honoured), then the intake prefix (image precheck +
input rail) runs HERE so the raw note is redacted and stored encrypted before
the case is handed to a worker — the raw note never crosses the queue and is
never persisted (I-6). The rest of the pipeline runs in ``jobs.worker``;
the API answers 202 + case ID.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, Response, UploadFile

from consilium.api.auth import (
    ApiRole,
    Identity,
    require_auth,
    require_clinician,
    require_submitter,
)
from consilium.api.deps import AppDeps
from consilium.api.errors import problem
from consilium.api.schemas_http import (
    CadSummary,
    CaseListItem,
    CaseListView,
    CaseView,
    DraftView,
    EvidenceChip,
    FactView,
    FailureView,
    SubmitResponse,
    TimelineEvent,
)
from consilium.core.errors import InputRejected, NotFound, StatusConflict
from consilium.graph.edges import next_node
from consilium.graph.state import CaseState
from consilium.jobs.queue import JobQueue
from consilium.persistence.object_store import sha256_hex
from consilium.persistence.repositories import PersistedCase
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import Issue
from consilium.schemas.enums import CaseStatus, Laterality
from consilium.schemas.vision import RawVisionOutput
from consilium.vision.overlay import OverlayLayer, render_overlay

router = APIRouter(tags=["cases"])

UNVERIFIED_BANNER = "UNVERIFIED — critic did not approve"
IMAGE_PREFIX = "images"
OVERLAY_PREFIX = "overlays"


def _deps(request: Request) -> AppDeps:
    deps: AppDeps = request.app.state.deps
    return deps


def _parse_id(case_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(case_id)
    except ValueError:
        raise NotFound("CASE_NOT_FOUND") from None


async def _load(deps: AppDeps, case_id: str) -> tuple[PersistedCase, CaseState]:
    persisted = await deps.repos.load_case(_parse_id(case_id))
    if persisted is None:
        raise NotFound("CASE_NOT_FOUND")
    return persisted, CaseState.from_snapshot(persisted.state_json)


def _status_url(case_id: str) -> str:
    return f"/v1/cases/{case_id}"


# ------------------------------------------------------------------ submit


@router.post("/cases", status_code=202, response_model=SubmitResponse)
async def submit_case(
    request: Request,
    identity: Annotated[Identity, Depends(require_submitter)],
    image: Annotated[UploadFile, File()],
    note: Annotated[str | None, Form()] = None,
    laterality: Annotated[Laterality, Form()] = Laterality.UNKNOWN,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)] = None,
) -> Any:
    deps = _deps(request)
    if idempotency_key is not None:
        existing = await deps.repos.find_case_by_idempotency_key(idempotency_key)
        if existing is not None:
            return SubmitResponse(
                case_id=str(existing.id),
                status=CaseStatus(existing.status),
                status_url=_status_url(str(existing.id)),
            )

    max_bytes = deps.graph.thresholds.image_precheck.max_bytes
    data = await image.read(max_bytes + 1)  # image.filename is deliberately never read
    if not data:
        raise InputRejected("IMAGE_EMPTY")
    if len(data) > max_bytes:
        raise InputRejected("IMAGE_TOO_LARGE")
    image_uri = await deps.objects.put(data, prefix=IMAGE_PREFIX)
    state = await deps.runner.submit(
        image_uri=image_uri,
        image_sha256=sha256_hex(data),
        created_by=identity.sub,
        raw_note=note if note else None,
        laterality=laterality,
        idempotency_key=idempotency_key,
    )
    state = await deps.runner.run_intake(state)
    if state.status == CaseStatus.REJECTED_INPUT:
        reason = state.failure.type if state.failure is not None else "INPUT_REJECTED"
        return problem("input-rejected", reason=reason, extra={"case_id": state.case_id})
    if next_node(state) is not None:
        await JobQueue(deps.repos).enqueue(state.case_id)
    return SubmitResponse(
        case_id=state.case_id, status=state.status, status_url=_status_url(state.case_id)
    )


# ------------------------------------------------------------------ reads


@router.get("/cases", response_model=CaseListView)
async def list_cases(
    request: Request,
    identity: Annotated[Identity, Depends(require_auth)],
    status: Annotated[list[CaseStatus] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> CaseListView:
    """Newest first, optional status filter (the UI's queue views). Clinicians
    see every case; a submitter-only caller sees the cases they submitted."""
    owner = None if identity.has(ApiRole.CLINICIAN) else identity.sub
    rows = await _deps(request).repos.cases_list(
        statuses=tuple(s.value for s in status or ()), created_by=owner, limit=limit
    )
    items = []
    for row in rows:
        state = CaseState.from_snapshot(row.state_json)
        items.append(
            CaseListItem(
                case_id=str(row.id),
                status=CaseStatus(row.status),
                laterality=state.laterality,
                created_at=row.created_at,
                tier=None if state.cad is None else state.cad.classifier.tier.value,
                unverified=state.unverified,
            )
        )
    return CaseListView(items=tuple(items))


@router.get("/cases/{case_id}", response_model=CaseView)
async def get_case(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_auth)],
) -> CaseView:
    persisted, state = await _load(_deps(request), case_id)
    cad = state.cad
    return CaseView(
        case_id=str(persisted.id),
        status=CaseStatus(persisted.status),
        laterality=state.laterality,
        created_at=persisted.created_at,
        failure=(
            None
            if state.failure is None
            else FailureView(type=state.failure.type, reason=state.failure.reason)
        ),
        cad_summary=(
            None
            if cad is None
            else CadSummary(
                tier=cad.classifier.tier.value,
                p_calibrated=cad.classifier.p_calibrated,
                cdr_vertical=cad.cdr.vertical,
                quality_flags=tuple(f.value for f in cad.quality.flags),
                mask_qc_ok=cad.mask_qc.ok,
            )
        ),
        unverified=state.unverified,
        in_progress=next_node(state) is not None,
    )


def _cited_ids(draft: DirectorDraft) -> set[str]:
    ids: set[str] = set()
    for claim in draft.claims:
        ids.update(claim.evidence_refs)
        if claim.numeric_ref is not None:
            ids.add(claim.numeric_ref)
    for rec in draft.recommendations:
        ids.update(rec.triggered_by)
    return ids


@router.get("/cases/{case_id}/draft", response_model=DraftView)
async def get_draft(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_clinician)],
) -> DraftView:
    deps = _deps(request)
    persisted, state = await _load(deps, case_id)
    row = await deps.repos.draft_latest(persisted.id)
    if row is None:
        raise StatusConflict("DRAFT_NOT_AVAILABLE")
    draft = DirectorDraft.model_validate(row.draft_json, strict=False)
    issues = tuple(Issue.model_validate(i, strict=False) for i in (row.open_issues_json or []))
    cited = _cited_ids(draft)
    facts = state.global_ctx.facts if state.global_ctx is not None else ()
    evidence = {
        f.id: EvidenceChip(label=f.label, value=f.value, unit=f.unit)
        for f in facts
        if f.id in cited
    }
    return DraftView(
        case_id=str(persisted.id),
        case_status=CaseStatus(persisted.status),
        revision=row.revision,
        draft=draft,
        open_issues=issues if row.unverified else (),
        unverified=row.unverified,
        unverified_banner=UNVERIFIED_BANNER if row.unverified else None,
        overlay_url=f"/v1/cases/{persisted.id}/overlay",
        evidence=evidence,
    )


@router.get("/cases/{case_id}/overlay")
async def get_overlay(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_clinician)],
    layer: Annotated[OverlayLayer, Query()] = OverlayLayer.COMPOSITE,
) -> Response:
    """Server-rendered PNG (disc cyan, cup magenta), cached in the object
    store under ``sha256(case_id + mask RLEs [+ layer])``. ``layer`` selects
    the composite (default) or one stackable layer for the UI's toggles.
    Served directly (no presigned-URL facility in V1)."""
    deps = _deps(request)
    persisted, state = await _load(deps, case_id)
    key, png = await overlay_png(deps, persisted, state, layer)
    return Response(
        png,
        media_type="image/png",
        headers={"ETag": f'"{key}"', "Cache-Control": "private, max-age=3600"},
    )


async def overlay_png(
    deps: AppDeps, persisted: PersistedCase, state: CaseState, layer: OverlayLayer
) -> tuple[str, bytes]:
    """(cache key, PNG) for one overlay layer; rendered once, then cached."""
    cad = await deps.repos.cad_load(persisted.id)
    if cad is None or "disc_mask_rle" not in cad.raw_vision_json or state.image_uri is None:
        raise StatusConflict("OVERLAY_NOT_AVAILABLE")
    raw = RawVisionOutput.model_validate(cad.raw_vision_json, strict=False)
    material = f"{persisted.id}|{raw.disc_mask_rle}|{raw.cup_mask_rle}"
    if layer != OverlayLayer.COMPOSITE:  # composite keeps its original cache key
        material += f"|{layer.value}"
    key = hashlib.sha256(material.encode()).hexdigest()
    uri = deps.objects.uri_for(key, prefix=OVERLAY_PREFIX)
    if await deps.objects.exists(uri):
        return key, await deps.objects.load(uri)
    image = await deps.objects.load(state.image_uri)
    png = await asyncio.to_thread(render_overlay, image, raw, layer=layer)
    await deps.objects.put(png, prefix=OVERLAY_PREFIX, key=key)
    return key, png


@router.get("/cases/{case_id}/timeline", response_model=tuple[TimelineEvent, ...])
async def get_timeline(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_auth)],
) -> tuple[TimelineEvent, ...]:
    """Audit events in append order: name, opaque actor, time (no detail)."""
    deps = _deps(request)
    persisted, _state = await _load(deps, case_id)
    events = await deps.repos.audit_events(persisted.id)
    return tuple(TimelineEvent(event=e.event, actor=e.actor, at=e.at) for e in events)


@router.get("/cases/{case_id}/facts", response_model=tuple[FactView, ...])
async def get_facts(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_clinician)],
) -> tuple[FactView, ...]:
    """The case's C_global fact sheet (CAD + de-identified note facts), read-only.
    409 until the context has been built."""
    _persisted, state = await _load(_deps(request), case_id)
    if state.global_ctx is None:
        raise StatusConflict("FACTS_NOT_AVAILABLE")
    return tuple(
        FactView(id=f.id, label=f.label, value=f.value, unit=f.unit, source=f.source)
        for f in state.global_ctx.facts
    )
