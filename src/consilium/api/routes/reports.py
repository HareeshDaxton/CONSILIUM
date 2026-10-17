"""Screening report endpoints (paper Fig. 2: draft → critic → report; AGENTS.md §17).

  GET /v1/cases/{id}/report-document   the report as structured JSON (UI renders it)
  GET /v1/cases/{id}/export?format=…   the SAME document as PDF or Word (.docx)
  GET /v1/meta/status                  is the vision server reachable? is an LLM configured?

The document's state — approved / draft / unverified / rejected — is decided
by ``reporting.document`` from the stored case, never by the client. Drafts
are exported clearly labelled as drafts; only APPROVED* reports carry
clinician attribution (I-4). Clinician-only, like the draft itself.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response

from consilium.api.auth import Identity, require_auth, require_clinician
from consilium.api.deps import AppDeps
from consilium.api.routes.cases import _load, overlay_png
from consilium.api.schemas_http import SystemStatusView, VisionStatusView
from consilium.core.errors import StatusConflict
from consilium.graph.edges import next_node
from consilium.reporting.document import Approval, ReportDocument, build_report_document
from consilium.reporting.render import render_docx, render_pdf
from consilium.review.service import load_report
from consilium.schemas.agents import DirectorDraft
from consilium.schemas.audit import Issue
from consilium.schemas.enums import CaseStatus
from consilium.vision.client import SupportsHealth
from consilium.vision.overlay import OverlayLayer

router = APIRouter(tags=["reports"])

_APPROVED = frozenset({CaseStatus.APPROVED, CaseStatus.APPROVED_WITH_EDITS})


class ExportFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"


_MEDIA = {
    ExportFormat.PDF: "application/pdf",
    ExportFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _deps(request: Request) -> AppDeps:
    deps: AppDeps = request.app.state.deps
    return deps


async def _document(deps: AppDeps, case_id: str) -> tuple[ReportDocument, bytes | None]:
    persisted, state = await _load(deps, case_id)
    if next_node(state) is not None:
        raise StatusConflict("REPORT_IN_PROGRESS")
    row = await deps.repos.draft_latest(persisted.id)
    if row is None:
        raise StatusConflict("REPORT_NOT_AVAILABLE")  # e.g. failed before drafting
    status = CaseStatus(persisted.status)
    approval = None
    if status in _APPROVED:
        rv = await load_report(str(persisted.id), deps.repos)
        approval = Approval(
            approved_by=rv.approved_by,
            approved_at=rv.approved_at,
            version=rv.version,
            reason_codes=rv.reason_codes,
            draft_revision=rv.draft_revision,
            report=rv.report,
        )
    doc = build_report_document(
        case_id=str(persisted.id),
        case_status=status,
        laterality=state.laterality,
        created_at=persisted.created_at,
        generated_at=deps.clock.now(),
        cad=state.cad,
        facts=state.global_ctx.facts if state.global_ctx is not None else (),
        draft=DirectorDraft.model_validate(row.draft_json, strict=False),
        draft_revision=row.revision,
        unverified=row.unverified,
        open_issues=tuple(
            Issue.model_validate(i, strict=False) for i in (row.open_issues_json or [])
        ),
        approval=approval,
        calibration_fitted=deps.graph.models.calibration.fitted_on != "not-fitted",
    )
    try:
        _, overlay = await overlay_png(deps, persisted, state, OverlayLayer.COMPOSITE)
    except StatusConflict:
        overlay = None
    return doc, overlay


@router.get("/cases/{case_id}/report-document", response_model=ReportDocument)
async def get_report_document(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_clinician)],
) -> ReportDocument:
    doc, _overlay = await _document(_deps(request), case_id)
    return doc


@router.get("/cases/{case_id}/export")
async def export_report(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_clinician)],
    format: Annotated[ExportFormat, Query()] = ExportFormat.PDF,
) -> Response:
    doc, overlay = await _document(_deps(request), case_id)
    render = render_pdf if format == ExportFormat.PDF else render_docx
    data = await asyncio.to_thread(render, doc, overlay)
    filename = f"consilium-report-{doc.case_id[:8]}-{doc.state}.{format.value}"
    return Response(
        data,
        media_type=_MEDIA[format],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/meta/status", response_model=SystemStatusView)
async def system_status(
    request: Request,
    _: Annotated[Identity, Depends(require_auth)],
) -> SystemStatusView:
    deps = _deps(request)
    client = deps.graph.vision
    pin = deps.graph.models.endpoint.pinned_revision
    if isinstance(client, SupportsHealth):
        health = await client.health()
        vision = VisionStatusView(
            mode=deps.settings.vision_client_mode,
            reachable=health.reachable,
            revision_ok=health.reachable and health.revision == pin,
            device=health.device,
            detail=health.detail,
        )
    else:  # pragma: no cover - every shipped client supports health
        vision = VisionStatusView(
            mode=deps.settings.vision_client_mode, reachable=False, revision_ok=False
        )
    key = deps.settings.openai_api_key.get_secret_value().strip()
    return SystemStatusView(vision=vision, llm_configured=bool(key))
