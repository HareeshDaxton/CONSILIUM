"""Clinician-only review endpoints (AGENTS.md §17). Status logic lives in
``review/service.py``; this router only maps HTTP ↔ service. Reviewer
identity is taken from the verified token, never from the body."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from consilium.api.auth import Identity, require_clinician
from consilium.api.deps import AppDeps
from consilium.review.reason_codes import REASON_CODE_DESCRIPTIONS
from consilium.review.service import (
    ReportView,
    ReviewCommand,
    ReviewPolicy,
    ReviewResult,
    load_report,
    review,
)

router = APIRouter(tags=["review"])


def _deps(request: Request) -> AppDeps:
    deps: AppDeps = request.app.state.deps
    return deps


@router.post("/cases/{case_id}/review", response_model=ReviewResult)
async def post_review(
    case_id: str,
    cmd: ReviewCommand,
    request: Request,
    clinician: Annotated[Identity, Depends(require_clinician)],
) -> ReviewResult:
    deps = _deps(request)
    policy = ReviewPolicy(
        repos=deps.repos, thresholds=deps.graph.thresholds, routing=deps.graph.routing
    )
    return await review(case_id, cmd, reviewer_id=clinician.sub, policy=policy)


@router.get("/cases/{case_id}/report", response_model=ReportView)
async def get_report(
    case_id: str,
    request: Request,
    _: Annotated[Identity, Depends(require_clinician)],
) -> ReportView:
    return await load_report(case_id, _deps(request).repos)


@router.get("/meta/reason-codes")
async def reason_codes(
    _: Annotated[Identity, Depends(require_clinician)],
) -> list[dict[str, str]]:
    return [{"code": c.value, "description": d} for c, d in REASON_CODE_DESCRIPTIONS.items()]
