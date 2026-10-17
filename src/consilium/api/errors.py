"""RFC 7807 ``application/problem+json`` handlers (ARCHITECTURE.md §7).

Rules (I-6/I-7, phi-review skill):
  - Bodies carry a problem ``type``/``title``/``status`` and, for AppErrors, the
    short safe ``reason`` code only. ``AppError.detail`` is NEVER rendered.
  - Request-validation errors list field locations + error types only — never
    the offending input value or pydantic's message (which can quote it).
  - Unknown exceptions → 500 ``internal``; the log line carries the exception
    TYPE NAME only — no stack trace, prompt, note text, or filename in the
    response.
"""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from consilium.core.errors import AppError

__all__ = ["PROBLEM_STATUS", "install_problem_handlers", "problem"]

logger = logging.getLogger("consilium.api")

PROBLEM_MEDIA_TYPE = "application/problem+json"

PROBLEM_STATUS: dict[str, int] = {
    "input-rejected": 422,
    "validation-failed": 422,
    "unprocessable": 422,
    "not-found": 404,
    "conflict-status": 409,
    "forbidden": 403,
    "unauthorized": 401,
    "method-not-allowed": 405,
    "payload-too-large": 413,
    "internal": 500,
    "service-unavailable": 503,
}

_HTTP_TO_TYPE: dict[int, str] = {
    401: "unauthorized",
    403: "forbidden",
    404: "not-found",
    405: "method-not-allowed",
    409: "conflict-status",
    413: "payload-too-large",
    422: "validation-failed",
}


def problem(
    ptype: str,
    *,
    status: int | None = None,
    reason: str | None = None,
    extra: dict[str, object] | None = None,
) -> JSONResponse:
    code = status or PROBLEM_STATUS.get(ptype, 500)
    body: dict[str, object] = {
        "type": f"urn:consilium:problem:{ptype}",
        "title": HTTPStatus(code).phrase,
        "status": code,
    }
    if reason is not None:
        body["reason"] = reason
    if extra:
        body.update(extra)
    headers = {"WWW-Authenticate": "Bearer"} if code == 401 else None
    return JSONResponse(body, status_code=code, media_type=PROBLEM_MEDIA_TYPE, headers=headers)


async def _app_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    return problem(exc.problem_type, reason=exc.reason)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {"loc": [str(p) for p in err.get("loc", ())], "type": str(err.get("type", "invalid"))}
        for err in exc.errors()
    ]
    return problem("validation-failed", reason="REQUEST_INVALID", extra={"errors": errors})


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    ptype = _HTTP_TO_TYPE.get(exc.status_code, "internal")
    return problem(ptype, status=exc.status_code)


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # Type name only: str(exc) may carry user content (I-6).
    logger.error(
        "unhandled error",
        extra={"error_type": type(exc).__name__, "path_template": _route_template(request)},
    )
    return problem("internal", reason="INTERNAL_ERROR")


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return str(getattr(route, "path", "unmatched"))


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled)
