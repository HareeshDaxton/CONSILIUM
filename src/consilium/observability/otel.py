"""OpenTelemetry request tracing for the API (AGENTS.md §18).

Pure ASGI middleware: one span per HTTP request named by the ROUTE TEMPLATE
(``/v1/cases/{case_id}``), never the raw path or query string. Attributes are
method, route template and status code only — no headers, bodies, tokens,
filenames or image data (I-6/I-14). Exporter wiring (OTLP endpoint) is a
deployment concern; without a configured provider the API tracer is a no-op.
"""

from __future__ import annotations

from opentelemetry import trace
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["RequestTracingMiddleware"]

_tracer = trace.get_tracer("consilium.api")


class RequestTracingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method", "GET"))
        with _tracer.start_as_current_span(f"HTTP {method}") as span:
            span.set_attribute("http.request.method", method)
            status_holder: list[int] = []

            async def send_wrapper(message: Message) -> None:
                if message["type"] == "http.response.start":
                    status_holder.append(int(message["status"]))
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                route = scope.get("route")
                template = str(getattr(route, "path", "unmatched"))
                span.update_name(f"HTTP {method} {template}")
                span.set_attribute("http.route", template)
                if status_holder:
                    span.set_attribute("http.response.status_code", status_holder[0])
