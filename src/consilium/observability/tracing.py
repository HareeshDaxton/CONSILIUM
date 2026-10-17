"""OTel tracer provider + OTLP exporter wiring (AGENTS.md §18).

Called once by the production composition root. Without ``OTEL_ENDPOINT``
the global no-op provider stays in place (spans cost nothing). Spans never
carry bodies by default — see ``observability/instrument.py``.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from consilium.core.settings import Settings


def configure_tracing(settings: Settings, service: str = "consilium") -> bool:
    """Install an OTLP/HTTP exporter when configured. Returns True if installed."""
    if not settings.otel_endpoint:
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
    )
    trace.set_tracer_provider(provider)
    return True
