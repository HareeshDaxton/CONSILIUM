"""Observability (plan Section 14.1): metrics, span privacy, alerts <-> runbook."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

from consilium.llm.client_fake import FakeLLMClient
from consilium.observability import instrument
from consilium.observability import metrics as m
from consilium.observability.langfuse_sink import Generation, LangfuseSdkSink
from consilium.schemas.agents import SubReport
from consilium.schemas.audit import Issue, MergedVerdict
from consilium.schemas.enums import CaseStatus, IssueType
from consilium.vision import client_fake as vf
from consilium.vision.config import load_models_config

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "deploy" / "observability"
SECRET = "PATIENT-NOTE-TEXT-SENTINEL"
REPORT = {"role": "pharmacist", "findings": [], "recommendations": []}


def _val(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(instrument, "_tracer", provider.get_tracer("test"))
    return exporter


async def test_llm_calls_are_metered_and_spans_carry_no_bodies(
    spans: InMemorySpanExporter,
) -> None:
    from consilium.llm.client import ChatMessage

    sent: list[Generation] = []

    class Sink:
        def generation(self, gen: Generation) -> None:
            sent.append(gen)

    before = _val("consilium_llm_calls_total", agent="pharmacist", status="ok")
    llm = instrument.InstrumentedLLM(
        FakeLLMClient.valid(REPORT, agent="pharmacist", prompt_version="v1"), Sink()
    )
    await llm.generate(
        [ChatMessage(role="user", content=SECRET)],
        SubReport,
        model="snap",
        temperature=0.0,
        max_output_tokens=10,
        timeout_s=1.0,
        tags={"agent": "pharmacist", "prompt_version": "v1"},
    )
    assert _val("consilium_llm_calls_total", agent="pharmacist", status="ok") == before + 1
    (span,) = spans.get_finished_spans()
    assert span.attributes is not None and span.attributes["llm.agent"] == "pharmacist"
    assert SECRET not in json.dumps(dict(span.attributes))
    assert sent[0].status == "ok" and SECRET not in repr(sent[0])


async def test_vision_cold_start_and_success_are_counted(spans: InMemorySpanExporter) -> None:
    pin = load_models_config().endpoint.pinned_revision
    client = instrument.InstrumentedVision(vf.FakeVisionClient(revision=pin, cold_start_calls=1))
    cold = _val("consilium_vision_cold_start_total")
    with pytest.raises(vf.VisionTransient):
        await client.predict(b"img")
    out = await client.predict(b"img")
    assert out.endpoint_revision == pin
    assert _val("consilium_vision_cold_start_total") == cold + 1
    assert [s.attributes["vision.cold_start"] for s in spans.get_finished_spans()] == [True, False]  # type: ignore[index]


def test_transition_counters() -> None:
    na = _val("consilium_needs_attention_total")
    issue = Issue(type=IssueType.UNSUPPORTED, claim_id="c1", detail="d", raised_by="llm_critic")
    rejected = MergedVerdict(approved=False, issues=(issue,), round=3, critic_ran=True)
    crit = _val("consilium_critic_issues_total", type="unsupported", layer="llm_critic")
    m.observe_node("audit", 0.1, CaseStatus.AUDITING, CaseStatus.NEEDS_ATTENTION, rejected)
    assert _val("consilium_needs_attention_total") == na + 1
    assert _val("consilium_critic_issues_total", type="unsupported", layer="llm_critic") == crit + 1
    m.record_review("APPROVE", ())
    assert _val("consilium_review_decisions_total", action="APPROVE") >= 1


def test_langfuse_sdk_errors_never_propagate() -> None:
    class Broken:
        def start_observation(self, **_: Any) -> Any:
            raise RuntimeError("langfuse down")

    LangfuseSdkSink(Broken()).generation(Generation("a", "m", "v1", 1, 1, 1.0, "ok"))


def test_every_alert_has_a_runbook_entry_and_known_metrics() -> None:
    rules = yaml.safe_load((OBS / "alerts.yml").read_text(encoding="utf-8"))
    alerts = [r for g in rules["groups"] for r in g["rules"]]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    known = {
        name
        for metric in REGISTRY.collect()
        for name in (
            metric.name,
            f"{metric.name}_total",
            f"{metric.name}_bucket",
            f"{metric.name}_count",
        )
    }
    dashboard = json.loads((OBS / "grafana" / "dashboards" / "consilium.json").read_text("utf-8"))
    exprs = [a["expr"] for a in alerts] + [
        t["expr"] for p in dashboard["panels"] for t in p["targets"]
    ]
    for alert in alerts:
        assert f"| `{alert['alert']}` |" in readme, alert["alert"]
    for expr in exprs:
        for used in re.findall(r"consilium_[a-z0-9_]+", expr):
            assert used in known, used
