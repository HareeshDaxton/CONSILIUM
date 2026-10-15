"""NeMo Guardrails wiring — the MODEL-BASED layer of the rails (AGENTS.md §14).

Division of responsibility: cheap exact patterns live in ``input_checks`` /
``output_checks`` and run first; this module covers only what needs a model
(chiefly prompt-injection detection on the note, and a semantic policy check
on the draft). Config: ``configs/guardrails/config.yml`` + ``rails/*.co``.

Engines (mirrors the LLMClient/VisionClient pattern):
  - ``NemoRailEngine`` — real NeMo (``nemoguardrails``), imported LAZILY so
    the API image and the offline test suite never need it. It is handed the
    REDACTED note only (I-6/I-7); the raw note never reaches a provider.
  - ``FakeRailEngine`` — deterministic trigger-substring engine for tests (§20).

``GuardedRails`` wraps an engine with timing, Prometheus metrics (rail
latency histogram, trigger counter — injected registry, no global state) and
an in-memory latency recorder for p50/p95. Fail closed (I-5): an engine
error or timeout is a block (``RAIL_UNAVAILABLE``), never a silent pass.

Documented gap list per rule: ``GAPS`` (what each rail does NOT catch).
"""

from __future__ import annotations

import asyncio
import importlib
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from prometheus_client import CollectorRegistry, Counter, Histogram

from consilium.core.errors import PolicyViolation
from consilium.schemas.base import Strict

RailName = Literal["input", "output"]

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "guardrails"

# Rule names as they appear in audit events / metrics.
INPUT_RULE = "input_note_injection"
OUTPUT_RULE = "output_draft_policy"
UNAVAILABLE_RULE = "rail_unavailable"
_BLOCK_MARKER = "RAIL_BLOCKED:"


@dataclass(frozen=True)
class RailGap:
    rule: str
    gap: str


# What each rule does NOT catch — reviewed with the rail config; every gap is
# either covered by another layer (named) or an accepted V1 limitation.
GAPS: tuple[RailGap, ...] = (
    RailGap(
        INPUT_RULE,
        "Model judgement: paraphrased or novel injections may pass "
        "(deterministic input_checks patterns + untrusted-data prompt contract "
        "+ schema-validated extraction are the backstops).",
    ),
    RailGap(
        INPUT_RULE,
        "Runs on the REDACTED note: injections hidden inside text that "
        "de-identification replaced are not seen (they are also not seen by the LLM).",
    ),
    RailGap(
        INPUT_RULE,
        "Non-English injections depend on the rail model's multilingual "
        "ability; not measured in V1.",
    ),
    RailGap(
        OUTPUT_RULE,
        "Does not verify numbers or evidence (deterministic verifier + LLM critic own that, §13).",
    ),
    RailGap(
        OUTPUT_RULE,
        "Model judgement on tone/scope; exact diagnosis/dose/PHI-echo "
        "patterns are enforced deterministically in output_checks.",
    ),
    RailGap(
        UNAVAILABLE_RULE,
        "Fail closed: an unavailable rail model blocks the case "
        "(NEEDS_ATTENTION) — availability, not content, is the failure.",
    ),
)


class RailDecision(Strict):
    rail: RailName
    blocked: bool
    rule: str | None = None


class RailEngine(Protocol):
    async def check(self, rail: RailName, text: str) -> RailDecision: ...


@dataclass(frozen=True)
class FakeRailEngine:
    """Deterministic engine for offline tests: blocks when any trigger
    substring (case-insensitive) occurs in the text for that rail."""

    input_triggers: tuple[str, ...] = ()
    output_triggers: tuple[str, ...] = ()
    fail: bool = False  # simulate an unavailable rail model
    delay_s: float = 0.0

    async def check(self, rail: RailName, text: str) -> RailDecision:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise ConnectionError("fake rail unavailable")
        triggers = self.input_triggers if rail == "input" else self.output_triggers
        lowered = text.lower()
        hit = any(t.lower() in lowered for t in triggers)
        rule = (INPUT_RULE if rail == "input" else OUTPUT_RULE) if hit else None
        return RailDecision(rail=rail, blocked=hit, rule=rule)


class NemoRailEngine:
    """Real NeMo Guardrails engine (lazy import; not used by the default suite).

    Requires the optional ``nemoguardrails`` package (see Kimi.md manual steps).
    The colang flows answer with ``RAIL_BLOCKED:<rule>`` when they stop.
    """

    def __init__(self, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
        nemo = importlib.import_module("nemoguardrails")
        self._rails = nemo.LLMRails(nemo.RailsConfig.from_path(str(config_dir)))

    async def check(self, rail: RailName, text: str) -> RailDecision:
        if rail == "input":
            messages = [{"role": "user", "content": text}]
        else:
            messages = [{"role": "user", "content": "."}, {"role": "assistant", "content": text}]
        res = await self._rails.generate_async(messages=messages, options={"rails": [rail]})
        content = str(getattr(res, "response", [{}])[-1].get("content", ""))
        if _BLOCK_MARKER in content:
            rule = content.split(_BLOCK_MARKER, 1)[1].strip() or (
                INPUT_RULE if rail == "input" else OUTPUT_RULE
            )
            return RailDecision(rail=rail, blocked=True, rule=rule)
        return RailDecision(rail=rail, blocked=False)


@dataclass
class LatencyRecorder:
    samples: dict[str, list[float]] = field(default_factory=lambda: {"input": [], "output": []})

    def add(self, rail: RailName, seconds: float) -> None:
        self.samples[rail].append(seconds)

    def percentiles(self, rail: RailName) -> tuple[float, float]:
        """(p50, p95) in seconds; requires ≥2 samples."""
        qs = statistics.quantiles(self.samples[rail], n=100, method="inclusive")
        return qs[49], qs[94]


class RailMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.latency = Histogram(
            "consilium_rail_latency_seconds",
            "Guardrail (NeMo layer) latency",
            ["rail"],
            registry=registry,
        )
        self.triggers = Counter(
            "consilium_rail_triggers_total",
            "Guardrail triggers by rail and rule",
            ["rail", "rule"],
            registry=registry,
        )


class GuardedRails:
    """Engine + timing + metrics + fail-closed mapping to PolicyViolation."""

    def __init__(
        self,
        engine: RailEngine,
        *,
        registry: CollectorRegistry | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._engine = engine
        self._timeout_s = timeout_s
        self.metrics = RailMetrics(registry if registry is not None else CollectorRegistry())
        self.recorder = LatencyRecorder()

    async def check_input(self, redacted_note: str) -> None:
        await self._check("input", redacted_note, "NOTE_INJECTION_MODEL")

    async def check_output(self, draft_text: str) -> None:
        await self._check("output", draft_text, "DRAFT_POLICY_MODEL")

    async def _check(self, rail: RailName, text: str, reason: str) -> None:
        start = time.perf_counter()
        try:
            decision = await asyncio.wait_for(self._engine.check(rail, text), self._timeout_s)
        except Exception:  # fail closed: availability failure is a block (I-5)
            decision = RailDecision(rail=rail, blocked=True, rule=UNAVAILABLE_RULE)
            reason = "RAIL_UNAVAILABLE"
        finally:
            elapsed = time.perf_counter() - start
            self.metrics.latency.labels(rail=rail).observe(elapsed)
            self.recorder.add(rail, elapsed)
        if decision.blocked:
            rule = decision.rule or "unknown"
            self.metrics.triggers.labels(rail=rail, rule=rule).inc()
            raise PolicyViolation(
                reason, case_status="NEEDS_ATTENTION", detail=f"{rail}_rail.nemo:{rule}"
            )
