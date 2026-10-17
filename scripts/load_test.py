"""Load test of the pipeline on the dev stack (plan Section 14.3).

Runs N cases with C in flight through the REAL GraphRunner and the REAL local
Postgres (docker compose), with the offline canned LLM and fake vision server
plus EMULATED provider latency (``--llm-ms``, ``--vision-ms``). This measures
what we own: orchestration, persistence after every node, the audit loop and
DB contention under concurrency. It does NOT measure OpenAI or GPU latency or
rate limits. Those come from the live eval run (doc/eval_runs) and from the
endpoint itself.

Per-stage latency = time between consecutive node persists of a case (node
work + its Postgres transaction). Cost per case = tokens x list price
(configs/eval.yaml), an estimate. The canned client's token counts are
synthetic, so for real cost per case see the live eval report.

    uv run python scripts/load_test.py --cases 50 --concurrency 10 --out doc/load_test.md

Target (proposal, to be agreed): 0 failures, and end-to-end p95 <= 1.5x the
emulated critical path (vision + extractor + specialists + director + critic).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "tests" / "integration", REPO):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from graph_kit import (  # noqa: E402
    IMAGE,
    IMAGE_URI,
    NOTE_TEXT,
    PIN,
    clean_draft,
    expected_ctx,
    fake_llm,
)
from pydantic import BaseModel  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from consilium.core.settings import Settings  # noqa: E402
from consilium.eval.ablation import load_eval_config  # noqa: E402
from consilium.graph.build import GraphRunner  # noqa: E402
from consilium.graph.deps import MemoryImageStore, build_fake_deps  # noqa: E402
from consilium.graph.state import CaseState  # noqa: E402
from consilium.llm.client import ChatMessage, LLMClient, ParsedResponse  # noqa: E402
from consilium.persistence.db import make_session_factory  # noqa: E402
from consilium.persistence.encryption import AesGcmFieldEncryptor  # noqa: E402
from consilium.persistence.repositories import Repositories  # noqa: E402
from consilium.schemas.vision import RawVisionOutput  # noqa: E402
from consilium.vision import client_fake as vf  # noqa: E402
from consilium.vision.client import VisionClient  # noqa: E402

T = TypeVar("T", bound=BaseModel)
ADMIN_URL = os.environ.get(
    "CONSILIUM_TEST_ADMIN_URL", "postgresql+asyncpg://consilium:consilium@localhost:5432/postgres"
)
LOAD_DB = "consilium_load"
MODEL = "gpt-4o-mini-2024-07-18"


class SlowLLM:
    def __init__(self, inner: LLMClient, delay_s: float) -> None:
        self.inner, self.delay_s = inner, delay_s
        self.tokens: dict[str, int] = defaultdict(int)

    async def generate(
        self, messages: list[ChatMessage], response_model: type[T], **kw: Any
    ) -> ParsedResponse[T]:
        await asyncio.sleep(self.delay_s)
        resp = await self.inner.generate(messages, response_model, **kw)
        self.tokens["input"] += resp.input_tokens
        self.tokens["output"] += resp.output_tokens
        return resp


class SlowVision:
    def __init__(self, inner: VisionClient, delay_s: float) -> None:
        self.inner, self.delay_s = inner, delay_s

    async def predict(self, image_bytes: bytes) -> RawVisionOutput:
        await asyncio.sleep(self.delay_s)
        return await self.inner.predict(image_bytes)


async def _fresh_db() -> str:
    from alembic import command
    from alembic.config import Config

    admin = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{LOAD_DB}" WITH (FORCE)'))
        await conn.execute(text(f'CREATE DATABASE "{LOAD_DB}"'))
    await admin.dispose()
    url = ADMIN_URL.rsplit("/", 1)[0] + f"/{LOAD_DB}"
    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    await asyncio.to_thread(command.upgrade, cfg, "head")
    return url


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


async def run(cases: int, concurrency: int, llm_ms: int, vision_ms: int) -> dict[str, Any]:
    url = await _fresh_db()
    engine = create_async_engine(url, pool_size=concurrency + 5, pool_pre_ping=True)
    repos = Repositories(make_session_factory(engine), AesGcmFieldEncryptor(b"k" * 32))
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        OPENAI_API_KEY="sk-load-test-synthetic",
        DATABASE_URL=url,
        JWT_SECRET="load-test-jwt-secret-synthetic-32-bytes",  # noqa: S106 - synthetic
        LLM_CASE_TOKEN_BUDGET=10_000_000,
    )
    llm = SlowLLM(fake_llm(clean_draft(await expected_ctx())), llm_ms / 1000)
    vision = SlowVision(vf.FakeVisionClient(revision=PIN), vision_ms / 1000)
    deps = build_fake_deps(
        settings, llm=llm, vision=vision, image_store=MemoryImageStore({IMAGE_URI: IMAGE})
    )
    marks: dict[str, float] = {}
    node_ms: dict[str, list[float]] = defaultdict(list)

    def hook(node: str, state: CaseState) -> None:
        now = time.perf_counter()
        node_ms[node].append((now - marks.get(state.case_id, now)) * 1000)
        marks[state.case_id] = now

    runner = GraphRunner(deps, repos, on_node_persisted=hook)
    gate = asyncio.Semaphore(concurrency)
    e2e: list[float] = []
    statuses: dict[str, int] = defaultdict(int)

    async def one(i: int) -> None:
        async with gate:
            started = time.perf_counter()
            state = await runner.submit(
                image_uri=IMAGE_URI,
                image_sha256="a" * 64,
                created_by="load-test",
                raw_note=NOTE_TEXT,
                idempotency_key=f"load-{i}",
            )
            marks[state.case_id] = started
            final = await runner.run(state)
            e2e.append((time.perf_counter() - started) * 1000)
            statuses[final.status.value] += 1

    wall = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(cases)))
    wall = time.perf_counter() - wall
    await engine.dispose()

    price = load_eval_config().pricing[MODEL]
    cost = (llm.tokens["input"] * price["input"] + llm.tokens["output"] * price["output"]) / 1e6
    llm_calls_on_path = 5  # extractor, specialists (parallel), director, critic: 1 round each
    critical_ms = vision_ms + llm_ms * llm_calls_on_path
    p95 = _pct(e2e, 95)
    return {
        "cases": cases,
        "concurrency": concurrency,
        "emulated_llm_ms": llm_ms,
        "emulated_vision_ms": vision_ms,
        "statuses": dict(statuses),
        "throughput_cases_per_min": round(cases / wall * 60, 1),
        "e2e_ms": {"p50": round(statistics.median(e2e)), "p95": round(p95)},
        "critical_path_ms": critical_ms,
        "target_p95_ms": round(critical_ms * 1.5),
        "target_met": p95 <= critical_ms * 1.5 and set(statuses) == {"PENDING_REVIEW"},
        "per_node_ms": {
            n: {"p50": round(statistics.median(v)), "p95": round(_pct(v, 95))}
            for n, v in node_ms.items()
        },
        "est_cost_per_case_usd_synthetic_tokens": round(cost / cases, 6),
    }


def _markdown(r: dict[str, Any]) -> str:
    rows = "\n".join(f"| `{n}` | {v['p50']} | {v['p95']} |" for n, v in r["per_node_ms"].items())
    target = f"Target (proposal): 0 failures, p95 <= 1.5x critical path ({r['target_p95_ms']} ms)"
    verdict = "MET" if r["target_met"] else "NOT MET"
    return f"""# Load test (Phase 14.3)

Generated by `scripts/load_test.py`: the real GraphRunner and the real local Postgres, with the
canned LLM and fake vision plus emulated latency (LLM {r["emulated_llm_ms"]} ms per call, vision
{r["emulated_vision_ms"]} ms). This measures orchestration, persistence and DB contention, not
OpenAI or GPU latency (see `doc/eval_runs/` for the live run).

| Metric | Value |
|---|---|
| Cases / concurrency | {r["cases"]} / {r["concurrency"]} |
| Final statuses | {r["statuses"]} |
| Throughput | {r["throughput_cases_per_min"]} cases/min |
| End-to-end p50 / p95 | {r["e2e_ms"]["p50"]} / {r["e2e_ms"]["p95"]} ms |
| Emulated critical path | {r["critical_path_ms"]} ms |
| {target} | {verdict} |
| Est. cost per case (synthetic token counts) | ${r["est_cost_per_case_usd_synthetic_tokens"]} |

Per node (time from previous persist to this node's persist, ms):

| Node | p50 | p95 |
|---|---|---|
{rows}

Raw result:

```json
{json.dumps(r, indent=2)}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="load_test")
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--llm-ms", type=int, default=800)
    parser.add_argument("--vision-ms", type=int, default=1500)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(args.cases, args.concurrency, args.llm_ms, args.vision_ms))
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
