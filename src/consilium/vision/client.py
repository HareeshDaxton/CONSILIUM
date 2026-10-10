"""VisionClient protocol + retry policy (AGENTS.md §8.0, §13).

The endpoint may scale to zero (cold starts). 503s and transport timeouts are
*transient* and retried with exponential backoff + jitter, bounded by the
total wait budget (``VISION_TIMEOUT_S``, default 120s). If the budget is
exceeded the caller raises ``VisionFailure(VISION_UNAVAILABLE)`` — never a
degraded report (I-5).

Two implementations:
  - ``client_http.HttpVisionClient`` — the real Hugging Face Inference Endpoint
  - ``client_fake.FakeVisionClient`` — deterministic canned outputs (dev/tests)

Only ``client_http`` may import httpx (import-linter enforces this).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from consilium.core.errors import VisionFailure
from consilium.core.logging import get_logger

if TYPE_CHECKING:
    from consilium.schemas.vision import RawVisionOutput

logger = get_logger(__name__)


class VisionTransient(Exception):
    """Retryable vision call failure (503, connect timeout, cold start).

    Carries no payloads — detail strings must never contain image data (I-14).
    """


class VisionClient(Protocol):
    """The only interface the pipeline uses for remote vision inference."""

    async def predict(self, image_bytes: bytes) -> RawVisionOutput:
        """Run one inference pass. Raises VisionTransient (retryable) or
        VisionFailure (permanent: revision mismatch, malformed payload)."""
        ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry for cold-start tolerance. Total wall-clock budget wins
    over attempt count — a slow attempt consumes budget like any other."""

    total_budget_s: float = 120.0  # == VISION_TIMEOUT_S default
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 15.0
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.25  # +/- uniform jitter on each sleep
    max_attempts: int = 10


async def predict_with_retry(
    client: VisionClient,
    image_bytes: bytes,
    policy: RetryPolicy,
) -> RawVisionOutput:
    """Call ``client.predict`` retrying VisionTransient until the budget or
    attempt cap is exhausted, then fail closed with VISION_UNAVAILABLE."""
    started = time.monotonic()
    backoff = policy.initial_backoff_s
    attempt = 0
    while True:
        attempt += 1
        try:
            return await client.predict(image_bytes)
        except VisionTransient:
            elapsed = time.monotonic() - started
            if attempt >= policy.max_attempts or elapsed + backoff > policy.total_budget_s:
                logger.warning(
                    "vision.unavailable",
                    extra={"attempts": attempt, "elapsed_s": round(elapsed, 2)},
                )
                raise VisionFailure(
                    "VISION_UNAVAILABLE",
                    detail=f"endpoint not ready after {attempt} attempt(s) "
                    f"in {elapsed:.1f}s (budget {policy.total_budget_s:.1f}s)",
                ) from None
            sleep_s = backoff * (
                1.0
                + random.uniform(  # noqa: S311 — jitter, not crypto
                    -policy.jitter_fraction, policy.jitter_fraction
                )
            )
            logger.info(
                "vision.retry",
                extra={"attempt": attempt, "sleep_s": round(sleep_s, 2)},
            )
            await asyncio.sleep(sleep_s)
            backoff = min(backoff * policy.backoff_multiplier, policy.max_backoff_s)
