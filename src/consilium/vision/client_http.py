"""HTTP VisionClient — talks to the Hugging Face Inference Endpoint (I-13/I-14).

Sends the raw image bytes once, per call, to the private endpoint. The image
never goes anywhere else (never to the LLM provider, never into logs/traces).
Every response's ``endpoint_revision`` is checked against the pin — a mismatch
fails closed (§8.0).

This module is the ONLY place in src/ allowed to import httpx (import-linter).
"""

from __future__ import annotations

import httpx
from pydantic import ValidationError

from consilium.core.errors import VisionFailure
from consilium.core.logging import get_logger
from consilium.schemas.vision import RawVisionOutput
from consilium.vision.client import VisionHealth, VisionTransient

logger = get_logger(__name__)

# HF returns 503 while the model is loading (cold start); 502/504 from the
# gateway during scale-up are equally transient.
_TRANSIENT_STATUSES = frozenset({502, 503, 504})


class HttpVisionClient:
    """VisionClient over HTTPS. Reuse one instance per process (connection pool)."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        token: str,
        pinned_revision: str,
        request_timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,  # tests: in-process fake server
    ) -> None:
        self._pinned_revision = pinned_revision
        self._client = httpx.AsyncClient(
            base_url=endpoint_url,
            headers={
                "Authorization": f"Bearer {token}",
                # Deva's server is published through ngrok; this header stops the
                # free tier from answering with its HTML interstitial page.
                "ngrok-skip-browser-warning": "1",
            },
            timeout=httpx.Timeout(request_timeout_s),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self, timeout_s: float = 5.0) -> VisionHealth:
        """GET /healthz on the vision server (Deva's GPU laptop / HF endpoint)."""
        try:
            response = await self._client.get("/healthz", timeout=timeout_s)
        except httpx.HTTPError as exc:
            return VisionHealth(reachable=False, detail=type(exc).__name__)
        if response.status_code != 200:
            return VisionHealth(reachable=False, detail=f"status {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            return VisionHealth(reachable=False, detail="non-JSON health response")
        revision = body.get("revision") if isinstance(body, dict) else None
        device = body.get("device") if isinstance(body, dict) else None
        return VisionHealth(
            reachable=True,
            revision=revision if isinstance(revision, str) else None,
            device=device if isinstance(device, str) else None,
        )

    async def predict(self, image_bytes: bytes) -> RawVisionOutput:
        try:
            response = await self._client.post(
                "/predict",
                content=image_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.TransportError as exc:
            # Any connection-level failure (timeout, connect, dropped read/write,
            # protocol error — e.g. a tunnel closing mid-request) is retryable:
            # inference is stateless, and the retry budget still fails closed
            # as VISION_UNAVAILABLE instead of an unmapped INTERNAL error.
            raise VisionTransient(f"transport: {type(exc).__name__}") from exc

        if response.status_code in _TRANSIENT_STATUSES:
            raise VisionTransient(f"endpoint status {response.status_code} (cold start/scale-up)")
        if response.status_code != 200:
            raise VisionFailure(
                "VISION_UNAVAILABLE",
                detail=f"endpoint returned non-retryable status {response.status_code}",
            )

        try:
            output = RawVisionOutput.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise VisionFailure(
                "VISION_MALFORMED_PAYLOAD",
                detail=f"response failed schema validation: {type(exc).__name__}",
            ) from exc

        if output.endpoint_revision != self._pinned_revision:
            # Fail closed + alert (AGENTS.md §8.0): the deployed model no longer
            # matches what this repo's thresholds/calibration were tuned against.
            raise VisionFailure(
                "VISION_REVISION_MISMATCH",
                detail=f"endpoint revision != pinned revision ({self._pinned_revision})",
            )
        return output
