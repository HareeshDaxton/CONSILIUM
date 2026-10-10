"""Section 2.5 — VisionClient HTTP contract + retry policy (AGENTS.md §8.0).

Network is mocked with respx — the default suite never calls the network (§20).
"""

import httpx
import pytest
import respx

from consilium.core.errors import VisionFailure
from consilium.schemas.vision import RawVisionOutput
from consilium.vision.client import RetryPolicy, VisionTransient, predict_with_retry
from consilium.vision.client_fake import build_output
from consilium.vision.client_http import HttpVisionClient

PIN = "consilium-vision-abc123def456"
URL = "https://endpoint.example.com"
IMAGE = b"\x89PNG fake-image-bytes"


def _payload(revision: str = PIN) -> dict[str, object]:
    return build_output(p_raw=0.4, revision=revision).model_dump(mode="json")


def _client() -> HttpVisionClient:
    return HttpVisionClient(
        endpoint_url=URL, token="secret-token", pinned_revision=PIN, request_timeout_s=5.0
    )


class TestHttpContract:
    async def test_success_returns_schema(self) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").respond(200, json=_payload())
            out = await _client().predict(IMAGE)
        assert isinstance(out, RawVisionOutput)
        assert out.endpoint_revision == PIN

    async def test_bearer_token_and_octet_stream_sent(self) -> None:
        with respx.mock(base_url=URL) as mock:
            route = mock.post("/predict").respond(200, json=_payload())
            await _client().predict(IMAGE)
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer secret-token"
        assert request.headers["Content-Type"] == "application/octet-stream"
        assert request.content == IMAGE

    async def test_503_is_transient(self) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").respond(503, json={"detail": "loading"})
            with pytest.raises(VisionTransient):
                await _client().predict(IMAGE)

    @pytest.mark.parametrize("status", [502, 504])
    async def test_gateway_statuses_transient(self, status: int) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").respond(status)
            with pytest.raises(VisionTransient):
                await _client().predict(IMAGE)

    async def test_400_is_permanent(self) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").respond(400, json={"detail": "bad request"})
            with pytest.raises(VisionFailure, match="VISION_UNAVAILABLE"):
                await _client().predict(IMAGE)

    async def test_timeout_is_transient(self) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").mock(side_effect=httpx.ConnectTimeout("boom"))
            with pytest.raises(VisionTransient):
                await _client().predict(IMAGE)

    async def test_revision_mismatch_fails_closed(self) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").respond(200, json=_payload(revision="consilium-vision-OTHER"))
            with pytest.raises(VisionFailure, match="VISION_REVISION_MISMATCH"):
                await _client().predict(IMAGE)

    async def test_malformed_payload_fails_closed(self) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").respond(200, json={"p_raw": 2.5})  # invalid + missing fields
            with pytest.raises(VisionFailure, match="VISION_MALFORMED_PAYLOAD"):
                await _client().predict(IMAGE)

    async def test_non_json_payload_fails_closed(self) -> None:
        with respx.mock(base_url=URL) as mock:
            mock.post("/predict").respond(200, content=b"<html>not json</html>")
            with pytest.raises(VisionFailure, match="VISION_MALFORMED_PAYLOAD"):
                await _client().predict(IMAGE)


class TestRetryPolicy:
    def _policy(self, **kw: float | int) -> RetryPolicy:
        base: dict[str, float | int] = {
            "total_budget_s": 5.0,
            "initial_backoff_s": 0.001,
            "max_backoff_s": 0.005,
            "jitter_fraction": 0.0,
            "max_attempts": 5,
        }
        base.update(kw)
        return RetryPolicy(**base)  # type: ignore[arg-type]

    async def test_recovers_after_cold_start(self) -> None:
        calls = 0

        class Flaky:
            async def predict(self, image_bytes: bytes) -> RawVisionOutput:
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise VisionTransient("cold")
                return build_output(p_raw=0.5, revision=PIN)

        out = await predict_with_retry(Flaky(), IMAGE, self._policy())
        assert out.endpoint_revision == PIN
        assert calls == 3

    async def test_budget_exhaustion_fails_closed(self) -> None:
        class Never:
            async def predict(self, image_bytes: bytes) -> RawVisionOutput:
                raise VisionTransient("always cold")

        with pytest.raises(VisionFailure, match="VISION_UNAVAILABLE"):
            await predict_with_retry(Never(), IMAGE, self._policy(max_attempts=3))

    async def test_permanent_error_not_retried(self) -> None:
        calls = 0

        class Bad:
            async def predict(self, image_bytes: bytes) -> RawVisionOutput:
                nonlocal calls
                calls += 1
                raise VisionFailure("VISION_REVISION_MISMATCH")

        with pytest.raises(VisionFailure, match="VISION_REVISION_MISMATCH"):
            await predict_with_retry(Bad(), IMAGE, self._policy())
        assert calls == 1
