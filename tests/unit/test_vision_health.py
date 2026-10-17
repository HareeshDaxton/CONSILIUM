"""HttpVisionClient.health — how the upload page knows Deva's GPU server is up.
Offline: httpx MockTransport (no network)."""

from __future__ import annotations

import httpx

from consilium.vision.client import SupportsHealth
from consilium.vision.client_fake import FakeVisionClient
from consilium.vision.client_http import HttpVisionClient


def _client(handler) -> HttpVisionClient:  # type: ignore[no-untyped-def]
    return HttpVisionClient(
        endpoint_url="http://deva.test",
        token="t" * 20,
        pinned_revision="rev-1",
        transport=httpx.MockTransport(handler),
    )


async def test_reachable_server_reports_revision_and_device() -> None:
    client = _client(
        lambda req: httpx.Response(
            200, json={"status": "ok", "revision": "rev-1", "device": "cuda"}
        )
    )
    health = await client.health()
    assert (health.reachable, health.revision, health.device) == (True, "rev-1", "cuda")
    assert isinstance(client, SupportsHealth)


async def test_unreachable_and_bad_responses_are_not_reachable() -> None:
    def down(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert (await _client(down).health()).detail == "ConnectError"
    assert (await _client(lambda r: httpx.Response(502)).health()).reachable is False
    garbage = await _client(lambda r: httpx.Response(200, text="<html>")).health()
    assert garbage.reachable is False


async def test_fake_client_health() -> None:
    assert (await FakeVisionClient(revision="x").health()).reachable is True
    assert (await FakeVisionClient(revision="x", always_transient=True).health()).reachable is False


async def test_requests_carry_the_ngrok_bypass_header() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"status": "ok", "revision": "rev-1", "device": "cuda"})

    await _client(handler).health()
    assert seen[0].headers["ngrok-skip-browser-warning"] == "1"
    assert seen[0].headers["authorization"] == "Bearer " + "t" * 20
