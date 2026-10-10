"""Fake vision HTTP service (Phase 2, Section 2.5).

Stands in for the private HF Inference Endpoint in the local compose stack so
e2e flows can exercise real HTTP + retries without the real endpoint
(ARCHITECTURE.md §5.4.2, I-13).

Serves the same deterministic fixtures as ``FakeVisionClient``
(src/consilium/vision/client_fake.py): real RLE-encoded synthetic masks, a
revision taken from configs/models.yaml so the pin check passes locally, and a
FAKE_COLD_START_CALLS env knob to simulate scale-to-zero (503 for the first N
requests).
"""

import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from consilium.vision.client_fake import build_output
from consilium.vision.config import load_models_config

REVISION = os.environ.get("FAKE_VISION_REVISION") or load_models_config().endpoint.pinned_revision
COLD_START_CALLS = int(os.environ.get("FAKE_VISION_COLD_START_CALLS", "0"))

app = FastAPI(title="consilium-fake-vision", docs_url=None, redoc_url=None)

_payload = build_output(p_raw=0.5, revision=REVISION).model_dump(mode="json")
_calls = 0


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "fake-vision", "revision": REVISION}


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    global _calls
    _calls += 1
    await request.body()  # consume the image; never logged or stored (I-14)
    if _calls <= COLD_START_CALLS:
        return JSONResponse({"detail": "model loading"}, status_code=503)
    return JSONResponse(_payload)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)  # noqa: S104  # container service
