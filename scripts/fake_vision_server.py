"""Fake vision HTTP service (Phase 0, Section 0.4 — stub).

Stands in for the private HF Inference Endpoint in the local compose stack so
e2e flows can exercise real HTTP + retries without the real endpoint
(ARCHITECTURE.md §5.4.2, I-13).

This is the STUB: it returns one canned RawVisionOutput-shaped payload with a
fake revision. Phase 2 replaces the payload logic with deterministic canned
outputs keyed by sha256(image) plus the failure modes (cold_start, flaky_503,
revision_mismatch, malformed, empty_mask) from SKILLS.md vision-endpoint.
The RLE strings below are placeholders — the real RLE codec lands in
src/consilium/vision/rle.py (Phase 2, Section 2.3).
"""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

FAKE_REVISION = os.environ.get("FAKE_VISION_REVISION", "sha256:fake/dev-fake-000")

app = FastAPI(title="consilium-fake-vision", docs_url=None, redoc_url=None)

CANNED_PAYLOAD: dict[str, object] = {
    "p_raw": 0.5,
    "disc_mask_rle": "STUB_RLE_NOT_REAL",
    "cup_mask_rle": "STUB_RLE_NOT_REAL",
    "mask_width": 1024,
    "mask_height": 1024,
    "endpoint_revision": FAKE_REVISION,
}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "fake-vision", "revision": FAKE_REVISION}


@app.post("/predict")
async def predict() -> JSONResponse:
    # Stub: ignores the body entirely. P2 keys responses by sha256(image).
    return JSONResponse(CANNED_PAYLOAD)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)  # noqa: S104  # container service
