"""CONSILIUM vision server — runs on Deva's GPU laptop.

Receives one fundus image from the CONSILIUM backend, runs
  * SwinV2    (glaucoma classifier)           -> P(glaucoma)
  * SegFormer (optic disc / cup segmenter)    -> disc + cup masks
on the GPU and sends back the RAW outputs only (wire.py). Everything clinical —
cup-to-disc ratio, risk tier, quality checks, the report — is computed by the
CONSILIUM project, never here (no CDR, no verdict, no thresholds in this folder).

On start it: creates a shared token (first run), downloads the two pinned models
(first run), loads them on the GPU, opens an ngrok tunnel and prints the public
URL + token to give to Hareesh.

Privacy: images are processed in memory, never written to disk or logged.

Run:  run.bat  (Windows)  /  ./run.sh  (Linux)  — or:  python server.py
"""

from __future__ import annotations

import hmac
import io
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
import wire
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    Swinv2ForImageClassification,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("consilium-vision")

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"
MODELS_DIR = HERE / "models"


# ------------------------------------------------------------------ settings


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines) — no extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ensure_token() -> str:
    """First run: generate the shared secret and save it in Deva/.env."""
    token = os.environ.get("CONSILIUM_VISION_TOKEN", "").strip()
    if len(token) >= 16:
        return token
    token = secrets.token_urlsafe(32)
    if ENV_FILE.exists():
        lines = [
            ln
            for ln in ENV_FILE.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("CONSILIUM_VISION_TOKEN=")
        ]
    else:
        template = HERE / ".env.example"
        lines = template.read_text(encoding="utf-8").splitlines() if template.exists() else []
        lines = [ln for ln in lines if not ln.strip().startswith("CONSILIUM_VISION_TOKEN=")]
    lines.append(f"CONSILIUM_VISION_TOKEN={token}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["CONSILIUM_VISION_TOKEN"] = token
    log.info("Created a new shared token in %s", ENV_FILE)
    return token


def _ensure_models() -> None:
    """First run: download the pinned models into Deva/models/ (needs internet once)."""
    manifest = MODELS_DIR / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded.get("revision") == wire.REVISION:
            return
        log.warning("Deva/models/ holds other model versions - downloading the pinned ones")
    log.info("Downloading SwinV2 + SegFormer (about 290 MB, first run only)...")
    import download_models

    download_models.main()


def _pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        raise SystemExit("DEVICE=cuda, but PyTorch sees no CUDA GPU (install the CUDA build).")
    log.warning("No CUDA GPU found - running on CPU (slower, still correct).")
    return torch.device("cpu")


# ------------------------------------------------------------------ models


class Models:
    """Both models, loaded once from Deva/models/ at their pinned versions."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        cls_dir, seg_dir = MODELS_DIR / "classifier", MODELS_DIR / "segmenter"
        log.info("Loading classifier %s @ %s", wire.CLASSIFIER_ID, wire.CLASSIFIER_SHA[:12])
        self.cls_processor = AutoImageProcessor.from_pretrained(
            cls_dir, local_files_only=True, use_fast=False
        )
        self.classifier = (
            Swinv2ForImageClassification.from_pretrained(cls_dir, local_files_only=True)
            .to(device)
            .eval()
        )
        log.info("Loading segmenter %s @ %s", wire.SEGMENTER_ID, wire.SEGMENTER_SHA[:12])
        self.seg_processor = AutoImageProcessor.from_pretrained(
            seg_dir, local_files_only=True, use_fast=False
        )
        self.segmenter = (
            SegformerForSemanticSegmentation.from_pretrained(seg_dir, local_files_only=True)
            .to(device)
            .eval()
        )
        # Label indices from the model configs (never hard-coded).
        cls_labels = {v.lower(): int(k) for k, v in self.classifier.config.id2label.items()}
        seg_labels = {v.lower(): int(k) for k, v in self.segmenter.config.id2label.items()}
        self.glaucoma_idx = cls_labels["glaucoma"]
        self.disc_idx = seg_labels["optic disc"]
        self.cup_idx = seg_labels["optic cup"]
        self.lock = threading.Lock()  # one image at a time on the GPU

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, object]:
        """Raw outputs only: P(glaucoma) + disc/cup masks. No CDR, no verdict."""
        width, height = image.size
        mask_w, mask_h = wire.mask_size(width, height)
        with self.lock:
            cls_in = self.cls_processor(images=image, return_tensors="pt").to(self.device)
            logits = self.classifier(**cls_in).logits
            p_raw = float(torch.softmax(logits.float(), dim=-1)[0, self.glaucoma_idx].item())

            seg_in = self.seg_processor(images=image, return_tensors="pt").to(self.device)
            seg_logits = self.segmenter(**seg_in).logits  # (1, labels, h/4, w/4)
            upsampled = torch.nn.functional.interpolate(
                seg_logits.float(), size=(mask_h, mask_w), mode="bilinear", align_corners=False
            )
            labels = upsampled.argmax(dim=1)[0].cpu().numpy()
        disc = (labels == self.disc_idx) | (labels == self.cup_idx)  # nested: disc includes cup
        cup = labels == self.cup_idx
        return wire.build_response(p_raw, disc, cup)


# ------------------------------------------------------------------ HTTP app


def create_app(models: Models, token: str) -> FastAPI:
    app = FastAPI(title="consilium-vision (Deva)", docs_url=None, redoc_url=None, openapi_url=None)

    def authorised(request: Request) -> bool:
        scheme, _, given = request.headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(given.strip(), token)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        gpu = torch.cuda.get_device_name(0) if models.device.type == "cuda" else None
        return {
            "status": "ok",
            "revision": wire.REVISION,
            "device": models.device.type,
            "gpu": gpu,
        }

    @app.post("/predict")
    async def predict(request: Request) -> JSONResponse:
        if not authorised(request):
            raise HTTPException(status_code=401, detail="bad or missing bearer token")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=422, detail="empty body")
        if len(body) > wire.MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image too large")
        try:
            with Image.open(io.BytesIO(body)) as img:
                image = img.convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=422, detail="not a decodable image") from None

        import anyio

        started = time.perf_counter()
        payload = await anyio.to_thread.run_sync(models.predict, image)
        log.info(
            "image processed: %dx%d px in %.0f ms (not stored)",
            image.width,
            image.height,
            (time.perf_counter() - started) * 1000,
        )
        return JSONResponse(payload)

    return app


def _banner(public_url: str | None, port: int, token: str, device: torch.device) -> None:
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (no CUDA GPU)"
    url = public_url or f"http://<this-PC-IP>:{port}   (ngrok not running - same Wi-Fi only)"
    line = "=" * 72
    print(
        f"\n{line}\n  CONSILIUM vision server is READY on {gpu}\n{line}\n"
        f"  Send these two lines to Hareesh (he pastes them into his .env):\n\n"
        f"    VISION_ENDPOINT_URL={url}\n"
        f"    VISION_ENDPOINT_TOKEN={token}\n\n"
        f"  Check from any browser: {(public_url or f'http://localhost:{port}')}/healthz\n"
        f"  Keep this window open. Ctrl+C to stop.\n{line}\n",
        flush=True,
    )


def main() -> int:
    _load_dotenv(ENV_FILE)
    token = _ensure_token()
    port = int(os.environ.get("PORT", "8100"))
    tunnel_mode = os.environ.get("TUNNEL", "ngrok").strip().lower()  # ngrok | none

    _ensure_models()
    device = _pick_device(os.environ.get("DEVICE", "auto").strip().lower())
    models = Models(device)
    models.predict(
        Image.fromarray(np.full((512, 512, 3), (150, 60, 30), dtype=np.uint8))
    )  # warm-up
    log.info("Models ready on %s - revision %s", device, wire.REVISION)

    public_url = None
    if tunnel_mode == "ngrok":
        from tunnel import open_ngrok

        public_url = open_ngrok(port)
    # Behind ngrok the server only needs to listen locally; without a tunnel
    # (TUNNEL=none, or ngrok failed) it listens on the LAN instead.
    default_host = "127.0.0.1" if public_url else "0.0.0.0"  # noqa: S104
    host = os.environ.get("HOST", default_host)
    _banner(public_url, port, token, device)
    try:
        uvicorn.run(create_app(models, token), host=host, port=port, log_level="warning")
    finally:
        if public_url:
            from tunnel import close_ngrok

            close_ngrok()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
