"""Self-test for Deva's vision server (no extra dependencies).

    python check_server.py                           # http://localhost:8100 + token from Deva/.env
    python check_server.py https://<ngrok-url>       # test the public URL exactly as Hareesh will
    python check_server.py <url> path/to/fundus.jpg  # with your own fundus image

Downloads the public example fundus image from the classifier's model card
if no image is given, calls /healthz and /predict, and checks the response is
exactly what the CONSILIUM backend expects.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import wire

EXAMPLE = f"https://huggingface.co/{wire.CLASSIFIER_ID}/resolve/{wire.CLASSIFIER_SHA}/example.jpg"


def _token() -> str:
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("CONSILIUM_VISION_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("CONSILIUM_VISION_TOKEN", "")


def _decode(rle: str, h: int, w: int) -> np.ndarray:
    mask = np.zeros(h * w, dtype=bool)
    pos, value = 0, False
    for count in (int(c) for c in rle.split()):
        if value:
            mask[pos : pos + count] = True
        pos += count
        value = not value
    assert pos in (0, h * w), "RLE does not cover the mask"
    return mask.reshape(h, w)


def main() -> int:
    url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100").rstrip("/")
    image = (
        Path(sys.argv[2]).read_bytes()
        if len(sys.argv) > 2
        else urllib.request.urlopen(EXAMPLE).read()
    )

    # the server URL is the operator's own (http/https) - self-test tool
    skip = {"ngrok-skip-browser-warning": "1"}  # free ngrok: never show the HTML warning page
    health_req = urllib.request.Request(f"{url}/healthz", headers=skip)  # noqa: S310
    health = json.load(urllib.request.urlopen(health_req, timeout=15))  # noqa: S310
    print("health:", health)

    req = urllib.request.Request(  # noqa: S310
        f"{url}/predict",
        data=image,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/octet-stream",
            **skip,
        },
    )
    out = json.load(urllib.request.urlopen(req, timeout=120))  # noqa: S310
    keys = {
        "p_raw",
        "disc_mask_rle",
        "cup_mask_rle",
        "mask_width",
        "mask_height",
        "endpoint_revision",
    }
    assert set(out) == keys, f"unexpected keys {set(out) ^ keys}"
    assert out["endpoint_revision"] == wire.REVISION, "revision differs from wire.REVISION"
    h, w = out["mask_height"], out["mask_width"]
    disc, cup = _decode(out["disc_mask_rle"], h, w), _decode(out["cup_mask_rle"], h, w)
    assert not (cup & ~disc).any(), "cup must lie inside disc (nested convention)"

    rows = lambda m: np.flatnonzero(m.any(axis=1))  # noqa: E731
    vcdr = (len(rows(cup)) / len(rows(disc))) if disc.any() else float("nan")
    print(f"OK  p_raw={out['p_raw']:.3f}  mask={w}x{h}", end="")
    print(f"  disc px={int(disc.sum())}  cup px={int(cup.sum())}")
    print(f"    rough vertical CDR (backend computes the real one) ~ {vcdr:.2f}")
    print("This server is ready for CONSILIUM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
