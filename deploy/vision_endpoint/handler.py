"""HF Inference Endpoint custom handler — SwinV2 + SegFormer (Section 2.4).

Contract (AGENTS.md §8.0): preprocessing happens HERE (consuming the exported
preprocessing_spec.json so train/serve skew is controlled in one place); the
response carries RAW logits-derived probability + RLE masks + the package
revision ONLY. All interpretation (calibration, tiers, CDR, QC) happens in
src/consilium/vision — never here.

Torch is imported lazily inside the handler class so the preprocessing helpers
stay importable (and parity-testable) without the training group installed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_spec(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "preprocessing_spec.json").read_text(encoding="utf-8"))


def preprocess_image(image_bytes: bytes, spec: dict) -> np.ndarray:
    """Decode + resize + normalize per the shared spec -> CHW float32.

    MUST stay numerically identical to training/common/dataset.py
    (_load_image_tensor) — the parity test enforces it.
    """
    resample = {"bilinear": Image.Resampling.BILINEAR, "nearest": Image.Resampling.NEAREST}[
        spec["resize_filter"]
    ]
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image = image.resize((spec["image_size"], spec["image_size"]), resample)
        array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(spec["mean"], dtype=np.float32)
    std = np.asarray(spec["std"], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)


def encode_rle(mask: np.ndarray) -> str:
    """Same format as consilium.vision.rle (parity-tested). Duplicated because
    the endpoint deploys without the consilium package."""
    flat = np.asarray(mask, dtype=bool).ravel()
    if not flat.any():
        return ""
    counts: list[int] = []
    current_value = False
    run = 0
    for px in flat:
        if bool(px) == current_value:
            run += 1
        else:
            counts.append(run)
            run = 1
            current_value = bool(px)
    counts.append(run)
    return " ".join(str(c) for c in counts)


class EndpointHandler:
    """HF custom handler: one instance per endpoint worker."""

    def __init__(self, model_dir: str) -> None:
        import torch  # endpoint-only dependency
        from transformers import AutoModelForImageClassification, SegformerForSemanticSegmentation

        self._torch = torch
        root = Path(model_dir)
        self._spec = load_spec(root)
        self._revision = json.loads((root / "manifest.json").read_text(encoding="utf-8"))[
            "revision"
        ]
        self._classifier = AutoModelForImageClassification.from_pretrained(root / "classifier")
        self._segmenter = SegformerForSemanticSegmentation.from_pretrained(root / "segmenter")
        self._classifier.eval()
        self._segmenter.eval()

    def __call__(self, data: dict) -> dict:
        return self.predict(data["inputs"])

    def predict(self, image_bytes: bytes) -> dict:
        """Returns the RawVisionOutput wire shape — nothing interpreted."""
        torch = self._torch
        array = preprocess_image(image_bytes, self._spec)
        pixel_values = torch.from_numpy(array).unsqueeze(0)
        with torch.no_grad():
            logits = self._classifier(pixel_values=pixel_values).logits
            p_raw = float(torch.softmax(logits, dim=-1)[0, 1])
            seg_logits = self._segmenter(pixel_values=pixel_values).logits
            upsampled = torch.nn.functional.interpolate(
                seg_logits,
                size=(self._spec["image_size"], self._spec["image_size"]),
                mode="bilinear",
                align_corners=False,
            )
            pred = upsampled.argmax(dim=1)[0].numpy()
        disc = pred >= 1  # nested convention: disc includes cup
        cup = pred == 2
        return {
            "p_raw": p_raw,
            "disc_mask_rle": encode_rle(disc),
            "cup_mask_rle": encode_rle(cup),
            "mask_width": self._spec["image_size"],
            "mask_height": self._spec["image_size"],
            "endpoint_revision": self._revision,
        }
