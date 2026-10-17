"""CONSILIUM vision wire format — the contract between this GPU server and the
CONSILIUM backend (`RawVisionOutput` in src/consilium/schemas/vision.py).

Pure numpy, no torch: the CONSILIUM repo's test suite imports THIS file to
check that what Deva's server sends decodes exactly with the backend's decoder.
Do not change anything here without changing the backend at the same time.

Response JSON (and nothing else — no tiers, no CDR, no interpretation):
    {
      "p_raw": 0.83,                 # P(glaucoma) straight from the classifier softmax
      "disc_mask_rle": "...",        # optic disc INCLUDING the cup (nested convention)
      "cup_mask_rle": "...",         # optic cup
      "mask_width": 1024, "mask_height": 991,
      "endpoint_revision": REVISION  # must equal configs/models.yaml pinned_revision
    }
"""

from __future__ import annotations

import numpy as np

# ---- pinned models (exact Hugging Face commits; Apache-2.0, fine-tuned on REFUGE)
CLASSIFIER_ID = "pamixsun/swinv2_tiny_for_glaucoma_classification"
CLASSIFIER_SHA = "a25a03d9ff23d6fbaf6cd7a329373253f4671c63"
SEGMENTER_ID = "pamixsun/segformer_for_optic_disc_cup_segmentation"
SEGMENTER_SHA = "a0463fb6c2bf184cf398e5fd82a8194e01e66e7f"

# The backend refuses any response whose revision differs from its pin
# (fail closed). Changing a model = changing this string AND the backend pin.
REVISION = f"hf:{CLASSIFIER_ID}@{CLASSIFIER_SHA[:12]}+hf:{SEGMENTER_ID}@{SEGMENTER_SHA[:12]}"

MAX_MASK_SIDE = 1024  # masks are returned at the image's aspect ratio, longest side <= this
MAX_IMAGE_BYTES = 20_000_000  # same cap as the backend's upload precheck


def mask_size(width: int, height: int, max_side: int = MAX_MASK_SIDE) -> tuple[int, int]:
    """(width, height) for the returned masks: original aspect ratio, capped."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def encode_rle(mask: np.ndarray) -> str:
    """Run lengths over the row-major flattened mask, alternating zero/one runs,
    STARTING with a zero-run (0 when the mask starts with a 1). All-zero = "".
    Byte-identical to consilium.vision.rle.encode_rle (vectorised)."""
    flat = np.asarray(mask, dtype=bool).ravel()
    if not flat.any():
        return ""
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs.insert(0, 0)
    return " ".join(str(int(r)) for r in runs)


def build_response(p_raw: float, disc: np.ndarray, cup: np.ndarray) -> dict[str, object]:
    """Assemble the wire payload from raw model outputs (no interpretation)."""
    disc = np.asarray(disc, dtype=bool)
    cup = np.asarray(cup, dtype=bool)
    if disc.shape != cup.shape or disc.ndim != 2:
        raise ValueError("disc and cup masks must be 2-D and the same shape")
    p = float(p_raw)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p_raw must be a probability")
    height, width = disc.shape
    return {
        "p_raw": p,
        "disc_mask_rle": encode_rle(disc),
        "cup_mask_rle": encode_rle(cup),
        "mask_width": int(width),
        "mask_height": int(height),
        "endpoint_revision": REVISION,
    }
