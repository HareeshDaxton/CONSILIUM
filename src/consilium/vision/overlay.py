"""Review-UI mask overlay rendering (AGENTS.md §17, ARCHITECTURE.md §5.2).

Renders the STORED endpoint masks (``cad_results.raw_vision_json``) over the
case image: disc outline cyan, cup outline magenta. The masks go through the
same in-repo post-processing as the CDR computation, so the overlay shows
exactly the regions the measurements came from. Pure + deterministic: no
network, no model; the image never leaves the process (I-14).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped]  # scipy ships no stubs

from consilium.schemas.vision import RawVisionOutput
from consilium.vision import postprocess, quality, rle

__all__ = ["CUP_RGB", "DISC_RGB", "render_overlay"]

DISC_RGB = (0, 255, 255)  # cyan
CUP_RGB = (255, 0, 255)  # magenta


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    if not mask.any():
        return mask
    inner = np.asarray(ndimage.binary_erosion(mask, iterations=width), dtype=bool)
    return np.asarray(mask & ~inner, dtype=bool)


def render_overlay(image_bytes: bytes, raw: RawVisionOutput, *, line_px: int = 2) -> bytes:
    """PNG bytes at mask resolution with disc/cup outlines drawn on the image."""
    disc = rle.decode_rle(raw.disc_mask_rle, raw.mask_height, raw.mask_width)
    cup = rle.decode_rle(raw.cup_mask_rle, raw.mask_height, raw.mask_width)
    disc, cup = postprocess.postprocess_masks(disc, cup)
    image = (quality.decode_image(image_bytes) * 255.0).round().astype(np.uint8)
    base = Image.fromarray(image).resize((raw.mask_width, raw.mask_height))
    canvas = np.array(base, dtype=np.uint8)
    canvas[_outline(disc.astype(bool), line_px)] = DISC_RGB
    canvas[_outline(cup.astype(bool), line_px)] = CUP_RGB
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    return buf.getvalue()
