"""Review-UI mask overlay rendering (AGENTS.md §17, ARCHITECTURE.md §5.2).

Renders the STORED endpoint masks (``cad_results.raw_vision_json``) over the
case image: disc outline cyan, cup outline magenta. The masks go through the
same in-repo post-processing as the CDR computation, so the overlay shows
exactly the regions the measurements came from. Pure + deterministic: no
network, no model; the image never leaves the process (I-14).

Layers (``OverlayLayer``): ``composite`` (outlines drawn on the image — the
default), ``image`` (the fundus at mask resolution, no outlines), ``disc`` /
``cup`` (transparent RGBA with only that outline opaque). The review UI stacks
the separate layers and toggles/fades them with CSS — all mask geometry stays
server-side, the browser does no mask math.
"""

from __future__ import annotations

import io
from enum import StrEnum

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped]  # scipy ships no stubs

from consilium.schemas.vision import RawVisionOutput
from consilium.vision import postprocess, quality, rle

__all__ = ["CUP_RGB", "DISC_RGB", "OverlayLayer", "render_overlay"]

DISC_RGB = (0, 255, 255)  # cyan
CUP_RGB = (255, 0, 255)  # magenta


class OverlayLayer(StrEnum):
    COMPOSITE = "composite"
    IMAGE = "image"
    DISC = "disc"
    CUP = "cup"


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    if not mask.any():
        return mask
    inner = np.asarray(ndimage.binary_erosion(mask, iterations=width), dtype=bool)
    return np.asarray(mask & ~inner, dtype=bool)


def _png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _outline_layer(outline: np.ndarray, rgb: tuple[int, int, int]) -> bytes:
    rgba = np.zeros((*outline.shape, 4), dtype=np.uint8)
    rgba[outline] = (*rgb, 255)
    return _png(rgba)


def render_overlay(
    image_bytes: bytes,
    raw: RawVisionOutput,
    *,
    layer: OverlayLayer = OverlayLayer.COMPOSITE,
    line_px: int = 2,
) -> bytes:
    """PNG bytes at mask resolution for the requested layer (see module doc)."""
    disc = rle.decode_rle(raw.disc_mask_rle, raw.mask_height, raw.mask_width)
    cup = rle.decode_rle(raw.cup_mask_rle, raw.mask_height, raw.mask_width)
    disc, cup = postprocess.postprocess_masks(disc, cup)
    disc_line = _outline(disc.astype(bool), line_px)
    cup_line = _outline(cup.astype(bool), line_px)
    if layer == OverlayLayer.DISC:
        return _outline_layer(disc_line, DISC_RGB)
    if layer == OverlayLayer.CUP:
        return _outline_layer(cup_line, CUP_RGB)
    image = (quality.decode_image(image_bytes) * 255.0).round().astype(np.uint8)
    base = Image.fromarray(image).resize((raw.mask_width, raw.mask_height))
    canvas = np.array(base, dtype=np.uint8)
    if layer == OverlayLayer.COMPOSITE:
        canvas[disc_line] = DISC_RGB
        canvas[cup_line] = CUP_RGB
    return _png(canvas)
