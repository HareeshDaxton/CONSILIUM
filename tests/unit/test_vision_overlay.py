"""Section 12.3 — overlay layers: the composite is unchanged, and the separate
image / disc / cup layers stack back to exactly the composite (so the UI can
toggle and fade them with CSS alone — no mask math in the browser)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from consilium.vision import pipeline
from consilium.vision.client_fake import build_output
from consilium.vision.overlay import CUP_RGB, DISC_RGB, OverlayLayer, render_overlay

IMAGE = pipeline.synthetic_fundus_bytes(256)
RAW = build_output(p_raw=0.5, revision="sha256:test")


def _px(png: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(png)))


def test_default_layer_is_the_composite() -> None:
    assert render_overlay(IMAGE, RAW) == render_overlay(IMAGE, RAW, layer=OverlayLayer.COMPOSITE)


@pytest.mark.parametrize("layer", list(OverlayLayer))
def test_every_layer_is_at_mask_resolution(layer: OverlayLayer) -> None:
    arr = _px(render_overlay(IMAGE, RAW, layer=layer))
    assert arr.shape[:2] == (RAW.mask_height, RAW.mask_width)


@pytest.mark.parametrize(
    ("layer", "rgb"), [(OverlayLayer.DISC, DISC_RGB), (OverlayLayer.CUP, CUP_RGB)]
)
def test_outline_layers_are_transparent_except_their_outline(
    layer: OverlayLayer, rgb: tuple[int, int, int]
) -> None:
    arr = _px(render_overlay(IMAGE, RAW, layer=layer))
    assert arr.shape[2] == 4
    opaque = arr[..., 3] == 255
    assert opaque.any()
    assert set(np.unique(arr[..., 3])) <= {0, 255}
    assert (arr[opaque][:, :3] == rgb).all()


def test_layers_stack_to_the_composite() -> None:
    image = _px(render_overlay(IMAGE, RAW, layer=OverlayLayer.IMAGE))
    disc = _px(render_overlay(IMAGE, RAW, layer=OverlayLayer.DISC))
    cup = _px(render_overlay(IMAGE, RAW, layer=OverlayLayer.CUP))
    stacked = image.copy()
    stacked[disc[..., 3] == 255] = DISC_RGB
    stacked[cup[..., 3] == 255] = CUP_RGB  # cup drawn last, as in the composite
    composite = _px(render_overlay(IMAGE, RAW))
    assert np.array_equal(stacked, composite)
    assert not np.array_equal(image, composite)  # the image layer carries no outlines
