"""Deva's GPU vision server speaks exactly the backend's wire contract.

``Deva/wire.py`` (shipped to the GPU laptop, no consilium imports) must:
  - encode masks so that ``consilium.vision.rle.decode_rle`` restores them
    bit-for-bit (and byte-identically to the backend encoder);
  - build payloads that validate as ``RawVisionOutput``;
  - report the SAME revision the backend pins in configs/models.yaml (a
    mismatch would fail every case closed with VISION_REVISION_MISMATCH).
Pure numpy — no torch needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from consilium.schemas.vision import RawVisionOutput
from consilium.vision import rle
from consilium.vision.config import load_models_config

_WIRE = Path(__file__).resolve().parents[2] / "Deva" / "wire.py"


def _wire() -> ModuleType:
    spec = importlib.util.spec_from_file_location("deva_wire", _WIRE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WIRE = _wire()


def test_revision_matches_the_backend_pin() -> None:
    models = load_models_config()
    assert models.endpoint.pinned_revision == WIRE.REVISION
    assert models.classifier.model_id == WIRE.CLASSIFIER_ID
    assert models.classifier.revision == WIRE.CLASSIFIER_SHA
    assert models.segmenter.model_id == WIRE.SEGMENTER_ID
    assert models.segmenter.revision == WIRE.SEGMENTER_SHA


@settings(max_examples=200, deadline=None)
@given(arrays(dtype=bool, shape=st.tuples(st.integers(1, 40), st.integers(1, 40))))
def test_rle_round_trips_through_the_backend_decoder(mask: np.ndarray) -> None:
    encoded = WIRE.encode_rle(mask)
    assert encoded == rle.encode_rle(mask)  # byte-identical to the backend encoder
    decoded = rle.decode_rle(encoded, mask.shape[0], mask.shape[1])
    assert np.array_equal(decoded, mask)


def test_payload_validates_as_raw_vision_output() -> None:
    disc = np.zeros((60, 80), dtype=bool)
    disc[10:50, 20:60] = True
    cup = np.zeros_like(disc)
    cup[20:40, 30:50] = True
    payload = WIRE.build_response(0.73, disc, cup)
    out = RawVisionOutput.model_validate(payload)
    assert (out.mask_width, out.mask_height) == (80, 60)
    assert out.endpoint_revision == load_models_config().endpoint.pinned_revision
    assert np.array_equal(rle.decode_rle(out.cup_mask_rle, 60, 80), cup)


@pytest.mark.parametrize(
    ("size", "expected"),
    [((2124, 2056), (1024, 991)), ((512, 512), (512, 512)), ((800, 1600), (512, 1024))],
)
def test_masks_keep_the_image_aspect_ratio(
    size: tuple[int, int], expected: tuple[int, int]
) -> None:
    assert WIRE.mask_size(*size) == expected


def test_payload_rejects_bad_inputs() -> None:
    m = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        WIRE.build_response(1.5, m, m)
    with pytest.raises(ValueError):
        WIRE.build_response(0.5, m, np.zeros((4, 5), dtype=bool))
