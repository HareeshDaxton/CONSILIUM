"""FakeVisionClient — deterministic canned vision outputs for dev/tests (§8.0).

Mirrors the LLMClient fake pattern: no network, no torch, fully deterministic.
Outputs are keyed by sha256(image_bytes) so e2e tests can pin specific
responses to specific fixture images; unknown images get the default fixture.

Failure modes cover the §4/§8.0 failure edges:
  - ``cold_start_calls=N``  — first N calls raise VisionTransient (scale-to-zero)
  - ``always_transient=True`` — never recovers (VISION_UNAVAILABLE path)
  - ``force_revision=...``   — returns outputs with a wrong revision (pin check)
  - ``empty_mask=True``      — all-zero masks (degenerate-mask pipeline path)
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from consilium.schemas.vision import RawVisionOutput
from consilium.vision.client import VisionTransient
from consilium.vision.rle import encode_rle


def _disc_cup_masks(
    height: int = 256, width: int = 256, *, empty: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic concentric disc/cup (vCDR 0.5) — plausible, not random noise."""
    disc = np.zeros((height, width), dtype=bool)
    cup = np.zeros((height, width), dtype=bool)
    if empty:
        return disc, cup
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    r_disc, r_cup = height // 4, height // 8
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_disc**2
    cup = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_cup**2
    return disc, cup


def build_output(
    *,
    p_raw: float,
    revision: str,
    empty_mask: bool = False,
    mask_size: tuple[int, int] = (256, 256),
) -> RawVisionOutput:
    height, width = mask_size
    disc, cup = _disc_cup_masks(height, width, empty=empty_mask)
    return RawVisionOutput(
        p_raw=p_raw,
        disc_mask_rle=encode_rle(disc),
        cup_mask_rle=encode_rle(cup),
        mask_width=width,
        mask_height=height,
        endpoint_revision=revision,
    )


def default_fixtures(revision: str) -> dict[str, RawVisionOutput]:
    """Named canned outputs. Keys are semantic; callers key by sha256(image)."""
    return {
        "normal": build_output(p_raw=0.25, revision=revision),
        "suspect": build_output(p_raw=0.8, revision=revision),
        "empty_mask": build_output(p_raw=0.5, revision=revision, empty_mask=True),
    }


class FakeVisionClient:
    """Deterministic VisionClient. See module docstring for failure modes."""

    def __init__(
        self,
        *,
        revision: str,
        default: RawVisionOutput | None = None,
        keyed: Mapping[str, RawVisionOutput] | None = None,  # sha256(image) -> output
        cold_start_calls: int = 0,
        always_transient: bool = False,
        force_revision: str | None = None,
    ) -> None:
        self._default = default or build_output(p_raw=0.5, revision=revision)
        self._keyed = dict(keyed or {})
        self._cold_start_calls = cold_start_calls
        self._always_transient = always_transient
        self._force_revision = force_revision
        self.calls = 0

    def register(self, image_bytes: bytes, output: RawVisionOutput) -> None:
        self._keyed[hashlib.sha256(image_bytes).hexdigest()] = output

    async def predict(self, image_bytes: bytes) -> RawVisionOutput:
        self.calls += 1
        if self._always_transient or self.calls <= self._cold_start_calls:
            raise VisionTransient("fake cold start")
        key = hashlib.sha256(image_bytes).hexdigest()
        output = self._keyed.get(key, self._default)
        if self._force_revision is not None:
            output = output.model_copy(update={"endpoint_revision": self._force_revision})
        return output
