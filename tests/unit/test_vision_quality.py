"""Section 2.3 — quality gates (AGENTS.md §8.4)."""

import io

import numpy as np
import pytest
from PIL import Image

from consilium.core.errors import InputRejected
from consilium.schemas.enums import QualityFlag
from consilium.vision.config import ImagePrecheckConfig, PostInferenceQualityConfig
from consilium.vision.quality import assess_quality, decode_image, mask_qc, precheck_image_bytes

PRECHECK = ImagePrecheckConfig(
    max_bytes=1_000_000, allowed_formats=("PNG", "JPEG"), min_resolution_px=32
)
QUALITY = PostInferenceQualityConfig(
    underexposed_below=0.12,
    overexposed_above=0.90,
    blur_laplacian_var_below=0.0002,
    fundus_red_blue_margin=0.05,
    min_disc_area_fraction=0.005,
    max_disc_area_fraction=0.30,
)


def _png_bytes(array: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt)
    return buf.getvalue()


def _rgb(value: tuple[float, float, float], size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0], img[..., 1], img[..., 2] = value
    return img


def _good_disc(size: int = 64) -> np.ndarray:
    """Disc covering ~12% of the image (within the plausible range)."""
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - size // 2) ** 2 + (xx - size // 2) ** 2) <= int(size * 0.2) ** 2


def _sharp_fundus(size: int = 64) -> np.ndarray:
    """Reddish, textured, mid-brightness — passes all checks."""
    rng = np.random.default_rng(0)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = 0.55 + rng.normal(0, 0.15, (size, size))
    img[..., 1] = 0.30 + rng.normal(0, 0.10, (size, size))
    img[..., 2] = 0.15 + rng.normal(0, 0.05, (size, size))
    return np.clip(img, 0, 1)


class TestPrecheck:
    def test_valid_png_passes(self) -> None:
        data = _png_bytes(np.zeros((64, 64, 3), dtype=np.uint8))
        precheck_image_bytes(data, PRECHECK)  # no raise

    def test_garbage_bytes_rejected(self) -> None:
        with pytest.raises(InputRejected) as exc:
            precheck_image_bytes(b"not an image at all", PRECHECK)
        assert exc.value.reason == "IMAGE_UNDECODABLE"

    def test_oversize_rejected(self) -> None:
        with pytest.raises(InputRejected, match="IMAGE_TOO_LARGE"):
            precheck_image_bytes(b"x" * (PRECHECK.max_bytes + 1), PRECHECK)

    def test_format_allowlist(self) -> None:
        data = _png_bytes(np.zeros((64, 64, 3), dtype=np.uint8), fmt="GIF")
        with pytest.raises(InputRejected) as exc:
            precheck_image_bytes(data, PRECHECK)
        assert exc.value.reason == "UNSUPPORTED_IMAGE_FORMAT"

    def test_low_resolution_rejected(self) -> None:
        data = _png_bytes(np.zeros((16, 16, 3), dtype=np.uint8))
        with pytest.raises(InputRejected) as exc:
            precheck_image_bytes(data, PRECHECK)
        assert exc.value.reason == "LOW_RESOLUTION"


class TestAssessQuality:
    def test_good_fundus_is_gradable(self) -> None:
        result = assess_quality(_sharp_fundus(), _good_disc(), QUALITY)
        assert result.gradable
        assert result.flags == ()
        assert result.score > 0.9

    def test_underexposed(self) -> None:
        result = assess_quality(_rgb((0.02, 0.02, 0.01)), _good_disc(), QUALITY)
        assert QualityFlag.UNDEREXPOSED in result.flags
        assert not result.gradable

    def test_overexposed(self) -> None:
        result = assess_quality(_rgb((0.99, 0.98, 0.97)), _good_disc(), QUALITY)
        assert QualityFlag.OVEREXPOSED in result.flags

    def test_blurred(self) -> None:
        img = _sharp_fundus()
        for _ in range(30):  # heavy box blur kills the Laplacian variance
            img = (
                np.roll(img, 1, 0)
                + np.roll(img, -1, 0)
                + np.roll(img, 1, 1)
                + np.roll(img, -1, 1)
                + img
            ) / 5
        result = assess_quality(img, _good_disc(), QUALITY)
        assert QualityFlag.BLURRED in result.flags

    def test_not_fundus(self) -> None:
        rng = np.random.default_rng(0)
        blue = np.clip(0.1 + rng.normal(0, 0.1, (64, 64, 3)), 0, 1).astype(np.float32)
        blue[..., 2] += 0.5  # blue-dominant: not fundus-like
        result = assess_quality(blue, _good_disc(), QUALITY)
        assert QualityFlag.NOT_FUNDUS in result.flags

    def test_disc_not_visible(self) -> None:
        tiny_disc = np.zeros((64, 64), dtype=bool)
        tiny_disc[32, 32] = True
        result = assess_quality(_sharp_fundus(), tiny_disc, QUALITY)
        assert QualityFlag.DISC_NOT_VISIBLE in result.flags

    def test_decode_round_trip(self) -> None:
        data = _png_bytes(np.full((64, 64, 3), 128, dtype=np.uint8))
        img = decode_image(data)
        assert img.shape == (64, 64, 3)
        assert img.mean() == pytest.approx(128 / 255, abs=0.01)


class TestMaskQC:
    def test_clean_masks_ok(self) -> None:
        disc = _good_disc()
        yy, xx = np.mgrid[0:64, 0:64]
        cup = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 25
        qc = mask_qc(disc, cup, QUALITY)
        assert qc.ok
        assert qc.cup_within_disc and qc.single_component_each and qc.disc_area_plausible

    def test_cup_outside_disc_fails(self) -> None:
        disc = _good_disc()
        cup = np.zeros((64, 64), dtype=bool)
        cup[0:3, 0:3] = True
        qc = mask_qc(disc, cup, QUALITY)
        assert not qc.cup_within_disc
        assert not qc.ok

    def test_two_disc_blobs_fail(self) -> None:
        disc = _good_disc() | _good_disc().take(indices=range(64), axis=1)  # same
        disc[2:6, 2:6] = True  # second detached blob
        qc = mask_qc(disc, np.zeros((64, 64), dtype=bool), QUALITY)
        assert not qc.single_component_each
        assert not qc.ok

    def test_implausible_disc_area_fails(self) -> None:
        disc = np.ones((64, 64), dtype=bool)  # 100% of image — implausible
        qc = mask_qc(disc, np.zeros((64, 64), dtype=bool), QUALITY)
        assert not qc.disc_area_plausible
        assert not qc.ok
