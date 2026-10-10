"""Section 2.3 — CDR, post-processing, RLE (AGENTS.md §8.2, §8.4)."""

import numpy as np
import pytest

from consilium.vision.cdr import compute_cdr, compute_measurements, vertical_diameter_px
from consilium.vision.postprocess import (
    clip_cup_to_disc,
    fill_holes,
    largest_component,
    postprocess_masks,
)
from consilium.vision.rle import decode_rle, encode_rle


def rect(shape: tuple[int, int], top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


class TestVerticalCDR:
    @pytest.mark.parametrize(
        ("cup_rows", "expected"),
        [(30, 0.3), (50, 0.5), (70, 0.7), (90, 0.9)],
    )
    def test_concentric_known_ratios(self, cup_rows: int, expected: float) -> None:
        disc = rect((200, 200), 50, 50, 100, 100)
        cup = rect((200, 200), 50 + (100 - cup_rows) // 2, 75, cup_rows, 50)
        cdr = compute_cdr(compute_measurements(disc, cup))
        assert cdr.vertical == pytest.approx(expected)
        assert cdr.mask_convention == "disc_includes_cup"

    def test_cup_zero(self) -> None:
        disc = rect((200, 200), 50, 50, 100, 100)
        cup = np.zeros((200, 200), dtype=bool)
        assert compute_cdr(compute_measurements(disc, cup)).vertical == 0.0

    def test_cup_equals_disc(self) -> None:
        disc = rect((200, 200), 50, 50, 100, 100)
        assert compute_cdr(compute_measurements(disc, disc.copy())).vertical == 1.0

    def test_tilted_mask_uses_vertical_extent(self) -> None:
        """Diagonal cup: vertical extent 50, horizontal extent 99 — the
        vertical CDR must use 50 (height ratio), not 99 (width)."""
        disc = rect((200, 200), 0, 0, 100, 200)
        cup = np.zeros((200, 200), dtype=bool)
        for i in range(50):
            cup[i, 2 * i] = True
        assert vertical_diameter_px(cup) == 50
        assert compute_cdr(compute_measurements(disc, cup)).vertical == pytest.approx(0.5)

    def test_area_based_nested_convention(self) -> None:
        """disc area 10000, cup 2500 → sqrt(0.25) = 0.5. The paper's rim-only
        formula would give sqrt(2500/12500) ≈ 0.44 — we deliberately differ."""
        disc = rect((200, 200), 0, 0, 100, 100)
        cup = rect((200, 200), 25, 25, 50, 50)
        assert compute_cdr(compute_measurements(disc, cup)).area_based == pytest.approx(0.5)

    def test_empty_disc_raises(self) -> None:
        empty = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="empty disc"):
            compute_measurements(empty, empty)


class TestPostprocess:
    def test_largest_component_keeps_biggest_blob(self) -> None:
        disc = rect((100, 100), 0, 0, 60, 60) | rect((100, 100), 70, 70, 20, 20)
        cleaned = largest_component(disc)
        assert cleaned.sum() == 3600
        assert not cleaned[80, 80]

    def test_fill_holes(self) -> None:
        ring = rect((100, 100), 10, 10, 60, 60) & ~rect((100, 100), 30, 30, 20, 20)
        filled = fill_holes(ring)
        assert filled[40, 40]

    def test_cup_outside_disc_is_clipped(self) -> None:
        disc = rect((100, 100), 0, 0, 50, 100)
        cup = rect((100, 100), 25, 0, 50, 100)  # half inside, half below disc
        clipped = clip_cup_to_disc(cup, disc)
        assert clipped.sum() == 25 * 100
        assert not (clipped & ~disc).any()

    def test_two_blobs_and_cup_overflow_full_pipeline(self) -> None:
        disc = rect((200, 200), 10, 10, 100, 100) | rect((200, 200), 150, 150, 40, 40)
        cup = rect((200, 200), 30, 30, 50, 50) | rect((200, 200), 120, 120, 20, 20)
        disc_clean, cup_clean = postprocess_masks(disc, cup)
        assert cup_clean.sum() == 50 * 50  # stray blob dropped, in-disc cup kept
        assert disc_clean.sum() == 100 * 100  # small blob dropped
        cdr = compute_cdr(compute_measurements(disc_clean, cup_clean))
        assert cdr.vertical == pytest.approx(0.5)


class TestRLE:
    def test_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        mask = rng.random((37, 53)) > 0.6
        assert np.array_equal(decode_rle(encode_rle(mask), 37, 53), mask)

    def test_all_zeros_is_empty_string(self) -> None:
        mask = np.zeros((4, 5), dtype=bool)
        assert encode_rle(mask) == ""
        assert not decode_rle("", 4, 5).any()

    def test_all_ones(self) -> None:
        mask = np.ones((4, 5), dtype=bool)
        assert encode_rle(mask) == "0 20"
        assert decode_rle("0 20", 4, 5).all()

    @pytest.mark.parametrize("bad", ["1 2 x", "1 -2 17", "1 2 3", "0 19"])
    def test_malformed_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="malformed RLE"):
            decode_rle(bad, 4, 5)

    def test_zero_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            decode_rle("", 0, 5)
