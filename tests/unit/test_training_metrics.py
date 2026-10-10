"""Section 2.2 — training metrics (pure numpy, offline)."""

import numpy as np
import pytest
from training.classification.metrics import (
    auroc,
    brier_score,
    expected_calibration_error,
    operating_point,
    reliability_curve,
)
from training.segmentation.metrics import (
    dice_score,
    iou_score,
    vcdr_mae,
    vertical_cdr,
)


class TestAUROC:
    def test_perfect_separation(self) -> None:
        assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_reversed(self) -> None:
        assert auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0

    def test_all_ties_is_chance(self) -> None:
        assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5

    def test_known_value(self) -> None:
        # pos scores 0.8/0.7 vs neg scores 0.1/0.75: pos wins 3 of 4 pairs
        assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.75, 0.8, 0.7])) == 0.75

    def test_single_class_undefined(self) -> None:
        with pytest.raises(ValueError, match="single-class"):
            auroc(np.array([1, 1]), np.array([0.5, 0.6]))


class TestCalibrationMetrics:
    def test_brier_known(self) -> None:
        assert brier_score(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
        assert brier_score(np.array([0, 1]), np.array([0.5, 0.5])) == 0.25

    def test_ece_perfect_calibration(self) -> None:
        y = np.array([0, 1, 0, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        assert expected_calibration_error(y, p, n_bins=2) == 0.0

    def test_ece_worst_case(self) -> None:
        y = np.array([0, 1])
        p = np.array([1.0, 0.0])
        assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(1.0)

    def test_reliability_curve_shape(self) -> None:
        curve = reliability_curve(np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.6, 0.9]), n_bins=2)
        assert curve["counts"] == [2, 2]
        assert len(curve["bin_edges"]) == 3


class TestOperatingPoint:
    def test_requires_exactly_one_target(self) -> None:
        y, p = np.array([0, 1]), np.array([0.2, 0.8])
        with pytest.raises(ValueError, match="exactly one"):
            operating_point(y, p)
        with pytest.raises(ValueError, match="exactly one"):
            operating_point(y, p, target_sensitivity=0.9, target_specificity=0.9)

    def test_target_specificity_met(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.7, 0.6, 0.9])  # no perfect threshold exists
        point = operating_point(y, p, target_specificity=1.0)
        assert point.specificity == 1.0
        assert point.sensitivity == pytest.approx(0.5)

    def test_degenerate_target_reachable_via_inf_threshold(self) -> None:
        """ROC convention: specificity 1.0 is always reachable (predict all
        negative) even with perfectly reversed scores — never crash a run."""
        y = np.array([0, 1])
        p = np.array([0.9, 0.1])
        point = operating_point(y, p, target_specificity=1.0)
        assert point.specificity == 1.0
        assert point.sensitivity == 0.0

    def test_target_sensitivity_met(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.7, 0.6, 0.9])
        point = operating_point(y, p, target_sensitivity=1.0)
        assert point.sensitivity == 1.0
        assert point.specificity == pytest.approx(0.5)  # best spec at full recall


def _concentric(size: int = 100, disc_r: int = 40, cup_r: int = 0) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    disc = dist <= disc_r
    cup = dist <= cup_r if cup_r > 0 else np.zeros_like(disc)
    return disc, cup


class TestSegmentationMetrics:
    def test_dice_identical(self) -> None:
        disc, _ = _concentric()
        assert dice_score(disc, disc) == 1.0

    def test_dice_disjoint(self) -> None:
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        a[0, 0] = True
        b[3, 3] = True
        assert dice_score(a, b) == 0.0

    def test_dice_both_empty_defined(self) -> None:
        empty = np.zeros((4, 4), dtype=bool)
        assert dice_score(empty, empty) == 1.0
        assert iou_score(empty, empty) == 1.0

    def test_iou_half_overlap(self) -> None:
        a = np.zeros((2, 4), dtype=bool)
        b = np.zeros((2, 4), dtype=bool)
        a[0, :3] = True
        b[0, 1:] = True  # intersection 2, union 4
        assert iou_score(a, b) == pytest.approx(0.5)

    def test_vertical_cdr_concentric(self) -> None:
        disc, cup = _concentric(disc_r=40, cup_r=20)
        # extents: disc 81 rows, cup 41 rows
        assert vertical_cdr(disc, cup) == pytest.approx(41 / 81)

    def test_vertical_cdr_cup_zero(self) -> None:
        disc, cup = _concentric(disc_r=40, cup_r=0)
        assert vertical_cdr(disc, cup) == 0.0

    def test_vertical_cdr_empty_disc_raises(self) -> None:
        empty = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="empty disc"):
            vertical_cdr(empty, empty)

    def test_vcdr_mae(self) -> None:
        gt_disc, gt_cup = _concentric(disc_r=40, cup_r=20)
        pred_disc, pred_cup = _concentric(disc_r=40, cup_r=30)
        expected = abs(41 / 81 - 61 / 81)
        assert vcdr_mae(pred_disc, pred_cup, gt_disc, gt_cup) == pytest.approx(expected)
