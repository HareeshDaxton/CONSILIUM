"""Segmentation evaluation metrics (P2, Section 2.2; AGENTS.md §8.3).

Dice / IoU for disc and cup, and vertical-CDR MAE against expert annotation.
Masks are boolean arrays under the nested convention (disc mask INCLUDES cup).

CDR math delegates to the CANONICAL serving implementation in
``src/consilium/vision/cdr.py`` (Section 2.3) — training may import src
(the §5 boundary only forbids the reverse), so the number reported at
training time is exactly the number served.
"""

from __future__ import annotations

import numpy as np

from consilium.vision.cdr import compute_cdr, compute_measurements


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """2|A∩B| / (|A|+|B|). Both-empty is defined as 1.0 (perfect agreement
    on absence), matching common medical-segmentation convention."""
    a = np.asarray(pred, dtype=bool)
    b = np.asarray(target, dtype=bool)
    intersection = float((a & b).sum())
    denominator = float(a.sum() + b.sum())
    if denominator == 0.0:
        return 1.0
    return 2.0 * intersection / denominator


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    a = np.asarray(pred, dtype=bool)
    b = np.asarray(target, dtype=bool)
    union = float((a | b).sum())
    if union == 0.0:
        return 1.0
    return float((a & b).sum()) / union


def vertical_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> float:
    """Cup vertical diameter / disc vertical diameter (AGENTS.md §8.2 PRIMARY).

    Delegates to consilium.vision.cdr — the canonical serving implementation.
    Raises on an empty disc (degenerate mask is a QC failure, not a number).
    """
    measurements = compute_measurements(disc_mask, cup_mask)
    return compute_cdr(measurements).vertical


def vcdr_mae(
    pred_disc: np.ndarray,
    pred_cup: np.ndarray,
    gt_disc: np.ndarray,
    gt_cup: np.ndarray,
) -> float:
    """Mean absolute error of predicted vs annotated vertical CDR for one case."""
    return abs(vertical_cdr(pred_disc, pred_cup) - vertical_cdr(gt_disc, gt_cup))
