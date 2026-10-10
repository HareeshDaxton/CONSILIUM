"""Classifier evaluation metrics (P2, Section 2.2; AGENTS.md §8.3).

Pure numpy — no torch, no network — so they are unit-testable offline and
reusable by eval/. Reported per dataset: AUROC, sensitivity @ fixed
specificity and specificity @ fixed sensitivity, Brier score, ECE, and the
reliability curve (logged as an MLflow artifact).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def _as_arrays(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    return y, p


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U formulation; exact with ties. 1.0 perfect, 0.5 chance."""
    y, p = _as_arrays(y_true, y_score)
    pos = p[y == 1]
    neg = p[y == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUROC undefined for a single-class input")
    # count pos > neg plus half ties, vectorized
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y, p = _as_arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y, p = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float


def operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_sensitivity: float | None = None,
    target_specificity: float | None = None,
) -> OperatingPoint:
    """Threshold meeting a stated operating point (§8.3).

    Exactly one target must be given. The chosen threshold is the one that
    meets-or-exceeds the target while maximizing the other metric. Candidate
    thresholds follow ROC convention and include ``+inf`` (predict all
    negative), so any target in [0, 1] is reachable — possibly degenerately
    (sensitivity 0 at specificity 1). Never raises on a reachable target.
    """
    if (target_sensitivity is None) == (target_specificity is None):
        raise ValueError("give exactly one of target_sensitivity / target_specificity")
    y, p = _as_arrays(y_true, y_score)
    thresholds = np.concatenate([np.unique(p), [np.inf]])
    best: OperatingPoint | None = None
    for threshold in thresholds:
        predicted = p >= threshold
        tp = float(((predicted == 1) & (y == 1)).sum())
        fp = float(((predicted == 1) & (y == 0)).sum())
        fn = float(((predicted == 0) & (y == 1)).sum())
        tn = float(((predicted == 0) & (y == 0)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        candidate = OperatingPoint(float(threshold), sensitivity, specificity)
        if (
            target_sensitivity is not None
            and sensitivity >= target_sensitivity
            and (best is None or candidate.specificity > best.specificity)
        ):
            best = candidate
        if (
            target_specificity is not None
            and specificity >= target_specificity
            and (best is None or candidate.sensitivity > best.sensitivity)
        ):
            best = candidate
    if best is None:
        raise ValueError("no threshold meets the requested operating point")
    return best


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> dict[str, list[float | int]]:
    """Reliability diagram data (logged as JSON artifact; plotting is a
    reporting concern, not a training dependency)."""
    y, p = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted: list[float] = []
    fraction_positive: list[float] = []
    counts: list[int] = []
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        counts.append(int(mask.sum()))
        mean_predicted.append(float(p[mask].mean()) if mask.any() else 0.0)
        fraction_positive.append(float(y[mask].mean()) if mask.any() else 0.0)
    return {
        "bin_edges": [float(e) for e in edges],
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "counts": counts,
    }
