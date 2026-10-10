"""Temperature scaling calibration (AGENTS.md §8.3).

Raw softmax scores are not probabilities. The calibrator is BACKEND code with
versioned parameters (configs/models.yaml -> calibration.temperature) operating
on endpoint outputs — it never lives inside the endpoint.

    p_calibrated = sigmoid(logit(p_raw) / T)

Fitting (``fit_temperature``) must run on VALIDATION data only, never test.
Pure numpy — no scipy (its C extensions may be unavailable on locked-down
hosts; a golden-section search needs nothing but arithmetic).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-7


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return float(np.log(p / (1.0 - p)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def calibrate(p_raw: float, temperature: float) -> float:
    """Apply temperature scaling to one raw probability."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return sigmoid(logit(p_raw) / temperature)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood at a given temperature."""
    scaled = logits / temperature
    # stable logistic loss: max(z,0) - z*y + log(1 + exp(-|z|))
    losses = np.maximum(scaled, 0) - scaled * labels + np.log1p(np.exp(-np.abs(scaled)))
    return float(losses.mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 50.0),
    tol: float = 1e-4,
) -> float:
    """Golden-section search for the NLL-minimizing temperature.

    VALIDATION data only — fitting on test leaks the operating point (§8.3).
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if z.shape != y.shape or z.size == 0:
        raise ValueError("logits and labels must be non-empty and equal length")
    lo, hi = bounds
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc, fd = nll(z, y, c), nll(z, y, d)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = nll(z, y, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = nll(z, y, d)
    return float((lo + hi) / 2.0)
