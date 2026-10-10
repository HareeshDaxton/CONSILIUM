"""Cup-to-disc ratio computation (AGENTS.md §8.2 — read before touching).

- ``vertical`` = cup vertical diameter / disc vertical diameter. PRIMARY;
  this is the number agents and reports use.
- ``area_based`` = sqrt(cup_area / disc_area), disc mask INCLUDES cup
  (nested convention). Secondary, for comparison with the paper/MedChat.
- NEVER apply the paper's sqrt(|cup| / (|cup| + |disc|)) — with the nested
  convention that double-counts the cup and biases low.

This module is the CANONICAL implementation for serving; training metrics
(training/segmentation/metrics.py) mirror it and switch to importing it.
"""

from __future__ import annotations

import numpy as np

from consilium.schemas.vision import CDRMetrics, Measurements


def vertical_diameter_px(mask: np.ndarray) -> int:
    """Vertical extent (max row - min row + 1). 0 for an empty mask."""
    rows = np.where(np.asarray(mask, dtype=bool).any(axis=1))[0]
    if rows.size == 0:
        return 0
    return int(rows[-1] - rows[0] + 1)


def compute_measurements(disc: np.ndarray, cup: np.ndarray) -> Measurements:
    disc = np.asarray(disc, dtype=bool)
    cup = np.asarray(cup, dtype=bool)
    disc_area = int(disc.sum())
    if disc_area == 0:
        raise ValueError("empty disc mask — degenerate, route to mask QC / quality gate")
    return Measurements(
        disc_area_px=disc_area,
        cup_area_px=int(cup.sum()),
        disc_vertical_diameter_px=vertical_diameter_px(disc),
        cup_vertical_diameter_px=vertical_diameter_px(cup),
    )


def compute_cdr(measurements: Measurements) -> CDRMetrics:
    return CDRMetrics(
        vertical=min(
            1.0,
            measurements.cup_vertical_diameter_px / measurements.disc_vertical_diameter_px,
        ),
        area_based=float(np.sqrt(measurements.cup_area_px / measurements.disc_area_px)),
    )
