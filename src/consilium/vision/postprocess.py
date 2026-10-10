"""Mask post-processing (AGENTS.md §8.4): largest-component selection, hole
filling, and cup-clip-to-disc. Runs on RAW endpoint masks before any
measurement — the numbers the pipeline reports describe THESE masks.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]  # scipy ships no stubs


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (8-connectivity)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    return np.asarray(labels == int(sizes.argmax()), dtype=bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return np.asarray(ndimage.binary_fill_holes(np.asarray(mask, dtype=bool)), dtype=bool)


def clip_cup_to_disc(cup: np.ndarray, disc: np.ndarray) -> np.ndarray:
    """Nested convention: cup ⊆ disc. Pixels outside the disc are dropped."""
    result: np.ndarray = np.asarray(cup, dtype=bool) & np.asarray(disc, dtype=bool)
    return result


def postprocess_masks(disc: np.ndarray, cup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full deterministic cleanup: largest component each, fill holes, clip."""
    disc_clean = fill_holes(largest_component(disc))
    cup_clean = clip_cup_to_disc(fill_holes(largest_component(cup)), disc_clean)
    return disc_clean, cup_clean
