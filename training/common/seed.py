"""Seeding (AGENTS.md §21: seed everything in training/eval; record seeds).

Torch is imported lazily so this module stays usable without the training
dependency group installed.
"""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch RNGs. Full bitwise determinism across CUDA is
    NOT promised (cuDNN autotune); the seed is always recorded on the run."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # training group not installed (e.g. api image)
        pass
