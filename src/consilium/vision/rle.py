"""Run-length encoding for the endpoint wire format (RawVisionOutput).

Format (documented in the endpoint contract, Section 2.4): space-separated
integer run counts over the row-major flattened binary mask, alternating
zero/one runs, STARTING with a zero-run (which may have count 0 when the mask
begins with a 1). Empty string = all-zero mask.

    mask [0 0 1 1 0]  ->  "2 2 1"
    all ones (n px)   ->  "0 n"
    all zeros         ->  "" (empty string)
"""

from __future__ import annotations

import numpy as np


def encode_rle(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=bool).ravel()
    if not flat.any():
        return ""
    counts: list[int] = []
    current_value = False  # runs always start counting zeros
    run = 0
    for px in flat:
        if bool(px) == current_value:
            run += 1
        else:
            counts.append(run)
            run = 1
            current_value = bool(px)
    counts.append(run)
    return " ".join(str(c) for c in counts)


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    total = height * width
    if total <= 0:
        raise ValueError(f"mask dimensions must be positive, got {height}x{width}")
    rle = rle.strip()
    if not rle:
        return np.zeros(total, dtype=bool).reshape(height, width)
    try:
        counts = [int(tok) for tok in rle.split()]
    except ValueError as exc:
        raise ValueError(f"malformed RLE: non-integer token in {rle[:40]!r}") from exc
    if any(c < 0 for c in counts):
        raise ValueError("malformed RLE: negative run count")
    if sum(counts) != total:
        raise ValueError(f"malformed RLE: counts sum to {sum(counts)}, expected {total}")
    mask = np.zeros(total, dtype=bool)
    pos = 0
    value = False  # first run is zeros
    for count in counts:
        if value and count:
            mask[pos : pos + count] = True
        pos += count
        value = not value
    return mask.reshape(height, width)
