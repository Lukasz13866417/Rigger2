"""Small deterministic utilities for temporal boolean signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def true_intervals(mask: ArrayLike) -> tuple[tuple[int, int], ...]:
    """Return half-open intervals covering every contiguous true run in a 1D mask."""
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))


def clean_contact_mask(
    mask: ArrayLike,
    *,
    min_true_frames: int,
    max_false_gap_frames: int,
) -> NDArray[np.bool_]:
    """Close short internal false gaps and remove short true islands."""
    if min_true_frames < 1 or max_false_gap_frames < 0:
        raise ValueError("cleanup frame counts must be nonnegative and min_true_frames >= 1")
    result = np.asarray(mask, dtype=np.bool_).copy()
    if result.ndim != 1:
        raise ValueError("mask must be one-dimensional")

    for start, stop in true_intervals(result):
        if stop - start < min_true_frames:
            result[start:stop] = False
    false_intervals = true_intervals(~result)
    for start, stop in false_intervals:
        bounded = start > 0 and stop < result.size
        if bounded and stop - start <= max_false_gap_frames:
            result[start:stop] = True
    return result
