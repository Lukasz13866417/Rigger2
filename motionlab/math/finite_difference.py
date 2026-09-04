"""Batch-safe finite differences with explicit physical sample intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def finite_difference(
    value: ArrayLike,
    dt_s: float,
    *,
    time_axis: int = 0,
) -> NDArray[np.float64]:
    """Differentiate samples using central interiors and second-order endpoint formulas."""
    samples = np.asarray(value, dtype=np.float64)
    if samples.ndim == 0:
        raise ValueError("value must have a time dimension")
    if not np.all(np.isfinite(samples)):
        raise ValueError("value must contain only finite samples")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    axis = time_axis if time_axis >= 0 else samples.ndim + time_axis
    if axis < 0 or axis >= samples.ndim:
        raise ValueError("time_axis is outside the input dimensions")
    moved = np.moveaxis(samples, axis, 0)
    result = np.zeros_like(moved, dtype=np.float64)
    if moved.shape[0] == 1:
        return np.moveaxis(result, 0, axis)
    if moved.shape[0] == 2:
        difference = (moved[1] - moved[0]) / dt_s
        result[0] = difference
        result[1] = difference
        return np.moveaxis(result, 0, axis)
    result[1:-1] = (moved[2:] - moved[:-2]) / (2.0 * dt_s)
    result[0] = (-3.0 * moved[0] + 4.0 * moved[1] - moved[2]) / (2.0 * dt_s)
    result[-1] = (3.0 * moved[-1] - 4.0 * moved[-2] + moved[-3]) / (2.0 * dt_s)
    return np.moveaxis(result, 0, axis)
