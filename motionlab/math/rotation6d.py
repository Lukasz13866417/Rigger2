"""Continuous 6D rotation representation using the first two matrix columns."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.constants import FLOAT_EPS

FloatArray = NDArray[np.float64]


def matrix_to_rotation_6d(matrix: ArrayLike) -> FloatArray:
    """Convert rotation matrices to concatenated first and second columns."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim < 2 or value.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have trailing shape (3, 3), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("matrix contains NaN or infinite values")
    return np.concatenate((value[..., :, 0], value[..., :, 1]), axis=-1)


def rotation_6d_to_matrix(rotation_6d: ArrayLike) -> FloatArray:
    """Convert 6D vectors to proper rotation matrices with Gram-Schmidt projection."""
    value = np.asarray(rotation_6d, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 6:
        raise ValueError(f"rotation_6d must have trailing shape (6,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("rotation_6d contains NaN or infinite values")

    first = value[..., :3]
    second = value[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < FLOAT_EPS):
        raise ValueError("rotation_6d first column is degenerate")
    basis_x = first / first_norm
    second_orthogonal = second - np.sum(basis_x * second, axis=-1, keepdims=True) * basis_x
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < FLOAT_EPS):
        raise ValueError("rotation_6d columns are linearly dependent")
    basis_y = second_orthogonal / second_norm
    basis_z = np.cross(basis_x, basis_y)
    return np.stack((basis_x, basis_y, basis_z), axis=-1)
