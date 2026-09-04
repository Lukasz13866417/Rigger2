"""Explicit source-to-canonical coordinate metadata and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.constants import (
    CANONICAL_FORWARD_AXIS,
    CANONICAL_HANDEDNESS,
    CANONICAL_UP_AXIS,
)
from motionlab.motion._validation import frozen_array


def axis_vector(axis: str) -> NDArray[np.float64]:
    """Convert a signed Cartesian axis token to a unit vector."""
    normalized = axis.strip().upper()
    sign = -1.0 if normalized.startswith("-") else 1.0
    name = normalized.lstrip("+-")
    if name not in {"X", "Y", "Z"}:
        raise ValueError(f"axis must be signed X, Y, or Z; got {axis!r}")
    vector = np.zeros(3, dtype=np.float64)
    vector[{"X": 0, "Y": 1, "Z": 2}[name]] = sign
    return vector


@dataclass(frozen=True)
class CoordinateContract:
    """Coordinate provenance for values already converted to MotionLab canonical space."""

    source_handedness: Literal["right", "left"]
    source_up_axis: str
    source_forward_axis: str
    source_unit_scale_to_m: float
    source_to_canonical_matrix: NDArray[np.float64]
    source: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.source_handedness not in {"right", "left"}:
            raise ValueError("source_handedness must be 'right' or 'left'")
        up = axis_vector(self.source_up_axis)
        forward = axis_vector(self.source_forward_axis)
        if abs(float(np.dot(up, forward))) > 1.0e-12:
            raise ValueError("source up and forward axes must be perpendicular")
        if not np.isfinite(self.source_unit_scale_to_m) or self.source_unit_scale_to_m <= 0.0:
            raise ValueError("source_unit_scale_to_m must be finite and positive")
        matrix = frozen_array(
            self.source_to_canonical_matrix,
            dtype=np.dtype(np.float64),
            name="source_to_canonical_matrix",
        )
        if matrix.shape != (3, 3):
            raise ValueError("source_to_canonical_matrix must have shape [3,3]")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-7, rtol=0.0):
            raise ValueError("source_to_canonical_matrix must be orthonormal")
        determinant = float(np.linalg.det(matrix))
        if not np.isclose(abs(determinant), 1.0, atol=1.0e-7, rtol=0.0):
            raise ValueError("source_to_canonical_matrix must be invertible without scale")
        source_right = np.cross(up, forward)
        if self.source_handedness == "left":
            source_right *= -1.0
        expected = np.stack((source_right, up, forward), axis=0)
        if not np.allclose(matrix, expected, atol=1.0e-7, rtol=0.0):
            raise ValueError("source_to_canonical_matrix is inconsistent with declared axes")
        if not self.source.strip():
            raise ValueError("coordinate source must be nonempty")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("coordinate confidence must be finite and within [0,1]")
        object.__setattr__(self, "source_up_axis", self.source_up_axis.strip().upper())
        object.__setattr__(self, "source_forward_axis", self.source_forward_axis.strip().upper())
        object.__setattr__(self, "source_to_canonical_matrix", matrix)
        object.__setattr__(self, "source_unit_scale_to_m", float(self.source_unit_scale_to_m))
        object.__setattr__(self, "confidence", float(self.confidence))

    @property
    def canonical_handedness(self) -> str:
        return CANONICAL_HANDEDNESS

    @property
    def canonical_up_axis(self) -> str:
        return CANONICAL_UP_AXIS

    @property
    def canonical_forward_axis(self) -> str:
        return CANONICAL_FORWARD_AXIS

    @property
    def canonical_unit(self) -> str:
        return "meter"

    @property
    def canonical_to_source_matrix(self) -> NDArray[np.float64]:
        result = np.array(self.source_to_canonical_matrix.T, copy=True, order="C")
        result.setflags(write=False)
        return result

    def source_to_canonical(self, value: ArrayLike) -> NDArray[np.float64]:
        """Transform vectors with trailing shape three into canonical axes and metres."""
        vector = np.asarray(value, dtype=np.float64)
        if vector.ndim == 0 or vector.shape[-1] != 3 or not np.all(np.isfinite(vector)):
            raise ValueError("coordinate values must be finite with trailing shape [3]")
        return np.asarray(
            vector @ self.source_to_canonical_matrix.T * self.source_unit_scale_to_m,
            dtype=np.float64,
        )

    def canonical_to_source(self, value: ArrayLike) -> NDArray[np.float64]:
        """Invert the source-to-canonical vector conversion."""
        vector = np.asarray(value, dtype=np.float64)
        if vector.ndim == 0 or vector.shape[-1] != 3 or not np.all(np.isfinite(vector)):
            raise ValueError("coordinate values must be finite with trailing shape [3]")
        return np.asarray(
            vector @ self.source_to_canonical_matrix / self.source_unit_scale_to_m,
            dtype=np.float64,
        )
