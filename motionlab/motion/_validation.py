"""Internal helpers for immutable array and JSON boundary validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def frozen_array(value: ArrayLike, *, dtype: np.dtype[Any], name: str) -> NDArray[Any]:
    """Copy an array into contiguous, finite, read-only storage."""
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    array.setflags(write=False)
    return array


def frozen_json_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    """Validate a mapping as JSON data and return a detached read-only top-level copy."""
    copied = dict(value)
    try:
        encoded = json.dumps(copied, sort_keys=True, allow_nan=False)
        detached = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-compatible finite values") from exc
    if not isinstance(detached, dict):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(detached)
