"""Batch-safe quaternion operations using active ``[w, x, y, z]`` rotations.

``quaternion_multiply(q_left, q_right)`` composes rotations so that ``q_right``
is applied to a column vector first and ``q_left`` is applied second.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.constants import FLOAT_EPS

FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike, *, trailing: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != trailing:
        raise ValueError(f"{name} must have trailing shape ({trailing},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def quaternion_normalize(quaternion: ArrayLike) -> FloatArray:
    """Normalize one quaternion or a batch of quaternions."""
    q = _as_float_array(quaternion, trailing=4, name="quaternion")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < FLOAT_EPS):
        raise ValueError("cannot normalize a zero-length quaternion")
    return np.asarray(q / norm, dtype=np.float64)


def quaternion_conjugate(quaternion: ArrayLike) -> FloatArray:
    """Return the quaternion conjugate."""
    q = _as_float_array(quaternion, trailing=4, name="quaternion")
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_inverse(quaternion: ArrayLike) -> FloatArray:
    """Return the multiplicative inverse of a nonzero quaternion."""
    q = _as_float_array(quaternion, trailing=4, name="quaternion")
    squared_norm = np.sum(q * q, axis=-1, keepdims=True)
    if np.any(squared_norm < FLOAT_EPS * FLOAT_EPS):
        raise ValueError("cannot invert a zero-length quaternion")
    return np.asarray(quaternion_conjugate(q) / squared_norm, dtype=np.float64)


def quaternion_multiply(q_left: ArrayLike, q_right: ArrayLike) -> FloatArray:
    """Compose active rotations, applying ``q_right`` before ``q_left``."""
    left = _as_float_array(q_left, trailing=4, name="q_left")
    right = _as_float_array(q_right, trailing=4, name="q_right")
    left, right = np.broadcast_arrays(left, right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_rotate_vector(quaternion: ArrayLike, vector: ArrayLike) -> FloatArray:
    """Rotate 3D vectors by active quaternions without constructing matrices."""
    q = quaternion_normalize(quaternion)
    v = _as_float_array(vector, trailing=3, name="vector")
    q_vector = q[..., 1:]
    q_vector, v = np.broadcast_arrays(q_vector, v)
    scalar = np.broadcast_to(q[..., :1], (*q_vector.shape[:-1], 1))
    twice_cross = 2.0 * np.cross(q_vector, v)
    return v + scalar * twice_cross + np.cross(q_vector, twice_cross)


def shortest_arc_quaternion(vector_from: ArrayLike, vector_to: ArrayLike) -> FloatArray:
    """Return the shortest active rotation aligning one nonzero 3D vector with another."""
    source = _as_float_array(vector_from, trailing=3, name="vector_from")
    target = _as_float_array(vector_to, trailing=3, name="vector_to")
    source, target = np.broadcast_arrays(source, target)
    source_norm = np.linalg.norm(source, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    if np.any(source_norm < FLOAT_EPS) or np.any(target_norm < FLOAT_EPS):
        raise ValueError("shortest-arc vectors must be nonzero")
    source = source / source_norm
    target = target / target_norm
    dot = np.clip(np.sum(source * target, axis=-1, keepdims=True), -1.0, 1.0)
    cross = np.cross(source, target)
    quaternion = np.concatenate((1.0 + dot, cross), axis=-1)

    antiparallel = dot[..., 0] < -1.0 + 1.0e-7
    if np.any(antiparallel):
        selected = source[antiparallel]
        basis_index = np.argmin(np.abs(selected), axis=-1)
        basis = np.zeros_like(selected)
        basis[np.arange(selected.shape[0]), basis_index] = 1.0
        axis = np.cross(selected, basis)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        quaternion[antiparallel, 0] = 0.0
        quaternion[antiparallel, 1:] = axis
    return quaternion_normalize(quaternion)


def axis_angle_to_quaternion(axis_angle: ArrayLike) -> FloatArray:
    """Convert tangent/axis-angle vectors in radians to unit quaternions."""
    value = _as_float_array(axis_angle, trailing=3, name="axis_angle")
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1.0e-7
    angle_sq = angle * angle
    scale[small] = 0.5 - angle_sq[small] / 48.0 + angle_sq[small] ** 2 / 3840.0
    np.divide(np.sin(half_angle), angle, out=scale, where=~small)
    return quaternion_normalize(np.concatenate((np.cos(half_angle), value * scale), axis=-1))


def quaternion_to_axis_angle(quaternion: ArrayLike) -> FloatArray:
    """Convert quaternions to shortest-path axis-angle vectors in radians."""
    q = quaternion_normalize(quaternion)
    q = np.where(q[..., :1] < 0.0, -q, q)
    vector = q[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(q[..., :1], 0.0, 1.0))
    scale = np.empty_like(vector_norm)
    small = vector_norm < 1.0e-7
    scale[small] = 2.0
    np.divide(angle, vector_norm, out=scale, where=~small)
    return np.asarray(vector * scale, dtype=np.float64)


def quaternion_log(quaternion: ArrayLike) -> FloatArray:
    """Map unit rotations to 3D tangent vectors using the shortest branch."""
    return quaternion_to_axis_angle(quaternion)


def quaternion_exp(tangent: ArrayLike) -> FloatArray:
    """Map 3D tangent vectors to unit rotations."""
    return axis_angle_to_quaternion(tangent)


def quaternion_difference(q_from: ArrayLike, q_to: ArrayLike) -> FloatArray:
    """Return the local rotation taking ``q_from`` to ``q_to`` by right multiplication."""
    return quaternion_normalize(quaternion_multiply(quaternion_inverse(q_from), q_to))


def quaternion_geodesic_distance(q_first: ArrayLike, q_second: ArrayLike) -> FloatArray:
    """Return shortest angular distance in radians for corresponding rotations."""
    first = quaternion_normalize(q_first)
    second = quaternion_normalize(q_second)
    first, second = np.broadcast_arrays(first, second)
    absolute_dot = np.abs(np.sum(first * second, axis=-1))
    return np.asarray(2.0 * np.arccos(np.clip(absolute_dot, 0.0, 1.0)), dtype=np.float64)


def quaternion_to_matrix(quaternion: ArrayLike) -> FloatArray:
    """Convert unit quaternions to active 3-by-3 rotation matrices."""
    q = quaternion_normalize(quaternion)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        axis=-1,
    ).reshape((*q.shape[:-1], 3, 3))


def matrix_to_quaternion(matrix: ArrayLike) -> FloatArray:
    """Convert active 3-by-3 rotation matrices to canonical unit quaternions."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim < 2 or value.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have trailing shape (3, 3), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("matrix contains NaN or infinite values")

    flat = value.reshape((-1, 3, 3))
    result = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(max(1.0 + trace, FLOAT_EPS))
            result[index] = (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        else:
            diagonal = np.diag(rotation)
            major = int(np.argmax(diagonal))
            if major == 0:
                radicand = 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
                scale = 2.0 * np.sqrt(max(radicand, FLOAT_EPS))
                result[index] = (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            elif major == 1:
                radicand = 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
                scale = 2.0 * np.sqrt(max(radicand, FLOAT_EPS))
                result[index] = (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            else:
                radicand = 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
                scale = 2.0 * np.sqrt(max(radicand, FLOAT_EPS))
                result[index] = (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
    result = quaternion_normalize(result)
    result = np.where(result[:, :1] < 0.0, -result, result)
    return result.reshape((*value.shape[:-2], 4))


def quaternion_slerp(q_start: ArrayLike, q_end: ArrayLike, fraction: ArrayLike) -> FloatArray:
    """Spherically interpolate over the shortest arc with broadcastable fractions."""
    start = quaternion_normalize(q_start)
    end = quaternion_normalize(q_end)
    start, end = np.broadcast_arrays(start, end)
    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0.0, -end, end)
    dot = np.abs(dot)

    target_shape = dot.shape[:-1]
    amount = np.broadcast_to(np.asarray(fraction, dtype=np.float64), target_shape)[..., None]
    close = dot > 0.9995
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    denominator = np.sin(angle)
    safe_denominator = np.where(np.abs(denominator) < FLOAT_EPS, 1.0, denominator)
    spherical = (
        np.sin((1.0 - amount) * angle) / safe_denominator * start
        + np.sin(amount * angle) / safe_denominator * end
    )
    linear = start + amount * (end - start)
    return quaternion_normalize(np.where(close, linear, spherical))


def unwrap_quaternion_signs(quaternion: ArrayLike, *, time_axis: int = -2) -> FloatArray:
    """Flip equivalent quaternion signs so adjacent samples have nonnegative dot products."""
    q = quaternion_normalize(quaternion)
    axis = time_axis if time_axis >= 0 else q.ndim + time_axis
    if axis < 0 or axis >= q.ndim - 1:
        raise ValueError("time_axis must identify a non-quaternion dimension")
    moved = np.moveaxis(q, axis, 0).copy()
    for frame in range(1, moved.shape[0]):
        flip = np.sum(moved[frame - 1] * moved[frame], axis=-1, keepdims=True) < 0.0
        moved[frame] = np.where(flip, -moved[frame], moved[frame])
    return np.moveaxis(moved, 0, axis)


def quaternion_interval_angular_velocity(
    quaternion: ArrayLike,
    timestamps_s: ArrayLike,
    *,
    time_axis: int = -2,
) -> FloatArray:
    """Compute local-frame SO(3) angular velocity between adjacent samples.

    The result has one fewer sample on ``time_axis`` and is expressed in radians per second in
    the earlier sample's local coordinate frame.
    """
    q = unwrap_quaternion_signs(quaternion, time_axis=time_axis)
    axis = time_axis if time_axis >= 0 else q.ndim + time_axis
    moved = np.moveaxis(q, axis, 0)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.shape[0] != moved.shape[0]:
        raise ValueError("timestamps_s must be one-dimensional and match the time dimension")
    dt = np.diff(timestamps)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    relative = quaternion_multiply(quaternion_inverse(moved[:-1]), moved[1:])
    tangent = quaternion_log(relative)
    divisor_shape = (dt.shape[0],) + (1,) * (tangent.ndim - 1)
    velocity = tangent / dt.reshape(divisor_shape)
    return np.moveaxis(velocity, 0, axis)
