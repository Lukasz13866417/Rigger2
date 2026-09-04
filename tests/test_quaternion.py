from __future__ import annotations

import numpy as np
import pytest

from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_difference,
    quaternion_exp,
    quaternion_geodesic_distance,
    quaternion_interval_angular_velocity,
    quaternion_inverse,
    quaternion_log,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_rotate_vector,
    quaternion_slerp,
    quaternion_to_matrix,
    unwrap_quaternion_signs,
)
from motionlab.math.rotation6d import matrix_to_rotation_6d, rotation_6d_to_matrix


def test_normalize_and_inverse_identity() -> None:
    quaternion = np.array([2.0, -0.5, 1.25, 0.75])
    normalized = quaternion_normalize(quaternion)
    identity = quaternion_multiply(normalized, quaternion_inverse(normalized))
    np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1.0e-12)


def test_zero_quaternion_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        quaternion_normalize([0.0, 0.0, 0.0, 0.0])


def test_composition_order_matches_sequential_vector_rotation() -> None:
    quarter_turn_x = axis_angle_to_quaternion([np.pi / 2.0, 0.0, 0.0])
    quarter_turn_z = axis_angle_to_quaternion([0.0, 0.0, np.pi / 2.0])
    vector = np.array([0.0, 1.0, 0.0])
    composed = quaternion_multiply(quarter_turn_z, quarter_turn_x)
    sequential = quaternion_rotate_vector(
        quarter_turn_z,
        quaternion_rotate_vector(quarter_turn_x, vector),
    )
    np.testing.assert_allclose(quaternion_rotate_vector(composed, vector), sequential, atol=1e-12)


def test_axis_angle_log_exp_and_matrix_random_round_trips() -> None:
    rng = np.random.default_rng(1234)
    tangent = rng.normal(size=(256, 3))
    tangent *= rng.uniform(0.0, np.pi - 1.0e-4, size=(256, 1)) / np.linalg.norm(
        tangent, axis=-1, keepdims=True
    )
    quaternion = axis_angle_to_quaternion(tangent)
    reconstructed = quaternion_exp(quaternion_log(quaternion))
    np.testing.assert_allclose(
        quaternion_geodesic_distance(quaternion, reconstructed),
        0.0,
        atol=5.0e-8,
    )

    matrix = quaternion_to_matrix(quaternion)
    matrix_round_trip = quaternion_to_matrix(matrix_to_quaternion(matrix))
    np.testing.assert_allclose(matrix_round_trip, matrix, atol=1.0e-12)


def test_near_pi_rotation_round_trip() -> None:
    tangent = np.array([0.0, np.pi, 0.0])
    quaternion = axis_angle_to_quaternion(tangent)
    matrix = quaternion_to_matrix(quaternion)
    round_trip = quaternion_to_matrix(matrix_to_quaternion(matrix))
    np.testing.assert_allclose(round_trip, matrix, atol=1e-12)


def test_slerp_endpoints_midpoint_and_sign_equivalence() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    half_turn = axis_angle_to_quaternion([0.0, np.pi, 0.0])
    np.testing.assert_allclose(quaternion_slerp(identity, half_turn, 0.0), identity, atol=1e-12)
    np.testing.assert_allclose(
        quaternion_geodesic_distance(quaternion_slerp(identity, half_turn, 1.0), half_turn),
        0.0,
        atol=1e-12,
    )
    midpoint = quaternion_slerp(identity, -half_turn, 0.5)
    rotated = quaternion_rotate_vector(midpoint, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 0.0, -1.0], atol=1e-12)


def test_sign_unwrap_and_difference_convention() -> None:
    target = axis_angle_to_quaternion([0.1, -0.2, 0.3])
    sequence = np.stack((target, -target, target))
    unwrapped = unwrap_quaternion_signs(sequence)
    assert np.all(np.sum(unwrapped[:-1] * unwrapped[1:], axis=-1) >= 0.0)

    source = axis_angle_to_quaternion([-0.1, 0.0, 0.2])
    delta = quaternion_difference(source, target)
    repaired = quaternion_multiply(source, delta)
    np.testing.assert_allclose(quaternion_geodesic_distance(repaired, target), 0.0, atol=1e-12)


def test_rotation_6d_round_trip() -> None:
    rotations = quaternion_to_matrix(axis_angle_to_quaternion([[0.1, 0.2, 0.3], [-0.7, 0.4, 1.2]]))
    encoded = matrix_to_rotation_6d(rotations)
    assert encoded.shape == (2, 6)
    np.testing.assert_allclose(rotation_6d_to_matrix(encoded), rotations, atol=1e-12)


def test_constant_angular_velocity_uses_physical_timestamps() -> None:
    timestamps = np.linspace(0.0, 1.0, 61)
    angular_speed = 1.75
    tangent = np.zeros((timestamps.size, 3))
    tangent[:, 1] = angular_speed * timestamps
    velocity = quaternion_interval_angular_velocity(
        axis_angle_to_quaternion(tangent),
        timestamps,
    )
    expected = np.zeros_like(velocity)
    expected[:, 1] = angular_speed
    np.testing.assert_allclose(velocity, expected, atol=1e-11)
