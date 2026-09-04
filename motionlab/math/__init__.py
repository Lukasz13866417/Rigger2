"""Rotation and interpolation mathematics used throughout MotionLab."""

from motionlab.math.finite_difference import finite_difference
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_exp,
    quaternion_geodesic_distance,
    quaternion_inverse,
    quaternion_log,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_rotate_vector,
    quaternion_slerp,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    shortest_arc_quaternion,
    unwrap_quaternion_signs,
)
from motionlab.math.rotation6d import matrix_to_rotation_6d, rotation_6d_to_matrix

__all__ = [
    "axis_angle_to_quaternion",
    "finite_difference",
    "matrix_to_rotation_6d",
    "quaternion_exp",
    "quaternion_geodesic_distance",
    "quaternion_inverse",
    "quaternion_log",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_rotate_vector",
    "quaternion_slerp",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rotation_6d_to_matrix",
    "shortest_arc_quaternion",
    "unwrap_quaternion_signs",
]
