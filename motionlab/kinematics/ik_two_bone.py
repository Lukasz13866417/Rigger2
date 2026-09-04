"""Analytic two-segment position IK with explicit reach clamping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.quaternion import (
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    shortest_arc_quaternion,
)
from motionlab.motion.skeleton import Skeleton


@dataclass(frozen=True)
class TwoBoneIKResult:
    """Solved local rotations and end-effector residual for one pose."""

    local_quat_wxyz: NDArray[np.float32]
    target_world_m: NDArray[np.float64]
    achieved_world_m: NDArray[np.float64]
    residual_m: float
    target_was_clamped: bool


def _world_delta_to_local(
    parent_global: NDArray[np.float64],
    world_delta: NDArray[np.float64],
) -> NDArray[np.float64]:
    return quaternion_multiply(
        quaternion_multiply(quaternion_inverse(parent_global), world_delta),
        parent_global,
    )


def solve_two_bone_ik(
    skeleton: Skeleton,
    local_quat_wxyz: ArrayLike,
    root_translation_m: ArrayLike,
    *,
    hip: int,
    knee: int,
    end: int,
    target_world_m: ArrayLike,
    preferred_bend_direction_world: ArrayLike | None = None,
) -> TwoBoneIKResult:
    """Move a direct hip-knee-end chain toward a world target using two analytic rotations."""
    if int(skeleton.parents[knee]) != hip or int(skeleton.parents[end]) != knee:
        raise ValueError("two-bone IK requires a direct hip -> knee -> end chain")
    local = np.asarray(local_quat_wxyz, dtype=np.float64)
    if local.shape != (skeleton.num_joints, 4):
        raise ValueError(f"local_quat_wxyz must have shape ({skeleton.num_joints}, 4)")
    root_translation = np.asarray(root_translation_m, dtype=np.float64)
    target = np.asarray(target_world_m, dtype=np.float64)
    if root_translation.shape != (3,) or target.shape != (3,):
        raise ValueError("root_translation_m and target_world_m must have shape (3,)")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(root_translation)):
        raise ValueError("IK pose inputs must be finite")
    if not np.all(np.isfinite(target)):
        raise ValueError("IK target must be finite")
    solved = quaternion_normalize(local)
    position, global_rotation = forward_kinematics_numpy(
        skeleton,
        solved,
        root_translation,
    )
    hip_position = position[hip]
    knee_position = position[knee]
    end_position = position[end]
    upper_length = float(np.linalg.norm(knee_position - hip_position))
    lower_length = float(np.linalg.norm(end_position - knee_position))
    if upper_length < 1.0e-8 or lower_length < 1.0e-8:
        raise ValueError("two-bone IK chain contains a zero-length segment")

    target_vector = target - hip_position
    target_distance = float(np.linalg.norm(target_vector))
    if target_distance < 1.0e-8:
        raise ValueError("IK target coincides with hip; bend direction is undefined")
    direction = target_vector / target_distance
    minimum_reach = abs(upper_length - lower_length) + 1.0e-7
    maximum_reach = upper_length + lower_length - 1.0e-7
    clamped_distance = float(np.clip(target_distance, minimum_reach, maximum_reach))
    clamped_target = hip_position + direction * clamped_distance
    target_was_clamped = not np.isclose(clamped_distance, target_distance, atol=1.0e-9)

    current_bend = knee_position - hip_position
    if preferred_bend_direction_world is None:
        bend_reference = current_bend
    else:
        bend_reference = np.asarray(preferred_bend_direction_world, dtype=np.float64)
        if bend_reference.shape != (3,) or not np.all(np.isfinite(bend_reference)):
            raise ValueError("preferred_bend_direction_world must be a finite 3D vector")
    bend_perpendicular = bend_reference - np.dot(bend_reference, direction) * direction
    if np.linalg.norm(bend_perpendicular) < 1.0e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, direction))) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        bend_perpendicular = fallback - np.dot(fallback, direction) * direction
    bend_direction = bend_perpendicular / np.linalg.norm(bend_perpendicular)
    along = (
        upper_length * upper_length
        - lower_length * lower_length
        + clamped_distance * clamped_distance
    ) / (2.0 * clamped_distance)
    perpendicular = np.sqrt(max(upper_length * upper_length - along * along, 0.0))
    desired_knee = hip_position + direction * along + bend_direction * perpendicular

    hip_world_delta = shortest_arc_quaternion(
        knee_position - hip_position,
        desired_knee - hip_position,
    )
    hip_parent = int(skeleton.parents[hip])
    if hip_parent == -1:
        hip_local_delta = hip_world_delta
    else:
        hip_local_delta = _world_delta_to_local(global_rotation[hip_parent], hip_world_delta)
    solved[hip] = quaternion_multiply(hip_local_delta, solved[hip])

    position, global_rotation = forward_kinematics_numpy(
        skeleton,
        solved,
        root_translation,
    )
    knee_world_delta = shortest_arc_quaternion(
        position[end] - position[knee],
        clamped_target - position[knee],
    )
    knee_parent_rotation = global_rotation[hip]
    knee_local_delta = _world_delta_to_local(knee_parent_rotation, knee_world_delta)
    solved[knee] = quaternion_multiply(knee_local_delta, solved[knee])
    solved = quaternion_normalize(solved)

    achieved_position, _ = forward_kinematics_numpy(
        skeleton,
        solved,
        root_translation,
    )
    achieved = achieved_position[end]
    return TwoBoneIKResult(
        local_quat_wxyz=solved.astype(np.float32),
        target_world_m=target.copy(),
        achieved_world_m=achieved.copy(),
        residual_m=float(np.linalg.norm(achieved - target)),
        target_was_clamped=target_was_clamped,
    )
