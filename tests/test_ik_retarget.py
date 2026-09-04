from __future__ import annotations

import numpy as np
import pytest

from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.kinematics.ik_two_bone import solve_two_bone_ik
from motionlab.kinematics.retarget import retarget_rest_relative
from motionlab.math.quaternion import (
    quaternion_geodesic_distance,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_rotate_vector,
    shortest_arc_quaternion,
)
from motionlab.motion.skeleton import Skeleton
from motionlab.testing.synthetic import make_nonidentity_target_humanoid, make_synthetic_walk


def test_shortest_arc_handles_parallel_and_antiparallel_vectors() -> None:
    identity = shortest_arc_quaternion([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    opposite = shortest_arc_quaternion([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    rotated = quaternion_rotate_vector(opposite, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [-1.0, 0.0, 0.0], atol=1e-12)


def test_two_bone_ik_reaches_target_and_preserves_unrelated_joints() -> None:
    clip = make_synthetic_walk(num_frames=2)
    skeleton = clip.skeleton
    local = skeleton.rest_local_quat_wxyz.copy()
    root = np.array([0.0, 0.93, 0.0])
    position, _ = forward_kinematics_numpy(skeleton, local, root)
    target = position[skeleton.roles["left_ankle"]] + np.array([0.1, 0.06, 0.0])
    result = solve_two_bone_ik(
        skeleton,
        local,
        root,
        hip=skeleton.roles["left_hip"],
        knee=skeleton.roles["left_knee"],
        end=skeleton.roles["left_ankle"],
        target_world_m=target,
    )
    assert result.residual_m < 1.0e-5
    assert not result.target_was_clamped
    np.testing.assert_allclose(result.local_quat_wxyz[13], local[13])


def test_two_bone_ik_clamps_unreachable_target() -> None:
    clip = make_synthetic_walk(num_frames=2)
    skeleton = clip.skeleton
    result = solve_two_bone_ik(
        skeleton,
        skeleton.rest_local_quat_wxyz,
        [0.0, 0.93, 0.0],
        hip=skeleton.roles["left_hip"],
        knee=skeleton.roles["left_knee"],
        end=skeleton.roles["left_ankle"],
        target_world_m=[10.0, 10.0, 10.0],
    )
    assert result.target_was_clamped
    assert result.residual_m > 1.0


def test_identity_retarget_preserves_pose_and_root_motion() -> None:
    clip = make_synthetic_walk(num_frames=15)
    retargeted = retarget_rest_relative(clip, clip.skeleton)
    np.testing.assert_allclose(retargeted.local_quat_wxyz, clip.local_quat_wxyz, atol=1e-6)
    np.testing.assert_allclose(retargeted.root_translation_m, clip.root_translation_m, atol=1e-7)
    assert retargeted.metadata["root_scale_ratio"] == pytest.approx(1.0)


def test_retarget_scales_root_trajectory_by_leg_length() -> None:
    clip = make_synthetic_walk(num_frames=5)
    source = clip.skeleton
    target = Skeleton(
        joint_names=source.joint_names,
        parents=source.parents,
        rest_offsets_m=source.rest_offsets_m * 1.5,
        rest_local_quat_wxyz=source.rest_local_quat_wxyz,
        roles=source.roles,
        markers=source.markers,
        metadata={"name": "scaled_target"},
    )
    retargeted = retarget_rest_relative(clip, target)
    np.testing.assert_allclose(
        retargeted.root_translation_m,
        clip.root_translation_m * 1.5,
        atol=1e-6,
    )


def test_nonidentity_retarget_preserves_rest_relative_semantic_motion() -> None:
    source = make_synthetic_walk(num_frames=31)
    target = make_nonidentity_target_humanoid(scale=1.25)
    result = retarget_rest_relative(source, target)
    assert result.skeleton.content_hash == target.content_hash
    assert result.metadata["source_skeleton_id"] != result.metadata["target_skeleton_id"]
    assert result.metadata["root_scale_ratio"] == pytest.approx(1.25)
    for role in ("left_shoulder", "right_shoulder", "left_hip", "right_knee"):
        source_joint = source.skeleton.roles[role]
        target_joint = target.roles[role]
        source_delta = quaternion_multiply(
            quaternion_inverse(source.skeleton.rest_local_quat_wxyz[source_joint]),
            source.local_quat_wxyz[:, source_joint],
        )
        target_delta = quaternion_multiply(
            quaternion_inverse(target.rest_local_quat_wxyz[target_joint]),
            result.local_quat_wxyz[:, target_joint],
        )
        np.testing.assert_allclose(
            quaternion_geodesic_distance(source_delta, target_delta),
            0.0,
            atol=1.0e-6,
        )
