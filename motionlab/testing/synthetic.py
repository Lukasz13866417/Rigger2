"""Small deterministic humanoid and walk fixtures requiring no external assets."""

from __future__ import annotations

import numpy as np

from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.kinematics.ik_two_bone import solve_two_bone_ik
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
)
from motionlab.motion.clip import MotionClip
from motionlab.motion.markers import MarkerSpec
from motionlab.motion.skeleton import Skeleton


def make_synthetic_humanoid() -> Skeleton:
    """Create a compact humanoid hierarchy with explicit heel and toe markers."""
    names = (
        "Hips",
        "Spine",
        "Chest",
        "Head",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
    )
    parents = np.asarray((-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 2, 10, 11, 2, 13, 14), dtype=np.int32)
    offsets = np.asarray(
        (
            (0.00, 0.00, 0.00),
            (0.00, 0.20, 0.00),
            (0.00, 0.25, 0.00),
            (0.00, 0.30, 0.00),
            (0.10, -0.05, 0.00),
            (0.00, -0.45, 0.00),
            (0.00, -0.43, 0.00),
            (-0.10, -0.05, 0.00),
            (0.00, -0.45, 0.00),
            (0.00, -0.43, 0.00),
            (0.20, 0.15, 0.00),
            (0.30, 0.00, 0.00),
            (0.25, 0.00, 0.00),
            (-0.20, 0.15, 0.00),
            (-0.30, 0.00, 0.00),
            (-0.25, 0.00, 0.00),
        ),
        dtype=np.float32,
    )
    rest = np.zeros((len(names), 4), dtype=np.float32)
    rest[:, 0] = 1.0
    roles = {
        "root": 0,
        "pelvis": 0,
        "spine_1": 1,
        "chest": 2,
        "head": 3,
        "left_hip": 4,
        "left_knee": 5,
        "left_ankle": 6,
        "left_foot": 6,
        "right_hip": 7,
        "right_knee": 8,
        "right_ankle": 9,
        "right_foot": 9,
        "left_shoulder": 10,
        "left_elbow": 11,
        "left_wrist": 12,
        "right_shoulder": 13,
        "right_elbow": 14,
        "right_wrist": 15,
    }
    markers = {
        "left_heel": MarkerSpec(joint="LeftFoot", local_offset_m=(0.0, -0.04, -0.08)),
        "left_toe": MarkerSpec(joint="LeftFoot", local_offset_m=(0.0, -0.04, 0.16)),
        "right_heel": MarkerSpec(joint="RightFoot", local_offset_m=(0.0, -0.04, -0.08)),
        "right_toe": MarkerSpec(joint="RightFoot", local_offset_m=(0.0, -0.04, 0.16)),
    }
    return Skeleton(
        joint_names=names,
        parents=parents,
        rest_offsets_m=offsets,
        rest_local_quat_wxyz=rest,
        roles=roles,
        markers=markers,
        metadata={"fixture": "synthetic_humanoid_v1"},
    )


def make_nonidentity_target_humanoid(*, scale: float = 1.25) -> Skeleton:
    """Create a proportionally different target with non-identity arm rest orientations."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    source = make_synthetic_humanoid()
    rest = source.rest_local_quat_wxyz.copy()
    rest[source.roles["left_shoulder"]] = axis_angle_to_quaternion([0.0, 0.0, 0.25])
    rest[source.roles["right_shoulder"]] = axis_angle_to_quaternion([0.0, 0.0, -0.25])
    markers = {
        name: MarkerSpec(
            joint=marker.joint,
            local_offset_m=(
                scale * marker.local_offset_m[0],
                scale * marker.local_offset_m[1],
                scale * marker.local_offset_m[2],
            ),
        )
        for name, marker in source.markers.items()
    }
    return Skeleton(
        joint_names=source.joint_names,
        parents=source.parents,
        rest_offsets_m=source.rest_offsets_m * scale,
        rest_local_quat_wxyz=rest,
        roles=source.roles,
        markers=markers,
        metadata={
            "name": "synthetic_nonidentity_target",
            "fixture": "synthetic_nonidentity_target_v1",
            "scale_from_source": scale,
        },
    )


def make_synthetic_walk(*, num_frames: int = 121, fps: float = 60.0) -> MotionClip:
    """Create a crude but deterministic periodic walk for integration smoke tests."""
    if num_frames < 2:
        raise ValueError("num_frames must be at least two")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    skeleton = make_synthetic_humanoid()
    phase = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False, dtype=np.float64)
    rotations = np.broadcast_to(
        skeleton.rest_local_quat_wxyz,
        (num_frames, skeleton.num_joints, 4),
    ).copy()

    joint_angles = {
        4: 0.42 * np.sin(phase),
        5: 0.35 * np.maximum(0.0, -np.sin(phase)),
        7: 0.42 * np.sin(phase + np.pi),
        8: 0.35 * np.maximum(0.0, -np.sin(phase + np.pi)),
        10: -0.32 * np.sin(phase),
        13: -0.32 * np.sin(phase + np.pi),
    }
    for joint, angle in joint_angles.items():
        tangent = np.zeros((num_frames, 3), dtype=np.float64)
        tangent[:, 0] = angle
        rotations[:, joint] = axis_angle_to_quaternion(tangent).astype(np.float32)

    time_s = np.arange(num_frames, dtype=np.float64) / fps
    root = np.zeros((num_frames, 3), dtype=np.float32)
    root[:, 1] = (0.93 + 0.015 * np.cos(2.0 * phase)).astype(np.float32)
    root[:, 2] = (1.2 * time_s).astype(np.float32)
    return MotionClip(
        skeleton=skeleton,
        local_quat_wxyz=rotations,
        root_translation_m=root,
        fps=fps,
        metadata={
            "fixture": "synthetic_walk_v1",
            "loop_kind": "root_motion",
            "expected_speed_mps": 1.2,
        },
    )


def _foot_target(root: np.ndarray, time_s: float, *, side: str) -> np.ndarray:
    shifted_time = time_s + (0.5 if side == "right" else 0.0)
    cycle = np.floor(shifted_time)
    cycle_phase = shifted_time - cycle
    if cycle_phase < 0.5:
        relative_z = 0.25 - cycle_phase
        height = 0.04
    else:
        swing = (cycle_phase - 0.5) / 0.5
        smooth = swing * swing * (3.0 - 2.0 * swing)
        relative_z = -0.25 + 0.5 * smooth
        height = 0.04 + 0.12 * np.sin(np.pi * swing)
    return np.array(
        [0.10 if side == "left" else -0.10, height, root[2] + relative_z],
        dtype=np.float64,
    )


def make_synthetic_contact_walk(*, num_cycles: int = 2, fps: float = 60.0) -> MotionClip:
    """Create an IK-authored alternating walk with stationary stance-foot markers."""
    if num_cycles < 1:
        raise ValueError("num_cycles must be positive")
    if not np.isfinite(fps) or fps < 4.0:
        raise ValueError("fps must be finite and at least 4")
    frame_count = round(num_cycles * fps)
    skeleton = make_synthetic_humanoid()
    local = np.broadcast_to(
        skeleton.rest_local_quat_wxyz,
        (frame_count, skeleton.num_joints, 4),
    ).copy()
    time_s = np.arange(frame_count, dtype=np.float64) / fps
    phase = 2.0 * np.pi * time_s
    root = np.zeros((frame_count, 3), dtype=np.float32)
    root[:, 1] = (0.93 + 0.01 * np.cos(2.0 * phase)).astype(np.float32)
    root[:, 2] = time_s.astype(np.float32)
    for joint, sign in ((10, -1.0), (13, 1.0)):
        tangent = np.zeros((frame_count, 3), dtype=np.float64)
        tangent[:, 0] = sign * 0.28 * np.sin(phase)
        local[:, joint] = axis_angle_to_quaternion(tangent).astype(np.float32)

    for frame, timestamp in enumerate(time_s):
        for side in ("left", "right"):
            hip = skeleton.roles[f"{side}_hip"]
            knee = skeleton.roles[f"{side}_knee"]
            end = skeleton.roles[f"{side}_ankle"]
            result = solve_two_bone_ik(
                skeleton,
                local[frame],
                root[frame],
                hip=hip,
                knee=knee,
                end=end,
                target_world_m=_foot_target(root[frame], float(timestamp), side=side),
                preferred_bend_direction_world=(0.0, 0.0, 1.0),
            )
            local[frame] = result.local_quat_wxyz
            _, global_rotation = forward_kinematics_numpy(skeleton, local[frame], root[frame])
            parent = int(skeleton.parents[end])
            local[frame, end] = quaternion_multiply(
                quaternion_inverse(global_rotation[parent]),
                [1.0, 0.0, 0.0, 0.0],
            ).astype(np.float32)
    return MotionClip(
        skeleton=skeleton,
        local_quat_wxyz=local,
        root_translation_m=root,
        fps=fps,
        metadata={
            "fixture": "synthetic_contact_walk_v1",
            "loop_kind": "root_motion",
            "expected_speed_mps": 1.0,
            "ground_plane": [0.0, 1.0, 0.0, 0.0],
            "num_cycles": num_cycles,
        },
    )
