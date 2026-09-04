"""Root-motion-aware cyclic endpoint compatibility metric."""

from __future__ import annotations

import numpy as np

from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.quaternion import (
    quaternion_geodesic_distance,
    quaternion_interval_angular_velocity,
    quaternion_inverse,
    quaternion_multiply,
)
from motionlab.metrics.base import DiagnosticEvent, MetricResult
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult


def loop_seam_metric(clip: MotionClip, contacts: ContactResult | None = None) -> MetricResult:
    """Compare endpoint poses and derivatives after ignoring permitted root displacement."""
    position, _ = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    root = clip.skeleton.root_index
    relative = position - position[:, root : root + 1]
    if clip.num_frames >= 2:
        last_step = quaternion_multiply(
            quaternion_inverse(clip.local_quat_wxyz[-2]),
            clip.local_quat_wxyz[-1],
        )
        predicted_next_rotation = quaternion_multiply(clip.local_quat_wxyz[-1], last_step)
        predicted_next_position = relative[-1] + (relative[-1] - relative[-2])
    else:
        predicted_next_rotation = clip.local_quat_wxyz[-1]
        predicted_next_position = relative[-1]
    pose_by_joint = quaternion_geodesic_distance(clip.local_quat_wxyz[0], predicted_next_rotation)
    pose_error = float(np.mean(pose_by_joint))
    position_error = float(np.mean(np.linalg.norm(relative[0] - predicted_next_position, axis=-1)))

    if clip.num_frames >= 3:
        root_velocity = np.diff(clip.root_translation_m.astype(np.float64), axis=0) * clip.fps
        root_velocity_error = float(np.linalg.norm(root_velocity[0] - root_velocity[-1]))
        angular_velocity = quaternion_interval_angular_velocity(
            clip.local_quat_wxyz,
            clip.timestamps_s,
            time_axis=0,
        )
        angular_velocity_error = float(
            np.mean(np.linalg.norm(angular_velocity[0] - angular_velocity[-1], axis=-1))
        )
    else:
        root_velocity_error = 0.0
        angular_velocity_error = 0.0
    contact_mismatch = (
        0.0 if contacts is None else float(np.mean(contacts.hard[0] != contacts.hard[-1]))
    )
    weighted = (
        pose_error
        + 4.0 * position_error
        + 0.25 * root_velocity_error
        + 0.1 * angular_velocity_error
        + contact_mismatch
    )
    if weighted < 1.0e-6:
        weighted = 0.0
    events: tuple[DiagnosticEvent, ...] = ()
    if weighted > 1.0e-4:
        worst_joint = int(np.argmax(pose_by_joint))
        events = (
            DiagnosticEvent(
                type="loop_seam",
                frames=(max(0, clip.num_frames - 3), clip.num_frames),
                joints=(clip.skeleton.joint_names[worst_joint],),
                severity=weighted,
                confidence=1.0 if contacts is not None else 0.85,
                evidence={
                    "pose_error_rad": pose_error,
                    "position_error_m": position_error,
                    "root_velocity_error_mps": root_velocity_error,
                    "angular_velocity_error_rad_s": angular_velocity_error,
                    "contact_mismatch_fraction": contact_mismatch,
                },
                suggested_operators=("close_loop",),
            ),
        )
    return MetricResult(
        name="loop_seam",
        clip_value=weighted,
        frame_values=None,
        joint_frame_values=None,
        events=events,
        units="weighted_seam",
        confidence=1.0 if contacts is not None else 0.85,
        metadata={
            "pose_error_rad": pose_error,
            "position_error_m": position_error,
            "root_velocity_error_mps": root_velocity_error,
            "angular_velocity_error_rad_s": angular_velocity_error,
            "contact_mismatch_fraction": contact_mismatch,
            "root_translation_delta_m": (clip.root_translation_m[-1] - clip.root_translation_m[0])
            .astype(float)
            .tolist(),
        },
    )
