"""Configured joint-limit diagnostics with explicit validity and confidence."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from motionlab.math.quaternion import (
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_axis_angle,
)
from motionlab.metrics.base import DiagnosticEvent, MetricResult
from motionlab.motion.clip import MotionClip
from motionlab.motion.markers import JointLimitSpec
from motionlab.processing._signals import true_intervals


def _axis_confidence(clip: MotionClip, joint: int) -> float:
    raw = clip.skeleton.metadata.get("local_axis_confidence")
    if isinstance(raw, dict):
        raw = raw.get(clip.skeleton.joint_names[joint], raw.get(str(joint)))
    if raw is None:
        raw = 0.25 if clip.skeleton.metadata.get("source_format") == "BVH" else 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _required_channels(spec: JointLimitSpec) -> tuple[str, ...]:
    if spec.representation == "euler_xyz":
        return ("x_deg", "y_deg", "z_deg")
    return ("swing_x_deg", "swing_z_deg", "twist_deg")


def _coordinates(
    relative_quaternion: NDArray[np.float64], representation: str
) -> NDArray[np.float64]:
    quaternion = quaternion_normalize(relative_quaternion)
    quaternion = np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    if representation == "euler_xyz":
        xyzw = quaternion[..., (1, 2, 3, 0)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return np.asarray(Rotation.from_quat(xyzw).as_euler("XYZ", degrees=True))

    twist = np.zeros_like(quaternion)
    twist[..., 0] = quaternion[..., 0]
    twist[..., 2] = quaternion[..., 2]
    norm = np.linalg.norm(twist, axis=-1, keepdims=True)
    ambiguous = norm[..., 0] < 1.0e-8
    twist = np.divide(twist, norm, out=np.zeros_like(twist), where=norm >= 1.0e-8)
    twist[ambiguous, 0] = 1.0
    swing = quaternion_multiply(quaternion, quaternion_inverse(twist))
    swing_vector_deg = np.rad2deg(quaternion_to_axis_angle(swing))
    twist_deg = np.rad2deg(2.0 * np.arctan2(twist[..., 2], twist[..., 0]))
    return np.stack((swing_vector_deg[..., 0], swing_vector_deg[..., 2], twist_deg), axis=-1)


def _bounds(spec: JointLimitSpec) -> tuple[tuple[float, float], ...] | None:
    values = tuple(getattr(spec, name) for name in _required_channels(spec))
    if any(value is None for value in values):
        return None
    return tuple(value for value in values if value is not None)


def joint_limit_metric(clip: MotionClip) -> MetricResult:
    """Measure angular degrees outside all valid per-rig configured limits."""
    violation = np.zeros((clip.num_frames, clip.num_joints), dtype=np.float64)
    per_joint: dict[str, dict[str, object]] = {}
    valid_joints: list[int] = []
    joint_confidence: dict[int, float] = {}
    events: list[DiagnosticEvent] = []

    for joint, spec in sorted(clip.skeleton.joint_limits.items()):
        name = clip.skeleton.joint_names[joint]
        bounds = _bounds(spec)
        axis_confidence = _axis_confidence(clip, joint)
        valid = (
            spec.valid and bounds is not None and spec.confidence > 0.0 and axis_confidence > 0.0
        )
        confidence = spec.confidence * axis_confidence if valid else 0.0
        per_joint[name] = {
            "valid": valid,
            "confidence": confidence,
            "limit_confidence": spec.confidence,
            "axis_confidence": axis_confidence,
            "representation": spec.representation,
            "source": spec.source,
            "reason": (None if valid else "disabled_or_incomplete_bounds_or_untrusted_local_axis"),
        }
        if not valid or bounds is None:
            continue
        valid_joints.append(joint)
        joint_confidence[joint] = confidence
        relative = quaternion_multiply(
            quaternion_inverse(clip.skeleton.rest_local_quat_wxyz[joint]),
            clip.local_quat_wxyz[:, joint],
        )
        coordinates = _coordinates(relative, spec.representation)
        channel_violation = np.stack(
            [
                np.maximum(
                    np.maximum(lower - coordinates[:, channel], 0.0),
                    coordinates[:, channel] - upper,
                )
                for channel, (lower, upper) in enumerate(bounds)
            ],
            axis=-1,
        )
        violation[:, joint] = np.max(channel_violation, axis=-1)
        violating = violation[:, joint] > 1.0e-6
        channel_names = _required_channels(spec)
        for start, stop in true_intervals(violating):
            local_slice = channel_violation[start:stop]
            peak_flat = int(np.argmax(local_slice))
            peak_frame_offset, peak_channel = np.unravel_index(peak_flat, local_slice.shape)
            events.append(
                DiagnosticEvent(
                    type="joint_limit_violation",
                    frames=(start, stop),
                    joints=(name,),
                    severity=float(np.max(local_slice)),
                    confidence=confidence,
                    evidence={
                        "representation": spec.representation,
                        "peak_channel": channel_names[peak_channel],
                        "peak_frame": start + int(peak_frame_offset),
                        "degrees_beyond_limit": float(np.max(local_slice)),
                        "integrated_violation_degree_seconds": float(
                            np.sum(violation[start:stop, joint]) / clip.fps
                        ),
                    },
                    suggested_operators=("joint_limit_projection",),
                )
            )

    configured_count = len(clip.skeleton.joint_limits)
    valid_count = len(valid_joints)
    if valid_joints:
        valid_values = violation[:, valid_joints]
        clip_value: float | None = float(np.max(valid_values))
        violating_fraction = float(np.mean(valid_values > 1.0e-6))
        frame_values = np.max(valid_values, axis=1).astype(float).tolist()
        confidence = float(np.mean([joint_confidence[joint] for joint in valid_joints]))
        confidence *= valid_count / max(configured_count, 1)
    else:
        clip_value = None
        violating_fraction = 0.0
        frame_values = None
        confidence = 0.0
    return MetricResult(
        name="joint_limits",
        clip_value=clip_value,
        frame_values=frame_values,
        joint_frame_values=(violation.astype(float).tolist() if valid_joints else None),
        events=tuple(events),
        units="degrees_beyond_limit",
        confidence=confidence,
        metadata={
            "valid": bool(valid_joints),
            "configured_joint_count": configured_count,
            "valid_joint_count": valid_count,
            "invalid_joint_count": configured_count - valid_count,
            "rig_joint_coverage_fraction": valid_count / clip.num_joints,
            "violating_joint_frame_fraction": violating_fraction,
            "integrated_violation_degree_seconds": float(np.sum(valid_values) / clip.fps)
            if valid_joints
            else 0.0,
            "valid_joint_indices": valid_joints,
            "per_joint": per_joint,
            "relative_to": "rest_local_quat_wxyz",
            "swing_twist_axis": "local_y",
            "euler_convention": "intrinsic_XYZ",
        },
    )
