"""Quaternion-geodesic angular jerk and localized pop diagnostics."""

from __future__ import annotations

import numpy as np

from motionlab.math.finite_difference import finite_difference
from motionlab.math.quaternion import quaternion_interval_angular_velocity
from motionlab.metrics.base import DiagnosticEvent, MetricResult
from motionlab.motion.clip import MotionClip
from motionlab.processing._signals import true_intervals


def _time_gradient(value: np.ndarray, dt: float) -> np.ndarray:
    return finite_difference(value, dt, time_axis=0)


def smoothness_metric(clip: MotionClip) -> MetricResult:
    """Measure per-joint angular jerk in rad/s^3 using SO(3) differences."""
    if clip.num_frames < 2:
        angular_velocity = np.zeros((clip.num_frames, clip.num_joints, 3), dtype=np.float64)
    else:
        intervals = quaternion_interval_angular_velocity(
            clip.local_quat_wxyz,
            clip.timestamps_s,
            time_axis=0,
        )
        angular_velocity = np.empty((clip.num_frames, clip.num_joints, 3), dtype=np.float64)
        angular_velocity[:-1] = intervals
        angular_velocity[-1] = intervals[-1]
    acceleration = _time_gradient(angular_velocity, 1.0 / clip.fps)
    jerk = _time_gradient(acceleration, 1.0 / clip.fps)
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)
    clip_value = float(np.percentile(jerk_magnitude, 99.0))

    median = np.median(jerk_magnitude, axis=0)
    mad = np.median(np.abs(jerk_magnitude - median[None, :]), axis=0)
    threshold = median + np.maximum(6.0 * 1.4826 * mad, 1.0e-3)
    events: list[DiagnosticEvent] = []
    for joint in range(clip.num_joints):
        for start, stop in true_intervals(jerk_magnitude[:, joint] > threshold[joint]):
            if stop - start > max(5, clip.num_frames // 8):
                event_type = "joint_jitter"
                operator = "smooth_joint"
            else:
                event_type = "joint_pop"
                operator = "smooth_joint"
            peak = float(np.max(jerk_magnitude[start:stop, joint]))
            events.append(
                DiagnosticEvent(
                    type=event_type,
                    frames=(start, stop),
                    joints=(clip.skeleton.joint_names[joint],),
                    severity=peak,
                    confidence=0.75,
                    evidence={
                        "peak_angular_jerk_rad_s3": peak,
                        "robust_threshold_rad_s3": float(threshold[joint]),
                    },
                    suggested_operators=(operator,),
                )
            )
    return MetricResult(
        name="angular_smoothness",
        clip_value=clip_value,
        frame_values=np.max(jerk_magnitude, axis=1).astype(float).tolist(),
        joint_frame_values=jerk_magnitude.astype(float).tolist(),
        events=tuple(events),
        units="rad/s^3_p99",
        confidence=1.0,
        metadata={"statistic": "99th percentile angular jerk", "derivative": "SO(3) log"},
    )
