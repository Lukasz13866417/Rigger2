"""Root horizontal speed and target-adherence diagnostics."""

from __future__ import annotations

import numpy as np

from motionlab.math.finite_difference import finite_difference
from motionlab.metrics.base import DiagnosticEvent, MetricResult
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import normalize_ground_plane


def speed_metric(
    clip: MotionClip,
    *,
    ground_plane: np.ndarray | tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
    target_speed_mps: float | None = None,
) -> MetricResult:
    """Measure plane-tangential root speed in metres per second."""
    plane = normalize_ground_plane(ground_plane)
    velocity = finite_difference(clip.root_translation_m, 1.0 / clip.fps)
    tangent = velocity - np.sum(velocity * plane[:3], axis=-1, keepdims=True) * plane[:3]
    speed = np.linalg.norm(tangent, axis=-1)
    average = float(np.mean(speed))
    metadata: dict[str, float | None] = {
        "median_speed_mps": float(np.median(speed)),
        "target_speed_mps": target_speed_mps,
        "absolute_target_error_mps": None,
        "relative_target_error": None,
    }
    events: tuple[DiagnosticEvent, ...] = ()
    if target_speed_mps is not None:
        if not np.isfinite(target_speed_mps) or target_speed_mps < 0.0:
            raise ValueError("target_speed_mps must be finite and nonnegative")
        absolute_error = abs(average - target_speed_mps)
        relative_error = absolute_error / max(target_speed_mps, 1.0e-6)
        metadata["absolute_target_error_mps"] = absolute_error
        metadata["relative_target_error"] = relative_error
        if relative_error > 0.02:
            events = (
                DiagnosticEvent(
                    type="speed_mismatch",
                    frames=(0, clip.num_frames),
                    severity=relative_error,
                    confidence=1.0,
                    evidence={
                        "average_speed_mps": average,
                        "target_speed_mps": target_speed_mps,
                        "relative_error": relative_error,
                    },
                    suggested_operators=("adjust_speed",),
                ),
            )
    return MetricResult(
        name="root_speed",
        clip_value=average,
        frame_values=speed.astype(float).tolist(),
        events=events,
        units="m/s",
        confidence=1.0,
        metadata=metadata,
    )
