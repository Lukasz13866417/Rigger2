"""Timestamp-aware fixed-rate motion resampling."""

from __future__ import annotations

import numpy as np

from motionlab.core.provenance import append_operation_lineage
from motionlab.math.quaternion import quaternion_slerp, unwrap_quaternion_signs
from motionlab.motion.clip import MotionClip


def resample_motion(clip: MotionClip, target_fps: float) -> MotionClip:
    """Resample root translation linearly and local rotations with shortest-path SLERP."""
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be finite and positive")
    if np.isclose(target_fps, clip.fps, rtol=0.0, atol=1.0e-12):
        return clip.copy()
    if clip.num_frames == 1:
        metadata = {**dict(clip.metadata), "resampled_from_fps": clip.fps}
        metadata = append_operation_lineage(
            clip,
            metadata,
            operation="resample_motion",
            kind="transform",
            parameters={"source_fps": clip.fps, "target_fps": float(target_fps)},
            version="motionlab-0.1.0",
        )
        return MotionClip(
            skeleton=clip.skeleton,
            local_quat_wxyz=clip.local_quat_wxyz,
            root_translation_m=clip.root_translation_m,
            fps=target_fps,
            metadata=metadata,
        )

    old_time = clip.timestamps_s
    # A fixed-rate MotionClip cannot represent an irregular final partial interval. Keep every
    # target-grid sample inside the source duration and drop only the final sub-frame remainder.
    new_count = max(1, int(np.floor(clip.duration_s * target_fps + 1.0e-9)) + 1)
    new_time = np.arange(new_count, dtype=np.float64) / target_fps
    right = np.searchsorted(old_time, new_time, side="right")
    right = np.clip(right, 1, clip.num_frames - 1)
    left = right - 1
    interval = old_time[right] - old_time[left]
    fraction = (new_time - old_time[left]) / interval

    unwrapped = unwrap_quaternion_signs(clip.local_quat_wxyz, time_axis=0)
    rotations = quaternion_slerp(
        unwrapped[left],
        unwrapped[right],
        fraction[:, None],
    ).astype(np.float32)
    translations = np.stack(
        [
            np.interp(new_time, old_time, clip.root_translation_m[:, component])
            for component in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "resampled_from_fps": clip.fps,
            "resampled_to_fps": float(target_fps),
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="resample_motion",
        kind="transform",
        parameters={"source_fps": clip.fps, "target_fps": float(target_fps)},
        version="motionlab-0.1.0",
    )
    return MotionClip(
        skeleton=clip.skeleton,
        local_quat_wxyz=rotations,
        root_translation_m=translations,
        fps=target_fps,
        metadata=metadata,
    )
