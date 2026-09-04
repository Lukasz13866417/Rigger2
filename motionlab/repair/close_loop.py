"""Tangent-space loop seam closure with explicit before/after diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motionlab.core.provenance import append_operation_lineage
from motionlab.math.quaternion import (
    quaternion_exp,
    quaternion_inverse,
    quaternion_log,
    quaternion_multiply,
    quaternion_normalize,
)
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult


@dataclass(frozen=True)
class LoopClosureResult:
    """Closed loop and weighted seam improvement."""

    repaired: MotionClip
    before_seam: float
    after_seam: float
    reduction_fraction: float
    seam_window_frames: int


def close_loop(
    clip: MotionClip,
    *,
    contacts: ContactResult | None = None,
    seam_window_frames: int = 12,
    strength: float = 1.0,
) -> LoopClosureResult:
    """Reduce a cyclic pose/velocity seam without appending a duplicate first frame."""
    if clip.num_frames < 4:
        raise ValueError("loop closure requires at least four frames")
    if seam_window_frames < 2:
        raise ValueError("seam_window_frames must be at least two")
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and within [0,1]")
    window = min(seam_window_frames, clip.num_frames - 1)
    before = loop_seam_metric(clip, contacts)
    before_value = before.clip_value or 0.0
    local = clip.local_quat_wxyz.astype(np.float64).copy()
    first_step = quaternion_multiply(quaternion_inverse(local[0]), local[1])
    first_step_tangent = quaternion_log(first_step)
    if window == 2:
        ramp = np.ones(2, dtype=np.float64)
    else:
        ramp = np.concatenate((np.linspace(0.0, 1.0, window - 1, dtype=np.float64), [1.0]))
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    window_start = clip.num_frames - window
    for offset, weight in enumerate(ramp, start=window_start):
        distance_to_next = clip.num_frames - offset
        ideal = quaternion_multiply(
            local[0],
            quaternion_exp(-distance_to_next * first_step_tangent),
        )
        correction = quaternion_log(quaternion_multiply(quaternion_inverse(local[offset]), ideal))
        local[offset] = quaternion_multiply(
            local[offset],
            quaternion_exp(correction * (weight * strength)),
        )

    root = clip.root_translation_m.astype(np.float64).copy()
    first_root_step = root[1] - root[0]
    last_root_anchor = root[-1].copy()
    for offset, weight in enumerate(ramp, start=window_start):
        steps_before_last = clip.num_frames - 1 - offset
        ideal_root = last_root_anchor - steps_before_last * first_root_step
        root[offset] += (ideal_root - root[offset]) * (weight * strength)

    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "close_loop",
            "operator_parameters": {
                "seam_window_frames": seam_window_frames,
                "strength": strength,
                "no_duplicate_endpoint": True,
            },
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="close_loop",
        kind="repair",
        parameters={
            "seam_window_frames": seam_window_frames,
            "strength": strength,
            "no_duplicate_endpoint": True,
        },
        version="motionlab-0.1.0",
    )
    repaired = clip.with_updates(
        local_quat_wxyz=quaternion_normalize(local).astype(np.float32),
        root_translation_m=root.astype(np.float32),
        metadata=metadata,
    )
    after = loop_seam_metric(repaired, contacts)
    after_value = after.clip_value or 0.0
    reduction = 0.0 if before_value <= 1.0e-12 else 1.0 - after_value / before_value
    return LoopClosureResult(
        repaired=repaired,
        before_seam=before_value,
        after_seam=after_value,
        reduction_fraction=reduction,
        seam_window_frames=window,
    )
