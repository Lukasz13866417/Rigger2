"""Explicit world-origin and yaw-facing canonicalization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from motionlab.core.provenance import append_operation_lineage
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_rotate_vector,
)
from motionlab.motion.clip import MotionClip


def canonicalize_origin_and_facing(
    clip: MotionClip,
    *,
    forward_world: ArrayLike | None = None,
) -> MotionClip:
    """Move initial horizontal root position to zero and rotate facing toward canonical +Z.

    If ``forward_world`` is omitted, horizontal root displacement supplies the facing estimate.
    Vertical root height and the full transformed trajectory are preserved.
    """
    if forward_world is None:
        forward = clip.root_translation_m[-1] - clip.root_translation_m[0]
    else:
        forward = np.asarray(forward_world, dtype=np.float64)
        if forward.shape != (3,) or not np.all(np.isfinite(forward)):
            raise ValueError("forward_world must be a finite 3D vector")
    horizontal = np.asarray([forward[0], 0.0, forward[2]], dtype=np.float64)
    magnitude = np.linalg.norm(horizontal)
    if magnitude < 1.0e-8:
        raise ValueError("cannot infer facing from near-zero horizontal direction")
    yaw = float(np.arctan2(horizontal[0], horizontal[2]))
    world_correction = axis_angle_to_quaternion([0.0, -yaw, 0.0])

    origin = np.asarray(
        [clip.root_translation_m[0, 0], 0.0, clip.root_translation_m[0, 2]],
        dtype=np.float64,
    )
    translated = clip.root_translation_m.astype(np.float64) - origin
    root_translation = quaternion_rotate_vector(world_correction, translated).astype(np.float32)
    local = clip.local_quat_wxyz.copy()
    root = clip.skeleton.root_index
    local[:, root] = quaternion_multiply(world_correction, local[:, root]).astype(np.float32)
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "canonical_origin_source_m": origin.tolist(),
            "canonical_facing_yaw_correction_rad": -yaw,
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="canonicalize_origin_and_facing",
        kind="transform",
        parameters={
            "origin_source_m": origin.tolist(),
            "yaw_correction_rad": -yaw,
        },
        version="motionlab-0.1.0",
    )
    return MotionClip(
        skeleton=clip.skeleton,
        local_quat_wxyz=local,
        root_translation_m=root_translation,
        fps=clip.fps,
        metadata=metadata,
    )
