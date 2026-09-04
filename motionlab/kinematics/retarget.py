"""Conservative rest-relative rotation transfer for known semantic humanoid rigs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from motionlab.core.provenance import append_operation_lineage
from motionlab.math.quaternion import quaternion_inverse, quaternion_multiply, quaternion_normalize
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton


def _average_leg_length(skeleton: Skeleton) -> float:
    lengths: list[float] = []
    for side in ("left", "right"):
        knee = skeleton.roles.get(f"{side}_knee")
        ankle = skeleton.roles.get(f"{side}_ankle")
        if knee is not None and ankle is not None:
            lengths.append(
                float(np.linalg.norm(skeleton.rest_offsets_m[knee]))
                + float(np.linalg.norm(skeleton.rest_offsets_m[ankle]))
            )
    if not lengths:
        raise ValueError("retarget scale requires mapped hip-knee-ankle leg roles")
    return float(np.mean(lengths))


def retarget_rest_relative(
    source: MotionClip,
    target_skeleton: Skeleton,
    *,
    basis_correction_quat_wxyz: Mapping[str, ArrayLike] | None = None,
    scale_root_translation: bool = True,
) -> MotionClip:
    """Transfer semantic-joint rest-relative deltas and optionally scale root translation."""
    shared_roles = sorted(set(source.skeleton.roles).intersection(target_skeleton.roles))
    if not shared_roles:
        raise ValueError("source and target skeletons have no shared semantic roles")
    correction = {} if basis_correction_quat_wxyz is None else basis_correction_quat_wxyz
    local = np.broadcast_to(
        target_skeleton.rest_local_quat_wxyz,
        (source.num_frames, target_skeleton.num_joints, 4),
    ).copy()
    assigned_targets: set[int] = set()
    for role in shared_roles:
        source_joint = source.skeleton.roles[role]
        target_joint = target_skeleton.roles[role]
        if target_joint in assigned_targets:
            continue
        source_delta = quaternion_multiply(
            quaternion_inverse(source.skeleton.rest_local_quat_wxyz[source_joint]),
            source.local_quat_wxyz[:, source_joint],
        )
        if role in correction:
            basis = quaternion_normalize(correction[role])
            source_delta = quaternion_multiply(
                quaternion_multiply(basis, source_delta),
                quaternion_inverse(basis),
            )
        local[:, target_joint] = quaternion_multiply(
            target_skeleton.rest_local_quat_wxyz[target_joint],
            source_delta,
        ).astype(np.float32)
        assigned_targets.add(target_joint)

    scale = 1.0
    if scale_root_translation:
        scale = _average_leg_length(target_skeleton) / _average_leg_length(source.skeleton)
    root_translation = (source.root_translation_m * scale).astype(np.float32)
    metadata = {
        **dict(source.metadata),
        "parent_motion_id": source.content_hash,
        "operation": "retarget_rest_relative",
        "source_skeleton_id": source.skeleton.content_hash,
        "target_skeleton_id": target_skeleton.content_hash,
        "shared_roles": shared_roles,
        "root_scale_ratio": scale,
        "basis_corrected_roles": sorted(correction),
    }
    metadata = append_operation_lineage(
        source,
        metadata,
        operation="retarget_rest_relative",
        kind="retarget",
        parameters={
            "source_skeleton_id": source.skeleton.content_hash,
            "target_skeleton_id": target_skeleton.content_hash,
            "shared_roles": shared_roles,
            "root_scale_ratio": scale,
            "basis_corrected_roles": sorted(correction),
        },
        version="motionlab-0.1.0",
        retargeter_version="motionlab-0.1.0",
    )
    return MotionClip(
        skeleton=target_skeleton,
        local_quat_wxyz=local,
        root_translation_m=root_translation,
        fps=source.fps,
        metadata=metadata,
    )
