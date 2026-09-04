"""Contact-window foot locking using two-bone IK and optional orientation preservation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.core.provenance import append_operation_lineage
from motionlab.kinematics.fk import forward_kinematics_numpy, marker_world_positions
from motionlab.kinematics.ik_two_bone import solve_two_bone_ik
from motionlab.math.finite_difference import finite_difference
from motionlab.math.quaternion import quaternion_inverse, quaternion_multiply, quaternion_normalize
from motionlab.metrics.foot_sliding import foot_sliding_metric
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult, normalize_ground_plane


@dataclass(frozen=True)
class FootLockResult:
    """Repaired clip and transparent before/after operator diagnostics."""

    repaired: MotionClip
    side: str
    interval: tuple[int, int]
    anchor_world_m: tuple[float, float, float]
    before_slip_cm: float
    after_slip_cm: float
    maximum_ik_target_residual_m: float
    clamped_frames: tuple[int, ...]

    @property
    def maximum_ik_residual_m(self) -> float:
        """Backward-compatible alias for the explicit target residual."""
        return self.maximum_ik_target_residual_m

    @property
    def actual_post_repair_marker_slip_cm(self) -> float:
        """Return measured sole-marker slip after applying the operator."""
        return self.after_slip_cm


def _smoothstep(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(value * value * (3.0 - 2.0 * value), dtype=np.float64)


def _lock_weights(length: int, blend_in_frames: int, blend_out_frames: int) -> np.ndarray:
    weight = np.ones(length, dtype=np.float64)
    if blend_in_frames > 0:
        count = min(blend_in_frames + 1, length)
        weight[:count] *= _smoothstep(np.linspace(0.0, 1.0, count))
    if blend_out_frames > 0:
        count = min(blend_out_frames + 1, length)
        weight[-count:] *= _smoothstep(np.linspace(1.0, 0.0, count))
    return weight


def _interval_contacts(
    clip: MotionClip,
    *,
    side: str,
    start: int,
    stop: int,
    ground_plane: ArrayLike,
) -> ContactResult:
    plane = normalize_ground_plane(ground_plane)
    marker_names = ("left_heel", "left_toe", "right_heel", "right_toe")
    markers = marker_world_positions(clip, marker_names)
    position = np.stack([markers[name] for name in marker_names], axis=1)
    height = position @ plane[:3] + plane[3]
    velocity = finite_difference(position, 1.0 / clip.fps)
    normal_velocity = np.sum(velocity * plane[:3], axis=-1, keepdims=True) * plane[:3]
    speed = np.linalg.norm(velocity - normal_velocity, axis=-1)
    hard = np.zeros((clip.num_frames, 4), dtype=np.bool_)
    selected = (0, 1) if side == "left" else (2, 3)
    hard[start:stop, selected] = True
    confidence = hard.astype(np.float32)
    return ContactResult(
        marker_names=marker_names,
        hard=hard,
        confidence=confidence,
        height_m=height.astype(np.float32),
        tangential_speed_mps=speed.astype(np.float32),
        ground_plane=plane.astype(np.float32),
        source="operator_interval",
        ground_confidence=1.0,
    )


def lock_foot(
    clip: MotionClip,
    *,
    side: Literal["left", "right"],
    start_frame: int,
    stop_frame: int,
    anchor_mode: Literal["first", "median"] = "first",
    lock_strength: float = 1.0,
    blend_in_frames: int = 3,
    blend_out_frames: int = 3,
    preserve_foot_orientation: bool = True,
    pelvis_compensation_limit_m: float = 0.03,
    anchor_offset_m: ArrayLike = (0.0, 0.0, 0.0),
    anchor_offset_end_m: ArrayLike | None = None,
    ground_plane: ArrayLike = (0.0, 1.0, 0.0, 0.0),
) -> FootLockResult:
    """Lock one ankle trajectory over a half-open contact interval using two-bone IK."""
    if start_frame < 0 or stop_frame > clip.num_frames or start_frame >= stop_frame:
        raise ValueError("foot-lock interval must be a valid nonempty frame range")
    if not np.isfinite(lock_strength) or not 0.0 <= lock_strength <= 1.0:
        raise ValueError("lock_strength must be finite and within [0,1]")
    if blend_in_frames < 0 or blend_out_frames < 0:
        raise ValueError("blend frame counts must be nonnegative")
    if not np.isfinite(pelvis_compensation_limit_m) or pelvis_compensation_limit_m < 0.0:
        raise ValueError("pelvis_compensation_limit_m must be finite and nonnegative")
    anchor_offset = np.asarray(anchor_offset_m, dtype=np.float64)
    if anchor_offset.shape != (3,) or not np.all(np.isfinite(anchor_offset)):
        raise ValueError("anchor_offset_m must be a finite 3D vector")
    anchor_offset_end = (
        anchor_offset
        if anchor_offset_end_m is None
        else np.asarray(anchor_offset_end_m, dtype=np.float64)
    )
    if anchor_offset_end.shape != (3,) or not np.all(np.isfinite(anchor_offset_end)):
        raise ValueError("anchor_offset_end_m must be a finite 3D vector")
    required_roles = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
    missing_roles = [role for role in required_roles if role not in clip.skeleton.roles]
    if missing_roles:
        raise ValueError(f"foot lock requires semantic roles: {missing_roles}")
    hip, knee, end = (clip.skeleton.roles[role] for role in required_roles)
    position, global_rotation = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    end_trajectory = position[:, end]
    if anchor_mode == "first":
        anchor = end_trajectory[start_frame].copy()
    else:
        anchor = np.median(end_trajectory[start_frame:stop_frame], axis=0)
    anchor = anchor + anchor_offset
    weights = (
        _lock_weights(
            stop_frame - start_frame,
            blend_in_frames,
            blend_out_frames,
        )
        * lock_strength
    )
    repaired_local = clip.local_quat_wxyz.copy()
    repaired_root = clip.root_translation_m.copy()
    residuals: list[float] = []
    clamped_frames: list[int] = []
    for local_frame, frame in enumerate(range(start_frame, stop_frame)):
        weight = weights[local_frame]
        if weight <= 0.0:
            continue
        trajectory_fraction = local_frame / max(stop_frame - start_frame - 1, 1)
        moving_anchor = anchor + trajectory_fraction * (anchor_offset_end - anchor_offset)
        target = (1.0 - weight) * end_trajectory[frame] + weight * moving_anchor
        if np.linalg.norm(target - end_trajectory[frame]) < 1.0e-7:
            continue
        result = solve_two_bone_ik(
            clip.skeleton,
            repaired_local[frame],
            repaired_root[frame],
            hip=hip,
            knee=knee,
            end=end,
            target_world_m=target,
        )
        if result.target_was_clamped and pelvis_compensation_limit_m > 0.0:
            frame_position, _ = forward_kinematics_numpy(
                clip.skeleton,
                repaired_local[frame],
                repaired_root[frame],
            )
            hip_to_target = target - frame_position[hip]
            target_distance = np.linalg.norm(hip_to_target)
            if target_distance > 1.0e-8:
                compensation = min(
                    result.residual_m + 2.0e-7,
                    pelvis_compensation_limit_m,
                )
                repaired_root[frame] += (hip_to_target / target_distance * compensation).astype(
                    np.float32
                )
                result = solve_two_bone_ik(
                    clip.skeleton,
                    repaired_local[frame],
                    repaired_root[frame],
                    hip=hip,
                    knee=knee,
                    end=end,
                    target_world_m=target,
                )
        repaired_local[frame] = result.local_quat_wxyz
        residuals.append(result.residual_m)
        if result.target_was_clamped:
            clamped_frames.append(frame)
        if preserve_foot_orientation:
            _, repaired_global = forward_kinematics_numpy(
                clip.skeleton,
                repaired_local[frame],
                repaired_root[frame],
            )
            parent = int(clip.skeleton.parents[end])
            desired_local = quaternion_multiply(
                quaternion_inverse(repaired_global[parent]),
                global_rotation[frame, end],
            )
            repaired_local[frame, end] = quaternion_normalize(desired_local).astype(np.float32)

    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "lock_foot",
            "operator_parameters": {
                "side": side,
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "anchor_mode": anchor_mode,
                "lock_strength": lock_strength,
                "blend_in_frames": blend_in_frames,
                "blend_out_frames": blend_out_frames,
                "preserve_foot_orientation": preserve_foot_orientation,
                "pelvis_compensation_limit_m": pelvis_compensation_limit_m,
                "anchor_offset_m": anchor_offset.tolist(),
                "anchor_offset_end_m": anchor_offset_end.tolist(),
            },
            "ik_clamped_frames": clamped_frames,
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="lock_foot",
        kind="repair",
        parameters={
            "side": side,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "anchor_mode": anchor_mode,
            "lock_strength": lock_strength,
            "blend_in_frames": blend_in_frames,
            "blend_out_frames": blend_out_frames,
            "preserve_foot_orientation": preserve_foot_orientation,
            "pelvis_compensation_limit_m": pelvis_compensation_limit_m,
            "anchor_offset_m": anchor_offset.tolist(),
            "anchor_offset_end_m": anchor_offset_end.tolist(),
        },
        version="motionlab-0.1.0",
    )
    repaired = clip.with_updates(
        local_quat_wxyz=repaired_local,
        root_translation_m=repaired_root,
        metadata=metadata,
    )
    before_contacts = _interval_contacts(
        clip,
        side=side,
        start=start_frame,
        stop=stop_frame,
        ground_plane=ground_plane,
    )
    after_contacts = _interval_contacts(
        repaired,
        side=side,
        start=start_frame,
        stop=stop_frame,
        ground_plane=ground_plane,
    )
    before_slip = foot_sliding_metric(clip, before_contacts).clip_value or 0.0
    after_slip = foot_sliding_metric(repaired, after_contacts).clip_value or 0.0
    return FootLockResult(
        repaired=repaired,
        side=side,
        interval=(start_frame, stop_frame),
        anchor_world_m=(float(anchor[0]), float(anchor[1]), float(anchor[2])),
        before_slip_cm=before_slip,
        after_slip_cm=after_slip,
        maximum_ik_target_residual_m=max(residuals, default=0.0),
        clamped_frames=tuple(clamped_frames),
    )
