"""Deterministic temporal, coordination, speed, and soft-style corruptions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.core.provenance import append_operation_lineage
from motionlab.corruptions._supervision import (
    build_corruption_result,
    joint_part,
    resolve_ground_plane,
    smooth_zero_boundary_envelope,
    validate_interval,
)
from motionlab.corruptions.base import CorruptionResult, IneffectiveCorruptionError
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_geodesic_distance,
    quaternion_multiply,
    quaternion_slerp,
)
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.metrics.speed import speed_metric
from motionlab.motion.clip import MotionClip
from motionlab.rigging.semantic_roles import FunctionalPart

JointReference = int | str


def _resolve_joint(clip: MotionClip, reference: JointReference) -> int:
    if isinstance(reference, int):
        if reference < 0 or reference >= clip.num_joints:
            raise ValueError("joint index is out of range")
        return reference
    if reference in clip.skeleton.roles:
        return clip.skeleton.roles[reference]
    try:
        return clip.skeleton.joint_names.index(reference)
    except ValueError as exc:
        raise ValueError(f"unknown joint or role {reference!r}") from exc


def _resolve_joints(clip: MotionClip, references: Sequence[JointReference]) -> tuple[int, ...]:
    joints = tuple(dict.fromkeys(_resolve_joint(clip, reference) for reference in references))
    if not joints:
        raise ValueError("at least one joint is required")
    return joints


def _part_masks(
    clip: MotionClip,
    joints: Sequence[int],
    frames: NDArray[np.bool_],
) -> dict[FunctionalPart, NDArray[np.bool_]]:
    return {joint_part(clip.skeleton, joint): frames for joint in joints}


def _rotation_interventions(
    joints: Sequence[int], frames: NDArray[np.bool_]
) -> list[tuple[int, tuple[int, int, int], NDArray[np.bool_]]]:
    return [(joint, (3, 4, 5), frames) for joint in joints]


def _with_corruption_metadata(
    clip: MotionClip,
    *,
    operation: str,
    seed: int,
    parameters: dict[str, object],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        **dict(clip.metadata),
        "parent_motion_id": clip.content_hash,
        "operation": operation,
        "corruption_seed": seed,
    }
    return append_operation_lineage(
        clip,
        metadata,
        operation=operation,
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )


def _jitter_noise(
    rng: np.random.Generator,
    length: int,
    joint_count: int,
    *,
    mechanism: Literal["white_tangent_noise", "band_limited_correlated"],
) -> NDArray[np.float64]:
    noise = rng.normal(size=(length, joint_count, 3))
    if mechanism == "band_limited_correlated":
        common = rng.normal(size=(length, 1, 3))
        noise = 0.65 * common + 0.35 * noise
        kernel = np.asarray((0.15, 0.35, 0.35, 0.15), dtype=np.float64)
        for joint in range(joint_count):
            for axis in range(3):
                noise[:, joint, axis] = np.convolve(noise[:, joint, axis], kernel, mode="same")
    rms = float(np.sqrt(np.mean(noise * noise)))
    if rms < np.finfo(np.float64).eps:
        raise ValueError("generated jitter has zero energy")
    return np.asarray(noise / rms, dtype=np.float64)


def corrupt_joint_jitter(
    clip: MotionClip,
    *,
    joints: Sequence[JointReference],
    start_frame: int,
    stop_frame: int,
    amplitude_rad: float,
    seed: int,
    mechanism: Literal["white_tangent_noise", "band_limited_correlated"] = ("white_tangent_noise"),
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Apply seeded tangent-space jitter and measure localized angular jerk."""
    validate_interval(clip, start_frame, stop_frame)
    if not np.isfinite(amplitude_rad) or amplitude_rad < 0.0:
        raise ValueError("amplitude_rad must be finite and nonnegative")
    selected = _resolve_joints(clip, joints)
    length = stop_frame - start_frame
    envelope = smooth_zero_boundary_envelope(length)
    rng = np.random.default_rng(seed)
    tangent = (
        amplitude_rad
        * envelope[:, None, None]
        * _jitter_noise(rng, length, len(selected), mechanism=mechanism)
    )
    local = clip.local_quat_wxyz.copy()
    for local_index, joint in enumerate(selected):
        delta = axis_angle_to_quaternion(tangent[:, local_index])
        local[start_frame:stop_frame, joint] = quaternion_multiply(
            local[start_frame:stop_frame, joint], delta
        ).astype(np.float32)
    parameters: dict[str, object] = {
        "mechanism": mechanism,
        "joint_indices": list(selected),
        "joint_names": [clip.skeleton.joint_names[joint] for joint in selected],
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "amplitude_rad": amplitude_rad,
    }
    corrupted = clip.with_updates(
        local_quat_wxyz=local,
        metadata=_with_corruption_metadata(
            clip,
            operation=f"corrupt_joint_jitter_{mechanism}",
            seed=seed,
            parameters=parameters,
        ),
    )
    before_metric = smoothness_metric(clip)
    after_metric = smoothness_metric(corrupted)
    support = slice(max(0, start_frame - 2), min(clip.num_frames, stop_frame + 2))
    before_values = np.asarray(before_metric.joint_frame_values, dtype=np.float64)[support][
        :, selected
    ]
    after_values = np.asarray(after_metric.joint_frame_values, dtype=np.float64)[support][
        :, selected
    ]
    before = float(np.percentile(before_values, 95.0))
    after = float(np.percentile(after_values, 95.0))
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start_frame:stop_frame] = envelope > 0.0
    symptom_frames = np.zeros(clip.num_frames, dtype=np.bool_)
    symptom_frames[support] = amplitude_rad > 0.0
    parts = _part_masks(clip, selected, symptom_frames)
    partition = "train" if mechanism == "white_tangent_noise" else "heldout_eval"
    return build_corruption_result(
        clip,
        corrupted,
        family="joint_jitter",
        mechanism=mechanism,
        interventions=_rotation_interventions(selected, affected),
        symptoms=parts,
        responsibilities={part: (frames, 1.0, 0.85) for part, frames in parts.items()},
        metric_name="localized_angular_jerk_p95_rad_s3",
        metric_before=before,
        metric_after=after,
        minimum_worsening=1.0e-3,
        evidence_confidence=1.0,
        requested_severity=amplitude_rad,
        parameters=parameters,
        seed=seed,
        catalog_partition=partition,
        strict_postcondition=strict_postcondition,
        schema_metadata={"symptom_uses_derivative_support_window": True},
    )


def corrupt_joint_jitter_white(
    clip: MotionClip,
    *,
    joints: Sequence[JointReference],
    start_frame: int,
    stop_frame: int,
    amplitude_rad: float,
    seed: int,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Generate the training jitter implementation."""
    return corrupt_joint_jitter(
        clip,
        joints=joints,
        start_frame=start_frame,
        stop_frame=stop_frame,
        amplitude_rad=amplitude_rad,
        seed=seed,
        mechanism="white_tangent_noise",
        strict_postcondition=strict_postcondition,
    )


def corrupt_joint_jitter_band_limited(
    clip: MotionClip,
    *,
    joints: Sequence[JointReference],
    start_frame: int,
    stop_frame: int,
    amplitude_rad: float,
    seed: int,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Generate the held-out correlated, band-limited jitter implementation."""
    return corrupt_joint_jitter(
        clip,
        joints=joints,
        start_frame=start_frame,
        stop_frame=stop_frame,
        amplitude_rad=amplitude_rad,
        seed=seed,
        mechanism="band_limited_correlated",
        strict_postcondition=strict_postcondition,
    )


def _arm_joints(clip: MotionClip, sides: Sequence[str]) -> tuple[int, ...]:
    roles = [
        role
        for side in sides
        for role in (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        if role in clip.skeleton.roles
    ]
    if not roles:
        raise ValueError("limb-phase corruption requires authored arm roles")
    return _resolve_joints(clip, roles)


def corrupt_limb_phase_shift(
    clip: MotionClip,
    *,
    start_frame: int,
    stop_frame: int,
    shift_frames: int,
    side: Literal["left", "right"] = "left",
    bilateral: bool = False,
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Shift arm channels relative to the lower-body timeline with smooth boundaries."""
    validate_interval(clip, start_frame, stop_frame)
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if shift_frames == 0:
        raise ValueError("shift_frames must be nonzero")
    sides = ("left", "right") if bilateral else (side,)
    selected = _arm_joints(clip, sides)
    length = stop_frame - start_frame
    envelope = smooth_zero_boundary_envelope(length)
    frames = np.arange(start_frame, stop_frame)
    shifted_frames = np.mod(frames - shift_frames, clip.num_frames)
    local = clip.local_quat_wxyz.copy()
    for joint in selected:
        local[start_frame:stop_frame, joint] = quaternion_slerp(
            clip.local_quat_wxyz[start_frame:stop_frame, joint],
            clip.local_quat_wxyz[shifted_frames, joint],
            envelope,
        ).astype(np.float32)
    mechanism = "bilateral_circular_shift" if bilateral else "unilateral_circular_shift"
    parameters: dict[str, object] = {
        "mechanism": mechanism,
        "sides": list(sides),
        "joint_indices": list(selected),
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "shift_frames": shift_frames,
    }
    corrupted = clip.with_updates(
        local_quat_wxyz=local,
        metadata=_with_corruption_metadata(
            clip,
            operation=f"corrupt_limb_phase_{mechanism}",
            seed=seed,
            parameters=parameters,
        ),
    )
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start_frame:stop_frame] = envelope > 0.0
    deviation = quaternion_geodesic_distance(
        clip.local_quat_wxyz[:, selected], corrupted.local_quat_wxyz[:, selected]
    )
    measured = float(np.sqrt(np.mean(np.square(deviation[affected]))))
    parts = _part_masks(clip, selected, affected)
    return build_corruption_result(
        clip,
        corrupted,
        family="limb_phase_mismatch",
        mechanism=mechanism,
        interventions=_rotation_interventions(selected, affected),
        symptoms=parts,
        responsibilities={part: (frames, 1.0, 0.75) for part, frames in parts.items()},
        metric_name="source_relative_arm_phase_deviation_rad_rms",
        metric_before=0.0,
        metric_after=measured,
        minimum_worsening=1.0e-5,
        evidence_confidence=0.75,
        requested_severity=float(abs(shift_frames)),
        parameters=parameters,
        seed=seed,
        catalog_partition="heldout_eval" if bilateral else "train",
        strict_postcondition=strict_postcondition,
        schema_metadata={"symptom_is_source_relative_phase_deviation": True},
    )


def _time_warp_map(start_frame: int, stop_frame: int, strength: float) -> NDArray[np.float64]:
    length = stop_frame - start_frame
    unit = np.linspace(0.0, 1.0, length, dtype=np.float64)
    warped = unit + strength * np.sin(2.0 * np.pi * unit) / (2.0 * np.pi)
    source = start_frame + warped * (length - 1)
    if np.any(np.diff(source) <= 0.0):
        raise ValueError("time-warp strength did not produce a monotonic map")
    return source


def corrupt_time_warp(
    clip: MotionClip,
    *,
    start_frame: int,
    stop_frame: int,
    strength: float,
    warp_root_only: bool = False,
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Warp pose or root timing monotonically while keeping the other stream unchanged."""
    validate_interval(clip, start_frame, stop_frame)
    if not np.isfinite(strength) or not 0.0 < abs(strength) < 0.95:
        raise ValueError("strength magnitude must be finite and within (0,0.95)")
    source = _time_warp_map(start_frame, stop_frame, strength)
    left = np.floor(source).astype(np.int64)
    right = np.minimum(left + 1, stop_frame - 1)
    fraction = source - left
    root = clip.root_translation_m.copy()
    local = clip.local_quat_wxyz.copy()
    selected: tuple[int, ...]
    if warp_root_only:
        for axis in range(3):
            root[start_frame:stop_frame, axis] = np.interp(
                source,
                np.arange(clip.num_frames),
                clip.root_translation_m[:, axis],
            ).astype(np.float32)
        selected = (clip.skeleton.root_index,)
        mechanism = "root_stream_monotonic_warp"
    else:
        local[start_frame:stop_frame] = quaternion_slerp(
            clip.local_quat_wxyz[left],
            clip.local_quat_wxyz[right],
            fraction[:, None],
        ).astype(np.float32)
        selected = tuple(range(clip.num_joints))
        mechanism = "pose_stream_monotonic_warp"
    parameters: dict[str, object] = {
        "mechanism": mechanism,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "strength": strength,
        "warp_root_only": warp_root_only,
    }
    corrupted = clip.with_updates(
        local_quat_wxyz=local,
        root_translation_m=root,
        metadata=_with_corruption_metadata(
            clip,
            operation=f"corrupt_time_warp_{mechanism}",
            seed=seed,
            parameters=parameters,
        ),
    )
    map_changed = np.abs(source - np.arange(start_frame, stop_frame)) > 1.0e-9
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start_frame:stop_frame] = map_changed
    if warp_root_only:
        delta = corrupted.root_translation_m - clip.root_translation_m
        measured = 100.0 * float(np.sqrt(np.mean(np.square(delta[affected]))))
        interventions = [(clip.skeleton.root_index, (0, 1, 2), affected)]
        parts = {FunctionalPart.ROOT_PELVIS: affected}
        metric_name = "source_relative_root_timing_deviation_cm_rms"
    else:
        deviation = quaternion_geodesic_distance(clip.local_quat_wxyz, corrupted.local_quat_wxyz)
        measured = float(np.sqrt(np.mean(np.square(deviation[affected]))))
        interventions = _rotation_interventions(selected, affected)
        parts = _part_masks(clip, selected, affected)
        metric_name = "source_relative_pose_timing_deviation_rad_rms"
    return build_corruption_result(
        clip,
        corrupted,
        family="cadence_inconsistency",
        mechanism=mechanism,
        interventions=interventions,
        symptoms=parts,
        responsibilities={part: (frames, 1.0, 0.75) for part, frames in parts.items()},
        metric_name=metric_name,
        metric_before=0.0,
        metric_after=measured,
        minimum_worsening=1.0e-5,
        evidence_confidence=0.75,
        requested_severity=abs(strength),
        parameters=parameters,
        seed=seed,
        catalog_partition="heldout_eval" if warp_root_only else "train",
        strict_postcondition=strict_postcondition,
        schema_metadata={"monotonic_warp_map": source.tolist()},
    )


def corrupt_speed_root_progression(
    clip: MotionClip,
    *,
    speed_scale: float,
    seed: int = 0,
    ground_plane: ArrayLike | None = None,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Scale root progression without changing stride pose channels."""
    if not np.isfinite(speed_scale) or speed_scale <= 0.0 or np.isclose(speed_scale, 1.0):
        raise ValueError("speed_scale must be finite, positive, and different from one")
    plane = resolve_ground_plane(clip, ground_plane)
    origin = clip.root_translation_m[0].astype(np.float64)
    delta = clip.root_translation_m.astype(np.float64) - origin
    normal_component = np.sum(delta * plane[:3], axis=-1, keepdims=True) * plane[:3]
    tangential = delta - normal_component
    root = origin + normal_component + speed_scale * tangential
    parameters: dict[str, object] = {
        "mechanism": "root_progression_scale",
        "speed_scale": speed_scale,
        "ground_plane": plane.tolist(),
    }
    corrupted = clip.with_updates(
        root_translation_m=root.astype(np.float32),
        metadata=_with_corruption_metadata(
            clip,
            operation="corrupt_speed_root_progression",
            seed=seed,
            parameters=parameters,
        ),
    )
    target_raw = clip.metadata.get("expected_speed_mps")
    clean_speed = speed_metric(clip, ground_plane=plane)
    target = float(clean_speed.clip_value or 0.0) if target_raw is None else float(target_raw)
    before_metric = speed_metric(clip, ground_plane=plane, target_speed_mps=target)
    after_metric = speed_metric(corrupted, ground_plane=plane, target_speed_mps=target)
    before = float(before_metric.metadata["absolute_target_error_mps"] or 0.0)
    after = float(after_metric.metadata["absolute_target_error_mps"] or 0.0)
    affected = np.linalg.norm(root - clip.root_translation_m, axis=-1) > 1.0e-9
    channels = tuple(
        axis
        for axis in range(3)
        if np.any(np.abs(root[:, axis] - clip.root_translation_m[:, axis]) > 1.0e-9)
    )
    return build_corruption_result(
        clip,
        corrupted,
        family="speed_inconsistency",
        mechanism="root_progression_scale",
        interventions=[(clip.skeleton.root_index, channels, affected)],
        symptoms={FunctionalPart.ROOT_PELVIS: affected},
        responsibilities={FunctionalPart.ROOT_PELVIS: (affected, 1.0, 1.0)},
        metric_name="absolute_target_speed_error_mps",
        metric_before=before,
        metric_after=after,
        minimum_worsening=1.0e-5,
        evidence_confidence=1.0,
        requested_severity=abs(speed_scale - 1.0),
        parameters=parameters,
        seed=seed,
        catalog_partition="train",
        strict_postcondition=strict_postcondition,
        schema_metadata={"task_target_source": "metadata_or_clean_speed"},
    )


def corrupt_leg_stride_amplitude(
    clip: MotionClip,
    *,
    amplitude_scale: float,
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Scale leg pose excursion while retaining root speed; held out for speed/stride tests."""
    if (
        not np.isfinite(amplitude_scale)
        or amplitude_scale <= 0.0
        or np.isclose(amplitude_scale, 1.0)
    ):
        raise ValueError("amplitude_scale must be finite, positive, and different from one")
    leg_roles = [
        role
        for side in ("left", "right")
        for role in (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
        if role in clip.skeleton.roles
    ]
    selected = _resolve_joints(clip, leg_roles)
    local = clip.local_quat_wxyz.copy()
    for joint in selected:
        rest = np.broadcast_to(clip.skeleton.rest_local_quat_wxyz[joint], (clip.num_frames, 4))
        local[:, joint] = quaternion_slerp(
            rest,
            clip.local_quat_wxyz[:, joint],
            np.full(clip.num_frames, amplitude_scale),
        ).astype(np.float32)
    parameters: dict[str, object] = {
        "mechanism": "leg_pose_amplitude_scale",
        "amplitude_scale": amplitude_scale,
        "joint_indices": list(selected),
    }
    corrupted = clip.with_updates(
        local_quat_wxyz=local,
        metadata=_with_corruption_metadata(
            clip,
            operation="corrupt_leg_stride_amplitude",
            seed=seed,
            parameters=parameters,
        ),
    )
    deviation = quaternion_geodesic_distance(
        clip.local_quat_wxyz[:, selected], corrupted.local_quat_wxyz[:, selected]
    )
    measured = float(np.sqrt(np.mean(np.square(deviation))))
    affected = np.any(deviation > 1.0e-9, axis=1)
    parts = _part_masks(clip, selected, affected)
    return build_corruption_result(
        clip,
        corrupted,
        family="speed_inconsistency",
        mechanism="leg_pose_amplitude_scale",
        interventions=_rotation_interventions(selected, affected),
        symptoms=parts,
        responsibilities={part: (frames, 1.0, 0.7) for part, frames in parts.items()},
        metric_name="source_relative_leg_stride_pose_deviation_rad_rms",
        metric_before=0.0,
        metric_after=measured,
        minimum_worsening=1.0e-5,
        evidence_confidence=0.7,
        requested_severity=abs(amplitude_scale - 1.0),
        parameters=parameters,
        seed=seed,
        catalog_partition="heldout_eval",
        strict_postcondition=strict_postcondition,
        schema_metadata={"root_speed_is_held_fixed": True},
    )


def corrupt_pelvis_vertical_curve(
    clip: MotionClip,
    *,
    amplitude_scale: float,
    seed: int = 0,
) -> CorruptionResult:
    """Scale pelvis vertical excursion as a soft, non-preference corruption."""
    if not np.isfinite(amplitude_scale) or amplitude_scale < 0.0:
        raise ValueError("amplitude_scale must be finite and nonnegative")
    if np.isclose(amplitude_scale, 1.0):
        raise ValueError("amplitude_scale must differ from one")
    root = clip.root_translation_m.copy()
    center = float(np.mean(root[:, 1]))
    root[:, 1] = (center + amplitude_scale * (root[:, 1] - center)).astype(np.float32)
    affected = np.abs(root[:, 1] - clip.root_translation_m[:, 1]) > 1.0e-9
    measured = 100.0 * float(
        np.sqrt(np.mean(np.square(root[:, 1] - clip.root_translation_m[:, 1])))
    )
    if measured < 1.0e-5:
        raise IneffectiveCorruptionError("pelvis vertical curve edit produced no measurable change")
    parameters: dict[str, object] = {
        "mechanism": "vertical_amplitude_scale",
        "amplitude_scale": amplitude_scale,
        "source_vertical_center_m": center,
    }
    corrupted = clip.with_updates(
        root_translation_m=root,
        metadata=_with_corruption_metadata(
            clip,
            operation="corrupt_pelvis_vertical_curve",
            seed=seed,
            parameters=parameters,
        ),
    )
    return build_corruption_result(
        clip,
        corrupted,
        family="pelvis_curve",
        mechanism="vertical_amplitude_scale",
        interventions=[(clip.skeleton.root_index, (1,), affected)],
        symptoms={FunctionalPart.ROOT_PELVIS: affected},
        responsibilities={FunctionalPart.ROOT_PELVIS: (affected, 1.0, 0.5)},
        metric_name="source_relative_pelvis_vertical_deviation_cm_rms",
        metric_before=0.0,
        metric_after=measured,
        minimum_worsening=1.0e-5,
        evidence_confidence=0.5,
        requested_severity=abs(amplitude_scale - 1.0),
        parameters=parameters,
        seed=seed,
        catalog_partition="train",
        hard_negative_candidate=False,
        strict_postcondition=False,
        schema_metadata={"preference_label": "withheld_soft_style_change"},
    )


def corrupt_style_region_blend(
    clip: MotionClip,
    donor: MotionClip,
    *,
    parts: Sequence[FunctionalPart],
    blend_weight: float,
    seed: int = 0,
) -> CorruptionResult:
    """Blend donor body regions while withholding an automatic quality preference."""
    if clip.skeleton.content_hash != donor.skeleton.content_hash:
        raise ValueError("style-region blend requires the same skeleton")
    if clip.num_frames != donor.num_frames or clip.fps != donor.fps:
        raise ValueError("style-region blend requires aligned source timelines")
    if not parts:
        raise ValueError("style-region blend requires at least one functional part")
    if not np.isfinite(blend_weight) or not 0.0 < blend_weight <= 1.0:
        raise ValueError("blend_weight must be within (0,1]")
    selected_parts = set(parts)
    joints = tuple(
        joint
        for joint in range(clip.num_joints)
        if joint_part(clip.skeleton, joint) in selected_parts
    )
    if not joints:
        raise ValueError("selected style regions contain no joints")
    local = clip.local_quat_wxyz.copy()
    local[:, joints] = quaternion_slerp(
        clip.local_quat_wxyz[:, joints],
        donor.local_quat_wxyz[:, joints],
        np.full((clip.num_frames, len(joints)), blend_weight),
    ).astype(np.float32)
    parameters: dict[str, object] = {
        "mechanism": "aligned_region_blend",
        "donor_motion_id": donor.content_hash,
        "functional_parts": [part.name.lower() for part in parts],
        "blend_weight": blend_weight,
    }
    corrupted = clip.with_updates(
        local_quat_wxyz=local,
        metadata=_with_corruption_metadata(
            clip,
            operation="corrupt_style_region_blend",
            seed=seed,
            parameters=parameters,
        ),
    )
    deviation = quaternion_geodesic_distance(
        clip.local_quat_wxyz[:, joints], corrupted.local_quat_wxyz[:, joints]
    )
    affected = np.any(deviation > 1.0e-9, axis=1)
    measured = float(np.sqrt(np.mean(np.square(deviation))))
    if measured < 1.0e-5:
        raise IneffectiveCorruptionError("style-region blend produced no measurable change")
    part_masks = {part: affected for part in parts}
    return build_corruption_result(
        clip,
        corrupted,
        family="style_incoherence",
        mechanism="aligned_region_blend",
        interventions=_rotation_interventions(joints, affected),
        symptoms=part_masks,
        responsibilities={part: (affected, 1.0, 0.35) for part in parts},
        metric_name="source_relative_region_pose_deviation_rad_rms",
        metric_before=0.0,
        metric_after=measured,
        minimum_worsening=1.0e-5,
        evidence_confidence=0.35,
        requested_severity=blend_weight,
        parameters=parameters,
        seed=seed,
        catalog_partition="train",
        hard_negative_candidate=False,
        strict_postcondition=False,
        schema_metadata={"preference_label": "withheld_style_incoherence"},
    )
