"""Measured hard-negative corruptions for physical and temporal defects."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from motionlab.core.provenance import append_operation_lineage, provenance_from_clip
from motionlab.corruptions._supervision import (
    build_corruption_result,
    contact_part_frames,
    dense_supervision_masks,
    joint_part,
    measure_contact_supervision,
    resolve_ground_plane,
    smooth_zero_boundary_envelope,
    validate_interval,
    verify_worsening,
)
from motionlab.corruptions.base import CorruptionResult
from motionlab.kinematics.fk import forward_kinematics_numpy, marker_world_positions
from motionlab.kinematics.ik_two_bone import solve_two_bone_ik
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_slerp,
)
from motionlab.metrics.ground import ground_metrics
from motionlab.metrics.joint_limits import joint_limit_metric
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult, detect_foot_contacts
from motionlab.rigging.semantic_roles import FunctionalPart

JointReference: TypeAlias = int | str


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


def _rotation_axis(axis_local: ArrayLike) -> np.ndarray:
    axis = np.asarray(axis_local, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axis_local must be a finite 3D vector")
    length = float(np.linalg.norm(axis))
    if length < 1.0e-8:
        raise ValueError("axis_local must be nonzero")
    return axis / length


def corrupt_ground_penetration_root_offset(
    clip: MotionClip,
    *,
    start_frame: int,
    stop_frame: int,
    depth_m: float,
    seed: int = 0,
    source_contacts: ContactResult | None = None,
    ground_plane: ArrayLike | None = None,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Push the root smoothly into the ground plane over a contact interval."""
    validate_interval(clip, start_frame, stop_frame)
    if not np.isfinite(depth_m) or depth_m < 0.0:
        raise ValueError("depth_m must be finite and nonnegative")
    plane = resolve_ground_plane(clip, ground_plane)
    envelope = smooth_zero_boundary_envelope(stop_frame - start_frame)
    root = clip.root_translation_m.copy()
    root[start_frame:stop_frame] -= (envelope[:, None] * depth_m * plane[:3]).astype(np.float32)
    parameters: dict[str, object] = {
        "mechanism": "root_vertical_offset",
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "depth_m": depth_m,
        "ground_normal": plane[:3].tolist(),
    }
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_ground_penetration_root_offset",
            "corruption_seed": seed,
            "corruption_interval": [start_frame, stop_frame],
            "corruption_depth_m": depth_m,
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="corrupt_ground_penetration_root_offset",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(root_translation_m=root, metadata=metadata)
    contact_supervision, fixed_clean, fixed_corrupted = measure_contact_supervision(
        clip,
        corrupted,
        source_contacts=source_contacts,
        ground_plane=plane,
    )
    before_metric, _ = ground_metrics(fixed_clean)
    after_metric, _ = ground_metrics(fixed_corrupted)
    before = float(before_metric.clip_value or 0.0)
    after = float(after_metric.clip_value or 0.0)
    postcondition, measured_severity, preference_confidence = verify_worsening(
        metric_name="maximum_ground_penetration_cm",
        before=before,
        after=after,
        minimum_worsening=0.01,
        evidence_confidence=min(before_metric.confidence, after_metric.confidence),
        requested_active=depth_m > 0.0,
        strict=strict_postcondition,
    )
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start_frame:stop_frame] = envelope > 0.0
    symptoms = contact_part_frames(fixed_clean, affected)
    responsibilities: dict[
        FunctionalPart,
        tuple[np.ndarray, float, float],
    ] = {
        FunctionalPart.ROOT_PELVIS: (affected, 0.8, 0.9),
        **{part: (frames, 1.0, 0.9) for part, frames in symptoms.items()},
    }
    channels = tuple(index for index, component in enumerate(plane[:3]) if abs(component) > 1.0e-12)
    intervention, symptom, responsibility, responsibility_confidence = dense_supervision_masks(
        clip,
        family="ground_penetration",
        interventions=[(clip.skeleton.root_index, channels, affected)],
        symptoms=symptoms,
        responsibilities=responsibilities,
    )
    provenance = provenance_from_clip(clip)
    return CorruptionResult(
        corrupted_motion=corrupted,
        clean_motion=clip,
        corruption_family="ground_penetration",
        corruption_mechanism="root_vertical_offset",
        intervention_mask=intervention,
        symptom_mask=symptom,
        responsibility_target=responsibility,
        responsibility_confidence=responsibility_confidence,
        measured_metrics_before={"maximum_ground_penetration_cm": before},
        measured_metrics_after={"maximum_ground_penetration_cm": after},
        measured_severity=measured_severity,
        preference_confidence=preference_confidence,
        source_motion_id=clip.content_hash,
        corruption_seed=seed,
        split_lineage_id=provenance.split_lineage_id,
        requested_severity_parameter=depth_m,
        generation_parameters=parameters,
        postcondition=postcondition,
        contact_supervision=contact_supervision,
        catalog_partition="train",
        schema_metadata={
            "intervention_is_exact": True,
            "symptom_is_contact_supported": True,
            "fine_joint_causal_blame_is_supervised": False,
        },
    )


def corrupt_joint_pop_rotation_pulse(
    clip: MotionClip,
    *,
    joint: JointReference,
    frame: int,
    angle_rad: float,
    axis_local: ArrayLike = (1.0, 0.0, 0.0),
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Inject one local-rotation sample and verify increased maximum angular jerk."""
    if frame < 0 or frame >= clip.num_frames:
        raise ValueError("frame is out of range")
    if not np.isfinite(angle_rad) or angle_rad < 0.0:
        raise ValueError("angle_rad must be finite and nonnegative")
    joint_index = _resolve_joint(clip, joint)
    axis = _rotation_axis(axis_local)
    local = clip.local_quat_wxyz.copy()
    delta = axis_angle_to_quaternion(angle_rad * axis)
    local[frame, joint_index] = quaternion_multiply(local[frame, joint_index], delta).astype(
        np.float32
    )
    parameters: dict[str, object] = {
        "mechanism": "local_rotation_pulse",
        "joint_index": joint_index,
        "joint_name": clip.skeleton.joint_names[joint_index],
        "frame": frame,
        "angle_rad": angle_rad,
        "axis_local": axis.tolist(),
    }
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_joint_pop_rotation_pulse",
            "corruption_seed": seed,
            "corruption_frame": frame,
            "corruption_angle_rad": angle_rad,
            "corruption_joint": clip.skeleton.joint_names[joint_index],
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="corrupt_joint_pop_rotation_pulse",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(local_quat_wxyz=local, metadata=metadata)
    before_metric = smoothness_metric(clip)
    after_metric = smoothness_metric(corrupted)
    before_joint_values = np.asarray(before_metric.joint_frame_values, dtype=np.float64)
    after_joint_values = np.asarray(after_metric.joint_frame_values, dtype=np.float64)
    support = slice(max(0, frame - 2), min(clip.num_frames, frame + 3))
    before = float(np.max(before_joint_values[support, joint_index], initial=0.0))
    after = float(np.max(after_joint_values[support, joint_index], initial=0.0))
    postcondition, measured_severity, preference_confidence = verify_worsening(
        metric_name="localized_target_joint_peak_angular_jerk_rad_s3",
        before=before,
        after=after,
        minimum_worsening=1.0e-3,
        evidence_confidence=min(before_metric.confidence, after_metric.confidence),
        requested_active=angle_rad > 0.0,
        strict=strict_postcondition,
    )
    intervention_frames = np.zeros(clip.num_frames, dtype=np.bool_)
    intervention_frames[frame] = angle_rad > 0.0
    symptom_frames = np.zeros(clip.num_frames, dtype=np.bool_)
    symptom_frames[max(0, frame - 2) : min(clip.num_frames, frame + 3)] = angle_rad > 0.0
    part = joint_part(clip.skeleton, joint_index)
    channels = tuple(3 + index for index, component in enumerate(axis) if abs(component) > 1.0e-12)
    intervention, symptom, responsibility, responsibility_confidence = dense_supervision_masks(
        clip,
        family="joint_pop",
        interventions=[(joint_index, channels, intervention_frames)],
        symptoms={part: symptom_frames},
        responsibilities={part: (symptom_frames, 1.0, 0.95)},
    )
    provenance = provenance_from_clip(clip)
    return CorruptionResult(
        corrupted_motion=corrupted,
        clean_motion=clip,
        corruption_family="joint_pop",
        corruption_mechanism="local_rotation_pulse",
        intervention_mask=intervention,
        symptom_mask=symptom,
        responsibility_target=responsibility,
        responsibility_confidence=responsibility_confidence,
        measured_metrics_before={"localized_target_joint_peak_angular_jerk_rad_s3": before},
        measured_metrics_after={"localized_target_joint_peak_angular_jerk_rad_s3": after},
        measured_severity=measured_severity,
        preference_confidence=preference_confidence,
        source_motion_id=clip.content_hash,
        corruption_seed=seed,
        split_lineage_id=provenance.split_lineage_id,
        requested_severity_parameter=angle_rad,
        generation_parameters=parameters,
        postcondition=postcondition,
        catalog_partition="train",
        schema_metadata={
            "intervention_is_exact": True,
            "symptom_uses_derivative_support_window": True,
            "fine_joint_causal_blame_is_supervised": False,
        },
    )


def corrupt_loop_seam_end_rotation(
    clip: MotionClip,
    *,
    joint: JointReference,
    seam_window_frames: int,
    angle_rad: float,
    axis_local: ArrayLike = (0.0, 1.0, 0.0),
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Ramp a joint to a mismatched end pose and verify loop-seam degradation."""
    if seam_window_frames < 3 or seam_window_frames > clip.num_frames:
        raise ValueError("seam_window_frames must be within [3, number of frames]")
    if not np.isfinite(angle_rad) or angle_rad < 0.0:
        raise ValueError("angle_rad must be finite and nonnegative")
    joint_index = _resolve_joint(clip, joint)
    axis = _rotation_axis(axis_local)
    blend = np.linspace(0.0, 1.0, seam_window_frames, dtype=np.float64)
    blend = blend * blend * (3.0 - 2.0 * blend)
    delta = axis_angle_to_quaternion(blend[:, None] * angle_rad * axis)
    start = clip.num_frames - seam_window_frames
    local = clip.local_quat_wxyz.copy()
    local[start:, joint_index] = quaternion_multiply(local[start:, joint_index], delta).astype(
        np.float32
    )
    parameters: dict[str, object] = {
        "mechanism": "end_rotation_offset",
        "joint_index": joint_index,
        "joint_name": clip.skeleton.joint_names[joint_index],
        "seam_window_frames": seam_window_frames,
        "angle_rad": angle_rad,
        "axis_local": axis.tolist(),
    }
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_loop_seam_end_rotation",
            "corruption_seed": seed,
            "corruption_seam_window_frames": seam_window_frames,
            "corruption_angle_rad": angle_rad,
            "corruption_joint": clip.skeleton.joint_names[joint_index],
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="corrupt_loop_seam_end_rotation",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(local_quat_wxyz=local, metadata=metadata)
    before_metric = loop_seam_metric(clip)
    after_metric = loop_seam_metric(corrupted)
    before = float(before_metric.clip_value or 0.0)
    after = float(after_metric.clip_value or 0.0)
    postcondition, measured_severity, preference_confidence = verify_worsening(
        metric_name="weighted_loop_seam",
        before=before,
        after=after,
        minimum_worsening=1.0e-5,
        evidence_confidence=min(before_metric.confidence, after_metric.confidence),
        requested_active=angle_rad > 0.0,
        strict=strict_postcondition,
    )
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start + 1 :] = angle_rad > 0.0
    part = joint_part(clip.skeleton, joint_index)
    channels = tuple(3 + index for index, component in enumerate(axis) if abs(component) > 1.0e-12)
    intervention, symptom, responsibility, responsibility_confidence = dense_supervision_masks(
        clip,
        family="loop_seam",
        interventions=[(joint_index, channels, affected)],
        symptoms={part: affected},
        responsibilities={part: (affected, 1.0, 0.9)},
    )
    provenance = provenance_from_clip(clip)
    return CorruptionResult(
        corrupted_motion=corrupted,
        clean_motion=clip,
        corruption_family="loop_seam",
        corruption_mechanism="end_rotation_offset",
        intervention_mask=intervention,
        symptom_mask=symptom,
        responsibility_target=responsibility,
        responsibility_confidence=responsibility_confidence,
        measured_metrics_before={"weighted_loop_seam": before},
        measured_metrics_after={"weighted_loop_seam": after},
        measured_severity=measured_severity,
        preference_confidence=preference_confidence,
        source_motion_id=clip.content_hash,
        corruption_seed=seed,
        split_lineage_id=provenance.split_lineage_id,
        requested_severity_parameter=angle_rad,
        generation_parameters=parameters,
        postcondition=postcondition,
        catalog_partition="train",
        schema_metadata={
            "intervention_is_exact": True,
            "symptom_is_seam_window": True,
            "fine_joint_causal_blame_is_supervised": False,
        },
    )


def corrupt_floating_contact_root_offset(
    clip: MotionClip,
    *,
    start_frame: int,
    stop_frame: int,
    height_m: float,
    seed: int = 0,
    source_contacts: ContactResult | None = None,
    ground_plane: ArrayLike | None = None,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Lift the root over source-contact frames and verify floating-contact height."""
    validate_interval(clip, start_frame, stop_frame)
    if not np.isfinite(height_m) or height_m < 0.0:
        raise ValueError("height_m must be finite and nonnegative")
    plane = resolve_ground_plane(clip, ground_plane)
    envelope = smooth_zero_boundary_envelope(stop_frame - start_frame)
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start_frame:stop_frame] = envelope > 0.0
    root = clip.root_translation_m.copy()
    root[start_frame:stop_frame] += (envelope[:, None] * height_m * plane[:3]).astype(np.float32)
    parameters: dict[str, object] = {
        "mechanism": "root_vertical_offset",
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "height_m": height_m,
        "ground_normal": plane[:3].tolist(),
    }
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_floating_contact_root_offset",
            "corruption_seed": seed,
            "corruption_interval": [start_frame, stop_frame],
            "corruption_height_m": height_m,
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="corrupt_floating_contact_root_offset",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(root_translation_m=root, metadata=metadata)
    contact_supervision, fixed_clean, fixed_corrupted = measure_contact_supervision(
        clip,
        corrupted,
        source_contacts=source_contacts,
        ground_plane=plane,
    )
    _, before_metric = ground_metrics(fixed_clean)
    _, after_metric = ground_metrics(fixed_corrupted)
    before = float(before_metric.clip_value or 0.0)
    after = float(after_metric.clip_value or 0.0)
    symptoms = contact_part_frames(fixed_clean, affected)
    responsibilities: dict[FunctionalPart, tuple[np.ndarray, float, float]] = {
        FunctionalPart.ROOT_PELVIS: (affected, 0.8, 0.9),
        **{part: (frames, 1.0, 0.9) for part, frames in symptoms.items()},
    }
    channels = tuple(index for index, component in enumerate(plane[:3]) if abs(component) > 1.0e-12)
    return build_corruption_result(
        clip,
        corrupted,
        family="floating_contact",
        mechanism="root_vertical_offset",
        interventions=[(clip.skeleton.root_index, channels, affected)],
        symptoms=symptoms,
        responsibilities=responsibilities,
        metric_name="maximum_floating_contact_cm",
        metric_before=before,
        metric_after=after,
        minimum_worsening=0.01,
        evidence_confidence=min(before_metric.confidence, after_metric.confidence),
        requested_severity=height_m,
        parameters=parameters,
        seed=seed,
        catalog_partition="train",
        contact_supervision=contact_supervision,
        strict_postcondition=strict_postcondition,
        schema_metadata={"symptom_is_contact_supported": True},
    )


def corrupt_joint_pop_smooth_pulse(
    clip: MotionClip,
    *,
    joint: JointReference,
    start_frame: int,
    stop_frame: int,
    angle_rad: float,
    axis_local: ArrayLike = (1.0, 0.0, 0.0),
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Apply a compact zero-boundary rotation pulse reserved for pop evaluation."""
    validate_interval(clip, start_frame, stop_frame)
    if not np.isfinite(angle_rad) or angle_rad < 0.0:
        raise ValueError("angle_rad must be finite and nonnegative")
    joint_index = _resolve_joint(clip, joint)
    axis = _rotation_axis(axis_local)
    envelope = smooth_zero_boundary_envelope(stop_frame - start_frame)
    delta = axis_angle_to_quaternion(envelope[:, None] * angle_rad * axis)
    local = clip.local_quat_wxyz.copy()
    local[start_frame:stop_frame, joint_index] = quaternion_multiply(
        local[start_frame:stop_frame, joint_index], delta
    ).astype(np.float32)
    parameters: dict[str, object] = {
        "mechanism": "smooth_rotation_pulse",
        "joint_index": joint_index,
        "joint_name": clip.skeleton.joint_names[joint_index],
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "angle_rad": angle_rad,
        "axis_local": axis.tolist(),
    }
    metadata = append_operation_lineage(
        clip,
        {
            **dict(clip.metadata),
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_joint_pop_smooth_pulse",
            "corruption_seed": seed,
            "corruption_interval": [start_frame, stop_frame],
            "corruption_joint": clip.skeleton.joint_names[joint_index],
            "corruption_angle_rad": angle_rad,
        },
        operation="corrupt_joint_pop_smooth_pulse",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(local_quat_wxyz=local, metadata=metadata)
    before_metric = smoothness_metric(clip)
    after_metric = smoothness_metric(corrupted)
    support = slice(max(0, start_frame - 2), min(clip.num_frames, stop_frame + 2))
    before_values = np.asarray(before_metric.joint_frame_values, dtype=np.float64)
    after_values = np.asarray(after_metric.joint_frame_values, dtype=np.float64)
    before = float(np.max(before_values[support, joint_index], initial=0.0))
    after = float(np.max(after_values[support, joint_index], initial=0.0))
    intervention_frames = np.zeros(clip.num_frames, dtype=np.bool_)
    intervention_frames[start_frame:stop_frame] = envelope > 0.0
    symptom_frames = np.zeros(clip.num_frames, dtype=np.bool_)
    symptom_frames[support] = angle_rad > 0.0
    part = joint_part(clip.skeleton, joint_index)
    channels = tuple(3 + index for index, component in enumerate(axis) if abs(component) > 1.0e-12)
    return build_corruption_result(
        clip,
        corrupted,
        family="joint_pop",
        mechanism="smooth_rotation_pulse",
        interventions=[(joint_index, channels, intervention_frames)],
        symptoms={part: symptom_frames},
        responsibilities={part: (symptom_frames, 1.0, 0.9)},
        metric_name="localized_target_joint_peak_angular_jerk_rad_s3",
        metric_before=before,
        metric_after=after,
        minimum_worsening=1.0e-3,
        evidence_confidence=1.0,
        requested_severity=angle_rad,
        parameters=parameters,
        seed=seed,
        catalog_partition="heldout_eval",
        strict_postcondition=strict_postcondition,
        schema_metadata={"symptom_uses_derivative_support_window": True},
    )


def corrupt_loop_seam_root_velocity(
    clip: MotionClip,
    *,
    seam_window_frames: int,
    velocity_delta_mps: float,
    direction_world: ArrayLike = (0.0, 0.0, 1.0),
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Introduce a terminal root-velocity mismatch as a held-out seam mechanism."""
    if seam_window_frames < 3 or seam_window_frames > clip.num_frames:
        raise ValueError("seam_window_frames must be within [3, number of frames]")
    if not np.isfinite(velocity_delta_mps) or velocity_delta_mps < 0.0:
        raise ValueError("velocity_delta_mps must be finite and nonnegative")
    direction = np.asarray(direction_world, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction_world must be a finite 3D vector")
    length = float(np.linalg.norm(direction))
    if length < 1.0e-8:
        raise ValueError("direction_world must be nonzero")
    direction /= length
    start = clip.num_frames - seam_window_frames
    time = np.arange(seam_window_frames, dtype=np.float64) / clip.fps
    duration = max(time[-1], np.finfo(np.float64).eps)
    displacement = 0.5 * velocity_delta_mps * (time * time / duration)
    root = clip.root_translation_m.copy()
    root[start:] += (displacement[:, None] * direction).astype(np.float32)
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start + 1 :] = velocity_delta_mps > 0.0
    parameters: dict[str, object] = {
        "mechanism": "root_velocity_mismatch",
        "seam_window_frames": seam_window_frames,
        "velocity_delta_mps": velocity_delta_mps,
        "direction_world": direction.tolist(),
    }
    metadata = append_operation_lineage(
        clip,
        {
            **dict(clip.metadata),
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_loop_seam_root_velocity",
            "corruption_seed": seed,
            "corruption_seam_window_frames": seam_window_frames,
            "corruption_velocity_delta_mps": velocity_delta_mps,
        },
        operation="corrupt_loop_seam_root_velocity",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(root_translation_m=root, metadata=metadata)
    before_metric = loop_seam_metric(clip)
    after_metric = loop_seam_metric(corrupted)
    before = float(before_metric.clip_value or 0.0)
    after = float(after_metric.clip_value or 0.0)
    channels = tuple(index for index, component in enumerate(direction) if abs(component) > 1.0e-12)
    return build_corruption_result(
        clip,
        corrupted,
        family="loop_seam",
        mechanism="root_velocity_mismatch",
        interventions=[(clip.skeleton.root_index, channels, affected)],
        symptoms={FunctionalPart.ROOT_PELVIS: affected},
        responsibilities={FunctionalPart.ROOT_PELVIS: (affected, 1.0, 0.95)},
        metric_name="weighted_loop_seam",
        metric_before=before,
        metric_after=after,
        minimum_worsening=1.0e-5,
        evidence_confidence=min(before_metric.confidence, after_metric.confidence),
        requested_severity=velocity_delta_mps,
        parameters=parameters,
        seed=seed,
        catalog_partition="heldout_eval",
        strict_postcondition=strict_postcondition,
        schema_metadata={"symptom_is_seam_window": True},
    )


def corrupt_foot_clearance_ik_lowering(
    clip: MotionClip,
    *,
    side: str,
    start_frame: int,
    stop_frame: int,
    clearance_loss_m: float,
    seed: int = 0,
    source_contacts: ContactResult | None = None,
    ground_plane: ArrayLike | None = None,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Lower a swing ankle with two-bone IK and measure the resulting clearance loss."""
    validate_interval(clip, start_frame, stop_frame)
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not np.isfinite(clearance_loss_m) or clearance_loss_m < 0.0:
        raise ValueError("clearance_loss_m must be finite and nonnegative")
    roles = tuple(f"{side}_{name}" for name in ("hip", "knee", "ankle"))
    missing = [role for role in roles if role not in clip.skeleton.roles]
    if missing:
        raise ValueError(f"foot-clearance corruption requires roles: {missing}")
    hip, knee, ankle = (clip.skeleton.roles[role] for role in roles)
    plane = resolve_ground_plane(clip, ground_plane)
    source = (
        detect_foot_contacts(clip, ground_plane=plane)
        if source_contacts is None
        else source_contacts
    )
    marker_indices = [
        index for index, name in enumerate(source.marker_names) if name.startswith(f"{side}_")
    ]
    if not marker_indices:
        raise ValueError(f"source contacts contain no {side} foot markers")
    envelope = smooth_zero_boundary_envelope(stop_frame - start_frame)
    interval_weight = np.zeros(clip.num_frames, dtype=np.float64)
    interval_weight[start_frame:stop_frame] = envelope
    swing = ~np.any(source.hard[:, marker_indices], axis=1)
    affected = (interval_weight > 0.0) & swing
    if clearance_loss_m > 0.0 and not np.any(affected):
        raise ValueError("foot-clearance interval contains no source swing frames")
    clean_position, _ = forward_kinematics_numpy(
        clip.skeleton, clip.local_quat_wxyz, clip.root_translation_m
    )
    local = clip.local_quat_wxyz.copy()
    for frame in np.flatnonzero(affected):
        target = clean_position[frame, ankle] - (
            clearance_loss_m * interval_weight[frame] * plane[:3]
        )
        solved = solve_two_bone_ik(
            clip.skeleton,
            local[frame],
            clip.root_translation_m[frame],
            hip=hip,
            knee=knee,
            end=ankle,
            target_world_m=target,
            preferred_bend_direction_world=(0.0, 0.0, 1.0),
        )
        local[frame] = solved.local_quat_wxyz
    parameters: dict[str, object] = {
        "mechanism": "swing_ankle_ik_lowering",
        "side": side,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "clearance_loss_m": clearance_loss_m,
    }
    metadata = append_operation_lineage(
        clip,
        {
            **dict(clip.metadata),
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_foot_clearance_ik_lowering",
            "corruption_seed": seed,
            "corruption_interval": [start_frame, stop_frame],
            "corruption_clearance_loss_m": clearance_loss_m,
            "corruption_side": side,
        },
        operation="corrupt_foot_clearance_ik_lowering",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(local_quat_wxyz=local, metadata=metadata)
    contact_supervision, _, _ = measure_contact_supervision(
        clip,
        corrupted,
        source_contacts=source,
        ground_plane=plane,
    )
    marker_names = tuple(source.marker_names[index] for index in marker_indices)
    clean_markers = marker_world_positions(clip, marker_names)
    corrupted_markers = marker_world_positions(corrupted, marker_names)
    clearance_losses: list[np.ndarray] = []
    for marker_name in marker_names:
        clean_height = clean_markers[marker_name] @ plane[:3] + plane[3]
        corrupted_height = corrupted_markers[marker_name] @ plane[:3] + plane[3]
        clearance_losses.append(clean_height[affected] - corrupted_height[affected])
    measured_loss_cm = (
        0.0
        if not clearance_losses or not np.any(affected)
        else 100.0 * float(np.max(np.stack(clearance_losses), initial=0.0))
    )
    part = FunctionalPart.LEFT_LEG if side == "left" else FunctionalPart.RIGHT_LEG
    return build_corruption_result(
        clip,
        corrupted,
        family="foot_clearance",
        mechanism="swing_ankle_ik_lowering",
        interventions=[(hip, (3, 4, 5), affected), (knee, (3, 4, 5), affected)],
        symptoms={part: affected},
        responsibilities={part: (affected, 1.0, 0.9)},
        metric_name="maximum_source_relative_clearance_loss_cm",
        metric_before=0.0,
        metric_after=measured_loss_cm,
        minimum_worsening=0.01,
        evidence_confidence=source.ground_confidence,
        requested_severity=clearance_loss_m,
        parameters=parameters,
        seed=seed,
        catalog_partition="train",
        contact_supervision=contact_supervision,
        strict_postcondition=strict_postcondition,
        schema_metadata={"symptom_is_source_swing_supported": True},
    )


def corrupt_joint_limit_excess(
    clip: MotionClip,
    *,
    joint: JointReference,
    start_frame: int,
    stop_frame: int,
    excess_deg: float,
    seed: int = 0,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Drive one configured joint beyond an authored upper angular limit."""
    validate_interval(clip, start_frame, stop_frame)
    if not np.isfinite(excess_deg) or excess_deg < 0.0:
        raise ValueError("excess_deg must be finite and nonnegative")
    joint_index = _resolve_joint(clip, joint)
    spec = clip.skeleton.joint_limits.get(joint_index)
    if spec is None or not spec.valid or spec.confidence <= 0.0:
        raise ValueError("joint-limit corruption requires a valid authored limit")
    if spec.representation == "euler_xyz":
        candidates = (
            ("x_deg", (1.0, 0.0, 0.0)),
            ("y_deg", (0.0, 1.0, 0.0)),
            ("z_deg", (0.0, 0.0, 1.0)),
        )
    else:
        candidates = (
            ("swing_x_deg", (1.0, 0.0, 0.0)),
            ("swing_z_deg", (0.0, 0.0, 1.0)),
            ("twist_deg", (0.0, 1.0, 0.0)),
        )
    selected = next(
        (
            (name, axis, getattr(spec, name))
            for name, axis in candidates
            if getattr(spec, name) is not None
        ),
        None,
    )
    if selected is None:
        raise ValueError("joint limit does not contain a complete supported channel bound")
    channel_name, axis, bounds = selected
    assert bounds is not None
    target_deg = bounds[1] + excess_deg
    target_relative = axis_angle_to_quaternion(np.deg2rad(target_deg) * np.asarray(axis))
    target_local = quaternion_multiply(
        clip.skeleton.rest_local_quat_wxyz[joint_index], target_relative
    )
    envelope = smooth_zero_boundary_envelope(stop_frame - start_frame)
    target_batch = np.broadcast_to(target_local, (stop_frame - start_frame, 4))
    local = clip.local_quat_wxyz.copy()
    local[start_frame:stop_frame, joint_index] = quaternion_slerp(
        local[start_frame:stop_frame, joint_index], target_batch, envelope
    ).astype(np.float32)
    affected = np.zeros(clip.num_frames, dtype=np.bool_)
    affected[start_frame:stop_frame] = envelope > 0.0
    parameters: dict[str, object] = {
        "mechanism": "authored_limit_excess",
        "joint_index": joint_index,
        "joint_name": clip.skeleton.joint_names[joint_index],
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "channel": channel_name,
        "target_deg": target_deg,
        "excess_deg": excess_deg,
    }
    metadata = append_operation_lineage(
        clip,
        {
            **dict(clip.metadata),
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_joint_limit_excess",
            "corruption_seed": seed,
            "corruption_interval": [start_frame, stop_frame],
            "corruption_joint": clip.skeleton.joint_names[joint_index],
            "corruption_excess_deg": excess_deg,
        },
        operation="corrupt_joint_limit_excess",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(local_quat_wxyz=local, metadata=metadata)
    before_metric = joint_limit_metric(clip)
    after_metric = joint_limit_metric(corrupted)
    if before_metric.clip_value is None or after_metric.clip_value is None:
        raise ValueError("joint-limit metric is unavailable for this rig")
    part = joint_part(clip.skeleton, joint_index)
    return build_corruption_result(
        clip,
        corrupted,
        family="joint_limit",
        mechanism="authored_limit_excess",
        interventions=[(joint_index, (3, 4, 5), affected)],
        symptoms={part: affected},
        responsibilities={part: (affected, 1.0, spec.confidence)},
        metric_name="maximum_joint_limit_excess_deg",
        metric_before=float(before_metric.clip_value),
        metric_after=float(after_metric.clip_value),
        minimum_worsening=0.01,
        evidence_confidence=min(before_metric.confidence, after_metric.confidence),
        requested_severity=excess_deg,
        parameters=parameters,
        seed=seed,
        catalog_partition="train",
        strict_postcondition=strict_postcondition,
        schema_metadata={"symptom_uses_authored_joint_limit": True},
    )
