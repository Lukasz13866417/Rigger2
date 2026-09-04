"""Measured foot-slide corruptions with training and held-out mechanisms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from motionlab.core.provenance import append_operation_lineage, provenance_from_clip
from motionlab.corruptions._supervision import (
    contact_part_frames,
    dense_supervision_masks,
    measure_contact_supervision,
    smooth_zero_boundary_envelope,
    validate_interval,
    verify_worsening,
)
from motionlab.corruptions.base import CorruptionResult
from motionlab.math.quaternion import axis_angle_to_quaternion, quaternion_multiply
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult
from motionlab.rigging.semantic_roles import FunctionalPart


def _foot_slide_result(
    clip: MotionClip,
    corrupted: MotionClip,
    *,
    mechanism: str,
    interval_frames: np.ndarray,
    interventions: list[tuple[int, tuple[int, ...], np.ndarray]],
    requested_severity: float,
    parameters: dict[str, object],
    seed: int,
    catalog_partition: str,
    source_contacts: ContactResult | None,
    ground_plane: ArrayLike | None,
    strict_postcondition: bool,
) -> CorruptionResult:
    contacts, fixed_clean, fixed_corrupted = measure_contact_supervision(
        clip,
        corrupted,
        source_contacts=source_contacts,
        ground_plane=ground_plane,
    )
    from motionlab.metrics.foot_sliding import foot_sliding_metric

    before_metric = foot_sliding_metric(clip, fixed_clean)
    after_metric = foot_sliding_metric(corrupted, fixed_corrupted)
    before = float(before_metric.clip_value or 0.0)
    after = float(after_metric.clip_value or 0.0)
    postcondition, measured_severity, preference_confidence = verify_worsening(
        metric_name="foot_sliding_cm_per_contact_interval",
        before=before,
        after=after,
        minimum_worsening=0.01,
        evidence_confidence=min(before_metric.confidence, after_metric.confidence),
        requested_active=requested_severity > 0.0,
        strict=strict_postcondition,
    )
    symptoms = contact_part_frames(fixed_clean, interval_frames)
    responsibilities: dict[
        FunctionalPart,
        tuple[np.ndarray, float, float],
    ] = {
        FunctionalPart.ROOT_PELVIS: (interval_frames, 0.5, 0.65),
        **{part: (frames, 1.0, 0.9) for part, frames in symptoms.items()},
    }
    intervention, symptom, responsibility, responsibility_confidence = dense_supervision_masks(
        clip,
        family="foot_slide",
        interventions=interventions,
        symptoms=symptoms,
        responsibilities=responsibilities,
    )
    provenance = provenance_from_clip(clip)
    return CorruptionResult(
        corrupted_motion=corrupted,
        clean_motion=clip,
        corruption_family="foot_slide",
        corruption_mechanism=mechanism,
        intervention_mask=intervention,
        symptom_mask=symptom,
        responsibility_target=responsibility,
        responsibility_confidence=responsibility_confidence,
        measured_metrics_before={"foot_sliding_cm_per_contact_interval": before},
        measured_metrics_after={"foot_sliding_cm_per_contact_interval": after},
        measured_severity=measured_severity,
        preference_confidence=preference_confidence,
        source_motion_id=clip.content_hash,
        corruption_seed=seed,
        split_lineage_id=provenance.split_lineage_id,
        requested_severity_parameter=requested_severity,
        generation_parameters=parameters,
        postcondition=postcondition,
        contact_supervision=contacts,
        catalog_partition=catalog_partition,
        schema_metadata={
            "intervention_is_exact": True,
            "symptom_is_contact_supported": True,
            "responsibility_is_functional_part_supervision": True,
            "fine_joint_causal_blame_is_supervised": False,
        },
    )


def corrupt_foot_slide_root_drift(
    clip: MotionClip,
    *,
    start_frame: int,
    stop_frame: int,
    distance_m: float,
    direction_world: ArrayLike = (1.0, 0.0, 0.0),
    seed: int = 0,
    source_contacts: ContactResult | None = None,
    ground_plane: ArrayLike | None = None,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Add a zero-boundary root drift pulse without compensating planted feet."""
    validate_interval(clip, start_frame, stop_frame)
    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("distance_m must be finite and nonnegative")
    direction = np.asarray(direction_world, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction_world must be a finite 3D vector")
    length = np.linalg.norm(direction)
    if length < 1.0e-8:
        raise ValueError("direction_world must be nonzero")
    direction /= length

    interval_frames = stop_frame - start_frame
    envelope = smooth_zero_boundary_envelope(interval_frames)
    displacement = envelope[:, None] * distance_m * direction
    root = clip.root_translation_m.copy()
    root[start_frame:stop_frame] += displacement.astype(np.float32)
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_foot_slide_root_drift",
            "corruption_seed": seed,
            "corruption_interval": [start_frame, stop_frame],
            "corruption_distance_m": distance_m,
            "corruption_direction_world": direction.tolist(),
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="corrupt_foot_slide_root_drift",
        kind="corruption",
        parameters={
            "mechanism": "root_drift",
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "distance_m": distance_m,
            "direction_world": direction.tolist(),
        },
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(root_translation_m=root, metadata=metadata)
    affected_frames = np.zeros(clip.num_frames, dtype=np.bool_)
    affected_frames[start_frame:stop_frame] = envelope > 0.0
    parameters: dict[str, object] = {
        "mechanism": "root_drift",
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "distance_m": distance_m,
        "direction_world": direction.tolist(),
    }
    channels = tuple(index for index, component in enumerate(direction) if abs(component) > 1.0e-12)
    return _foot_slide_result(
        clip,
        corrupted,
        mechanism="root_drift",
        interval_frames=affected_frames,
        interventions=[(clip.skeleton.root_index, channels, affected_frames)],
        requested_severity=distance_m,
        parameters=parameters,
        seed=seed,
        catalog_partition="train",
        source_contacts=source_contacts,
        ground_plane=ground_plane,
        strict_postcondition=strict_postcondition,
    )


def corrupt_foot_slide_hip_rotation_drift(
    clip: MotionClip,
    *,
    side: str,
    start_frame: int,
    stop_frame: int,
    angle_rad: float,
    axis_local: ArrayLike = (1.0, 0.0, 0.0),
    seed: int = 0,
    source_contacts: ContactResult | None = None,
    ground_plane: ArrayLike | None = None,
    strict_postcondition: bool = True,
) -> CorruptionResult:
    """Rotate a stance hip smoothly; reserved as a held-out foot-slide mechanism."""
    validate_interval(clip, start_frame, stop_frame)
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not np.isfinite(angle_rad) or angle_rad < 0.0:
        raise ValueError("angle_rad must be finite and nonnegative")
    role = f"{side}_hip"
    if role not in clip.skeleton.roles:
        raise ValueError(f"foot-slide hip corruption requires role {role!r}")
    hip = clip.skeleton.roles[role]
    axis = np.asarray(axis_local, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axis_local must be a finite 3D vector")
    axis_length = float(np.linalg.norm(axis))
    if axis_length < 1.0e-8:
        raise ValueError("axis_local must be nonzero")
    axis /= axis_length
    envelope = smooth_zero_boundary_envelope(stop_frame - start_frame)
    tangent = envelope[:, None] * angle_rad * axis
    delta = axis_angle_to_quaternion(tangent)
    local = clip.local_quat_wxyz.copy()
    local[start_frame:stop_frame, hip] = quaternion_multiply(
        local[start_frame:stop_frame, hip],
        delta,
    ).astype(np.float32)
    metadata = dict(clip.metadata)
    parameters: dict[str, object] = {
        "mechanism": "hip_rotation_drift",
        "side": side,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "angle_rad": angle_rad,
        "axis_local": axis.tolist(),
    }
    metadata.update(
        {
            "parent_motion_id": clip.content_hash,
            "operation": "corrupt_foot_slide_hip_rotation_drift",
            "corruption_seed": seed,
            "corruption_interval": [start_frame, stop_frame],
            "corruption_angle_rad": angle_rad,
            "corruption_axis_local": axis.tolist(),
            "corruption_side": side,
        }
    )
    metadata = append_operation_lineage(
        clip,
        metadata,
        operation="corrupt_foot_slide_hip_rotation_drift",
        kind="corruption",
        parameters=parameters,
        version="motionlab-0.1.0",
        seed=seed,
    )
    corrupted = clip.with_updates(local_quat_wxyz=local, metadata=metadata)
    affected_frames = np.zeros(clip.num_frames, dtype=np.bool_)
    affected_frames[start_frame:stop_frame] = envelope > 0.0
    channels = tuple(3 + index for index, component in enumerate(axis) if abs(component) > 1.0e-12)
    return _foot_slide_result(
        clip,
        corrupted,
        mechanism="hip_rotation_drift",
        interval_frames=affected_frames,
        interventions=[(hip, channels, affected_frames)],
        requested_severity=angle_rad,
        parameters=parameters,
        seed=seed,
        catalog_partition="heldout_eval",
        source_contacts=source_contacts,
        ground_plane=ground_plane,
        strict_postcondition=strict_postcondition,
    )
