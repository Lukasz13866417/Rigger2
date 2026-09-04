"""Shared measurement and dense-label helpers for procedural corruptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.corruptions.base import (
    CORRUPTION_CHANNEL_NAMES,
    CORRUPTION_DEFECT_NAMES,
    CORRUPTION_PART_NAMES,
    ContactSupervision,
    CorruptionResult,
    IneffectiveCorruptionError,
    MeasuredPostcondition,
)
from motionlab.kinematics.fk import marker_world_positions
from motionlab.math.finite_difference import finite_difference
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton
from motionlab.processing.contacts import (
    ContactResult,
    detect_foot_contacts,
    normalize_ground_plane,
)
from motionlab.rigging.semantic_roles import FunctionalPart, functional_part_for_role


def smooth_zero_boundary_envelope(length: int) -> NDArray[np.float64]:
    """Return a deterministic sin-squared pulse with exact zero endpoints."""
    if length < 3:
        raise ValueError("a smooth corruption interval must contain at least three frames")
    phase = np.linspace(0.0, np.pi, length, dtype=np.float64)
    envelope = np.sin(phase) ** 2
    envelope[[0, -1]] = 0.0
    return envelope


def validate_interval(clip: MotionClip, start_frame: int, stop_frame: int) -> None:
    """Validate one nontrivial half-open corruption interval."""
    if start_frame < 0 or stop_frame > clip.num_frames or stop_frame - start_frame < 3:
        raise ValueError("corruption interval must contain at least three valid frames")


def resolve_ground_plane(clip: MotionClip, supplied: ArrayLike | None) -> NDArray[np.float64]:
    """Resolve an explicit or clip-authored ground plane."""
    if supplied is not None:
        return normalize_ground_plane(supplied)
    authored = clip.metadata.get("ground_plane", (0.0, 1.0, 0.0, 0.0))
    return normalize_ground_plane(authored)


def _fixed_contact_evidence(clip: MotionClip, source: ContactResult) -> ContactResult:
    """Recompute physical evidence while retaining the privileged source contact mask."""
    markers = marker_world_positions(clip, source.marker_names)
    positions = np.stack([markers[name] for name in source.marker_names], axis=1)
    plane = normalize_ground_plane(source.ground_plane)
    signed_height = positions @ plane[:3] + plane[3]
    velocity = finite_difference(positions, 1.0 / clip.fps, time_axis=0)
    normal_velocity = np.sum(velocity * plane[:3], axis=-1, keepdims=True) * plane[:3]
    tangential_speed = np.linalg.norm(velocity - normal_velocity, axis=-1)
    return ContactResult(
        marker_names=source.marker_names,
        hard=source.hard,
        confidence=source.confidence,
        height_m=signed_height.astype(np.float32),
        tangential_speed_mps=tangential_speed.astype(np.float32),
        ground_plane=plane.astype(np.float32),
        source=f"fixed_source_truth:{source.source}",
        ground_confidence=source.ground_confidence,
        metadata={"truth_source": source.source},
    )


def measure_contact_supervision(
    clean: MotionClip,
    corrupted: MotionClip,
    *,
    source_contacts: ContactResult | None = None,
    ground_plane: ArrayLike | None = None,
) -> tuple[ContactSupervision, ContactResult, ContactResult]:
    """Build separate source-truth and inference contact views for a motion pair."""
    plane = resolve_ground_plane(clean, ground_plane)
    source = (
        detect_foot_contacts(clean, ground_plane=plane)
        if source_contacts is None
        else source_contacts
    )
    if source.hard.shape[0] != clean.num_frames:
        raise ValueError("source contact frame count must match the clean clip")
    if not np.allclose(source.ground_plane, plane, atol=1.0e-6, rtol=0.0):
        raise ValueError("source contacts and requested ground plane disagree")
    fixed_clean = _fixed_contact_evidence(clean, source)
    fixed_corrupted = _fixed_contact_evidence(corrupted, source)
    inferred_clean = detect_foot_contacts(clean, ground_plane=plane)
    inferred_corrupted = detect_foot_contacts(corrupted, ground_plane=plane)
    supervision = ContactSupervision(
        marker_names=source.marker_names,
        truth_hard=source.hard,
        truth_confidence=source.confidence,
        inferred_clean_hard=inferred_clean.hard,
        inferred_clean_confidence=inferred_clean.confidence,
        inferred_corrupted_hard=inferred_corrupted.hard,
        inferred_corrupted_confidence=inferred_corrupted.confidence,
        ground_plane=plane.astype(np.float32),
        truth_source=(
            f"clean_inference_proxy:{source.source}"
            if source_contacts is None
            else f"provided_source_truth:{source.source}"
        ),
        inference_source=inferred_corrupted.source,
        ground_confidence=source.ground_confidence,
    )
    return supervision, fixed_clean, fixed_corrupted


def part_slot(part: FunctionalPart) -> int:
    """Map the public one-based anatomical enum into the dense eight-part schema."""
    normalized = FunctionalPart.OTHER_ACCESSORY if part == FunctionalPart.UNKNOWN else part
    return int(normalized) - 1


def joint_part(skeleton: Skeleton, joint: int) -> FunctionalPart:
    """Resolve a joint's authored role to a functional part without name guessing."""
    if joint < 0 or joint >= skeleton.num_joints:
        raise ValueError("joint index is out of range")
    parts = {
        functional_part_for_role(role)
        for role, role_joint in skeleton.roles.items()
        if role_joint == joint
    }
    if not parts:
        return FunctionalPart.OTHER_ACCESSORY
    if len(parts) != 1:
        raise ValueError("authored aliases for one joint map to different functional parts")
    return next(iter(parts))


def contact_part_frames(
    contacts: ContactResult,
    interval_mask: NDArray[np.bool_],
) -> dict[FunctionalPart, NDArray[np.bool_]]:
    """Return left/right leg symptom frames supported by source contact truth."""
    if interval_mask.shape != (contacts.hard.shape[0],):
        raise ValueError("interval mask and contacts must have matching frame counts")
    result: dict[FunctionalPart, NDArray[np.bool_]] = {}
    for side, part in (
        ("left", FunctionalPart.LEFT_LEG),
        ("right", FunctionalPart.RIGHT_LEG),
    ):
        indices = [
            index
            for index, marker_name in enumerate(contacts.marker_names)
            if marker_name.startswith(f"{side}_")
        ]
        if indices:
            active = interval_mask & np.any(contacts.hard[:, indices], axis=1)
            if np.any(active):
                result[part] = active
    return result


def dense_supervision_masks(
    clip: MotionClip,
    *,
    family: str,
    interventions: Sequence[tuple[int, Sequence[int], NDArray[np.bool_]]],
    symptoms: Mapping[FunctionalPart, NDArray[np.bool_]],
    responsibilities: Mapping[FunctionalPart, tuple[NDArray[np.bool_], float, float]],
) -> tuple[
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Build the distinct intervention, symptom, and responsibility tensors."""
    try:
        defect = CORRUPTION_DEFECT_NAMES.index(family)
    except ValueError as exc:
        raise ValueError(f"unknown corruption family {family!r}") from exc
    intervention = np.zeros(
        (clip.num_frames, clip.num_joints, len(CORRUPTION_CHANNEL_NAMES)),
        dtype=np.bool_,
    )
    symptom = np.zeros(
        (clip.num_frames, len(CORRUPTION_PART_NAMES), len(CORRUPTION_DEFECT_NAMES)),
        dtype=np.bool_,
    )
    responsibility = np.zeros_like(symptom, dtype=np.float32)
    responsibility_confidence = np.zeros_like(symptom, dtype=np.float32)
    for joint, channels, frames in interventions:
        if frames.shape != (clip.num_frames,):
            raise ValueError("intervention frame masks must have shape [T]")
        if joint < 0 or joint >= clip.num_joints:
            raise ValueError("intervention joint index is out of range")
        if any(channel < 0 or channel >= len(CORRUPTION_CHANNEL_NAMES) for channel in channels):
            raise ValueError("intervention channel index is out of range")
        for channel in channels:
            intervention[frames, joint, channel] = True
    for part, frames in symptoms.items():
        if frames.shape != (clip.num_frames,):
            raise ValueError("symptom frame masks must have shape [T]")
        symptom[frames, part_slot(part), defect] = True
    for part, (frames, target, confidence) in responsibilities.items():
        if frames.shape != (clip.num_frames,):
            raise ValueError("responsibility frame masks must have shape [T]")
        if not 0.0 <= target <= 1.0 or not 0.0 <= confidence <= 1.0:
            raise ValueError("responsibility values and confidence must be within [0,1]")
        responsibility[frames, part_slot(part), defect] = target
        responsibility_confidence[frames, part_slot(part), defect] = confidence
    return intervention, symptom, responsibility, responsibility_confidence


def verify_worsening(
    *,
    metric_name: str,
    before: float,
    after: float,
    minimum_worsening: float,
    evidence_confidence: float,
    requested_active: bool,
    strict: bool,
) -> tuple[MeasuredPostcondition, float, float]:
    """Validate a candidate hard negative and derive measured severity/confidence."""
    if not all(np.isfinite(value) for value in (before, after, evidence_confidence)):
        raise ValueError("corruption measurements must be finite")
    if not 0.0 <= evidence_confidence <= 1.0:
        raise ValueError("evidence confidence must be within [0,1]")
    passed = after - before >= minimum_worsening
    hard_negative = requested_active and passed and evidence_confidence > 0.0
    postcondition = MeasuredPostcondition(
        metric_name=metric_name,
        before=before,
        after=after,
        minimum_worsening=minimum_worsening,
        passed=passed,
        hard_negative=hard_negative,
    )
    if requested_active and strict and not hard_negative:
        raise IneffectiveCorruptionError(
            f"{metric_name} did not worsen by the required {minimum_worsening:g}: "
            f"before={before:g}, after={after:g}"
        )
    measured_severity = max(0.0, postcondition.delta)
    if not hard_negative:
        return postcondition, measured_severity, 0.0
    margin_confidence = min(
        1.0,
        measured_severity / max(abs(after), minimum_worsening, np.finfo(np.float64).eps),
    )
    return postcondition, measured_severity, evidence_confidence * margin_confidence


def build_corruption_result(
    clean: MotionClip,
    corrupted: MotionClip,
    *,
    family: str,
    mechanism: str,
    interventions: Sequence[tuple[int, Sequence[int], NDArray[np.bool_]]],
    symptoms: Mapping[FunctionalPart, NDArray[np.bool_]],
    responsibilities: Mapping[FunctionalPart, tuple[NDArray[np.bool_], float, float]],
    metric_name: str,
    metric_before: float,
    metric_after: float,
    minimum_worsening: float,
    evidence_confidence: float,
    requested_severity: float,
    parameters: Mapping[str, object],
    seed: int,
    catalog_partition: str,
    contact_supervision: ContactSupervision | None = None,
    hard_negative_candidate: bool = True,
    strict_postcondition: bool = True,
    schema_metadata: Mapping[str, object] | None = None,
) -> CorruptionResult:
    """Assemble one measured result using the shared dense schema and lineage rules."""
    postcondition, measured_severity, preference_confidence = verify_worsening(
        metric_name=metric_name,
        before=metric_before,
        after=metric_after,
        minimum_worsening=minimum_worsening,
        evidence_confidence=evidence_confidence,
        requested_active=requested_severity > 0.0 and hard_negative_candidate,
        strict=strict_postcondition and hard_negative_candidate,
    )
    intervention, symptom, responsibility, responsibility_confidence = dense_supervision_masks(
        clean,
        family=family,
        interventions=interventions,
        symptoms=symptoms,
        responsibilities=responsibilities,
    )
    from motionlab.core.provenance import provenance_from_clip

    provenance = provenance_from_clip(clean)
    metadata = {
        "intervention_is_exact": True,
        "fine_joint_causal_blame_is_supervised": False,
        **({} if schema_metadata is None else dict(schema_metadata)),
    }
    return CorruptionResult(
        corrupted_motion=corrupted,
        clean_motion=clean,
        corruption_family=family,
        corruption_mechanism=mechanism,
        intervention_mask=intervention,
        symptom_mask=symptom,
        responsibility_target=responsibility,
        responsibility_confidence=responsibility_confidence,
        measured_metrics_before={metric_name: metric_before},
        measured_metrics_after={metric_name: metric_after},
        measured_severity=measured_severity,
        preference_confidence=preference_confidence,
        source_motion_id=clean.content_hash,
        corruption_seed=seed,
        split_lineage_id=provenance.split_lineage_id,
        requested_severity_parameter=requested_severity,
        generation_parameters=parameters,
        postcondition=postcondition,
        contact_supervision=contact_supervision,
        catalog_partition=catalog_partition,
        schema_metadata=metadata,
    )
