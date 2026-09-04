"""Leakage-safe fixed-rig feature and target construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motionlab.core.provenance import provenance_from_clip
from motionlab.corruptions._supervision import joint_part, part_slot
from motionlab.corruptions.base import (
    CORRUPTION_CHANNEL_NAMES,
    CORRUPTION_DEFECT_NAMES,
    CORRUPTION_PART_NAMES,
    CorruptionResult,
)
from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.finite_difference import finite_difference
from motionlab.math.quaternion import (
    quaternion_geodesic_distance,
    quaternion_interval_angular_velocity,
    quaternion_rotate_vector,
    quaternion_to_matrix,
)
from motionlab.math.rotation6d import matrix_to_rotation_6d
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.motion._validation import frozen_json_mapping
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import FOOT_MARKER_NAMES, detect_foot_contacts
from motionlab.processing.phase import estimate_gait_phase

DATASET_SAMPLE_VERSION = "motionlab.fixed_rig_sample.v4"

NEURAL_INPUT_FIELDS = frozenset(
    {
        "local_rot6d",
        "root_translation",
        "root_velocity",
        "root_angular_velocity",
        "joint_position_rel",
        "joint_velocity_rel",
        "joint_angular_velocity",
        "contact_soft",
        "contact_valid",
        "phase_sincos",
        "phase_valid",
        "target_speed",
        "style_id",
        "loop_mode",
        "joint_functional_part",
    }
)

TARGET_FIELDS = frozenset(
    {
        "intervention_mask",
        "symptom_mask",
        "responsibility_target",
        "responsibility_confidence",
        "defect_mask",
        "joint_defect_mask",
        "defect_present",
        "preference_valid",
        "preference_clean_better",
        "is_clean",
        "no_target_defect",
        "approximately_equal_quality",
        "quality_equivalence_valid",
        "clean_local_quat",
        "clean_root_translation",
    }
)

FORBIDDEN_ARRAY_FIELDS = frozenset(
    {
        "source_dataset",
        "source_clip_id",
        "source_take_id",
        "source_path",
        "corruption_family",
        "corruption_mechanism",
        "corruption_seed",
        "requested_severity_parameter",
        "split",
        "split_lineage_id",
        "counterfactual_group_id",
        "catalog_mechanism_key",
        "gait_target",
        "gait_event_labels",
        "gait_phase_bins",
        "sample_role",
        "deterministic_baseline_metrics",
        "equivalent_transform",
    }
)


@dataclass(frozen=True)
class TrainingSample:
    """One fixed-length feature/target record with metadata kept outside neural arrays."""

    sample_id: str
    arrays: Mapping[str, NDArray[Any]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.sample_id.startswith("sample:sha256:"):
            raise ValueError("sample_id must be content addressed")
        keys = set(self.arrays)
        if keys != NEURAL_INPUT_FIELDS | TARGET_FIELDS:
            missing = sorted((NEURAL_INPUT_FIELDS | TARGET_FIELDS).difference(keys))
            unexpected = sorted(keys.difference(NEURAL_INPUT_FIELDS | TARGET_FIELDS))
            raise ValueError(
                f"sample array fields mismatch; missing={missing}, unexpected={unexpected}"
            )
        if keys & FORBIDDEN_ARRAY_FIELDS:
            raise ValueError("metadata-only fields cannot be persisted as neural arrays")
        arrays: dict[str, NDArray[Any]] = {}
        for name, raw in self.arrays.items():
            value = np.array(raw, copy=True, order="C")
            if value.dtype == np.dtype(object):
                raise ValueError(f"sample array {name!r} cannot have object dtype")
            if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise ValueError(f"sample array {name!r} contains non-finite values")
            value.setflags(write=False)
            arrays[name] = value
        time = arrays["local_rot6d"].shape[0]
        joints = arrays["local_rot6d"].shape[1]
        if arrays["local_rot6d"].shape != (time, joints, 6):
            raise ValueError("local_rot6d must have shape [T,J,6]")
        expected_time_fields = {
            "root_translation": (time, 3),
            "root_velocity": (time, 3),
            "root_angular_velocity": (time, 3),
            "joint_position_rel": (time, joints, 3),
            "joint_velocity_rel": (time, joints, 3),
            "joint_angular_velocity": (time, joints, 3),
            "contact_soft": (time, len(FOOT_MARKER_NAMES)),
            "contact_valid": (time, len(FOOT_MARKER_NAMES)),
            "phase_sincos": (time, 2),
            "phase_valid": (time,),
            "intervention_mask": (time, joints, len(CORRUPTION_CHANNEL_NAMES)),
            "symptom_mask": (time, len(CORRUPTION_PART_NAMES), len(CORRUPTION_DEFECT_NAMES)),
            "responsibility_target": (
                time,
                len(CORRUPTION_PART_NAMES),
                len(CORRUPTION_DEFECT_NAMES),
            ),
            "responsibility_confidence": (
                time,
                len(CORRUPTION_PART_NAMES),
                len(CORRUPTION_DEFECT_NAMES),
            ),
            "defect_mask": (time, len(CORRUPTION_DEFECT_NAMES)),
            "joint_defect_mask": (time, joints, len(CORRUPTION_DEFECT_NAMES)),
            "clean_local_quat": (time, joints, 4),
            "clean_root_translation": (time, 3),
            "joint_functional_part": (joints,),
        }
        for name, shape in expected_time_fields.items():
            if arrays[name].shape != shape:
                raise ValueError(f"sample array {name!r} must have shape {shape}")
        if arrays["defect_present"].shape != (len(CORRUPTION_DEFECT_NAMES),):
            raise ValueError("defect_present must have shape [number of defects]")
        for name in (
            "target_speed",
            "style_id",
            "loop_mode",
            "preference_valid",
            "preference_clean_better",
            "is_clean",
            "no_target_defect",
            "approximately_equal_quality",
            "quality_equivalence_valid",
        ):
            if arrays[name].shape != (1,):
                raise ValueError(f"sample array {name!r} must have shape [1]")
        metadata = frozen_json_mapping(self.metadata, name="training sample metadata")
        if metadata.get("sample_id") != self.sample_id:
            raise ValueError("sample metadata ID does not match sample_id")
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "metadata", metadata)

    @property
    def neural_inputs(self) -> Mapping[str, NDArray[Any]]:
        """Return only the audited neural-input whitelist, excluding targets and metadata."""
        return MappingProxyType({name: self.arrays[name] for name in sorted(NEURAL_INPUT_FIELDS)})

    @property
    def targets(self) -> Mapping[str, NDArray[Any]]:
        """Return supervision arrays separately from model inputs."""
        return MappingProxyType({name: self.arrays[name] for name in sorted(TARGET_FIELDS)})


def _angular_velocity(local_quat: NDArray[np.float32], fps: float) -> NDArray[np.float32]:
    if local_quat.shape[0] < 2:
        return np.zeros((*local_quat.shape[:2], 3), dtype=np.float32)
    interval = quaternion_interval_angular_velocity(
        local_quat,
        np.arange(local_quat.shape[0], dtype=np.float64) / fps,
        time_axis=0,
    )
    result = np.empty((*local_quat.shape[:2], 3), dtype=np.float32)
    result[:-1] = interval.astype(np.float32)
    result[-1] = result[-2]
    return result


def _inverse_yaw_rotate(
    value: NDArray[np.float64], yaw: NDArray[np.float64]
) -> NDArray[np.float64]:
    result = value.copy()
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    x = value[..., 0]
    z = value[..., 2]
    while cosine.ndim < x.ndim:
        cosine = cosine[..., None]
        sine = sine[..., None]
    result[..., 0] = cosine * x - sine * z
    result[..., 2] = sine * x + cosine * z
    return result


def motion_feature_arrays(clip: MotionClip) -> dict[str, NDArray[Any]]:
    """Encode only inference-safe per-frame arrays for a motion clip."""
    position, global_rotation = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    root = clip.skeleton.root_index
    forward = quaternion_rotate_vector(global_rotation[:, root], (0.0, 0.0, 1.0))
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    relative = position - position[:, root : root + 1]
    relative[..., 1] = position[..., 1]
    facing_relative = _inverse_yaw_rotate(relative, yaw)
    root_feature = clip.root_translation_m.astype(np.float64).copy()
    root_feature[:, (0, 2)] -= root_feature[0, (0, 2)]
    root_velocity = finite_difference(root_feature, 1.0 / clip.fps, time_axis=0)
    joint_velocity = finite_difference(facing_relative, 1.0 / clip.fps, time_axis=0)
    angular_velocity = _angular_velocity(clip.local_quat_wxyz, clip.fps)
    root_angular_velocity = angular_velocity[:, root]

    contact_soft = np.zeros((clip.num_frames, len(FOOT_MARKER_NAMES)), dtype=np.float32)
    contact_valid = np.zeros_like(contact_soft, dtype=np.bool_)
    phase_sincos = np.zeros((clip.num_frames, 2), dtype=np.float32)
    phase_sincos[:, 1] = 1.0
    phase_valid = np.zeros(clip.num_frames, dtype=np.bool_)
    try:
        contacts = detect_foot_contacts(clip)
        contact_soft = contacts.confidence.copy()
        contact_valid[:] = True
        phase = estimate_gait_phase(contacts)
        phase_sincos = phase.phase_sincos.copy()
        phase_valid = phase.valid.copy()
    except ValueError:
        pass

    target_raw = clip.metadata.get("expected_speed_mps")
    target_speed = (
        float(np.mean(np.linalg.norm(root_velocity[:, (0, 2)], axis=-1)))
        if target_raw is None
        else float(target_raw)
    )
    style_raw = clip.metadata.get("style_id", -1)
    style_id = int(style_raw) if isinstance(style_raw, int) else -1
    loop_mode = {"root_motion": 1, "in_place": 2}.get(str(clip.metadata.get("loop_kind")), 0)
    functional_part = np.asarray(
        [part_slot(joint_part(clip.skeleton, joint)) for joint in range(clip.num_joints)],
        dtype=np.int8,
    )
    return {
        "local_rot6d": matrix_to_rotation_6d(quaternion_to_matrix(clip.local_quat_wxyz)).astype(
            np.float32
        ),
        "root_translation": root_feature.astype(np.float32),
        "root_velocity": root_velocity.astype(np.float32),
        "root_angular_velocity": root_angular_velocity,
        "joint_position_rel": facing_relative.astype(np.float32),
        "joint_velocity_rel": joint_velocity.astype(np.float32),
        "joint_angular_velocity": angular_velocity,
        "contact_soft": contact_soft,
        "contact_valid": contact_valid,
        "phase_sincos": phase_sincos,
        "phase_valid": phase_valid,
        "target_speed": np.asarray([target_speed], dtype=np.float32),
        "style_id": np.asarray([style_id], dtype=np.int64),
        "loop_mode": np.asarray([loop_mode], dtype=np.int64),
        "joint_functional_part": functional_part,
    }


def _sample_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"sample:sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _base_metadata(
    clean: MotionClip,
    motion: MotionClip,
    *,
    source_window_start: int,
    corruption_family: str | None,
    corruption_mechanism: str | None,
    corruption_seed: int | None,
    requested_severity_parameter: float | None,
    measured_severity: float | None,
    preference_confidence: float,
    catalog_partition: str,
    extra: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    provenance = provenance_from_clip(clean)
    identity = {
        "format_version": DATASET_SAMPLE_VERSION,
        "source_motion_id": clean.content_hash,
        "motion_id": motion.content_hash,
        "split_lineage_id": provenance.split_lineage_id,
        "source_window_start": source_window_start,
        "corruption_family": corruption_family,
        "corruption_mechanism": corruption_mechanism,
        "corruption_seed": corruption_seed,
        "requested_severity_parameter": requested_severity_parameter,
    }
    sample_id = _sample_id(identity)
    target_speed_raw = motion.metadata.get("expected_speed_mps")
    target_speed = float(target_speed_raw) if target_speed_raw is not None else None
    deterministic = grade_motion_deterministic(
        motion,
        target_speed_mps=target_speed,
    )
    speed_metadata = deterministic.metrics["root_speed"].metadata
    deterministic_baseline_metrics = {
        "foot_sliding_cm": deterministic.measured["maximum_stance_slip_cm"],
        "ground_penetration_cm": deterministic.measured["maximum_ground_penetration_cm"],
        "floating_contact_cm": deterministic.measured["maximum_floating_contact_cm"],
        "loop_seam": deterministic.measured["loop_seam"],
        "angular_smoothness_rad_s3_p99": deterministic.metrics["angular_smoothness"].clip_value,
        "absolute_target_speed_error_mps": speed_metadata["absolute_target_error_mps"],
        "average_speed_mps": deterministic.measured["average_speed_mps"],
        "cadence_steps_per_minute": deterministic.measured["cadence_steps_per_minute"],
        "stride_time_s": deterministic.measured["stride_time_s"],
        "stride_length_m": deterministic.measured["stride_length_m"],
        "step_width_m": deterministic.measured["step_width_m"],
        "gait_asymmetry_percent": deterministic.measured["gait_asymmetry_percent"],
    }
    metadata = {
        **identity,
        "sample_id": sample_id,
        "source_dataset": provenance.source_dataset,
        "source_clip_id": provenance.source_clip_id,
        "source_take_id": provenance.source_take_id,
        "split": provenance.split,
        "clean_confidence": provenance.clean_confidence,
        "catalog_partition": catalog_partition,
        "measured_severity": measured_severity,
        "preference_confidence": preference_confidence,
        # Persisted for separated baselines only. This metadata is forbidden from neural arrays.
        "deterministic_baseline_metrics": deterministic_baseline_metrics,
        "neural_input_fields": sorted(NEURAL_INPUT_FIELDS),
        "metadata_only_fields": sorted(FORBIDDEN_ARRAY_FIELDS),
        "facing_frame": "per_frame_root_yaw_removed; vertical positions remain world-height",
        **({} if extra is None else dict(extra)),
    }
    return sample_id, metadata


def _target_arrays(
    clean: MotionClip,
    *,
    intervention: NDArray[np.bool_],
    symptom: NDArray[np.bool_],
    responsibility: NDArray[np.float32],
    responsibility_confidence: NDArray[np.float32],
    preference_valid: bool,
    is_clean: bool,
    no_target_defect: bool,
    approximately_equal_quality: bool,
    quality_equivalence_valid: bool,
) -> dict[str, NDArray[Any]]:
    defect_mask = np.any(symptom, axis=1).astype(np.float32)
    defect_present = np.any(defect_mask > 0.0, axis=0).astype(np.float32)
    joint_defect = np.empty(
        (clean.num_frames, clean.num_joints, len(CORRUPTION_DEFECT_NAMES)),
        dtype=np.float32,
    )
    for joint in range(clean.num_joints):
        joint_defect[:, joint] = responsibility[:, part_slot(joint_part(clean.skeleton, joint))]
    return {
        "intervention_mask": intervention,
        "symptom_mask": symptom,
        "responsibility_target": responsibility,
        "responsibility_confidence": responsibility_confidence,
        "defect_mask": defect_mask,
        "joint_defect_mask": joint_defect,
        "defect_present": defect_present,
        "preference_valid": np.asarray([preference_valid], dtype=np.bool_),
        "preference_clean_better": np.asarray([preference_valid], dtype=np.bool_),
        "is_clean": np.asarray([is_clean], dtype=np.bool_),
        "no_target_defect": np.asarray([no_target_defect], dtype=np.bool_),
        "approximately_equal_quality": np.asarray([approximately_equal_quality], dtype=np.bool_),
        "quality_equivalence_valid": np.asarray([quality_equivalence_valid], dtype=np.bool_),
        "clean_local_quat": clean.local_quat_wxyz,
        "clean_root_translation": clean.root_translation_m,
    }


def corruption_training_sample(
    result: CorruptionResult,
    *,
    source_window_start: int,
    extra_metadata: Mapping[str, Any] | None = None,
) -> TrainingSample:
    """Convert a corruption result into a fixed-rig sample without privileged inputs."""
    arrays = motion_feature_arrays(result.corrupted_motion)
    arrays.update(
        _target_arrays(
            result.clean_motion,
            intervention=result.intervention_mask,
            symptom=result.symptom_mask,
            responsibility=result.responsibility_target,
            responsibility_confidence=result.responsibility_confidence,
            preference_valid=result.postcondition.hard_negative,
            is_clean=False,
            no_target_defect=False,
            approximately_equal_quality=False,
            quality_equivalence_valid=result.postcondition.hard_negative,
        )
    )
    sample_id, metadata = _base_metadata(
        result.clean_motion,
        result.corrupted_motion,
        source_window_start=source_window_start,
        corruption_family=result.corruption_family,
        corruption_mechanism=result.corruption_mechanism,
        corruption_seed=result.corruption_seed,
        requested_severity_parameter=result.requested_severity_parameter,
        measured_severity=result.measured_severity,
        preference_confidence=result.preference_confidence,
        catalog_partition=result.catalog_partition,
        extra={
            "hard_negative": result.postcondition.hard_negative,
            "sample_role": (
                "composite"
                if result.corruption_family == "composite"
                else "hard_corruption"
                if result.postcondition.hard_negative
                else "soft_corruption"
            ),
            "postcondition_metric": result.postcondition.metric_name,
            "postcondition_before": result.postcondition.before,
            "postcondition_after": result.postcondition.after,
            "generation_parameters": dict(result.generation_parameters),
            "corruption_schema_metadata": dict(result.schema_metadata),
            **({} if extra_metadata is None else dict(extra_metadata)),
        },
    )
    return TrainingSample(sample_id=sample_id, arrays=arrays, metadata=metadata)


def clean_training_sample(
    clip: MotionClip,
    *,
    source_window_start: int,
    extra_metadata: Mapping[str, Any] | None = None,
) -> TrainingSample:
    """Build one clean reference sample with zero dense defect labels."""
    semantic_shape = (
        clip.num_frames,
        len(CORRUPTION_PART_NAMES),
        len(CORRUPTION_DEFECT_NAMES),
    )
    arrays = motion_feature_arrays(clip)
    arrays.update(
        _target_arrays(
            clip,
            intervention=np.zeros(
                (clip.num_frames, clip.num_joints, len(CORRUPTION_CHANNEL_NAMES)),
                dtype=np.bool_,
            ),
            symptom=np.zeros(semantic_shape, dtype=np.bool_),
            responsibility=np.zeros(semantic_shape, dtype=np.float32),
            responsibility_confidence=np.zeros(semantic_shape, dtype=np.float32),
            preference_valid=False,
            is_clean=True,
            no_target_defect=True,
            approximately_equal_quality=True,
            quality_equivalence_valid=True,
        )
    )
    sample_id, metadata = _base_metadata(
        clip,
        clip,
        source_window_start=source_window_start,
        corruption_family=None,
        corruption_mechanism=None,
        corruption_seed=None,
        requested_severity_parameter=None,
        measured_severity=0.0,
        preference_confidence=0.0,
        catalog_partition="clean",
        extra={"sample_role": "clean", **({} if extra_metadata is None else extra_metadata)},
    )
    return TrainingSample(sample_id=sample_id, arrays=arrays, metadata=metadata)


def equivalent_training_sample(
    clean: MotionClip,
    equivalent: MotionClip,
    *,
    source_window_start: int,
    control_mechanism: str,
    seed: int,
    transform_parameters: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
) -> TrainingSample:
    """Build a transformed real-motion control with clean/no-defect targets."""
    if clean.content_hash == equivalent.content_hash:
        raise ValueError("equivalent controls must not duplicate the clean motion")
    semantic_shape = (
        clean.num_frames,
        len(CORRUPTION_PART_NAMES),
        len(CORRUPTION_DEFECT_NAMES),
    )
    arrays = motion_feature_arrays(equivalent)
    arrays.update(
        _target_arrays(
            clean,
            intervention=_motion_intervention_mask(clean, equivalent),
            symptom=np.zeros(semantic_shape, dtype=np.bool_),
            responsibility=np.zeros(semantic_shape, dtype=np.float32),
            responsibility_confidence=np.zeros(semantic_shape, dtype=np.float32),
            preference_valid=False,
            is_clean=True,
            no_target_defect=True,
            approximately_equal_quality=True,
            quality_equivalence_valid=True,
        )
    )
    sample_id, metadata = _base_metadata(
        clean,
        equivalent,
        source_window_start=source_window_start,
        corruption_family=None,
        corruption_mechanism=control_mechanism,
        corruption_seed=seed,
        requested_severity_parameter=None,
        measured_severity=0.0,
        preference_confidence=0.0,
        catalog_partition="equivalent",
        extra={
            "sample_role": "equivalent_control",
            "equivalent_transform": {
                "mechanism": control_mechanism,
                "parameters": dict(transform_parameters),
            },
            **({} if extra_metadata is None else dict(extra_metadata)),
        },
    )
    return TrainingSample(sample_id=sample_id, arrays=arrays, metadata=metadata)


def _motion_intervention_mask(clean: MotionClip, edited: MotionClip) -> NDArray[np.bool_]:
    """Measure channels changed by a control transform without inferring causal symptoms."""
    if clean.num_frames != edited.num_frames:
        raise ValueError("clean and edited controls must have the same frame count")
    if clean.skeleton.content_hash != edited.skeleton.content_hash:
        raise ValueError("clean and edited controls must use the same skeleton")
    mask = np.zeros(
        (clean.num_frames, clean.num_joints, len(CORRUPTION_CHANNEL_NAMES)), dtype=np.bool_
    )
    root = clean.skeleton.root_index
    changed_root = np.abs(edited.root_translation_m - clean.root_translation_m) > 1.0e-7
    for axis in range(3):
        mask[:, root, axis] = changed_root[:, axis]
    changed_rotation = (
        quaternion_geodesic_distance(clean.local_quat_wxyz, edited.local_quat_wxyz) > 1.0e-7
    )
    mask[:, :, 3:] = changed_rotation[:, :, None]
    return mask


def sham_training_sample(
    clean: MotionClip,
    sham: MotionClip,
    *,
    source_window_start: int,
    target_family: str,
    control_mechanism: str,
    seed: int,
    matched_corruption_sample_id: str,
    target_metric_name: str,
    target_metric_delta: float,
    extra_metadata: Mapping[str, Any] | None = None,
) -> TrainingSample:
    """Build an edited no-symptom control with explicit quality-equivalence targets."""
    semantic_shape = (
        clean.num_frames,
        len(CORRUPTION_PART_NAMES),
        len(CORRUPTION_DEFECT_NAMES),
    )
    arrays = motion_feature_arrays(sham)
    arrays.update(
        _target_arrays(
            clean,
            intervention=_motion_intervention_mask(clean, sham),
            symptom=np.zeros(semantic_shape, dtype=np.bool_),
            responsibility=np.zeros(semantic_shape, dtype=np.float32),
            responsibility_confidence=np.zeros(semantic_shape, dtype=np.float32),
            preference_valid=False,
            is_clean=False,
            no_target_defect=True,
            approximately_equal_quality=True,
            quality_equivalence_valid=True,
        )
    )
    sample_id, metadata = _base_metadata(
        clean,
        sham,
        source_window_start=source_window_start,
        corruption_family=target_family,
        corruption_mechanism=control_mechanism,
        corruption_seed=seed,
        requested_severity_parameter=0.0,
        measured_severity=max(0.0, target_metric_delta),
        preference_confidence=0.0,
        catalog_partition="sham",
        extra={
            "hard_negative": False,
            "sample_role": "sham_control",
            "target_defect_family": target_family,
            "matched_corruption_sample_id": matched_corruption_sample_id,
            "postcondition_metric": target_metric_name,
            "postcondition_before": None,
            "postcondition_after": None,
            "sham_target_metric_delta": target_metric_delta,
            "generation_parameters": {},
            **({} if extra_metadata is None else dict(extra_metadata)),
        },
    )
    return TrainingSample(sample_id=sample_id, arrays=arrays, metadata=metadata)
