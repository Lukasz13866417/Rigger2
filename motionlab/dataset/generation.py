"""Resumable deterministic generation of fixed-window corruption datasets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from motionlab._version import __version__
from motionlab.core.provenance import append_operation_lineage, provenance_from_clip
from motionlab.corruptions import (
    CorruptionResult,
    compose_corruptions,
    corrupt_floating_contact_root_offset,
    corrupt_foot_clearance_ik_lowering,
    corrupt_foot_slide_hip_rotation_drift,
    corrupt_foot_slide_root_drift,
    corrupt_ground_penetration_root_offset,
    corrupt_joint_jitter_band_limited,
    corrupt_joint_jitter_white,
    corrupt_joint_limit_excess,
    corrupt_joint_pop_rotation_pulse,
    corrupt_joint_pop_smooth_pulse,
    corrupt_leg_stride_amplitude,
    corrupt_limb_phase_shift,
    corrupt_loop_seam_end_rotation,
    corrupt_loop_seam_root_velocity,
    corrupt_pelvis_vertical_curve,
    corrupt_speed_root_progression,
    corrupt_time_warp,
    verify_ordinal_chain,
)
from motionlab.corruptions._supervision import measure_contact_supervision
from motionlab.corruptions.base import IneffectiveCorruptionError
from motionlab.corruptions.catalog import CORRUPTION_CATALOG_VERSION
from motionlab.dataset.audit import (
    CollateralDefectError,
    CollateralFinding,
    single_defect_policy,
    validate_single_defect,
)
from motionlab.dataset.features import (
    DATASET_SAMPLE_VERSION,
    TrainingSample,
    clean_training_sample,
    corruption_training_sample,
    equivalent_training_sample,
    sham_training_sample,
)
from motionlab.dataset.io import (
    file_sha256,
    fit_training_normalization,
    load_normalization_statistics,
    load_training_sample,
    save_normalization_statistics,
    save_training_sample,
)
from motionlab.dataset.targeting import (
    ObservedSeverityBin,
    ObservedSeverityScale,
    SeverityTargetingError,
    target_observed_severity,
)
from motionlab.io.npz import load_motion_npz
from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_rotate_vector,
)
from motionlab.metrics.foot_sliding import foot_sliding_metric
from motionlab.motion.clip import MotionClip
from motionlab.processing._signals import true_intervals
from motionlab.processing.contacts import detect_foot_contacts
from motionlab.processing.phase import estimate_gait_phase, extract_gait_events
from motionlab.processing.resample import resample_motion
from motionlab.repair.foot_lock import lock_foot

DATASET_GENERATION_VERSION = "motionlab.corruption_dataset.v7"
SEVERITY_LABELS = ("near_threshold", "subtle", "moderate", "clear", "severe")


class DatasetGenerationConfig(BaseModel):
    """Frozen configuration controlling windowing and corruption sampling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 1234
    target_fps: float = Field(default=60.0, gt=0.0)
    window_frames: int = Field(default=160, ge=16)
    stride_frames: int = Field(default=32, ge=1)
    include_heldout_mechanisms: bool = True
    include_soft_corruptions: bool = True
    composition_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    maximum_windows_per_source: int | None = Field(default=None, ge=1)
    minimum_clean_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    severity_search_evaluations: int = Field(default=14, ge=4, le=64)
    include_sham_controls: bool = True
    cross_mechanism_relative_tolerance: float = Field(default=0.2, gt=0.0, le=1.0)
    equivalent_global_yaw_degrees: tuple[float, ...] = ()
    mechanisms: tuple[str, ...] | None = None

    @field_validator("equivalent_global_yaw_degrees")
    @classmethod
    def equivalent_yaws_are_distinct_and_nonzero(
        cls, value: tuple[float, ...]
    ) -> tuple[float, ...]:
        if any(not np.isfinite(item) or abs(item) < 1.0e-6 or abs(item) > 180.0 for item in value):
            raise ValueError(
                "equivalent yaw angles must be finite, nonzero, and within 180 degrees"
            )
        if len(set(value)) != len(value):
            raise ValueError("equivalent yaw angles must be unique")
        return value


Generator = Callable[[MotionClip, float, int, bool], CorruptionResult]


@dataclass(frozen=True)
class AutomaticMechanism:
    """One automatic generator with an observed-symptom severity contract."""

    key: str
    partition: str
    hard_negative: bool
    generate: Generator
    parameter_bounds: tuple[float, float] | None = None
    severity_scale: ObservedSeverityScale | None = None
    soft_parameter_values: tuple[float, ...] = ()
    search_strategy: Literal["monotone_binary", "bounded_grid"] = "monotone_binary"
    parameter_quantum: float | None = None

    def __post_init__(self) -> None:
        if self.partition not in {"train", "heldout_eval"}:
            raise ValueError("automatic mechanism has an invalid catalog partition")
        if self.hard_negative:
            if self.parameter_bounds is None or self.severity_scale is None:
                raise ValueError("hard mechanisms require parameter bounds and a severity scale")
            if self.soft_parameter_values:
                raise ValueError("hard mechanisms cannot declare soft parameter presets")
        elif not self.soft_parameter_values:
            raise ValueError("soft mechanisms require explicit non-ordinal parameter presets")


def _severity_scale(metric_name: str, edges: Sequence[float]) -> ObservedSeverityScale:
    if len(edges) != len(SEVERITY_LABELS) + 1:
        raise ValueError("observed severity scales require six ordered edges")
    return ObservedSeverityScale(
        metric_name=metric_name,
        bins=tuple(
            ObservedSeverityBin(label, float(lower), float(upper))
            for label, lower, upper in zip(SEVERITY_LABELS, edges[:-1], edges[1:], strict=True)
        ),
    )


def _central_interval(clip: MotionClip, fraction: float = 0.65) -> tuple[int, int]:
    length = max(7, min(clip.num_frames, round(clip.num_frames * fraction)))
    start = max(0, (clip.num_frames - length) // 2)
    return start, start + length


def _longest_foot_interval(
    clip: MotionClip,
    *,
    contact: bool,
    preferred_side: str | None = None,
) -> tuple[str, int, int]:
    if preferred_side not in {None, "left", "right"}:
        raise ValueError("preferred_side must be left, right, or None")
    contacts = detect_foot_contacts(clip)
    candidates: list[tuple[int, str, int, int]] = []
    for side in ("left", "right"):
        indices = [
            index for index, name in enumerate(contacts.marker_names) if name.startswith(f"{side}_")
        ]
        mask = np.any(contacts.hard[:, indices], axis=1)
        if not contact:
            mask = ~mask
        for start, stop in true_intervals(mask):
            if stop - start >= 5:
                candidates.append((stop - start, side, start, stop))
    if not candidates:
        state = "contact" if contact else "swing"
        raise ValueError(f"source clip has no usable foot {state} interval")
    preferred = [item for item in candidates if item[1] == preferred_side]
    if preferred:
        candidates = preferred
    center = 0.5 * clip.num_frames
    _, side, start, stop = max(
        candidates,
        key=lambda item: (
            -abs(0.5 * (item[2] + item[3]) - center),
            item[0],
            -item[2],
        ),
    )
    return side, start, stop


def _preferred_joint(clip: MotionClip) -> int:
    for role in ("chest", "spine_1", "left_knee", "right_knee"):
        if role in clip.skeleton.roles:
            return clip.skeleton.roles[role]
    return min(1, clip.num_joints - 1)


def _preferred_jitter_joints(clip: MotionClip) -> tuple[int, ...]:
    joints = tuple(
        dict.fromkeys(
            clip.skeleton.roles[role]
            for role in ("chest", "left_shoulder", "right_shoulder")
            if role in clip.skeleton.roles
        )
    )
    return joints or (_preferred_joint(clip),)


def _foot_root(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    _, start, stop = _longest_foot_interval(
        clip, contact=True, preferred_side=("left", "right")[seed % 2]
    )
    return corrupt_foot_slide_root_drift(
        clip,
        start_frame=start,
        stop_frame=stop,
        distance_m=severity,
        direction_world=(1.0, 0.0, 0.0),
        seed=seed,
        strict_postcondition=strict,
    )


def _foot_hip(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    side, start, stop = _longest_foot_interval(
        clip, contact=True, preferred_side=("left", "right")[seed % 2]
    )
    return corrupt_foot_slide_hip_rotation_drift(
        clip,
        side=side,
        start_frame=start,
        stop_frame=stop,
        angle_rad=severity,
        axis_local=(0.0, 1.0, 0.0),
        seed=seed,
        strict_postcondition=strict,
    )


def _penetration(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    _, start, stop = _longest_foot_interval(
        clip, contact=True, preferred_side=("left", "right")[seed % 2]
    )
    return corrupt_ground_penetration_root_offset(
        clip,
        start_frame=start,
        stop_frame=stop,
        depth_m=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _floating(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    _, start, stop = _longest_foot_interval(
        clip, contact=True, preferred_side=("left", "right")[seed % 2]
    )
    return corrupt_floating_contact_root_offset(
        clip,
        start_frame=start,
        stop_frame=stop,
        height_m=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _joint_pop(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    return corrupt_joint_pop_rotation_pulse(
        clip,
        joint=_preferred_joint(clip),
        frame=clip.num_frames // 2,
        angle_rad=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _joint_pop_smooth(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    center = clip.num_frames // 2
    return corrupt_joint_pop_smooth_pulse(
        clip,
        joint=_preferred_joint(clip),
        start_frame=center - 3,
        stop_frame=center + 4,
        angle_rad=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _jitter_white(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    start, stop = _central_interval(clip)
    return corrupt_joint_jitter_white(
        clip,
        joints=_preferred_jitter_joints(clip),
        start_frame=start,
        stop_frame=stop,
        amplitude_rad=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _jitter_band(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    start, stop = _central_interval(clip)
    return corrupt_joint_jitter_band_limited(
        clip,
        joints=_preferred_jitter_joints(clip),
        start_frame=start,
        stop_frame=stop,
        amplitude_rad=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _limb_phase(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    start, stop = _central_interval(clip, 0.8)
    return corrupt_limb_phase_shift(
        clip,
        start_frame=start,
        stop_frame=stop,
        shift_frames=max(1, round(severity)),
        side="left",
        seed=seed,
        strict_postcondition=strict,
    )


def _limb_phase_bilateral(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    start, stop = _central_interval(clip, 0.8)
    return corrupt_limb_phase_shift(
        clip,
        start_frame=start,
        stop_frame=stop,
        shift_frames=max(1, round(severity)),
        bilateral=True,
        seed=seed,
        strict_postcondition=strict,
    )


def _pose_warp(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    start, stop = _central_interval(clip, 0.85)
    return corrupt_time_warp(
        clip,
        start_frame=start,
        stop_frame=stop,
        strength=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _root_warp(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    start, stop = _central_interval(clip, 0.85)
    return corrupt_time_warp(
        clip,
        start_frame=start,
        stop_frame=stop,
        strength=severity,
        warp_root_only=True,
        seed=seed,
        strict_postcondition=strict,
    )


def _speed(clip: MotionClip, severity: float, seed: int, strict: bool = True) -> CorruptionResult:
    return corrupt_speed_root_progression(
        clip,
        speed_scale=1.0 + severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _stride(clip: MotionClip, severity: float, seed: int, strict: bool = True) -> CorruptionResult:
    return corrupt_leg_stride_amplitude(
        clip,
        amplitude_scale=1.0 - severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _loop_rotation(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    candidates = tuple(
        corrupt_loop_seam_end_rotation(
            clip,
            joint=_preferred_joint(clip),
            seam_window_frames=min(12, clip.num_frames),
            angle_rad=severity,
            axis_local=axis,
            seed=seed,
            strict_postcondition=False,
        )
        for axis in (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    result = max(candidates, key=lambda candidate: candidate.measured_severity)
    if strict and not result.postcondition.hard_negative:
        raise IneffectiveCorruptionError("end-rotation candidates did not worsen the loop seam")
    return result


def _loop_velocity(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    return corrupt_loop_seam_root_velocity(
        clip,
        seam_window_frames=min(12, clip.num_frames),
        velocity_delta_mps=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _clearance(
    clip: MotionClip, severity: float, seed: int, strict: bool = True
) -> CorruptionResult:
    side, start, stop = _longest_foot_interval(
        clip, contact=False, preferred_side=("left", "right")[seed % 2]
    )
    return corrupt_foot_clearance_ik_lowering(
        clip,
        side=side,
        start_frame=start,
        stop_frame=stop,
        clearance_loss_m=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _limit(clip: MotionClip, severity: float, seed: int, strict: bool = True) -> CorruptionResult:
    valid = [
        joint
        for joint, spec in clip.skeleton.joint_limits.items()
        if spec.valid and spec.confidence > 0.0
    ]
    if not valid:
        raise ValueError("source rig has no valid authored joint limit")
    center = clip.num_frames // 2
    return corrupt_joint_limit_excess(
        clip,
        joint=valid[0],
        start_frame=center - 5,
        stop_frame=center + 6,
        excess_deg=severity,
        seed=seed,
        strict_postcondition=strict,
    )


def _pelvis(clip: MotionClip, severity: float, seed: int, strict: bool = True) -> CorruptionResult:
    del strict
    return corrupt_pelvis_vertical_curve(clip, amplitude_scale=1.0 + severity, seed=seed)


_FOOT_SLIDE_SCALE = _severity_scale(
    "foot_sliding_cm_per_contact_interval", (0.25, 1.0, 3.0, 7.0, 15.0, 30.0)
)
_PENETRATION_SCALE = _severity_scale(
    "maximum_ground_penetration_cm", (0.2, 0.75, 1.5, 3.0, 5.0, 8.0)
)
_FLOATING_SCALE = _severity_scale("maximum_floating_contact_cm", (0.2, 0.75, 1.5, 3.0, 5.0, 8.0))
_POP_SCALE = _severity_scale(
    "localized_target_joint_peak_angular_jerk_rad_s3",
    (800.0, 3000.0, 7000.0, 14000.0, 26000.0, 52000.0),
)
_JITTER_SCALE = _severity_scale(
    "localized_angular_jerk_p95_rad_s3",
    (500.0, 1500.0, 3000.0, 6000.0, 10000.0, 18000.0),
)
_LIMB_PHASE_SCALE = _severity_scale(
    "source_relative_arm_phase_deviation_rad_rms", (0.01, 0.025, 0.05, 0.08, 0.12, 0.18)
)
_POSE_CADENCE_SCALE = _severity_scale(
    "source_relative_pose_timing_deviation_rad_rms", (0.02, 0.06, 0.11, 0.17, 0.23, 0.32)
)
_ROOT_CADENCE_SCALE = _severity_scale(
    "source_relative_root_timing_deviation_cm_rms", (0.5, 2.0, 4.0, 7.0, 10.0, 15.0)
)
_ROOT_SPEED_SCALE = _severity_scale(
    "absolute_target_speed_error_mps", (0.015, 0.05, 0.1, 0.2, 0.35, 0.55)
)
_LEG_SPEED_SCALE = _severity_scale(
    "source_relative_leg_stride_pose_deviation_rad_rms", (0.015, 0.05, 0.1, 0.2, 0.32, 0.48)
)
_LOOP_SCALE = _severity_scale("weighted_loop_seam", (0.0001, 0.002, 0.01, 0.03, 0.07, 0.15))
_CLEARANCE_SCALE = _severity_scale(
    "maximum_source_relative_clearance_loss_cm", (0.2, 0.7, 1.5, 3.0, 5.0, 8.5)
)
_LIMIT_SCALE = _severity_scale("maximum_joint_limit_excess_deg", (1.0, 5.0, 10.0, 20.0, 35.0, 55.0))


AUTOMATIC_MECHANISMS = (
    AutomaticMechanism(
        "foot_slide/root_drift",
        "train",
        True,
        _foot_root,
        parameter_bounds=(0.002, 0.165),
        severity_scale=_FOOT_SLIDE_SCALE,
    ),
    AutomaticMechanism(
        "foot_slide/hip_rotation_drift",
        "heldout_eval",
        True,
        _foot_hip,
        parameter_bounds=(0.004, 0.8),
        severity_scale=_FOOT_SLIDE_SCALE,
    ),
    AutomaticMechanism(
        "ground_penetration/root_vertical_offset",
        "train",
        True,
        _penetration,
        parameter_bounds=(0.002, 0.081),
        severity_scale=_PENETRATION_SCALE,
    ),
    AutomaticMechanism(
        "floating_contact/root_vertical_offset",
        "train",
        True,
        _floating,
        parameter_bounds=(0.006, 0.09),
        severity_scale=_FLOATING_SCALE,
    ),
    AutomaticMechanism(
        "joint_pop/local_rotation_pulse",
        "train",
        True,
        _joint_pop,
        parameter_bounds=(0.007, 0.49),
        severity_scale=_POP_SCALE,
    ),
    AutomaticMechanism(
        "joint_pop/smooth_rotation_pulse",
        "heldout_eval",
        True,
        _joint_pop_smooth,
        parameter_bounds=(0.015, 0.78),
        severity_scale=_POP_SCALE,
    ),
    AutomaticMechanism(
        "joint_jitter/white_tangent_noise",
        "train",
        True,
        _jitter_white,
        parameter_bounds=(0.001, 0.05),
        severity_scale=_JITTER_SCALE,
    ),
    AutomaticMechanism(
        "joint_jitter/band_limited_correlated",
        "heldout_eval",
        True,
        _jitter_band,
        parameter_bounds=(0.002, 0.085),
        severity_scale=_JITTER_SCALE,
    ),
    AutomaticMechanism(
        "limb_phase_mismatch/unilateral_circular_shift",
        "train",
        True,
        _limb_phase,
        parameter_bounds=(1.0, 28.0),
        severity_scale=_LIMB_PHASE_SCALE,
        parameter_quantum=1.0,
    ),
    AutomaticMechanism(
        "limb_phase_mismatch/bilateral_circular_shift",
        "heldout_eval",
        True,
        _limb_phase_bilateral,
        parameter_bounds=(1.0, 28.0),
        severity_scale=_LIMB_PHASE_SCALE,
        parameter_quantum=1.0,
    ),
    AutomaticMechanism(
        "cadence_inconsistency/pose_stream_monotonic_warp",
        "train",
        True,
        _pose_warp,
        parameter_bounds=(0.03, 0.949),
        severity_scale=_POSE_CADENCE_SCALE,
    ),
    AutomaticMechanism(
        "cadence_inconsistency/root_stream_monotonic_warp",
        "heldout_eval",
        True,
        _root_warp,
        parameter_bounds=(0.03, 0.949),
        severity_scale=_ROOT_CADENCE_SCALE,
    ),
    AutomaticMechanism(
        "speed_inconsistency/root_progression_scale",
        "train",
        True,
        _speed,
        parameter_bounds=(0.015, 2.5),
        severity_scale=_ROOT_SPEED_SCALE,
    ),
    AutomaticMechanism(
        "speed_inconsistency/leg_pose_amplitude_scale",
        "heldout_eval",
        True,
        _stride,
        parameter_bounds=(0.025, 0.86),
        severity_scale=_LEG_SPEED_SCALE,
    ),
    AutomaticMechanism(
        "loop_seam/end_rotation_offset",
        "train",
        True,
        _loop_rotation,
        parameter_bounds=(0.0005, 3.2),
        severity_scale=_LOOP_SCALE,
    ),
    AutomaticMechanism(
        "loop_seam/root_velocity_mismatch",
        "heldout_eval",
        True,
        _loop_velocity,
        parameter_bounds=(0.01, 0.75),
        severity_scale=_LOOP_SCALE,
    ),
    AutomaticMechanism(
        "foot_clearance/swing_ankle_ik_lowering",
        "train",
        True,
        _clearance,
        parameter_bounds=(0.0015, 0.075),
        severity_scale=_CLEARANCE_SCALE,
    ),
    AutomaticMechanism(
        "joint_limit/authored_limit_excess",
        "train",
        True,
        _limit,
        parameter_bounds=(1.01, 54.99),
        severity_scale=_LIMIT_SCALE,
    ),
    AutomaticMechanism(
        "pelvis_curve/vertical_amplitude_scale",
        "train",
        False,
        _pelvis,
        soft_parameter_values=(0.25, 0.5, 1.0),
    ),
)


def _stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256()
    digest.update(str(base).encode())
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode())
    return int.from_bytes(digest.digest()[:4], "big", signed=False)


def _counterfactual_group_id(window: MotionClip, window_start: int) -> str:
    payload = {
        "source_window_motion_id": window.content_hash,
        "source_window_start": window_start,
        "version": DATASET_GENERATION_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"counterfactual:sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _global_yaw_equivalent(clip: MotionClip, angle_degrees: float, seed: int) -> MotionClip:
    """Rotate a clean motion's world heading without changing its intrinsic animation."""
    angle_rad = float(np.deg2rad(angle_degrees))
    delta = axis_angle_to_quaternion((0.0, angle_rad, 0.0))
    root = clip.skeleton.root_index
    local = clip.local_quat_wxyz.copy()
    local[:, root] = quaternion_multiply(delta, local[:, root]).astype(np.float32)
    origin = clip.root_translation_m[0].astype(np.float64)
    translated = clip.root_translation_m.astype(np.float64) - origin
    root_translation = quaternion_rotate_vector(delta, translated) + origin
    parameters = {"angle_degrees": angle_degrees}
    metadata = append_operation_lineage(
        clip,
        dict(clip.metadata),
        operation="equivalent_global_yaw",
        kind="transform",
        parameters=parameters,
        version=__version__,
        seed=seed,
    )
    metadata["equivalent_control"] = {
        "mechanism": "global_heading_rotation",
        **parameters,
    }
    return clip.with_updates(
        local_quat_wxyz=local,
        root_translation_m=root_translation.astype(np.float32),
        metadata=metadata,
    )


def _gait_target_metadata(
    clip: MotionClip,
    result: CorruptionResult,
) -> dict[str, Any]:
    """Describe semantic clean-source gait events targeted by a corruption."""
    parameters = result.generation_parameters
    family = result.corruption_family
    if family == "loop_seam":
        return {
            "gait_event_labels": ["loop_seam"],
            "gait_phase_bins": [],
            "target_side": None,
            "gait_target": {
                "reference_source": "clean_source_motion",
                "start_frame": max(0, clip.num_frames - 3),
                "stop_frame": clip.num_frames,
                "events": [{"label": "loop_seam", "frame": clip.num_frames - 1}],
            },
        }

    start = int(parameters.get("start_frame", parameters.get("frame", 0)))
    if "frame" in parameters:
        stop = min(clip.num_frames, start + 1)
    else:
        stop = int(parameters.get("stop_frame", clip.num_frames))
    start = max(0, min(start, clip.num_frames - 1))
    stop = max(start + 1, min(stop, clip.num_frames))
    contacts = detect_foot_contacts(clip)
    phase = estimate_gait_phase(contacts)
    marker_index = {name: index for index, name in enumerate(contacts.marker_names)}
    labels: set[str] = set()
    event_records: list[dict[str, Any]] = []
    requested_mode: str | None
    if family == "foot_clearance":
        requested_mode = "swing"
    elif family in {"foot_slide", "ground_penetration", "floating_contact"}:
        requested_mode = "stance"
    else:
        requested_mode = None
    side_scores: dict[str, float] = {}
    for side in ("left", "right"):
        indices = (marker_index[f"{side}_heel"], marker_index[f"{side}_toe"])
        foot_contact = np.any(contacts.hard[start:stop, indices], axis=1)
        side_scores[side] = float(np.mean(foot_contact))
        selected = requested_mode == "stance" and np.mean(foot_contact) >= 0.5
        selected = selected or (requested_mode == "swing" and np.mean(foot_contact) < 0.5)
        if requested_mode is not None and selected:
            labels.add(f"{side}_{requested_mode}")
    both_contact = np.any(
        contacts.hard[start:stop, (marker_index["left_heel"], marker_index["left_toe"])],
        axis=1,
    ) & np.any(
        contacts.hard[start:stop, (marker_index["right_heel"], marker_index["right_toe"])],
        axis=1,
    )
    if np.any(both_contact):
        labels.add("double_support")
    for event in extract_gait_events(contacts):
        if start <= event.frame < stop:
            label = f"{event.side}_{event.kind}"
            labels.add(label)
            event_records.append(
                {
                    "label": label,
                    "frame": event.frame,
                    "confidence": event.confidence,
                }
            )
    if family in {
        "limb_phase_mismatch",
        "cadence_inconsistency",
        "speed_inconsistency",
    }:
        labels.add("gait_cycle")
    valid_phase = phase.phase_rad[start:stop][phase.valid[start:stop]]
    phase_bins = sorted(
        {
            f"phase_octant_{int(np.floor((float(value) % (2.0 * np.pi)) / (0.25 * np.pi))) % 8}"
            for value in valid_phase
        }
    )
    explicit_side = parameters.get("side")
    inferred_side = (
        min(side_scores, key=lambda side: side_scores[side])
        if requested_mode == "swing"
        else max(side_scores, key=lambda side: side_scores[side])
    )
    target_side = (
        str(explicit_side)
        if explicit_side in {"left", "right"}
        else inferred_side
        if family in {"foot_slide", "ground_penetration", "floating_contact", "foot_clearance"}
        else None
    )
    return {
        "gait_event_labels": sorted(labels),
        "gait_phase_bins": phase_bins,
        "target_side": target_side,
        "gait_target": {
            "reference_source": "privileged_clean_contact_schedule",
            "start_frame": start,
            "stop_frame": stop,
            "requested_mode": requested_mode,
            "events": event_records,
        },
    }


def _config_payload(config: DatasetGenerationConfig) -> dict[str, Any]:
    return {
        "format_version": DATASET_GENERATION_VERSION,
        "sample_version": DATASET_SAMPLE_VERSION,
        "catalog_version": CORRUPTION_CATALOG_VERSION,
        "generator_code_version": __version__,
        "config": config.model_dump(mode="json"),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_lines(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )


def _audit_breakdowns(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate generation attempts without discarding rejection evidence."""

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        accepted = [record for record in selected if bool(record.get("accepted"))]
        rejected = [record for record in selected if not bool(record.get("accepted"))]
        severities = [
            float(record["actual_observed_severity"])
            for record in accepted
            if isinstance(record.get("actual_observed_severity"), (int, float))
        ]
        return {
            "attempted": len(selected),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_fraction": len(accepted) / len(selected) if selected else 0.0,
            "rejection_reasons": dict(
                sorted(
                    Counter(
                        str(record.get("rejection_reason", "unknown")) for record in rejected
                    ).items()
                )
            ),
            "collateral_defect_rejection_count": sum(
                bool(record.get("collateral_defect_rejection")) for record in rejected
            ),
            "actual_observed_severity": _distribution(severities),
        }

    dimensions = {
        "defect_family": sorted(
            {str(record["defect_family"]) for record in records if record.get("defect_family")}
        ),
        "corruption_mechanism": sorted(
            {
                str(record["corruption_mechanism"])
                for record in records
                if record.get("corruption_mechanism")
            }
        ),
        "severity_bin": sorted(
            {str(record["severity_bin"]) for record in records if record.get("severity_bin")}
        ),
        "gait_event": sorted(
            {str(label) for record in records for label in record.get("gait_events", []) if label}
        ),
        "side": sorted({str(record["side"]) for record in records if record.get("side")}),
        "source_clip": sorted(
            {str(record["source_clip"]) for record in records if record.get("source_clip")}
        ),
    }
    breakdowns: dict[str, Any] = {"overall": summarize(records)}
    for dimension, values in dimensions.items():
        breakdowns[f"by_{dimension}"] = {
            value: summarize(
                [
                    record
                    for record in records
                    if (
                        value in record.get("gait_events", [])
                        if dimension == "gait_event"
                        else str(record.get(dimension)) == value
                    )
                ]
            )
            for value in values
        }
    sham_records = [record for record in records if record.get("attempt_kind") == "sham"]
    breakdowns["sham"] = {
        **summarize(sham_records),
        "accepted_without_target_symptom": sum(
            bool(record.get("accepted")) and bool(record.get("target_symptom_absent"))
            for record in sham_records
        ),
    }
    return breakdowns


def _save_or_reuse(path: Path, sample: TrainingSample) -> bool:
    """Return true when an identical existing sample was reused."""
    if path.exists():
        try:
            existing = load_training_sample(path)
            same = existing.sample_id == sample.sample_id and existing.metadata == sample.metadata
            same = same and all(
                np.array_equal(existing.arrays[name], value)
                for name, value in sample.arrays.items()
            )
            if same:
                return True
        except ValueError:
            pass
    save_training_sample(path, sample)
    return False


def _sample_record(
    output: Path,
    path: Path,
    sample: TrainingSample,
    *,
    data_split: str,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "path": os.path.relpath(path, output),
        "sha256": file_sha256(path),
        "data_split": data_split,
        "source_split": sample.metadata["split"],
        "split_lineage_id": sample.metadata["split_lineage_id"],
        "source_motion_id": sample.metadata["source_motion_id"],
        "source_clip_id": sample.metadata["source_clip_id"],
        "source_window_start": sample.metadata["source_window_start"],
        "source_path": str(source_path),
        "corruption_family": sample.metadata["corruption_family"],
        "corruption_mechanism": sample.metadata["corruption_mechanism"],
        "catalog_mechanism_key": sample.metadata.get("catalog_mechanism_key"),
        "catalog_partition": sample.metadata["catalog_partition"],
        "measured_severity": sample.metadata["measured_severity"],
        "hard_negative": sample.metadata.get("hard_negative", False),
        "severity_label": sample.metadata.get("severity_label"),
        "severity_bin": sample.metadata.get("severity_bin"),
        "ordinal_chain_id": sample.metadata.get("ordinal_chain_id"),
        "ordinal_rank": sample.metadata.get("ordinal_rank"),
        "counterfactual_group_id": sample.metadata.get("counterfactual_group_id"),
        "sample_role": sample.metadata.get("sample_role"),
        "no_target_defect": bool(sample.arrays["no_target_defect"].item()),
        "approximately_equal_quality": bool(sample.arrays["approximately_equal_quality"].item()),
        "gait_event_labels": sample.metadata.get("gait_event_labels", []),
        "gait_phase_bins": sample.metadata.get("gait_phase_bins", []),
        "target_side": sample.metadata.get("target_side"),
        "collateral_defect_families": sample.metadata.get("collateral_defect_families", []),
        "matched_corruption_sample_id": sample.metadata.get("matched_corruption_sample_id"),
        "matched_corruption_mechanism": sample.metadata.get("matched_corruption_mechanism"),
    }


def _mechanisms(config: DatasetGenerationConfig) -> tuple[AutomaticMechanism, ...]:
    selected: list[AutomaticMechanism] = []
    requested = None if config.mechanisms is None else set(config.mechanisms)
    known = {mechanism.key for mechanism in AUTOMATIC_MECHANISMS}
    if requested is not None:
        unknown = requested.difference(known)
        if unknown:
            raise ValueError(f"unknown automatic corruption mechanisms: {sorted(unknown)}")
    for mechanism in AUTOMATIC_MECHANISMS:
        if requested is not None and mechanism.key not in requested:
            continue
        if mechanism.partition == "heldout_eval" and not config.include_heldout_mechanisms:
            continue
        if not mechanism.hard_negative and not config.include_soft_corruptions:
            continue
        selected.append(mechanism)
    if not selected:
        raise ValueError("dataset configuration selected no corruption mechanisms")
    return tuple(selected)


def _load_sources(
    paths: Sequence[Path], config: DatasetGenerationConfig
) -> list[tuple[Path, MotionClip]]:
    if not paths:
        raise ValueError("dataset generation requires at least one source motion")
    sources: list[tuple[Path, MotionClip]] = []
    skeleton_id: str | None = None
    lineage_splits: dict[str, str] = {}
    for raw_path in sorted((Path(path) for path in paths), key=lambda value: str(value)):
        clip = load_motion_npz(raw_path)
        if not np.isclose(clip.fps, config.target_fps, atol=1.0e-12, rtol=0.0):
            clip = resample_motion(clip, config.target_fps)
        provenance = provenance_from_clip(clip)
        if (
            not provenance.lineage_complete
            or provenance.split_lineage_id is None
            or provenance.split is None
        ):
            raise ValueError(f"source motion requires complete split lineage: {raw_path}")
        if (
            provenance.clean_confidence is None
            or provenance.clean_confidence < config.minimum_clean_confidence
        ):
            raise ValueError(
                f"source motion clean confidence is missing or below threshold: {raw_path}"
            )
        previous = lineage_splits.setdefault(provenance.split_lineage_id, provenance.split)
        if previous != provenance.split:
            raise ValueError("one split lineage appears in multiple source splits")
        current_skeleton = clip.skeleton.content_hash
        if skeleton_id is None:
            skeleton_id = current_skeleton
        elif current_skeleton != skeleton_id:
            raise ValueError("fixed-rig dataset generation requires one skeleton")
        if clip.num_frames < config.window_frames:
            raise ValueError(
                f"source clip has {clip.num_frames} frames, fewer than window length "
                f"{config.window_frames}: {raw_path}"
            )
        sources.append((raw_path, clip))
    return sources


def _window_starts(clip: MotionClip, config: DatasetGenerationConfig) -> tuple[int, ...]:
    starts = tuple(range(0, clip.num_frames - config.window_frames + 1, config.stride_frames))
    if config.maximum_windows_per_source is not None:
        starts = starts[: config.maximum_windows_per_source]
    return starts


def _preference_records(
    clean_sample: TrainingSample,
    samples: Sequence[TrainingSample],
    results: Sequence[CorruptionResult],
    chain_id: str,
) -> list[dict[str, Any]]:
    ordered_samples = (clean_sample, *samples)
    records: list[dict[str, Any]] = []
    confidence = min(result.preference_confidence for result in results)
    for rank, (better, worse) in enumerate(pairwise(ordered_samples)):
        payload = {
            "chain_id": chain_id,
            "rank_edge": rank,
            "better_sample_id": better.sample_id,
            "worse_sample_id": worse.sample_id,
            "confidence": confidence,
            "metric_name": results[0].postcondition.metric_name,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        records.append(
            {
                "preference_id": (
                    f"preference:sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"
                ),
                **payload,
            }
        )
    return records


def _matched_foot_slide_sham(
    clean: MotionClip,
    result: CorruptionResult,
    negative_sample: TrainingSample,
    *,
    source_window_start: int,
    counterfactual_group_id: str,
    seed: int,
    gait_metadata: Mapping[str, Any],
) -> TrainingSample:
    """Compensate a root-drift intervention so the trusted stance foot remains fixed."""
    target_side = gait_metadata.get("target_side")
    side = str(target_side) if target_side in {"left", "right"} else None
    generated_start = result.generation_parameters.get("start_frame")
    generated_stop = result.generation_parameters.get("stop_frame")
    if side is None or generated_start is None or generated_stop is None:
        side, start, stop = _longest_foot_interval(
            clean, contact=True, preferred_side=("left", "right")[seed % 2]
        )
    else:
        start, stop = int(generated_start), int(generated_stop)
    locked = lock_foot(
        result.corrupted_motion,
        side=side,  # type: ignore[arg-type]
        start_frame=start,
        stop_frame=stop,
        lock_strength=1.0,
        blend_in_frames=2,
        blend_out_frames=2,
        pelvis_compensation_limit_m=0.12,
    )
    repaired = locked.repaired
    maximum_sham_delta_cm = 0.2
    root_matches = np.allclose(
        repaired.root_translation_m,
        result.corrupted_motion.root_translation_m,
        atol=1.0e-7,
        rtol=0.0,
    )
    delta = float("inf")
    fallback_reason: str | None = None
    if root_matches:
        foot = clean.skeleton.roles[f"{side}_ankle"]
        toe = clean.skeleton.roles.get(f"{side}_toe")
        orientation_joints = (foot,) if toe is None else (foot, toe)
        _, target_global = forward_kinematics_numpy(
            result.corrupted_motion.skeleton,
            result.corrupted_motion.local_quat_wxyz,
            result.corrupted_motion.root_translation_m,
        )
        repaired_local = repaired.local_quat_wxyz.copy()
        for frame in range(start, stop):
            for joint in orientation_joints:
                _, repaired_global = forward_kinematics_numpy(
                    repaired.skeleton,
                    repaired_local[frame],
                    repaired.root_translation_m[frame],
                )
                parent = int(repaired.skeleton.parents[joint])
                repaired_local[frame, joint] = quaternion_multiply(
                    quaternion_inverse(repaired_global[parent]),
                    target_global[start, joint],
                ).astype(np.float32)
        repaired = repaired.with_updates(local_quat_wxyz=repaired_local)
        _, fixed_clean, fixed_sham = measure_contact_supervision(clean, repaired)
        before = float(foot_sliding_metric(clean, fixed_clean).clip_value or 0.0)
        after = float(foot_sliding_metric(repaired, fixed_sham).clip_value or 0.0)
        delta = after - before
        if delta > maximum_sham_delta_cm:
            fallback_reason = f"contact-compensated sham retained +{delta:g} cm target slip"
    else:
        fallback_reason = "contact compensation changed the root intervention"
    control_mechanism = "root_drift_contact_compensated"
    if fallback_reason is not None:
        # A reversible edit round-trip is the robust no-symptom control when fixed-rig IK cannot
        # preserve a real, near-extended stance. The tiny vertical dither makes the persisted
        # neural signal non-identical without creating tangential foot motion.
        dithered_root = clean.root_translation_m.copy()
        phase = np.linspace(0.0, np.pi, stop - start, dtype=np.float64)
        dithered_root[start:stop, 1] += (1.0e-6 * np.sin(phase)).astype(np.float32)
        sham_metadata = dict(clean.metadata)
        sham_metadata.update(
            {
                "parent_motion_id": clean.content_hash,
                "operation": "sham_reversible_root_roundtrip",
                "operator_parameters": {
                    "applied_and_inverted_mechanism": result.corruption_mechanism,
                    "vertical_dither_m": 1.0e-6,
                    "fallback_reason": fallback_reason,
                },
            }
        )
        sham_metadata = append_operation_lineage(
            clean,
            sham_metadata,
            operation="sham_reversible_root_roundtrip",
            kind="transform",
            parameters={
                "applied_and_inverted_mechanism": result.corruption_mechanism,
                "vertical_dither_m": 1.0e-6,
            },
            version="motionlab-0.1.0",
            seed=seed,
        )
        repaired = clean.with_updates(root_translation_m=dithered_root, metadata=sham_metadata)
        delta = 0.0
        control_mechanism = "root_drift_reversible_roundtrip"
    return sham_training_sample(
        clean,
        repaired,
        source_window_start=source_window_start,
        target_family="foot_slide",
        control_mechanism=control_mechanism,
        seed=seed,
        matched_corruption_sample_id=negative_sample.sample_id,
        target_metric_name="foot_sliding_cm_per_contact_interval",
        target_metric_delta=delta,
        extra_metadata={
            "counterfactual_group_id": counterfactual_group_id,
            "matched_corruption_mechanism": result.corruption_mechanism,
            "matched_observed_severity": result.measured_severity,
            "severity_label": negative_sample.metadata.get("severity_label"),
            "severity_bin": negative_sample.metadata.get("severity_bin"),
            "quality_equivalence_basis": {
                "target_metric_delta": delta,
                "maximum_allowed_target_metric_delta": maximum_sham_delta_cm,
                "root_intervention_matches_negative": fallback_reason is None,
                "contact_compensation": (
                    "two_bone_ik_foot_lock"
                    if fallback_reason is None
                    else "reversible_roundtrip_fallback"
                ),
                "fallback_reason": fallback_reason,
            },
            **dict(gait_metadata),
        },
    )


def _record_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{prefix}:sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _consistency_records(
    samples: Sequence[TrainingSample],
    *,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    """Create targeted equality/contrast records without aligning whole-motion latents."""
    records: list[dict[str, Any]] = []
    sample_by_id = {sample.sample_id: sample for sample in samples}
    clean_by_group = {
        str(sample.metadata.get("counterfactual_group_id")): sample
        for sample in samples
        if sample.metadata.get("sample_role") == "clean"
    }
    comparable: dict[tuple[str, str, str, str], list[TrainingSample]] = {}
    for sample in samples:
        if sample.metadata.get("sample_role") != "hard_corruption":
            continue
        key = (
            str(sample.metadata.get("counterfactual_group_id")),
            str(sample.metadata.get("corruption_family")),
            str(sample.metadata.get("postcondition_metric")),
            str(sample.metadata.get("severity_label")),
        )
        comparable.setdefault(key, []).append(sample)
    for (group_id, family, metric, label), candidates in sorted(comparable.items()):
        ordered = sorted(candidates, key=lambda item: str(item.metadata["corruption_mechanism"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.metadata["corruption_mechanism"] == right.metadata["corruption_mechanism"]:
                    continue
                left_value = float(left.metadata["measured_severity"])
                right_value = float(right.metadata["measured_severity"])
                relative_difference = abs(left_value - right_value) / max(
                    abs(left_value), abs(right_value), 1.0e-12
                )
                if relative_difference > relative_tolerance:
                    continue
                payload = {
                    "target_type": "per_defect_severity_equal",
                    "counterfactual_group_id": group_id,
                    "family": family,
                    "metric_name": metric,
                    "severity_bin": label,
                    "sample_a_id": left.sample_id,
                    "sample_b_id": right.sample_id,
                    "sample_a_measured_severity": left_value,
                    "sample_b_measured_severity": right_value,
                    "relative_difference": relative_difference,
                    "maximum_relative_difference": relative_tolerance,
                    "whole_motion_latent_alignment": False,
                }
                records.append({"consistency_id": _record_id("consistency", payload), **payload})

    for sham in samples:
        if sham.metadata.get("sample_role") != "sham_control":
            continue
        group_id = str(sham.metadata.get("counterfactual_group_id"))
        clean = clean_by_group[group_id]
        matched_id = str(sham.metadata["matched_corruption_sample_id"])
        matched = sample_by_id[matched_id]
        equality_payload = {
            "target_type": "approximately_equal_quality",
            "counterfactual_group_id": group_id,
            "family": sham.metadata.get("target_defect_family"),
            "sample_a_id": clean.sample_id,
            "sample_b_id": sham.sample_id,
            "whole_motion_latent_alignment": False,
        }
        records.append(
            {
                "consistency_id": _record_id("consistency", equality_payload),
                **equality_payload,
            }
        )
        contrast_payload = {
            "target_type": "matched_sham_defect_contrast",
            "counterfactual_group_id": group_id,
            "family": sham.metadata.get("target_defect_family"),
            "corrupted_sample_id": matched.sample_id,
            "sham_sample_id": sham.sample_id,
            "matched_severity_bin": matched.metadata.get("severity_label"),
        }
        records.append(
            {
                "consistency_id": _record_id("consistency", contrast_payload),
                **contrast_payload,
            }
        )
    return sorted(records, key=lambda record: str(record["consistency_id"]))


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "p50": None, "mean": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "p50": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _dataset_diagnostics(
    records: Sequence[Mapping[str, Any]],
    mechanisms: Sequence[AutomaticMechanism],
    *,
    source_count: int,
    attempts: Counter[str],
    accepted: Counter[str],
    postcondition_passed: Counter[str],
    search_evaluations: Counter[str],
    rejection_reasons: Mapping[str, Counter[str]],
    collateral_attempts: Counter[str],
    collateral_rejections: Counter[str],
) -> dict[str, Any]:
    mechanism_reports: dict[str, Any] = {}
    for mechanism in mechanisms:
        relevant = [
            record
            for record in records
            if record.get("catalog_mechanism_key") == mechanism.key
            and record.get("sample_role") in {"hard_corruption", "soft_corruption"}
        ]
        bins = Counter(str(record.get("severity_label")) for record in relevant)
        gait = Counter(
            str(label) for record in relevant for label in record.get("gait_event_labels", [])
        )
        gait_phase = Counter(
            str(label) for record in relevant for label in record.get("gait_phase_bins", [])
        )
        sides = Counter(
            str(record["target_side"])
            for record in relevant
            if record.get("target_side") in {"left", "right"}
        )
        source_clips = {str(record["source_clip_id"]) for record in relevant}
        target_labels = (
            SEVERITY_LABELS
            if mechanism.hard_negative
            else tuple(f"soft_{index + 1}" for index in range(len(mechanism.soft_parameter_values)))
        )
        attempted = attempts[mechanism.key]
        collateral_checked = collateral_attempts[mechanism.key]
        mechanism_reports[mechanism.key] = {
            "attempted": attempted,
            "accepted": accepted[mechanism.key],
            "rejected": attempted - accepted[mechanism.key],
            "acceptance_fraction": (accepted[mechanism.key] / attempted if attempted else 0.0),
            "postcondition_pass_count": postcondition_passed[mechanism.key],
            "postcondition_pass_rate": (
                postcondition_passed[mechanism.key] / attempted if attempted else 0.0
            ),
            "parameter_search_evaluations": search_evaluations[mechanism.key],
            "rejection_reasons": dict(sorted(rejection_reasons.get(mechanism.key, {}).items())),
            "observed_severity_distribution": _distribution(
                [float(record["measured_severity"]) for record in relevant]
            ),
            "severity_bin_counts": {label: bins[label] for label in target_labels},
            "severity_bins_populated": {label: bins[label] > 0 for label in target_labels},
            "all_severity_bins_populated": all(bins[label] > 0 for label in target_labels),
            "collateral_validation_count": collateral_checked,
            "collateral_defect_rejection_count": collateral_rejections[mechanism.key],
            "collateral_defect_rejection_rate": (
                collateral_rejections[mechanism.key] / collateral_checked
                if collateral_checked
                else 0.0
            ),
            "source_clip_count": len(source_clips),
            "source_clip_coverage_fraction": (
                len(source_clips) / source_count if source_count else 0.0
            ),
            "gait_event_coverage": dict(sorted(gait.items())),
            "gait_phase_coverage": dict(sorted(gait_phase.items())),
            "side_coverage": dict(sorted(sides.items())),
            "sham_control_count": sum(
                record.get("sample_role") == "sham_control"
                and record.get("matched_corruption_mechanism") == mechanism.key.split("/", 1)[1]
                for record in records
            ),
        }
    family_reports: dict[str, Any] = {}
    for family in sorted({mechanism.key.split("/", 1)[0] for mechanism in mechanisms}):
        selected = [item for item in mechanisms if item.key.startswith(f"{family}/")]
        accepted_mechanisms = [
            item.key for item in selected if mechanism_reports[item.key]["accepted"] > 0
        ]
        family_reports[family] = {
            "selected_mechanism_count": len(selected),
            "accepted_mechanism_count": len(accepted_mechanisms),
            "accepted_mechanisms": accepted_mechanisms,
            "heldout_mechanism_selected": any(
                item.partition == "heldout_eval" for item in selected
            ),
        }
    return {
        "mechanisms": mechanism_reports,
        "families": family_reports,
        "production_generalization_inferred": False,
    }


def generate_corruption_dataset(
    source_paths: Sequence[Path],
    output_directory: Path,
    *,
    config: DatasetGenerationConfig | None = None,
) -> dict[str, Any]:
    """Generate or resume a deterministic fixed-rig corruption dataset."""
    settings = DatasetGenerationConfig() if config is None else config
    mechanisms = _mechanisms(settings)
    sources = _load_sources(source_paths, settings)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    config_payload = _config_payload(settings)
    source_state = [
        {
            "path": str(path),
            "motion_id": clip.content_hash,
            "split_lineage_id": provenance_from_clip(clip).split_lineage_id,
            "split": provenance_from_clip(clip).split,
        }
        for path, clip in sources
    ]
    state = {**config_payload, "sources": source_state}
    state_text = json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n"
    state_path = output / "generation_state.json"
    if state_path.exists() and state_path.read_text(encoding="utf-8") != state_text:
        raise ValueError("existing dataset generation state differs from this invocation")
    _atomic_write(state_path, state_text)

    split_registry = {
        provenance_from_clip(clip).split_lineage_id: provenance_from_clip(clip).split
        for _, clip in sources
    }
    _atomic_write(
        output / "splits.json",
        json.dumps(
            {
                "format_version": DATASET_GENERATION_VERSION,
                "assignment_unit": "source_take_split_lineage",
                "splits": split_registry,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    records: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    all_samples: list[TrainingSample] = []
    rejection_counter: Counter[str] = Counter()
    rejection_reasons: dict[str, Counter[str]] = {}
    attempts: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    postcondition_passed: Counter[str] = Counter()
    search_evaluations: Counter[str] = Counter()
    collateral_attempts: Counter[str] = Counter()
    collateral_rejections: Counter[str] = Counter()
    reused_count = 0
    window_count = 0

    for source_path, source in sources:
        provenance = provenance_from_clip(source)
        assert provenance.split is not None
        for window_start in _window_starts(source, settings):
            window_count += 1
            window = source.slice_frames(window_start, window_start + settings.window_frames)
            counterfactual_group_id = _counterfactual_group_id(window, window_start)
            clean_sample = clean_training_sample(
                window,
                source_window_start=window_start,
                extra_metadata={"counterfactual_group_id": counterfactual_group_id},
            )
            clean_path = (
                output / "samples" / provenance.split / f"{clean_sample.sample_id[14:]}.npz"
            )
            reused_count += int(_save_or_reuse(clean_path, clean_sample))
            records.append(
                _sample_record(
                    output,
                    clean_path,
                    clean_sample,
                    data_split=provenance.split,
                    source_path=source_path,
                )
            )
            all_samples.append(clean_sample)

            for yaw_degrees in settings.equivalent_global_yaw_degrees:
                equivalent_seed = _stable_seed(
                    settings.seed,
                    window.content_hash,
                    "equivalent_global_yaw",
                    yaw_degrees,
                )
                equivalent_motion = _global_yaw_equivalent(window, yaw_degrees, equivalent_seed)
                equivalent_sample = equivalent_training_sample(
                    window,
                    equivalent_motion,
                    source_window_start=window_start,
                    control_mechanism="global_heading_rotation",
                    seed=equivalent_seed,
                    transform_parameters={"angle_degrees": yaw_degrees},
                    extra_metadata={"counterfactual_group_id": counterfactual_group_id},
                )
                equivalent_path = (
                    output
                    / "samples"
                    / provenance.split
                    / f"{equivalent_sample.sample_id[14:]}.npz"
                )
                reused_count += int(_save_or_reuse(equivalent_path, equivalent_sample))
                records.append(
                    _sample_record(
                        output,
                        equivalent_path,
                        equivalent_sample,
                        data_split=provenance.split,
                        source_path=source_path,
                    )
                )
                all_samples.append(equivalent_sample)
                audit_records.append(
                    {
                        "attempt_kind": "equivalent_control",
                        "source_clip": provenance.source_clip_id,
                        "source_motion_id": source.content_hash,
                        "source_split": provenance.split,
                        "source_window_start": window_start,
                        "defect_family": None,
                        "corruption_mechanism": "global_heading_rotation",
                        "catalog_partition": "equivalent",
                        "severity_bin": None,
                        "side": None,
                        "gait_events": [],
                        "accepted": True,
                        "rejection_reason": None,
                        "rejection_detail": None,
                        "collateral_defect_rejection": False,
                        "actual_observed_severity": 0.0,
                        "angle_degrees": yaw_degrees,
                    }
                )

            accepted_train_results: dict[str, dict[str, CorruptionResult]] = {}
            for mechanism in mechanisms:
                results: list[CorruptionResult] = []
                labels: list[str] = []
                findings_by_motion: dict[str, tuple[CollateralFinding, ...]] = {}
                if mechanism.hard_negative:
                    assert mechanism.severity_scale is not None
                    targets: Sequence[ObservedSeverityBin | tuple[str, float]] = (
                        mechanism.severity_scale.bins
                    )
                else:
                    targets = tuple(
                        (f"soft_{index + 1}", parameter)
                        for index, parameter in enumerate(mechanism.soft_parameter_values)
                    )
                for target in targets:
                    if isinstance(target, ObservedSeverityBin):
                        severity_label = target.label
                        severity_parameter = None
                    else:
                        severity_label, severity_parameter = target
                    audit_base: dict[str, Any] = {
                        "attempt_kind": "corruption",
                        "source_clip": provenance.source_clip_id,
                        "source_motion_id": source.content_hash,
                        "source_split": provenance.split,
                        "source_window_start": window_start,
                        "defect_family": mechanism.key.split("/", 1)[0],
                        "corruption_mechanism": mechanism.key,
                        "catalog_partition": mechanism.partition,
                        "severity_bin": severity_label,
                        "side": None,
                        "gait_events": [],
                    }
                    if isinstance(target, ObservedSeverityBin):
                        audit_base.update(
                            {
                                "target_observed_severity_lower": target.lower,
                                "target_observed_severity_upper": target.upper,
                                "target_observed_severity_midpoint": (target.lower + target.upper)
                                / 2.0,
                            }
                        )
                    attempts[mechanism.key] += 1
                    corruption_seed = _stable_seed(
                        settings.seed,
                        window.content_hash,
                        mechanism.key,
                        severity_label,
                    )
                    try:
                        if isinstance(target, ObservedSeverityBin):
                            assert mechanism.severity_scale is not None
                            assert mechanism.parameter_bounds is not None
                            targeted = target_observed_severity(
                                window,
                                generator=mechanism.generate,
                                severity_scale=mechanism.severity_scale,
                                severity_bin=target,
                                parameter_bounds=mechanism.parameter_bounds,
                                seed=corruption_seed,
                                maximum_evaluations=settings.severity_search_evaluations,
                                strategy=mechanism.search_strategy,
                                parameter_quantum=mechanism.parameter_quantum,
                            )
                            search_evaluations[mechanism.key] += len(targeted.evaluated_parameters)
                            result = targeted.result
                            postcondition_passed[mechanism.key] += 1
                            collateral_attempts[mechanism.key] += 1
                            policy = single_defect_policy(
                                primary_family=result.corruption_family,
                                primary_metric=result.postcondition.metric_name,
                                required_primary_delta=target.lower,
                            )
                            result, collateral_findings = validate_single_defect(result, policy)
                            findings_by_motion[result.corrupted_motion.content_hash] = (
                                collateral_findings
                            )
                        else:
                            assert severity_parameter is not None
                            result = mechanism.generate(
                                window, severity_parameter, corruption_seed, True
                            )
                            postcondition_passed[mechanism.key] += int(result.postcondition.passed)
                    except CollateralDefectError as exc:
                        collateral_rejections[mechanism.key] += 1
                        rejection_reasons.setdefault(mechanism.key, Counter())[
                            "collateral_defect"
                        ] += 1
                        rejection_counter[f"{mechanism.key}: collateral_defect: {exc}"] += 1
                        audit_records.append(
                            {
                                **audit_base,
                                "accepted": False,
                                "rejection_reason": "collateral_defect",
                                "rejection_detail": str(exc),
                                "collateral_defect_rejection": True,
                                "actual_observed_severity": None,
                            }
                        )
                        continue
                    except SeverityTargetingError as exc:
                        search_evaluations[mechanism.key] += settings.severity_search_evaluations
                        rejection_reasons.setdefault(mechanism.key, Counter())[
                            "severity_bin_unreached"
                        ] += 1
                        rejection_counter[f"{mechanism.key}: severity_bin_unreached: {exc}"] += 1
                        audit_records.append(
                            {
                                **audit_base,
                                "accepted": False,
                                "rejection_reason": "severity_bin_unreached",
                                "rejection_detail": str(exc),
                                "collateral_defect_rejection": False,
                                "actual_observed_severity": None,
                            }
                        )
                        continue
                    except (ValueError, IneffectiveCorruptionError) as exc:
                        rejection_reasons.setdefault(mechanism.key, Counter())["generation"] += 1
                        rejection_counter[f"{mechanism.key}: {type(exc).__name__}: {exc}"] += 1
                        audit_records.append(
                            {
                                **audit_base,
                                "accepted": False,
                                "rejection_reason": "generation",
                                "rejection_detail": f"{type(exc).__name__}: {exc}",
                                "collateral_defect_rejection": False,
                                "actual_observed_severity": None,
                            }
                        )
                        continue
                    gait_metadata = _gait_target_metadata(window, result)
                    target_midpoint = audit_base.get("target_observed_severity_midpoint")
                    audit_records.append(
                        {
                            **audit_base,
                            "accepted": True,
                            "rejection_reason": None,
                            "rejection_detail": None,
                            "collateral_defect_rejection": False,
                            "actual_observed_severity": result.measured_severity,
                            "requested_severity_parameter": result.requested_severity_parameter,
                            "absolute_target_midpoint_error": (
                                None
                                if target_midpoint is None
                                else abs(result.measured_severity - float(target_midpoint))
                            ),
                            "side": gait_metadata.get("target_side"),
                            "gait_events": gait_metadata.get("gait_event_labels", []),
                        }
                    )
                    results.append(result)
                    labels.append(severity_label)
                    accepted[mechanism.key] += 1
                chain_id: str | None = None
                if mechanism.hard_negative and len(results) == len(SEVERITY_LABELS):
                    try:
                        chain = verify_ordinal_chain(results)
                    except ValueError as exc:
                        rejection_reasons.setdefault(mechanism.key, Counter())[
                            "ordinal_verification"
                        ] += 1
                        rejection_counter[f"{mechanism.key}: ordinal: {exc}"] += 1
                    else:
                        chain_payload = {
                            "source_motion_id": chain.source_motion_id,
                            "mechanism": mechanism.key,
                            "ordered_motion_ids": chain.ordered_motion_ids,
                            "metric": chain.metric_name,
                            "values": chain.measured_values,
                        }
                        encoded = json.dumps(chain_payload, sort_keys=True, separators=(",", ":"))
                        chain_id = f"ordinal:sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"
                mechanism_samples: list[TrainingSample] = []
                for rank, (severity_label, result) in enumerate(
                    zip(labels, results, strict=True), start=1
                ):
                    severity_target = result.schema_metadata.get("severity_targeting")
                    gait_metadata = _gait_target_metadata(window, result)
                    collateral_findings = findings_by_motion.get(
                        result.corrupted_motion.content_hash, ()
                    )
                    sample = corruption_training_sample(
                        result,
                        source_window_start=window_start,
                        extra_metadata={
                            "severity_label": severity_label,
                            "catalog_mechanism_key": mechanism.key,
                            "severity_bin": severity_target,
                            "ordinal_chain_id": chain_id,
                            "ordinal_rank": rank if chain_id is not None else None,
                            "counterfactual_group_id": counterfactual_group_id,
                            "collateral_defect_families": sorted(
                                finding.defect_family for finding in collateral_findings
                            ),
                            "single_defect": not collateral_findings,
                            **gait_metadata,
                        },
                    )
                    data_split = (
                        "heldout_corruptor"
                        if mechanism.partition == "heldout_eval"
                        else provenance.split
                    )
                    sample_path = output / "samples" / data_split / f"{sample.sample_id[14:]}.npz"
                    reused_count += int(_save_or_reuse(sample_path, sample))
                    records.append(
                        _sample_record(
                            output,
                            sample_path,
                            sample,
                            data_split=data_split,
                            source_path=source_path,
                        )
                    )
                    all_samples.append(sample)
                    mechanism_samples.append(sample)
                    if settings.include_sham_controls and mechanism.key == "foot_slide/root_drift":
                        sham_audit_base = {
                            "attempt_kind": "sham",
                            "source_clip": provenance.source_clip_id,
                            "source_motion_id": source.content_hash,
                            "source_split": provenance.split,
                            "source_window_start": window_start,
                            "defect_family": mechanism.key.split("/", 1)[0],
                            "corruption_mechanism": mechanism.key,
                            "catalog_partition": mechanism.partition,
                            "severity_bin": severity_label,
                            "side": gait_metadata.get("target_side"),
                            "gait_events": gait_metadata.get("gait_event_labels", []),
                        }
                        if isinstance(severity_target, Mapping):
                            sham_audit_base.update(
                                {
                                    "target_observed_severity_lower": severity_target.get(
                                        "bin_lower_inclusive"
                                    ),
                                    "target_observed_severity_upper": severity_target.get(
                                        "bin_upper_exclusive"
                                    ),
                                }
                            )
                        sham_seed = _stable_seed(
                            settings.seed,
                            window.content_hash,
                            mechanism.key,
                            severity_label,
                            "sham",
                        )
                        try:
                            sham = _matched_foot_slide_sham(
                                window,
                                result,
                                sample,
                                source_window_start=window_start,
                                counterfactual_group_id=counterfactual_group_id,
                                seed=sham_seed,
                                gait_metadata=gait_metadata,
                            )
                        except ValueError as exc:
                            rejection_reasons.setdefault(mechanism.key, Counter())[
                                "sham_control"
                            ] += 1
                            rejection_counter[f"{mechanism.key}: sham_control: {exc}"] += 1
                            audit_records.append(
                                {
                                    **sham_audit_base,
                                    "accepted": False,
                                    "rejection_reason": "sham_control",
                                    "rejection_detail": str(exc),
                                    "collateral_defect_rejection": False,
                                    "actual_observed_severity": None,
                                    "target_symptom_absent": False,
                                }
                            )
                        else:
                            sham_path = (
                                output / "samples" / provenance.split / f"{sham.sample_id[14:]}.npz"
                            )
                            reused_count += int(_save_or_reuse(sham_path, sham))
                            records.append(
                                _sample_record(
                                    output,
                                    sham_path,
                                    sham,
                                    data_split=provenance.split,
                                    source_path=source_path,
                                )
                            )
                            all_samples.append(sham)
                            audit_records.append(
                                {
                                    **sham_audit_base,
                                    "accepted": True,
                                    "rejection_reason": None,
                                    "rejection_detail": None,
                                    "collateral_defect_rejection": False,
                                    "actual_observed_severity": float(
                                        sham.metadata["sham_target_metric_delta"]
                                    ),
                                    "target_symptom_absent": bool(
                                        sham.arrays["no_target_defect"].item()
                                    ),
                                }
                            )
                if chain_id is not None:
                    preferences.extend(
                        _preference_records(clean_sample, mechanism_samples, results, chain_id)
                    )
                if mechanism.partition == "train" and mechanism.hard_negative:
                    accepted_train_results[mechanism.key] = dict(zip(labels, results, strict=True))

            composition_draw = _stable_seed(settings.seed, window.content_hash, "composition") / (
                2**32 - 1
            )
            if (
                settings.composition_fraction > 0.0
                and composition_draw < settings.composition_fraction
            ):
                speed_mechanism = next(
                    item
                    for item in AUTOMATIC_MECHANISMS
                    if item.key == "speed_inconsistency/root_progression_scale"
                )
                jitter_mechanism = next(
                    item
                    for item in AUTOMATIC_MECHANISMS
                    if item.key == "joint_jitter/white_tangent_noise"
                )
                if (
                    speed_mechanism.key in accepted_train_results
                    and jitter_mechanism.key in accepted_train_results
                    and "moderate" in accepted_train_results[speed_mechanism.key]
                    and "moderate" in accepted_train_results[jitter_mechanism.key]
                ):
                    composition_seed = _stable_seed(
                        settings.seed, window.content_hash, "composition"
                    )
                    speed_parameter = accepted_train_results[speed_mechanism.key][
                        "moderate"
                    ].requested_severity_parameter
                    jitter_parameter = accepted_train_results[jitter_mechanism.key][
                        "moderate"
                    ].requested_severity_parameter

                    def speed_operator(
                        value: MotionClip,
                        spec: AutomaticMechanism = speed_mechanism,
                        operation_seed: int = composition_seed,
                        parameter: float = speed_parameter,
                    ) -> CorruptionResult:
                        return spec.generate(
                            value,
                            parameter,
                            operation_seed,
                            True,
                        )

                    def jitter_operator(
                        value: MotionClip,
                        spec: AutomaticMechanism = jitter_mechanism,
                        operation_seed: int = composition_seed,
                        parameter: float = jitter_parameter,
                    ) -> CorruptionResult:
                        return spec.generate(
                            value,
                            parameter,
                            _stable_seed(operation_seed, "jitter"),
                            True,
                        )

                    attempts["composite"] += 1
                    try:
                        composed = compose_corruptions(
                            window,
                            (speed_operator, jitter_operator),
                            seed=composition_seed,
                        )
                    except (ValueError, IneffectiveCorruptionError) as exc:
                        rejection_counter[f"composite: {type(exc).__name__}: {exc}"] += 1
                    else:
                        accepted["composite"] += 1
                        sample = corruption_training_sample(
                            composed,
                            source_window_start=window_start,
                            extra_metadata={
                                "severity_label": "composed_medium",
                                "severity_bin": None,
                                "ordinal_chain_id": None,
                                "ordinal_rank": None,
                                "counterfactual_group_id": counterfactual_group_id,
                                "collateral_defect_families": [],
                                "single_defect": False,
                                "gait_event_labels": ["gait_cycle"],
                                "target_side": None,
                            },
                        )
                        sample_path = (
                            output / "samples" / provenance.split / f"{sample.sample_id[14:]}.npz"
                        )
                        reused_count += int(_save_or_reuse(sample_path, sample))
                        records.append(
                            _sample_record(
                                output,
                                sample_path,
                                sample,
                                data_split=provenance.split,
                                source_path=source_path,
                            )
                        )
                        all_samples.append(sample)

    if not any(record["data_split"] == "train" for record in records):
        raise ValueError("generated dataset contains no training samples")
    if any(
        record["data_split"] == "train" and record["catalog_partition"] == "heldout_eval"
        for record in records
    ):
        raise AssertionError("held-out corruption mechanism leaked into training")
    records.sort(key=lambda record: str(record["sample_id"]))
    preferences.sort(key=lambda record: str(record["preference_id"]))
    audit_records.sort(
        key=lambda record: (
            str(record.get("source_clip")),
            int(record.get("source_window_start", 0)),
            str(record.get("corruption_mechanism")),
            str(record.get("severity_bin")),
            str(record.get("attempt_kind")),
        )
    )
    consistency = _consistency_records(
        all_samples,
        relative_tolerance=settings.cross_mechanism_relative_tolerance,
    )
    manifest_path = output / "manifest.jsonl"
    preference_path = output / "preferences.jsonl"
    consistency_path = output / "consistency.jsonl"
    audit_path = output / "audit.jsonl"
    _atomic_write(manifest_path, _json_lines(records))
    _atomic_write(preference_path, _json_lines(preferences))
    _atomic_write(consistency_path, _json_lines(consistency))
    _atomic_write(audit_path, _json_lines(audit_records))
    normalization = fit_training_normalization(all_samples)
    normalization_path = save_normalization_statistics(output, normalization)

    by_split = Counter(str(record["data_split"]) for record in records)
    by_family = Counter(
        "clean" if record["corruption_family"] is None else str(record["corruption_family"])
        for record in records
    )
    by_role = Counter(str(record["sample_role"]) for record in records)
    pass_rates = {
        key: {
            "attempted": attempts[key],
            "accepted": accepted[key],
            "rejected": attempts[key] - accepted[key],
            "postcondition_passed": postcondition_passed[key],
            "postcondition_pass_fraction": (
                postcondition_passed[key] / attempts[key] if attempts[key] else 0.0
            ),
            "acceptance_fraction": accepted[key] / attempts[key] if attempts[key] else 0.0,
        }
        for key in sorted(attempts)
    }
    summary = {
        **config_payload,
        "source_count": len(sources),
        "source_window_count": window_count,
        "sample_count": len(records),
        "preference_pair_count": len(preferences),
        "consistency_target_count": len(consistency),
        "counterfactual_group_count": len(
            {str(record["counterfactual_group_id"]) for record in records}
        ),
        "sham_control_count": by_role["sham_control"],
        "split_lineage_count": len(split_registry),
        "counts_by_data_split": dict(sorted(by_split.items())),
        "counts_by_corruption_family": dict(sorted(by_family.items())),
        "counts_by_sample_role": dict(sorted(by_role.items())),
        "mechanism_postcondition_pass_rates": pass_rates,
        "dataset_diagnostics": _dataset_diagnostics(
            records,
            mechanisms,
            source_count=len(sources),
            attempts=attempts,
            accepted=accepted,
            postcondition_passed=postcondition_passed,
            search_evaluations=search_evaluations,
            rejection_reasons=rejection_reasons,
            collateral_attempts=collateral_attempts,
            collateral_rejections=collateral_rejections,
        ),
        "audit_breakdowns": _audit_breakdowns(audit_records),
        "rejections": dict(sorted(rejection_counter.items())),
        "heldout_mechanisms_absent_from_training": True,
        "normalization_fit_split": "train_only",
        "files": {
            "manifest": {"path": manifest_path.name, "sha256": file_sha256(manifest_path)},
            "preferences": {
                "path": preference_path.name,
                "sha256": file_sha256(preference_path),
            },
            "consistency": {
                "path": consistency_path.name,
                "sha256": file_sha256(consistency_path),
            },
            "audit": {"path": audit_path.name, "sha256": file_sha256(audit_path)},
            "splits": {
                "path": "splits.json",
                "sha256": file_sha256(output / "splits.json"),
            },
            "normalization": {
                "path": normalization_path.name,
                "sha256": file_sha256(normalization_path),
            },
        },
    }
    _atomic_write(
        output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return {**summary, "resumed_existing_sample_count": reused_count}


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read dataset JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"dataset JSON must contain an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read dataset JSONL: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    return records


def validate_corruption_dataset(directory: Path) -> dict[str, Any]:
    """Verify dataset manifests, samples, split lineage, normalization, and holdout isolation."""
    root = Path(directory).resolve()
    summary = _load_json_object(root / "summary.json")
    if summary.get("format_version") != DATASET_GENERATION_VERSION:
        raise ValueError("unsupported corruption dataset version")
    files = summary.get("files")
    if not isinstance(files, dict):
        raise ValueError("dataset summary files declaration is invalid")
    required_files = {
        "manifest": "manifest.jsonl",
        "preferences": "preferences.jsonl",
        "consistency": "consistency.jsonl",
        "audit": "audit.jsonl",
        "splits": "splits.json",
        "normalization": "normalization.json",
    }
    paths: dict[str, Path] = {}
    for name, expected_name in required_files.items():
        record = files.get(name)
        if not isinstance(record, dict) or record.get("path") != expected_name:
            raise ValueError(f"dataset summary declaration is invalid for {name!r}")
        path = root / expected_name
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"dataset checksum mismatch for {name!r}")
        paths[name] = path
    manifest = _load_jsonl(paths["manifest"])
    preferences = _load_jsonl(paths["preferences"])
    consistency = _load_jsonl(paths["consistency"])
    audit = _load_jsonl(paths["audit"])
    splits_payload = _load_json_object(paths["splits"])
    splits = splits_payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("dataset split registry is invalid")
    sample_ids: set[str] = set()
    counterfactual_lineage: dict[str, str] = {}
    for record in manifest:
        sample_id = record.get("sample_id")
        relative = record.get("path")
        if not isinstance(sample_id, str) or sample_id in sample_ids:
            raise ValueError("dataset manifest contains an invalid or duplicate sample ID")
        if not isinstance(relative, str):
            raise ValueError("dataset manifest sample path must be a string")
        sample_path = (root / relative).resolve()
        if not sample_path.is_relative_to(root / "samples"):
            raise ValueError("dataset sample path escapes the samples directory")
        if file_sha256(sample_path) != record.get("sha256"):
            raise ValueError(f"dataset sample checksum mismatch: {sample_id}")
        sample = load_training_sample(sample_path)
        if sample.sample_id != sample_id:
            raise ValueError("dataset sample ID does not match its manifest record")
        lineage = sample.metadata.get("split_lineage_id")
        source_split = sample.metadata.get("split")
        if not isinstance(lineage, str) or splits.get(lineage) != source_split:
            raise ValueError("dataset sample violates its source split-lineage assignment")
        if (
            record.get("data_split") == "train"
            and sample.metadata.get("catalog_partition") == "heldout_eval"
        ):
            raise ValueError("held-out corruption mechanism appears in training")
        counterfactual_group = sample.metadata.get("counterfactual_group_id")
        if not isinstance(counterfactual_group, str) or not counterfactual_group.startswith(
            "counterfactual:sha256:"
        ):
            raise ValueError("dataset sample has no valid counterfactual group ID")
        previous_lineage = counterfactual_lineage.setdefault(counterfactual_group, lineage)
        if previous_lineage != lineage:
            raise ValueError("one counterfactual group crosses split lineage")
        role = sample.metadata.get("sample_role")
        if role not in {
            "clean",
            "equivalent_control",
            "hard_corruption",
            "soft_corruption",
            "sham_control",
            "composite",
        }:
            raise ValueError("dataset sample has an invalid sample role")
        if role in {"equivalent_control", "sham_control"} and (
            not bool(sample.arrays["no_target_defect"].item())
            or not bool(sample.arrays["approximately_equal_quality"].item())
        ):
            raise ValueError("control is missing explicit no-defect/equal-quality targets")
        if role == "hard_corruption":
            severity_bin = sample.metadata.get("severity_bin")
            measured = sample.metadata.get("measured_severity")
            if not isinstance(severity_bin, dict) or not isinstance(measured, (int, float)):
                raise ValueError("hard corruption has no observed-severity bin declaration")
            lower = severity_bin.get("bin_lower_inclusive")
            upper = severity_bin.get("bin_upper_exclusive")
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                raise ValueError("hard corruption severity-bin bounds are invalid")
            if not float(lower) <= float(measured) < float(upper):
                raise ValueError("hard corruption measured severity is outside its target bin")
        sample_ids.add(sample_id)
    if len(manifest) != summary.get("sample_count"):
        raise ValueError("dataset sample count does not match the summary")
    for record in preferences:
        if (
            record.get("better_sample_id") not in sample_ids
            or record.get("worse_sample_id") not in sample_ids
        ):
            raise ValueError("preference record references an unknown sample")
    if len(preferences) != summary.get("preference_pair_count"):
        raise ValueError("dataset preference count does not match the summary")
    for record in consistency:
        referenced = {
            str(record[field])
            for field in (
                "sample_a_id",
                "sample_b_id",
                "corrupted_sample_id",
                "sham_sample_id",
            )
            if field in record
        }
        if not referenced or not referenced.issubset(sample_ids):
            raise ValueError("consistency record references an unknown sample")
    if len(consistency) != summary.get("consistency_target_count"):
        raise ValueError("dataset consistency-target count does not match the summary")
    if _audit_breakdowns(audit) != summary.get("audit_breakdowns"):
        raise ValueError("dataset audit breakdowns do not match the attempt records")
    for record in manifest:
        matched = record.get("matched_corruption_sample_id")
        if matched is not None and matched not in sample_ids:
            raise ValueError("sham control references an unknown matched corruption")
    normalization = load_normalization_statistics(root)
    expected_training_ids = {
        str(record["sample_id"])
        for record in manifest
        if record.get("source_split") == "train"
        and record.get("catalog_partition") in {"clean", "equivalent", "train", "sham"}
    }
    if set(normalization.training_sample_ids) != expected_training_ids:
        raise ValueError("normalization was not fit on exactly the declared training samples")
    return {
        "format_version": DATASET_GENERATION_VERSION,
        "sample_count": len(manifest),
        "preference_pair_count": len(preferences),
        "consistency_target_count": len(consistency),
        "counterfactual_group_count": len(counterfactual_lineage),
        "sham_control_count": sum(
            record.get("sample_role") == "sham_control" for record in manifest
        ),
        "training_normalization_sample_count": len(normalization.training_sample_ids),
        "split_lineage_count": len(splits),
        "heldout_mechanisms_absent_from_training": True,
    }
