"""Versioned corruption catalog and mechanism holdout declarations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from motionlab.corruptions.base import CorruptionResult

CORRUPTION_CATALOG_VERSION = "motionlab.corruption_catalog.v1"


@dataclass(frozen=True)
class CorruptionCatalogEntry:
    """One deterministic mechanism and its dataset role."""

    family: str
    mechanism: str
    callable_name: str
    measured_metric: str
    partition: str
    hard_negative: bool
    description: str

    def __post_init__(self) -> None:
        if self.partition not in {"train", "heldout_eval"}:
            raise ValueError("catalog partition must be 'train' or 'heldout_eval'")


CORRUPTION_CATALOG = (
    CorruptionCatalogEntry(
        family="foot_slide",
        mechanism="root_drift",
        callable_name="corrupt_foot_slide_root_drift",
        measured_metric="foot_sliding_cm_per_contact_interval",
        partition="train",
        hard_negative=True,
        description="Smooth root drift without stance-foot compensation.",
    ),
    CorruptionCatalogEntry(
        family="foot_slide",
        mechanism="hip_rotation_drift",
        callable_name="corrupt_foot_slide_hip_rotation_drift",
        measured_metric="foot_sliding_cm_per_contact_interval",
        partition="heldout_eval",
        hard_negative=True,
        description="Smooth stance-hip rotation reserved for mechanism generalization tests.",
    ),
    CorruptionCatalogEntry(
        family="ground_penetration",
        mechanism="root_vertical_offset",
        callable_name="corrupt_ground_penetration_root_offset",
        measured_metric="maximum_ground_penetration_cm",
        partition="train",
        hard_negative=True,
        description="Smooth root displacement into the authored ground plane.",
    ),
    CorruptionCatalogEntry(
        family="floating_contact",
        mechanism="root_vertical_offset",
        callable_name="corrupt_floating_contact_root_offset",
        measured_metric="maximum_floating_contact_cm",
        partition="train",
        hard_negative=True,
        description="Smooth root lift over source-supported contact frames.",
    ),
    CorruptionCatalogEntry(
        family="joint_pop",
        mechanism="local_rotation_pulse",
        callable_name="corrupt_joint_pop_rotation_pulse",
        measured_metric="localized_target_joint_peak_angular_jerk_rad_s3",
        partition="train",
        hard_negative=True,
        description="Single-sample local rotation producing a measured angular-jerk pop.",
    ),
    CorruptionCatalogEntry(
        family="joint_pop",
        mechanism="smooth_rotation_pulse",
        callable_name="corrupt_joint_pop_smooth_pulse",
        measured_metric="localized_target_joint_peak_angular_jerk_rad_s3",
        partition="heldout_eval",
        hard_negative=True,
        description="Compact zero-boundary rotation pulse reserved for generalization tests.",
    ),
    CorruptionCatalogEntry(
        family="joint_jitter",
        mechanism="white_tangent_noise",
        callable_name="corrupt_joint_jitter_white",
        measured_metric="localized_angular_jerk_p95_rad_s3",
        partition="train",
        hard_negative=True,
        description="Seeded white-ish local tangent noise over selected joints.",
    ),
    CorruptionCatalogEntry(
        family="joint_jitter",
        mechanism="band_limited_correlated",
        callable_name="corrupt_joint_jitter_band_limited",
        measured_metric="localized_angular_jerk_p95_rad_s3",
        partition="heldout_eval",
        hard_negative=True,
        description="Correlated filtered tangent jitter held out from training.",
    ),
    CorruptionCatalogEntry(
        family="limb_phase_mismatch",
        mechanism="unilateral_circular_shift",
        callable_name="corrupt_limb_phase_shift",
        measured_metric="source_relative_arm_phase_deviation_rad_rms",
        partition="train",
        hard_negative=True,
        description="Smooth unilateral arm-channel phase shift relative to gait.",
    ),
    CorruptionCatalogEntry(
        family="limb_phase_mismatch",
        mechanism="bilateral_circular_shift",
        callable_name="corrupt_limb_phase_shift",
        measured_metric="source_relative_arm_phase_deviation_rad_rms",
        partition="heldout_eval",
        hard_negative=True,
        description="Bilateral phase shift held out from mechanism training.",
    ),
    CorruptionCatalogEntry(
        family="cadence_inconsistency",
        mechanism="pose_stream_monotonic_warp",
        callable_name="corrupt_time_warp",
        measured_metric="source_relative_pose_timing_deviation_rad_rms",
        partition="train",
        hard_negative=True,
        description="Monotonic pose-time warp with root trajectory held fixed.",
    ),
    CorruptionCatalogEntry(
        family="cadence_inconsistency",
        mechanism="root_stream_monotonic_warp",
        callable_name="corrupt_time_warp",
        measured_metric="source_relative_root_timing_deviation_cm_rms",
        partition="heldout_eval",
        hard_negative=True,
        description="Monotonic root-time warp with pose channels held fixed.",
    ),
    CorruptionCatalogEntry(
        family="speed_inconsistency",
        mechanism="root_progression_scale",
        callable_name="corrupt_speed_root_progression",
        measured_metric="absolute_target_speed_error_mps",
        partition="train",
        hard_negative=True,
        description="Root progression scaling without corresponding stride adjustment.",
    ),
    CorruptionCatalogEntry(
        family="speed_inconsistency",
        mechanism="leg_pose_amplitude_scale",
        callable_name="corrupt_leg_stride_amplitude",
        measured_metric="source_relative_leg_stride_pose_deviation_rad_rms",
        partition="heldout_eval",
        hard_negative=True,
        description="Leg excursion scaling with root speed held fixed.",
    ),
    CorruptionCatalogEntry(
        family="loop_seam",
        mechanism="end_rotation_offset",
        callable_name="corrupt_loop_seam_end_rotation",
        measured_metric="weighted_loop_seam",
        partition="train",
        hard_negative=True,
        description="Smooth terminal rotation ramp creating an endpoint discontinuity.",
    ),
    CorruptionCatalogEntry(
        family="loop_seam",
        mechanism="root_velocity_mismatch",
        callable_name="corrupt_loop_seam_root_velocity",
        measured_metric="weighted_loop_seam",
        partition="heldout_eval",
        hard_negative=True,
        description="Terminal root-velocity mismatch reserved for seam generalization tests.",
    ),
    CorruptionCatalogEntry(
        family="pelvis_curve",
        mechanism="vertical_amplitude_scale",
        callable_name="corrupt_pelvis_vertical_curve",
        measured_metric="source_relative_pelvis_vertical_deviation_cm_rms",
        partition="train",
        hard_negative=False,
        description="Soft pelvis-bounce edit with automatic preference withheld.",
    ),
    CorruptionCatalogEntry(
        family="foot_clearance",
        mechanism="swing_ankle_ik_lowering",
        callable_name="corrupt_foot_clearance_ik_lowering",
        measured_metric="maximum_source_relative_clearance_loss_cm",
        partition="train",
        hard_negative=True,
        description="Contact-aware swing-ankle lowering through two-bone IK.",
    ),
    CorruptionCatalogEntry(
        family="joint_limit",
        mechanism="authored_limit_excess",
        callable_name="corrupt_joint_limit_excess",
        measured_metric="maximum_joint_limit_excess_deg",
        partition="train",
        hard_negative=True,
        description="Authored-limit-aware local rotation beyond a configured bound.",
    ),
    CorruptionCatalogEntry(
        family="style_incoherence",
        mechanism="aligned_region_blend",
        callable_name="corrupt_style_region_blend",
        measured_metric="source_relative_region_pose_deviation_rad_rms",
        partition="train",
        hard_negative=False,
        description="Aligned donor-region blend with automatic quality preference withheld.",
    ),
)


def catalog_entry(family: str, mechanism: str) -> CorruptionCatalogEntry:
    """Look up a declared mechanism by stable family/mechanism key."""
    for entry in CORRUPTION_CATALOG:
        if entry.family == family and entry.mechanism == mechanism:
            return entry
    raise KeyError(f"unknown corruption catalog entry {family!r}/{mechanism!r}")


@dataclass(frozen=True)
class VerifiedPreferenceChain:
    """An ordinal clean-to-severe chain verified by one measured metric."""

    source_motion_id: str
    ordered_motion_ids: tuple[str, ...]
    metric_name: str
    measured_values: tuple[float, ...]
    confidence: float


def verify_ordinal_chain(results: Sequence[CorruptionResult]) -> VerifiedPreferenceChain:
    """Verify caller-supplied mild-to-severe order from actual post-generation metrics."""
    if not results:
        raise ValueError("an ordinal chain requires at least one corruption")
    first = results[0]
    for result in results:
        if result.source_motion_id != first.source_motion_id:
            raise ValueError("ordinal chain corruptions must share one clean source")
        if (
            result.corruption_family != first.corruption_family
            or result.corruption_mechanism != first.corruption_mechanism
        ):
            raise ValueError("ordinal chain corruptions must share one family and mechanism")
        if result.postcondition.metric_name != first.postcondition.metric_name:
            raise ValueError("ordinal chain corruptions must use one measured metric")
        if not result.postcondition.hard_negative:
            raise ValueError("ordinal chain contains an unverified hard negative")
        if result.postcondition.before != first.postcondition.before:
            raise ValueError("ordinal chain corruptions must share one metric baseline")
    values = tuple(result.postcondition.after for result in results)
    if any(right <= left for left, right in pairwise(values)):
        raise ValueError("measured defect values do not verify the requested ordinal order")
    return VerifiedPreferenceChain(
        source_motion_id=first.source_motion_id,
        ordered_motion_ids=(
            first.clean_motion.content_hash,
            *(result.corrupted_motion.content_hash for result in results),
        ),
        metric_name=first.postcondition.metric_name,
        measured_values=(first.postcondition.before, *values),
        confidence=min(result.preference_confidence for result in results),
    )
