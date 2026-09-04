"""Deterministic, measured procedural corruptions and training artifacts."""

from motionlab.corruptions.artifacts import (
    CORRUPTION_ARTIFACT_VERSION,
    CorruptionArtifact,
    load_corruption_artifact,
    save_corruption_artifact,
)
from motionlab.corruptions.base import (
    CORRUPTION_CHANNEL_NAMES,
    CORRUPTION_DEFECT_NAMES,
    CORRUPTION_PART_NAMES,
    ContactSupervision,
    CorruptionResult,
    IneffectiveCorruptionError,
    MeasuredPostcondition,
)
from motionlab.corruptions.catalog import (
    CORRUPTION_CATALOG,
    CORRUPTION_CATALOG_VERSION,
    CorruptionCatalogEntry,
    VerifiedPreferenceChain,
    catalog_entry,
    verify_ordinal_chain,
)
from motionlab.corruptions.composition import CorruptionOperator, compose_corruptions
from motionlab.corruptions.foot_slide import (
    corrupt_foot_slide_hip_rotation_drift,
    corrupt_foot_slide_root_drift,
)
from motionlab.corruptions.physical import (
    corrupt_floating_contact_root_offset,
    corrupt_foot_clearance_ik_lowering,
    corrupt_ground_penetration_root_offset,
    corrupt_joint_limit_excess,
    corrupt_joint_pop_rotation_pulse,
    corrupt_joint_pop_smooth_pulse,
    corrupt_loop_seam_end_rotation,
    corrupt_loop_seam_root_velocity,
)
from motionlab.corruptions.temporal import (
    corrupt_joint_jitter,
    corrupt_joint_jitter_band_limited,
    corrupt_joint_jitter_white,
    corrupt_leg_stride_amplitude,
    corrupt_limb_phase_shift,
    corrupt_pelvis_vertical_curve,
    corrupt_speed_root_progression,
    corrupt_style_region_blend,
    corrupt_time_warp,
)

__all__ = [
    "CORRUPTION_ARTIFACT_VERSION",
    "CORRUPTION_CATALOG",
    "CORRUPTION_CATALOG_VERSION",
    "CORRUPTION_CHANNEL_NAMES",
    "CORRUPTION_DEFECT_NAMES",
    "CORRUPTION_PART_NAMES",
    "ContactSupervision",
    "CorruptionArtifact",
    "CorruptionCatalogEntry",
    "CorruptionOperator",
    "CorruptionResult",
    "IneffectiveCorruptionError",
    "MeasuredPostcondition",
    "VerifiedPreferenceChain",
    "catalog_entry",
    "compose_corruptions",
    "corrupt_floating_contact_root_offset",
    "corrupt_foot_clearance_ik_lowering",
    "corrupt_foot_slide_hip_rotation_drift",
    "corrupt_foot_slide_root_drift",
    "corrupt_ground_penetration_root_offset",
    "corrupt_joint_jitter",
    "corrupt_joint_jitter_band_limited",
    "corrupt_joint_jitter_white",
    "corrupt_joint_limit_excess",
    "corrupt_joint_pop_rotation_pulse",
    "corrupt_joint_pop_smooth_pulse",
    "corrupt_leg_stride_amplitude",
    "corrupt_limb_phase_shift",
    "corrupt_loop_seam_end_rotation",
    "corrupt_loop_seam_root_velocity",
    "corrupt_pelvis_vertical_curve",
    "corrupt_speed_root_progression",
    "corrupt_style_region_blend",
    "corrupt_time_warp",
    "load_corruption_artifact",
    "save_corruption_artifact",
    "verify_ordinal_chain",
]
