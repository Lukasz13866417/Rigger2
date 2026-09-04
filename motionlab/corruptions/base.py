"""Dense, immutable supervision contract for deterministic corruptions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motionlab.motion._validation import frozen_array, frozen_json_mapping
from motionlab.motion.clip import MotionClip

CORRUPTION_CHANNEL_NAMES = (
    "root_translation_x",
    "root_translation_y",
    "root_translation_z",
    "local_rotation_x",
    "local_rotation_y",
    "local_rotation_z",
)
CORRUPTION_PART_NAMES = (
    "root_pelvis",
    "trunk_spine",
    "head_neck",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "other_accessory",
)
CORRUPTION_DEFECT_NAMES = (
    "foot_slide",
    "ground_penetration",
    "floating_contact",
    "loop_seam",
    "joint_limit",
    "joint_pop",
    "joint_jitter",
    "speed_inconsistency",
    "limb_phase_mismatch",
    "cadence_inconsistency",
    "pelvis_curve",
    "foot_clearance",
    "style_incoherence",
)


class IneffectiveCorruptionError(ValueError):
    """Raised when a requested hard negative fails its measured postcondition."""


@dataclass(frozen=True)
class MeasuredPostcondition:
    """Measured evidence that the intended defect became worse."""

    metric_name: str
    before: float
    after: float
    minimum_worsening: float
    passed: bool
    hard_negative: bool

    def __post_init__(self) -> None:
        values = (self.before, self.after, self.minimum_worsening)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("postcondition values must be finite")
        if self.minimum_worsening < 0.0:
            raise ValueError("minimum_worsening must be nonnegative")
        expected = self.after - self.before >= self.minimum_worsening
        if self.passed != expected:
            raise ValueError("postcondition passed flag does not match its measured values")
        if self.hard_negative and not self.passed:
            raise ValueError("only a passed measured postcondition can be a hard negative")

    @property
    def delta(self) -> float:
        """Return the measured increase in defect severity."""
        return self.after - self.before


@dataclass(frozen=True)
class ContactSupervision:
    """Privileged source contacts kept separate from inference-available estimates."""

    marker_names: tuple[str, ...]
    truth_hard: NDArray[np.bool_]
    truth_confidence: NDArray[np.float32]
    inferred_clean_hard: NDArray[np.bool_]
    inferred_clean_confidence: NDArray[np.float32]
    inferred_corrupted_hard: NDArray[np.bool_]
    inferred_corrupted_confidence: NDArray[np.float32]
    ground_plane: NDArray[np.float32]
    truth_source: str
    inference_source: str
    ground_confidence: float

    def __post_init__(self) -> None:
        marker_names = tuple(self.marker_names)
        if not marker_names or any(not name for name in marker_names):
            raise ValueError("contact marker names must be nonempty")
        if len(set(marker_names)) != len(marker_names):
            raise ValueError("contact marker names must be unique")
        truth_hard = frozen_array(
            self.truth_hard, dtype=np.dtype(np.bool_), name="truth contact mask"
        )
        truth_confidence = frozen_array(
            self.truth_confidence,
            dtype=np.dtype(np.float32),
            name="truth contact confidence",
        )
        inferred_clean_hard = frozen_array(
            self.inferred_clean_hard,
            dtype=np.dtype(np.bool_),
            name="clean inferred contact mask",
        )
        inferred_clean_confidence = frozen_array(
            self.inferred_clean_confidence,
            dtype=np.dtype(np.float32),
            name="clean inferred contact confidence",
        )
        inferred_corrupted_hard = frozen_array(
            self.inferred_corrupted_hard,
            dtype=np.dtype(np.bool_),
            name="corrupted inferred contact mask",
        )
        inferred_corrupted_confidence = frozen_array(
            self.inferred_corrupted_confidence,
            dtype=np.dtype(np.float32),
            name="corrupted inferred contact confidence",
        )
        expected = truth_hard.shape
        if len(expected) != 2 or expected[1] != len(marker_names):
            raise ValueError("contact arrays must have shape [T, number of markers]")
        arrays = (
            truth_confidence,
            inferred_clean_hard,
            inferred_clean_confidence,
            inferred_corrupted_hard,
            inferred_corrupted_confidence,
        )
        if any(array.shape != expected for array in arrays):
            raise ValueError("all contact supervision arrays must have the same shape")
        for name, confidence in (
            ("truth", truth_confidence),
            ("clean inferred", inferred_clean_confidence),
            ("corrupted inferred", inferred_corrupted_confidence),
        ):
            if np.any((confidence < 0.0) | (confidence > 1.0)):
                raise ValueError(f"{name} contact confidence must be within [0,1]")
        plane = frozen_array(
            self.ground_plane,
            dtype=np.dtype(np.float32),
            name="contact ground plane",
        )
        if plane.shape != (4,):
            raise ValueError("contact ground plane must have shape [4]")
        if not self.truth_source.strip() or not self.inference_source.strip():
            raise ValueError("contact supervision sources must be nonempty")
        if not np.isfinite(self.ground_confidence) or not 0.0 <= self.ground_confidence <= 1.0:
            raise ValueError("ground_confidence must be finite and within [0,1]")
        object.__setattr__(self, "marker_names", marker_names)
        object.__setattr__(self, "truth_hard", truth_hard)
        object.__setattr__(self, "truth_confidence", truth_confidence)
        object.__setattr__(self, "inferred_clean_hard", inferred_clean_hard)
        object.__setattr__(self, "inferred_clean_confidence", inferred_clean_confidence)
        object.__setattr__(self, "inferred_corrupted_hard", inferred_corrupted_hard)
        object.__setattr__(self, "inferred_corrupted_confidence", inferred_corrupted_confidence)
        object.__setattr__(self, "ground_plane", plane)


@dataclass(frozen=True)
class CorruptionResult:
    """A corrupted clip plus dense training labels and measured preference evidence."""

    corrupted_motion: MotionClip
    clean_motion: MotionClip
    corruption_family: str
    corruption_mechanism: str
    intervention_mask: NDArray[np.bool_]
    symptom_mask: NDArray[np.bool_]
    responsibility_target: NDArray[np.float32]
    responsibility_confidence: NDArray[np.float32]
    measured_metrics_before: Mapping[str, float | None]
    measured_metrics_after: Mapping[str, float | None]
    measured_severity: float
    preference_confidence: float
    source_motion_id: str
    corruption_seed: int
    split_lineage_id: str | None
    requested_severity_parameter: float
    generation_parameters: Mapping[str, Any]
    postcondition: MeasuredPostcondition
    contact_supervision: ContactSupervision | None = None
    catalog_partition: str = "train"
    schema_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.clean_motion.num_frames != self.corrupted_motion.num_frames:
            raise ValueError("clean and corrupted clips must have the same frame count")
        if self.clean_motion.skeleton.content_hash != self.corrupted_motion.skeleton.content_hash:
            raise ValueError("clean and corrupted clips must use the same skeleton")
        if not self.corruption_family.strip() or not self.corruption_mechanism.strip():
            raise ValueError("corruption family and mechanism must be nonempty")
        if self.corruption_family not in (*CORRUPTION_DEFECT_NAMES, "composite"):
            raise ValueError(f"unknown corruption family {self.corruption_family!r}")
        if self.catalog_partition not in {"train", "heldout_eval"}:
            raise ValueError("catalog_partition must be 'train' or 'heldout_eval'")
        if self.source_motion_id != self.clean_motion.content_hash:
            raise ValueError("source_motion_id must identify the clean motion")
        if self.split_lineage_id is not None and not self.split_lineage_id.strip():
            raise ValueError("split_lineage_id must be nonempty when supplied")

        frames = self.corrupted_motion.num_frames
        joints = self.corrupted_motion.num_joints
        intervention = frozen_array(
            self.intervention_mask,
            dtype=np.dtype(np.bool_),
            name="intervention mask",
        )
        symptom = frozen_array(
            self.symptom_mask,
            dtype=np.dtype(np.bool_),
            name="symptom mask",
        )
        responsibility = frozen_array(
            self.responsibility_target,
            dtype=np.dtype(np.float32),
            name="responsibility target",
        )
        responsibility_confidence = frozen_array(
            self.responsibility_confidence,
            dtype=np.dtype(np.float32),
            name="responsibility confidence",
        )
        expected_intervention = (frames, joints, len(CORRUPTION_CHANNEL_NAMES))
        expected_semantic = (frames, len(CORRUPTION_PART_NAMES), len(CORRUPTION_DEFECT_NAMES))
        if intervention.shape != expected_intervention:
            raise ValueError(f"intervention_mask must have shape {expected_intervention}")
        for array_name, array_value in (
            ("symptom_mask", symptom),
            ("responsibility_target", responsibility),
            ("responsibility_confidence", responsibility_confidence),
        ):
            if array_value.shape != expected_semantic:
                raise ValueError(f"{array_name} must have shape {expected_semantic}")
        if np.any((responsibility < 0.0) | (responsibility > 1.0)):
            raise ValueError("responsibility_target must be within [0,1]")
        if np.any((responsibility_confidence < 0.0) | (responsibility_confidence > 1.0)):
            raise ValueError("responsibility_confidence must be within [0,1]")
        if np.any((responsibility > 0.0) & (responsibility_confidence == 0.0)):
            raise ValueError("nonzero responsibility targets require nonzero confidence")
        if (
            self.contact_supervision is not None
            and self.contact_supervision.truth_hard.shape[0] != frames
        ):
            raise ValueError("contact supervision frame count must match the motion")

        for scalar_name, scalar_value in (
            ("measured_severity", self.measured_severity),
            ("requested_severity_parameter", self.requested_severity_parameter),
        ):
            if not np.isfinite(scalar_value) or scalar_value < 0.0:
                raise ValueError(f"{scalar_name} must be finite and nonnegative")
        if (
            not np.isfinite(self.preference_confidence)
            or not 0.0 <= self.preference_confidence <= 1.0
        ):
            raise ValueError("preference_confidence must be finite and within [0,1]")
        if not np.isclose(
            self.measured_severity,
            max(0.0, self.postcondition.delta),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("measured_severity must equal the nonnegative measured metric delta")
        if self.postcondition.hard_negative and self.preference_confidence <= 0.0:
            raise ValueError("hard negatives require positive preference confidence")

        before = frozen_json_mapping(self.measured_metrics_before, name="metrics before")
        after = frozen_json_mapping(self.measured_metrics_after, name="metrics after")
        parameters = frozen_json_mapping(self.generation_parameters, name="generation parameters")
        schema_metadata = frozen_json_mapping(self.schema_metadata, name="schema metadata")
        object.__setattr__(self, "intervention_mask", intervention)
        object.__setattr__(self, "symptom_mask", symptom)
        object.__setattr__(self, "responsibility_target", responsibility)
        object.__setattr__(self, "responsibility_confidence", responsibility_confidence)
        object.__setattr__(self, "measured_metrics_before", before)
        object.__setattr__(self, "measured_metrics_after", after)
        object.__setattr__(self, "generation_parameters", parameters)
        object.__setattr__(self, "schema_metadata", schema_metadata)

    @property
    def corrupted(self) -> MotionClip:
        """Compatibility alias for the original vertical-slice API."""
        return self.corrupted_motion

    @property
    def repair_target(self) -> MotionClip:
        """Compatibility alias for the original vertical-slice API."""
        return self.clean_motion

    @property
    def defect_type(self) -> str:
        """Compatibility alias for ``corruption_family``."""
        return self.corruption_family

    @property
    def affected_frames(self) -> NDArray[np.bool_]:
        """Return frames containing an exact generator intervention."""
        result: NDArray[np.bool_] = np.asarray(
            np.any(self.intervention_mask, axis=(1, 2)),
            dtype=np.bool_,
        )
        result.setflags(write=False)
        return result

    @property
    def affected_joints(self) -> NDArray[np.bool_]:
        """Return the legacy dense joint intervention mask ``[T,J]``."""
        result: NDArray[np.bool_] = np.asarray(
            np.any(self.intervention_mask, axis=2),
            dtype=np.bool_,
        )
        result.setflags(write=False)
        return result

    @property
    def severity_parameter(self) -> float:
        """Compatibility alias for the requested, not measured, severity."""
        return self.requested_severity_parameter

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Compatibility alias for generator parameters."""
        return self.generation_parameters

    @property
    def seed(self) -> int:
        """Compatibility alias for ``corruption_seed``."""
        return self.corruption_seed
