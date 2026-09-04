"""Leakage-safe tensorization of persisted fixed-rig samples."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from motionlab.corruptions.base import CORRUPTION_DEFECT_NAMES
from motionlab.critic.model import FIRST_CRITIC_DEFECTS
from motionlab.dataset.features import TrainingSample, motion_feature_arrays
from motionlab.dataset.io import (
    NORMALIZED_FIELDS,
    NormalizationStatistics,
    load_normalization_statistics,
    load_training_sample,
)
from motionlab.motion.clip import MotionClip

SEVERITY_RANK = {
    "near_threshold": 0.2,
    "subtle": 0.4,
    "moderate": 0.6,
    "clear": 0.8,
    "severe": 1.0,
}

SUMMARY_METRIC_NAMES = (
    "foot_sliding_cm",
    "ground_penetration_cm",
    "floating_contact_cm",
    "loop_seam",
    "angular_smoothness_rad_s3_p99",
    "absolute_target_speed_error_mps",
    "average_speed_mps",
    "cadence_steps_per_minute",
    "stride_time_s",
    "stride_length_m",
    "step_width_m",
    "gait_asymmetry_percent",
)

RELEVANT_SUMMARY_METRIC = {
    "foot_slide": 0,
    "ground_penetration": 1,
    "floating_contact": 2,
    "loop_seam": 3,
    "joint_pop": 4,
    "joint_jitter": 4,
    "speed_inconsistency": 5,
}


def load_manifest(dataset_directory: Path) -> list[dict[str, Any]]:
    """Load the sorted dataset manifest."""
    path = Path(dataset_directory) / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} must be an object")
        records.append(value)
    return records


def _normalized_array(
    arrays: Mapping[str, Any],
    name: str,
    normalization: NormalizationStatistics,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    value = np.asarray(arrays[name], dtype=np.float32)
    if name not in NORMALIZED_FIELDS:
        return value
    return np.asarray(
        (value - normalization.means[name]) / normalization.scales[name],
        dtype=np.float32,
    )


def _encode_feature_arrays(
    arrays: Mapping[str, Any], normalization: NormalizationStatistics
) -> torch.Tensor:
    joint = np.concatenate(
        (
            np.asarray(arrays["local_rot6d"], dtype=np.float32),
            _normalized_array(arrays, "joint_position_rel", normalization),
            _normalized_array(arrays, "joint_velocity_rel", normalization),
            _normalized_array(arrays, "joint_angular_velocity", normalization),
        ),
        axis=-1,
    )
    frames = joint.shape[0]
    target_speed = _normalized_array(arrays, "target_speed", normalization)
    global_features = np.concatenate(
        (
            _normalized_array(arrays, "root_translation", normalization),
            _normalized_array(arrays, "root_velocity", normalization),
            _normalized_array(arrays, "root_angular_velocity", normalization),
            np.asarray(arrays["contact_soft"], dtype=np.float32),
            np.asarray(arrays["contact_valid"], dtype=np.float32),
            np.asarray(arrays["phase_sincos"], dtype=np.float32),
            np.asarray(arrays["phase_valid"], dtype=np.float32)[:, None],
            np.broadcast_to(target_speed, (frames, 1)),
        ),
        axis=-1,
    )
    encoded = np.concatenate((joint.reshape(frames, -1), global_features), axis=-1)
    return torch.from_numpy(np.ascontiguousarray(encoded, dtype=np.float32))


def encode_frame_features(
    sample: TrainingSample,
    normalization: NormalizationStatistics,
) -> torch.Tensor:
    """Flatten the audited fixed-joint features and append inference-safe globals."""
    return _encode_feature_arrays(sample.arrays, normalization)


def encode_motion_clip(clip: MotionClip, normalization: NormalizationStatistics) -> torch.Tensor:
    """Encode a candidate motion directly, without constructing targets or grading metrics."""
    return _encode_feature_arrays(motion_feature_arrays(clip), normalization)


def deterministic_summary_features(sample: TrainingSample) -> torch.Tensor:
    """Load exact deterministic summaries kept outside the learned model's inputs."""
    stored = sample.metadata.get("deterministic_baseline_metrics")
    if isinstance(stored, Mapping):
        stored_values = [stored.get(name) for name in SUMMARY_METRIC_NAMES]
        return torch.tensor(
            [0.0 if value is None else float(value) for value in stored_values],
            dtype=torch.float32,
        )

    # Backward-compatible proxies for older sample artifacts. New datasets always take the
    # exact-metric path above.
    arrays = sample.arrays
    joint_parts = np.asarray(arrays["joint_functional_part"], dtype=np.int64)
    leg_indices = np.flatnonzero((joint_parts == 5) | (joint_parts == 6))
    if leg_indices.size == 0:
        leg_indices = np.arange(np.asarray(arrays["joint_position_rel"]).shape[1])
    leg_speed = np.linalg.norm(
        np.asarray(arrays["joint_velocity_rel"], dtype=np.float64)[:, leg_indices], axis=-1
    )
    contact_weight = np.mean(np.asarray(arrays["contact_soft"], dtype=np.float64), axis=-1)
    weighted_leg_speed = np.max(leg_speed, axis=1) * contact_weight
    leg_height = np.asarray(arrays["joint_position_rel"], dtype=np.float64)[:, leg_indices, 1]
    root_velocity = np.asarray(arrays["root_velocity"], dtype=np.float64)
    root_speed = np.linalg.norm(root_velocity[:, (0, 2)], axis=-1)
    target_speed = float(np.asarray(arrays["target_speed"]).item())
    angular = np.asarray(arrays["joint_angular_velocity"], dtype=np.float64)
    angular_speed = np.linalg.norm(angular, axis=-1)
    angular_acceleration = np.diff(angular, axis=0)
    root_acceleration = np.diff(root_velocity, axis=0)
    local_rot = np.asarray(arrays["local_rot6d"], dtype=np.float64)
    endpoint_seam = np.linalg.norm(local_rot[-1] - local_rot[0]) + np.linalg.norm(
        root_velocity[-1] - root_velocity[0]
    )
    legacy_values = np.asarray(
        [
            np.quantile(weighted_leg_speed, 0.95),
            max(0.0, -float(np.min(leg_height))),
            max(0.0, float(np.quantile(np.min(leg_height, axis=1), 0.9))),
            endpoint_seam,
            float(np.max(np.linalg.norm(angular_acceleration, axis=-1))),
            float(np.quantile(angular_speed, 0.99)),
            abs(float(np.mean(root_speed)) - target_speed),
            float(np.quantile(np.linalg.norm(root_acceleration, axis=-1), 0.99)),
        ],
        dtype=np.float32,
    )
    values = np.zeros(len(SUMMARY_METRIC_NAMES), dtype=np.float32)
    values[:5] = legacy_values[:5]
    values[5] = legacy_values[6]
    values[6] = float(np.mean(root_speed))
    return torch.from_numpy(values)


def encode_targets(
    sample: TrainingSample,
    *,
    defect_names: Sequence[str] = FIRST_CRITIC_DEFECTS,
) -> dict[str, torch.Tensor]:
    """Select only the first-experiment defects and create unitless ordinal severity targets."""
    indices = [CORRUPTION_DEFECT_NAMES.index(name) for name in defect_names]
    frame = np.asarray(sample.arrays["defect_mask"], dtype=np.float32)[:, indices]
    part = np.asarray(sample.arrays["symptom_mask"], dtype=np.float32)[:, :, indices]
    clip = np.asarray(sample.arrays["defect_present"], dtype=np.float32)[indices]
    rank = float(SEVERITY_RANK.get(str(sample.metadata.get("severity_label")), 0.0))
    severity = clip * rank
    return {
        "frame_target": torch.from_numpy(np.ascontiguousarray(frame)),
        "part_target": torch.from_numpy(np.ascontiguousarray(part)),
        "clip_target": torch.from_numpy(np.ascontiguousarray(clip)),
        "severity_target": torch.from_numpy(np.ascontiguousarray(severity)),
    }


SplitRegime = Literal["train", "validation", "known_test", "heldout_mechanism", "all"]


def records_for_regime(
    records: Sequence[Mapping[str, Any]], regime: SplitRegime
) -> list[dict[str, Any]]:
    """Apply lineage-safe experiment splits without consulting sample arrays."""
    if regime == "all":
        return [dict(record) for record in records]
    if regime == "heldout_mechanism":
        return [
            dict(record)
            for record in records
            if record.get("data_split") == "heldout_corruptor"
            and record.get("source_split") == "test"
        ]
    expected_split = {
        "train": "train",
        "validation": "validation",
        "known_test": "test",
    }[regime]
    return [
        dict(record)
        for record in records
        if record.get("data_split") == expected_split
        and record.get("catalog_partition") in {"clean", "equivalent", "train", "sham"}
    ]


class FixedRigSampleDataset(Dataset[dict[str, Any]]):
    """In-memory metadata with optional decoded-sample caching."""

    def __init__(
        self,
        dataset_directory: Path,
        *,
        regime: SplitRegime,
        defect_names: Sequence[str] = FIRST_CRITIC_DEFECTS,
        cache_samples: bool = True,
    ) -> None:
        self.root = Path(dataset_directory)
        self.normalization = load_normalization_statistics(self.root)
        self.records = records_for_regime(load_manifest(self.root), regime)
        if not self.records:
            raise ValueError(f"dataset has no records for regime {regime!r}")
        self.defect_names = tuple(defect_names)
        self.cache_samples = cache_samples
        self._cache: dict[int, TrainingSample] = {}
        first = self._sample(0)
        self.num_joints = int(first.arrays["local_rot6d"].shape[1])
        self.sequence_frames = int(first.arrays["local_rot6d"].shape[0])

    def _sample(self, index: int) -> TrainingSample:
        if index not in self._cache:
            sample = load_training_sample(self.root / str(self.records[index]["path"]))
            if self.cache_samples:
                self._cache[index] = sample
            return sample
        return self._cache[index]

    def sample(self, index: int) -> TrainingSample:
        """Return the decoded persisted sample for audit and reconstruction tooling."""
        return self._sample(index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._sample(index)
        if sample.arrays["local_rot6d"].shape[1] != self.num_joints:
            raise ValueError("fixed-rig dataset contains multiple joint counts")
        result: dict[str, Any] = {
            "frame_features": encode_frame_features(sample, self.normalization),
            "summary_features": deterministic_summary_features(sample),
            **encode_targets(sample, defect_names=self.defect_names),
            "sample_id": sample.sample_id,
            "source_clip_id": str(sample.metadata.get("source_clip_id", "")),
            "source_split": str(sample.metadata.get("split", "")),
            "sample_role": str(sample.metadata.get("sample_role", "")),
            "corruption_family": str(sample.metadata.get("corruption_family") or "clean"),
            "corruption_mechanism": str(
                sample.metadata.get("catalog_mechanism_key")
                or sample.metadata.get("corruption_mechanism")
                or "clean"
            ),
            "severity_label": str(sample.metadata.get("severity_label") or "clean"),
            "ordinal_chain_id": str(sample.metadata.get("ordinal_chain_id") or ""),
            "ordinal_rank": int(sample.metadata.get("ordinal_rank") or 0),
            "measured_severity": float(sample.metadata.get("measured_severity") or 0.0),
            "sample_path": str(self.records[index]["path"]),
            "source_path": str(self.records[index].get("source_path") or ""),
        }
        return result
