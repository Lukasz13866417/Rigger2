"""Counterfactual forensic audit for every sham control."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.model import FIRST_CRITIC_DEFECTS
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.quaternion import matrix_to_quaternion, quaternion_geodesic_distance
from motionlab.math.rotation6d import rotation_6d_to_matrix
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.motion.clip import MotionClip
from motionlab.visualization.skeleton_plot import save_motion_comparison_strip

SHAM_AUDIT_VERSION = "motionlab.sham_forensic_audit.v1"

_FAMILY_METRIC = {
    "foot_slide": "maximum_stance_slip_cm",
    "ground_penetration": "maximum_ground_penetration_cm",
    "floating_contact": "maximum_floating_contact_cm",
    "loop_seam": "loop_seam",
    "joint_pop": "angular_smoothness_rad_s3_p99",
    "joint_jitter": "angular_smoothness_rad_s3_p99",
    "speed_inconsistency": "absolute_target_speed_error_mps",
}

# These are audit materiality thresholds, not production validity limits.  They only decide
# whether a positive prediction has deterministic evidence for some other known artifact.
_MATERIAL_DELTA = {
    "maximum_stance_slip_cm": 0.2,
    "maximum_ground_penetration_cm": 0.2,
    "maximum_floating_contact_cm": 0.2,
    "loop_seam": 0.02,
    "angular_smoothness_rad_s3_p99": 100.0,
    "absolute_target_speed_error_mps": 0.02,
    "maximum_joint_limit_violation_deg": 0.2,
}


def motions_from_sample(
    dataset: FixedRigSampleDataset, index: int
) -> tuple[MotionClip, MotionClip]:
    """Reconstruct clean and encoded candidate clips from one persisted sample."""
    sample = dataset.sample(index)
    source_path = Path(str(dataset.records[index]["source_path"]))
    source = load_motion_npz(source_path)
    clean_root = np.asarray(sample.arrays["clean_root_translation"], dtype=np.float32)
    clean = MotionClip(
        skeleton=source.skeleton,
        local_quat_wxyz=sample.arrays["clean_local_quat"],
        root_translation_m=clean_root,
        fps=source.fps,
        metadata={**dict(source.metadata), **dict(sample.metadata), "audit_role": "clean"},
    )
    root = np.asarray(sample.arrays["root_translation"], dtype=np.float32).copy()
    # The inference feature intentionally removes absolute horizontal origin.  Re-aligning it to
    # the clean first frame restores a comparable world trajectory without using a learned input.
    root[:, (0, 2)] += clean_root[0, (0, 2)]
    candidate = MotionClip(
        skeleton=source.skeleton,
        local_quat_wxyz=matrix_to_quaternion(
            rotation_6d_to_matrix(sample.arrays["local_rot6d"])
        ).astype(np.float32),
        root_translation_m=root,
        fps=source.fps,
        metadata={**dict(source.metadata), **dict(sample.metadata), "audit_role": "sham"},
    )
    return clean, candidate


def motion_deterministic_metrics(clip: MotionClip) -> dict[str, float | None]:
    """Return the compact deterministic evidence used in critic and optimizer audits."""
    target_raw = clip.metadata.get(
        "target_speed",
        clip.metadata.get("target_speed_mps", clip.metadata.get("expected_speed_mps")),
    )
    target_speed = None if target_raw is None else float(target_raw)
    report = grade_motion_deterministic(clip, target_speed_mps=target_speed)
    return {
        **report.measured,
        "angular_smoothness_rad_s3_p99": report.metrics["angular_smoothness"].clip_value,
        "absolute_target_speed_error_mps": report.metrics["root_speed"].metadata.get(
            "absolute_target_error_mps"
        ),
    }


def motion_distances(clean: MotionClip, candidate: MotionClip) -> dict[str, float]:
    """Measure root, local-rotation, and FK joint distance between matching clips."""
    root_difference = candidate.root_translation_m - clean.root_translation_m
    rotation = quaternion_geodesic_distance(clean.local_quat_wxyz, candidate.local_quat_wxyz)
    clean_position, _ = forward_kinematics_numpy(
        clean.skeleton, clean.local_quat_wxyz, clean.root_translation_m
    )
    candidate_position, _ = forward_kinematics_numpy(
        candidate.skeleton,
        candidate.local_quat_wxyz,
        candidate.root_translation_m,
    )
    return {
        "root_translation_rms_m": float(
            np.sqrt(np.mean(np.sum(root_difference * root_difference, axis=-1)))
        ),
        "local_rotation_geodesic_rms_deg": float(np.rad2deg(np.sqrt(np.mean(rotation * rotation)))),
        "joint_world_position_rms_m": float(
            np.sqrt(np.mean(np.sum((candidate_position - clean_position) ** 2, axis=-1)))
        ),
    }


def _metric_deltas(
    before: Mapping[str, float | None], after: Mapping[str, float | None]
) -> dict[str, float | None]:
    return {
        name: None
        if before.get(name) is None or after.get(name) is None
        else float(after[name]) - float(before[name])  # type: ignore[arg-type]
        for name in sorted(set(before) | set(after))
    }


def audit_sham_controls(
    dataset: FixedRigSampleDataset,
    prediction: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Audit every persisted sham, including non-test splits and non-target head responses."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, metadata in enumerate(prediction["metadata"]):
        if metadata["sample_role"] != "sham_control":
            continue
        family = str(metadata["corruption_family"])
        clean, sham = motions_from_sample(dataset, index)
        stem = str(metadata["sample_id"]).removeprefix("sample:sha256:")
        sample_directory = output / stem
        save_motion_npz(sample_directory / "clean.npz", clean)
        save_motion_npz(sample_directory / "sham.npz", sham)
        save_motion_comparison_strip(clean, sham, sample_directory / "comparison.png")
        before = motion_deterministic_metrics(clean)
        after = motion_deterministic_metrics(sham)
        deltas = _metric_deltas(before, after)
        target_metric = _FAMILY_METRIC[family]
        target_delta = deltas.get(target_metric)
        material_other = [
            name
            for name, threshold in _MATERIAL_DELTA.items()
            if name != target_metric
            and deltas.get(name) is not None
            and float(deltas[name]) > threshold  # type: ignore[arg-type]
        ]
        probabilities = {
            name: float(prediction["clip_probability"][index, defect_index])
            for defect_index, name in enumerate(FIRST_CRITIC_DEFECTS)
        }
        severities = {
            name: float(prediction["clip_severity"][index, defect_index])
            for defect_index, name in enumerate(FIRST_CRITIC_DEFECTS)
        }
        ranks = {
            name: float(prediction["family_ranking_score"][index, defect_index])
            for defect_index, name in enumerate(FIRST_CRITIC_DEFECTS)
        }
        target_positive = probabilities[family] >= 0.5
        classification = "no_target_false_positive"
        if target_positive and material_other:
            classification = "legitimate_other_artifact_present"
        elif target_positive:
            classification = "target_head_shortcut"
        records.append(
            {
                "sample_id": metadata["sample_id"],
                "source_clip_id": metadata["source_clip_id"],
                "source_split": metadata["source_split"],
                "target_family": family,
                "declared_target_metric": dataset.sample(index).metadata.get(
                    "postcondition_metric"
                ),
                "audited_target_metric": target_metric,
                "target_probability": probabilities[family],
                "target_severity": severities[family],
                "target_family_rank": ranks[family],
                "all_probabilities": probabilities,
                "all_severities": severities,
                "all_family_ranks": ranks,
                "deterministic_before": before,
                "deterministic_after": after,
                "deterministic_delta": deltas,
                "target_metric_delta": target_delta,
                "material_other_artifacts": material_other,
                "distance": motion_distances(clean, sham),
                "classification": classification,
                "artifacts": {
                    "clean": str(sample_directory / "clean.npz"),
                    "sham": str(sample_directory / "sham.npz"),
                    "comparison": str(sample_directory / "comparison.png"),
                },
            }
        )
    target_false_positives = [
        record for record in records if float(record["target_probability"]) >= 0.5
    ]
    any_false_positives = [
        record
        for record in records
        if max(float(value) for value in record["all_probabilities"].values()) >= 0.5
    ]
    report = {
        "format_version": SHAM_AUDIT_VERSION,
        "sample_count": len(records),
        "target_head_false_positive_count": len(target_false_positives),
        "target_head_false_positive_rate": (
            None if not records else len(target_false_positives) / len(records)
        ),
        "any_head_false_positive_count": len(any_false_positives),
        "any_head_false_positive_rate": (
            None if not records else len(any_false_positives) / len(records)
        ),
        "classification_counts": {
            label: sum(record["classification"] == label for record in records)
            for label in (
                "no_target_false_positive",
                "target_head_shortcut",
                "legitimate_other_artifact_present",
            )
        },
        "materiality_thresholds": _MATERIAL_DELTA,
        "records": records,
    }
    (output / "sham_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
