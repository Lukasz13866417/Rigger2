"""Guarded, evaluation-only assessment for the external MotionCritic baseline."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from motionlab.perceptual.comparator_readiness import wilson_interval

MOTIONCRITIC_BASELINE_VERSION = "motionlab.motioncritic_baseline_assessment.v1"
_EXPECTED_JOINT_COUNT = 24
_EXPECTED_SEQUENCE_FRAMES = 60


def _canonical_preference(value: Any) -> str | None:
    return {
        "a_better": "a_better",
        "b_better": "b_better",
        "approximately_equal": "approximately_equal",
        "first_better": "a_better",
        "second_better": "b_better",
        "effectively_equal": "approximately_equal",
    }.get(str(value))


def _human_preference(row: dict[str, Any]) -> str | None:
    for key in ("human_preference", "pair_outcome_canonical"):
        if key in row:
            value = _canonical_preference(row.get(key))
            if value is not None:
                return value
    if (
        row.get("supervision_category") == "HUMAN_LABELED"
        or row.get("supervision_origin") == "human"
    ):
        return _canonical_preference(row.get("preference"))
    return None


def assess_motioncritic_compatibility(rig_contract: dict[str, Any]) -> dict[str, Any]:
    """Determine whether using the pretrained baseline would be representation-defensible."""
    joint_count = rig_contract.get("joint_count", rig_contract.get("num_joints"))
    skeleton_model = str(
        rig_contract.get("skeleton_model", rig_contract.get("skeleton_convention", "unknown"))
    )
    rotation_representation = str(
        rig_contract.get("rotation_representation", rig_contract.get("rotation_format", "unknown"))
    )
    root_representation = str(rig_contract.get("root_representation", "unknown"))
    source_frames = rig_contract.get("sequence_frames", rig_contract.get("num_frames"))
    blockers: list[str] = []
    if joint_count != _EXPECTED_JOINT_COUNT:
        blockers.append(f"joint_count_mismatch:expected_{_EXPECTED_JOINT_COUNT}_got_{joint_count}")
    if skeleton_model.lower() != "smpl":
        blockers.append("skeleton_is_not_verified_smpl")
    if rotation_representation.lower() not in {"local_axis_angle", "axis_angle_local"}:
        blockers.append("rotation_stream_is_not_verified_local_smpl_axis_angle")
    if root_representation.lower() not in {"xyz", "root_xyz"}:
        blockers.append("root_xyz_contract_missing")
    if not bool(rig_contract.get("verified_smpl_joint_mapping", False)):
        blockers.append("verified_smpl_joint_mapping_missing")
    if not bool(rig_contract.get("smpl_assets_available", False)):
        blockers.append("required_smpl_assets_unavailable")
    if source_frames != _EXPECTED_SEQUENCE_FRAMES and not bool(
        rig_contract.get("validated_resampling_to_60_frames", False)
    ):
        blockers.append("validated_60_frame_conversion_missing")
    status = "ready_for_evaluation_only" if not blockers else "deferred_incompatible_representation"
    return {
        "format_version": MOTIONCRITIC_BASELINE_VERSION,
        "status": status,
        "blockers": blockers,
        "current_contract": {
            "joint_count": joint_count,
            "skeleton_model": skeleton_model,
            "rotation_representation": rotation_representation,
            "root_representation": root_representation,
            "sequence_frames": source_frames,
        },
        "expected_contract": {
            "joint_count": _EXPECTED_JOINT_COUNT,
            "skeleton_model": "SMPL",
            "rotation_representation": "local_axis_angle",
            "root_representation": "XYZ",
            "sequence_frames": _EXPECTED_SEQUENCE_FRAMES,
            "tensor_shape": "[batch, 60, 25, 3]: 24 SMPL joints plus root XYZ",
        },
        "mapping_attempted": False,
        "checkpoint_loaded": False,
        "labels_adapted_or_used_for_fitting": False,
        "upstream_repository": "https://github.com/ou524u/MotionCritic",
        "decision": (
            "defer rather than report a representation-confounded baseline"
            if blockers
            else "permit pretrained inference and human-agreement evaluation only"
        ),
    }


def _agreement_metric(values: list[bool]) -> dict[str, Any]:
    successes = sum(values)
    return {
        "pair_count": len(values),
        "agreement_count": successes,
        "agreement": None if not values else successes / len(values),
        "agreement_95_percent_wilson_interval": wilson_interval(successes, len(values)),
    }


def motioncritic_pairwise_agreement(
    human_pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    """Compare frozen external predictions with direct human judgments, without adaptation."""
    if compatibility.get("status") != "ready_for_evaluation_only":
        return {
            "status": "not_run_incompatible_representation",
            "pairwise_agreement_with_human_judgments": None,
            "compatibility": compatibility,
            "labels_adapted_or_used_for_fitting": False,
            "checkpoint_or_model_updated": False,
        }
    prediction_by_id: dict[str, str] = {}
    for row in predictions:
        pair_id = row.get("pair_id", row.get("observation_id"))
        preference = _canonical_preference(row.get("prediction", row.get("predicted_preference")))
        if pair_id is None or preference is None:
            continue
        key = str(pair_id)
        if key in prediction_by_id:
            raise ValueError(f"duplicate MotionCritic prediction: {key}")
        prediction_by_id[key] = preference
    comparisons: list[dict[str, Any]] = []
    missing: list[str] = []
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row in human_pairs:
        human = _human_preference(row)
        if human is None:
            continue
        raw_pair_id = row.get("pair_id", row.get("observation_id"))
        if raw_pair_id is None:
            raise ValueError("direct human pair needs pair_id or observation_id")
        pair_id = str(raw_pair_id)
        predicted = prediction_by_id.get(pair_id)
        if predicted is None:
            missing.append(pair_id)
            continue
        family = str(row.get("perturbation_mechanism", row.get("family", "unknown")))
        agreed = predicted == human
        by_family[family].append(agreed)
        comparisons.append(
            {
                "pair_id": pair_id,
                "family": family,
                "direct_human_preference": human,
                "prediction": predicted,
                "agreement": agreed,
            }
        )
    values = [bool(row["agreement"]) for row in comparisons]
    return {
        "status": "complete" if comparisons and not missing else "incomplete_human_pair_coverage",
        "pairwise_agreement_with_human_judgments": _agreement_metric(values),
        "by_perturbation_family": {
            family: _agreement_metric(family_values)
            for family, family_values in sorted(by_family.items())
        },
        "direct_human_pair_count": sum(_human_preference(row) is not None for row in human_pairs),
        "evaluated_pair_count": len(comparisons),
        "missing_prediction_pair_ids": sorted(missing),
        "comparisons": comparisons,
        "compatibility": compatibility,
        "labels_adapted_or_used_for_fitting": False,
        "checkpoint_or_model_updated": False,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    value: Any
    if path.suffix.lower() == ".jsonl":
        value = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("pairs", value.get("predictions"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON/JSONL rows: {path}")
    return value


def prepare_motioncritic_baseline_report(
    rig_contract_path: Path,
    output_path: Path,
    *,
    human_pairs_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    """Write compatibility now and agreement later, if defensible predictions exist."""
    input_paths = {
        Path(value).resolve()
        for value in (rig_contract_path, human_pairs_path, predictions_path)
        if value is not None
    }
    output = Path(output_path)
    if output.resolve() in input_paths:
        raise ValueError("MotionCritic report must not overwrite an input artifact")
    rig_contract = json.loads(Path(rig_contract_path).read_text(encoding="utf-8"))
    if not isinstance(rig_contract, dict):
        raise ValueError("rig contract must be a JSON object")
    compatibility = assess_motioncritic_compatibility(rig_contract)
    if human_pairs_path is None and predictions_path is None:
        report: dict[str, Any] = compatibility
    elif human_pairs_path is None or predictions_path is None:
        raise ValueError("human pairs and predictions must be supplied together")
    else:
        report = motioncritic_pairwise_agreement(
            _read_rows(Path(human_pairs_path)),
            _read_rows(Path(predictions_path)),
            compatibility=compatibility,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
