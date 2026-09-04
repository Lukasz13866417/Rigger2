"""Evaluation-only readiness checks for perceptual comparator ablations.

The functions in this module consume precomputed predictions.  They never instantiate, fit, or
update a critic, so the complete evaluation contract can be exercised with synthetic fixtures
while human labeling is still in progress.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from motionlab.dataset.io import file_sha256
from motionlab.io.npz import load_motion_npz
from motionlab.perceptual.dataset import (
    load_perceptual_pairs,
    objective_equivalence_control,
)
from motionlab.perceptual.observation_auth import (
    resolve_frozen_evidence_paths,
    validate_current_protocol_observations,
)
from motionlab.perceptual.subjective import (
    FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION,
    SUBJECTIVE_PROTOCOL_VERSION,
)

COMPARATOR_READINESS_VERSION = "motionlab.comparator_readiness.v1"
AUTHENTICATED_COMPARATOR_PAIRS_VERSION = "motionlab.authenticated_comparator_pairs.v1"
COMPARATOR_PREDICTION_ADAPTER_VERSION = "motionlab.comparator_prediction_adapter.v1"
POST_LABEL_COMPARATOR_EVALUATION_VERSION = "motionlab.post_label_comparator_evaluation.v1"
COMPARATOR_VARIANTS = (
    "absolute_q",
    "relative_comparator_frozen",
    "relative_comparator_finetuned",
)
PREFERENCE_CLASSES = ("a_better", "b_better", "approximately_equal")
EvidenceOrigin = Literal["synthetic_fixture", "human_pilot"]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """Compute a two-sided Wilson score interval for one binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _canonical_preference(value: Any) -> str | None:
    mapping = {
        "a_better": "a_better",
        "b_better": "b_better",
        "approximately_equal": "approximately_equal",
        "first_better": "a_better",
        "second_better": "b_better",
        "effectively_equal": "approximately_equal",
    }
    return mapping.get(str(value))


def _pair_id(row: dict[str, Any]) -> str:
    value = row.get("pair_id", row.get("observation_id"))
    if value is None:
        raise ValueError("comparator evaluation rows require pair_id or observation_id")
    return str(value)


def _direct_human_preference(row: dict[str, Any]) -> str | None:
    for key in ("human_preference", "pair_outcome_canonical"):
        if key in row:
            value = _canonical_preference(row.get(key))
            if value is not None:
                return value
    if row.get("format_version") == AUTHENTICATED_COMPARATOR_PAIRS_VERSION:
        return None
    if (
        row.get("supervision_category") == "HUMAN_LABELED"
        or row.get("supervision_origin") == "human"
    ):
        return _canonical_preference(row.get("preference"))
    return None


def _reference_preference(row: dict[str, Any], *, allow_synthetic_fallback: bool) -> str | None:
    direct_human = _direct_human_preference(row)
    if direct_human is not None or not allow_synthetic_fallback:
        return direct_human
    return _canonical_preference(row.get("preference"))


def _declares_human_evidence(row: dict[str, Any]) -> bool:
    authentication = row.get("evidence_authentication")
    return (
        row.get("format_version") == AUTHENTICATED_COMPARATOR_PAIRS_VERSION
        and row.get("supervision_category") == "HUMAN_LABELED"
        and row.get("supervision_origin") == "human"
        and isinstance(authentication, dict)
        and authentication.get("current_protocol_validated") is True
        and isinstance(authentication.get("freeze_id"), str)
        and bool(authentication["freeze_id"])
    )


def _validate_human_evidence(pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        raise ValueError("human_pilot evidence requires at least one direct human pair")
    undeclared = [_pair_id(row) for row in pairs if not _declares_human_evidence(row)]
    if undeclared:
        raise ValueError(
            "human_pilot evidence contains rows without direct-human provenance: "
            f"{sorted(undeclared)}"
        )
    synthetic_fallbacks = [
        _pair_id(row)
        for row in pairs
        if _direct_human_preference(row) is None
        and _canonical_preference(row.get("preference")) is not None
    ]
    if synthetic_fallbacks:
        raise ValueError(
            "human_pilot evidence must not fall back to synthetic preferences: "
            f"{sorted(synthetic_fallbacks)}"
        )
    unscorable = [_pair_id(row) for row in pairs if _direct_human_preference(row) is None]
    if unscorable:
        raise ValueError(
            f"human_pilot evidence contains unscorable direct-human targets: {sorted(unscorable)}"
        )
    missing_model_lineage = [
        _pair_id(row)
        for row in pairs
        if not isinstance(row.get("model_pair_id"), str) or not row["model_pair_id"]
    ]
    if missing_model_lineage:
        raise ValueError(
            f"human_pilot evidence lacks model-pair lineage: {sorted(missing_model_lineage)}"
        )
    missing_equivalence_declaration = [
        _pair_id(row) for row in pairs if "objective_equivalence_control" not in row
    ]
    if missing_equivalence_declaration:
        raise ValueError(
            "human_pilot evidence lacks an objective-equivalence declaration: "
            f"{sorted(missing_equivalence_declaration)}"
        )


def _explicit_equivalent_control(row: dict[str, Any]) -> bool:
    value = row.get("objective_equivalence_control", False)
    if not isinstance(value, bool):
        raise ValueError("objective_equivalence_control must be boolean when supplied")
    return value


def _subset(row: dict[str, Any]) -> str:
    explicit = row.get("subset")
    if explicit is not None:
        return str(explicit)
    if bool(row.get("adversarial_origin", row.get("is_adversarial", False))):
        return "adversarial"
    source_split = str(row.get("source_split", "other"))
    mechanism_partition = str(row.get("mechanism_partition", "other"))
    if source_split == "test" and mechanism_partition == "heldout":
        return "heldout_mechanism"
    if source_split == "test" and mechanism_partition == "train":
        return "heldout_source"
    return source_split if source_split in {"train", "validation"} else "other"


def _prediction_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows = value.get("predictions", value.get("pair_predictions"))
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _variant_predictions(value: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = value.get("variants", value)
    if not isinstance(raw, dict):
        raise ValueError("comparator predictions must contain a variants object")
    unknown = sorted(set(str(key) for key in raw) - set(COMPARATOR_VARIANTS))
    if unknown:
        raise ValueError(f"unknown comparator prediction variants: {unknown}")
    return {variant: _prediction_rows(raw.get(variant)) for variant in COMPARATOR_VARIANTS}


def _probabilities(row: dict[str, Any], key: str) -> dict[str, float] | None:
    raw = row.get(key)
    if not isinstance(raw, dict):
        return None
    result: dict[str, float] = {}
    for label in PREFERENCE_CLASSES:
        value = raw.get(label)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            return None
        result[label] = numeric
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        return None
    return result


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row["correct"] is not None]
    correct = sum(bool(row["correct"]) for row in labeled)
    return {
        "pair_count": len(labeled),
        "correct_count": correct,
        "accuracy": None if not labeled else correct / len(labeled),
        "accuracy_95_percent_wilson_interval": wilson_interval(correct, len(labeled)),
    }


def _evaluate_variant(
    pairs: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    swap_tolerance: float,
    allow_synthetic_reference: bool,
) -> dict[str, Any]:
    predictions: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in prediction_rows:
        pair_id = _pair_id(row)
        if pair_id in predictions:
            duplicate_ids.add(pair_id)
        predictions[pair_id] = row
    if duplicate_ids:
        raise ValueError(f"duplicate comparator predictions: {sorted(duplicate_ids)}")
    expected_pair_ids = {_pair_id(pair) for pair in pairs}
    unexpected = sorted(set(predictions) - expected_pair_ids)
    evaluated: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    swap_errors: list[float] = []
    for pair in pairs:
        pair_id = _pair_id(pair)
        prediction_row = predictions.get(pair_id)
        if prediction_row is None:
            missing.append(pair_id)
            continue
        forward = _probabilities(prediction_row, "probabilities")
        swapped = _probabilities(prediction_row, "swapped_probabilities")
        if forward is None or swapped is None:
            invalid.append(pair_id)
            continue
        prediction = max(
            PREFERENCE_CLASSES, key=lambda label: (forward[label], -PREFERENCE_CLASSES.index(label))
        )
        reference = _reference_preference(
            pair,
            allow_synthetic_fallback=allow_synthetic_reference,
        )
        direct_human = _direct_human_preference(pair)
        swap_error = max(
            abs(forward["a_better"] - swapped["b_better"]),
            abs(forward["b_better"] - swapped["a_better"]),
            abs(forward["approximately_equal"] - swapped["approximately_equal"]),
        )
        swap_errors.append(swap_error)
        evaluated.append(
            {
                "pair_id": pair_id,
                "subset": _subset(pair),
                "family": str(pair.get("perturbation_mechanism", pair.get("family", "unknown"))),
                "reference_preference": reference,
                "direct_human_preference": direct_human,
                "prediction": prediction,
                "correct": None if reference is None else prediction == reference,
                "direct_human_correct": (
                    None if direct_human is None else prediction == direct_human
                ),
                "equivalent_control": _explicit_equivalent_control(pair),
                "adversarial": bool(
                    pair.get("adversarial_origin", pair.get("is_adversarial", False))
                ),
                "swap_consistency_error": swap_error,
            }
        )
    by_family = {
        family: _metric([row for row in evaluated if row["family"] == family])
        for family in sorted({str(row["family"]) for row in evaluated})
    }
    direct_human_rows = [
        {**row, "correct": row["direct_human_correct"]}
        for row in evaluated
        if row["direct_human_preference"] is not None
    ]
    equivalent = [row for row in evaluated if row["equivalent_control"]]
    equivalent_false = sum(row["prediction"] != "approximately_equal" for row in equivalent)
    exact_swaps = sum(error == 0.0 for error in swap_errors)
    within_tolerance = sum(error <= swap_tolerance for error in swap_errors)
    return {
        "status": "ready"
        if not missing and not invalid and not unexpected and len(evaluated) == len(pairs)
        else "incomplete",
        "input_pair_count": len(pairs),
        "prediction_count": len(prediction_rows),
        "evaluated_pair_count": len(evaluated),
        "missing_pair_ids": sorted(missing),
        "unexpected_prediction_pair_ids": unexpected,
        "invalid_probability_pair_ids": sorted(invalid),
        "overall": _metric(evaluated),
        "heldout_source": _metric([row for row in evaluated if row["subset"] == "heldout_source"]),
        "heldout_mechanism": _metric(
            [row for row in evaluated if row["subset"] == "heldout_mechanism"]
        ),
        "direct_human_only": _metric(direct_human_rows),
        "equivalent_controls": {
            "pair_count": len(equivalent),
            "false_preference_count": equivalent_false,
            "false_preference_rate": (
                None if not equivalent else equivalent_false / len(equivalent)
            ),
            "false_preference_rate_95_percent_wilson_interval": wilson_interval(
                equivalent_false, len(equivalent)
            ),
        },
        "adversarial": _metric([row for row in evaluated if row["adversarial"]]),
        "by_perturbation_family": by_family,
        "swap_consistency": {
            "pair_count": len(swap_errors),
            "exact_zero_error_count": exact_swaps,
            "exact_zero_error_rate": (None if not swap_errors else exact_swaps / len(swap_errors)),
            "within_tolerance_count": within_tolerance,
            "within_tolerance_rate": (
                None if not swap_errors else within_tolerance / len(swap_errors)
            ),
            "tolerance": swap_tolerance,
            "maximum_probability_error": max(swap_errors, default=None),
            "mean_probability_error": (
                None if not swap_errors else sum(swap_errors) / len(swap_errors)
            ),
        },
        "pair_scores": evaluated,
    }


def build_comparator_readiness_report(
    pairs: list[dict[str, Any]],
    variant_predictions: dict[str, Any],
    *,
    evidence_origin: EvidenceOrigin,
    swap_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Evaluate all planned variants from precomputed outputs without fitting any model."""
    if evidence_origin not in {"synthetic_fixture", "human_pilot"}:
        raise ValueError("evidence_origin must be synthetic_fixture or human_pilot")
    if swap_tolerance < 0.0 or not math.isfinite(swap_tolerance):
        raise ValueError("swap_tolerance must be finite and nonnegative")
    pair_ids = [_pair_id(row) for row in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("comparator evaluation pair IDs must be unique")
    if evidence_origin == "human_pilot":
        _validate_human_evidence(pairs)
    for row in pairs:
        _explicit_equivalent_control(row)
    predictions = _variant_predictions(variant_predictions)
    variants = {
        name: _evaluate_variant(
            pairs,
            predictions[name],
            swap_tolerance=swap_tolerance,
            allow_synthetic_reference=evidence_origin == "synthetic_fixture",
        )
        for name in COMPARATOR_VARIANTS
    }
    input_coverage_counts = {
        "heldout_source": sum(_subset(row) == "heldout_source" for row in pairs),
        "heldout_mechanism": sum(_subset(row) == "heldout_mechanism" for row in pairs),
        "direct_human_only": sum(_direct_human_preference(row) is not None for row in pairs),
        "equivalent_controls": sum(_explicit_equivalent_control(row) for row in pairs),
        "adversarial_pairs": sum(
            bool(row.get("adversarial_origin", row.get("is_adversarial", False))) for row in pairs
        ),
    }
    evaluation_contract = {name: count > 0 for name, count in input_coverage_counts.items()}
    evaluation_contract["exact_and_tolerance_swap_consistency"] = bool(pairs) and all(
        row["swap_consistency"]["pair_count"] > 0 for row in variants.values()
    )
    evaluation_contract["wilson_confidence_intervals"] = bool(pairs) and all(
        row["overall"]["accuracy_95_percent_wilson_interval"] is not None
        for row in variants.values()
    )
    predictions_aligned = all(row["status"] == "ready" for row in variants.values())
    ready = predictions_aligned and all(evaluation_contract.values())
    return {
        "format_version": COMPARATOR_READINESS_VERSION,
        "status": "evaluation_machinery_ready" if ready else "evaluation_inputs_incomplete",
        "evidence_origin": evidence_origin,
        "synthetic_fixture_results_are_not_substantive_perceptual_findings": (
            evidence_origin == "synthetic_fixture"
        ),
        "training_invoked": False,
        "critic_parameters_updated": False,
        "variants": variants,
        "required_variant_order": list(COMPARATOR_VARIANTS),
        "supported_capabilities": {
            "heldout_source": True,
            "heldout_mechanism": True,
            "direct_human_only": True,
            "equivalent_controls": True,
            "adversarial_pairs": True,
            "exact_and_tolerance_swap_consistency": True,
            "wilson_confidence_intervals": True,
        },
        "input_coverage_counts": input_coverage_counts,
        "evaluation_contract": evaluation_contract,
        "prediction_alignment_complete": predictions_aligned,
        "human_labels_are_evaluation_targets_not_adaptation_inputs": True,
    }


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_comparator_readiness_report(
    pairs_path: Path,
    predictions_path: Path,
    output_path: Path,
    *,
    evidence_origin: EvidenceOrigin,
    swap_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """File wrapper for the evaluation-only comparator readiness report."""
    inputs = {Path(pairs_path).resolve(), Path(predictions_path).resolve()}
    output = Path(output_path)
    if output.resolve() in inputs:
        raise ValueError("comparator readiness output must not overwrite an input artifact")
    pairs = _load_json_or_jsonl(Path(pairs_path))
    predictions = _load_json_or_jsonl(Path(predictions_path))
    if isinstance(pairs, dict):
        pairs = pairs.get("pairs")
    if not isinstance(pairs, list) or not all(isinstance(row, dict) for row in pairs):
        raise ValueError("pairs must be a JSON/JSONL list or a JSON object containing pairs")
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a JSON object keyed by comparator variant")
    report = build_comparator_readiness_report(
        pairs,
        predictions,
        evidence_origin=evidence_origin,
        swap_tolerance=swap_tolerance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _read_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = _load_json_or_jsonl(Path(path))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must contain JSON objects")
    return rows


def _flip_preference(value: str) -> str:
    return {
        "a_better": "b_better",
        "b_better": "a_better",
        "approximately_equal": "approximately_equal",
    }[value]


def _observation_preference(value: Any) -> str | None:
    return {
        "first_better": "a_better",
        "second_better": "b_better",
        "effectively_equal": "approximately_equal",
        "not_sure": None,
    }.get(str(value))


def _pair_motion_hashes(
    pair_dataset_directory: Path,
    record: dict[str, Any],
) -> tuple[str, str]:
    root = Path(pair_dataset_directory)
    return (
        load_motion_npz(root / str(record["motion_a"])).content_hash,
        load_motion_npz(root / str(record["motion_b"])).content_hash,
    )


def _completed_pilot_contract(manifest: dict[str, Any], observations: list[dict[str, Any]]) -> bool:
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("pilot manifest has no completion contract")
    expected_base_ids: set[str] = set()
    for session in sessions:
        trials = session.get("trials") if isinstance(session, dict) else None
        if not isinstance(trials, list):
            raise ValueError("pilot manifest has a malformed completion contract")
        expected_base_ids.update(
            str(trial["trial_id"])
            for trial in trials
            if isinstance(trial, dict) and trial.get("trial_id") is not None
        )
    policy = manifest.get("adaptive_policy")
    expected_adaptive = (
        int(policy.get("maximum_followups_total", 0))
        if isinstance(policy, dict) and policy.get("enabled") is True
        else 0
    )
    adaptive_pool = manifest.get("adaptive_comparison_pool", [])
    if not isinstance(adaptive_pool, list):
        raise ValueError("pilot manifest has a malformed adaptive completion contract")
    adaptive_ids = {
        str(trial["trial_id"])
        for trial in adaptive_pool
        if isinstance(trial, dict) and trial.get("trial_id") is not None
    }
    by_rater: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        by_rater.setdefault(str(row.get("rater_id")), []).append(row)
    for rows in by_rater.values():
        completed_base = {
            str(row["trial_id"])
            for row in rows
            if row.get("trial_id") is not None and str(row["trial_id"]) in expected_base_ids
        }
        completed_adaptive = {
            str(row["trial_id"])
            for row in rows
            if row.get("trial_id") is not None and str(row["trial_id"]) in adaptive_ids
        }
        if completed_base == expected_base_ids and len(completed_adaptive) >= expected_adaptive:
            return True
    return False


def build_authenticated_comparator_pair_export(
    pair_dataset_directory: Path,
    pilot_manifest_path: Path,
    observations_path: Path,
    protocol_freeze_path: Path,
    pilot_report_path: Path,
    *,
    served_playlist_path: Path | None = None,
) -> dict[str, Any]:
    """Export scorable direct judgments only after re-authenticating the frozen pilot evidence."""
    pilot_path = Path(pilot_manifest_path).resolve()
    observations_path = Path(observations_path).resolve()
    freeze_path = Path(protocol_freeze_path).resolve()
    manifest = _read_json_object(pilot_path, label="pilot manifest")
    if manifest.get("protocol_version") not in {
        SUBJECTIVE_PROTOCOL_VERSION,
        FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION,
    }:
        raise ValueError("comparator export requires the frozen current subjective protocol")

    served_path, protocol_validation = resolve_frozen_evidence_paths(
        pilot_path,
        observations_path,
        freeze_path,
        served_playlist_path=served_playlist_path,
    )
    observations = _read_jsonl_objects(observations_path, label="observation log")
    served = _read_jsonl_objects(served_path, label="served-playlist log")
    validate_current_protocol_observations(observations, manifest, served)
    if not _completed_pilot_contract(manifest, observations):
        raise ValueError("comparator export requires completion of the frozen pilot contract")

    pilot_report_path = Path(pilot_report_path).resolve()
    try:
        pilot_report_path.relative_to(pilot_path.parent)
    except ValueError:
        pass
    else:
        raise ValueError("subjective analysis report must remain outside the active pilot")
    pilot_report = _read_json_object(pilot_report_path, label="subjective pilot report")
    completion = pilot_report.get("pilot_completion")
    report_validation = pilot_report.get("observation_validation")
    if (
        pilot_report.get("format_version") != "motionlab.subjective_analysis.v3"
        or pilot_report.get("status") != "pilot_complete_ready_for_inspection"
        or not isinstance(completion, dict)
        or completion.get("complete") is not True
    ):
        raise ValueError("comparator export requires a completed subjective pilot report")
    if (
        not isinstance(report_validation, dict)
        or report_validation.get("protocol_freeze_validated") is not True
        or report_validation.get("served_trial_linkage_validated") is not True
        or report_validation.get("freeze_id") != protocol_validation.get("freeze_id")
        or report_validation.get("validated_observation_count") != len(observations)
    ):
        raise ValueError("subjective pilot report is not bound to the authenticated evidence")
    if pilot_report.get("critic_retraining_started") is not False:
        raise ValueError("subjective pilot report does not preserve the no-retraining milestone")

    stimulus_meta = {
        str(row["stimulus_id"]): row
        for row in manifest.get("stimuli", [])
        if isinstance(row, dict) and row.get("stimulus_id") is not None
    }
    pair_records = load_perceptual_pairs(Path(pair_dataset_directory))
    records_by_id = {str(row["pair_id"]): row for row in pair_records}
    motion_hashes: dict[str, tuple[str, str]] = {}
    exported: list[dict[str, Any]] = []
    excluded_not_sure: list[str] = []
    excluded_non_pair = 0
    for observation in observations:
        if observation.get("task_type") != "pair_comparison":
            excluded_non_pair += 1
            continue
        observation_id = str(observation["observation_id"])
        human_preference = _observation_preference(observation.get("pair_outcome_canonical"))
        if human_preference is None:
            excluded_not_sure.append(observation_id)
            continue
        model_pair_id = observation.get("pair_id")
        if not isinstance(model_pair_id, str) or model_pair_id not in records_by_id:
            raise ValueError(
                f"direct human observation {observation_id} has no known model-pair lineage"
            )
        record = records_by_id[model_pair_id]
        stimulus_ids = observation.get("stimulus_ids")
        if not isinstance(stimulus_ids, list) or len(stimulus_ids) != 2:
            raise ValueError(f"direct human observation {observation_id} is not a pair")
        stimulus_hashes = tuple(
            str(stimulus_meta[str(stimulus_id)]["motion_content_hash"])
            for stimulus_id in stimulus_ids
        )
        expected_hashes = motion_hashes.setdefault(
            model_pair_id,
            _pair_motion_hashes(Path(pair_dataset_directory), record),
        )
        if stimulus_hashes == expected_hashes:
            model_orientation = "observation_order_matches_model_a_b"
        elif stimulus_hashes == tuple(reversed(expected_hashes)):
            model_orientation = "observation_order_matches_model_b_a"
            human_preference = _flip_preference(human_preference)
        else:
            raise ValueError(
                f"direct human observation {observation_id} motions do not match its model pair"
            )
        is_equivalent, equivalence_provenance = objective_equivalence_control(record)
        exported.append(
            {
                "format_version": AUTHENTICATED_COMPARATOR_PAIRS_VERSION,
                "pair_id": observation_id,
                "observation_id": observation_id,
                "model_pair_id": model_pair_id,
                "model_orientation": model_orientation,
                "human_preference": human_preference,
                "pair_outcome_canonical": observation["pair_outcome_canonical"],
                "supervision_category": "HUMAN_LABELED",
                "supervision_origin": "human",
                "source_clip_id": record["source_clip_id"],
                "source_split": record["source_split"],
                "mechanism_partition": record["mechanism_partition"],
                "perturbation_mechanism": record["perturbation_mechanism"],
                "adversarial_origin": bool(record["adversarial_origin"]),
                "objective_equivalence_control": is_equivalent,
                "objective_equivalence_control_provenance": equivalence_provenance,
                "rater_id": observation["rater_id"],
                "session_id": observation["session_id"],
                "trial_id": observation["trial_id"],
                "confidence": observation.get("confidence"),
                "reason_tags": observation.get("reason_tags", []),
                "evidence_authentication": {
                    "current_protocol_validated": True,
                    "protocol_id": protocol_validation.get("protocol_id"),
                    "freeze_id": protocol_validation.get("freeze_id"),
                    "served_trial_linkage_validated": True,
                    "pilot_report_completion_validated": True,
                },
            }
        )
    if not exported:
        raise ValueError("authenticated pilot contains no scorable direct-human comparisons")
    return {
        "format_version": AUTHENTICATED_COMPARATOR_PAIRS_VERSION,
        "evidence_origin": "human_pilot",
        "pairs": exported,
        "scorable_direct_human_pair_count": len(exported),
        "excluded_not_sure_observation_ids": sorted(excluded_not_sure),
        "excluded_non_pair_observation_count": excluded_non_pair,
        "observation_level_wilson_intervals_do_not_treat_repeats_as_new_raters": True,
        "authentication": {
            "protocol_id": protocol_validation.get("protocol_id"),
            "freeze_id": protocol_validation.get("freeze_id"),
            "validated_observation_count": len(observations),
            "pilot_manifest_sha256": file_sha256(pilot_path),
            "observations_sha256": file_sha256(observations_path),
            "served_playlist_sha256": file_sha256(served_path),
            "pilot_report_sha256": file_sha256(pilot_report_path),
        },
    }


def _rows_by_model_pair(rows: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} pair scores are missing")
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{label} pair score has no pair_id")
        if pair_id in indexed:
            duplicates.append(pair_id)
        indexed[pair_id] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate pair scores: {sorted(set(duplicates))}")
    return indexed


def _absolute_q_probabilities(row: dict[str, Any]) -> dict[str, float]:
    probability = row.get("probability_a_better")
    score_a = row.get("score_a")
    score_b = row.get("score_b")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
        or isinstance(score_a, bool)
        or not isinstance(score_a, (int, float))
        or not math.isfinite(float(score_a))
        or isinstance(score_b, bool)
        or not isinstance(score_b, (int, float))
        or not math.isfinite(float(score_b))
    ):
        raise ValueError("absolute_q pair score is malformed")
    delta = float(score_a) - float(score_b)
    expected_probability = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, delta))))
    if not math.isclose(float(probability), expected_probability, rel_tol=1.0e-6, abs_tol=1.0e-8):
        raise ValueError("absolute_q probability disagrees with its scalar scores")

    # Preserve the original absolute-q evaluator's explicit +/-0.1 equivalence band while
    # producing a symmetric three-class distribution for the shared evaluation machinery.
    equivalence_support = max(0.0, 1.0 - abs(float(probability) - 0.5) / 0.1)
    directional_support = 1.0 - equivalence_support
    return {
        "a_better": directional_support * float(probability),
        "b_better": directional_support * (1.0 - float(probability)),
        "approximately_equal": equivalence_support,
    }


def build_aligned_comparator_predictions(
    pair_export: dict[str, Any],
    absolute_evaluation: dict[str, Any],
    relative_experiment: dict[str, Any],
) -> dict[str, Any]:
    """Adapt native ablation outputs onto the same authenticated observation-level pairs."""
    pairs = pair_export.get("pairs")
    if not isinstance(pairs, list) or not all(isinstance(row, dict) for row in pairs):
        raise ValueError("authenticated comparator pair export is malformed")
    _validate_human_evidence(pairs)
    if absolute_evaluation.get("format_version") != "motionlab.fixed_rig_perceptual_evaluation.v1":
        raise ValueError("unsupported absolute_q evaluation artifact")
    if (
        relative_experiment.get("format_version") != "motionlab.relative_comparator_experiment.v3"
        or relative_experiment.get("status") != "complete"
    ):
        raise ValueError("relative-comparator experiment is not a completed v3 evaluation")
    selection = relative_experiment.get("variant_selection")
    selection_supervision = relative_experiment.get("selection_supervision")
    if (
        not isinstance(selection, dict)
        or selection.get("heldout_human_labels_used_for_selection") is not False
        or selection.get("heldout_human_audit_policy_used_for_selection") is not False
        or not isinstance(selection_supervision, dict)
        or selection_supervision.get("heldout_human_labels_used") is not False
        or selection_supervision.get("heldout_human_audit_policy_used") is not False
    ):
        raise ValueError("relative-comparator report lacks held-out-isolated selection provenance")
    selection_sha256 = selection_supervision.get("records_and_targets_sha256")
    if not isinstance(selection_sha256, str) or len(selection_sha256) != 64:
        raise ValueError("relative-comparator report lacks a selection-supervision digest")

    absolute_by_id = _rows_by_model_pair(absolute_evaluation.get("pair_scores"), label="absolute_q")
    ablations = relative_experiment.get("ablations")
    if not isinstance(ablations, dict):
        raise ValueError("relative-comparator report has no ablations")
    relative_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in ("relative_comparator_frozen", "relative_comparator_finetuned"):
        ablation = ablations.get(variant)
        if (
            not isinstance(ablation, dict)
            or ablation.get("selection_records_and_targets_sha256") != selection_sha256
        ):
            raise ValueError(f"{variant} is not bound to the selected supervision contract")
        metrics = ablation.get("metrics") if isinstance(ablation, dict) else None
        rows = metrics.get("pair_scores") if isinstance(metrics, dict) else None
        relative_by_variant[variant] = _rows_by_model_pair(rows, label=variant)

    predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in COMPARATOR_VARIANTS}
    for pair in pairs:
        evaluation_pair_id = _pair_id(pair)
        model_pair_id = str(pair["model_pair_id"])
        absolute_row = absolute_by_id.get(model_pair_id)
        if absolute_row is None:
            raise ValueError(f"absolute_q is missing model pair: {model_pair_id}")
        forward = _absolute_q_probabilities(absolute_row)
        predictions["absolute_q"].append(
            {
                "pair_id": evaluation_pair_id,
                "model_pair_id": model_pair_id,
                "probabilities": forward,
                "swapped_probabilities": {
                    "a_better": forward["b_better"],
                    "b_better": forward["a_better"],
                    "approximately_equal": forward["approximately_equal"],
                },
                "adapter": "native_scalar_score_with_original_0.1_equivalence_band",
            }
        )
        for variant, rows_by_id in relative_by_variant.items():
            source = rows_by_id.get(model_pair_id)
            if source is None:
                raise ValueError(f"{variant} is missing model pair: {model_pair_id}")
            forward_probabilities = _probabilities(source, "probabilities")
            swapped_probabilities = _probabilities(source, "swapped_probabilities")
            if forward_probabilities is None or swapped_probabilities is None:
                raise ValueError(f"{variant} has malformed probabilities: {model_pair_id}")
            predictions[variant].append(
                {
                    "pair_id": evaluation_pair_id,
                    "model_pair_id": model_pair_id,
                    "probabilities": forward_probabilities,
                    "swapped_probabilities": swapped_probabilities,
                    "adapter": "native_three_class_probabilities",
                }
            )
    return {
        "format_version": COMPARATOR_PREDICTION_ADAPTER_VERSION,
        "variants": {variant: {"predictions": rows} for variant, rows in predictions.items()},
        "absolute_q_equivalence_mapping": {
            "native_probability_band_half_width": 0.1,
            "mapping": "linear_equivalence_support_inside_band_then_symmetric_direction_mass",
        },
        "pair_count": len(pairs),
        "training_invoked": False,
    }


def prepare_post_label_comparator_evaluation(
    pair_dataset_directory: Path,
    pilot_manifest_path: Path,
    observations_path: Path,
    protocol_freeze_path: Path,
    pilot_report_path: Path,
    absolute_evaluation_path: Path,
    relative_experiment_path: Path,
    output_directory: Path,
    *,
    served_playlist_path: Path | None = None,
    swap_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Run the common post-label three-ablation evaluation without fitting any model."""
    output = Path(output_directory).resolve()
    pilot_root = Path(pilot_manifest_path).resolve().parent
    try:
        output.relative_to(pilot_root)
    except ValueError:
        pass
    else:
        raise ValueError("post-label comparator output must remain outside the active pilot")
    targets = {
        "pairs": output / "authenticated_direct_human_pairs.json",
        "predictions": output / "aligned_ablation_predictions.json",
        "evaluation": output / "comparator_evaluation.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"post-label comparator outputs already exist: {existing}")
    pair_export = build_authenticated_comparator_pair_export(
        pair_dataset_directory,
        pilot_manifest_path,
        observations_path,
        protocol_freeze_path,
        pilot_report_path,
        served_playlist_path=served_playlist_path,
    )
    absolute = _read_json_object(Path(absolute_evaluation_path), label="absolute_q evaluation")
    relative = _read_json_object(
        Path(relative_experiment_path), label="relative-comparator experiment"
    )
    aligned = build_aligned_comparator_predictions(pair_export, absolute, relative)
    evaluation = build_comparator_readiness_report(
        pair_export["pairs"],
        aligned,
        evidence_origin="human_pilot",
        swap_tolerance=swap_tolerance,
    )
    evaluation.update(
        {
            "post_label_format_version": POST_LABEL_COMPARATOR_EVALUATION_VERSION,
            "authenticated_pair_export": str(targets["pairs"]),
            "aligned_prediction_export": str(targets["predictions"]),
            "input_provenance": {
                "absolute_evaluation_sha256": file_sha256(Path(absolute_evaluation_path)),
                "relative_experiment_sha256": file_sha256(Path(relative_experiment_path)),
                **pair_export["authentication"],
            },
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (targets["pairs"], pair_export),
        (targets["predictions"], aligned),
        (targets["evaluation"], evaluation),
    ):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return evaluation
