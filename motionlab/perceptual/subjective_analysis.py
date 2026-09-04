"""Session-aware ordinal and pairwise analysis for subjective motion judgments."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import spearmanr

from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.perceptual.observation_auth import (
    resolve_frozen_evidence_paths,
    served_observation_key,
    validate_current_protocol_observations,
)
from motionlab.perceptual.subjective import (
    FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION,
    PILOT_MANIFEST_VERSION,
    RAW_OBSERVATION_VERSION,
    SUBJECTIVE_PROTOCOL_VERSION,
)

SUBJECTIVE_ANALYSIS_VERSION = "motionlab.subjective_analysis.v3"
TRAINING_EVIDENCE_VERSION = "motionlab.subjective_training_evidence.v2"
LEGACY_PILOT_MANIFEST_VERSION = "motionlab.subjective_pilot.v1"
REVISED_PILOT_MANIFEST_VERSION = "motionlab.subjective_pilot.v2"
LEGACY_RAW_OBSERVATION_VERSION = "motionlab.subjective_observation.v1"
REVISED_RAW_OBSERVATION_VERSION = "motionlab.subjective_observation.v2"
INSPECTION_TELEMETRY_VERSION = "motionlab.inspection_telemetry.v1"
CALIBRATION_ONLY = "CALIBRATION_ONLY"
HARD_FEASIBLE_PERCEPTUAL = "HARD_FEASIBLE_PERCEPTUAL"
EVALUATION_ANCHOR = "EVALUATION_ANCHOR"
_KNOWN_POPULATIONS = {
    CALIBRATION_ONLY,
    HARD_FEASIBLE_PERCEPTUAL,
    EVALUATION_ANCHOR,
}
_KNOWN_COHORTS = {"calibration_easy", "hard_feasible"}
_CALIBRATION_SEVERITY_RANK = {
    "clean": 0,
    "mild": 1,
    "medium": 2,
    "strong": 3,
    "severe": 4,
}
_MIN_FAMILY_PERCEPTIBILITY_OBSERVATIONS = 5
_MIN_FAMILY_PERCEPTIBILITY_LEVELS = 5
_MIN_EXPLORATORY_FAMILY_OBSERVATIONS = 3
_BASE_THRESHOLDS = np.asarray([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None]


def _row_populations(
    row: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    for key in ("stimulus_populations", "stimulus_population"):
        values = _string_list(row.get(key))
        if values:
            return values
    if stimulus_meta is None:
        return []
    return [
        str(stimulus_meta[stimulus_id]["stimulus_population"])
        for stimulus_id in _string_list(row.get("stimulus_ids"))
        if stimulus_id in stimulus_meta
        and stimulus_meta[stimulus_id].get("stimulus_population") is not None
    ]


def _row_cohorts(
    row: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    for key in ("measurement_cohorts", "measurement_cohort", "analysis_cohort"):
        values = _string_list(row.get(key))
        if values:
            return values
    if stimulus_meta is not None:
        values = [
            str(stimulus_meta[stimulus_id]["measurement_cohort"])
            for stimulus_id in _string_list(row.get("stimulus_ids"))
            if stimulus_id in stimulus_meta
            and stimulus_meta[stimulus_id].get("measurement_cohort") is not None
        ]
        if values:
            return values
    populations = _row_populations(row, stimulus_meta)
    if populations and all(value == HARD_FEASIBLE_PERCEPTUAL for value in populations):
        return ["hard_feasible"]
    if populations and all(value in {CALIBRATION_ONLY, EVALUATION_ANCHOR} for value in populations):
        return ["calibration_easy"]
    return []


def _row_training_eligible(
    row: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]] | None = None,
) -> bool:
    flag = row.get(
        "critic_training_eligible",
        row.get("perceptual_ground_truth_eligible"),
    )
    populations = _row_populations(row, stimulus_meta)
    return (
        flag is True
        and bool(populations)
        and all(value == HARD_FEASIBLE_PERCEPTUAL for value in populations)
    )


def _population_signature(
    row: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]],
) -> str:
    populations = sorted(set(_row_populations(row, stimulus_meta)))
    if not populations:
        return "LEGACY_UNCLASSIFIED"
    if len(populations) == 1:
        return populations[0]
    return "MIXED:" + "+".join(populations)


def _cohort_signature(
    row: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]],
) -> str:
    cohorts = sorted(set(_row_cohorts(row, stimulus_meta)))
    if not cohorts:
        return "legacy_unclassified"
    if len(cohorts) == 1:
        return cohorts[0]
    return "mixed:" + "+".join(cohorts)


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) and result >= 0.0 else None


def _rating_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        int(row["rating"])
        for row in rows
        if isinstance(row.get("rating"), int)
        and not isinstance(row.get("rating"), bool)
        and 1 <= int(row["rating"]) <= 7
    ]
    counts = {str(category): values.count(category) for category in range(1, 8)}
    total = len(values)
    return {
        "ordinal_observation_count": total,
        "counts": counts,
        "proportions": {
            key: (None if total == 0 else count / total) for key, count in counts.items()
        },
        "used_category_count": len(set(values)),
        "mean_rating_diagnostic_only": None if not values else float(np.mean(values)),
    }


def _difficulty_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = ("easy", "moderate", "difficult")
    values = [
        str(row.get("judgment_difficulty", row.get("post_rating_difficulty")))
        for row in rows
        if row.get("judgment_difficulty", row.get("post_rating_difficulty")) in allowed
    ]
    counts = {value: values.count(value) for value in allowed}
    return {
        "observation_count": len(rows),
        "answered_count": len(values),
        "response_rate": None if not rows else len(values) / len(rows),
        "counts": counts,
        "proportions_among_answers": {
            value: (None if not values else count / len(values)) for value, count in counts.items()
        },
        "difficult_rate_among_answers": (None if not values else counts["difficult"] / len(values)),
    }


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return None
    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def _weighted_kappa(first: list[int], second: list[int], categories: int = 7) -> float | None:
    if len(first) < 2 or len(first) != len(second):
        return None
    observed = np.zeros((categories, categories), dtype=np.float64)
    for left, right in zip(first, second, strict=True):
        observed[left - 1, right - 1] += 1.0
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    indices = np.arange(categories, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) / (categories - 1)) ** 2
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement <= 1.0e-12:
        return None
    return float(1.0 - np.sum(weights * observed) / expected_disagreement)


def _quality_standard_error(
    covariance: np.ndarray | None,
    source_parameter: int,
    variant_parameter: int,
) -> float | None:
    """Laplace-approximation error for the sum of correlated model parameters."""
    if covariance is None or covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        return None
    if max(source_parameter, variant_parameter) >= covariance.shape[0]:
        return None
    relevant = covariance[
        np.ix_([source_parameter, variant_parameter], [source_parameter, variant_parameter])
    ]
    if not np.all(np.isfinite(relevant)):
        return None
    relevant = 0.5 * (relevant + relevant.T)
    weights = np.asarray([1.0, 1.0], dtype=np.float64)
    variance = float(weights @ relevant @ weights)
    scale = max(1.0, float(np.max(np.abs(relevant))))
    if variance < -1.0e-10 * scale:
        return None
    return float(math.sqrt(max(variance, 0.0)))


def _fit_joint_latent_model(
    observations: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    singles = [
        row
        for row in observations
        if row.get("task_type") == "naturalness" and row.get("rating") is not None
    ]
    pairs = [
        row
        for row in observations
        if row.get("task_type") == "pair_comparison"
        and row.get("pair_outcome_canonical")
        in {"first_better", "second_better", "effectively_equal", "not_sure"}
    ]
    stimulus_meta = {str(row["stimulus_id"]): row for row in manifest["stimuli"]}
    item_ids = sorted(
        {
            str(value)
            for row in singles + pairs
            for value in row.get("stimulus_ids", [])
            if str(value) in stimulus_meta
        }
    )
    source_ids = sorted({str(stimulus_meta[item]["source_id"]) for item in item_ids})
    session_ids = sorted({str(row["session_id"]) for row in singles})
    rater_ids = sorted({str(row["rater_id"]) for row in singles + pairs})
    if len(singles) < 8 or len(item_ids) < 3:
        return {
            "status": "insufficient_human_observations",
            "minimum_single_observations": 8,
            "single_observation_count": len(singles),
            "pair_observation_count": len(pairs),
            "item_quality": [],
            "source_baselines": [],
            "session_calibration": [],
            "rater_consistency": [],
            "pair_model": {"status": "not_fit"},
            "canonical_score_is_raw_mean": False,
        }
    item_index = {value: index for index, value in enumerate(item_ids)}
    source_index = {value: index for index, value in enumerate(source_ids)}
    session_index = {value: index for index, value in enumerate(session_ids)}
    rater_index = {value: index for index, value in enumerate(rater_ids)}
    ni, ns, nh, nr = len(item_ids), len(source_ids), len(session_ids), len(rater_ids)
    offsets = {
        "source": 0,
        "variant": ns,
        "session_offset": ns + ni,
        "session_log_scale": ns + ni + nh,
        "rater_log_noise": ns + ni + 2 * nh,
        "tie": ns + ni + 2 * nh + nr,
        "unsure": ns + ni + 2 * nh + nr + 1,
    }
    n_parameters = offsets["unsure"] + 1

    def quality(vector: np.ndarray, item_id: str) -> float:
        source = str(stimulus_meta[item_id]["source_id"])
        return float(
            vector[offsets["source"] + source_index[source]]
            + vector[offsets["variant"] + item_index[item_id]]
        )

    def objective(vector: np.ndarray) -> float:
        loss = 0.0
        for row in singles:
            item = str(row["stimulus_ids"][0])
            if item not in item_index:
                continue
            session = session_index[str(row["session_id"])]
            rater = rater_index[str(row["rater_id"])]
            thresholds = (
                vector[offsets["session_offset"] + session]
                + math.exp(
                    float(np.clip(vector[offsets["session_log_scale"] + session], -3.0, 3.0))
                )
                * _BASE_THRESHOLDS
            )
            noise = math.exp(float(np.clip(vector[offsets["rater_log_noise"] + rater], -3.0, 3.0)))
            q = quality(vector, item)
            category = int(row["rating"])
            lower = 0.0 if category == 1 else float(expit((thresholds[category - 2] - q) / noise))
            upper = 1.0 if category == 7 else float(expit((thresholds[category - 1] - q) / noise))
            loss -= math.log(max(upper - lower, 1.0e-12))
        outcomes = {"first_better": 0, "second_better": 1, "effectively_equal": 2, "not_sure": 3}
        for row in pairs:
            first, second = [str(value) for value in row["stimulus_ids"]]
            if first not in item_index or second not in item_index:
                continue
            rater = rater_index[str(row["rater_id"])]
            noise = math.exp(float(np.clip(vector[offsets["rater_log_noise"] + rater], -3.0, 3.0)))
            delta = (quality(vector, first) - quality(vector, second)) / noise
            logits = np.asarray(
                [0.5 * delta, -0.5 * delta, vector[offsets["tie"]], vector[offsets["unsure"]]],
                dtype=np.float64,
            )
            loss -= float(logits[outcomes[str(row["pair_outcome_canonical"])]] - logsumexp(logits))
        # Weakly identify the structured model while preserving source-relative effects.
        loss += 0.08 * float(np.sum(vector[offsets["source"] : offsets["variant"]] ** 2))
        loss += 0.12 * float(np.sum(vector[offsets["variant"] : offsets["session_offset"]] ** 2))
        loss += 0.1 * float(
            np.sum(vector[offsets["session_offset"] : offsets["rater_log_noise"]] ** 2)
        )
        loss += 0.15 * float(np.sum(vector[offsets["rater_log_noise"] : offsets["tie"]] ** 2))
        loss += 0.03 * float(vector[offsets["tie"]] ** 2 + vector[offsets["unsure"]] ** 2)
        return loss

    initial = np.zeros(n_parameters, dtype=np.float64)
    initial[offsets["tie"]] = -0.6
    initial[offsets["unsure"]] = -1.0
    result = minimize(
        objective, initial, method="L-BFGS-B", options={"maxiter": 700, "ftol": 1.0e-10}
    )
    vector = np.asarray(result.x, dtype=np.float64)
    covariance: np.ndarray | None
    try:
        approximate_covariance = np.asarray(result.hess_inv.todense(), dtype=np.float64)
        covariance = (
            approximate_covariance
            if approximate_covariance.shape == (n_parameters, n_parameters)
            else None
        )
    except (AttributeError, TypeError, ValueError):
        covariance = None

    item_rows = []
    for item in item_ids:
        source = str(stimulus_meta[item]["source_id"])
        source_parameter = offsets["source"] + source_index[source]
        variant_parameter = offsets["variant"] + item_index[item]
        q = quality(vector, item)
        se = _quality_standard_error(covariance, source_parameter, variant_parameter)
        item_rows.append(
            {
                "stimulus_id": item,
                "source_id": source,
                "family": stimulus_meta[item]["family"],
                "stimulus_population": stimulus_meta[item].get("stimulus_population"),
                "measurement_cohort": stimulus_meta[item].get("measurement_cohort"),
                "critic_training_eligible": bool(
                    stimulus_meta[item].get(
                        "critic_training_eligible",
                        stimulus_meta[item].get("perceptual_ground_truth_eligible", False),
                    )
                ),
                "latent_quality": q,
                "quality_95_percent_interval": None
                if se is None
                else [q - 1.96 * se, q + 1.96 * se],
                "quality_interval_method": (
                    None
                    if se is None
                    else "laplace_approximation_from_lbfgs_inverse_hessian_with_full_covariance"
                ),
                "source_relative_variant_effect": float(vector[variant_parameter]),
                "observation_count": sum(
                    item in row.get("stimulus_ids", []) for row in singles + pairs
                ),
                "raw_mean_rating_diagnostic_only": float(
                    np.mean([row["rating"] for row in singles if row["stimulus_ids"][0] == item])
                )
                if any(row["stimulus_ids"][0] == item for row in singles)
                else None,
            }
        )
    sessions = []
    for session_id, index in session_index.items():
        offset = float(vector[offsets["session_offset"] + index])
        scale = math.exp(float(vector[offsets["session_log_scale"] + index]))
        sessions.append(
            {
                "session_id": session_id,
                "offset": offset,
                "scale": scale,
                "category_thresholds": (offset + scale * _BASE_THRESHOLDS).tolist(),
            }
        )
    raters = []
    for rater_id, index in rater_index.items():
        noise = math.exp(float(vector[offsets["rater_log_noise"] + index]))
        raters.append(
            {"rater_id": rater_id, "inconsistency_scale": noise, "consistency": 1.0 / noise}
        )
    sources = [
        {"source_id": value, "latent_baseline": float(vector[offsets["source"] + index])}
        for value, index in source_index.items()
    ]
    return {
        "status": "fit" if result.success else "fit_with_optimizer_warning",
        "optimizer_message": str(result.message),
        "negative_log_posterior": float(result.fun),
        "single_observation_count": len(singles),
        "pair_observation_count": len(pairs),
        "item_quality": sorted(item_rows, key=lambda row: row["stimulus_id"]),
        "source_baselines": sources,
        "session_calibration": sessions,
        "rater_consistency": raters,
        "pair_model": {
            "type": "Davidson-style four-outcome Bradley-Terry",
            "log_tie_propensity": float(vector[offsets["tie"]]),
            "log_not_sure_propensity": float(vector[offsets["unsure"]]),
        },
        "canonical_score_is_raw_mean": False,
        "quality_interval_method": (
            "laplace_approximation_from_lbfgs_inverse_hessian_with_full_covariance"
        ),
        "quality_intervals_are_optimizer_approximations": True,
        "joint_single_and_pair_fit": True,
    }


def _direct_pair_vs_ordinal_evidence(
    observations: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compare direct pair judgments with a model fitted to ordinal rows only.

    Keeping the pair rows out of the comparison model is important: otherwise a reported
    agreement could partly be the pair judgment agreeing with itself through the joint fit.
    """
    ordinal_rows = [
        row
        for row in observations
        if row.get("task_type") == "naturalness" and row.get("rating") is not None
    ]
    direct_rows = [
        row
        for row in observations
        if row.get("task_type") == "pair_comparison"
        and row.get("pair_outcome_canonical")
        in {"first_better", "second_better", "effectively_equal", "not_sure"}
        and len(_string_list(row.get("stimulus_ids"))) == 2
    ]
    ordinal_fit = _fit_joint_latent_model(ordinal_rows, manifest)
    ordinal_items = {str(row["stimulus_id"]): row for row in ordinal_fit.get("item_quality", [])}
    records: list[dict[str, Any]] = []
    decisive_agreement: list[bool] = []
    interval_supported_agreement: list[bool] = []
    tie_overlap: list[bool] = []
    by_family_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stimulus_meta = {str(row["stimulus_id"]): row for row in manifest["stimuli"]}
    covered = 0
    for row in direct_rows:
        first_id, second_id = _string_list(row["stimulus_ids"])
        first = ordinal_items.get(first_id)
        second = ordinal_items.get(second_id)
        direct_outcome = str(row["pair_outcome_canonical"])
        record: dict[str, Any] = {
            "observation_id": row.get("observation_id"),
            "pair_id": row.get("pair_id"),
            "stimulus_ids": [first_id, second_id],
            "direct_pair_outcome": direct_outcome,
            "ordinal_model_coverage": first is not None and second is not None,
            "ordinal_quality_delta_first_minus_second": None,
            "ordinal_derived_direction": None,
            "ordinal_95_percent_interval_relation": None,
            "directional_agreement": None,
            "families": sorted(
                {
                    str(stimulus_meta[stimulus_id].get("family", "unknown"))
                    for stimulus_id in (first_id, second_id)
                    if stimulus_id in stimulus_meta
                }
            ),
        }
        if first is None or second is None:
            records.append(record)
            for family in record["families"]:
                by_family_values[family].append(record)
            continue
        covered += 1
        delta = float(first["latent_quality"] - second["latent_quality"])
        ordinal_direction = (
            "first_better" if delta > 0.0 else "second_better" if delta < 0.0 else "equal"
        )
        first_interval = first.get("quality_95_percent_interval")
        second_interval = second.get("quality_95_percent_interval")
        interval_relation: str | None = None
        if first_interval is not None and second_interval is not None:
            if float(first_interval[0]) > float(second_interval[1]):
                interval_relation = "first_better_nonoverlapping"
            elif float(second_interval[0]) > float(first_interval[1]):
                interval_relation = "second_better_nonoverlapping"
            else:
                interval_relation = "overlap"
        directional = None
        if direct_outcome in {"first_better", "second_better"} and ordinal_direction != "equal":
            directional = direct_outcome == ordinal_direction
            decisive_agreement.append(directional)
        interval_supported = None
        if interval_relation in {
            "first_better_nonoverlapping",
            "second_better_nonoverlapping",
        } and direct_outcome in {"first_better", "second_better"}:
            interval_supported = interval_relation.startswith(direct_outcome)
            interval_supported_agreement.append(interval_supported)
        if direct_outcome == "effectively_equal" and interval_relation is not None:
            tie_overlap.append(interval_relation == "overlap")
        record.update(
            {
                "ordinal_quality_delta_first_minus_second": delta,
                "ordinal_derived_direction": ordinal_direction,
                "ordinal_95_percent_interval_relation": interval_relation,
                "directional_agreement": directional,
                "interval_supported_directional_agreement": interval_supported,
            }
        )
        records.append(record)
        for family in record["families"]:
            by_family_values[family].append(record)
    status = (
        "insufficient_ordinal_observations"
        if not ordinal_fit["status"].startswith("fit")
        else "no_direct_pair_observations"
        if not direct_rows
        else "ready"
    )
    by_family: dict[str, Any] = {}
    for family, values in sorted(by_family_values.items()):
        agreements = [
            bool(value["directional_agreement"])
            for value in values
            if value["directional_agreement"] is not None
        ]
        by_family[family] = {
            "direct_pair_observation_count": len(values),
            "ordinal_covered_direct_pair_count": sum(
                bool(value["ordinal_model_coverage"]) for value in values
            ),
            "decisive_comparison_count": len(agreements),
            "directional_agreement": (None if not agreements else float(np.mean(agreements))),
        }
    return {
        "status": status,
        "ordinal_model_status": ordinal_fit["status"],
        "ordinal_model_excludes_direct_pairs": True,
        "direct_pair_observation_count": len(direct_rows),
        "ordinal_covered_direct_pair_count": covered,
        "uncovered_direct_pair_count": len(direct_rows) - covered,
        "decisive_comparison_count": len(decisive_agreement),
        "directional_agreement": (
            None if not decisive_agreement else float(np.mean(decisive_agreement))
        ),
        "nonoverlapping_interval_comparison_count": len(interval_supported_agreement),
        "nonoverlapping_interval_directional_agreement": (
            None
            if not interval_supported_agreement
            else float(np.mean(interval_supported_agreement))
        ),
        "direct_tie_comparison_count": len(tie_overlap),
        "direct_tie_with_ordinal_interval_overlap_rate": (
            None if not tie_overlap else float(np.mean(tie_overlap))
        ),
        "not_sure_count": sum(
            row.get("pair_outcome_canonical") == "not_sure" for row in direct_rows
        ),
        "by_family": by_family,
        "records": records,
    }


def _repeat_metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        for group in row.get("hidden_repeat_group_ids", []):
            groups[(str(row["rater_id"]), str(row["task_type"]), str(group))].append(row)
    first: list[int] = []
    second: list[int] = []
    pair_agreement: list[bool] = []
    comparable_groups = 0
    for rows in groups.values():
        rows.sort(key=lambda row: str(row["timestamp_utc"]))
        if len(rows) < 2:
            continue
        comparable_groups += 1
        if rows[0].get("rating") is not None and rows[1].get("rating") is not None:
            first.append(int(rows[0]["rating"]))
            second.append(int(rows[1]["rating"]))
        elif rows[0].get("pair_outcome_canonical") is not None:
            pair_agreement.append(
                rows[0]["pair_outcome_canonical"] == rows[1]["pair_outcome_canonical"]
            )
    return {
        "repeat_group_count": comparable_groups,
        "ordinal_repeat_pair_count": len(first),
        "ordinal_exact_agreement": None
        if not first
        else float(np.mean(np.asarray(first) == np.asarray(second))),
        "ordinal_mean_absolute_difference": None
        if not first
        else float(np.mean(np.abs(np.asarray(first) - np.asarray(second)))),
        "ordinal_test_retest_spearman": _correlation(
            [float(value) for value in first], [float(value) for value in second]
        ),
        "ordinal_quadratic_weighted_kappa": _weighted_kappa(first, second),
        "pair_repeat_agreement": None if not pair_agreement else float(np.mean(pair_agreement)),
        "repeats_from_one_human_count_as_independent_raters": False,
    }


def _repeat_metrics_by_cohort(
    observations: list[dict[str, Any]],
    stimulus_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort in sorted(_KNOWN_COHORTS):
        rows = [row for row in observations if _cohort_signature(row, stimulus_meta) == cohort]
        result[cohort] = {
            "observation_count": len(rows),
            **_repeat_metrics(rows),
        }
    extra_cohorts = sorted(
        {
            _cohort_signature(row, stimulus_meta)
            for row in observations
            if _cohort_signature(row, stimulus_meta) not in _KNOWN_COHORTS
        }
    )
    for cohort in extra_cohorts:
        rows = [row for row in observations if _cohort_signature(row, stimulus_meta) == cohort]
        result[cohort] = {
            "observation_count": len(rows),
            **_repeat_metrics(rows),
        }
    return result


def _inspection_usage(observations: list[dict[str, Any]]) -> dict[str, Any]:
    rate_keys = ("0.25", "0.5", "1", "1.5", "2")
    replay_totals: list[float] = []
    inspection_seconds: list[float] = []
    speed_change_counts: list[float] = []
    playback_ms = {key: 0.0 for key in rate_keys}
    final_rate_counts: Counter[str] = Counter()
    final_view_counts: Counter[str] = Counter()
    camera_kind_counts: Counter[str] = Counter()
    trials_with_speed_change = 0
    trials_with_camera_change = 0
    telemetry_count = 0

    def rate_key(value: Any) -> str | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric < 0.0:
            return None
        for key in rate_keys:
            if math.isclose(numeric, float(key), abs_tol=1.0e-9):
                return key
        return None

    for row in observations:
        telemetry = row.get("inspection_telemetry")
        typed = telemetry if isinstance(telemetry, dict) else {}
        if typed:
            telemetry_count += 1
        raw_replay = typed.get("replay_count", row.get("replay_count", []))
        replay = raw_replay if isinstance(raw_replay, list) else []
        replay_values = [
            float(value)
            for value in replay
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ]
        replay_totals.append(sum(replay_values))

        total_inspection_ms = _finite_nonnegative(typed.get("total_inspection_time_ms"))
        if total_inspection_ms is not None:
            inspection_seconds.append(total_inspection_ms / 1000.0)

        speed_events = typed.get("speed_change_events", [])
        speed_event_count = (
            sum(isinstance(event, dict) for event in speed_events)
            if isinstance(speed_events, list)
            else 0
        )
        speed_change_counts.append(float(speed_event_count))
        trials_with_speed_change += speed_event_count > 0

        raw_speed_time = typed.get("playback_wall_time_by_speed_ms", {})
        if isinstance(raw_speed_time, dict):
            for raw_rate, raw_duration in raw_speed_time.items():
                key = rate_key(raw_rate)
                duration = _finite_nonnegative(raw_duration)
                if key is not None and duration is not None:
                    playback_ms[key] += duration

        final_rate = rate_key(typed.get("final_playback_rate"))
        if final_rate is not None:
            final_rate_counts[final_rate] += 1

        camera_events = typed.get("camera_change_events", [])
        valid_camera_events = (
            [event for event in camera_events if isinstance(event, dict)]
            if isinstance(camera_events, list)
            else []
        )
        for event in valid_camera_events:
            kind = str(event.get("kind", "unknown"))
            camera_kind_counts[kind] += 1
        # Explicit summary counts are authoritative for zoom/view/overlay and avoid losing
        # aggregate information if a bounded event log was truncated.
        for kind, field in (
            ("zoom", "zoom_change_count"),
            ("view", "view_change_count"),
            ("overlay", "overlay_change_count"),
        ):
            explicit = typed.get(field)
            if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
                camera_kind_counts[kind] += max(
                    0,
                    explicit - sum(str(event.get("kind")) == kind for event in valid_camera_events),
                )
        trials_with_camera_change += bool(valid_camera_events) or any(
            isinstance(typed.get(field), int) and typed.get(field, 0) > 0
            for field in ("zoom_change_count", "view_change_count", "overlay_change_count")
        )
        final_camera = typed.get("final_camera_state")
        if isinstance(final_camera, dict) and final_camera.get("preset") is not None:
            final_view_counts[str(final_camera["preset"])] += 1

    total_playback_ms = sum(playback_ms.values())
    return {
        "telemetry_format_version": INSPECTION_TELEMETRY_VERSION,
        "observation_count": len(observations),
        "telemetry_observation_count": telemetry_count,
        "missing_telemetry_count": len(observations) - telemetry_count,
        "replay_count_per_trial": _quantiles(replay_totals),
        "trials_with_replay_rate": (
            None
            if not replay_totals
            else float(np.mean(np.asarray(replay_totals, dtype=np.float64) > 0.0))
        ),
        "total_inspection_time_seconds": _quantiles(inspection_seconds),
        "speed_controls": {
            "speed_change_count": int(sum(speed_change_counts)),
            "speed_changes_per_trial": _quantiles(speed_change_counts),
            "trials_with_speed_change_rate": (
                None if not observations else trials_with_speed_change / len(observations)
            ),
            "playback_wall_time_seconds_by_speed": {
                key: value / 1000.0 for key, value in playback_ms.items()
            },
            "playback_wall_time_share_by_speed": {
                key: (None if total_playback_ms <= 0.0 else value / total_playback_ms)
                for key, value in playback_ms.items()
            },
            "final_playback_rate_counts": dict(sorted(final_rate_counts.items())),
        },
        "camera_controls": {
            "camera_change_count": int(sum(camera_kind_counts.values())),
            "trials_with_camera_change_rate": (
                None if not observations else trials_with_camera_change / len(observations)
            ),
            "change_counts_by_kind": dict(sorted(camera_kind_counts.items())),
            "zoom_change_count": int(camera_kind_counts["zoom"]),
            "view_change_count": int(camera_kind_counts["view"]),
            "reset_count": int(camera_kind_counts["reset"]),
            "orbit_change_count": int(camera_kind_counts["orbit"]),
            "overlay_change_count": int(camera_kind_counts["overlay"]),
            "final_view_preset_counts": dict(sorted(final_view_counts.items())),
        },
    }


def _row_families(
    row: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]],
) -> list[str]:
    calibration_families = {
        str(stimulus_meta[stimulus_id]["calibration_family"])
        for stimulus_id in _string_list(row.get("stimulus_ids"))
        if stimulus_id in stimulus_meta
        and stimulus_meta[stimulus_id].get("calibration_family") is not None
    }
    if calibration_families:
        return sorted(calibration_families)
    result = set(_string_list(row.get("families")))
    for stimulus_id in _string_list(row.get("stimulus_ids")):
        item = stimulus_meta.get(stimulus_id, {})
        if item.get("family") is not None:
            result.add(str(item["family"]))
    return sorted(result)


def _difficulty_breakdown(
    observations: list[dict[str, Any]],
    stimulus_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_cohort: dict[str, Any] = {}
    for cohort in sorted(
        _KNOWN_COHORTS | {_cohort_signature(row, stimulus_meta) for row in observations}
    ):
        rows = [row for row in observations if _cohort_signature(row, stimulus_meta) == cohort]
        by_cohort[cohort] = _difficulty_summary(rows)
    families = sorted(
        {family for row in observations for family in _row_families(row, stimulus_meta)}
    )
    return {
        "overall": _difficulty_summary(observations),
        "by_measurement_cohort": by_cohort,
        "by_family": {
            family: _difficulty_summary(
                [row for row in observations if family in _row_families(row, stimulus_meta)]
            )
            for family in families
        },
        "revised_protocol_requires_prompt_after_quality_judgment": True,
    }


def _calibration_ladder_value(
    row: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]],
) -> tuple[str, int] | None:
    stimulus_ids = _string_list(row.get("stimulus_ids"))
    if len(stimulus_ids) != 1 or row.get("rating") is None:
        return None
    item = stimulus_meta.get(stimulus_ids[0], {})
    family = item.get("calibration_family", row.get("calibration_family"))
    severity = item.get("calibration_severity", row.get("calibration_severity"))
    if family is None or str(severity) not in _CALIBRATION_SEVERITY_RANK:
        return None
    return str(family), _CALIBRATION_SEVERITY_RANK[str(severity)]


def _severity_order_accuracy(values: list[tuple[str, int, int]]) -> tuple[int, float | None]:
    comparisons: list[float] = []
    by_rater: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rater_id, severity, rating in values:
        by_rater[rater_id].append((severity, rating))
    for rows in by_rater.values():
        for left, right in combinations(rows, 2):
            if left[0] == right[0]:
                continue
            less_severe, more_severe = (left, right) if left[0] < right[0] else (right, left)
            if less_severe[1] > more_severe[1]:
                comparisons.append(1.0)
            elif less_severe[1] == more_severe[1]:
                comparisons.append(0.5)
            else:
                comparisons.append(0.0)
    return len(comparisons), None if not comparisons else float(np.mean(comparisons))


def _family_perceptibility(
    observations: list[dict[str, Any]],
    stimulus_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    families = sorted(
        {family for row in observations for family in _row_families(row, stimulus_meta)}
    )
    family_rows: dict[str, Any] = {}
    severity_rankable: list[dict[str, Any]] = []
    exploratory_rankable: list[dict[str, Any]] = []
    for family in families:
        rows = [row for row in observations if family in _row_families(row, stimulus_meta)]
        ladder: list[tuple[str, int, int]] = []
        for row in rows:
            value = _calibration_ladder_value(row, stimulus_meta)
            rating = row.get("rating")
            if (
                value is not None
                and value[0] == family
                and isinstance(rating, int)
                and not isinstance(rating, bool)
                and 1 <= rating <= 7
            ):
                ladder.append((str(row.get("rater_id", "")), value[1], rating))
        severity_levels = sorted({value[1] for value in ladder})
        ordering_count, ordering_accuracy = _severity_order_accuracy(ladder)
        severity_correlation = _correlation(
            [float(value[1]) for value in ladder],
            [float(8 - value[2]) for value in ladder],
        )
        confidence = [
            float(row["confidence"])
            for row in rows
            if isinstance(row.get("confidence"), int)
            and not isinstance(row.get("confidence"), bool)
            and 1 <= int(row["confidence"]) <= 5
        ]
        pair_rows = [row for row in rows if row.get("task_type") == "pair_comparison"]
        not_sure_count = sum(row.get("pair_outcome_canonical") == "not_sure" for row in pair_rows)
        decisive_count = sum(
            row.get("pair_outcome_canonical") in {"first_better", "second_better"}
            for row in pair_rows
        )
        sufficient = (
            len(ladder) >= _MIN_FAMILY_PERCEPTIBILITY_OBSERVATIONS
            and len(severity_levels) >= _MIN_FAMILY_PERCEPTIBILITY_LEVELS
        )
        difficulty = _difficulty_summary(rows)
        repeats = _repeat_metrics(rows)
        exploratory_components: dict[str, float] = {}
        if difficulty["answered_count"] >= _MIN_EXPLORATORY_FAMILY_OBSERVATIONS:
            answers = float(difficulty["answered_count"])
            exploratory_components["post_rating_ease"] = (
                float(difficulty["counts"]["easy"]) + 0.5 * float(difficulty["counts"]["moderate"])
            ) / answers
        if len(pair_rows) >= _MIN_EXPLORATORY_FAMILY_OBSERVATIONS:
            exploratory_components["pair_answer_certainty"] = 1.0 - (
                not_sure_count / len(pair_rows)
            )
        if len(confidence) >= _MIN_EXPLORATORY_FAMILY_OBSERVATIONS:
            exploratory_components["normalized_confidence"] = (
                float(np.mean(confidence)) - 1.0
            ) / 4.0
        if (
            repeats["ordinal_repeat_pair_count"] >= 2
            and repeats["ordinal_exact_agreement"] is not None
        ):
            exploratory_components["exact_repeat_agreement"] = float(
                repeats["ordinal_exact_agreement"]
            )
        exploratory_score = (
            None
            if len(rows) < _MIN_EXPLORATORY_FAMILY_OBSERVATIONS or not exploratory_components
            else float(np.mean(list(exploratory_components.values())))
        )
        result = {
            "status": "sufficient_for_ranking" if sufficient else "insufficient_data",
            "observation_count": len(rows),
            "ordinal_ladder_observation_count": len(ladder),
            "distinct_severity_level_count": len(severity_levels),
            "required_observation_count": _MIN_FAMILY_PERCEPTIBILITY_OBSERVATIONS,
            "required_severity_level_count": _MIN_FAMILY_PERCEPTIBILITY_LEVELS,
            "severity_to_unnaturalness_spearman": severity_correlation,
            "severity_order_comparison_count": ordering_count,
            "severity_order_accuracy_ties_half_credit": ordering_accuracy,
            "mean_confidence": None if not confidence else float(np.mean(confidence)),
            "pair_observation_count": len(pair_rows),
            "pair_not_sure_rate": (None if not pair_rows else not_sure_count / len(pair_rows)),
            "pair_decisive_preference_rate": (
                None if not pair_rows else decisive_count / len(pair_rows)
            ),
            "judgment_difficulty": difficulty,
            "repeat_agreement": repeats,
            "exploratory_perceptibility_score": exploratory_score,
            "exploratory_perceptibility_components": exploratory_components,
            "exploratory_rank_eligible": exploratory_score is not None,
        }
        family_rows[family] = result
        if sufficient:
            severity_rankable.append({"family": family, **result})
        if exploratory_score is not None and family != "clean_reference":
            exploratory_rankable.append({"family": family, **result})
    severity_rankable.sort(
        key=lambda row: (
            -float(row["severity_order_accuracy_ties_half_credit"] or 0.0),
            -float(row["severity_to_unnaturalness_spearman"] or 0.0),
            str(row["family"]),
        )
    )
    exploratory_rankable.sort(
        key=lambda row: (
            -float(row["exploratory_perceptibility_score"]),
            str(row["family"]),
        )
    )
    return {
        "ranking_basis": (
            "exploratory_mean_of_supported_post_rating_ease_pair_certainty_"
            "confidence_and_repeat_signals"
        ),
        "ranking_status": (
            "insufficient_attainable_scored_evidence"
            if not exploratory_rankable
            else "exploratory_self_report_and_consistency_ranking"
        ),
        "minimum_scored_observations_for_exploratory_ranking": (
            _MIN_EXPLORATORY_FAMILY_OBSERVATIONS
        ),
        "ranked_easiest_to_hardest": [row["family"] for row in exploratory_rankable],
        "ranking_is_exploratory_not_perceptual_accuracy": True,
        "tutorial_ladders_are_unscored_and_cannot_supply_ranking_observations": True,
        "severity_ladder_ranking_basis": "severity_order_accuracy_ties_half_credit",
        "minimum_observations_for_ranking": _MIN_FAMILY_PERCEPTIBILITY_OBSERVATIONS,
        "minimum_distinct_severity_levels_for_ranking": _MIN_FAMILY_PERCEPTIBILITY_LEVELS,
        "severity_ladder_ranked_easiest_to_hardest": [row["family"] for row in severity_rankable],
        "families": family_rows,
        "intended_severity_is_calibration_metadata_not_critic_ground_truth": True,
    }


def _measurement_quality_by_protocol(
    observations: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    default_settings = manifest.get("viewing_settings", {})

    def protocol(row: dict[str, Any]) -> str:
        settings = row.get("viewing_settings", default_settings)
        if isinstance(settings, dict) and settings.get("protocol") is not None:
            return str(settings["protocol"])
        return str(row.get("render_protocol_hash", manifest.get("render_protocol_hash", "unknown")))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[protocol(row)].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        confidence = [
            float(row["confidence"])
            for row in rows
            if isinstance(row.get("confidence"), int)
            and not isinstance(row.get("confidence"), bool)
            and 1 <= int(row["confidence"]) <= 5
        ]
        return {
            "observation_count": len(rows),
            "confidence_answered_count": len(confidence),
            "mean_confidence": None if not confidence else float(np.mean(confidence)),
            "confidence_quantiles": _quantiles(confidence),
            "repeat_agreement": _repeat_metrics(rows),
            "judgment_difficulty": _difficulty_summary(rows),
            "inspection_usage": _inspection_usage(rows),
        }

    summaries = {name: summarize(rows) for name, rows in sorted(grouped.items())}
    mannequin_rows: list[dict[str, Any]] = []
    skeleton_rows: list[dict[str, Any]] = []
    for name, rows in sorted(grouped.items()):
        lowered = name.lower()
        if "mannequin" in lowered:
            mannequin_rows.extend(rows)
        if "skeleton" in lowered:
            skeleton_rows.extend(rows)

    mannequin_summary = summarize(mannequin_rows) if mannequin_rows else None
    skeleton_summary = summarize(skeleton_rows) if skeleton_rows else None
    if not observations:
        status = "awaiting_human_pilot_no_protocol_observations"
    elif mannequin_summary is None:
        status = "legacy_or_unknown_protocol_only_no_mannequin_observations"
    elif skeleton_summary is None:
        status = "mannequin_absolute_only_no_matched_skeleton_baseline"
    elif (
        mannequin_summary["repeat_agreement"]["ordinal_repeat_pair_count"] < 2
        or skeleton_summary["repeat_agreement"]["ordinal_repeat_pair_count"] < 2
    ):
        status = "inconclusive_insufficient_matched_repeat_data"
    else:
        status = "exploratory_unmatched_combined_protocol_comparison"

    confidence_delta = None
    exact_repeat_delta = None
    repeat_mae_improvement = None
    if mannequin_summary is not None and skeleton_summary is not None:
        left = mannequin_summary["mean_confidence"]
        right = skeleton_summary["mean_confidence"]
        if left is not None and right is not None:
            confidence_delta = float(left - right)
        left_exact = mannequin_summary["repeat_agreement"]["ordinal_exact_agreement"]
        right_exact = skeleton_summary["repeat_agreement"]["ordinal_exact_agreement"]
        if left_exact is not None and right_exact is not None:
            exact_repeat_delta = float(left_exact - right_exact)
        left_mae = mannequin_summary["repeat_agreement"]["ordinal_mean_absolute_difference"]
        right_mae = skeleton_summary["repeat_agreement"]["ordinal_mean_absolute_difference"]
        if left_mae is not None and right_mae is not None:
            repeat_mae_improvement = float(right_mae - left_mae)
    return {
        "status": status,
        "comparison_scope": "combined_renderer_and_inspection_protocol_not_mannequin_only",
        "protocols": summaries,
        "mannequin_protocol_aggregate": mannequin_summary,
        "skeleton_protocol_aggregate": skeleton_summary,
        "mean_confidence_delta_mannequin_minus_skeleton": confidence_delta,
        "exact_repeat_agreement_delta_mannequin_minus_skeleton": exact_repeat_delta,
        "repeat_mae_improvement_skeleton_minus_mannequin": repeat_mae_improvement,
        "causal_mannequin_only_claim_permitted": False,
    }


def _anchor_drift(observations: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = [
        row
        for row in observations
        if row.get("anchor_status") == "hidden_session_anchor" and row.get("rating") is not None
    ]
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        sessions[str(row["session_id"])].append(row)
    rows = []
    for session_id, values in sorted(sessions.items()):
        x = [float(row["trial_index"]) for row in values]
        y = [float(row["rating"]) for row in values]
        slope = None
        if len(values) >= 3 and np.ptp(x) > 0:
            slope = float(np.polyfit(x, y, 1)[0])
        rows.append(
            {
                "session_id": session_id,
                "anchor_observation_count": len(values),
                "mean_rating_diagnostic_only": float(np.mean(y)),
                "within_session_rating_drift_per_trial": slope,
                "anchor_stimulus_ids": sorted(
                    {
                        str(stimulus_id)
                        for row in values
                        for stimulus_id in row.get("stimulus_ids", [])
                    }
                ),
            }
        )
    by_rater_anchor: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        stimulus_ids = [str(value) for value in row.get("stimulus_ids", [])]
        if len(stimulus_ids) == 1:
            by_rater_anchor[(str(row.get("rater_id", "")), stimulus_ids[0])].append(row)
    matched: list[dict[str, Any]] = []
    signed_deltas: list[float] = []
    for (rater_id, stimulus_id), values in sorted(by_rater_anchor.items()):
        by_session: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in values:
            raw_index = row.get("session_index")
            session_index = (
                int(raw_index)
                if isinstance(raw_index, int) and not isinstance(raw_index, bool)
                else 0
            )
            by_session[(session_index, str(row.get("session_id", "")))].append(float(row["rating"]))
        ordered = sorted((key, float(np.mean(ratings))) for key, ratings in by_session.items())
        if len(ordered) < 2:
            continue
        (baseline_index, baseline_session), baseline_mean = ordered[0]
        comparisons = []
        for (session_index, session_id), mean_rating in ordered[1:]:
            delta = mean_rating - baseline_mean
            signed_deltas.append(delta)
            comparisons.append(
                {
                    "session_index": session_index,
                    "session_id": session_id,
                    "mean_rating": mean_rating,
                    "delta_from_first_matched_session": delta,
                }
            )
        matched.append(
            {
                "rater_id": rater_id,
                "stimulus_id": stimulus_id,
                "baseline_session_index": baseline_index,
                "baseline_session_id": baseline_session,
                "baseline_mean_rating": baseline_mean,
                "later_session_comparisons": comparisons,
            }
        )
    return {
        "session_anchor_drift": rows,
        "matched_anchor_cross_session_drift": matched,
        "matched_anchor_cross_session_summary": {
            "matched_rater_anchor_count": len(matched),
            "comparison_count": len(signed_deltas),
            "mean_signed_rating_delta": (
                None if not signed_deltas else float(np.mean(signed_deltas))
            ),
            "mean_absolute_rating_delta": (
                None if not signed_deltas else float(np.mean(np.abs(signed_deltas)))
            ),
        },
        "pooled_session_means_are_composition_sensitive": True,
        "anchors_define_ground_truth_score_7": False,
    }


def _ab_swap_metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row.get("task_type") == "pair_comparison":
            pairs[(str(row["rater_id"]), str(row.get("pair_id")))].append(row)
    swap_checks: list[bool] = []
    for rows in pairs.values():
        for left, right in combinations(rows, 2):
            if left.get("presentation_order") == list(
                reversed(right.get("presentation_order", []))
            ):
                swap_checks.append(
                    left.get("pair_outcome_canonical") == right.get("pair_outcome_canonical")
                )
    equivalent = [
        row
        for row in observations
        if row.get("equivalence_control") and row.get("task_type") == "pair_comparison"
    ]
    false_preference = [
        row
        for row in equivalent
        if row.get("pair_outcome_canonical") in {"first_better", "second_better"}
    ]
    return {
        "a_b_swap_check_count": len(swap_checks),
        "a_b_swap_consistency": None if not swap_checks else float(np.mean(swap_checks)),
        "equivalent_control_count": len(equivalent),
        "equivalent_control_false_preference_rate": None
        if not equivalent
        else len(false_preference) / len(equivalent),
    }


def _transitivity(observations: list[dict[str, Any]]) -> dict[str, Any]:
    wins: set[tuple[str, str]] = set()
    nodes: set[str] = set()
    for row in observations:
        if row.get("task_type") != "pair_comparison" or row.get("pair_outcome_canonical") not in {
            "first_better",
            "second_better",
        }:
            continue
        first, second = [str(value) for value in row["stimulus_ids"]]
        nodes.update((first, second))
        wins.add(
            (first, second) if row["pair_outcome_canonical"] == "first_better" else (second, first)
        )
    cycles = 0
    triples = 0
    for a, b, c in combinations(sorted(nodes), 3):
        directions = ((a, b) in wins, (b, c) in wins, (c, a) in wins)
        reverse = ((b, a) in wins, (c, b) in wins, (a, c) in wins)
        if all(directions) or all(reverse):
            cycles += 1
        if all(x or y for x, y in zip(directions, reverse, strict=True)):
            triples += 1
    return {
        "fully_observed_triplet_count": triples,
        "intransitive_cycle_count": cycles,
        "transitivity_rate": None if triples == 0 else 1.0 - cycles / triples,
    }


def _breakdown(observations: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_key = {"family": "families", "style": "styles", "source": "source_ids"}[key]
    for row in observations:
        for value in row.get(raw_key, []):
            groups[str(value)].append(row)
    result = {}
    for value, rows in sorted(groups.items()):
        repeats = _repeat_metrics(rows)
        result[value] = {
            "observation_count": len(rows),
            "repeat_group_count": repeats["repeat_group_count"],
            "ordinal_exact_agreement": repeats["ordinal_exact_agreement"],
            "pair_repeat_agreement": repeats["pair_repeat_agreement"],
        }
    return result


def _reliability_dashboard(
    observations: list[dict[str, Any]],
    manifest: dict[str, Any],
    latent: dict[str, Any],
) -> dict[str, Any]:
    singles = [row for row in observations if row.get("rating") is not None]
    response_times = [
        value / 1000.0
        for row in observations
        if (value := _finite_nonnegative(row.get("response_time_ms"))) is not None
    ]
    stimulus_meta = {str(row["stimulus_id"]): row for row in manifest["stimuli"]}
    inspection = _inspection_usage(observations)
    rating_distribution = _rating_distribution(observations)
    quality_by_id = {row["stimulus_id"]: row for row in latent.get("item_quality", [])}
    counts = Counter(value for row in observations for value in row.get("stimulus_ids", []))
    needs_more = []
    for item in manifest["stimuli"]:
        if item.get("stimulus_population") == CALIBRATION_ONLY:
            continue
        fitted = quality_by_id.get(item["stimulus_id"])
        interval = None if fitted is None else fitted.get("quality_95_percent_interval")
        width = None if interval is None else float(interval[1] - interval[0])
        repeat_rows = [
            row
            for row in observations
            if item["stimulus_id"] in row.get("stimulus_ids", []) and row.get("rating") is not None
        ]
        disagreement = (
            None
            if len(repeat_rows) < 2
            else int(
                max(row["rating"] for row in repeat_rows)
                - min(row["rating"] for row in repeat_rows)
            )
        )
        priority = (
            1.0 / (counts[item["stimulus_id"]] + 1.0)
            + (0.0 if width is None else width)
            + (0.0 if disagreement is None else disagreement)
        )
        needs_more.append(
            {
                "stimulus_id": item["stimulus_id"],
                "observation_count": counts[item["stimulus_id"]],
                "latent_interval_width": width,
                "repeat_rating_range": disagreement,
                "priority": priority,
            }
        )
    mode: dict[str, Any] = {}
    for name in ("same_viewport_toggle", "sequential_neutral_gap"):
        rows = [row for row in observations if row.get("comparison_mode") == name]
        mode_response_times = [
            value / 1000.0
            for row in rows
            if (value := _finite_nonnegative(row.get("response_time_ms"))) is not None
        ]
        mode_replays = [
            sum(
                int(value)
                for value in row.get("replay_count", [])
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            )
            for row in rows
            if isinstance(row.get("replay_count", []), list)
        ]
        mode[name] = {
            "count": len(rows),
            "response_time_seconds": _quantiles(mode_response_times),
            "not_sure_rate": None
            if not rows
            else float(np.mean([row.get("pair_outcome_canonical") == "not_sure" for row in rows])),
            "replays_per_trial": None if not mode_replays else float(np.mean(mode_replays)),
        }
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_session[str(row["session_id"])].append(row)
    fatigue = []
    for session_id, rows in sorted(by_session.items()):
        trial_index = [float(row["trial_index"]) for row in rows]
        fatigue.append(
            {
                "session_id": session_id,
                "trial_count": len(rows),
                "response_time_trend": _correlation(
                    trial_index, [float(row["response_time_ms"]) for row in rows]
                ),
                "rating_trend": _correlation(
                    [
                        float(row["trial_index"])
                        for row in singles
                        if row["session_id"] == session_id
                    ],
                    [float(row["rating"]) for row in singles if row["session_id"] == session_id],
                ),
            }
        )
    return {
        "observation_count": len(observations),
        "rater_count": len({row["rater_id"] for row in observations}),
        "session_count": len({row["session_id"] for row in observations}),
        "category_usage": rating_distribution["counts"],
        "used_category_count": rating_distribution["used_category_count"],
        "rating_distribution": {
            "overall": rating_distribution,
            "by_measurement_cohort": {
                cohort: _rating_distribution(
                    [row for row in observations if _cohort_signature(row, stimulus_meta) == cohort]
                )
                for cohort in sorted(
                    _KNOWN_COHORTS | {_cohort_signature(row, stimulus_meta) for row in observations}
                )
            },
            "by_stimulus_population": {
                population: _rating_distribution(
                    [
                        row
                        for row in observations
                        if _population_signature(row, stimulus_meta) == population
                    ]
                )
                for population in sorted(
                    _KNOWN_POPULATIONS
                    | {_population_signature(row, stimulus_meta) for row in observations}
                )
            },
        },
        "repeat_agreement": _repeat_metrics(observations),
        "repeat_agreement_by_measurement_cohort": _repeat_metrics_by_cohort(
            observations, stimulus_meta
        ),
        "anchor_calibration": _anchor_drift(observations),
        "response_time_seconds": _quantiles(response_times),
        "replay_count_per_trial": inspection["replay_count_per_trial"],
        "trials_with_replay_rate": inspection["trials_with_replay_rate"],
        "inspection_usage": inspection,
        "judgment_difficulty": _difficulty_breakdown(observations, stimulus_meta),
        "family_perceptibility": _family_perceptibility(observations, stimulus_meta),
        "mannequin_and_inspection_measurement_quality": _measurement_quality_by_protocol(
            observations, manifest
        ),
        "training_eligibility": _training_eligibility_summary(observations, latent, stimulus_meta),
        "a_b_order": _ab_swap_metrics(observations),
        "pairwise_fit": {
            **_transitivity(observations),
            "model_status": latent["status"],
            "negative_log_posterior": latent.get("negative_log_posterior"),
        },
        "reliability_by_family": _breakdown(observations, "family"),
        "reliability_by_style": _breakdown(observations, "style"),
        "reliability_by_source": _breakdown(observations, "source"),
        "comparison_presentation_mode": mode,
        "fatigue": fatigue,
        "items_needing_more_judgments": sorted(
            needs_more, key=lambda row: (-row["priority"], row["stimulus_id"])
        )[:20],
    }


def _synthetic_agreement(
    pair_dataset_directory: Path,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = {str(row["pair_id"]): row for row in load_perceptual_pairs(pair_dataset_directory)}
    family: dict[str, list[bool]] = defaultdict(list)
    human_rows = 0
    for row in observations:
        pair_id = row.get("pair_id")
        if row.get("task_type") != "pair_comparison" or pair_id not in pairs:
            continue
        generated = pairs[str(pair_id)].get("preference")
        if generated not in {"a_better", "b_better", "approximately_equal"}:
            continue
        human_rows += 1
        expected = {
            "a_better": "first_better",
            "b_better": "second_better",
            "approximately_equal": "effectively_equal",
        }[str(generated)]
        family[str(pairs[str(pair_id)]["perturbation_mechanism"])].append(
            row.get("pair_outcome_canonical") == expected
        )
    return {
        "human_and_synthetic_supervision_kept_separate": True,
        "comparison_count": human_rows,
        "by_family": {
            name: {"count": len(values), "agreement": float(np.mean(values))}
            for name, values in sorted(family.items())
        },
    }


def _eligible_stimulus_ids(
    observations: list[dict[str, Any]],
    stimulus_meta: dict[str, dict[str, Any]],
) -> set[str]:
    observed_eligible = {
        stimulus_id
        for row in observations
        if _row_training_eligible(row, stimulus_meta)
        for stimulus_id in _string_list(row.get("stimulus_ids"))
    }
    result: set[str] = set()
    for stimulus_id in observed_eligible:
        item = stimulus_meta.get(stimulus_id, {})
        item_flag = item.get(
            "critic_training_eligible",
            item.get("perceptual_ground_truth_eligible"),
        )
        if item.get("stimulus_population") == HARD_FEASIBLE_PERCEPTUAL and item_flag is True:
            result.add(stimulus_id)
    return result


def _training_eligibility_summary(
    observations: list[dict[str, Any]],
    latent: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    eligible_ids = _eligible_stimulus_ids(observations, stimulus_meta)
    eligible_rows = [
        row
        for row in observations
        if _row_training_eligible(row, stimulus_meta)
        and bool(_string_list(row.get("stimulus_ids")))
        and all(
            stimulus_id in eligible_ids for stimulus_id in _string_list(row.get("stimulus_ids"))
        )
    ]
    eligible_observation_ids = {id(row) for row in eligible_rows}
    excluded_rows = [row for row in observations if id(row) not in eligible_observation_ids]
    latent_rows = list(latent.get("item_quality", []))
    excluded_by_population = Counter(
        _population_signature(row, stimulus_meta) for row in excluded_rows
    )
    return {
        "strict_contract": (
            "explicit critic/perceptual-ground-truth eligibility and exclusively "
            "HARD_FEASIBLE_PERCEPTUAL stimuli"
        ),
        "eligible_observation_count": len(eligible_rows),
        "excluded_observation_count": len(excluded_rows),
        "excluded_observation_counts_by_population": dict(sorted(excluded_by_population.items())),
        "eligible_latent_item_count": sum(
            str(row.get("stimulus_id")) in eligible_ids for row in latent_rows
        ),
        "excluded_latent_item_count": sum(
            str(row.get("stimulus_id")) not in eligible_ids for row in latent_rows
        ),
        "legacy_unclassified_is_training_eligible": False,
        "calibration_and_evaluation_anchor_training_eligible": False,
    }


def _training_evidence(
    observations: list[dict[str, Any]],
    latent: dict[str, Any],
    stimulus_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    eligible_ids = _eligible_stimulus_ids(observations, stimulus_meta)
    for row in observations:
        row_ids = _string_list(row.get("stimulus_ids"))
        if (
            not _row_training_eligible(row, stimulus_meta)
            or not row_ids
            or not all(stimulus_id in eligible_ids for stimulus_id in row_ids)
        ):
            continue
        evidence.append(
            {
                "format_version": TRAINING_EVIDENCE_VERSION,
                "evidence_kind": "raw_human_ordinal"
                if row.get("rating") is not None
                else "raw_human_pairwise",
                "supervision_origin": "human",
                "observation_id": row["observation_id"],
                "stimulus_ids": row["stimulus_ids"],
                "source_ids": row["source_ids"],
                "rater_id": row["rater_id"],
                "session_id": row["session_id"],
                "ordinal_rating": row.get("rating"),
                "pair_outcome": row.get("pair_outcome_canonical"),
                "is_tie": row.get("pair_outcome_canonical") == "effectively_equal",
                "is_not_sure": row.get("pair_outcome_canonical") == "not_sure",
                "confidence": row.get("confidence"),
                "reason_tags": row.get("reason_tags", []),
                "family": row.get("families", []),
                "stimulus_populations": _row_populations(row, stimulus_meta),
                "measurement_cohorts": _row_cohorts(row, stimulus_meta),
                "critic_training_eligible": True,
            }
        )
    for row in latent.get("item_quality", []):
        if str(row.get("stimulus_id")) not in eligible_ids:
            continue
        interval = row.get("quality_95_percent_interval")
        width = None if interval is None else float(interval[1] - interval[0])
        evidence.append(
            {
                "format_version": TRAINING_EVIDENCE_VERSION,
                "evidence_kind": "fitted_human_latent_quality",
                "supervision_origin": "human_model",
                "stimulus_ids": [row["stimulus_id"]],
                "source_ids": [row["source_id"]],
                "latent_quality": row["latent_quality"],
                "latent_uncertainty_interval": interval,
                "source_relative_variant_effect": row["source_relative_variant_effect"],
                "uncertainty_sampling_weight": None if width is None else 1.0 / (1.0 + width),
                "rater_session_effects_retained_in_model": True,
                "stimulus_population": HARD_FEASIBLE_PERCEPTUAL,
                "critic_training_eligible": True,
            }
        )
    return evidence


def _latent_model_observations(
    observations: list[dict[str, Any]],
    stimulus_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    # Visible calibration judgments are measurement-process diagnostics, never latent quality
    # ground truth. Legacy rows have no purpose metadata and remain analyzable as their historical
    # cohort, while still being excluded from revised training evidence.
    return [
        row for row in observations if CALIBRATION_ONLY not in _row_populations(row, stimulus_meta)
    ]


def _served_observation_key(row: dict[str, Any], *, observation: bool) -> tuple[Any, ...]:
    return served_observation_key(row, observation=observation)


def _validate_completed_observations(
    observations: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    require_current_protocol_fields: bool,
    served_playlist: list[dict[str, Any]] | None,
) -> None:
    """Reject malformed, duplicated, unserved, or manifest-inconsistent evidence."""
    if require_current_protocol_fields:
        if served_playlist is None:
            raise ValueError("current-protocol observations require a served-playlist log")
        validate_current_protocol_observations(observations, manifest, served_playlist)
        return
    stimulus_meta = {
        str(row["stimulus_id"]): row
        for row in manifest.get("stimuli", [])
        if isinstance(row, dict) and row.get("stimulus_id") is not None
    }
    supported_observations = {
        RAW_OBSERVATION_VERSION,
        LEGACY_RAW_OBSERVATION_VERSION,
        REVISED_RAW_OBSERVATION_VERSION,
    }
    served_keys = (
        None
        if served_playlist is None
        else {_served_observation_key(row, observation=False) for row in served_playlist}
    )
    seen_observation_ids: set[str] = set()
    for index, row in enumerate(observations):
        observation_id = row.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError(f"observation row {index} has no observation_id")
        if observation_id in seen_observation_ids:
            raise ValueError(f"duplicate subjective observation_id: {observation_id}")
        seen_observation_ids.add(observation_id)
        if row.get("format_version") not in supported_observations:
            raise ValueError("unsupported raw subjective observation")
        if row.get("render_protocol_hash") != manifest.get("render_protocol_hash"):
            raise ValueError("observation render protocol does not match the pilot")
        stimulus_ids = _string_list(row.get("stimulus_ids"))
        unknown = [value for value in stimulus_ids if value not in stimulus_meta]
        if unknown:
            raise ValueError(f"observation references unknown pilot stimuli: {unknown}")
        task_type = row.get("task_type")
        if task_type in {"naturalness", "style_adherence"}:
            rating = row.get("rating")
            if (
                len(stimulus_ids) != 1
                or not isinstance(rating, int)
                or isinstance(rating, bool)
                or not 1 <= rating <= 7
                or row.get("pair_outcome_canonical") is not None
            ):
                raise ValueError(f"observation {observation_id} has no valid completed rating")
        elif task_type == "pair_comparison":
            if (
                len(stimulus_ids) != 2
                or len(set(stimulus_ids)) != 2
                or row.get("rating") is not None
                or row.get("pair_outcome_canonical")
                not in {"first_better", "second_better", "effectively_equal", "not_sure"}
            ):
                raise ValueError(f"observation {observation_id} has no valid completed comparison")
        else:
            raise ValueError(f"observation {observation_id} has an unsupported task type")
        expected_sources = [stimulus_meta[value].get("source_id") for value in stimulus_ids]
        supplied_sources = row.get("source_ids")
        if isinstance(supplied_sources, list):
            expected_source_set = {str(value) for value in expected_sources}
            if {str(value) for value in supplied_sources} != expected_source_set:
                raise ValueError(
                    f"observation {observation_id} source lineage disagrees with pilot"
                )
            if require_current_protocol_fields and supplied_sources != list(
                dict.fromkeys(expected_sources)
            ):
                raise ValueError(
                    f"observation {observation_id} source ordering disagrees with pilot"
                )
        elif require_current_protocol_fields:
            raise ValueError(f"observation {observation_id} has no source lineage")
        expected_variants = [stimulus_meta[value].get("variant_id") for value in stimulus_ids]
        supplied_variants = row.get("variant_ids")
        if supplied_variants is not None and supplied_variants != expected_variants:
            raise ValueError(f"observation {observation_id} variant lineage disagrees with pilot")
        if require_current_protocol_fields and supplied_variants is None:
            raise ValueError(f"observation {observation_id} has no variant lineage")
        for required in ("rater_id", "session_id", "trial_id"):
            if not isinstance(row.get(required), str) or not row.get(required):
                raise ValueError(f"observation {observation_id} has no {required}")
        if (
            served_keys is not None
            and _served_observation_key(row, observation=True) not in served_keys
        ):
            raise ValueError(f"observation {observation_id} is not linked to a served trial")


def _validate_analysis_evidence(
    pilot_manifest_path: Path,
    observations_path: Path,
    manifest: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    protocol_freeze_path: Path | None,
    served_playlist_path: Path | None,
) -> dict[str, Any]:
    current_protocol = manifest.get("protocol_version") in {
        SUBJECTIVE_PROTOCOL_VERSION,
        FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION,
    }
    served_rows: list[dict[str, Any]] | None = None
    if current_protocol:
        freeze = (
            Path(protocol_freeze_path).resolve()
            if protocol_freeze_path is not None
            else Path(pilot_manifest_path).with_name("protocol_freeze.json")
        )
        if not freeze.is_file():
            raise ValueError("current-protocol analysis requires its protocol freeze")
        served, validation = resolve_frozen_evidence_paths(
            pilot_manifest_path,
            observations_path,
            freeze,
            served_playlist_path=served_playlist_path,
        )
        if not served.is_file():
            raise ValueError("current-protocol analysis requires its served-playlist log")
        served_rows = _read_jsonl(served)
        mode = "frozen_current_protocol_with_exact_served_trial_authentication"
    else:
        validation = None
        mode = "synthetic_or_legacy_schema_and_manifest_validation"
    _validate_completed_observations(
        observations,
        manifest,
        require_current_protocol_fields=current_protocol,
        served_playlist=served_rows,
    )
    return {
        "mode": mode,
        "validated_observation_count": len(observations),
        "protocol_freeze_validated": validation is not None,
        "canonical_frozen_evidence_paths_validated": validation is not None,
        "served_trial_linkage_validated": served_rows is not None,
        "manifest_trial_metadata_authenticated": current_protocol,
        "freeze_id": None if validation is None else validation["freeze_id"],
    }


def _pilot_completion_audit(
    observations: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Measure completion against the manifest rather than a fit-size heuristic."""
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        return {
            "completion_contract_available": False,
            "complete": False,
            "reason": "manifest_has_no_session_trial_contract",
            "expected_base_trial_count_per_rater": None,
            "expected_adaptive_trial_count_per_rater": None,
            "expected_total_trial_count_per_rater": None,
            "completed_rater_count": 0,
        }
    expected_base: set[str] = set()
    for session in sessions:
        if not isinstance(session, dict) or not isinstance(session.get("trials"), list):
            return {
                "completion_contract_available": False,
                "complete": False,
                "reason": "manifest_session_has_no_trial_contract",
                "expected_base_trial_count_per_rater": None,
                "expected_adaptive_trial_count_per_rater": None,
                "expected_total_trial_count_per_rater": None,
                "completed_rater_count": 0,
            }
        expected_base.update(
            str(trial["trial_id"])
            for trial in session["trials"]
            if isinstance(trial, dict) and trial.get("trial_id") is not None
        )
    policy = manifest.get("adaptive_policy", {})
    expected_adaptive = (
        int(policy.get("maximum_followups_total", 0))
        if isinstance(policy, dict) and policy.get("enabled")
        else 0
    )
    by_rater: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        rater_id = row.get("rater_id")
        if isinstance(rater_id, str) and rater_id:
            by_rater[rater_id].append(row)
    coverage: list[dict[str, Any]] = []
    for rows in by_rater.values():
        base_ids = {
            str(row["trial_id"])
            for row in rows
            if row.get("trial_id") is not None and str(row["trial_id"]) in expected_base
        }
        adaptive_ids = {
            str(row["trial_id"])
            for row in rows
            if row.get("trial_id") is not None
            and str(row.get("schedule_reason", "")).startswith("adaptive:")
        }
        coverage.append(
            {
                "completed_base_trial_count": len(base_ids),
                "completed_adaptive_trial_count": len(adaptive_ids),
                "complete": len(base_ids) == len(expected_base)
                and len(adaptive_ids) >= expected_adaptive,
            }
        )
    complete_count = sum(bool(row["complete"]) for row in coverage)
    best_base = max((int(row["completed_base_trial_count"]) for row in coverage), default=0)
    best_adaptive = max((int(row["completed_adaptive_trial_count"]) for row in coverage), default=0)
    return {
        "completion_contract_available": True,
        "complete": complete_count > 0,
        "reason": (
            "at_least_one_rater_completed_manifest_contract"
            if complete_count
            else "no_rater_has_completed_manifest_contract"
        ),
        "expected_base_trial_count_per_rater": len(expected_base),
        "expected_adaptive_trial_count_per_rater": expected_adaptive,
        "expected_total_trial_count_per_rater": len(expected_base) + expected_adaptive,
        "observed_rater_count": len(coverage),
        "completed_rater_count": complete_count,
        "best_completed_base_trial_count": best_base,
        "best_completed_adaptive_trial_count": best_adaptive,
    }


def analyze_subjective_pilot(
    pair_dataset_directory: Path,
    pilot_manifest_path: Path,
    observations_path: Path,
    output_directory: Path,
    *,
    protocol_freeze_path: Path | None = None,
    served_playlist_path: Path | None = None,
) -> dict[str, Any]:
    """Fit the joint model and export reliability/training artifacts without retraining."""
    pilot_manifest_path = Path(pilot_manifest_path).resolve()
    observations_path = Path(observations_path).resolve()
    manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    supported_pilots = {
        PILOT_MANIFEST_VERSION,
        LEGACY_PILOT_MANIFEST_VERSION,
        REVISED_PILOT_MANIFEST_VERSION,
    }
    if manifest.get("format_version") not in supported_pilots:
        raise ValueError("unsupported pilot manifest")
    output = Path(output_directory).resolve()
    if manifest.get("protocol_version") in {
        SUBJECTIVE_PROTOCOL_VERSION,
        FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION,
    }:
        try:
            output.relative_to(pilot_manifest_path.parent)
        except ValueError:
            pass
        else:
            raise ValueError("analysis output must remain outside the active pilot directory")
    if output.exists():
        raise ValueError(f"analysis output must be fresh: {output}")
    observations = _read_jsonl(observations_path)
    observation_validation = _validate_analysis_evidence(
        pilot_manifest_path,
        observations_path,
        manifest,
        observations,
        protocol_freeze_path=protocol_freeze_path,
        served_playlist_path=served_playlist_path,
    )
    stimulus_meta = {str(row["stimulus_id"]): row for row in manifest["stimuli"]}
    latent_observations = _latent_model_observations(observations, stimulus_meta)
    latent = _fit_joint_latent_model(latent_observations, manifest)
    dashboard = _reliability_dashboard(observations, manifest, latent)
    direct_pair_vs_ordinal = _direct_pair_vs_ordinal_evidence(latent_observations, manifest)
    dashboard["direct_pair_vs_ordinal_evidence"] = direct_pair_vs_ordinal
    agreement = _synthetic_agreement(pair_dataset_directory, observations)
    training_evidence = _training_evidence(observations, latent, stimulus_meta)
    completion = _pilot_completion_audit(observations, manifest)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "latent_quality.json", latent)
    _write_json(output / "reliability_dashboard.json", dashboard)
    _write_json(output / "direct_pair_vs_ordinal_evidence.json", direct_pair_vs_ordinal)
    _write_json(output / "human_synthetic_agreement.json", agreement)
    _write_jsonl(output / "training_evidence.jsonl", training_evidence)
    status = (
        "pilot_complete_ready_for_inspection"
        if completion["complete"] and latent["status"].startswith("fit")
        else (
            "pilot_in_progress_analysis_available"
            if observations and latent["status"].startswith("fit")
            else "awaiting_human_pilot"
        )
    )
    report = {
        "format_version": SUBJECTIVE_ANALYSIS_VERSION,
        "status": status,
        "human_observation_count": len(observations),
        "latent_model_observation_count": len(latent_observations),
        "latent_model_status": latent["status"],
        "pilot_completion": completion,
        "observation_validation": observation_validation,
        "observation_format_counts": dict(
            sorted(Counter(str(row["format_version"]) for row in observations).items())
        ),
        "training_evidence_record_count": len(training_evidence),
        "training_eligibility": dashboard["training_eligibility"],
        "raw_mean_used_as_canonical_score": False,
        "critic_retraining_started": False,
        "critic_retraining_blocked_until_pilot_report_inspected": True,
        "outputs": {
            "latent_quality": str(output / "latent_quality.json"),
            "reliability_dashboard": str(output / "reliability_dashboard.json"),
            "direct_pair_vs_ordinal_evidence": str(output / "direct_pair_vs_ordinal_evidence.json"),
            "human_synthetic_agreement": str(output / "human_synthetic_agreement.json"),
            "training_evidence": str(output / "training_evidence.jsonl"),
        },
    }
    _write_json(output / "pilot_report.json", report)
    return report
