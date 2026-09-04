"""Reproducible initial repair/exploitation matrix for the fixed-rig critic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.forensics import motions_from_sample
from motionlab.optimization.cmaes import CMAOptimizationConfig, optimize_motion_with_cma

FIRST_OPTIMIZATION_FAMILIES = ("foot_slide", "joint_jitter", "joint_pop")


def _floating_is_eligible(evaluation: dict[str, Any]) -> bool:
    learned = evaluation["known_mechanism_unseen_source"]["floating_contact"]
    deterministic = evaluation["trivial_baselines"]["deterministic_metric"]["floating_contact"]
    learned_auroc = learned.get("defect_auroc")
    baseline_auroc = deterministic.get("auroc")
    learned_ap = learned.get("defect_average_precision")
    baseline_ap = deterministic.get("average_precision")
    return (
        learned_auroc is not None
        and baseline_auroc is not None
        and learned_ap is not None
        and baseline_ap is not None
        and float(learned_auroc) > float(baseline_auroc)
        and float(learned_ap) > float(baseline_ap)
    )


def _representative_index(dataset: FixedRigSampleDataset, family: str) -> int:
    candidates = [
        index
        for index, record in enumerate(dataset.records)
        if record.get("source_split") == "test"
        and record.get("catalog_partition") == "train"
        and record.get("sample_role") == "hard_corruption"
        and record.get("corruption_family") == family
    ]
    if not candidates:
        raise ValueError(f"no known-mechanism held-source sample for {family!r}")
    for severity in ("moderate", "subtle", "clear", "near_threshold", "severe"):
        matching = [
            index
            for index in candidates
            if dataset.records[index].get("severity_label") == severity
        ]
        if matching:
            return matching[0]
    return candidates[0]


def run_first_cma_experiment(
    dataset_directory: Path,
    checkpoint_path: Path,
    evaluation_report_path: Path,
    output_directory: Path,
    *,
    seeds: tuple[int, ...] = (3301, 3302, 3303),
    coarse_max_iterations: int = 4,
    refined_max_iterations: int = 5,
    population_size: int | None = 8,
) -> dict[str, Any]:
    """Run several seeds in separately persisted red-team and production modes."""
    if len(seeds) < 2:
        raise ValueError("the initial experiment requires several (at least two) seeds")
    evaluation = json.loads(Path(evaluation_report_path).read_text(encoding="utf-8"))
    families = list(FIRST_OPTIMIZATION_FAMILIES)
    floating_eligible = _floating_is_eligible(evaluation)
    if floating_eligible:
        families.append("floating_contact")
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    selected: dict[str, dict[str, Any]] = {}
    clips = {}
    for family in families:
        index = _representative_index(dataset, family)
        _, candidate = motions_from_sample(dataset, index)
        clips[family] = candidate
        selected[family] = {
            "sample_id": dataset.records[index]["sample_id"],
            "source_clip_id": dataset.records[index]["source_clip_id"],
            "severity_label": dataset.records[index]["severity_label"],
            "corruption_mechanism": dataset.records[index]["corruption_mechanism"],
        }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for family in families:
        for mode in ("red_team", "production"):
            for seed in seeds:
                run_output = output / family / mode / f"seed_{seed}"
                result = optimize_motion_with_cma(
                    clips[family],
                    checkpoint_path,
                    dataset_directory,
                    run_output,
                    config=CMAOptimizationConfig(
                        seed=seed,
                        mode=mode,
                        objective_weights={family: 1.0},
                        coarse_max_iterations=coarse_max_iterations,
                        refined_max_iterations=refined_max_iterations,
                        population_size=population_size,
                    ),
                )
                runs.append(
                    {
                        "family": family,
                        "mode": mode,
                        "seed": seed,
                        "sample": selected[family],
                        "objective_before": result.objective_before,
                        "objective_after": result.objective_after,
                        "category": result.category,
                        "category_label": result.category_label,
                        "adversarial_training_eligible": result.adversarial_training_eligible,
                        "report": str(result.report_path),
                    }
                )
                partial = {
                    "format_version": "motionlab.first_cma_experiment.v1",
                    "checkpoint": str(checkpoint_path),
                    "evaluation_report": str(evaluation_report_path),
                    "floating_contact_eligible": floating_eligible,
                    "families": families,
                    "seeds": list(seeds),
                    "selected_samples": selected,
                    "runs": runs,
                }
                (output / "experiment.json").write_text(
                    json.dumps(partial, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
    categories = {
        str(category): sum(run["category"] == category for run in runs) for category in (1, 2, 3, 4)
    }
    report_value = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    if not isinstance(report_value, dict):
        raise ValueError("persisted experiment report must be an object")
    report: dict[str, Any] = report_value
    report["category_counts"] = categories
    report["adversarial_training_candidates"] = [
        run["report"] for run in runs if run["adversarial_training_eligible"]
    ]
    (output / "experiment.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
