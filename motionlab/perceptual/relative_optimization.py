"""Gate-controlled anchor-relative perceptual CMA-ES."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from motionlab.critic.data import encode_motion_clip
from motionlab.critic.forensics import motion_distances
from motionlab.dataset.io import load_normalization_statistics
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.motion.clip import MotionClip
from motionlab.optimization.acceptance import assess_production_acceptance
from motionlab.optimization.block_cma import optimize_parameter_block_with_cma
from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.perceptual.optimization import (
    PerceptualOptimizationConfig,
    _acceptance_rejection,
    _perceptual_block,
)
from motionlab.perceptual.relative_experiment import load_relative_comparator

RELATIVE_CMA_VERSION = "motionlab.anchor_relative_perceptual_cma.v1"


@dataclass(frozen=True)
class AnchorComparison:
    probability_a_better: float
    probability_b_better: float
    probability_equal: float
    candidate_better_logit: float


def _relative_objective(
    anchor: MotionClip,
    comparator: Callable[[MotionClip, MotionClip], AnchorComparison],
    candidate: MotionClip,
) -> float:
    comparison = comparator(anchor, candidate)
    distance = motion_distances(anchor, candidate)
    edit = distance["joint_world_position_rms_m"] + (
        distance["local_rotation_geodesic_rms_deg"] / 30.0
    )
    return -comparison.candidate_better_logit + 1.0e-3 * edit


def run_gated_relative_perceptual_cma(
    pair_dataset_directory: Path,
    source_dataset_directory: Path,
    comparator_checkpoint: Path,
    relative_experiment_report: Path,
    output_directory: Path,
    *,
    config: PerceptualOptimizationConfig | None = None,
) -> dict[str, Any]:
    """Use only comparisons against the current anchor, recentering after acceptance."""
    settings = PerceptualOptimizationConfig() if config is None else config
    experiment = json.loads(Path(relative_experiment_report).read_text(encoding="utf-8"))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if not experiment["optimization_gate"]["passed"]:
        report = {
            "format_version": RELATIVE_CMA_VERSION,
            "status": "not_run_relative_comparator_gate_failed",
            "gate_reasons": experiment["optimization_gate"]["reasons"],
            "cma_es_invoked": False,
            "global_absolute_quality_assumed": False,
            "perceptual_cma_success_rate": None,
            "perceptual_cma_exploit_rate": None,
            "runs": [],
        }
        (output / "relative_perceptual_cma.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    if len(settings.seeds) < 3:
        raise ValueError("relative perceptual CMA requires at least three seeds")
    model, checkpoint = load_relative_comparator(comparator_checkpoint)
    if experiment.get("selected_variant") != checkpoint["variant"]:
        raise ValueError("comparator checkpoint does not match the selected gated variant")
    normalization = load_normalization_statistics(source_dataset_directory)

    def compare(anchor: MotionClip, candidate: MotionClip) -> AnchorComparison:
        features_a = encode_motion_clip(anchor, normalization)[None]
        features_b = encode_motion_clip(candidate, normalization)[None]
        probabilities = model(features_a, features_b).probabilities[0].detach().cpu().tolist()
        probability_b = float(probabilities[1])
        logit = math.log(max(probability_b, 1.0e-8) / max(1.0 - probability_b, 1.0e-8))
        return AnchorComparison(
            probability_a_better=float(probabilities[0]),
            probability_b_better=probability_b,
            probability_equal=float(probabilities[2]),
            candidate_better_logit=logit,
        )

    pairs = load_perceptual_pairs(pair_dataset_directory)
    selected_paths = []
    for record in pairs:
        if record["source_split"] == "test" and record["motion_a"] not in selected_paths:
            selected_paths.append(record["motion_a"])
        if len(selected_paths) >= settings.sources:
            break
    runs: list[dict[str, Any]] = []
    for source_offset, relative in enumerate(selected_paths):
        original = load_motion_npz(Path(pair_dataset_directory) / str(relative))
        for seed in settings.seeds:
            current = original
            evaluations = 0
            rounds = []
            started = time.perf_counter()
            for round_index in range(settings.rounds):
                anchor = current
                block = _perceptual_block(anchor, settings)
                rejection = partial(_acceptance_rejection, original)
                objective = partial(_relative_objective, anchor, compare)
                round_output = (
                    output
                    / f"source_{source_offset + 1}"
                    / f"seed_{seed}"
                    / f"round_{round_index + 1}"
                )
                result = optimize_parameter_block_with_cma(
                    anchor,
                    block,
                    objective,
                    round_output,
                    constraint_reasons=rejection,
                    seed=seed + round_index * 10_000,
                    maximum_evaluations=settings.evaluations_per_round,
                    initial_sigma=min(0.2, settings.coefficient_radius / 2.0),
                )
                evaluations += result.evaluations
                comparison = compare(anchor, result.motion)
                accepted = (
                    comparison.probability_b_better > 0.5
                    and comparison.probability_b_better
                    > max(comparison.probability_a_better, comparison.probability_equal)
                    and not rejection(result.motion)
                )
                if accepted:
                    current = result.motion
                rounds.append(
                    {
                        "round": round_index + 1,
                        "anchor_candidate_comparison": comparison.__dict__,
                        "accepted_and_recentered": accepted,
                        "evaluations": result.evaluations,
                        "optimization_report": str(result.report_path),
                    }
                )
            run_output = output / f"source_{source_offset + 1}" / f"seed_{seed}"
            save_motion_npz(run_output / "final.npz", current)
            acceptance = assess_production_acceptance(original, current)
            runs.append(
                {
                    "source_motion": str(relative),
                    "seed": seed,
                    "accepted_round_count": sum(row["accepted_and_recentered"] for row in rounds),
                    "deterministically_feasible": acceptance.all_required_satisfied,
                    "distance_from_source": motion_distances(original, current),
                    "evaluations": evaluations,
                    "runtime_seconds": time.perf_counter() - started,
                    "rounds": rounds,
                    "human_review_status": "pending",
                    "artifact": str(run_output / "final.npz"),
                }
            )
    score_successes = [
        run
        for run in runs
        if run["deterministically_feasible"] and int(run["accepted_round_count"]) > 0
    ]
    report = {
        "format_version": RELATIVE_CMA_VERSION,
        "status": "completed_pending_human_exploit_review",
        "selected_comparator_variant": checkpoint["variant"],
        "global_absolute_quality_assumed": False,
        "fitness": "logit P(candidate better than current anchor)",
        "recenter_after_each_accepted_round": True,
        "temporary_bradley_terry_ranking": {
            "used": False,
            "reason": "optional fallback requires evidence that anchor-only comparison is coarse",
        },
        "run_count": len(runs),
        "perceptual_cma_score_success_rate": len(score_successes) / max(len(runs), 1),
        "perceptual_cma_exploit_rate": None,
        "runs": runs,
    }
    (output / "relative_perceptual_cma.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
