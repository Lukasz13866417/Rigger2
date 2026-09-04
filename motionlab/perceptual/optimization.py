"""Gate-controlled trust-region CMA for a validated fixed-rig perceptual critic."""

from __future__ import annotations

import json
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
from motionlab.optimization.adaptive import (
    AdaptiveResidualLayout,
    AdaptiveSplineBlock,
    LocalSplinePatch,
    ResidualChannel,
)
from motionlab.optimization.block_cma import optimize_parameter_block_with_cma
from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.perceptual.experiment import load_perceptual_model

PERCEPTUAL_CMA_VERSION = "motionlab.perceptual_trust_region_cma.v1"


@dataclass(frozen=True)
class PerceptualOptimizationConfig:
    """Receding-horizon controls for locally defensible production edits."""

    seeds: tuple[int, ...] = (8301, 8302, 8303)
    rounds: int = 3
    evaluations_per_round: int = 300
    coefficient_radius: float = 0.35
    rotation_scale_rad: float = 0.08
    sources: int = 2


def _perceptual_block(
    motion: MotionClip,
    config: PerceptualOptimizationConfig,
) -> AdaptiveSplineBlock:
    roles = motion.skeleton.roles
    joints = tuple(
        dict.fromkeys(
            roles[role]
            for role in (
                "spine_1",
                "spine_2",
                "chest",
                "left_shoulder",
                "left_elbow",
                "right_shoulder",
                "right_elbow",
            )
            if role in roles
        )
    )
    axes = ("x", "y", "z")
    patches = tuple(
        LocalSplinePatch(
            "coarse",
            ResidualChannel(
                "rotation",
                joint,
                component,
                f"rotation:{motion.skeleton.joint_names[joint]}:{axes[component]}",
            ),
            0,
            motion.num_frames,
            4,
        )
        for joint in joints
        for component in range(3)
    )
    return AdaptiveSplineBlock(
        AdaptiveResidualLayout(patches),
        rotation_scale_rad=config.rotation_scale_rad,
        coefficient_bound=config.coefficient_radius,
    )


def _acceptance_rejection(reference: MotionClip, candidate: MotionClip) -> list[str]:
    acceptance = assess_production_acceptance(reference, candidate)
    reasons = list(acceptance.hard_invalid_reasons)
    reasons.extend(
        f"threshold:{name}"
        for name, result in acceptance.constraints.items()
        if result["required"] and not result["satisfied"]
    )
    return reasons


def _quality_objective(
    center: MotionClip,
    scorer: Callable[[MotionClip], float],
    candidate: MotionClip,
) -> float:
    distance = motion_distances(center, candidate)
    edit = distance["joint_world_position_rms_m"] + (
        distance["local_rotation_geodesic_rms_deg"] / 30.0
    )
    return -scorer(candidate) + 1.0e-3 * edit


def run_gated_perceptual_cma(
    pair_dataset_directory: Path,
    source_dataset_directory: Path,
    perceptual_checkpoint: Path,
    evaluation_report: Path,
    output_directory: Path,
    *,
    config: PerceptualOptimizationConfig | None = None,
) -> dict[str, Any]:
    """Run iterative local CMA only after the independent preference gate passes."""
    settings = PerceptualOptimizationConfig() if config is None else config
    evaluation = json.loads(Path(evaluation_report).read_text(encoding="utf-8"))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if not evaluation["optimization_gate"]["passed"]:
        misrankings = sorted(
            (
                row
                for row in evaluation.get("pair_scores", [])
                if row.get("subset") in {"heldout_source", "heldout_mechanism"}
                and row.get("correct") is False
            ),
            key=lambda row: float(row.get("confidence", 0.0)),
            reverse=True,
        )
        representative_exploits = [
            {
                "pair_id": row["pair_id"],
                "evaluation_subset": row["subset"],
                "mechanism": row["mechanism"],
                "expected_preference": row["preference"],
                "probability_a_better": row["probability_a_better"],
                "confidence": row["confidence"],
                "review_status": "queued_for_human_labeling",
            }
            for row in misrankings[:5]
        ]
        report = {
            "format_version": PERCEPTUAL_CMA_VERSION,
            "status": "not_run_preference_ranking_gate_failed",
            "gate_reasons": evaluation["optimization_gate"]["reasons"],
            "critic_score_improvement": None,
            "deterministic_feasibility_rate": None,
            "human_or_inspection_preference_rate": None,
            "perceptual_cma_success_rate": None,
            "perceptual_cma_exploit_rate": None,
            "representative_successful_refinements": [],
            "representative_critic_exploits": representative_exploits,
            "cma_es_invoked": False,
        }
        (output / "perceptual_cma.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    if len(settings.seeds) < 3:
        raise ValueError("perceptual CMA requires at least three seeds")
    model, _ = load_perceptual_model(perceptual_checkpoint)
    normalization = load_normalization_statistics(source_dataset_directory)

    def score(motion: MotionClip) -> float:
        features = encode_motion_clip(motion, normalization)
        return float(model(features[None]).score.item())

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
            before = score(original)
            evaluations = 0
            rounds = []
            started = time.perf_counter()
            for round_index in range(settings.rounds):
                block = _perceptual_block(current, settings)
                rejection = partial(_acceptance_rejection, original)
                objective = partial(_quality_objective, current, score)

                round_output = (
                    output
                    / f"source_{source_offset + 1}"
                    / f"seed_{seed}"
                    / f"round_{round_index + 1}"
                )
                result = optimize_parameter_block_with_cma(
                    current,
                    block,
                    objective,
                    round_output,
                    constraint_reasons=rejection,
                    seed=seed + round_index * 10_000,
                    maximum_evaluations=settings.evaluations_per_round,
                    initial_sigma=min(0.2, settings.coefficient_radius / 2.0),
                )
                evaluations += result.evaluations
                candidate_score = score(result.motion)
                if candidate_score > score(current) and not rejection(result.motion):
                    current = result.motion
                    accepted = True
                else:
                    accepted = False
                rounds.append(
                    {
                        "round": round_index + 1,
                        "candidate_score": candidate_score,
                        "accepted_and_recentered": accepted,
                        "evaluations": result.evaluations,
                        "optimization_report": str(result.report_path),
                    }
                )
            run_output = output / f"source_{source_offset + 1}" / f"seed_{seed}"
            save_motion_npz(run_output / "final.npz", current)
            final_acceptance = assess_production_acceptance(original, current)
            runs.append(
                {
                    "source_motion": str(relative),
                    "seed": seed,
                    "score_before": before,
                    "score_after": score(current),
                    "score_improvement": score(current) - before,
                    "deterministically_feasible": final_acceptance.all_required_satisfied,
                    "distance_from_source": motion_distances(original, current),
                    "evaluations": evaluations,
                    "runtime_seconds": time.perf_counter() - started,
                    "rounds": rounds,
                    "human_review_status": "pending",
                    "artifact": str(run_output / "final.npz"),
                }
            )
    successes = [
        run
        for run in runs
        if bool(run["deterministically_feasible"]) and float(run["score_improvement"]) > 0.0
    ]
    report = {
        "format_version": PERCEPTUAL_CMA_VERSION,
        "status": "completed_pending_human_exploit_review",
        "trust_region": {
            "rounds": settings.rounds,
            "coefficient_radius": settings.coefficient_radius,
            "rotation_scale_rad": settings.rotation_scale_rad,
            "recenter_after_each_accepted_round": True,
        },
        "run_count": len(runs),
        "perceptual_cma_score_success_rate": len(successes) / max(len(runs), 1),
        "perceptual_cma_exploit_rate": None,
        "exploit_rate_reason": "human or inspection labels are required",
        "runs": runs,
    }
    (output / "perceptual_cma.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
