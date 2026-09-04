"""CMA-ES validation on inverse motion problems representable by construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cma
import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.critic.forensics import motion_distances
from motionlab.io.npz import save_motion_npz
from motionlab.math.quaternion import quaternion_geodesic_distance
from motionlab.optimization.cmaes import (
    CMAOptimizationConfig,
    production_constraint_reasons,
)
from motionlab.optimization.spline import SplineResidualLayout, apply_spline_residual
from motionlab.testing.synthetic import make_synthetic_contact_walk

VALIDATION_VERSION = "motionlab.cma_representable_inverse.v1"
_UPPER_BODY_ROLES = (
    "spine_1",
    "chest",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)


def _apply_active_coefficients(
    clip: Any,
    layout: SplineResidualLayout,
    active_indices: NDArray[np.int64],
    values: ArrayLike,
    config: CMAOptimizationConfig,
) -> Any:
    full = np.zeros(layout.parameter_count, dtype=np.float64)
    full[active_indices] = np.asarray(values, dtype=np.float64)
    return apply_spline_residual(
        clip,
        layout,
        full,
        rotation_scale_rad=config.rotation_scale_rad,
        root_translation_scale_m=config.root_translation_scale_m,
        root_yaw_scale_rad=config.root_yaw_scale_rad,
    )


def _distance_objective(clean: Any, candidate: Any, rotation_scale_rad: float) -> float:
    root = candidate.root_translation_m - clean.root_translation_m
    rotation = quaternion_geodesic_distance(
        clean.local_quat_wxyz,
        candidate.local_quat_wxyz,
    )
    return float(
        np.mean(root * root) / (0.12**2) + np.mean(rotation * rotation) / (rotation_scale_rad**2)
    )


def _run_trial(
    dimension: int,
    seed: int,
    output_directory: Path,
    *,
    evaluations_per_dimension: int,
    success_objective: float,
) -> dict[str, Any]:
    clean_fixture = make_synthetic_contact_walk(num_cycles=2)
    clean = clean_fixture.with_updates(
        metadata={key: value for key, value in clean_fixture.metadata.items() if key != "loop_kind"}
    )
    config = CMAOptimizationConfig(
        seed=seed,
        mode="production",
        objective_source="deterministic_target",
        optimized_roles=_UPPER_BODY_ROLES,
    )
    layout = SplineResidualLayout.for_skeleton(
        clean.skeleton,
        control_points=8,
        optimized_roles=_UPPER_BODY_ROLES,
    )
    rotation_parameter_count = len(layout.joint_indices) * 3 * layout.control_points
    if dimension > rotation_parameter_count:
        raise ValueError(
            f"dimension {dimension} exceeds {rotation_parameter_count} safe rotation parameters"
        )
    active_indices = np.linspace(
        0,
        rotation_parameter_count - 1,
        dimension,
        dtype=np.int64,
    )
    if len(np.unique(active_indices)) != dimension:
        raise RuntimeError("active parameter selection produced duplicates")
    rng = np.random.default_rng(seed)
    corruption = rng.uniform(-0.35, 0.35, size=dimension)
    bad = _apply_active_coefficients(clean, layout, active_indices, corruption, config)
    optimum = -corruption
    recovered = _apply_active_coefficients(bad, layout, active_indices, optimum, config)
    clean_rejections = production_constraint_reasons(clean, clean, config)
    bad_rejections = production_constraint_reasons(bad, bad, config)
    optimum_rejections = production_constraint_reasons(bad, recovered, config)
    if clean_rejections or bad_rejections or optimum_rejections:
        raise RuntimeError(
            "representable inverse fixture is not production feasible: "
            f"clean={clean_rejections}, bad={bad_rejections}, optimum={optimum_rejections}"
        )
    known_objective = _distance_objective(clean, recovered, config.rotation_scale_rad)
    initial = np.zeros(dimension, dtype=np.float64)
    evaluation_count = 0
    evaluations_to_success: int | None = None
    best_values = initial.copy()
    best_objective = _distance_objective(clean, bad, config.rotation_scale_rad)
    history: list[dict[str, int | float]] = []
    budget = max(3000, evaluations_per_dimension * dimension)
    strategy = cma.CMAEvolutionStrategy(
        initial,
        0.25,
        {
            "bounds": [-config.coefficient_bound, config.coefficient_bound],
            "maxfevals": budget,
            "seed": seed,
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
            "ftarget": success_objective,
            "tolfun": 1.0e-14,
            "tolx": 1.0e-10,
        },
    )
    while not strategy.stop() and evaluation_count < budget:
        candidates = strategy.ask()
        objectives = []
        for values in candidates:
            candidate = _apply_active_coefficients(
                bad,
                layout,
                active_indices,
                values,
                config,
            )
            objective = _distance_objective(clean, candidate, config.rotation_scale_rad)
            objectives.append(objective)
            evaluation_count += 1
            if objective < best_objective:
                best_objective = objective
                best_values = np.asarray(values, dtype=np.float64).copy()
            if evaluations_to_success is None and objective <= success_objective:
                evaluations_to_success = evaluation_count
        strategy.tell(candidates, objectives)
        history.append(
            {
                "generation": len(history) + 1,
                "evaluations": evaluation_count,
                "best_objective": best_objective,
            }
        )
        if evaluations_to_success is not None:
            break
    final = _apply_active_coefficients(bad, layout, active_indices, best_values, config)
    distances = motion_distances(clean, final)
    final_rejections = production_constraint_reasons(bad, final, config)
    succeeded = evaluations_to_success is not None and not final_rejections
    output_directory.mkdir(parents=True, exist_ok=True)
    save_motion_npz(output_directory / "clean.npz", clean)
    save_motion_npz(output_directory / "corrupted.npz", bad)
    save_motion_npz(output_directory / "recovered.npz", final)
    np.savez_compressed(
        output_directory / "inverse_problem.npz",
        active_parameter_indices=active_indices,
        corruption_coefficients=corruption,
        known_optimum_coefficients=optimum,
        recovered_coefficients=best_values,
    )
    report = {
        "format_version": VALIDATION_VERSION,
        "dimension": dimension,
        "seed": seed,
        "parameterization": "same cubic SO(3) residual spline used by CMA; active subset",
        "active_parameter_indices": active_indices.tolist(),
        "corruption_coefficient_rms": float(np.sqrt(np.mean(corruption**2))),
        "known_optimum": optimum.tolist(),
        "known_optimum_objective": known_objective,
        "success_objective_threshold": success_objective,
        "evaluation_budget": budget,
        "evaluations": evaluation_count,
        "evaluations_to_success": evaluations_to_success,
        "success": succeeded,
        "final_objective": best_objective,
        "coefficient_rms_error_to_known_optimum": float(
            np.sqrt(np.mean((best_values - optimum) ** 2))
        ),
        "final_distance": distances,
        "production_constraints": {
            "clean": clean_rejections,
            "corrupted": bad_rejections,
            "known_optimum": optimum_rejections,
            "recovered": final_rejections,
        },
        "history": history,
        "artifacts": {
            "clean": str(output_directory / "clean.npz"),
            "corrupted": str(output_directory / "corrupted.npz"),
            "recovered": str(output_directory / "recovered.npz"),
            "coefficients": str(output_directory / "inverse_problem.npz"),
        },
    }
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_guaranteed_representable_cma_suite(
    output_directory: Path,
    *,
    dimensions: tuple[int, ...] = (10, 25, 50, 100),
    seeds: tuple[int, ...] = (4701, 4702, 4703),
    evaluations_per_dimension: int = 300,
    success_objective: float = 1.0e-6,
) -> dict[str, Any]:
    """Measure CMA recovery independently of critic and representation failures."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for dimension in dimensions:
        for seed in seeds:
            trial = _run_trial(
                dimension,
                seed,
                output / f"dimension_{dimension}" / f"seed_{seed}",
                evaluations_per_dimension=evaluations_per_dimension,
                success_objective=success_objective,
            )
            trials.append(trial)
            partial = {
                "format_version": VALIDATION_VERSION,
                "dimensions": list(dimensions),
                "seeds": list(seeds),
                "trials": trials,
            }
            (output / "suite.json").write_text(
                json.dumps(partial, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    by_dimension = {}
    for dimension in dimensions:
        selected = [trial for trial in trials if trial["dimension"] == dimension]
        successful = [trial for trial in selected if trial["success"]]
        by_dimension[str(dimension)] = {
            "trial_count": len(selected),
            "success_count": len(successful),
            "success_rate": len(successful) / len(selected),
            "median_evaluations_to_success": (
                None
                if not successful
                else float(np.median([trial["evaluations_to_success"] for trial in successful]))
            ),
            "maximum_final_objective": max(trial["final_objective"] for trial in selected),
            "maximum_rotation_rms_deg": max(
                trial["final_distance"]["local_rotation_geodesic_rms_deg"] for trial in selected
            ),
        }
    report_value = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    if not isinstance(report_value, dict):
        raise ValueError("persisted optimizer-validation suite must be an object")
    report: dict[str, Any] = report_value
    report["by_dimension"] = by_dimension
    report["overall_success_rate"] = float(np.mean([trial["success"] for trial in trials]))
    report["optimizer_decision"] = (
        "retain_cma_es"
        if report["overall_success_rate"] >= 0.9
        else "material_recovery_failure_requires_optimizer_review"
    )
    (output / "suite.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
