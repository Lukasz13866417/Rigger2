"""Two-stage BIPOP-CMA-ES over smooth motion residuals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cma
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from motionlab.critic.data import encode_motion_clip
from motionlab.critic.forensics import motion_deterministic_metrics, motion_distances
from motionlab.critic.model import FIRST_CRITIC_DEFECTS
from motionlab.critic.training import model_from_checkpoint
from motionlab.dataset.io import load_normalization_statistics
from motionlab.io.npz import save_motion_npz
from motionlab.math.quaternion import quaternion_geodesic_distance
from motionlab.motion.clip import MotionClip
from motionlab.optimization.spline import (
    DEFAULT_OPTIMIZED_ROLES,
    SplineResidualLayout,
    apply_spline_residual,
    refine_coefficients,
)
from motionlab.visualization.skeleton_plot import save_motion_comparison_strip

OPTIMIZATION_VERSION = "motionlab.bipop_cma_spline.v2"

_FAMILY_METRIC = {
    "foot_slide": "maximum_stance_slip_cm",
    "ground_penetration": "maximum_ground_penetration_cm",
    "floating_contact": "maximum_floating_contact_cm",
    "loop_seam": "loop_seam",
    "joint_pop": "angular_smoothness_rad_s3_p99",
    "joint_jitter": "angular_smoothness_rad_s3_p99",
    "speed_inconsistency": "absolute_target_speed_error_mps",
}

_TARGET_GAIN = {
    "maximum_stance_slip_cm": 0.2,
    "maximum_ground_penetration_cm": 0.2,
    "maximum_floating_contact_cm": 0.2,
    "loop_seam": 0.02,
    "angular_smoothness_rad_s3_p99": 100.0,
    "absolute_target_speed_error_mps": 0.02,
}

_COLLATERAL_WORSENING = {
    "maximum_stance_slip_cm": 0.5,
    "maximum_ground_penetration_cm": 0.5,
    "maximum_floating_contact_cm": 0.5,
    "loop_seam": 0.2,
    "angular_smoothness_rad_s3_p99": 500.0,
    "absolute_target_speed_error_mps": 0.05,
}


class CMAOptimizationConfig(BaseModel):
    """Frozen controls for one reproducible red-team or production optimization run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 3301
    mode: Literal["red_team", "production"] = "red_team"
    objective_source: Literal["mode_default", "learned_severity", "deterministic_target"] = (
        "mode_default"
    )
    objective_weights: dict[str, float] = Field(default_factory=lambda: {"foot_slide": 1.0})
    optimized_roles: tuple[str, ...] = DEFAULT_OPTIMIZED_ROLES
    coarse_control_points: int = 4
    refined_control_points: int = 8
    coarse_max_iterations: int = Field(default=6, ge=1)
    refined_max_iterations: int = Field(default=8, ge=1)
    bipop_restarts: int = Field(default=1, ge=1, le=8)
    population_size: int | None = Field(default=None, ge=4)
    initial_sigma: float = Field(default=0.25, gt=0.0)
    coefficient_bound: float = Field(default=1.5, gt=0.0)
    rotation_scale_rad: float = Field(default=0.35, gt=0.0)
    root_translation_scale_m: float = Field(default=0.12, gt=0.0)
    root_yaw_scale_rad: float = Field(default=0.35, gt=0.0)
    absurd_root_offset_m: float = Field(default=1.0, gt=0.0)
    absurd_rotation_offset_rad: float = Field(default=1.6, gt=0.0)
    maximum_absolute_normalized_feature: float = Field(default=1000.0, gt=0.0)
    production_speed_error_mps: float = Field(default=0.05, ge=0.0)
    production_penetration_cm: float = Field(default=2.0, ge=0.0)
    production_joint_limit_slack_deg: float = Field(default=0.5, ge=0.0)
    production_loop_seam_slack: float = Field(default=0.02, ge=0.0)
    production_displacement_relative_tolerance: float = Field(default=0.1, ge=0.0)

    @field_validator("objective_weights")
    @classmethod
    def objective_is_supported(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("objective_weights cannot be empty")
        unknown = set(value).difference(FIRST_CRITIC_DEFECTS)
        if unknown:
            raise ValueError(f"unknown objective defect families: {sorted(unknown)}")
        if any(not np.isfinite(weight) or weight <= 0.0 for weight in value.values()):
            raise ValueError("all objective weights must be finite and positive")
        total = sum(value.values())
        return {name: float(weight / total) for name, weight in sorted(value.items())}

    @model_validator(mode="after")
    def parameterization_is_fixed_for_first_experiment(self) -> CMAOptimizationConfig:
        if self.coarse_control_points != 4 or self.refined_control_points != 8:
            raise ValueError("the first learned experiment requires the 4-to-8 spline schedule")
        if self.mode == "production" and self.objective_source == "learned_severity":
            raise ValueError(
                "production cannot optimize learned hard-defect heads; use deterministic targets"
            )
        return self

    @property
    def resolved_objective_source(self) -> Literal["learned_severity", "deterministic_target"]:
        if self.objective_source == "mode_default":
            return "learned_severity" if self.mode == "red_team" else "deterministic_target"
        return self.objective_source


@dataclass(frozen=True)
class OptimizationResult:
    """Final outcome and artifact locations from one complete two-stage run."""

    output_directory: Path
    final_motion_path: Path
    report_path: Path
    objective_before: float
    objective_after: float
    category: int
    category_label: str
    adversarial_training_eligible: bool


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prediction(output: Any) -> dict[str, dict[str, float]]:
    return {
        "probability": {
            family: float(torch.sigmoid(output.clip_logits)[0, index].cpu())
            for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
        "severity": {
            family: float(output.clip_severity[0, index].cpu())
            for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
        "family_rank": {
            family: float(output.family_ranking_score[0, index].cpu())
            for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
    }


def production_constraint_reasons(
    original: MotionClip,
    candidate: MotionClip,
    config: CMAOptimizationConfig,
    *,
    original_metrics: Mapping[str, float | None] | None = None,
    candidate_metrics: Mapping[str, float | None] | None = None,
) -> list[str]:
    """Evaluate numerical, physical, and task constraints for a production candidate."""
    reasons: list[str] = []
    root_offset = np.linalg.norm(
        candidate.root_translation_m - original.root_translation_m, axis=-1
    )
    if float(np.max(root_offset)) > config.absurd_root_offset_m:
        reasons.append("absurd_root_offset")
    rotation_offset = quaternion_geodesic_distance(
        original.local_quat_wxyz, candidate.local_quat_wxyz
    )
    if float(np.max(rotation_offset)) > config.absurd_rotation_offset_rad:
        reasons.append("absurd_rotation_offset")
    if config.mode == "red_team":
        return reasons

    before = (
        motion_deterministic_metrics(original) if original_metrics is None else original_metrics
    )
    after = (
        motion_deterministic_metrics(candidate) if candidate_metrics is None else candidate_metrics
    )
    speed_error = after.get("absolute_target_speed_error_mps")
    declared_tolerance = candidate.metadata.get(
        "target_speed_tolerance", config.production_speed_error_mps
    )
    speed_tolerance = float(declared_tolerance)
    if not np.isfinite(speed_tolerance) or speed_tolerance < 0.0:
        reasons.append("invalid_target_speed_tolerance")
    elif speed_error is None or speed_error > speed_tolerance:
        reasons.append("target_speed")
    penetration = after.get("maximum_ground_penetration_cm")
    if penetration is None or penetration > config.production_penetration_cm:
        reasons.append("ground_penetration")
    if candidate.skeleton.joint_limits:
        original_limit = before.get("maximum_joint_limit_violation_deg") or 0.0
        candidate_limit = after.get("maximum_joint_limit_violation_deg")
        if (
            candidate_limit is None
            or candidate_limit > original_limit + config.production_joint_limit_slack_deg
        ):
            reasons.append("trusted_joint_limits")
    if original.metadata.get("loop_kind"):
        original_seam = before.get("loop_seam")
        candidate_seam = after.get("loop_seam")
        if (
            original_seam is None
            or candidate_seam is None
            or candidate_seam > original_seam + config.production_loop_seam_slack
        ):
            reasons.append("loop_invariant")
    original_displacement = (
        original.root_translation_m[-1, (0, 2)] - (original.root_translation_m[0, (0, 2)])
    )
    candidate_displacement = (
        candidate.root_translation_m[-1, (0, 2)] - (candidate.root_translation_m[0, (0, 2)])
    )
    original_length = float(np.linalg.norm(original_displacement))
    candidate_length = float(np.linalg.norm(candidate_displacement))
    tolerance = config.production_displacement_relative_tolerance
    if original_length > 1.0e-6 and (
        abs(candidate_length - original_length) / original_length > tolerance
        or float(np.dot(original_displacement, candidate_displacement)) <= 0.0
    ):
        reasons.append("task_displacement_invariant")
    return reasons


class _CriticObjective:
    def __init__(
        self,
        original: MotionClip,
        checkpoint_path: Path,
        dataset_directory: Path,
        config: CMAOptimizationConfig,
    ) -> None:
        self.original = original
        self.model = model_from_checkpoint(checkpoint_path)
        self.normalization = load_normalization_statistics(dataset_directory)
        self.config = config
        self.original_metrics = motion_deterministic_metrics(original)
        self.evaluations = 0
        self.rejections: dict[str, int] = {}
        self._prediction_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._metrics_cache: dict[str, dict[str, float | None]] = {
            original.content_hash: self.original_metrics
        }

    def predict(self, clip: MotionClip) -> tuple[float, dict[str, Any]]:
        cached = self._prediction_cache.get(clip.content_hash)
        if cached is not None:
            return cached
        features = encode_motion_clip(clip, self.normalization)
        if not np.all(np.isfinite(features.numpy())):
            raise ValueError("non-finite critic input")
        if float(torch.max(torch.abs(features))) > self.config.maximum_absolute_normalized_feature:
            raise ValueError("critic input exceeds declared inference bounds")
        with torch.no_grad():
            output = self.model(features[None])
        prediction = _prediction(output)
        if self.config.resolved_objective_source == "learned_severity":
            score = sum(
                weight * prediction["severity"][family]
                for family, weight in self.config.objective_weights.items()
            )
        else:
            metrics = self.deterministic_metrics(clip)
            components = []
            for family, weight in self.config.objective_weights.items():
                metric = _FAMILY_METRIC[family]
                value = metrics.get(metric)
                if value is None:
                    raise ValueError(f"deterministic target metric {metric!r} is unavailable")
                components.append(weight * float(value) / _TARGET_GAIN[metric])
            score = sum(components)
        result = (float(score), prediction)
        self._prediction_cache[clip.content_hash] = result
        return result

    def deterministic_metrics(self, clip: MotionClip) -> dict[str, float | None]:
        if clip.content_hash not in self._metrics_cache:
            self._metrics_cache[clip.content_hash] = motion_deterministic_metrics(clip)
        return self._metrics_cache[clip.content_hash]

    def rejection_reasons(self, candidate: MotionClip) -> list[str]:
        metrics = None if self.config.mode == "red_team" else self.deterministic_metrics(candidate)
        return production_constraint_reasons(
            self.original,
            candidate,
            self.config,
            original_metrics=self.original_metrics,
            candidate_metrics=metrics,
        )


def _outcome_category(
    config: CMAOptimizationConfig,
    objective_before: float,
    objective_after: float,
    before_metrics: Mapping[str, float | None],
    after_metrics: Mapping[str, float | None],
    distances: Mapping[str, float],
) -> tuple[int, str, bool]:
    improvement = objective_before - objective_after
    minimum_improvement = max(0.01, abs(objective_before) * 0.05)
    if improvement < minimum_improvement:
        return 4, "no_learned_score_improvement", False
    bizarre = (
        distances["root_translation_rms_m"] > 0.25
        or distances["local_rotation_geodesic_rms_deg"] > 25.0
        or distances["joint_world_position_rms_m"] > 0.25
    )
    target_metrics = {_FAMILY_METRIC[family] for family in config.objective_weights}
    collateral_worsening = False
    for metric, threshold in _COLLATERAL_WORSENING.items():
        before = before_metrics.get(metric)
        after = after_metrics.get(metric)
        if (
            metric not in target_metrics
            and before is not None
            and after is not None
            and float(after) - float(before) > threshold
        ):
            collateral_worsening = True
            break
    bizarre = bizarre or collateral_worsening
    if bizarre:
        return 3, "bizarre_or_implausible_solution", True
    measured_improvements: list[bool] = []
    for family in config.objective_weights:
        metric = _FAMILY_METRIC[family]
        before = before_metrics.get(metric)
        after = after_metrics.get(metric)
        measured_improvements.append(
            before is not None
            and after is not None
            and float(before) - float(after) >= _TARGET_GAIN[metric]
        )
    if measured_improvements and all(measured_improvements):
        return 1, "genuine_measured_and_learned_improvement", False
    return 2, "learned_score_exploit_without_measured_gain", True


def optimize_motion_with_cma(
    original: MotionClip,
    checkpoint_path: Path,
    dataset_directory: Path,
    output_directory: Path,
    *,
    config: CMAOptimizationConfig,
) -> OptimizationResult:
    """Run coarse-four then refined-eight BIPOP-CMA-ES and persist complete evidence."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    save_motion_npz(output / "original.npz", original)
    objective = _CriticObjective(original, checkpoint_path, dataset_directory, config)
    original_score, original_prediction = objective.predict(original)
    _json_write(output / "original_prediction.json", original_prediction)
    _json_write(output / "original_metrics.json", objective.original_metrics)
    _json_write(
        output / "run_config.json",
        {
            "format_version": OPTIMIZATION_VERSION,
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_directory),
            "config": config.model_dump(mode="json"),
            "objective_definition": (
                "weighted selected-family learned severity vector"
                if config.resolved_objective_source == "learned_severity"
                else "weighted directly measured deterministic target metrics"
            ),
            "objective_source": config.resolved_objective_source,
            "global_quality_score_used": False,
            "deterministic_metrics_in_red_team_objective": (
                config.mode == "red_team"
                and config.resolved_objective_source == "deterministic_target"
            ),
            "learned_hard_defect_heads_in_production_objective": False,
        },
    )

    global_generation = 0
    history: list[dict[str, Any]] = []
    best_clip = original
    best_coefficients: NDArray[np.float64] | None = None
    best_score = original_score
    best_prediction: dict[str, Any] = original_prediction

    def run_stage(
        name: str,
        layout: SplineResidualLayout,
        initial: NDArray[np.float64],
        maximum_iterations: int,
        stage_seed: int,
    ) -> tuple[NDArray[np.float64], MotionClip, float, dict[str, Any]]:
        nonlocal global_generation, best_clip, best_coefficients, best_score, best_prediction
        stage_best_coefficients = initial.copy()
        stage_best_clip = apply_spline_residual(
            original,
            layout,
            stage_best_coefficients,
            rotation_scale_rad=config.rotation_scale_rad,
            root_translation_scale_m=config.root_translation_scale_m,
            root_yaw_scale_rad=config.root_yaw_scale_rad,
        )
        stage_best_score, stage_best_prediction = objective.predict(stage_best_clip)
        if objective.rejection_reasons(stage_best_clip):
            stage_best_score = float("inf")

        def evaluate(values: ArrayLike) -> float:
            nonlocal stage_best_coefficients, stage_best_clip
            nonlocal stage_best_score, stage_best_prediction
            objective.evaluations += 1
            try:
                candidate = apply_spline_residual(
                    original,
                    layout,
                    values,
                    rotation_scale_rad=config.rotation_scale_rad,
                    root_translation_scale_m=config.root_translation_scale_m,
                    root_yaw_scale_rad=config.root_yaw_scale_rad,
                )
                reasons = objective.rejection_reasons(candidate)
                if reasons:
                    for reason in reasons:
                        objective.rejections[reason] = objective.rejections.get(reason, 0) + 1
                    return 1.0e6 + len(reasons)
                score, prediction = objective.predict(candidate)
            except (ValueError, FloatingPointError) as exc:
                reason = f"invalid:{type(exc).__name__}"
                objective.rejections[reason] = objective.rejections.get(reason, 0) + 1
                return 1.0e6
            if score < stage_best_score:
                stage_best_coefficients = np.asarray(values, dtype=np.float64).copy()
                stage_best_clip = candidate
                stage_best_score = score
                stage_best_prediction = prediction
            return score

        def callback(strategy: Any) -> None:
            nonlocal global_generation, best_clip, best_coefficients, best_score, best_prediction
            global_generation += 1
            generation_directory = output / "best_by_generation" / f"{global_generation:04d}"
            generation_directory.mkdir(parents=True, exist_ok=True)
            save_motion_npz(generation_directory / "motion.npz", stage_best_clip)
            np.savez_compressed(
                generation_directory / "optimizer_state.npz",
                mean=np.asarray(strategy.mean, dtype=np.float64),
                sigma=np.asarray(float(strategy.sigma), dtype=np.float64),
                covariance=np.asarray(strategy.sm.C, dtype=np.float64),
                best_coefficients=stage_best_coefficients,
            )
            metrics = objective.deterministic_metrics(stage_best_clip)
            generation_record = {
                "generation": global_generation,
                "stage": name,
                "stage_iteration": int(strategy.countiter),
                "evaluations": int(strategy.countevals),
                "best_objective": (stage_best_score if np.isfinite(stage_best_score) else None),
                "feasible_best_found": bool(np.isfinite(stage_best_score)),
                "prediction": stage_best_prediction,
                "deterministic_metrics": metrics,
                "distance": motion_distances(original, stage_best_clip),
                "coefficient_path": str(generation_directory / "optimizer_state.npz"),
                "motion_path": str(generation_directory / "motion.npz"),
            }
            history.append(generation_record)
            _json_write(generation_directory / "record.json", generation_record)
            if stage_best_score < best_score:
                best_score = stage_best_score
                best_clip = stage_best_clip
                best_coefficients = stage_best_coefficients.copy()
                best_prediction = stage_best_prediction

        options: dict[str, Any] = {
            "bounds": [-config.coefficient_bound, config.coefficient_bound],
            "maxiter": maximum_iterations,
            "seed": stage_seed,
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
        }
        if config.population_size is not None:
            options["popsize"] = config.population_size
        cma.fmin2(
            evaluate,
            initial,
            config.initial_sigma,
            options,
            restarts=config.bipop_restarts,
            bipop=True,
            eval_initial_x=True,
            callback=callback,
        )
        return (
            stage_best_coefficients,
            stage_best_clip,
            stage_best_score,
            stage_best_prediction,
        )

    coarse_layout = SplineResidualLayout.for_skeleton(
        original.skeleton,
        control_points=config.coarse_control_points,
        optimized_roles=config.optimized_roles,
    )
    coarse_initial = np.zeros(coarse_layout.parameter_count, dtype=np.float64)
    coarse_values, _, _, _ = run_stage(
        "coarse_4",
        coarse_layout,
        coarse_initial,
        config.coarse_max_iterations,
        config.seed,
    )
    refined_layout = SplineResidualLayout.for_skeleton(
        original.skeleton,
        control_points=config.refined_control_points,
        optimized_roles=config.optimized_roles,
    )
    refined_initial = refine_coefficients(coarse_values, coarse_layout, refined_layout)
    refined_values, refined_clip, refined_score, refined_prediction = run_stage(
        "refined_8",
        refined_layout,
        refined_initial,
        config.refined_max_iterations,
        config.seed + 1,
    )
    if np.isfinite(refined_score) and refined_score <= best_score:
        best_clip = refined_clip
        best_score = refined_score
        best_coefficients = refined_values
        best_prediction = refined_prediction
    if best_coefficients is None:
        best_coefficients = refined_initial

    final_path = output / "final.npz"
    save_motion_npz(final_path, best_clip)
    np.savez_compressed(
        output / "final_spline_coefficients.npz",
        coefficients=best_coefficients,
        channel_names=np.asarray(refined_layout.channel_names, dtype=np.str_),
        control_points=np.asarray(refined_layout.control_points, dtype=np.int64),
    )
    save_motion_comparison_strip(original, best_clip, output / "preview.png")
    final_metrics = objective.deterministic_metrics(best_clip)
    distances = motion_distances(original, best_clip)
    final_rejections = objective.rejection_reasons(best_clip)
    production_feasible = config.mode != "production" or not final_rejections
    if production_feasible:
        category, category_label, adversarial = _outcome_category(
            config,
            original_score,
            best_score,
            objective.original_metrics,
            final_metrics,
            distances,
        )
    else:
        category, category_label, adversarial = (
            4,
            "no_feasible_production_improvement",
            False,
        )
    report = {
        "format_version": OPTIMIZATION_VERSION,
        "mode": config.mode,
        "objective_weights": config.objective_weights,
        "objective_before": original_score,
        "objective_after": best_score,
        "objective_improvement": original_score - best_score,
        "prediction_before": original_prediction,
        "prediction_after": best_prediction,
        "deterministic_metrics_before": objective.original_metrics,
        "deterministic_metrics_after": final_metrics,
        "distance": distances,
        "category": category,
        "category_label": category_label,
        "adversarial_training_eligible": adversarial,
        "evaluation_count": objective.evaluations,
        "rejection_counts": objective.rejections,
        "final_constraint_rejections": final_rejections,
        "production_feasible": production_feasible,
        "bipop": True,
        "coarse_control_points": coarse_layout.control_points,
        "refined_control_points": refined_layout.control_points,
        "optimized_joint_roles": list(refined_layout.joint_roles),
        "global_quality_score_used": False,
        "objective_source": config.resolved_objective_source,
        "deterministic_metrics_used_in_objective": (
            config.resolved_objective_source == "deterministic_target"
        ),
        "learned_hard_defect_heads_in_production_objective": False,
        "history": history,
        "artifacts": {
            "original": str(output / "original.npz"),
            "final": str(final_path),
            "coefficients": str(output / "final_spline_coefficients.npz"),
            "preview": str(output / "preview.png"),
        },
    }
    report_path = output / "optimization.json"
    _json_write(report_path, report)
    return OptimizationResult(
        output,
        final_path,
        report_path,
        original_score,
        best_score,
        category,
        category_label,
        adversarial,
    )
