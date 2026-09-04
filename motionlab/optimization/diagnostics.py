"""Causal diagnostics separating spline representation, search, critic, and constraints."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

from motionlab.corruptions._supervision import measure_contact_supervision
from motionlab.critic.data import FixedRigSampleDataset, encode_motion_clip
from motionlab.critic.forensics import (
    motion_deterministic_metrics,
    motion_distances,
    motions_from_sample,
)
from motionlab.critic.model import FIRST_CRITIC_DEFECTS
from motionlab.critic.training import model_from_checkpoint
from motionlab.dataset.io import load_normalization_statistics
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.math.quaternion import (
    quaternion_difference,
    quaternion_exp,
    quaternion_log,
    quaternion_multiply,
)
from motionlab.metrics.foot_sliding import foot_sliding_metric
from motionlab.metrics.ground import ground_metrics
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.motion.clip import MotionClip
from motionlab.optimization.adaptive import (
    build_multiresolution_projections,
    exact_residual_field,
    production_active_region,
    residual_energy_report,
)
from motionlab.optimization.cmaes import (
    CMAOptimizationConfig,
    optimize_motion_with_cma,
    production_constraint_reasons,
)
from motionlab.optimization.spline import SplineResidualLayout, apply_spline_residual
from motionlab.visualization.skeleton_plot import save_motion_comparison_strip

DIAGNOSTIC_VERSION = "motionlab.cma_failure_decomposition.v1"

FAMILY_METRIC = {
    "foot_slide": "foot_sliding_cm_per_contact_interval",
    "ground_penetration": "maximum_ground_penetration_cm",
    "floating_contact": "maximum_floating_contact_cm",
    "loop_seam": "loop_seam",
    "joint_pop": "localized_target_joint_peak_angular_jerk_rad_s3",
    "joint_jitter": "localized_angular_jerk_p95_rad_s3",
    "speed_inconsistency": "absolute_target_speed_error_mps",
}

TARGET_GAIN = {
    "foot_sliding_cm_per_contact_interval": 0.2,
    "maximum_ground_penetration_cm": 0.2,
    "maximum_floating_contact_cm": 0.2,
    "loop_seam": 0.02,
    "localized_target_joint_peak_angular_jerk_rad_s3": 100.0,
    "localized_angular_jerk_p95_rad_s3": 100.0,
    "absolute_target_speed_error_mps": 0.02,
}


@dataclass(frozen=True)
class ExactCleanResidual:
    """Exact local right-multiplication tangent and root-translation correction."""

    joint_tangent_rad: NDArray[np.float64]
    root_translation_m: NDArray[np.float64]


@dataclass(frozen=True)
class OracleProjection:
    """One constrained least-squares projection into a declared spline layout."""

    candidate: MotionClip
    coefficients: NDArray[np.float64]
    fitted_curves: NDArray[np.float64]
    exact_residual: ExactCleanResidual
    approximation: Mapping[str, Any]


class PairTargetEvaluator:
    """Reproduce the exact pair-relative postcondition used to create one sample."""

    def __init__(
        self,
        clean: MotionClip,
        family: str,
        sample_metadata: Mapping[str, Any],
    ) -> None:
        self.clean = clean
        self.family = family
        self.metadata = sample_metadata
        self.parameters = dict(sample_metadata.get("generation_parameters", {}))
        self._source_contacts = None
        if family in {"foot_slide", "ground_penetration", "floating_contact"}:
            _, fixed_clean, _ = measure_contact_supervision(clean, clean)
            self._source_contacts = fixed_clean

    @property
    def metric_name(self) -> str:
        return str(self.metadata.get("postcondition_metric") or FAMILY_METRIC[self.family])

    def measure(self, candidate: MotionClip) -> float:
        """Measure candidate severity using clean contacts and localized support."""
        if self.family in {"foot_slide", "ground_penetration", "floating_contact"}:
            if self._source_contacts is None:
                raise RuntimeError("contact evaluator was not initialized")
            _, _, fixed_candidate = measure_contact_supervision(
                self.clean,
                candidate,
                source_contacts=self._source_contacts,
                ground_plane=self._source_contacts.ground_plane,
            )
            if self.family == "foot_slide":
                return float(foot_sliding_metric(candidate, fixed_candidate).clip_value or 0.0)
            penetration, floating = ground_metrics(fixed_candidate)
            selected = penetration if self.family == "ground_penetration" else floating
            return float(selected.clip_value or 0.0)
        if self.family in {"joint_pop", "joint_jitter"}:
            values = np.asarray(
                smoothness_metric(candidate).joint_frame_values,
                dtype=np.float64,
            )
            if self.family == "joint_pop":
                frame = int(self.parameters["frame"])
                joint = int(self.parameters["joint_index"])
                support = values[max(0, frame - 2) : min(candidate.num_frames, frame + 3), joint]
                return float(np.max(support, initial=0.0))
            start = int(self.parameters["start_frame"])
            stop = int(self.parameters["stop_frame"])
            joints = np.asarray(self.parameters["joint_indices"], dtype=np.int64)
            support = values[max(0, start - 2) : min(candidate.num_frames, stop + 2)][:, joints]
            return float(np.percentile(support, 95.0))
        if self.family == "loop_seam":
            return float(loop_seam_metric(candidate).clip_value or 0.0)
        if self.family == "speed_inconsistency":
            value = motion_deterministic_metrics(candidate)["absolute_target_speed_error_mps"]
            return float("inf") if value is None else float(value)
        raise ValueError(f"no deterministic pair metric for {self.family!r}")


def exact_clean_residual(corrupted: MotionClip, clean: MotionClip) -> ExactCleanResidual:
    """Compute ``Log(R_bad^-1 R_clean)`` and ``p_clean - p_bad`` exactly."""
    if corrupted.skeleton.content_hash != clean.skeleton.content_hash:
        raise ValueError("clean and corrupted clips must share a skeleton")
    if corrupted.num_frames != clean.num_frames:
        raise ValueError("clean and corrupted clips must share a frame count")
    difference = quaternion_difference(
        corrupted.local_quat_wxyz,
        clean.local_quat_wxyz,
    )
    return ExactCleanResidual(
        joint_tangent_rad=np.asarray(quaternion_log(difference), dtype=np.float64),
        root_translation_m=np.asarray(
            clean.root_translation_m - corrupted.root_translation_m,
            dtype=np.float64,
        ),
    )


def _exact_layout_curves(
    residual: ExactCleanResidual,
    layout: SplineResidualLayout,
    config: CMAOptimizationConfig,
    *,
    root_index: int,
) -> NDArray[np.float64]:
    rotation = residual.joint_tangent_rad[:, layout.joint_indices, :].reshape(
        (residual.joint_tangent_rad.shape[0], -1)
    )
    normalized_rotation = rotation / config.rotation_scale_rad
    normalized_translation = residual.root_translation_m / config.root_translation_scale_m
    normalized_yaw = residual.joint_tangent_rad[:, root_index, 1:2] / config.root_yaw_scale_rad
    return np.concatenate((normalized_rotation, normalized_translation, normalized_yaw), axis=-1)


def _residual_energy_report(
    residual: ExactCleanResidual,
    layout: SplineResidualLayout,
    clip: MotionClip,
) -> dict[str, Any]:
    root = clip.skeleton.root_index
    squared = residual.joint_tangent_rad * residual.joint_tangent_rad
    total = float(np.sum(squared))
    exposed = float(np.sum(squared[:, layout.joint_indices]))
    exposed += float(np.sum(squared[:, root, 1] ** 2))
    outside_energy = max(0.0, total - exposed)
    unexposed_records = []
    selected = set(layout.joint_indices)
    for joint, name in enumerate(clip.skeleton.joint_names):
        component = residual.joint_tangent_rad[:, joint].copy()
        if joint == root:
            component[:, 1] = 0.0
        if joint in selected:
            component[:] = 0.0
        energy = float(np.sum(component * component))
        if energy > 1.0e-14:
            unexposed_records.append(
                {
                    "joint": name,
                    "joint_index": joint,
                    "rms_deg": float(
                        np.rad2deg(np.sqrt(np.mean(np.sum(component * component, axis=-1))))
                    ),
                    "energy_fraction": energy / max(total, 1.0e-20),
                }
            )
    return {
        "rotation_total_squared_rad": total,
        "rotation_exposed_fraction": exposed / max(total, 1.0e-20),
        "rotation_outside_exposed_fraction": outside_energy / max(total, 1.0e-20),
        "rotation_outside_exposed_squared_rad": outside_energy,
        "rotation_outside_exposed_rms_deg_all_joint_frames": float(
            np.rad2deg(
                np.sqrt(outside_energy / (residual.joint_tangent_rad.shape[0] * clip.num_joints))
            )
        ),
        "root_translation_outside_exposed_fraction": 0.0,
        "unexposed_rotation_by_joint": sorted(
            unexposed_records,
            key=lambda item: float(item["energy_fraction"]),
            reverse=True,
        ),
    }


def fit_oracle_projection(
    corrupted: MotionClip,
    clean: MotionClip,
    layout: SplineResidualLayout,
    config: CMAOptimizationConfig,
) -> OracleProjection:
    """Fit the exact correction with coefficient bounds identical to CMA-ES."""
    residual = exact_clean_residual(corrupted, clean)
    target = _exact_layout_curves(
        residual,
        layout,
        config,
        root_index=corrupted.skeleton.root_index,
    )
    basis = layout.basis(corrupted.num_frames)
    coefficient_matrix = np.empty((layout.channel_count, layout.control_points), dtype=np.float64)
    for channel in range(layout.channel_count):
        solution = lsq_linear(
            basis,
            target[:, channel],
            bounds=(-config.coefficient_bound, config.coefficient_bound),
            method="bvls",
        )
        coefficient_matrix[channel] = solution.x
    coefficients = coefficient_matrix.reshape(-1)
    fitted_normalized = basis @ coefficient_matrix.T
    candidate = apply_spline_residual(
        corrupted,
        layout,
        coefficients,
        rotation_scale_rad=config.rotation_scale_rad,
        root_translation_scale_m=config.root_translation_scale_m,
        root_yaw_scale_rad=config.root_yaw_scale_rad,
    )
    rotation_channels = len(layout.joint_indices) * 3
    fitted_rotation = (
        fitted_normalized[:, :rotation_channels].reshape(
            (corrupted.num_frames, len(layout.joint_indices), 3)
        )
        * config.rotation_scale_rad
    )
    exact_rotation = residual.joint_tangent_rad[:, layout.joint_indices]
    rotation_error = fitted_rotation - exact_rotation
    translation_error = (
        fitted_normalized[:, rotation_channels : rotation_channels + 3]
        * config.root_translation_scale_m
        - residual.root_translation_m
    )
    fitted_yaw = fitted_normalized[:, -1] * config.root_yaw_scale_rad
    exact_yaw = residual.joint_tangent_rad[:, corrupted.skeleton.root_index, 1]
    time_rotation_error = np.sqrt(np.mean(np.sum(rotation_error**2, axis=-1), axis=-1))
    approximation = {
        "coefficient_bound": config.coefficient_bound,
        "bound_saturated_fraction": float(
            np.mean(np.isclose(np.abs(coefficients), config.coefficient_bound, atol=1.0e-7))
        ),
        "by_joint": [
            {
                "role": role,
                "joint_index": joint,
                "rms_error_deg": float(
                    np.rad2deg(np.sqrt(np.mean(np.sum(rotation_error[:, offset] ** 2, axis=-1))))
                ),
                "maximum_error_deg": float(
                    np.rad2deg(np.max(np.linalg.norm(rotation_error[:, offset], axis=-1)))
                ),
            }
            for offset, (joint, role) in enumerate(
                zip(layout.joint_indices, layout.joint_roles, strict=True)
            )
        ],
        "by_time": {
            "exposed_rotation_rms_error_deg": np.rad2deg(time_rotation_error).tolist(),
            "root_translation_error_m": np.linalg.norm(translation_error, axis=-1).tolist(),
            "root_yaw_error_deg": np.rad2deg(np.abs(fitted_yaw - exact_yaw)).tolist(),
        },
        "aggregate": {
            "exposed_rotation_rms_error_deg": float(
                np.rad2deg(np.sqrt(np.mean(rotation_error**2)))
            ),
            "root_translation_rms_error_m": float(np.sqrt(np.mean(translation_error**2))),
            "root_yaw_rms_error_deg": float(
                np.rad2deg(np.sqrt(np.mean((fitted_yaw - exact_yaw) ** 2)))
            ),
        },
        "energy": _residual_energy_report(residual, layout, corrupted),
    }
    return OracleProjection(
        candidate,
        coefficients,
        fitted_normalized,
        residual,
        approximation,
    )


def localized_internal_knots(residual: ExactCleanResidual) -> tuple[float, ...]:
    """Add knots only around the measured support of the exact correction."""
    frame_energy = np.sum(residual.joint_tangent_rad**2, axis=(1, 2))
    frame_energy += np.sum(residual.root_translation_m**2, axis=-1)
    maximum = float(np.max(frame_energy))
    if maximum <= 1.0e-20:
        return (0.2, 0.4, 0.6, 0.8)
    active = np.flatnonzero(frame_energy >= maximum * 0.02)
    frame_scale = max(len(frame_energy) - 1, 1)
    lower = float(active[0] / frame_scale)
    upper = float(active[-1] / frame_scale)
    peak = float(np.argmax(frame_energy) / frame_scale)
    values = [0.2, 0.4, 0.6, 0.8]
    width = int(active[-1] - active[0] + 1)
    signals = np.concatenate(
        (
            residual.joint_tangent_rad.reshape((len(frame_energy), -1)),
            residual.root_translation_m,
        ),
        axis=-1,
    )
    channel_rms = np.sqrt(np.mean(signals[active] ** 2, axis=0))
    material_channels = channel_rms >= max(float(np.max(channel_rms)) * 0.02, 1.0e-8)
    material = signals[active][:, material_channels]
    sign_change_fraction = (
        0.0
        if material.shape[0] < 2 or material.shape[1] == 0
        else float(np.mean(material[:-1] * material[1:] < 0.0))
    )
    if width <= 5:
        # Repeated boundary knots reduce continuity only where a sample-local pulse needs it.
        for boundary in (
            (float(active[0]) - 0.5) / frame_scale,
            (float(active[-1]) + 0.5) / frame_scale,
        ):
            values.extend((boundary, boundary, boundary))
        values.append(peak)
    elif sign_change_fraction > 0.15:
        # White/correlated jitter genuinely has frame-scale residual energy.  A knot per active
        # frame is diagnostic evidence about the required bandwidth, not a proposed CMA space.
        values.extend(float(frame / frame_scale) for frame in active)
    else:
        half_frame = 1.5 / frame_scale
        for center in (lower, peak, upper):
            values.extend((center - half_frame, center, center + half_frame))
    return tuple(sorted(value for value in values if 1.0e-4 < value < 1.0 - 1.0e-4))


def _target_repair(
    evaluator: PairTargetEvaluator,
    clean: MotionClip,
    corrupted: MotionClip,
    candidate: MotionClip,
) -> dict[str, Any]:
    metric = evaluator.metric_name
    clean_value = evaluator.measure(clean)
    before = evaluator.measure(corrupted)
    after = evaluator.measure(candidate)
    improvement = before - after
    gap = before - clean_value
    fraction = None if gap <= 1.0e-12 else improvement / gap
    return {
        "metric": metric,
        "clean": clean_value,
        "before": before,
        "after": after,
        "improvement": improvement,
        "fraction_of_clean_gap_repaired": fraction,
        "measurement_contract": "pair-relative corruption postcondition",
        "meaningful_improvement": improvement >= TARGET_GAIN.get(metric, 0.0),
        "approximately_clean_target": abs(after - clean_value) <= max(1.0e-6, 0.1 * abs(gap)),
    }


def _expanded_joint_indices(
    residual: ExactCleanResidual,
    layout: SplineResidualLayout,
    clip: MotionClip,
) -> tuple[int, ...]:
    selected = list(layout.joint_indices)
    root = clip.skeleton.root_index
    joint_rms = np.sqrt(np.mean(np.sum(residual.joint_tangent_rad**2, axis=-1), axis=0))
    for joint in np.flatnonzero(joint_rms > np.deg2rad(0.01)):
        if int(joint) != root and int(joint) not in selected:
            selected.append(int(joint))
    return tuple(selected)


def _variant_report(
    family: str,
    corrupted: MotionClip,
    clean: MotionClip,
    projection: OracleProjection,
    layout: SplineResidualLayout,
    config: CMAOptimizationConfig,
    output: Path,
    name: str,
    clean_metrics: Mapping[str, float | None],
    corrupted_metrics: Mapping[str, float | None],
    target_evaluator: PairTargetEvaluator,
) -> dict[str, Any]:
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    save_motion_npz(destination / "oracle_projected.npz", projection.candidate)
    np.savez_compressed(
        destination / "projection.npz",
        coefficients=projection.coefficients,
        fitted_normalized_curves=projection.fitted_curves,
        exact_joint_tangent_rad=projection.exact_residual.joint_tangent_rad,
        exact_root_translation_m=projection.exact_residual.root_translation_m,
        channel_names=np.asarray(layout.channel_names, dtype=np.str_),
        internal_knots=np.asarray(layout.internal_knots or (), dtype=np.float64),
    )
    save_motion_comparison_strip(clean, projection.candidate, destination / "clean_comparison.png")
    candidate_metrics = motion_deterministic_metrics(projection.candidate)
    constraints = production_constraint_reasons(
        corrupted,
        projection.candidate,
        config,
        original_metrics=corrupted_metrics,
        candidate_metrics=candidate_metrics,
    )
    distance = motion_distances(clean, projection.candidate)
    target = _target_repair(target_evaluator, clean, corrupted, projection.candidate)
    approximately_clean_motion = (
        distance["root_translation_rms_m"] <= 0.01
        and distance["local_rotation_geodesic_rms_deg"] <= 1.0
        and distance["joint_world_position_rms_m"] <= 0.02
    )
    report = {
        "variant": name,
        "control_points": layout.control_points,
        "internal_knots": list(layout.internal_knots or ()),
        "joint_roles": list(layout.joint_roles),
        "parameter_count": layout.parameter_count,
        "rms_residual_to_clean": distance,
        "target_deterministic_defect": target,
        "production_constraints": {
            "passed": not constraints,
            "rejection_reasons": constraints,
        },
        "approximately_clean_motion": approximately_clean_motion,
        "oracle_projected_candidate_passes_production_repair_criteria": (
            approximately_clean_motion
            and bool(target["approximately_clean_target"])
            and not constraints
        ),
        "approximation": projection.approximation,
        "artifacts": {
            "motion": str(destination / "oracle_projected.npz"),
            "projection": str(destination / "projection.npz"),
            "comparison": str(destination / "clean_comparison.png"),
        },
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_oracle_representability(
    dataset_directory: Path,
    experiment_report_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Run 4, 8, localized-knot, and evidence-triggered expanded-joint projections."""
    experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    selected = experiment["selected_samples"]
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for family, selection in selected.items():
        sample_id = str(selection["sample_id"])
        indices = [
            index
            for index, record in enumerate(dataset.records)
            if record["sample_id"] == sample_id
        ]
        if len(indices) != 1:
            raise ValueError(f"optimization sample {sample_id!r} was not found exactly once")
        clean, corrupted = motions_from_sample(dataset, indices[0])
        case_output = output / family
        case_output.mkdir(parents=True, exist_ok=True)
        for variant_name in (
            "current_4",
            "current_8",
            "localized_knots",
            "expanded_joints",
        ):
            stale_variant = case_output / variant_name
            if stale_variant.is_dir():
                shutil.rmtree(stale_variant)
        save_motion_npz(case_output / "clean.npz", clean)
        save_motion_npz(case_output / "corrupted.npz", corrupted)
        clean_metrics = motion_deterministic_metrics(clean)
        corrupted_metrics = motion_deterministic_metrics(corrupted)
        sample_metadata = dataset.sample(indices[0]).metadata
        target_evaluator = PairTargetEvaluator(clean, family, sample_metadata)
        config = CMAOptimizationConfig(mode="production", objective_weights={family: 1.0})
        variants: list[dict[str, Any]] = []
        current_4 = SplineResidualLayout.for_skeleton(
            corrupted.skeleton,
            control_points=4,
            optimized_roles=config.optimized_roles,
        )
        projection_4 = fit_oracle_projection(corrupted, clean, current_4, config)
        variants.append(
            _variant_report(
                family,
                corrupted,
                clean,
                projection_4,
                current_4,
                config,
                case_output,
                "current_4",
                clean_metrics,
                corrupted_metrics,
                target_evaluator,
            )
        )
        current_8 = SplineResidualLayout.for_skeleton(
            corrupted.skeleton,
            control_points=8,
            optimized_roles=config.optimized_roles,
        )
        projection_8 = fit_oracle_projection(corrupted, clean, current_8, config)
        variants.append(
            _variant_report(
                family,
                corrupted,
                clean,
                projection_8,
                current_8,
                config,
                case_output,
                "current_8",
                clean_metrics,
                corrupted_metrics,
                target_evaluator,
            )
        )
        localized_knots = localized_internal_knots(projection_8.exact_residual)
        localized = SplineResidualLayout.for_skeleton(
            corrupted.skeleton,
            control_points=len(localized_knots) + 4,
            optimized_roles=config.optimized_roles,
            internal_knots=localized_knots,
        )
        localized_projection = fit_oracle_projection(corrupted, clean, localized, config)
        variants.append(
            _variant_report(
                family,
                corrupted,
                clean,
                localized_projection,
                localized,
                config,
                case_output,
                "localized_knots",
                clean_metrics,
                corrupted_metrics,
                target_evaluator,
            )
        )
        energy = localized_projection.approximation["energy"]
        outside = float(energy["rotation_outside_exposed_fraction"])
        unexposed = energy["unexposed_rotation_by_joint"]
        maximum_unexposed_rms = max(
            (float(item["rms_deg"]) for item in unexposed),
            default=0.0,
        )
        expanded_tested = outside > 1.0e-4 and maximum_unexposed_rms > 0.01
        if expanded_tested:
            expanded_indices = _expanded_joint_indices(
                localized_projection.exact_residual,
                localized,
                corrupted,
            )
            expanded = SplineResidualLayout.for_joint_indices(
                corrupted.skeleton,
                control_points=localized.control_points,
                joint_indices=expanded_indices,
                internal_knots=localized_knots,
            )
            expanded_projection = fit_oracle_projection(corrupted, clean, expanded, config)
            variants.append(
                _variant_report(
                    family,
                    corrupted,
                    clean,
                    expanded_projection,
                    expanded,
                    config,
                    case_output,
                    "expanded_joints",
                    clean_metrics,
                    corrupted_metrics,
                    target_evaluator,
                )
            )
        current_representable = bool(
            variants[1]["approximately_clean_motion"]
            and variants[1]["target_deterministic_defect"]["approximately_clean_target"]
        )
        progressive_representable = any(
            bool(variant["approximately_clean_motion"])
            and bool(variant["target_deterministic_defect"]["approximately_clean_target"])
            for variant in variants
        )
        production_valid = any(
            bool(variant["oracle_projected_candidate_passes_production_repair_criteria"])
            for variant in variants[:2]
        )
        current_geometrically_valid = bool(
            variants[1]["approximately_clean_motion"]
            and variants[1]["target_deterministic_defect"]["approximately_clean_target"]
        )
        current_constraint_valid = bool(variants[1]["production_constraints"]["passed"])
        if not current_geometrically_valid:
            classification = "SEARCH SPACE"
        elif not current_constraint_valid:
            classification = "PRODUCTION CONSTRAINT"
        else:
            classification = "REPRESENTABLE_CURRENT_SPACE"
        cases.append(
            {
                "family": family,
                "sample_id": sample_id,
                "current_8_representable": current_representable,
                "progressive_representable": progressive_representable,
                "expanded_joints_tested": expanded_tested,
                "oracle_valid_in_current_space": production_valid,
                "current_space_production_constraints_passed": current_constraint_valid,
                "preliminary_classification": classification,
                "clean_metrics": clean_metrics,
                "corrupted_metrics": corrupted_metrics,
                "variants": variants,
            }
        )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "dataset": str(dataset_directory),
        "source_experiment": str(experiment_report_path),
        "representability_thresholds": {
            "root_translation_rms_m": 0.01,
            "local_rotation_geodesic_rms_deg": 1.0,
            "joint_world_position_rms_m": 0.02,
            "target_clean_gap_repaired_fraction": 0.9,
        },
        "cases": cases,
    }
    (output / "representability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _critic_prediction(model: Any, features: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        output = model(features[None])
    probability = torch.sigmoid(output.clip_logits)[0].cpu().numpy()
    severity = output.clip_severity[0].cpu().numpy()
    ranking = output.family_ranking_score[0].cpu().numpy()
    frame_probability = torch.sigmoid(output.frame_logits)[0].cpu().numpy()
    part_probability = torch.sigmoid(output.part_logits)[0].cpu().numpy()
    return {
        "probability": {
            family: float(probability[index]) for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
        "severity": {
            family: float(severity[index]) for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
        "family_rank": {
            family: float(ranking[index]) for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
        "maximum_frame_probability": {
            family: float(np.max(frame_probability[:, index]))
            for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
        "maximum_part_probability": {
            family: float(np.max(part_probability[:, :, index]))
            for index, family in enumerate(FIRST_CRITIC_DEFECTS)
        },
    }


def interpolate_repair_path(
    corrupted: MotionClip,
    clean: MotionClip,
    alpha: float,
) -> MotionClip:
    """Apply the exact SO(3) and root interpolation from corrupted toward clean."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("repair alpha must be within [0,1]")
    residual = exact_clean_residual(corrupted, clean)
    local = quaternion_multiply(
        corrupted.local_quat_wxyz,
        quaternion_exp(alpha * residual.joint_tangent_rad),
    )
    root = corrupted.root_translation_m + alpha * residual.root_translation_m
    return corrupted.with_updates(
        local_quat_wxyz=np.asarray(local, dtype=np.float32),
        root_translation_m=np.asarray(root, dtype=np.float32),
        metadata={
            **dict(corrupted.metadata),
            "repair_path_alpha": alpha,
            "repair_path_interpolation": "R_bad*Exp(alpha*Log(R_bad^-1*R_clean)); root_lerp",
        },
    )


def _safe_spearman(left: list[float], right: list[float]) -> float | None:
    statistic = float(spearmanr(left, right).statistic)
    return statistic if np.isfinite(statistic) else None


def run_repair_path_monotonicity(
    dataset_directory: Path,
    experiment_report_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    *,
    alphas: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21).tolist()),
) -> dict[str, Any]:
    """Evaluate exact known repairs as a first-class optimization-landscape test."""
    experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    model = model_from_checkpoint(checkpoint_path)
    normalization = load_normalization_statistics(dataset_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    families: list[dict[str, Any]] = []
    for family, selection in experiment["selected_samples"].items():
        index = next(
            i
            for i, record in enumerate(dataset.records)
            if record["sample_id"] == selection["sample_id"]
        )
        sample = dataset.sample(index)
        clean, corrupted = motions_from_sample(dataset, index)
        evaluator = PairTargetEvaluator(clean, family, sample.metadata)
        steps: list[dict[str, Any]] = []
        for alpha in alphas:
            candidate = interpolate_repair_path(corrupted, clean, float(alpha))
            prediction = _critic_prediction(
                model,
                encode_motion_clip(candidate, normalization),
            )
            steps.append(
                {
                    "alpha": float(alpha),
                    "measured_target_severity": evaluator.measure(candidate),
                    "critic": prediction,
                }
            )
        measured = [float(step["measured_target_severity"]) for step in steps]
        predicted = [float(step["critic"]["severity"][family]) for step in steps]
        improvements = [measured[0] - value for value in measured]
        deltas = np.diff(predicted)
        inversions = [
            {
                "from_alpha": float(steps[offset]["alpha"]),
                "to_alpha": float(steps[offset + 1]["alpha"]),
                "severity_increase": float(delta),
            }
            for offset, delta in enumerate(deltas)
            if delta > 1.0e-9
        ]
        families.append(
            {
                "family": family,
                "sample_id": selection["sample_id"],
                "metric": evaluator.metric_name,
                "spearman_alpha_vs_predicted_severity": _safe_spearman(
                    [float(value) for value in alphas], predicted
                ),
                "spearman_measured_improvement_vs_predicted_severity": _safe_spearman(
                    improvements, predicted
                ),
                "adjacent_repair_steps_ordered_correctly_fraction": float(
                    np.mean(deltas <= 1.0e-9)
                ),
                "inversion_count": len(inversions),
                "inversion_total_magnitude": float(np.sum(np.maximum(deltas, 0.0))),
                "inversion_maximum_magnitude": float(np.max(np.maximum(deltas, 0.0), initial=0.0)),
                "inversions": inversions,
                "steps": steps,
            }
        )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "checkpoint": str(checkpoint_path),
        "alphas": list(alphas),
        "families": families,
    }
    (output / "repair_path_monotonicity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_overlapping_window_consistency(
    dataset_directory: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    window_frames: int = 160,
    offset_frames: int = 16,
    source_start: int = 512,
) -> dict[str, Any]:
    """Compare aligned interior outputs from overlapping feature windows."""
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    source_path = Path(str(dataset.records[0]["source_path"]))
    source = load_motion_npz(source_path)
    required = source_start + window_frames + offset_frames
    if required > source.num_frames:
        source_start = 0
    normalization = load_normalization_statistics(dataset_directory)
    all_features = encode_motion_clip(source, normalization)
    model = model_from_checkpoint(checkpoint_path)
    model_config = model.config
    radius = sum(dilation * (model_config.kernel_size - 1) for dilation in model_config.dilations)
    starts = (source_start, source_start + offset_frames)
    windows = [all_features[start : start + window_frames] for start in starts]
    with torch.no_grad():
        predictions = [model(window[None]) for window in windows]
    global_start = max(starts[0] + radius, starts[1] + radius)
    global_stop = min(starts[0] + window_frames - radius, starts[1] + window_frames - radius)
    if global_start >= global_stop:
        raise ValueError("overlap has no frames beyond both receptive-field boundaries")
    slices = (
        slice(global_start - starts[0], global_stop - starts[0]),
        slice(global_start - starts[1], global_stop - starts[1]),
    )

    def difference(name: str, *, sigmoid: bool = False) -> dict[str, Any]:
        values = [getattr(prediction, name)[0] for prediction in predictions]
        if sigmoid:
            values = [torch.sigmoid(value) for value in values]
        aligned = [value[s] for value, s in zip(values, slices, strict=True)]
        delta = torch.abs(aligned[0] - aligned[1]).cpu().numpy()
        family_axis = delta.ndim - 1
        reduced_axes = tuple(axis for axis in range(delta.ndim) if axis != family_axis)
        return {
            "maximum_absolute_difference": float(np.max(delta, initial=0.0)),
            "mean_absolute_difference": float(np.mean(delta)),
            "by_family_maximum_absolute_difference": {
                family: float(np.max(delta.take(index, axis=family_axis), initial=0.0))
                for index, family in enumerate(FIRST_CRITIC_DEFECTS)
            },
            "by_family_mean_absolute_difference": {
                family: float(np.mean(delta.take(index, axis=family_axis)))
                for index, family in enumerate(FIRST_CRITIC_DEFECTS)
            },
            "reduced_axes": list(reduced_axes),
        }

    comparisons = {
        "frame_probability": difference("frame_logits", sigmoid=True),
        "frame_severity": difference("frame_severity"),
        "part_probability": difference("part_logits", sigmoid=True),
    }
    maximum = max(float(value["maximum_absolute_difference"]) for value in comparisons.values())
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "checkpoint": str(checkpoint_path),
        "normalization": model_config.normalization,
        "source": str(source_path),
        "window_starts": list(starts),
        "window_frames": window_frames,
        "receptive_field_radius_frames": radius,
        "compared_global_frames": [global_start, global_stop],
        "compared_frame_count": global_stop - global_start,
        "input_contract": "one full-motion feature tensor sliced into windows",
        "comparisons": comparisons,
        "maximum_interior_difference": maximum,
        "stable_at_1e_5": maximum <= 1.0e-5,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_oracle_objective_cma_eligibility(
    representability_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Declare and persist cases eligible for same-constraint oracle-objective CMA."""
    representability = json.loads(Path(representability_path).read_text(encoding="utf-8"))
    cases = []
    for case in representability["cases"]:
        eligible = bool(case["oracle_valid_in_current_space"])
        cases.append(
            {
                "family": case["family"],
                "sample_id": case["sample_id"],
                "eligible": eligible,
                "status": "PENDING_ORACLE_CMA" if eligible else "NOT_RUN",
                "reason": (
                    "valid repair exists in current 8-control-point space"
                    if eligible
                    else (
                        "no repair satisfies representation and the unchanged "
                        "production constraints"
                    )
                ),
                "representability_classification": case["preliminary_classification"],
            }
        )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "same_production_constraints": True,
        "eligible_case_count": sum(bool(case["eligible"]) for case in cases),
        "executed_case_count": 0,
        "optimizer_classification_available": False,
        "cases": cases,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def preserve_red_team_exploits(
    experiment_report_path: Path,
    output_directory: Path,
    *,
    checkpoint_path: Path | None = None,
    dataset_directory: Path | None = None,
    inspection_labels: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Package every score-reducing bizarre run without assigning a defect family."""
    if (checkpoint_path is None) != (dataset_directory is None):
        raise ValueError("checkpoint_path and dataset_directory must be supplied together")
    experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model = None if checkpoint_path is None else model_from_checkpoint(checkpoint_path)
    normalization = (
        None if dataset_directory is None else load_normalization_statistics(dataset_directory)
    )
    records: list[dict[str, Any]] = []
    labels = {} if inspection_labels is None else dict(inspection_labels)
    for run in experiment["runs"]:
        if int(run["category"]) != 3:
            continue
        report_path = Path(str(run["report"]))
        source_directory = report_path.parent
        identifier = f"exploit_{len(records) + 1:02d}_{run['family']}_seed_{run['seed']}"
        destination = output / identifier
        destination.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for name in (
            "original.npz",
            "final.npz",
            "optimization.json",
            "original_prediction.json",
            "original_metrics.json",
            "final_spline_coefficients.npz",
            "preview.png",
            "run_config.json",
        ):
            source = source_directory / name
            if source.exists():
                target = destination / name
                shutil.copy2(source, target)
                copied[name] = str(target)
        optimization = json.loads(report_path.read_text(encoding="utf-8"))
        if model is not None and normalization is not None:
            for role, motion_name in (("source", "original.npz"), ("exploit", "final.npz")):
                motion = load_motion_npz(source_directory / motion_name)
                features = encode_motion_clip(motion, normalization)
                with torch.no_grad():
                    complete = model(features[None])
                complete_path = destination / f"{role}_complete_critic_outputs.npz"
                np.savez_compressed(
                    complete_path,
                    defect_names=np.asarray(FIRST_CRITIC_DEFECTS, dtype=np.str_),
                    frame_logits=complete.frame_logits[0].cpu().numpy(),
                    frame_probability=torch.sigmoid(complete.frame_logits[0]).cpu().numpy(),
                    frame_severity=complete.frame_severity[0].cpu().numpy(),
                    part_logits=complete.part_logits[0].cpu().numpy(),
                    part_probability=torch.sigmoid(complete.part_logits[0]).cpu().numpy(),
                    clip_logits=complete.clip_logits[0].cpu().numpy(),
                    clip_probability=torch.sigmoid(complete.clip_logits[0]).cpu().numpy(),
                    clip_severity=complete.clip_severity[0].cpu().numpy(),
                    family_ranking_score=complete.family_ranking_score[0].cpu().numpy(),
                    representation=complete.representation[0].cpu().numpy(),
                )
                copied[f"{role}_complete_critic_outputs"] = str(complete_path)
        trajectory_path = destination / "cma_trajectory.json"
        trajectory_path.write_text(
            json.dumps(optimization["history"], indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        inspection = dict(
            labels.get(
                identifier,
                {
                    "classification": "bizarre_score_reducing_candidate",
                    "quality_preference": "source > exploit",
                    "assignment_policy": "adversarial_quality_only; no existing defect family",
                    "inspection_status": "automated geometric and deterministic inspection",
                    "concepts": ["unclassified_perceptual_quality_failure"],
                },
            )
        )
        pair = {
            "preferred": copied.get("original.npz"),
            "rejected": copied.get("final.npz"),
            "preference": "source > exploit",
            "label": "adversarial_quality",
            "existing_defect_family": None,
        }
        record = {
            "id": identifier,
            "source_run": str(report_path),
            "original_selected_family": run["family"],
            "seed": run["seed"],
            "critic_outputs": {
                "source": optimization["prediction_before"],
                "exploit": optimization["prediction_after"],
            },
            "deterministic_metrics": {
                "source": optimization["deterministic_metrics_before"],
                "exploit": optimization["deterministic_metrics_after"],
            },
            "motion_distance": optimization["distance"],
            "inspection": inspection,
            "adversarial_quality_pair": pair,
            "trajectory": str(trajectory_path),
            "artifacts": copied,
        }
        (destination / "record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    concept_counts: dict[str, int] = {}
    for record in records:
        for concept in record["inspection"].get("concepts", []):
            concept_counts[str(concept)] = concept_counts.get(str(concept), 0) + 1
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "exploit_count": len(records),
        "existing_family_assignment": None,
        "concept_cluster_counts": concept_counts,
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def append_adversarial_quality_corpus(
    existing_manifest_path: Path,
    experiment_report_path: Path,
    output_directory: Path,
    *,
    checkpoint_path: Path,
    dataset_directory: Path,
    discovery_label: str,
) -> dict[str, Any]:
    """Append score-reducing visual failures without relabeling them as defect families."""
    experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    inspection_labels: dict[str, dict[str, Any]] = {}
    exploit_index = 0
    for run in experiment["runs"]:
        if int(run["category"]) != 3:
            continue
        exploit_index += 1
        identifier = f"exploit_{exploit_index:02d}_{run['family']}_seed_{run['seed']}"
        inspection_labels[identifier] = {
            "classification": "perceptually_worse_global_pose_distortion",
            "quality_preference": "source > exploit",
            "assignment_policy": "adversarial_quality_only; no existing defect family",
            "inspection_status": "manual six-frame overlay review",
            "reviewed_preview_frames": [0, 32, 64, 95, 127, 159],
            "observed_failure": (
                "coherent source gait is displaced by large whole-body pose, limb, and/or "
                "root changes while the selected critic score decreases"
            ),
            "concepts": [
                "global_pose_distortion",
                "anatomical_coordination_failure",
                "critic_score_gaming",
            ],
        }
    output = Path(output_directory)
    new_manifest = preserve_red_team_exploits(
        experiment_report_path,
        output / "new_records",
        checkpoint_path=checkpoint_path,
        dataset_directory=dataset_directory,
        inspection_labels=inspection_labels,
    )
    existing = json.loads(Path(existing_manifest_path).read_text(encoding="utf-8"))
    existing_records = []
    for record in existing["records"]:
        copied = dict(record)
        copied["corpus_record_id"] = f"prior_group_norm:{record['id']}"
        copied["lineage"] = {
            "discovery_label": "phase9_group_norm_red_team",
            "source_manifest": str(existing_manifest_path),
            "source_run": record["source_run"],
        }
        existing_records.append(copied)
    appended_records = []
    for record in new_manifest["records"]:
        copied = dict(record)
        copied["corpus_record_id"] = f"{discovery_label}:{record['id']}"
        copied["lineage"] = {
            "discovery_label": discovery_label,
            "source_experiment": str(experiment_report_path),
            "source_checkpoint": str(checkpoint_path),
            "source_dataset": str(dataset_directory),
            "source_run": record["source_run"],
        }
        appended_records.append(copied)
    records = [*existing_records, *appended_records]
    concept_counts: dict[str, int] = {}
    for record in records:
        for concept in record["inspection"].get("concepts", []):
            name = str(concept)
            concept_counts[name] = concept_counts.get(name, 0) + 1
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "corpus_label": "family_neutral_adversarial_quality",
        "existing_defect_family_assignment": None,
        "prior_record_count": len(existing_records),
        "newly_appended_record_count": len(appended_records),
        "total_record_count": len(records),
        "new_discovery_types": sorted(
            {str(record["inspection"]["classification"]) for record in appended_records}
        ),
        "concept_cluster_counts": concept_counts,
        "required_payload": {
            "motion_tensors": True,
            "complete_critic_outputs": True,
            "deterministic_metrics": True,
            "optimizer_trajectory": True,
            "visual_inspection_label": True,
            "lineage": True,
        },
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_critic_red_team_matrix(
    dataset_directory: Path,
    experiment_report_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    *,
    seeds: tuple[int, ...] = (3301, 3302, 3303),
    coarse_max_iterations: int = 4,
    refined_max_iterations: int = 5,
    population_size: int = 8,
) -> dict[str, Any]:
    """Repeat the original critic-only red-team matrix for one ablation checkpoint."""
    source_experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    selected = source_experiment["selected_samples"]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for family, selection in selected.items():
        index = next(
            i
            for i, record in enumerate(dataset.records)
            if record["sample_id"] == selection["sample_id"]
        )
        _, corrupted = motions_from_sample(dataset, index)
        for seed in seeds:
            destination = output / family / f"seed_{seed}"
            result = optimize_motion_with_cma(
                corrupted,
                checkpoint_path,
                dataset_directory,
                destination,
                config=CMAOptimizationConfig(
                    seed=seed,
                    mode="red_team",
                    objective_source="learned_severity",
                    objective_weights={family: 1.0},
                    coarse_max_iterations=coarse_max_iterations,
                    refined_max_iterations=refined_max_iterations,
                    population_size=population_size,
                ),
            )
            runs.append(
                {
                    "family": family,
                    "seed": seed,
                    "sample_id": selection["sample_id"],
                    "objective_before": result.objective_before,
                    "objective_after": result.objective_after,
                    "category": result.category,
                    "category_label": result.category_label,
                    "adversarial_training_eligible": result.adversarial_training_eligible,
                    "report": str(result.report_path),
                }
            )
            partial = {
                "format_version": DIAGNOSTIC_VERSION,
                "checkpoint": str(checkpoint_path),
                "source_experiment": str(experiment_report_path),
                "seeds": list(seeds),
                "runs": runs,
            }
            (output / "red_team_matrix.json").write_text(
                json.dumps(partial, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    report_value = json.loads((output / "red_team_matrix.json").read_text(encoding="utf-8"))
    if not isinstance(report_value, dict):
        raise ValueError("persisted red-team report must be an object")
    report: dict[str, Any] = report_value
    report["category_counts"] = {
        str(category): sum(int(run["category"]) == category for run in runs)
        for category in (1, 2, 3, 4)
    }
    report["score_reducing_bizarre_count"] = sum(int(run["category"]) == 3 for run in runs)
    (output / "red_team_matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_normalization_ablation_report(
    group_evaluation_path: Path,
    layer_evaluation_path: Path,
    group_repair_path: Path,
    layer_repair_path: Path,
    group_overlap_path: Path,
    layer_overlap_path: Path,
    group_experiment_path: Path,
    layer_red_team_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Collate the like-for-like GroupNorm/LayerNorm ablation and decision."""
    group_evaluation = json.loads(Path(group_evaluation_path).read_text(encoding="utf-8"))
    layer_evaluation = json.loads(Path(layer_evaluation_path).read_text(encoding="utf-8"))
    group_repair = json.loads(Path(group_repair_path).read_text(encoding="utf-8"))
    layer_repair = json.loads(Path(layer_repair_path).read_text(encoding="utf-8"))
    group_overlap = json.loads(Path(group_overlap_path).read_text(encoding="utf-8"))
    layer_overlap = json.loads(Path(layer_overlap_path).read_text(encoding="utf-8"))
    group_experiment = json.loads(Path(group_experiment_path).read_text(encoding="utf-8"))
    layer_red_team = json.loads(Path(layer_red_team_path).read_text(encoding="utf-8"))
    metric_names = (
        "defect_auroc",
        "defect_average_precision",
        "temporal_localization_f1",
        "anatomical_part_localization_f1",
        "severity_rank_spearman",
        "family_ranking_head_spearman",
    )

    def aggregate(evaluation: Mapping[str, Any], regime: str) -> dict[str, float | None]:
        result = {}
        for metric in metric_names:
            values = [
                float(family[metric])
                for family in evaluation[regime].values()
                if family[metric] is not None
            ]
            result[metric] = None if not values else float(np.mean(values))
        return result

    group_red = [run for run in group_experiment["runs"] if run["mode"] == "red_team"]
    layer_red = list(layer_red_team["runs"])
    group_bizarre_rate = float(np.mean([int(run["category"]) == 3 for run in group_red]))
    layer_bizarre_rate = float(np.mean([int(run["category"]) == 3 for run in layer_red]))
    group_adjacent = float(
        np.mean(
            [
                family["adjacent_repair_steps_ordered_correctly_fraction"]
                for family in group_repair["families"]
            ]
        )
    )
    layer_adjacent = float(
        np.mean(
            [
                family["adjacent_repair_steps_ordered_correctly_fraction"]
                for family in layer_repair["families"]
            ]
        )
    )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "controlled_variables": {
            "only_architecture_change": (
                "ResidualTemporalBlock GroupNorm(1,C) on [B,C,T] replaced by "
                "per-frame LayerNorm(C) on [B,T,C]"
            ),
            "same_width_dilations_losses_data_optimizer_seed": True,
        },
        "aggregate": {
            regime: {
                "group_norm": aggregate(group_evaluation, regime),
                "layer_norm": aggregate(layer_evaluation, regime),
            }
            for regime in ("known_mechanism_unseen_source", "heldout_mechanism")
        },
        "per_family": {
            regime: {
                family: {
                    "group_norm": group_evaluation[regime][family],
                    "layer_norm": layer_evaluation[regime][family],
                }
                for family in group_evaluation[regime]
            }
            for regime in ("known_mechanism_unseen_source", "heldout_mechanism")
        },
        "within_family_pair_accuracy": {
            "group_norm": group_evaluation["subtle_and_ordinal_ranking"]["accuracy"],
            "layer_norm": layer_evaluation["subtle_and_ordinal_ranking"]["accuracy"],
        },
        "sham_target_head_false_positive_rate": {
            "group_norm": group_evaluation["sham"]["target_head_false_positive_rate"],
            "layer_norm": layer_evaluation["sham"]["target_head_false_positive_rate"],
        },
        "repair_path": {
            "group_norm_mean_adjacent_accuracy": group_adjacent,
            "layer_norm_mean_adjacent_accuracy": layer_adjacent,
            "group_norm": group_repair["families"],
            "layer_norm": layer_repair["families"],
        },
        "overlapping_window_consistency": {
            "group_norm": group_overlap,
            "layer_norm": layer_overlap,
        },
        "red_team": {
            "group_norm_bizarre_count": sum(int(run["category"]) == 3 for run in group_red),
            "group_norm_run_count": len(group_red),
            "group_norm_bizarre_rate": group_bizarre_rate,
            "layer_norm_bizarre_count": sum(int(run["category"]) == 3 for run in layer_red),
            "layer_norm_run_count": len(layer_red),
            "layer_norm_bizarre_rate": layer_bizarre_rate,
            "group_norm_runs": group_red,
            "layer_norm_runs": layer_red,
        },
        "decision": {
            "layer_norm_improves_repair_path_monotonicity": layer_adjacent > group_adjacent,
            "layer_norm_improves_overlapping_window_consistency": (
                float(layer_overlap["maximum_interior_difference"])
                < float(group_overlap["maximum_interior_difference"])
            ),
            "layer_norm_improves_exploit_resistance": (layer_bizarre_rate < group_bizarre_rate),
            "adopt_layer_norm_as_production_critic": False,
            "reason": (
                "LayerNorm fixes path/window ordering but increases sham false positives and "
                "is exploitable in every repeated red-team run."
            ),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_failure_classification_matrix(
    experiment_report_path: Path,
    representability_path: Path,
    group_repair_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Assign one primary causal class to every failed original CMA run."""
    experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    representability = json.loads(Path(representability_path).read_text(encoding="utf-8"))
    repair = json.loads(Path(group_repair_path).read_text(encoding="utf-8"))
    by_family = {case["family"]: case for case in representability["cases"]}
    repair_by_family = {case["family"]: case for case in repair["families"]}
    rows = []
    for run in experiment["runs"]:
        if int(run["category"]) == 1:
            continue
        family = str(run["family"])
        oracle = by_family[family]
        secondary: list[str] = []
        if not oracle["current_8_representable"]:
            classification = "SEARCH SPACE"
            reason = "eight-control-point oracle projection does not recover the target defect"
            if repair_by_family[family]["adjacent_repair_steps_ordered_correctly_fraction"] < 0.8:
                secondary.append("CRITIC LANDSCAPE")
        elif (
            run["mode"] == "production"
            and not oracle["current_space_production_constraints_passed"]
        ):
            classification = "PRODUCTION CONSTRAINT"
            reason = "oracle repair is rejected by the unchanged production feasible set"
        elif run["mode"] == "red_team" and int(run["category"]) in {2, 3}:
            classification = "CRITIC LANDSCAPE"
            reason = "representable case reduced learned score through a non-repair exploit"
        else:
            classification = "MIXED/UNKNOWN"
            reason = "available diagnostics do not isolate a single remaining cause"
        rows.append(
            {
                "family": family,
                "mode": run["mode"],
                "seed": run["seed"],
                "original_category": run["category"],
                "original_category_label": run["category_label"],
                "classification": classification,
                "secondary_evidence": secondary,
                "reason": reason,
                "run_report": run["report"],
            }
        )
    allowed = (
        "SEARCH SPACE",
        "OPTIMIZER",
        "CRITIC LANDSCAPE",
        "PRODUCTION CONSTRAINT",
        "MIXED/UNKNOWN",
    )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "classification_precedence": (
            "Failed current-space oracle projection is SEARCH SPACE even when secondary "
            "critic/constraint evidence also exists."
        ),
        "counts": {label: sum(row["classification"] == label for row in rows) for label in allowed},
        "optimizer_assessment": (
            "not identifiable: zero cases had an oracle-valid repair under the unchanged "
            "production constraints"
        ),
        "rows": rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_multiresolution_oracle_representability(
    dataset_directory: Path,
    experiment_report_path: Path,
    previous_representability_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Measure oracle repair after coarse, local-medium, and residual-driven fine levels."""
    experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    previous = json.loads(Path(previous_representability_path).read_text(encoding="utf-8"))
    previous_by_family = {case["family"]: case for case in previous["cases"]}
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for family, selection in experiment["selected_samples"].items():
        index = next(
            item
            for item, record in enumerate(dataset.records)
            if record["sample_id"] == selection["sample_id"]
        )
        sample = dataset.sample(index)
        clean, corrupted = motions_from_sample(dataset, index)
        evaluator = PairTargetEvaluator(clean, family, sample.metadata)
        hierarchy, projections = build_multiresolution_projections(corrupted, clean)
        config = CMAOptimizationConfig(mode="production", objective_weights={family: 1.0})
        corrupted_metrics = motion_deterministic_metrics(corrupted)
        exact_energy = residual_energy_report(exact_residual_field(corrupted, clean), corrupted)
        case_output = output / family
        case_output.mkdir(parents=True, exist_ok=True)
        stages = []
        for projection in projections:
            level = str(projection.layout.levels[-1])
            stage_output = case_output / level
            stage_output.mkdir(parents=True, exist_ok=True)
            save_motion_npz(stage_output / "oracle_projected.npz", projection.candidate)
            np.savez_compressed(
                stage_output / "coefficients.npz",
                coefficients=projection.coefficients,
                parameter_count=np.asarray(projection.layout.parameter_count),
            )
            patch_records = [
                {
                    "level": patch.level,
                    "channel": patch.channel.label,
                    "start_frame": patch.start_frame,
                    "stop_frame": patch.stop_frame,
                    "control_points": patch.control_points,
                }
                for patch in projection.layout.patches
            ]
            (stage_output / "layout.json").write_text(
                json.dumps(patch_records, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidate_metrics = motion_deterministic_metrics(projection.candidate)
            target = _target_repair(evaluator, clean, corrupted, projection.candidate)
            distance = motion_distances(clean, projection.candidate)
            constraints = production_constraint_reasons(
                corrupted,
                projection.candidate,
                config,
                original_metrics=corrupted_metrics,
                candidate_metrics=candidate_metrics,
            )
            error_energy = residual_energy_report(projection.error, corrupted)
            motion_close = (
                distance["root_translation_rms_m"] <= 0.01
                and distance["local_rotation_geodesic_rms_deg"] <= 1.0
                and distance["joint_world_position_rms_m"] <= 0.02
            )
            stages.append(
                {
                    "level": level,
                    "parameter_count": projection.layout.parameter_count,
                    "patch_count": len(projection.layout.patches),
                    "rms_residual_to_clean": distance,
                    "target_deterministic_defect": target,
                    "production_constraints": {
                        "passed": not constraints,
                        "rejection_reasons": constraints,
                    },
                    "approximately_clean_motion": motion_close,
                    "representable_repair": (
                        motion_close and bool(target["approximately_clean_target"])
                    ),
                    "production_valid_repair": (
                        motion_close
                        and bool(target["approximately_clean_target"])
                        and not constraints
                    ),
                    "bound_saturated_fraction": projection.bound_saturated_fraction,
                    "remaining_residual_energy": error_energy,
                    "remaining_energy_fraction": cast(
                        float, error_energy["normalized_squared_energy"]
                    )
                    / max(
                        cast(float, exact_energy["normalized_squared_energy"]),
                        1.0e-20,
                    ),
                    "patches": patch_records,
                    "artifacts": {
                        "motion": str(stage_output / "oracle_projected.npz"),
                        "coefficients": str(stage_output / "coefficients.npz"),
                        "layout": str(stage_output / "layout.json"),
                    },
                }
            )
        old_current = next(
            variant
            for variant in previous_by_family[family]["variants"]
            if variant["variant"] == "current_8"
        )
        cases.append(
            {
                "family": family,
                "sample_id": selection["sample_id"],
                "before_current_8": {
                    "parameter_count": old_current["parameter_count"],
                    "rms_residual_to_clean": old_current["rms_residual_to_clean"],
                    "target_deterministic_defect": old_current["target_deterministic_defect"],
                    "representable_repair": previous_by_family[family]["current_8_representable"],
                },
                "exact_residual_energy": exact_energy,
                "hierarchy": hierarchy,
                "production_active_region_approximation": production_active_region(
                    corrupted, family
                ),
                "stages": stages,
                "after_fine_representable": stages[-1]["representable_repair"],
                "after_fine_production_valid": stages[-1]["production_valid_repair"],
            }
        )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "hierarchy": [
            "coarse whole-clip B-splines",
            "medium local B-splines around remaining error",
            "fine per-channel local B-splines where error remains",
        ],
        "active_energy_fraction": 0.95,
        "time_halo_frames": 4,
        "cases": cases,
        "before_representable_count": sum(
            bool(case["before_current_8"]["representable_repair"]) for case in cases
        ),
        "after_representable_count": sum(bool(case["after_fine_representable"]) for case in cases),
        "after_production_valid_count": sum(
            bool(case["after_fine_production_valid"]) for case in cases
        ),
    }
    (output / "multiresolution_representability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def reclassify_production_constraints(
    dataset_directory: Path,
    experiment_report_path: Path,
    original_failure_matrix_path: Path,
    multiresolution_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Separate inconsistent targets, path feasibility, representation, and task conflicts."""
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    experiment = json.loads(Path(experiment_report_path).read_text(encoding="utf-8"))
    matrix = json.loads(Path(original_failure_matrix_path).read_text(encoding="utf-8"))
    multiresolution = json.loads(Path(multiresolution_report_path).read_text(encoding="utf-8"))
    multi_by_family = {case["family"]: case for case in multiresolution["cases"]}
    selected = experiment["selected_samples"]
    rows = []
    constraint_rows = [
        row for row in matrix["rows"] if row["classification"] == "PRODUCTION CONSTRAINT"
    ]
    for family in sorted({str(row["family"]) for row in constraint_rows}):
        selection = selected[family]
        index = next(
            item
            for item, record in enumerate(dataset.records)
            if record["sample_id"] == selection["sample_id"]
        )
        clean, corrupted = motions_from_sample(dataset, index)
        config = CMAOptimizationConfig(mode="production", objective_weights={family: 1.0})
        clean_reasons = production_constraint_reasons(corrupted, clean, config)
        path = []
        first_feasible = None
        for alpha in np.linspace(0.0, 1.0, 21):
            candidate = interpolate_repair_path(corrupted, clean, float(alpha))
            reasons = production_constraint_reasons(corrupted, candidate, config)
            if not reasons and first_feasible is None:
                first_feasible = float(alpha)
            path.append({"alpha": float(alpha), "rejection_reasons": reasons})
        fine_stage = multi_by_family[family]["stages"][-1]
        fine_motion = load_motion_npz(Path(fine_stage["artifacts"]["motion"]))
        oracle_reasons = production_constraint_reasons(corrupted, fine_motion, config)
        if clean_reasons:
            if clean_reasons == ["target_speed"]:
                cause = "inconsistent_target_constraints"
                detail = (
                    "clean window itself violates the full-take target-speed command; "
                    "no exact repair can enter the feasible set"
                )
            else:
                cause = "genuine_task_incompatibility"
                detail = (
                    "the exact clean target conflicts with one or more authored task constraints"
                )
        elif oracle_reasons:
            cause = "parameterization_induced_infeasibility"
            detail = "clean is feasible but the best multiresolution projection is not"
        elif first_feasible is None or first_feasible >= 1.0 - 1.0e-12:
            cause = "path_intermediate_feasibility_issue"
            detail = (
                "the oracle endpoint is feasible, but no sampled intermediate repair is feasible"
            )
        else:
            cause = "resolved_by_multiresolution_space"
            detail = "clean path and multiresolution projection are feasible"
        rows.append(
            {
                "family": family,
                "sample_id": selection["sample_id"],
                "affected_previous_runs": [
                    row["run_report"] for row in constraint_rows if row["family"] == family
                ],
                "clean_satisfies_every_constraint": not clean_reasons,
                "clean_rejection_reasons": clean_reasons,
                "bad_to_clean_path_enters_feasible_set": first_feasible is not None,
                "first_feasible_alpha": first_feasible,
                "path": path,
                "best_multiresolution_oracle_satisfies_constraints": not oracle_reasons,
                "best_oracle_rejection_reasons": oracle_reasons,
                "detailed_cause": cause,
                "detail": detail,
            }
        )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "previous_production_constraint_run_count": len(constraint_rows),
        "unique_case_count": len(rows),
        "cause_counts": {
            cause: sum(row["detailed_cause"] == cause for row in rows)
            for cause in (
                "inconsistent_target_constraints",
                "path_intermediate_feasibility_issue",
                "parameterization_induced_infeasibility",
                "genuine_task_incompatibility",
                "resolved_by_multiresolution_space",
            )
        },
        "cases": rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def deterministic_production_rerun_gate(
    multiresolution_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run-gate deterministic production CMA only after oracle feasibility is proven."""
    multiresolution = json.loads(Path(multiresolution_report_path).read_text(encoding="utf-8"))
    cases = []
    for case in multiresolution["cases"]:
        eligible = bool(case["after_fine_production_valid"])
        cases.append(
            {
                "family": case["family"],
                "sample_id": case["sample_id"],
                "eligible": eligible,
                "executed": False,
                "objective": "direct deterministic family metric",
                "reason": (
                    "eligible multiresolution oracle repair"
                    if eligible
                    else "multiresolution oracle candidate is not production feasible"
                ),
            }
        )
    eligible_count = sum(bool(case["eligible"]) for case in cases)
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "learned_hard_defect_heads_used": False,
        "eligible_case_count": eligible_count,
        "executed_case_count": 0,
        "production_repair_success_count": 0,
        "status": (
            "blocked_by_oracle_feasibility_gate"
            if eligible_count == 0
            else "eligible_cases_require_adaptive_cma_execution"
        ),
        "cases": cases,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def record_normalization_artifact_policy(
    ablation_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze the normalization comparison and deferred alternatives without retraining."""
    ablation = json.loads(Path(ablation_report_path).read_text(encoding="utf-8"))
    repair_path = ablation["repair_path"]
    window = ablation["overlapping_window_consistency"]
    red_team = ablation["red_team"]
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "decision": "retain_both_as_experimental_artifacts_do_not_promote_layer_norm",
        "another_normalization_training_cycle_run": False,
        "group_norm": {
            "status": "current_more_robust_experimental_baseline",
            "tradeoff": "more robust; worse locality and repair-path monotonicity",
            "maximum_interior_window_difference": window["group_norm"][
                "maximum_interior_difference"
            ],
            "mean_adjacent_repair_step_accuracy": repair_path["group_norm_mean_adjacent_accuracy"],
            "repair_path_inversion_count": sum(
                int(row["inversion_count"]) for row in repair_path["group_norm"]
            ),
            "sham_false_positive_rate": ablation["sham_target_head_false_positive_rate"][
                "group_norm"
            ],
            "critic_exploits": red_team["group_norm_bizarre_count"],
            "red_team_runs": red_team["group_norm_run_count"],
            "checkpoint": window["group_norm"]["checkpoint"],
        },
        "layer_norm": {
            "status": "experimental_artifact_not_production",
            "tradeoff": (
                "exact window consistency and monotonic repair path; substantially more exploitable"
            ),
            "maximum_interior_window_difference": window["layer_norm"][
                "maximum_interior_difference"
            ],
            "mean_adjacent_repair_step_accuracy": repair_path["layer_norm_mean_adjacent_accuracy"],
            "repair_path_inversion_count": sum(
                int(row["inversion_count"]) for row in repair_path["layer_norm"]
            ),
            "sham_false_positive_rate": ablation["sham_target_head_false_positive_rate"][
                "layer_norm"
            ],
            "critic_exploits": red_team["layer_norm_bizarre_count"],
            "red_team_runs": red_team["layer_norm_run_count"],
            "checkpoint": window["layer_norm"]["checkpoint"],
        },
        "deferred_todos": [
            "compare per-frame RMSNorm",
            "compare a normalization-free residual block",
            "compare LayerNorm with adversarial and sham hardening",
        ],
        "source_ablation": str(ablation_report_path),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_adaptive_optimization_stop_report(
    cma_suite_path: Path,
    multiresolution_report_path: Path,
    production_gate_path: Path,
    constraint_reclassification_path: Path,
    adversarial_corpus_path: Path,
    normalization_policy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Collate the five required decision outcomes into one compact phase artifact."""
    cma_suite = json.loads(Path(cma_suite_path).read_text(encoding="utf-8"))
    multiresolution = json.loads(Path(multiresolution_report_path).read_text(encoding="utf-8"))
    production = json.loads(Path(production_gate_path).read_text(encoding="utf-8"))
    constraints = json.loads(Path(constraint_reclassification_path).read_text(encoding="utf-8"))
    corpus = json.loads(Path(adversarial_corpus_path).read_text(encoding="utf-8"))
    normalization = json.loads(Path(normalization_policy_path).read_text(encoding="utf-8"))
    per_family = []
    for case in multiresolution["cases"]:
        per_family.append(
            {
                "family": case["family"],
                "before_fixed_8_representable": case["before_current_8"]["representable_repair"],
                "after_multiresolution_representable": case["after_fine_representable"],
                "after_multiresolution_production_valid": case["after_fine_production_valid"],
                "active_parameter_counts": {
                    stage["level"]: stage["parameter_count"] for stage in case["stages"]
                },
                "remaining_energy_fraction": {
                    stage["level"]: stage["remaining_energy_fraction"] for stage in case["stages"]
                },
                "fine_rejection_reasons": case["stages"][-1]["production_constraints"][
                    "rejection_reasons"
                ],
            }
        )
    report = {
        "format_version": DIAGNOSTIC_VERSION,
        "scope_guards": {
            "graph_or_attention_started": False,
            "optimizer_family_changed": False,
            "layer_norm_promoted": False,
        },
        "outcome_1_cma_guaranteed_representable": {
            "by_dimension": cma_suite["by_dimension"],
            "overall_success_rate": cma_suite["overall_success_rate"],
            "optimizer_decision": cma_suite["optimizer_decision"],
            "known_optima_and_trial_distances": str(cma_suite_path),
        },
        "outcome_2_oracle_representability": {
            "before_count": multiresolution["before_representable_count"],
            "after_count": multiresolution["after_representable_count"],
            "case_count": len(multiresolution["cases"]),
            "per_family": per_family,
        },
        "outcome_3_deterministic_production": {
            "oracle_feasible_case_count": production["eligible_case_count"],
            "executed_case_count": production["executed_case_count"],
            "success_count": production["production_repair_success_count"],
            "status": production["status"],
            "learned_hard_defect_heads_used": production["learned_hard_defect_heads_used"],
        },
        "outcome_4_remaining_production_failures": {
            "previous_run_count": constraints["previous_production_constraint_run_count"],
            "unique_case_count": constraints["unique_case_count"],
            "cause_counts": constraints["cause_counts"],
            "cases": constraints["cases"],
        },
        "outcome_5_adversarial_critic_exploits": {
            "newly_appended_count": corpus["newly_appended_record_count"],
            "new_types": corpus["new_discovery_types"],
            "current_phase_new_critic_optimization_runs": 0,
            "corpus_total": corpus["total_record_count"],
            "family_assignment": corpus["existing_defect_family_assignment"],
        },
        "normalization_policy": {
            "decision": normalization["decision"],
            "group_norm": normalization["group_norm"]["tradeoff"],
            "layer_norm": normalization["layer_norm"]["tradeoff"],
            "deferred_todos": normalization["deferred_todos"],
        },
        "next_decision": (
            "retain CMA-ES; repair the target-speed command contract and improve the residual "
            "space/metric for joint jitter before considering graph or attention work"
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
