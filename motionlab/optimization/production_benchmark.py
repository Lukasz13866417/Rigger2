"""Production-constrained and autonomous deterministic repair benchmarks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.forensics import (
    motion_deterministic_metrics,
    motion_distances,
    motions_from_sample,
)
from motionlab.io.npz import save_motion_npz
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.motion.clip import MotionClip
from motionlab.optimization.adaptive import (
    AdaptiveResidualLayout,
    AdaptiveSplineBlock,
    LocalSplinePatch,
    ResidualChannel,
    fit_adaptive_projection,
    joint_halo,
)
from motionlab.optimization.benchmark_semantics import (
    ProductionBenchmarkPair,
    prepare_synthetic_production_benchmark,
)
from motionlab.optimization.block_cma import optimize_parameter_block_with_cma
from motionlab.optimization.cmaes import CMAOptimizationConfig, production_constraint_reasons
from motionlab.optimization.jitter_blocks import JitterSmoothingBlock
from motionlab.repair.foot_lock import lock_foot

PRODUCTION_BENCHMARK_VERSION = "motionlab.autonomous_deterministic_repair.v1"
PRODUCTION_FAMILIES = ("foot_slide", "floating_contact", "joint_pop")
_FAMILY_OBJECTIVE_METRIC = {
    "foot_slide": "maximum_stance_slip_cm",
    "floating_contact": "maximum_floating_contact_cm",
    "joint_pop": "localized_peak_angular_jerk_rad_s3",
}
_MINIMUM_TARGET_GAIN = {
    "foot_slide": 0.2,
    "floating_contact": 0.2,
    "joint_pop": 100.0,
}
_COLLATERAL_TOLERANCE = {
    "maximum_stance_slip_cm": 0.5,
    "maximum_floating_contact_cm": 0.5,
    "angular_smoothness_rad_s3_p99": 500.0,
    "loop_seam": 0.2,
}
_PRIVILEGED_METADATA_KEYS = frozenset(
    {
        "audit_role",
        "catalog_mechanism_key",
        "catalog_partition",
        "clean_window_reference_speed",
        "collateral_defect_families",
        "corruption_family",
        "corruption_mechanism",
        "corruption_schema_metadata",
        "corruption_seed",
        "counterfactual_group_id",
        "generation_parameters",
        "gait_event_labels",
        "gait_phase_bins",
        "gait_target",
        "measured_severity",
        "metadata_only_fields",
        "ordinal_chain_id",
        "ordinal_rank",
        "parent_motion_id",
        "parent_clip_speed",
        "postcondition_after",
        "postcondition_before",
        "postcondition_metric",
        "preference_confidence",
        "production_benchmark_semantics_version",
        "requested_severity_parameter",
        "sample_id",
        "sample_role",
        "severity_bin",
        "severity_label",
        "single_defect",
        "target_side",
        "target_speed_source",
    }
)
_INFERENCE_METADATA_ALLOWLIST = frozenset(
    {
        "coordinate_convention",
        "expected_speed_mps",
        "facing_frame",
        "ground_plane",
        "loop_kind",
        "loop_mode",
        "speed_units",
        "target_speed",
        "target_speed_mps",
        "target_speed_tolerance",
        "task_target_speed",
    }
)


@dataclass(frozen=True)
class DeterministicObjective:
    """Inference-only target metric and the localized region used to evaluate it."""

    family: str
    metric_name: str
    joint_indices: tuple[int, ...]
    time_interval: tuple[int, int]
    localization: dict[str, Any]

    def value(self, clip: MotionClip) -> float:
        if self.family == "joint_pop":
            values = np.asarray(smoothness_metric(clip).joint_frame_values, dtype=np.float64)
            start, stop = self.time_interval
            return float(np.max(values[start:stop, self.joint_indices], initial=0.0))
        metrics = motion_deterministic_metrics(clip)
        key = (
            "maximum_stance_slip_cm"
            if self.family == "foot_slide"
            else "maximum_floating_contact_cm"
        )
        value = metrics[key]
        if value is None:
            raise ValueError(f"deterministic metric {key!r} is unavailable")
        return float(value)


@dataclass(frozen=True)
class _OptimizationEvidence:
    candidate: MotionClip
    objective: DeterministicObjective
    evaluations: int
    runtime_seconds: float
    parameter_count: int
    block_name: str
    report_path: Path
    selection: dict[str, Any]


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def inference_available_motion(clip: MotionClip) -> MotionClip:
    """Remove synthetic labels and clean/corrupt lineage before autonomous diagnosis."""
    metadata_only = clip.metadata.get("metadata_only_fields", ())
    dynamically_privileged = (
        {str(key) for key in metadata_only}
        if isinstance(metadata_only, (list, tuple, set))
        else set()
    )
    metadata = {
        key: value
        for key, value in clip.metadata.items()
        if key in _INFERENCE_METADATA_ALLOWLIST
        and key not in _PRIVILEGED_METADATA_KEYS
        and key not in dynamically_privileged
    }
    metadata["benchmark_input_contract"] = "inference_available_motion_and_task_requirements"
    return clip.with_updates(metadata=metadata)


def _diagnostic_events(clip: MotionClip, family: str) -> list[Any]:
    target = clip.metadata.get("target_speed")
    report = grade_motion_deterministic(
        clip,
        target_speed_mps=None if target is None else float(target),
    )
    if family == "foot_slide":
        return [
            event
            for metric in report.metrics.values()
            for event in metric.events
            if "foot_slide" in event.type
        ]
    return [
        event
        for metric in report.metrics.values()
        for event in metric.events
        if event.type == family
    ]


def _joint_pop_event_outlier_ratio(event: Any) -> float:
    """Normalize jerk by the joint's robust motion-specific detection threshold."""
    threshold = float(event.evidence.get("robust_threshold_rad_s3", 0.0))
    # Effectively static terminal joints can have a near-zero numerical threshold and
    # turn harmless floating-point noise into an unbounded ratio.
    if threshold < 0.1:
        return 0.0
    return float(event.severity) / threshold


def _objective_from_diagnostics(clip: MotionClip, family: str) -> DeterministicObjective:
    events = _diagnostic_events(clip, family)
    if family == "joint_pop":
        if not events:
            raise ValueError("joint-pop optimization requires a deterministic localized event")
        # Absolute jerk can be naturally large in energetic clips.  A synthetic pop is an
        # isolated within-joint outlier, so localization uses the detector's robust threshold.
        strongest = max(events, key=_joint_pop_event_outlier_ratio)
        indices = tuple(
            clip.skeleton.joint_names.index(name)
            for name in strongest.joints
            if name in clip.skeleton.joint_names
        )
        if not indices:
            raise ValueError("joint-pop event has no resolvable joint")
        start = max(0, int(strongest.frames[0]) - 8)
        stop = min(clip.num_frames, int(strongest.frames[1]) + 8)
        return DeterministicObjective(
            family,
            "localized_peak_angular_jerk_rad_s3",
            indices,
            (start, stop),
            {
                "source": "strongest_deterministic_joint_pop_event",
                "event_severity": float(strongest.severity),
                "event_outlier_ratio": _joint_pop_event_outlier_ratio(strongest),
                "event_frames": list(strongest.frames),
                "event_joints": list(strongest.joints),
            },
        )
    metric = "maximum_stance_slip_cm" if family == "foot_slide" else "maximum_floating_contact_cm"
    if events:
        strongest = max(events, key=lambda event: float(event.severity))
        start = max(0, int(strongest.frames[0]) - 8)
        stop = min(clip.num_frames, int(strongest.frames[1]) + 8)
    else:
        strongest = None
        start, stop = 0, clip.num_frames
    return DeterministicObjective(
        family,
        metric,
        (),
        (start, stop),
        {
            "source": (
                "strongest_deterministic_contact_event"
                if strongest is not None
                else "conservative_full_clip_fallback"
            ),
            "event_severity": None if strongest is None else float(strongest.severity),
            "event_frames": None if strongest is None else list(strongest.frames),
            "event_joints": None if strongest is None else list(strongest.joints),
        },
    )


def _dominant_pop_components(
    clip: MotionClip, objective: DeterministicObjective
) -> tuple[int, ...]:
    start, stop = objective.time_interval
    center = (start + stop) // 2
    if stop - start >= 3:
        values = np.asarray(
            smoothness_metric(clip).joint_frame_values,
            dtype=np.float64,
        )
        local = values[start:stop, objective.joint_indices]
        peak = np.unravel_index(int(np.argmax(local)), local.shape)
        center = start + int(peak[0])
    if center <= 0 or center >= clip.num_frames - 1:
        return (0, 1, 2)
    from motionlab.math.quaternion import quaternion_difference, quaternion_log, quaternion_slerp

    joint = objective.joint_indices[0]
    expected = quaternion_slerp(
        clip.local_quat_wxyz[center - 1, joint],
        clip.local_quat_wxyz[center + 1, joint],
        0.5,
    )
    deviation = np.abs(
        quaternion_log(quaternion_difference(expected, clip.local_quat_wxyz[center, joint]))
    )
    maximum = float(np.max(deviation))
    selected = tuple(int(index) for index in np.flatnonzero(deviation >= maximum * 0.15))
    return selected or (int(np.argmax(deviation)),)


def build_inference_adaptive_block(
    clip: MotionClip,
    objective: DeterministicObjective,
) -> tuple[AdaptiveSplineBlock, dict[str, Any]]:
    """Build coarse/local/fine patches from deterministic localization without clean data."""
    start, stop = objective.time_interval
    patches: list[LocalSplinePatch] = []
    components: tuple[int, ...]
    if objective.family in {"foot_slide", "floating_contact"}:
        components = (0, 2) if objective.family == "foot_slide" else (1,)
        axes = ("x", "y", "z")
        for component in components:
            channel = ResidualChannel(
                "root_translation",
                0,
                component,
                f"root_translation:{axes[component]}",
            )
            patches.extend(
                (
                    LocalSplinePatch("coarse", channel, 0, clip.num_frames, 4),
                    LocalSplinePatch("medium", channel, start, stop, 8),
                    LocalSplinePatch("fine", channel, start, stop, 12),
                )
            )
        selected_joints: tuple[int, ...] = ()
        selected_components = list(components)
    else:
        primary = objective.joint_indices
        selected_joints = joint_halo(clip, primary)
        components = _dominant_pop_components(clip, objective)
        axes = ("x", "y", "z")
        for joint in selected_joints:
            for component in components:
                channel = ResidualChannel(
                    "rotation",
                    joint,
                    component,
                    f"rotation:{clip.skeleton.joint_names[joint]}:{axes[component]}",
                )
                patches.append(LocalSplinePatch("coarse", channel, 0, clip.num_frames, 4))
                if joint in primary:
                    patches.extend(
                        (
                            LocalSplinePatch("medium", channel, start, stop, 8),
                            LocalSplinePatch(
                                "fine",
                                channel,
                                start,
                                stop,
                                max(4, stop - start + 3),
                            ),
                        )
                    )
        selected_components = list(components)
    block = AdaptiveSplineBlock(AdaptiveResidualLayout(tuple(patches)))
    return block, {
        "policy": "deterministic localization plus temporal and parent/child halo",
        "family": objective.family,
        "time_interval": [start, stop],
        "joint_indices": list(selected_joints),
        "joint_names": [clip.skeleton.joint_names[index] for index in selected_joints],
        "rotation_or_root_components": selected_components,
        "parameter_count": len(block.parameter_names),
        "clean_reference_used_for_selection": False,
    }


def _constraint_function(
    original: MotionClip,
    family: str,
    config: CMAOptimizationConfig,
) -> Any:
    before = motion_deterministic_metrics(original)
    target_key = {
        "foot_slide": "maximum_stance_slip_cm",
        "floating_contact": "maximum_floating_contact_cm",
        "joint_pop": "angular_smoothness_rad_s3_p99",
    }[family]

    def reasons(candidate: MotionClip) -> list[str]:
        after = motion_deterministic_metrics(candidate)
        result = production_constraint_reasons(
            original,
            candidate,
            config,
            original_metrics=before,
            candidate_metrics=after,
        )
        for metric, tolerance in _COLLATERAL_TOLERANCE.items():
            if metric == target_key:
                continue
            old = before.get(metric)
            new = after.get(metric)
            if old is not None and new is not None and float(new) > float(old) + tolerance:
                result.append(f"collateral:{metric}")
        return result

    return reasons


def _optimize_without_clean_reference(
    corrupted: MotionClip,
    family: str,
    output_directory: Path,
    *,
    seed: int,
    maximum_evaluations: int,
) -> _OptimizationEvidence:
    """Run diagnosis and CMA with no clean motion in this function's contract."""
    objective = _objective_from_diagnostics(corrupted, family)
    block, selection = build_inference_adaptive_block(corrupted, objective)
    config = CMAOptimizationConfig(
        seed=seed,
        mode="production",
        objective_source="deterministic_target",
        objective_weights={family: 1.0},
    )
    started = time.perf_counter()
    result = optimize_parameter_block_with_cma(
        corrupted,
        block,
        objective.value,
        output_directory,
        constraint_reasons=_constraint_function(corrupted, family, config),
        seed=seed,
        maximum_evaluations=maximum_evaluations,
    )
    retry_report: dict[str, Any] = {"used": False}
    total_evaluations = result.evaluations
    # An unlucky broad population can become almost entirely infeasible and trigger CMA's
    # flat-fitness stop.  Re-spend the remaining budget locally instead of changing optimizers.
    minimum_gain = _MINIMUM_TARGET_GAIN[family]
    initial_objective = objective.value(corrupted)
    remaining = maximum_evaluations - result.evaluations
    if (
        family in {"foot_slide", "floating_contact"}
        and result.objective > initial_objective - minimum_gain
        and remaining > 0
    ):
        retry = optimize_parameter_block_with_cma(
            corrupted,
            block,
            objective.value,
            output_directory / "constraint_aware_retry",
            constraint_reasons=_constraint_function(corrupted, family, config),
            seed=seed,
            maximum_evaluations=remaining,
            initial_sigma=0.08,
        )
        total_evaluations += retry.evaluations
        retry_report = {
            "used": True,
            "reason": "broad_population_flat_fitness_in_narrow_feasible_corridor",
            "initial_sigma": 0.08,
            "evaluations": retry.evaluations,
            "report": str(retry.report_path),
        }
        if retry.objective < result.objective:
            result = retry
    elapsed = time.perf_counter() - started
    selection = {**selection, "constraint_aware_retry": retry_report}
    return _OptimizationEvidence(
        candidate=result.motion,
        objective=objective,
        evaluations=total_evaluations,
        runtime_seconds=elapsed,
        parameter_count=len(block.parameter_names),
        block_name=block.name,
        report_path=result.report_path,
        selection=selection,
    )


def _specialized_foot_lock(
    corrupted: MotionClip,
    objective: DeterministicObjective,
    output_directory: Path,
    config: CMAOptimizationConfig,
) -> dict[str, Any]:
    events = _diagnostic_events(corrupted, "foot_slide")
    if not events:
        return {"available": False, "reason": "no deterministic foot-slide interval"}
    strongest = max(events, key=lambda event: float(event.severity))
    side = "left" if strongest.type.startswith("left") else "right"
    related = [event for event in events if event.type.startswith(side)]
    start, stop = map(int, strongest.frames)
    changed = True
    while changed:
        changed = False
        for event in related:
            event_start, event_stop = map(int, event.frames)
            if event_start <= stop + 4 and event_stop >= start - 4:
                new = (min(start, event_start), max(stop, event_stop))
                changed = changed or new != (start, stop)
                start, stop = new
    reject = _constraint_function(corrupted, "foot_slide", config)
    best: tuple[float, MotionClip, dict[str, Any], list[str]] | None = None
    attempts = 0
    for strength in (0.5, 0.75, 1.0):
        for blend in (2, 4):
            for anchor_mode in ("first", "median"):
                attempts += 1
                try:
                    repaired = lock_foot(
                        corrupted,
                        side=cast(Any, side),
                        start_frame=start,
                        stop_frame=stop,
                        anchor_mode=cast(Any, anchor_mode),
                        lock_strength=strength,
                        blend_in_frames=blend,
                        blend_out_frames=blend,
                        pelvis_compensation_limit_m=0.03,
                        ground_plane=corrupted.metadata.get("ground_plane", (0.0, 1.0, 0.0, 0.0)),
                    ).repaired
                    score = objective.value(repaired)
                    reasons = reject(repaired)
                except (ValueError, FloatingPointError):
                    continue
                settings = {
                    "side": side,
                    "interval": [start, stop],
                    "strength": strength,
                    "blend_frames": blend,
                    "anchor_mode": anchor_mode,
                }
                ranking = score if not reasons else 1.0e12 + score
                if best is None or ranking < (best[0] if not best[3] else 1.0e12 + best[0]):
                    best = (score, repaired, settings, reasons)
    if best is None:
        return {"available": False, "reason": "all foot-lock attempts failed"}
    output_directory.mkdir(parents=True, exist_ok=True)
    save_motion_npz(output_directory / "best.npz", best[1])
    return {
        "available": True,
        "operator": "lock_foot_two_bone_ik",
        "attempt_count": attempts,
        "target_metric": best[0],
        "constraint_reasons": best[3],
        "settings": best[2],
        "metrics": motion_deterministic_metrics(best[1]),
        "artifact": str(output_directory / "best.npz"),
    }


def _specialized_pop_smoothing(
    corrupted: MotionClip,
    objective: DeterministicObjective,
    output_directory: Path,
    config: CMAOptimizationConfig,
) -> dict[str, Any]:
    block = JitterSmoothingBlock(
        objective.joint_indices,
        objective.time_interval[0],
        objective.time_interval[1],
    )
    reject = _constraint_function(corrupted, "joint_pop", config)
    best: tuple[float, MotionClip, dict[str, Any], list[str]] | None = None
    attempts = 0
    for cutoff in (3.0, 6.0, 12.0):
        for strength in (0.5, 0.75, 1.0):
            for blend in (2.0, 4.0):
                attempts += 1
                values = block.initial_values
                values[0] = cutoff
                values[1] = strength
                values[2:4] = blend
                candidate = block.apply(corrupted, values)
                score = objective.value(candidate)
                reasons = reject(candidate)
                settings = {
                    "cutoff_frequency_hz": cutoff,
                    "smoothing_strength": strength,
                    "blend_frames": blend,
                }
                ranking = score if not reasons else 1.0e12 + score
                if best is None or ranking < (best[0] if not best[3] else 1.0e12 + best[0]):
                    best = (score, candidate, settings, reasons)
    assert best is not None
    output_directory.mkdir(parents=True, exist_ok=True)
    save_motion_npz(output_directory / "best.npz", best[1])
    return {
        "available": True,
        "operator": "local_butterworth_so3_smoothing",
        "attempt_count": attempts,
        "target_metric": best[0],
        "constraint_reasons": best[3],
        "settings": best[2],
        "metrics": motion_deterministic_metrics(best[1]),
        "artifact": str(output_directory / "best.npz"),
    }


def _trial_report(
    pair: ProductionBenchmarkPair,
    family: str,
    evidence: _OptimizationEvidence,
    output_directory: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    config = CMAOptimizationConfig(
        mode="production",
        objective_source="deterministic_target",
        objective_weights={family: 1.0},
    )
    before_value = evidence.objective.value(pair.corrupted)
    after_value = evidence.objective.value(evidence.candidate)
    after_reasons = _constraint_function(pair.corrupted, family, config)(evidence.candidate)
    oracle_block, _ = build_inference_adaptive_block(pair.corrupted, evidence.objective)
    oracle = fit_adaptive_projection(pair.corrupted, pair.clean, oracle_block.layout)
    oracle_value = evidence.objective.value(oracle.candidate)
    oracle_reasons = _constraint_function(pair.corrupted, family, config)(oracle.candidate)
    if family == "foot_slide":
        specialized = _specialized_foot_lock(
            pair.corrupted,
            evidence.objective,
            output_directory / "specialized_foot_lock",
            config,
        )
    elif family == "joint_pop":
        specialized = _specialized_pop_smoothing(
            pair.corrupted,
            evidence.objective,
            output_directory / "specialized_smoothing",
            config,
        )
    else:
        specialized = {
            "available": False,
            "reason": "no existing specialized operator for this family",
        }
    target_reduction = before_value - after_value
    success = not after_reasons and target_reduction >= _MINIMUM_TARGET_GAIN[family]
    return {
        "family": family,
        "seed": seed,
        "eligible": pair.eligible,
        "speed_semantics": {
            "task_target_speed": pair.task_target_speed,
            "clean_window_reference_speed": pair.clean_window_reference_speed,
            "parent_clip_speed": pair.parent_clip_speed,
            "target_speed": pair.target_speed,
            "target_speed_tolerance": pair.target_speed_tolerance,
            "target_speed_source": pair.target_speed_source,
        },
        "clean_endpoint_constraint_reasons": list(pair.clean_endpoint_constraint_reasons),
        "clean_reference_available_during_optimization": False,
        "objective": {
            "source": "deterministic_metric",
            "metric": evidence.objective.metric_name,
            "before": before_value,
            "after": after_value,
            "reduction": target_reduction,
        },
        "success": success,
        "constraint_reasons": after_reasons,
        "metrics_before": motion_deterministic_metrics(pair.corrupted),
        "metrics_after": motion_deterministic_metrics(evidence.candidate),
        "distance_from_clean_evaluation_only": motion_distances(pair.clean, evidence.candidate),
        "optimizer": {
            "family": "CMA-ES",
            "parameter_block": evidence.block_name,
            "parameter_count": evidence.parameter_count,
            "evaluations": evidence.evaluations,
            "runtime_seconds": evidence.runtime_seconds,
            "selection": evidence.selection,
            "report": str(evidence.report_path),
        },
        "oracle_projection_evaluation_only": {
            "target_metric": oracle_value,
            "constraint_reasons": oracle_reasons,
            "distance_from_clean": motion_distances(pair.clean, oracle.candidate),
            "parameter_count": oracle_block.layout.parameter_count,
        },
        "specialized_deterministic_repair": specialized,
    }


def select_production_benchmark_cases(
    dataset: FixedRigSampleDataset,
    *,
    sources_per_family: int = 2,
) -> list[tuple[str, int]]:
    """Select measurable moderate cases from distinct source clips.

    Clean motion is used here only by the benchmark curator to prevent a known metric-floor
    confound.  It is never forwarded to diagnosis, block construction, or optimization.
    """
    selected: list[tuple[str, int]] = []
    for family in PRODUCTION_FAMILIES:
        candidates = [
            index
            for index, record in enumerate(dataset.records)
            if record.get("sample_role") == "hard_corruption"
            and record.get("catalog_partition") == "train"
            and record.get("corruption_family") == family
            and record.get("severity_label") == "moderate"
        ]
        by_source: dict[str, tuple[float, int]] = {}
        for index in candidates:
            source = str(dataset.records[index]["source_clip_id"])
            clean, corrupted = motions_from_sample(dataset, index)
            objective = _objective_from_diagnostics(corrupted, family)
            margin = objective.value(corrupted) - objective.value(clean)
            incumbent = by_source.get(source)
            if incumbent is None or margin > incumbent[0]:
                by_source[source] = (margin, index)
        if len(by_source) < sources_per_family:
            raise ValueError(f"insufficient distinct eligible sources for {family}")
        ranked = sorted(
            by_source.items(),
            key=lambda item: (-item[1][0], item[0]),
        )
        for _, (_, index) in ranked[:sources_per_family]:
            selected.append((family, index))
    return selected


def audit_production_benchmark_eligibility(
    dataset: FixedRigSampleDataset,
) -> dict[str, Any]:
    """Compare corrected window targets with the inherited legacy target semantics."""
    by_family: dict[str, dict[str, int]] = {}
    for family in PRODUCTION_FAMILIES:
        counts = {
            "candidate_count": 0,
            "corrected_eligible_count": 0,
            "legacy_eligible_count": 0,
            "newly_eligible_after_speed_correction": 0,
            "corrected_ineligible_count": 0,
        }
        for index, record in enumerate(dataset.records):
            if (
                record.get("sample_role") != "hard_corruption"
                or record.get("catalog_partition") != "train"
                or record.get("corruption_family") != family
            ):
                continue
            counts["candidate_count"] += 1
            clean, corrupted = motions_from_sample(dataset, index)
            config = CMAOptimizationConfig(
                mode="production",
                objective_source="deterministic_target",
                objective_weights={family: 1.0},
            )
            legacy_reasons = production_constraint_reasons(corrupted, clean, config)
            legacy_eligible = not legacy_reasons
            pair = prepare_synthetic_production_benchmark(clean, corrupted, config)
            pair.assert_clean_endpoint_feasible()
            if legacy_eligible:
                counts["legacy_eligible_count"] += 1
            if pair.eligible:
                counts["corrected_eligible_count"] += 1
                if not legacy_eligible:
                    counts["newly_eligible_after_speed_correction"] += 1
            else:
                counts["corrected_ineligible_count"] += 1
        by_family[family] = counts
    totals = {
        key: sum(values[key] for values in by_family.values())
        for key in next(iter(by_family.values()))
    }
    return {
        "scope": "train-partition hard-corruption records in production families",
        "legacy_semantics": "inherited parent/full-clip expected_speed_mps",
        "corrected_semantics": "measured clean-window speed without external task target",
        "by_family": by_family,
        "totals": totals,
    }


def run_production_cma_benchmark(
    dataset_directory: Path,
    output_directory: Path,
    *,
    seeds: tuple[int, ...] = (6101, 6102, 6103),
    sources_per_family: int = 2,
    maximum_evaluations: int = 1_200,
) -> dict[str, Any]:
    """Run family-specified CMA trials after enforcing feasible clean endpoints."""
    if len(seeds) < 3:
        raise ValueError("production benchmark requires at least three seeds per case")
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    eligibility_audit = audit_production_benchmark_eligibility(dataset)
    cases = select_production_benchmark_cases(dataset, sources_per_family=sources_per_family)
    runs: list[dict[str, Any]] = []
    selections = []
    eligible_case_count = 0
    for family, index in cases:
        record = dataset.records[index]
        clean, corrupted = motions_from_sample(dataset, index)
        config = CMAOptimizationConfig(
            mode="production",
            objective_source="deterministic_target",
            objective_weights={family: 1.0},
        )
        pair = prepare_synthetic_production_benchmark(clean, corrupted, config)
        pair.assert_clean_endpoint_feasible()
        selection_objective = _objective_from_diagnostics(pair.corrupted, family)
        case_id = f"{family}__{record['source_clip_id']}__{str(record['sample_id'])[14:26]}"
        selection = {
            "case_id": case_id,
            "family": family,
            "sample_id": record["sample_id"],
            "source_clip_id": record["source_clip_id"],
            "eligible": pair.eligible,
            "ineligibility_reasons": list(pair.ineligibility_reasons),
            "benchmark_selection_target_margin": (
                selection_objective.value(pair.corrupted) - selection_objective.value(pair.clean)
            ),
            "clean_used_for_selection_only": True,
        }
        selections.append(selection)
        if not pair.eligible:
            continue
        eligible_case_count += 1
        clean_prepared = pair.clean
        corrupted_input = inference_available_motion(pair.corrupted)
        pair = ProductionBenchmarkPair(
            clean=clean_prepared,
            corrupted=corrupted_input,
            task_target_speed=pair.task_target_speed,
            clean_window_reference_speed=pair.clean_window_reference_speed,
            parent_clip_speed=pair.parent_clip_speed,
            target_speed=pair.target_speed,
            target_speed_tolerance=pair.target_speed_tolerance,
            target_speed_source=pair.target_speed_source,
            eligible=pair.eligible,
            ineligibility_reasons=pair.ineligibility_reasons,
            clean_endpoint_constraint_reasons=pair.clean_endpoint_constraint_reasons,
        )
        case_output = output / case_id
        save_motion_npz(case_output / "clean_evaluation_only.npz", pair.clean)
        save_motion_npz(case_output / "corrupted_input.npz", pair.corrupted)
        for seed in seeds:
            run_output = case_output / f"seed_{seed}"
            evidence = _optimize_without_clean_reference(
                pair.corrupted,
                family,
                run_output,
                seed=seed,
                maximum_evaluations=maximum_evaluations,
            )
            report = _trial_report(pair, family, evidence, run_output, seed=seed)
            report.update({"case_id": case_id, "sample_id": record["sample_id"]})
            _json_write(run_output / "benchmark.json", report)
            runs.append(report)
            _json_write(
                output / "production_benchmark.json",
                {
                    "format_version": PRODUCTION_BENCHMARK_VERSION,
                    "mode": "family_specified_deterministic_production",
                    "seeds": list(seeds),
                    "selected_cases": selections,
                    "runs": runs,
                },
            )
    by_family = {}
    for family in PRODUCTION_FAMILIES:
        family_runs = [run for run in runs if run["family"] == family]
        specialized_runs = [
            run for run in family_runs if run["specialized_deterministic_repair"].get("available")
        ]
        collateral_keys = (
            "maximum_stance_slip_cm",
            "maximum_ground_penetration_cm",
            "maximum_floating_contact_cm",
            "angular_smoothness_rad_s3_p99",
            "loop_seam",
            "absolute_target_speed_error_mps",
        )
        median_metric_deltas = {}
        for key in collateral_keys:
            deltas = [
                float(run["metrics_after"][key]) - float(run["metrics_before"][key])
                for run in family_runs
                if run["metrics_before"].get(key) is not None
                and run["metrics_after"].get(key) is not None
            ]
            median_metric_deltas[key] = None if not deltas else float(np.median(deltas))
        by_family[family] = {
            "run_count": len(family_runs),
            "success_count": sum(bool(run["success"]) for run in family_runs),
            "success_rate": (
                0.0
                if not family_runs
                else sum(bool(run["success"]) for run in family_runs) / len(family_runs)
            ),
            "median_evaluations": (
                None
                if not family_runs
                else float(np.median([run["optimizer"]["evaluations"] for run in family_runs]))
            ),
            "median_runtime_seconds": (
                None
                if not family_runs
                else float(np.median([run["optimizer"]["runtime_seconds"] for run in family_runs]))
            ),
            "median_target_metric_before": (
                None
                if not family_runs
                else float(np.median([run["objective"]["before"] for run in family_runs]))
            ),
            "median_target_metric_after": (
                None
                if not family_runs
                else float(np.median([run["objective"]["after"] for run in family_runs]))
            ),
            "median_target_reduction": (
                None
                if not family_runs
                else float(np.median([run["objective"]["reduction"] for run in family_runs]))
            ),
            "median_clean_rotation_rms_deg_evaluation_only": (
                None
                if not family_runs
                else float(
                    np.median(
                        [
                            run["distance_from_clean_evaluation_only"][
                                "local_rotation_geodesic_rms_deg"
                            ]
                            for run in family_runs
                        ]
                    )
                )
            ),
            "median_clean_root_rms_m_evaluation_only": (
                None
                if not family_runs
                else float(
                    np.median(
                        [
                            run["distance_from_clean_evaluation_only"]["root_translation_rms_m"]
                            for run in family_runs
                        ]
                    )
                )
            ),
            "median_oracle_projection_target_metric": (
                None
                if not family_runs
                else float(
                    np.median(
                        [
                            run["oracle_projection_evaluation_only"]["target_metric"]
                            for run in family_runs
                        ]
                    )
                )
            ),
            "median_deterministic_metric_deltas_after_minus_before": median_metric_deltas,
            "specialized_comparison": {
                "available_run_count": len(specialized_runs),
                "median_target_metric": (
                    None
                    if not specialized_runs
                    else float(
                        np.median(
                            [
                                run["specialized_deterministic_repair"]["target_metric"]
                                for run in specialized_runs
                            ]
                        )
                    )
                ),
                "generic_cma_lower_target_count": sum(
                    run["objective"]["after"]
                    < run["specialized_deterministic_repair"]["target_metric"]
                    for run in specialized_runs
                ),
            },
        }
    report = {
        "format_version": PRODUCTION_BENCHMARK_VERSION,
        "mode": "family_specified_deterministic_production",
        "clean_endpoint_invariant": (
            "every eligible synthetic production benchmark has a feasible clean endpoint"
        ),
        "selected_case_count": len(cases),
        "selected_eligible_case_count": eligible_case_count,
        "eligibility_audit": eligibility_audit,
        "executed_run_count": len(runs),
        "seeds": list(seeds),
        "selected_cases": selections,
        "by_family": by_family,
        "runs": runs,
    }
    _json_write(output / "production_benchmark.json", report)
    return report


def diagnose_repair_family(clip: MotionClip) -> tuple[str, dict[str, float]]:
    """Choose a repair family using only deterministic anomaly magnitudes."""
    metrics = motion_deterministic_metrics(clip)
    pop_events = _diagnostic_events(clip, "joint_pop")
    pop_outlier_ratio = max(
        (_joint_pop_event_outlier_ratio(event) for event in pop_events),
        default=0.0,
    )
    scores = {
        # Ratios near 1--3 occur naturally; an injected single-frame pop is normally an
        # order-of-magnitude outlier against the same joint's robust temporal baseline.
        "joint_pop": pop_outlier_ratio / 10.0,
        "floating_contact": float(metrics["maximum_floating_contact_cm"] or 0.0) / 5.0,
        "foot_slide": float(metrics["maximum_stance_slip_cm"] or 0.0) / 3.0,
    }
    return max(scores, key=scores.__getitem__), scores


def select_repair_family(
    clip: MotionClip,
    task_requirements: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Activate a permitted family block from task requirements, with diagnostics fallback."""
    requested_metric = task_requirements.get("primary_objective_metric")
    requested = next(
        (
            family
            for family, metric in _FAMILY_OBJECTIVE_METRIC.items()
            if metric == requested_metric
        ),
        None,
    )
    diagnostic_family, diagnostic_scores = diagnose_repair_family(clip)
    if requested is not None:
        return requested, {
            "source": "declared_task_objective",
            "primary_objective_metric": requested_metric,
            "diagnostic_recommendation": diagnostic_family,
            "diagnostic_scores": diagnostic_scores,
        }
    return diagnostic_family, {
        "source": "deterministic_diagnostic_fallback",
        "primary_objective_metric": requested_metric,
        "diagnostic_recommendation": diagnostic_family,
        "diagnostic_scores": diagnostic_scores,
    }


def run_autonomous_deterministic_repair_benchmark(
    dataset_directory: Path,
    output_directory: Path,
    *,
    seed: int = 7101,
    sources_per_family: int = 2,
    maximum_evaluations: int = 1_200,
) -> dict[str, Any]:
    """Select a block and optimize without exposing clean motion until evaluation."""
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for expected_family, index in select_production_benchmark_cases(
        dataset, sources_per_family=sources_per_family
    ):
        record = dataset.records[index]
        clean, corrupted = motions_from_sample(dataset, index)
        config = CMAOptimizationConfig(
            mode="production",
            objective_source="deterministic_target",
            objective_weights={expected_family: 1.0},
        )
        pair = prepare_synthetic_production_benchmark(clean, corrupted, config)
        if not pair.eligible:
            continue
        inference_input = inference_available_motion(pair.corrupted)
        task_requirements = {
            "primary_objective_metric": _FAMILY_OBJECTIVE_METRIC[expected_family],
            "hard_constraints": "production_invariants",
        }
        selected_family, block_selection = select_repair_family(
            inference_input,
            task_requirements,
        )
        case_id = (
            f"{expected_family}__{record['source_clip_id']}__{str(record['sample_id'])[14:26]}"
        )
        case_output = output / case_id
        evidence = _optimize_without_clean_reference(
            inference_input,
            selected_family,
            case_output,
            seed=seed,
            maximum_evaluations=maximum_evaluations,
        )
        evaluation_pair = ProductionBenchmarkPair(
            clean=pair.clean,
            corrupted=inference_input,
            task_target_speed=pair.task_target_speed,
            clean_window_reference_speed=pair.clean_window_reference_speed,
            parent_clip_speed=pair.parent_clip_speed,
            target_speed=pair.target_speed,
            target_speed_tolerance=pair.target_speed_tolerance,
            target_speed_source=pair.target_speed_source,
            eligible=pair.eligible,
            ineligibility_reasons=pair.ineligibility_reasons,
            clean_endpoint_constraint_reasons=pair.clean_endpoint_constraint_reasons,
        )
        row = _trial_report(
            evaluation_pair,
            selected_family,
            evidence,
            case_output,
            seed=seed,
        )
        row.update(
            {
                "case_id": case_id,
                "sample_id": record["sample_id"],
                "evaluation_only_expected_family": expected_family,
                "selected_family": selected_family,
                "family_selection_correct": selected_family == expected_family,
                "task_requirements": task_requirements,
                "block_selection": block_selection,
            }
        )
        _json_write(case_output / "autonomous_benchmark.json", row)
        rows.append(row)
    by_family = {}
    for family in PRODUCTION_FAMILIES:
        family_rows = [row for row in rows if row["evaluation_only_expected_family"] == family]
        collateral_keys = (
            "maximum_stance_slip_cm",
            "maximum_ground_penetration_cm",
            "maximum_floating_contact_cm",
            "angular_smoothness_rad_s3_p99",
            "loop_seam",
            "absolute_target_speed_error_mps",
        )
        median_metric_deltas = {}
        for key in collateral_keys:
            deltas = [
                float(row["metrics_after"][key]) - float(row["metrics_before"][key])
                for row in family_rows
                if row["metrics_before"].get(key) is not None
                and row["metrics_after"].get(key) is not None
            ]
            median_metric_deltas[key] = None if not deltas else float(np.median(deltas))
        by_family[family] = {
            "case_count": len(family_rows),
            "selection_accuracy": (
                0.0
                if not family_rows
                else sum(bool(row["family_selection_correct"]) for row in family_rows)
                / len(family_rows)
            ),
            "repair_success_rate": (
                0.0
                if not family_rows
                else sum(
                    bool(row["success"] and row["family_selection_correct"]) for row in family_rows
                )
                / len(family_rows)
            ),
            "median_target_reduction": (
                None
                if not family_rows
                else float(np.median([row["objective"]["reduction"] for row in family_rows]))
            ),
            "median_evaluations": (
                None
                if not family_rows
                else float(np.median([row["optimizer"]["evaluations"] for row in family_rows]))
            ),
            "median_runtime_seconds": (
                None
                if not family_rows
                else float(np.median([row["optimizer"]["runtime_seconds"] for row in family_rows]))
            ),
            "median_clean_rotation_rms_deg_evaluation_only": (
                None
                if not family_rows
                else float(
                    np.median(
                        [
                            row["distance_from_clean_evaluation_only"][
                                "local_rotation_geodesic_rms_deg"
                            ]
                            for row in family_rows
                        ]
                    )
                )
            ),
            "median_clean_root_rms_m_evaluation_only": (
                None
                if not family_rows
                else float(
                    np.median(
                        [
                            row["distance_from_clean_evaluation_only"]["root_translation_rms_m"]
                            for row in family_rows
                        ]
                    )
                )
            ),
            "median_deterministic_metric_deltas_after_minus_before": median_metric_deltas,
        }
    report = {
        "format_version": PRODUCTION_BENCHMARK_VERSION,
        "mode": "autonomous_inference_only_deterministic_repair",
        "clean_reference_available_before_completion": False,
        "input_contract": ["corrupted_animation", "task_requirements"],
        "system_contract": [
            "inference_available_diagnostics",
            "deterministic_metrics",
            "permitted_parameter_blocks",
            "hard_constraints",
        ],
        "case_count": len(rows),
        "overall_selection_accuracy": (
            0.0
            if not rows
            else sum(bool(row["family_selection_correct"]) for row in rows) / len(rows)
        ),
        "overall_diagnostic_only_recommendation_accuracy": (
            0.0
            if not rows
            else sum(
                row["block_selection"]["diagnostic_recommendation"]
                == row["evaluation_only_expected_family"]
                for row in rows
            )
            / len(rows)
        ),
        "overall_repair_success_rate": (
            0.0
            if not rows
            else sum(bool(row["success"] and row["family_selection_correct"]) for row in rows)
            / len(rows)
        ),
        "by_family": by_family,
        "cases": rows,
    }
    _json_write(output / "autonomous_benchmark.json", report)
    return report
