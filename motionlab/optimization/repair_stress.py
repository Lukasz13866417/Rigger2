"""Label-independent stress tests for the deterministic repair portfolio.

The active perceptual pilot must not be involved in this module.  It consumes only
persisted motion/corruption data, inference-available deterministic diagnostics and
declared task requirements.  Clean motion is withheld until the repair candidates
have been produced and is then used only for benchmark qualification and distance
reporting.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.forensics import (
    motion_deterministic_metrics,
    motion_distances,
    motions_from_sample,
)
from motionlab.io.npz import save_motion_npz
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.metrics.joint_limits import joint_limit_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.metrics.speed import speed_metric
from motionlab.motion.clip import MotionClip
from motionlab.optimization.acceptance import (
    ProductionAcceptance,
    assess_production_acceptance,
)
from motionlab.optimization.adaptive import (
    AdaptiveResidualLayout,
    AdaptiveSplineBlock,
    LocalSplinePatch,
    ResidualChannel,
)
from motionlab.optimization.benchmark_semantics import prepare_synthetic_production_benchmark
from motionlab.optimization.block_cma import optimize_parameter_block_with_cma
from motionlab.optimization.cmaes import CMAOptimizationConfig, production_constraint_reasons
from motionlab.optimization.jitter_blocks import JitterSmoothingBlock
from motionlab.optimization.parameter_blocks import LoopSeamBlock, SpeedCadenceBlock
from motionlab.optimization.production_benchmark import (
    DeterministicObjective,
    build_inference_adaptive_block,
    inference_available_motion,
)
from motionlab.repair.close_loop import close_loop
from motionlab.repair.foot_lock import lock_foot

DETERMINISTIC_REPAIR_STRESS_VERSION = "motionlab.deterministic_repair_stress.v2"

STRESS_FAMILIES = (
    "foot_slide",
    "floating_contact",
    "ground_penetration",
    "joint_pop",
    "joint_jitter",
    "loop_seam",
    "speed_inconsistency",
    "joint_limit",
)

_TARGET_METRIC = {
    "foot_slide": "maximum_stance_slip_cm",
    "floating_contact": "maximum_floating_contact_cm",
    "ground_penetration": "maximum_ground_penetration_cm",
    "joint_pop": "localized_peak_angular_jerk_rad_s3",
    "joint_jitter": "localized_peak_angular_jerk_rad_s3",
    "loop_seam": "loop_seam",
    "speed_inconsistency": "absolute_target_speed_error_mps",
    "joint_limit": "maximum_joint_limit_violation_deg",
}

_MINIMUM_GAIN = {
    "foot_slide": 0.2,
    "floating_contact": 0.2,
    "ground_penetration": 0.2,
    "joint_pop": 100.0,
    "joint_jitter": 100.0,
    "loop_seam": 0.02,
    "speed_inconsistency": 0.02,
    "joint_limit": 0.2,
}

_COLLATERAL_TOLERANCE = {
    "maximum_stance_slip_cm": 0.5,
    "maximum_ground_penetration_cm": 0.5,
    "maximum_floating_contact_cm": 0.5,
    "angular_smoothness_rad_s3_p99": 500.0,
    "loop_seam": 0.2,
    "absolute_target_speed_error_mps": 0.05,
    "maximum_joint_limit_violation_deg": 0.5,
}

_TARGET_CONSTRAINT_REASON = {
    "ground_penetration": "ground_penetration",
    "speed_inconsistency": "target_speed",
    "loop_seam": "loop_invariant",
}


@dataclass(frozen=True)
class StressCaseSelection:
    """One reproducibly selected real-window synthetic corruption."""

    family: str
    index: int
    sample_id: str
    source_clip_id: str
    source_window_start: int
    mechanism: str
    target_margin_evaluation_only: float
    production_claim_eligible: bool
    qualification_reasons: tuple[str, ...]


@dataclass(frozen=True)
class StressObjective:
    """A deterministic scalar fixed before any repair candidate is generated."""

    family: str
    metric_name: str
    joint_indices: tuple[int, ...] = ()
    time_interval: tuple[int, int] | None = None
    localization: Mapping[str, Any] | None = None
    use_localized_angular_peak: bool = False

    def value(self, clip: MotionClip) -> float:
        if self.use_localized_angular_peak and self.joint_indices:
            values = np.asarray(smoothness_metric(clip).joint_frame_values, dtype=np.float64)
            start, stop = self.time_interval or (0, clip.num_frames)
            return float(np.max(values[start:stop, self.joint_indices], initial=0.0))
        value = motion_deterministic_metrics(clip).get(self.metric_name)
        if value is None:
            raise ValueError(f"deterministic objective {self.metric_name!r} is unavailable")
        return float(value)


@dataclass(frozen=True)
class _MethodCandidate:
    method: str
    kind: str
    motion: MotionClip
    evaluations: int
    runtime_seconds: float
    parameter_block: str | None
    parameter_count: int
    evidence: Mapping[str, Any]
    portfolio_selection_eligible: bool = True
    execution_purpose: str = "specialized_primary"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_fresh_output(path: Path) -> None:
    """Prevent a prior clean-evaluation artifact from being visible to a new repair run."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"deterministic repair stress output must be new or empty: {path}")


def _target_metric_for_collateral(family: str) -> str:
    if family in {"joint_pop", "joint_jitter"}:
        return "angular_smoothness_rad_s3_p99"
    return _TARGET_METRIC[family]


def _optimization_config(family: str, *, seed: int) -> CMAOptimizationConfig:
    # A requested speed change necessarily authorizes endpoint-displacement change.  All
    # remaining production invariants stay enabled, and travel direction is still preserved.
    displacement_tolerance = 1.0 if family == "speed_inconsistency" else 0.1
    weight_family = family if family != "joint_limit" else "joint_pop"
    return CMAOptimizationConfig(
        seed=seed,
        mode="production",
        objective_source="deterministic_target",
        objective_weights={weight_family: 1.0},
        production_displacement_relative_tolerance=displacement_tolerance,
    )


def _non_target_constraint_function(
    original: MotionClip,
    family: str,
    config: CMAOptimizationConfig,
) -> Any:
    """Keep production invariants hard while leaving the target defect continuous."""
    before = motion_deterministic_metrics(original)
    target_metric = _target_metric_for_collateral(family)
    target_reason = _TARGET_CONSTRAINT_REASON.get(family)

    def reasons(candidate: MotionClip) -> list[str]:
        after = motion_deterministic_metrics(candidate)
        result = [
            reason
            for reason in production_constraint_reasons(
                original,
                candidate,
                config,
                original_metrics=before,
                candidate_metrics=after,
            )
            if reason != target_reason
        ]
        for metric, tolerance in _COLLATERAL_TOLERANCE.items():
            if metric == target_metric:
                continue
            old = before.get(metric)
            new = after.get(metric)
            if old is not None and new is not None and float(new) > float(old) + tolerance:
                result.append(f"collateral:{metric}")
        return list(dict.fromkeys(result))

    return reasons


def _event_localized_objective(clip: MotionClip, family: str) -> StressObjective:
    metric = smoothness_metric(clip)
    angular_events = [
        event
        for event in metric.events
        if event.type in {"joint_pop", "joint_jitter"}
        and float(event.evidence.get("robust_threshold_rad_s3", 0.0)) >= 0.1
    ]
    events = [event for event in angular_events if event.type == family]
    if not events:
        events = angular_events
    if not events:
        return StressObjective(
            family,
            "angular_smoothness_rad_s3_p99",
            localization={"source": "full_clip_p99_fallback"},
        )
    strongest = max(events, key=lambda event: float(event.severity))
    selected_events = [strongest]
    if family == "joint_jitter":
        # Broadband jitter often appears as many adjacent short "pop" events.  Aggregate
        # repeated robust-threshold exceedances by joint instead of selecting a long event on
        # an effectively static terminal joint (whose numerical threshold is meaningless).
        by_joint: dict[str, list[Any]] = {}
        for event in angular_events:
            for name in event.joints:
                by_joint.setdefault(name, []).append(event)
        ranked_joints = sorted(
            by_joint,
            key=lambda name: (
                -sum(
                    max(
                        0.0,
                        float(event.severity) - float(event.evidence["robust_threshold_rad_s3"]),
                    )
                    for event in by_joint[name]
                ),
                -len(by_joint[name]),
                name,
            ),
        )
        selected_names = ranked_joints[:4]
        selected_events = [event for name in selected_names for event in by_joint[name]]
    joints = tuple(
        clip.skeleton.joint_names.index(name)
        for name in dict.fromkeys(name for event in selected_events for name in event.joints)
        if name in clip.skeleton.joint_names
    )
    if not joints:
        return StressObjective(
            family,
            "angular_smoothness_rad_s3_p99",
            localization={"source": "full_clip_p99_unresolvable_event_fallback"},
        )
    event_start = min(int(event.frames[0]) for event in selected_events)
    event_stop = max(int(event.frames[1]) for event in selected_events)
    start = max(0, event_start - 8)
    stop = min(clip.num_frames, event_stop + 8)
    if stop - start < 16:
        missing = 16 - (stop - start)
        start = max(0, start - missing // 2)
        stop = min(clip.num_frames, stop + missing - missing // 2)
        start = max(0, stop - 16)
    return StressObjective(
        family,
        (
            "localized_peak_angular_jerk_rad_s3"
            if family == "joint_pop"
            else "angular_smoothness_rad_s3_p99"
        ),
        joints,
        (start, stop),
        {
            "source": f"strongest_deterministic_{strongest.type}_event",
            "event_frames": [event_start, event_stop],
            "event_joints": [clip.skeleton.joint_names[index] for index in joints],
            "event_severity": float(strongest.severity),
            "aggregated_event_count": len(selected_events),
        },
        use_localized_angular_peak=family == "joint_pop",
    )


def deterministic_stress_objective(clip: MotionClip, family: str) -> StressObjective:
    """Resolve a target from inference-available deterministic evidence only."""
    if family not in STRESS_FAMILIES:
        raise ValueError(f"unsupported deterministic stress family {family!r}")
    if family in {"joint_pop", "joint_jitter"}:
        return _event_localized_objective(clip, family)
    return StressObjective(family, _TARGET_METRIC[family], localization={"source": "clip_metric"})


def _ground_interval(clip: MotionClip, family: str) -> tuple[int, int]:
    report = grade_motion_deterministic(
        clip,
        target_speed_mps=float(clip.metadata["target_speed"]),
    )
    events = [
        event
        for metric in report.metrics.values()
        for event in metric.events
        if event.type == family
    ]
    if not events:
        return 0, clip.num_frames
    strongest = max(events, key=lambda event: float(event.severity))
    return max(0, int(strongest.frames[0]) - 8), min(clip.num_frames, int(strongest.frames[1]) + 8)


def _generic_block(clip: MotionClip, objective: StressObjective) -> Any:
    family = objective.family
    if family in {"foot_slide", "floating_contact"}:
        compatible = DeterministicObjective(
            family,
            objective.metric_name,
            (),
            _ground_interval(clip, family),
            dict(objective.localization or {}),
        )
        return build_inference_adaptive_block(clip, compatible)[0]
    if family == "ground_penetration":
        start, stop = _ground_interval(clip, family)
        channel = ResidualChannel("root_translation", 0, 1, "root_translation:y")
        return AdaptiveSplineBlock(
            AdaptiveResidualLayout(
                (
                    LocalSplinePatch("coarse", channel, 0, clip.num_frames, 4),
                    LocalSplinePatch("medium", channel, start, stop, 8),
                    LocalSplinePatch("fine", channel, start, stop, 12),
                )
            )
        )
    if family in {"joint_pop", "joint_jitter"}:
        if not objective.joint_indices or objective.time_interval is None:
            raise ValueError("angular stress case has no deterministic localized event")
        compatible = DeterministicObjective(
            "joint_pop",
            objective.metric_name,
            objective.joint_indices,
            objective.time_interval,
            dict(objective.localization or {}),
        )
        return build_inference_adaptive_block(clip, compatible)[0]
    if family == "loop_seam":
        return LoopSeamBlock()
    if family == "speed_inconsistency":
        return SpeedCadenceBlock()
    if family == "joint_limit":
        metric = joint_limit_metric(clip)
        if not metric.events:
            raise ValueError("joint-limit case has no trusted localized violation")
        strongest = max(metric.events, key=lambda event: float(event.severity))
        joint = clip.skeleton.joint_names.index(strongest.joints[0])
        start = max(0, int(strongest.frames[0]) - 8)
        stop = min(clip.num_frames, int(strongest.frames[1]) + 8)
        patches = tuple(
            LocalSplinePatch(
                "fine",
                ResidualChannel(
                    "rotation",
                    joint,
                    component,
                    f"rotation:{clip.skeleton.joint_names[joint]}:{axis}",
                ),
                start,
                stop,
                max(4, min(12, stop - start + 3)),
            )
            for component, axis in enumerate(("x", "y", "z"))
        )
        return AdaptiveSplineBlock(AdaptiveResidualLayout(patches))
    raise AssertionError("unreachable stress family")


def _best_grid_candidate(
    original: MotionClip,
    objective: StressObjective,
    family: str,
    candidates: Sequence[tuple[MotionClip, Mapping[str, Any]]],
    config: CMAOptimizationConfig,
) -> tuple[MotionClip, float, Mapping[str, Any], list[str]]:
    reject = _non_target_constraint_function(original, family, config)
    best: tuple[tuple[int, float], MotionClip, Mapping[str, Any], list[str]] | None = None
    for candidate, settings in candidates:
        reasons = reject(candidate)
        value = objective.value(candidate)
        key = (int(bool(reasons)), value)
        if best is None or key < best[0]:
            best = (key, candidate, settings, reasons)
    if best is None:
        raise ValueError("specialized repair grid was empty")
    return best[1], float(best[0][1]), best[2], best[3]


def _specialized_contact_height(
    original: MotionClip,
    objective: StressObjective,
    family: str,
    config: CMAOptimizationConfig,
) -> _MethodCandidate:
    started = time.perf_counter()
    signs = (
        -np.linspace(0.0, 0.08, 17) if family == "floating_contact" else np.linspace(0.0, 0.08, 17)
    )
    candidates = []
    for offset in signs:
        root = original.root_translation_m.copy()
        root[:, 1] += float(offset)
        candidates.append(
            (
                original.with_updates(
                    root_translation_m=root,
                    metadata={
                        **dict(original.metadata),
                        "deterministic_repair": {
                            "operator": "constant_root_height_contact_alignment",
                            "offset_m": float(offset),
                        },
                    },
                ),
                {"root_height_offset_m": float(offset)},
            )
        )
    motion, _, settings, reasons = _best_grid_candidate(
        original, objective, family, candidates, config
    )
    return _MethodCandidate(
        "specialized_contact_height_alignment",
        "specialized",
        motion,
        len(candidates),
        time.perf_counter() - started,
        None,
        1,
        {"settings": settings, "non_target_constraint_reasons": reasons},
    )


def _specialized_foot_lock_candidate(
    original: MotionClip,
    objective: StressObjective,
    config: CMAOptimizationConfig,
) -> _MethodCandidate | None:
    report = grade_motion_deterministic(
        original,
        target_speed_mps=float(original.metadata["target_speed"]),
    )
    events = [
        event
        for metric in report.metrics.values()
        for event in metric.events
        if "foot_slide" in event.type
    ]
    if not events:
        return None
    started = time.perf_counter()
    strongest = max(events, key=lambda event: float(event.severity))
    side: Literal["left", "right"] = "left" if strongest.type.startswith("left") else "right"
    start, stop = map(int, strongest.frames)
    candidates: list[tuple[MotionClip, Mapping[str, Any]]] = []
    for strength in (0.5, 0.75, 1.0):
        for blend in (2, 4):
            for anchor_mode in ("first", "median"):
                try:
                    result = lock_foot(
                        original,
                        side=side,
                        start_frame=start,
                        stop_frame=stop,
                        anchor_mode=anchor_mode,
                        lock_strength=strength,
                        blend_in_frames=blend,
                        blend_out_frames=blend,
                        pelvis_compensation_limit_m=0.03,
                        ground_plane=original.metadata.get("ground_plane", (0.0, 1.0, 0.0, 0.0)),
                    )
                except (ValueError, FloatingPointError):
                    continue
                candidates.append(
                    (
                        result.repaired,
                        {
                            "side": side,
                            "interval": [start, stop],
                            "strength": strength,
                            "blend_frames": blend,
                            "anchor_mode": anchor_mode,
                        },
                    )
                )
    if not candidates:
        return None
    motion, _, settings, reasons = _best_grid_candidate(
        original, objective, "foot_slide", candidates, config
    )
    return _MethodCandidate(
        "specialized_foot_lock_ik",
        "specialized",
        motion,
        len(candidates),
        time.perf_counter() - started,
        None,
        4,
        {"settings": settings, "non_target_constraint_reasons": reasons},
    )


def _specialized_smoothing_candidate(
    original: MotionClip,
    objective: StressObjective,
    config: CMAOptimizationConfig,
) -> _MethodCandidate | None:
    if not objective.joint_indices or objective.time_interval is None:
        return None
    started = time.perf_counter()
    block = JitterSmoothingBlock(
        objective.joint_indices,
        objective.time_interval[0],
        objective.time_interval[1],
    )
    candidates: list[tuple[MotionClip, Mapping[str, Any]]] = []
    for cutoff in (3.0, 6.0, 12.0):
        if cutoff >= original.fps / 2.0:
            continue
        for strength in (0.5, 0.75, 1.0):
            for blend in (2.0, 4.0):
                values = block.initial_values
                values[0] = cutoff
                values[1] = strength
                values[2:4] = blend
                candidates.append(
                    (
                        block.apply(original, values),
                        {
                            "cutoff_frequency_hz": cutoff,
                            "smoothing_strength": strength,
                            "blend_frames": blend,
                        },
                    )
                )
    motion, _, settings, reasons = _best_grid_candidate(
        original, objective, objective.family, candidates, config
    )
    return _MethodCandidate(
        "specialized_local_so3_smoothing",
        "specialized",
        motion,
        len(candidates),
        time.perf_counter() - started,
        block.name,
        len(block.parameter_names),
        {"settings": settings, "non_target_constraint_reasons": reasons},
    )


def _specialized_loop_candidate(
    original: MotionClip,
    objective: StressObjective,
    config: CMAOptimizationConfig,
) -> _MethodCandidate:
    started = time.perf_counter()
    candidates = []
    for window in (6, 12, 18):
        for strength in (0.25, 0.5, 0.75, 1.0):
            try:
                repaired = close_loop(
                    original,
                    seam_window_frames=min(window, original.num_frames - 1),
                    strength=strength,
                ).repaired
            except (ValueError, FloatingPointError):
                continue
            candidates.append((repaired, {"seam_window_frames": window, "strength": strength}))
    motion, _, settings, reasons = _best_grid_candidate(
        original, objective, "loop_seam", candidates, config
    )
    return _MethodCandidate(
        "specialized_loop_closure",
        "specialized",
        motion,
        len(candidates),
        time.perf_counter() - started,
        None,
        2,
        {"settings": settings, "non_target_constraint_reasons": reasons},
    )


def _specialized_speed_candidate(
    original: MotionClip,
    objective: StressObjective,
    config: CMAOptimizationConfig,
) -> _MethodCandidate | None:
    target_raw = original.metadata.get("target_speed")
    measured = speed_metric(original).clip_value
    if target_raw is None or measured is None or measured <= 1.0e-8:
        return None
    started = time.perf_counter()
    scale = float(target_raw) / float(measured)
    if not np.isfinite(scale) or scale <= 0.0 or scale > 3.0:
        return None
    plane = np.asarray(
        original.metadata.get("ground_plane", (0.0, 1.0, 0.0, 0.0)), dtype=np.float64
    )
    normal = plane[:3] / np.linalg.norm(plane[:3])
    root = original.root_translation_m.astype(np.float64)
    origin = root[0]
    displacement = root - origin
    normal_component = np.sum(displacement * normal, axis=-1, keepdims=True) * normal
    tangent = displacement - normal_component
    repaired_root = origin + normal_component + scale * tangent
    candidate = original.with_updates(
        root_translation_m=repaired_root.astype(np.float32),
        metadata={
            **dict(original.metadata),
            "deterministic_repair": {
                "operator": "direct_root_progression_rescale",
                "scale": scale,
                "target_speed_mps": float(target_raw),
                "measured_input_speed_mps": float(measured),
            },
        },
    )
    reasons = _non_target_constraint_function(original, "speed_inconsistency", config)(candidate)
    return _MethodCandidate(
        "specialized_direct_speed_rescale",
        "specialized",
        candidate,
        1,
        time.perf_counter() - started,
        None,
        1,
        {"scale": scale, "non_target_constraint_reasons": reasons},
    )


def _specialized_candidate(
    original: MotionClip,
    objective: StressObjective,
    config: CMAOptimizationConfig,
) -> _MethodCandidate | None:
    family = objective.family
    if family == "foot_slide":
        return _specialized_foot_lock_candidate(original, objective, config)
    if family in {"floating_contact", "ground_penetration"}:
        return _specialized_contact_height(original, objective, family, config)
    if family in {"joint_pop", "joint_jitter"}:
        return _specialized_smoothing_candidate(original, objective, config)
    if family == "loop_seam":
        return _specialized_loop_candidate(original, objective, config)
    if family == "speed_inconsistency":
        return _specialized_speed_candidate(original, objective, config)
    return None


def _generic_cma_candidate(
    corrupted: MotionClip,
    objective: StressObjective,
    family: str,
    output_directory: Path,
    config: CMAOptimizationConfig,
    *,
    seed: int,
    maximum_evaluations: int,
    production_fallback: bool,
) -> _MethodCandidate:
    """Run the common CMA baseline without access to privileged benchmark data."""
    started = time.perf_counter()
    block = _generic_block(corrupted, objective)
    purpose = (
        "production_fallback_and_evaluation_head_to_head"
        if production_fallback
        else "evaluation_only_head_to_head"
    )
    artifact_name = "cma_fallback" if production_fallback else "cma_evaluation_only"
    method = (
        "generic_deterministic_cma_fallback"
        if production_fallback
        else "generic_deterministic_cma_evaluation_only"
    )
    cma = optimize_parameter_block_with_cma(
        corrupted,
        block,
        objective.value,
        output_directory / artifact_name,
        constraint_reasons=_non_target_constraint_function(corrupted, family, config),
        seed=seed,
        maximum_evaluations=maximum_evaluations,
        initial_sigma=0.12 if family in {"ground_penetration", "speed_inconsistency"} else 0.25,
    )
    return _MethodCandidate(
        method,
        "generic_cma",
        cma.motion,
        cma.evaluations,
        time.perf_counter() - started,
        block.name,
        len(block.parameter_names),
        {
            "optimizer": "CMA-ES",
            "optimizer_report": str(cma.report_path),
            "head_to_head_baseline": True,
            "clean_reference_used": False,
            "purpose": purpose,
        },
        portfolio_selection_eligible=production_fallback,
        execution_purpose=purpose,
    )


def _inference_only_selection(
    corrupted: MotionClip,
    objective: StressObjective,
    candidates: Sequence[_MethodCandidate],
    config: CMAOptimizationConfig,
) -> tuple[str, str]:
    """Freeze portfolio selection before privileged clean evaluation begins."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        gate = _inference_production_gate(corrupted, objective, candidate.motion, config)
        rows.append(
            {
                "method": candidate.method,
                "kind": candidate.kind,
                "diagnostic_success": bool(gate["passed"]),
                "production_success": bool(gate["passed"]),
                "non_target_constraint_reasons": gate["non_target_constraint_reasons"],
                "objective": {"after": gate["target_after"]},
                "portfolio_selection_eligible": candidate.portfolio_selection_eligible,
            }
        )
    return select_specialized_first(rows, production_claim_eligible=True)


def _optimize_methods_without_clean_reference(
    corrupted: MotionClip,
    family: str,
    output_directory: Path,
    *,
    seed: int,
    maximum_evaluations: int,
) -> tuple[StressObjective, list[_MethodCandidate], dict[str, Any]]:
    """Generate production candidates and a CMA comparator without accepting clean motion.

    A successful specialist freezes the production choice before the generic CMA run.  That CMA
    run is retained strictly as evaluation-only head-to-head evidence and cannot participate in
    portfolio selection.  When the specialist fails or is unavailable, the same generic run is
    the declared production fallback as well as the comparison baseline.
    """
    objective = deterministic_stress_objective(corrupted, family)
    config = _optimization_config(family, seed=seed)
    methods: list[_MethodCandidate] = []
    specialized = _specialized_candidate(corrupted, objective, config)
    if specialized is not None:
        methods.append(specialized)

    specialized_gate: dict[str, Any] | None = None
    if specialized is not None:
        specialized_gate = _inference_production_gate(
            corrupted,
            objective,
            specialized.motion,
            config,
        )
        if specialized_gate["passed"]:
            selected_method = specialized.method
            selection_reason = "successful_specialized_operator"
            # Selection is deliberately frozen before this comparison run.  Its output is not a
            # production candidate even if it achieves a lower deterministic objective.
            methods.append(
                _generic_cma_candidate(
                    corrupted,
                    objective,
                    family,
                    output_directory,
                    config,
                    seed=seed,
                    maximum_evaluations=maximum_evaluations,
                    production_fallback=False,
                )
            )
            return (
                objective,
                methods,
                {
                    "policy": (
                        "specialized operator first for production; deterministic CMA fallback "
                        "only on failure/unavailability; generic CMA always measured head-to-head"
                    ),
                    "specialized_available": True,
                    "specialized_gate": specialized_gate,
                    "generic_cma_invoked": True,
                    "generic_cma_invoked_for_production": False,
                    "generic_cma_evaluation_only_invoked": True,
                    "route": "specialized_success_short_circuit",
                    "selected_method_without_clean_reference": selected_method,
                    "selection_reason_without_clean_reference": selection_reason,
                    "selection_frozen_before_evaluation_only_comparison": True,
                },
            )

    methods.append(
        _generic_cma_candidate(
            corrupted,
            objective,
            family,
            output_directory,
            config,
            seed=seed,
            maximum_evaluations=maximum_evaluations,
            production_fallback=True,
        )
    )
    selected_method, selection_reason = _inference_only_selection(
        corrupted,
        objective,
        methods,
        config,
    )
    return (
        objective,
        methods,
        {
            "policy": (
                "specialized operator first for production; deterministic CMA fallback only on "
                "failure/unavailability; generic CMA always measured head-to-head"
            ),
            "specialized_available": specialized is not None,
            "specialized_gate": specialized_gate,
            "generic_cma_invoked": True,
            "generic_cma_invoked_for_production": True,
            "generic_cma_evaluation_only_invoked": False,
            "route": (
                "specialized_failed_cma_fallback"
                if specialized is not None
                else "specialized_unavailable_cma_fallback"
            ),
            "selected_method_without_clean_reference": selected_method,
            "selection_reason_without_clean_reference": selection_reason,
            "selection_frozen_before_evaluation_only_comparison": False,
        },
    )


def _acceptance_payload(acceptance: ProductionAcceptance) -> dict[str, Any]:
    return {
        "raw_metrics": acceptance.raw_metrics,
        "constraints": acceptance.constraints,
        "hard_invalid_reasons_default_contract": list(acceptance.hard_invalid_reasons),
        "all_required_satisfied_default_contract": acceptance.all_required_satisfied,
        "satisficing_penalty": acceptance.satisficing_penalty,
    }


def select_specialized_first(
    method_rows: Sequence[Mapping[str, Any]],
    *,
    production_claim_eligible: bool,
) -> tuple[str, str]:
    """Choose a successful specialized repair before falling back to deterministic CMA."""
    if not method_rows:
        raise ValueError("portfolio selection requires at least one method row")
    selectable_rows = [
        row for row in method_rows if bool(row.get("portfolio_selection_eligible", True))
    ]
    if not selectable_rows:
        raise ValueError("portfolio selection has no production-eligible method rows")
    success_key = "production_success" if production_claim_eligible else "diagnostic_success"
    specialized = [
        row
        for row in selectable_rows
        if row["kind"] == "specialized" and bool(row.get(success_key))
    ]
    if specialized:
        selected = min(specialized, key=lambda row: float(row["objective"]["after"]))
        return str(selected["method"]), "successful_specialized_operator"
    fallback = [
        row
        for row in selectable_rows
        if row["kind"] == "generic_cma" and bool(row.get(success_key))
    ]
    if fallback:
        selected = min(fallback, key=lambda row: float(row["objective"]["after"]))
        return str(selected["method"]), "specialized_unavailable_or_unsuccessful_cma_fallback"
    feasible = [row for row in selectable_rows if not row["non_target_constraint_reasons"]]
    pool = feasible or selectable_rows
    selected = min(pool, key=lambda row: float(row["objective"]["after"]))
    return str(selected["method"]), "no_method_met_success_gate_best_deterministic_candidate"


def _bands_satisfied(acceptance: ProductionAcceptance) -> bool:
    return all(
        not bool(item["required"]) or bool(item["satisfied"])
        for item in acceptance.constraints.values()
    )


def _inference_production_gate(
    original: MotionClip,
    objective: StressObjective,
    candidate: MotionClip,
    config: CMAOptimizationConfig,
) -> dict[str, Any]:
    """Decide specialized success without clean motion or subjective information."""
    before = objective.value(original)
    after = objective.value(candidate)
    non_target_reasons = _non_target_constraint_function(original, objective.family, config)(
        candidate
    )
    full_reasons = production_constraint_reasons(original, candidate, config)
    acceptance = assess_production_acceptance(original, candidate)
    target_gain = before - after
    passed = (
        target_gain >= _MINIMUM_GAIN[objective.family]
        and not non_target_reasons
        and not full_reasons
        and _bands_satisfied(acceptance)
    )
    return {
        "passed": passed,
        "clean_reference_used": False,
        "subjective_or_learned_score_used": False,
        "target_metric": objective.metric_name,
        "target_before": before,
        "target_after": after,
        "target_gain": target_gain,
        "minimum_required_gain": _MINIMUM_GAIN[objective.family],
        "non_target_constraint_reasons": non_target_reasons,
        "full_production_constraint_reasons": full_reasons,
        "production_bands_satisfied": _bands_satisfied(acceptance),
    }


def _head_to_head_payload(method_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the specialist and generic CMA using post-optimization evidence only."""
    specialized = next(
        (row for row in method_rows if row["kind"] == "specialized"),
        None,
    )
    generic = next(
        (row for row in method_rows if row["kind"] == "generic_cma"),
        None,
    )
    if specialized is None or generic is None:
        return {
            "available": False,
            "reason": (
                "specialized_candidate_unavailable"
                if specialized is None
                else "generic_cma_candidate_unavailable"
            ),
            "clean_reference_used_during_optimization": False,
        }
    specialized_after = float(specialized["objective"]["after"])
    generic_after = float(generic["objective"]["after"])
    if np.isclose(specialized_after, generic_after, rtol=1.0e-9, atol=1.0e-12):
        winner = "tie"
    elif specialized_after < generic_after:
        winner = "specialized"
    else:
        winner = "generic_cma"
    return {
        "available": True,
        "comparison_scope": "evaluation_only; lower deterministic target metric is better",
        "clean_reference_used_during_optimization": False,
        "production_selection_affected": False,
        "specialized": {
            "method": specialized["method"],
            "target_after": specialized_after,
            "target_reduction": specialized["objective"]["reduction"],
            "diagnostic_success": specialized["diagnostic_success"],
            "production_success": specialized["production_success"],
            "evaluations": specialized["evaluations"],
            "runtime_seconds": specialized["runtime_seconds"],
            "distance_from_clean_evaluation_only": specialized[
                "distance_from_clean_evaluation_only"
            ],
        },
        "generic_cma": {
            "method": generic["method"],
            "execution_purpose": generic["execution_purpose"],
            "portfolio_selection_eligible": generic["portfolio_selection_eligible"],
            "target_after": generic_after,
            "target_reduction": generic["objective"]["reduction"],
            "diagnostic_success": generic["diagnostic_success"],
            "production_success": generic["production_success"],
            "evaluations": generic["evaluations"],
            "runtime_seconds": generic["runtime_seconds"],
            "distance_from_clean_evaluation_only": generic["distance_from_clean_evaluation_only"],
        },
        "target_after_generic_minus_specialized": generic_after - specialized_after,
        "target_objective_winner": winner,
    }


def _evaluate_case(
    clean: MotionClip,
    corrupted: MotionClip,
    family: str,
    objective: StressObjective,
    candidates: Sequence[_MethodCandidate],
    routing: Mapping[str, Any],
    output_directory: Path,
    *,
    production_claim_eligible: bool,
    qualification_reasons: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Use clean motion only after every candidate has been fixed."""
    before = objective.value(corrupted)
    config = _optimization_config(family, seed=seed)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        after = objective.value(candidate.motion)
        non_target_reasons = _non_target_constraint_function(corrupted, family, config)(
            candidate.motion
        )
        full_reasons = production_constraint_reasons(
            corrupted,
            candidate.motion,
            config,
        )
        acceptance = assess_production_acceptance(corrupted, candidate.motion)
        diagnostic_success = not non_target_reasons and before - after >= _MINIMUM_GAIN[family]
        production_success = (
            production_claim_eligible
            and diagnostic_success
            and not full_reasons
            and _bands_satisfied(acceptance)
        )
        method_directory = output_directory / candidate.method
        save_motion_npz(method_directory / "candidate.npz", candidate.motion)
        row = {
            "method": candidate.method,
            "kind": candidate.kind,
            "objective": {
                "source": "deterministic_metric_only",
                "metric": objective.metric_name,
                "before": before,
                "after": after,
                "reduction": before - after,
            },
            "diagnostic_success": diagnostic_success,
            "production_success": production_success if production_claim_eligible else None,
            "non_target_constraint_reasons": non_target_reasons,
            "full_production_constraint_reasons": full_reasons,
            "production_acceptance": _acceptance_payload(acceptance),
            "metrics_after": motion_deterministic_metrics(candidate.motion),
            "distance_from_clean_evaluation_only": motion_distances(clean, candidate.motion),
            "distance_from_corrupted": motion_distances(corrupted, candidate.motion),
            "evaluations": candidate.evaluations,
            "runtime_seconds": candidate.runtime_seconds,
            "parameter_block": candidate.parameter_block,
            "parameter_count": candidate.parameter_count,
            "portfolio_selection_eligible": candidate.portfolio_selection_eligible,
            "execution_purpose": candidate.execution_purpose,
            "evidence": dict(candidate.evidence),
            "artifact": str(method_directory / "candidate.npz"),
        }
        _write_json(method_directory / "method.json", row)
        rows.append(row)
    selected_method = str(routing["selected_method_without_clean_reference"])
    reason = str(routing["selection_reason_without_clean_reference"])
    selected_candidate = next(
        (candidate for candidate in candidates if candidate.method == selected_method),
        None,
    )
    if selected_candidate is None:
        raise ValueError("pre-evaluation portfolio selection does not name a candidate")
    if not selected_candidate.portfolio_selection_eligible:
        raise ValueError("evaluation-only CMA comparison cannot be selected for production")
    selected_motion = selected_candidate.motion
    save_motion_npz(output_directory / "selected.npz", selected_motion)
    return {
        "family": family,
        "clean_reference_available_during_repair": False,
        "learned_perceptual_objective_used": False,
        "objective": {
            "metric": objective.metric_name,
            "minimum_required_gain": _MINIMUM_GAIN[family],
            "localization": dict(objective.localization or {}),
            "joint_indices": list(objective.joint_indices),
            "time_interval": (
                None if objective.time_interval is None else list(objective.time_interval)
            ),
            "uses_localized_angular_peak": objective.use_localized_angular_peak,
        },
        "production_claim_eligible": production_claim_eligible,
        "qualification_reasons": list(qualification_reasons),
        "metrics_before": motion_deterministic_metrics(corrupted),
        "distance_corrupted_from_clean_evaluation_only": motion_distances(clean, corrupted),
        "methods": rows,
        "specialized_vs_generic_cma": _head_to_head_payload(rows),
        "repair_routing": dict(routing),
        "portfolio": {
            "policy": (
                "successful specialized operator first for production; deterministic CMA is a "
                "fallback only, while its evaluation-only comparison cannot alter selection"
            ),
            "selected_method": selected_method,
            "selection_reason": reason,
            "selection_frozen_without_clean_reference": True,
            "evaluation_only_comparison_cannot_select": True,
            "selected_artifact": str(output_directory / "selected.npz"),
        },
    }


def _clean_endpoint_qualification(
    clean: MotionClip,
    family: str,
    *,
    seed: int,
    minimum_joint_limit_confidence: float,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    acceptance = assess_production_acceptance(clean, clean)
    if not _bands_satisfied(acceptance):
        reasons.extend(
            f"clean_endpoint_band:{name}"
            for name, row in acceptance.constraints.items()
            if bool(row["required"]) and not bool(row["satisfied"])
        )
    reasons.extend(
        f"clean_endpoint_constraint:{reason}"
        for reason in production_constraint_reasons(
            clean, clean, _optimization_config(family, seed=seed)
        )
    )
    if family == "loop_seam" and not clean.metadata.get("loop_kind"):
        reasons.append("source_window_has_no_trustworthy_cyclic_task_metadata")
    if family == "joint_limit":
        metric = joint_limit_metric(clean)
        if not bool(metric.metadata.get("valid")):
            reasons.append("no_valid_authored_joint_limits")
        if metric.confidence < minimum_joint_limit_confidence:
            reasons.append("joint_limit_or_local_axis_confidence_below_threshold")
    return not reasons, tuple(dict.fromkeys(reasons))


def _prior_source_exclusions(path: Path | None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {family: set() for family in STRESS_FAMILIES}
    if path is None:
        return result
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prior_sources: set[str] = set()
    for row in payload.get("selected_cases", []):
        if row.get("source_clip_id") is not None:
            prior_sources.add(str(row["source_clip_id"]))
    # "Additional styles" is global, not family-relative: a style used for a prior slide
    # trial is not new merely because this run applies a different corruption to it.
    for excluded in result.values():
        excluded.update(prior_sources)
    return result


def select_additional_repair_stress_cases(
    dataset: FixedRigSampleDataset,
    *,
    sources_per_family: int = 2,
    prior_production_report: Path | None = None,
    seed: int = 9201,
    minimum_joint_limit_confidence: float = 0.5,
) -> tuple[list[StressCaseSelection], dict[str, Any]]:
    """Select distinct styles not already used by the prior production benchmark."""
    if sources_per_family < 1:
        raise ValueError("sources_per_family must be positive")
    exclusions = _prior_source_exclusions(prior_production_report)
    selections: list[StressCaseSelection] = []
    coverage: dict[str, Any] = {}
    for family in STRESS_FAMILIES:
        by_source: dict[str, StressCaseSelection] = {}
        candidate_count = 0
        unavailable_reasons: set[str] = set()
        for index, record in enumerate(dataset.records):
            if (
                record.get("sample_role") != "hard_corruption"
                or record.get("corruption_family") != family
                or record.get("severity_label") != "moderate"
                or record.get("catalog_partition") != "train"
            ):
                continue
            source = str(record["source_clip_id"])
            if source in exclusions[family]:
                continue
            candidate_count += 1
            clean, corrupted = motions_from_sample(dataset, index)
            config = _optimization_config(family, seed=seed)
            pair = prepare_synthetic_production_benchmark(clean, corrupted, config)
            clean_prepared = pair.clean
            corrupted_prepared = pair.corrupted
            try:
                objective = deterministic_stress_objective(corrupted_prepared, family)
                margin = objective.value(corrupted_prepared) - objective.value(clean_prepared)
            except ValueError as exc:
                unavailable_reasons.add(str(exc))
                continue
            eligible, reasons = _clean_endpoint_qualification(
                clean_prepared,
                family,
                seed=seed,
                minimum_joint_limit_confidence=minimum_joint_limit_confidence,
            )
            if not pair.eligible:
                reasons = (*reasons, *pair.ineligibility_reasons)
                eligible = False
            selection = StressCaseSelection(
                family=family,
                index=index,
                sample_id=str(record["sample_id"]),
                source_clip_id=source,
                source_window_start=int(record.get("source_window_start", 0)),
                mechanism=str(record.get("corruption_mechanism", "unknown")),
                target_margin_evaluation_only=margin,
                production_claim_eligible=eligible,
                qualification_reasons=tuple(dict.fromkeys(reasons)),
            )
            current = by_source.get(source)
            if current is None or selection.target_margin_evaluation_only > (
                current.target_margin_evaluation_only
            ):
                by_source[source] = selection
        ranked = sorted(
            by_source.values(),
            key=lambda row: (
                not row.production_claim_eligible,
                -row.target_margin_evaluation_only,
                row.source_clip_id,
            ),
        )
        chosen = ranked[:sources_per_family]
        selections.extend(chosen)
        status = "selected" if chosen else "skipped"
        if family == "joint_limit" and not chosen:
            unavailable_reasons.add(
                "phase8/100STYLE fixed-rig dataset contains no authored joint-limit cases"
            )
        coverage[family] = {
            "status": status,
            "candidate_record_count_after_prior_source_exclusion": candidate_count,
            "distinct_candidate_source_count": len(by_source),
            "selected_case_count": len(chosen),
            "selected_source_clip_ids": [row.source_clip_id for row in chosen],
            "prior_source_exclusions": sorted(exclusions[family]),
            "unavailable_reasons": sorted(unavailable_reasons),
        }
    return selections, coverage


def run_deterministic_stress_case(
    clean: MotionClip,
    corrupted: MotionClip,
    family: str,
    output_directory: Path,
    *,
    seed: int = 9201,
    maximum_evaluations: int = 240,
    minimum_joint_limit_confidence: float = 0.5,
) -> dict[str, Any]:
    """Run one synthetic bad-to-clean case while isolating clean from repair code."""
    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive")
    if family not in STRESS_FAMILIES:
        raise ValueError(f"unsupported deterministic stress family {family!r}")
    config = _optimization_config(family, seed=seed)
    pair = prepare_synthetic_production_benchmark(clean, corrupted, config)
    eligible, reasons = _clean_endpoint_qualification(
        pair.clean,
        family,
        seed=seed,
        minimum_joint_limit_confidence=minimum_joint_limit_confidence,
    )
    if not pair.eligible:
        eligible = False
        reasons = tuple(dict.fromkeys((*reasons, *pair.ineligibility_reasons)))
    output = Path(output_directory)
    _require_fresh_output(output)
    inference_input = inference_available_motion(pair.corrupted)
    save_motion_npz(output / "corrupted_input.npz", inference_input)
    objective, candidates, routing = _optimize_methods_without_clean_reference(
        inference_input,
        family,
        output,
        seed=seed,
        maximum_evaluations=maximum_evaluations,
    )
    # Materialize privileged evaluation data only after the repair route and every candidate
    # on that route have been finalized.
    save_motion_npz(output / "clean_evaluation_only.npz", pair.clean)
    report = _evaluate_case(
        pair.clean,
        inference_input,
        family,
        objective,
        candidates,
        routing,
        output,
        production_claim_eligible=eligible,
        qualification_reasons=reasons,
        seed=seed,
    )
    report["format_version"] = DETERMINISTIC_REPAIR_STRESS_VERSION
    report["speed_semantics"] = {
        "task_target_speed": pair.task_target_speed,
        "clean_window_reference_speed": pair.clean_window_reference_speed,
        "parent_clip_speed": pair.parent_clip_speed,
        "target_speed": pair.target_speed,
        "target_speed_tolerance": pair.target_speed_tolerance,
        "target_speed_source": pair.target_speed_source,
    }
    _write_json(output / "stress_case.json", report)
    return report


def run_deterministic_repair_stress_test(
    dataset_directory: Path,
    output_directory: Path,
    *,
    prior_production_report: Path | None = None,
    seed: int = 9201,
    sources_per_family: int = 2,
    maximum_evaluations: int = 240,
    minimum_joint_limit_confidence: float = 0.5,
) -> dict[str, Any]:
    """Run a bounded specialized-versus-CMA stress test on additional real styles."""
    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive")
    dataset = FixedRigSampleDataset(Path(dataset_directory), regime="all", cache_samples=False)
    output = Path(output_directory)
    _require_fresh_output(output)
    output.mkdir(parents=True, exist_ok=True)
    selections, coverage = select_additional_repair_stress_cases(
        dataset,
        sources_per_family=sources_per_family,
        prior_production_report=prior_production_report,
        seed=seed,
        minimum_joint_limit_confidence=minimum_joint_limit_confidence,
    )
    runs: list[dict[str, Any]] = []
    for selection in selections:
        clean, corrupted = motions_from_sample(dataset, selection.index)
        pair = prepare_synthetic_production_benchmark(
            clean,
            corrupted,
            _optimization_config(selection.family, seed=seed),
        )
        inference_input = inference_available_motion(pair.corrupted)
        case_id = (
            f"{selection.family}__{selection.source_clip_id}__"
            f"{selection.sample_id.removeprefix('sample:sha256:')[:12]}"
        )
        case_output = output / "cases" / case_id
        save_motion_npz(case_output / "corrupted_input.npz", inference_input)
        objective, candidates, routing = _optimize_methods_without_clean_reference(
            inference_input,
            selection.family,
            case_output,
            seed=seed,
            maximum_evaluations=maximum_evaluations,
        )
        # Do not expose a filesystem clean target until candidate generation is complete.
        save_motion_npz(case_output / "clean_evaluation_only.npz", pair.clean)
        row = _evaluate_case(
            pair.clean,
            inference_input,
            selection.family,
            objective,
            candidates,
            routing,
            case_output,
            production_claim_eligible=selection.production_claim_eligible,
            qualification_reasons=selection.qualification_reasons,
            seed=seed,
        )
        row.update(
            {
                "case_id": case_id,
                "sample_id": selection.sample_id,
                "source_clip_id": selection.source_clip_id,
                "source_window_start": selection.source_window_start,
                "mechanism": selection.mechanism,
                "target_margin_used_for_curation_evaluation_only": (
                    selection.target_margin_evaluation_only
                ),
                "speed_semantics": {
                    "task_target_speed": pair.task_target_speed,
                    "clean_window_reference_speed": pair.clean_window_reference_speed,
                    "parent_clip_speed": pair.parent_clip_speed,
                    "target_speed": pair.target_speed,
                    "target_speed_tolerance": pair.target_speed_tolerance,
                    "target_speed_source": pair.target_speed_source,
                },
            }
        )
        _write_json(case_output / "stress_case.json", row)
        runs.append(row)
        _write_json(
            output / "deterministic_repair_stress.json",
            {
                "format_version": DETERMINISTIC_REPAIR_STRESS_VERSION,
                "status": "in_progress",
                "runs": runs,
                "coverage": coverage,
            },
        )

    by_family: dict[str, Any] = {}
    for family in STRESS_FAMILIES:
        rows = [row for row in runs if row["family"] == family]
        eligible = [row for row in rows if row["production_claim_eligible"]]
        selected_methods = [row["portfolio"]["selected_method"] for row in rows]
        production_successes = [
            next(
                method["production_success"]
                for method in row["methods"]
                if method["method"] == row["portfolio"]["selected_method"]
            )
            for row in eligible
        ]
        specialized_rows = [
            method for row in rows for method in row["methods"] if method["kind"] == "specialized"
        ]
        cma_rows = [
            method for row in rows for method in row["methods"] if method["kind"] == "generic_cma"
        ]
        comparisons = [
            row["specialized_vs_generic_cma"]
            for row in rows
            if row["specialized_vs_generic_cma"]["available"]
        ]
        eligible_specialized = [
            method
            for row in eligible
            for method in row["methods"]
            if method["kind"] == "specialized"
        ]
        eligible_cma = [
            method
            for row in eligible
            for method in row["methods"]
            if method["kind"] == "generic_cma"
        ]
        by_family[family] = {
            **coverage[family],
            "executed_case_count": len(rows),
            "production_eligible_case_count": len(eligible),
            "production_success_count": sum(bool(value) for value in production_successes),
            "production_success_rate": (
                None
                if not production_successes
                else sum(bool(value) for value in production_successes) / len(production_successes)
            ),
            "portfolio_selected_method_counts": {
                method: selected_methods.count(method) for method in sorted(set(selected_methods))
            },
            "median_specialized_target_after": (
                None
                if not specialized_rows
                else float(np.median([row["objective"]["after"] for row in specialized_rows]))
            ),
            "median_cma_target_after": (
                None
                if not cma_rows
                else float(np.median([row["objective"]["after"] for row in cma_rows]))
            ),
            "median_cma_evaluations": (
                None if not cma_rows else float(np.median([row["evaluations"] for row in cma_rows]))
            ),
            "median_cma_runtime_seconds": (
                None
                if not cma_rows
                else float(np.median([row["runtime_seconds"] for row in cma_rows]))
            ),
            "generic_cma_executed_case_count": len(cma_rows),
            "head_to_head_case_count": len(comparisons),
            "head_to_head_target_objective_wins": {
                winner: sum(
                    comparison["target_objective_winner"] == winner for comparison in comparisons
                )
                for winner in ("specialized", "generic_cma", "tie")
            },
            "specialized_production_success_count_evaluation_only": sum(
                bool(method["production_success"]) for method in eligible_specialized
            ),
            "generic_cma_production_success_count_evaluation_only": sum(
                bool(method["production_success"]) for method in eligible_cma
            ),
            "generic_cma_execution_purpose_counts": {
                purpose: sum(row["execution_purpose"] == purpose for row in cma_rows)
                for purpose in sorted({str(row["execution_purpose"]) for row in cma_rows})
            },
        }

    eligible_selected = [
        next(
            method
            for method in row["methods"]
            if method["method"] == row["portfolio"]["selected_method"]
        )
        for row in runs
        if row["production_claim_eligible"]
    ]
    all_method_rows = [method for row in runs for method in row["methods"]]
    all_cma_rows = [method for method in all_method_rows if method["kind"] == "generic_cma"]
    all_specialized_rows = [method for method in all_method_rows if method["kind"] == "specialized"]
    production_route_rows = [
        method
        for method in all_method_rows
        if method["kind"] == "specialized"
        or method["execution_purpose"] == "production_fallback_and_evaluation_head_to_head"
    ]
    evaluation_only_overhead_rows = [
        method
        for method in all_method_rows
        if method["execution_purpose"] == "evaluation_only_head_to_head"
    ]
    comparisons = [
        row["specialized_vs_generic_cma"]
        for row in runs
        if row["specialized_vs_generic_cma"]["available"]
    ]
    if len(all_cma_rows) != len(runs):
        raise RuntimeError("generic CMA baseline must execute exactly once for every stress case")
    if len(comparisons) != len(runs):
        raise RuntimeError(
            "every executed stress case must have a specialized-versus-CMA comparison"
        )
    report = {
        "format_version": DETERMINISTIC_REPAIR_STRESS_VERSION,
        "status": "complete",
        "scope": "additional 100STYLE-derived fixed-rig windows",
        "label_independent": True,
        "active_human_observations_read": False,
        "learned_perceptual_objective_used": False,
        "optimizer": "CMA-ES",
        "portfolio_policy": (
            "successful specialized operator first for production; deterministic CMA fallback "
            "only after failure/unavailability; generic CMA head-to-head runs never alter "
            "production selection"
        ),
        "generic_cma_head_to_head_contract": {
            "executed_for_every_case": len(all_cma_rows) == len(runs),
            "generic_cma_executed_case_count": len(all_cma_rows),
            "specialized_comparison_case_count": len(comparisons),
            "executed_case_count": len(runs),
            "clean_reference_used_during_optimization": False,
            "evaluation_only_runs_can_affect_portfolio_selection": False,
            "fallback_runs_double_as_evaluation_baselines": True,
            "target_objective_wins": {
                winner: sum(
                    comparison["target_objective_winner"] == winner for comparison in comparisons
                )
                for winner in ("specialized", "generic_cma", "tie")
            },
        },
        "clean_reference_contract": (
            "withheld until all repair candidates are complete; evaluation and benchmark "
            "qualification only"
        ),
        "seed": seed,
        "maximum_evaluations_per_cma_case": maximum_evaluations,
        "prior_production_report": (
            None if prior_production_report is None else str(prior_production_report)
        ),
        "selected_case_count": len(selections),
        "executed_case_count": len(runs),
        "production_eligible_case_count": len(eligible_selected),
        "production_success_count": sum(
            bool(row["production_success"]) for row in eligible_selected
        ),
        "production_success_rate": (
            None
            if not eligible_selected
            else sum(bool(row["production_success"]) for row in eligible_selected)
            / len(eligible_selected)
        ),
        "method_execution_totals": {
            "production_route": {
                "method_attempt_count": len(production_route_rows),
                "evaluations": sum(int(row["evaluations"]) for row in production_route_rows),
                "runtime_seconds": sum(
                    float(row["runtime_seconds"]) for row in production_route_rows
                ),
                "includes": "all specialist attempts plus the five required CMA fallbacks",
            },
            "evaluation_only_baseline_overhead": {
                "method_attempt_count": len(evaluation_only_overhead_rows),
                "evaluations": sum(
                    int(row["evaluations"]) for row in evaluation_only_overhead_rows
                ),
                "runtime_seconds": sum(
                    float(row["runtime_seconds"]) for row in evaluation_only_overhead_rows
                ),
                "can_affect_production_selection": False,
            },
            "all_executed_methods": {
                "method_attempt_count": len(all_method_rows),
                "evaluations": sum(int(row["evaluations"]) for row in all_method_rows),
                "runtime_seconds": sum(float(row["runtime_seconds"]) for row in all_method_rows),
            },
            "specialized": {
                "case_count": len(all_specialized_rows),
                "evaluations": sum(int(row["evaluations"]) for row in all_specialized_rows),
                "runtime_seconds": sum(
                    float(row["runtime_seconds"]) for row in all_specialized_rows
                ),
            },
            "generic_cma": {
                "case_count": len(all_cma_rows),
                "evaluations": sum(int(row["evaluations"]) for row in all_cma_rows),
                "runtime_seconds": sum(float(row["runtime_seconds"]) for row in all_cma_rows),
                "evaluation_only_comparator_count": sum(
                    row["execution_purpose"] == "evaluation_only_head_to_head"
                    for row in all_cma_rows
                ),
                "production_fallback_count": sum(
                    row["execution_purpose"] == "production_fallback_and_evaluation_head_to_head"
                    for row in all_cma_rows
                ),
            },
        },
        "coverage": coverage,
        "by_family": by_family,
        "runs": runs,
        "limitations": [
            (
                "Phase8/100STYLE carries no authored trustworthy joint-limit metadata, so "
                "joint-limit repair is gated out instead of fabricating limits."
            ),
            (
                "Phase8 loop-seam examples are ordinary windows without a declared cyclic task; "
                "their results are diagnostic comparisons and not production-success claims."
            ),
            (
                "The speed benchmark repairs root progression with the compact SpeedCadenceBlock; "
                "it makes no standalone cadence-success claim because the dataset has no explicit "
                "cadence task target, and source-relative leg-style amplitude has no "
                "inference-available target metric."
            ),
        ],
    }
    _write_json(output / "deterministic_repair_stress.json", report)
    return report
