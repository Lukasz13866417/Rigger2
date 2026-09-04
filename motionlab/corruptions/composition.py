"""Bounded sequential composition of independently measured hard negatives."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from motionlab.core.provenance import provenance_from_clip
from motionlab.corruptions._supervision import measure_contact_supervision, resolve_ground_plane
from motionlab.corruptions.base import (
    CORRUPTION_DEFECT_NAMES,
    CorruptionResult,
    IneffectiveCorruptionError,
    MeasuredPostcondition,
)
from motionlab.kinematics.fk import marker_world_positions
from motionlab.math.quaternion import quaternion_geodesic_distance
from motionlab.metrics.foot_sliding import foot_sliding_metric
from motionlab.metrics.ground import ground_metrics
from motionlab.metrics.joint_limits import joint_limit_metric
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.metrics.speed import speed_metric
from motionlab.motion.clip import MotionClip

CorruptionOperator = Callable[[MotionClip], CorruptionResult]


def _active_frames(result: CorruptionResult) -> NDArray[np.bool_]:
    if result.corruption_family not in CORRUPTION_DEFECT_NAMES:
        return np.asarray(np.any(result.intervention_mask, axis=(1, 2)), dtype=np.bool_)
    defect = CORRUPTION_DEFECT_NAMES.index(result.corruption_family)
    return np.asarray(np.any(result.symptom_mask[:, :, defect], axis=1), dtype=np.bool_)


def _active_joints(result: CorruptionResult) -> NDArray[np.int64]:
    return np.asarray(
        np.flatnonzero(np.any(result.intervention_mask, axis=(0, 2))),
        dtype=np.int64,
    )


def _rotation_statistic(
    clean: MotionClip,
    final: MotionClip,
    result: CorruptionResult,
    *,
    percentile: float,
) -> tuple[float, float]:
    joints = _active_joints(result)
    frames = _active_frames(result)
    if joints.size == 0 or not np.any(frames):
        raise ValueError("constituent contains no active rotation support")
    before_values = np.asarray(smoothness_metric(clean).joint_frame_values, dtype=np.float64)
    after_values = np.asarray(smoothness_metric(final).joint_frame_values, dtype=np.float64)
    before = float(np.percentile(before_values[frames][:, joints], percentile))
    after = float(np.percentile(after_values[frames][:, joints], percentile))
    return before, after


def _final_constituent_measurement(
    clean: MotionClip,
    final: MotionClip,
    result: CorruptionResult,
) -> tuple[float, float]:
    family = result.corruption_family
    if family in {"foot_slide", "ground_penetration", "floating_contact"}:
        _, fixed_clean, fixed_final = measure_contact_supervision(clean, final)
        if family == "foot_slide":
            return (
                float(foot_sliding_metric(clean, fixed_clean).clip_value or 0.0),
                float(foot_sliding_metric(final, fixed_final).clip_value or 0.0),
            )
        clean_ground = ground_metrics(fixed_clean)
        final_ground = ground_metrics(fixed_final)
        index = 0 if family == "ground_penetration" else 1
        return (
            float(clean_ground[index].clip_value or 0.0),
            float(final_ground[index].clip_value or 0.0),
        )
    if family == "loop_seam":
        return (
            float(loop_seam_metric(clean).clip_value or 0.0),
            float(loop_seam_metric(final).clip_value or 0.0),
        )
    if family == "joint_pop":
        return _rotation_statistic(clean, final, result, percentile=100.0)
    if family == "joint_jitter":
        return _rotation_statistic(clean, final, result, percentile=95.0)
    if family == "speed_inconsistency" and result.corruption_mechanism == "root_progression_scale":
        plane = resolve_ground_plane(clean, None)
        target_raw = clean.metadata.get("expected_speed_mps")
        baseline = speed_metric(clean, ground_plane=plane)
        target = float(baseline.clip_value or 0.0) if target_raw is None else float(target_raw)
        before = speed_metric(clean, ground_plane=plane, target_speed_mps=target)
        after = speed_metric(final, ground_plane=plane, target_speed_mps=target)
        return (
            float(before.metadata["absolute_target_error_mps"] or 0.0),
            float(after.metadata["absolute_target_error_mps"] or 0.0),
        )
    if family in {"limb_phase_mismatch", "cadence_inconsistency", "speed_inconsistency"}:
        frames = _active_frames(result)
        if result.corruption_mechanism == "root_stream_monotonic_warp":
            delta = final.root_translation_m - clean.root_translation_m
            return 0.0, 100.0 * float(np.sqrt(np.mean(np.square(delta[frames]))))
        joints = _active_joints(result)
        deviation = quaternion_geodesic_distance(
            clean.local_quat_wxyz[:, joints], final.local_quat_wxyz[:, joints]
        )
        return 0.0, float(np.sqrt(np.mean(np.square(deviation[frames]))))
    if family == "foot_clearance":
        frames = _active_frames(result)
        side = str(result.generation_parameters["side"])
        marker_names = tuple(name for name in clean.skeleton.markers if name.startswith(f"{side}_"))
        plane = resolve_ground_plane(clean, None)
        clean_markers = marker_world_positions(clean, marker_names)
        final_markers = marker_world_positions(final, marker_names)
        losses = [
            (clean_markers[name] @ plane[:3] - final_markers[name] @ plane[:3])[frames]
            for name in marker_names
        ]
        return 0.0, 100.0 * float(np.max(np.stack(losses), initial=0.0))
    if family == "joint_limit":
        limit_before = joint_limit_metric(clean).clip_value
        limit_after = joint_limit_metric(final).clip_value
        if limit_before is None or limit_after is None:
            raise ValueError("joint-limit metric became unavailable during composition")
        return float(limit_before), float(limit_after)
    raise ValueError(f"unsupported composition constituent family {family!r}")


def compose_corruptions(
    clean: MotionClip,
    operators: Sequence[CorruptionOperator],
    *,
    seed: int,
    maximum_constituents: int = 2,
) -> CorruptionResult:
    """Apply up to a bounded number of training mechanisms and union their dense labels."""
    if maximum_constituents < 2 or maximum_constituents > 3:
        raise ValueError("maximum_constituents must be two or three")
    if len(operators) < 2 or len(operators) > maximum_constituents:
        raise ValueError("composition must contain two through maximum_constituents operators")
    results: list[CorruptionResult] = []
    current = clean
    for operator in operators:
        result = operator(current)
        if result.clean_motion.content_hash != current.content_hash:
            raise ValueError("composition operator returned a result for a different input motion")
        if result.catalog_partition != "train":
            raise ValueError("held-out corruption mechanisms cannot enter training compositions")
        if not result.postcondition.hard_negative:
            raise ValueError("composition constituents must be measured hard negatives")
        if result.corruption_family == "composite":
            raise ValueError("nested corruption compositions are not supported")
        results.append(result)
        current = result.corrupted_motion

    intervention = np.logical_or.reduce([result.intervention_mask for result in results])
    symptom = np.logical_or.reduce([result.symptom_mask for result in results])
    responsibility = np.maximum.reduce([result.responsibility_target for result in results])
    responsibility_confidence = np.maximum.reduce(
        [result.responsibility_confidence for result in results]
    )
    final_measurements = [
        _final_constituent_measurement(clean, current, result) for result in results
    ]
    normalized_margins = []
    for result, (before, after) in zip(results, final_measurements, strict=True):
        margin = (after - before) / result.postcondition.minimum_worsening
        if margin < 1.0:
            raise IneffectiveCorruptionError(
                f"composition erased {result.corruption_family}/{result.corruption_mechanism}: "
                f"before={before:g}, final={after:g}"
            )
        normalized_margins.append(margin)
    minimum_margin = float(min(normalized_margins))
    postcondition = MeasuredPostcondition(
        metric_name="minimum_normalized_constituent_worsening",
        before=0.0,
        after=minimum_margin,
        minimum_worsening=1.0,
        passed=minimum_margin >= 1.0,
        hard_negative=minimum_margin >= 1.0,
    )
    constituents = [
        {
            "family": result.corruption_family,
            "mechanism": result.corruption_mechanism,
            "seed": result.corruption_seed,
            "requested_severity_parameter": result.requested_severity_parameter,
            "metric_name": result.postcondition.metric_name,
            "metric_before": final_measurements[index][0],
            "metric_after": final_measurements[index][1],
            "metric_delta": final_measurements[index][1] - final_measurements[index][0],
            "minimum_worsening": result.postcondition.minimum_worsening,
        }
        for index, result in enumerate(results)
    ]
    metric_before = {
        f"constituent_{index}:{result.postcondition.metric_name}": final_measurements[index][0]
        for index, result in enumerate(results)
    }
    metric_after = {
        f"constituent_{index}:{result.postcondition.metric_name}": final_measurements[index][1]
        for index, result in enumerate(results)
    }
    try:
        contact_supervision, _, _ = measure_contact_supervision(clean, current)
    except ValueError:
        contact_supervision = None
    provenance = provenance_from_clip(clean)
    return CorruptionResult(
        corrupted_motion=current,
        clean_motion=clean,
        corruption_family="composite",
        corruption_mechanism="+".join(
            f"{result.corruption_family}/{result.corruption_mechanism}" for result in results
        ),
        intervention_mask=intervention,
        symptom_mask=symptom,
        responsibility_target=responsibility,
        responsibility_confidence=responsibility_confidence,
        measured_metrics_before=metric_before,
        measured_metrics_after=metric_after,
        measured_severity=minimum_margin,
        preference_confidence=min(result.preference_confidence for result in results),
        source_motion_id=clean.content_hash,
        corruption_seed=seed,
        split_lineage_id=provenance.split_lineage_id,
        requested_severity_parameter=float(len(results)),
        generation_parameters={
            "composition_kind": "sequential",
            "constituents": constituents,
            "maximum_constituents": maximum_constituents,
        },
        postcondition=postcondition,
        contact_supervision=contact_supervision,
        catalog_partition="train",
        schema_metadata={
            "intervention_merge": "logical_or",
            "symptom_merge": "logical_or_by_defect",
            "responsibility_merge": "elementwise_max",
            "constituents_remeasured_on_final_composition": True,
            "fine_joint_causal_blame_is_supervised": False,
        },
    )
