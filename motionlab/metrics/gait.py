"""Confidence-aware gait timing and geometry measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from motionlab.kinematics.fk import marker_world_positions
from motionlab.metrics.base import MetricResult
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult
from motionlab.processing.phase import GaitEvent, extract_gait_events


@dataclass(frozen=True)
class _Sample:
    frame: int
    value: float
    confidence: float
    side: str | None = None


def _unambiguous_strikes(contacts: ContactResult) -> tuple[GaitEvent, ...]:
    strikes = [event for event in extract_gait_events(contacts) if event.kind == "heel_strike"]
    sides_by_frame: dict[int, set[str]] = {}
    for event in strikes:
        sides_by_frame.setdefault(event.frame, set()).add(event.side)
    return tuple(event for event in strikes if len(sides_by_frame[event.frame]) == 1)


def _sample_metric(
    *,
    name: str,
    units: str,
    frame_count: int,
    samples: Sequence[_Sample],
    ground_confidence: float,
    structure_confidence: float,
    description: str,
) -> MetricResult:
    dense = np.full(frame_count, np.nan, dtype=np.float64)
    for sample in samples:
        dense[sample.frame] = sample.value
    side_summary: dict[str, dict[str, float | int | None]] = {}
    for side in ("left", "right"):
        values = [sample.value for sample in samples if sample.side == side]
        side_summary[side] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
        }
    sample_confidence = (
        float(np.mean([sample.confidence for sample in samples])) if samples else 0.0
    )
    return MetricResult(
        name=name,
        clip_value=float(np.mean([sample.value for sample in samples])) if samples else None,
        frame_values=dense.astype(float).tolist(),
        units=units,
        confidence=sample_confidence * ground_confidence * structure_confidence,
        metadata={
            "description": description,
            "sample_count": len(samples),
            "side_summary": side_summary,
            "dense_missing_value": "NaN",
        },
    )


def _relative_asymmetry_percent(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = 0.5 * (abs(left) + abs(right))
    if denominator <= 1.0e-12:
        return 0.0
    return 100.0 * abs(left - right) / denominator


def gait_metrics(
    clip: MotionClip,
    contacts: ContactResult,
) -> tuple[MetricResult, MetricResult, MetricResult, MetricResult, MetricResult]:
    """Measure cadence, stride timing/length, step width, and bilateral asymmetry.

    Spatial quantities are measured on the configured ground plane. Forward is inferred from the
    clip's net plane-tangential root displacement, so spatial metrics are unavailable for clips
    without a stable travel direction while temporal metrics remain valid.
    """
    if contacts.hard.shape[0] != clip.num_frames:
        raise ValueError("contact frame count must match clip")
    strikes = _unambiguous_strikes(contacts)
    alternations = sum(current.side != previous.side for previous, current in pairwise(strikes))
    structure_confidence = alternations / (len(strikes) - 1) if len(strikes) >= 2 else 0.0

    cadence_samples: list[_Sample] = []
    for previous, current in pairwise(strikes):
        elapsed_s = (current.frame - previous.frame) / clip.fps
        if elapsed_s > 0.0:
            cadence_samples.append(
                _Sample(
                    frame=current.frame,
                    value=60.0 / elapsed_s,
                    confidence=min(previous.confidence, current.confidence),
                    side=current.side,
                )
            )

    stride_time_samples: list[_Sample] = []
    by_side = {
        side: [event for event in strikes if event.side == side] for side in ("left", "right")
    }
    for side, side_strikes in by_side.items():
        for previous, current in pairwise(side_strikes):
            stride_time_samples.append(
                _Sample(
                    frame=current.frame,
                    value=(current.frame - previous.frame) / clip.fps,
                    confidence=min(previous.confidence, current.confidence),
                    side=side,
                )
            )

    normal = np.asarray(contacts.ground_plane[:3], dtype=np.float64)
    travel = np.asarray(clip.root_translation_m[-1] - clip.root_translation_m[0], dtype=np.float64)
    travel -= np.dot(travel, normal) * normal
    travel_distance = float(np.linalg.norm(travel))
    spatial_valid = travel_distance > 1.0e-6
    stride_length_samples: list[_Sample] = []
    step_width_samples: list[_Sample] = []
    if spatial_valid:
        forward = travel / travel_distance
        lateral = np.cross(normal, forward)
        lateral /= np.linalg.norm(lateral)
        positions = marker_world_positions(
            clip,
            ("left_heel", "left_toe", "right_heel", "right_toe"),
        )
        foot_centers = {
            side: 0.5 * (positions[f"{side}_heel"] + positions[f"{side}_toe"])
            for side in ("left", "right")
        }
        for side, side_strikes in by_side.items():
            for previous, current in pairwise(side_strikes):
                displacement = (
                    foot_centers[side][current.frame] - foot_centers[side][previous.frame]
                )
                stride_length_samples.append(
                    _Sample(
                        frame=current.frame,
                        value=abs(float(np.dot(displacement, forward))),
                        confidence=min(previous.confidence, current.confidence),
                        side=side,
                    )
                )
        for strike in strikes:
            separation = foot_centers["left"][strike.frame] - foot_centers["right"][strike.frame]
            step_width_samples.append(
                _Sample(
                    frame=strike.frame,
                    value=abs(float(np.dot(separation, lateral))),
                    confidence=strike.confidence,
                    side=strike.side,
                )
            )

    cadence = _sample_metric(
        name="cadence",
        units="steps_per_minute",
        samples=cadence_samples,
        description="Mean reciprocal interval between consecutive unambiguous heel strikes.",
        frame_count=clip.num_frames,
        ground_confidence=contacts.ground_confidence,
        structure_confidence=structure_confidence,
    )
    stride_time = _sample_metric(
        name="stride_time",
        units="seconds",
        samples=stride_time_samples,
        description="Time between consecutive heel strikes of the same foot.",
        frame_count=clip.num_frames,
        ground_confidence=contacts.ground_confidence,
        structure_confidence=structure_confidence,
    )
    stride_length = _sample_metric(
        name="stride_length",
        units="meters",
        samples=stride_length_samples,
        description="Same-foot heel-strike displacement along inferred travel direction.",
        frame_count=clip.num_frames,
        ground_confidence=contacts.ground_confidence,
        structure_confidence=structure_confidence,
    )
    step_width = _sample_metric(
        name="step_width",
        units="meters",
        samples=step_width_samples,
        description="Lateral distance between foot centers at each heel strike.",
        frame_count=clip.num_frames,
        ground_confidence=contacts.ground_confidence,
        structure_confidence=structure_confidence,
    )

    component_asymmetry: dict[str, float | None] = {}
    for name, metric in (
        ("stride_time", stride_time),
        ("stride_length", stride_length),
        ("step_width", step_width),
    ):
        side_summary = metric.metadata["side_summary"]
        assert isinstance(side_summary, dict)
        left = side_summary["left"]["mean"]
        right = side_summary["right"]["mean"]
        component_asymmetry[name] = _relative_asymmetry_percent(left, right)
    available_asymmetry = [value for value in component_asymmetry.values() if value is not None]
    contributing_confidence = [
        metric.confidence
        for metric in (stride_time, stride_length, step_width)
        if metric.clip_value is not None
    ]
    asymmetry = MetricResult(
        name="gait_asymmetry",
        clip_value=(float(np.mean(available_asymmetry)) if available_asymmetry else None),
        units="percent",
        confidence=(float(np.mean(contributing_confidence)) if contributing_confidence else 0.0),
        metadata={
            "definition": "absolute left-right difference divided by bilateral mean",
            "components_percent": component_asymmetry,
            "spatial_valid": spatial_valid,
            "travel_distance_m": travel_distance,
        },
    )
    return cadence, stride_time, stride_length, step_width, asymmetry
