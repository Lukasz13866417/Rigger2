"""Explicit task and reference semantics for synthetic production-repair benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from motionlab.metrics.speed import speed_metric
from motionlab.motion.clip import MotionClip
from motionlab.optimization.cmaes import CMAOptimizationConfig, production_constraint_reasons

BENCHMARK_SEMANTICS_VERSION = "motionlab.production_benchmark_semantics.v1"


@dataclass(frozen=True)
class ProductionBenchmarkPair:
    """A bad-to-clean pair with an explicit task target and eligibility decision."""

    clean: MotionClip
    corrupted: MotionClip
    task_target_speed: float | None
    clean_window_reference_speed: float
    parent_clip_speed: float | None
    target_speed: float
    target_speed_tolerance: float
    target_speed_source: str
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    clean_endpoint_constraint_reasons: tuple[str, ...]

    def assert_clean_endpoint_feasible(self) -> None:
        """Enforce the defining invariant for every eligible synthetic benchmark."""
        if self.eligible and self.clean_endpoint_constraint_reasons:
            raise AssertionError(
                "eligible synthetic production benchmark has an infeasible clean endpoint"
            )


def _optional_nonnegative(value: object, name: str) -> float | None:
    if value is None:
        return None
    number = float(cast(Any, value))
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _with_speed_semantics(
    clip: MotionClip,
    *,
    task_target_speed: float | None,
    clean_window_reference_speed: float,
    parent_clip_speed: float | None,
    target_speed: float,
    tolerance: float,
    source: str,
) -> MotionClip:
    metadata = dict(clip.metadata)
    metadata.update(
        {
            "production_benchmark_semantics_version": BENCHMARK_SEMANTICS_VERSION,
            "task_target_speed": task_target_speed,
            "clean_window_reference_speed": clean_window_reference_speed,
            "parent_clip_speed": parent_clip_speed,
            "target_speed": target_speed,
            "target_speed_tolerance": tolerance,
            "target_speed_source": source,
            "speed_units": "m/s",
            # Compatibility aliases are synchronized to the explicit benchmark target.
            "target_speed_mps": target_speed,
            "expected_speed_mps": target_speed,
        }
    )
    return clip.with_updates(metadata=metadata)


def prepare_synthetic_production_benchmark(
    clean: MotionClip,
    corrupted: MotionClip,
    config: CMAOptimizationConfig,
    *,
    task_target_speed: float | None = None,
    target_speed_tolerance: float | None = None,
) -> ProductionBenchmarkPair:
    """Declare speed semantics and reject pairs whose clean endpoint is infeasible."""
    if clean.num_frames != corrupted.num_frames:
        raise ValueError("clean and corrupted benchmark clips must have matching frame counts")
    if clean.skeleton.content_hash != corrupted.skeleton.content_hash:
        raise ValueError("clean and corrupted benchmark clips must share a skeleton")
    explicit_task = _optional_nonnegative(
        task_target_speed
        if task_target_speed is not None
        else clean.metadata.get("task_target_speed"),
        "task_target_speed",
    )
    tolerance = (
        config.production_speed_error_mps
        if target_speed_tolerance is None
        else float(target_speed_tolerance)
    )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("target_speed_tolerance must be finite and nonnegative")
    clean_speed_value = speed_metric(clean).clip_value
    if clean_speed_value is None:
        raise ValueError("clean-window speed is unavailable")
    clean_speed = float(clean_speed_value)
    parent_speed = _optional_nonnegative(
        clean.metadata.get("parent_clip_speed", clean.metadata.get("expected_speed_mps")),
        "parent_clip_speed",
    )
    target = clean_speed if explicit_task is None else explicit_task
    source = "measured_clean_window_speed" if explicit_task is None else "external_task_speed"
    prepared_clean = _with_speed_semantics(
        clean,
        task_target_speed=explicit_task,
        clean_window_reference_speed=clean_speed,
        parent_clip_speed=parent_speed,
        target_speed=target,
        tolerance=tolerance,
        source=source,
    )
    prepared_corrupted = _with_speed_semantics(
        corrupted,
        task_target_speed=explicit_task,
        clean_window_reference_speed=clean_speed,
        parent_clip_speed=parent_speed,
        target_speed=target,
        tolerance=tolerance,
        source=source,
    )
    clean_reasons = tuple(production_constraint_reasons(prepared_corrupted, prepared_clean, config))
    reasons: list[str] = []
    if explicit_task is not None and abs(clean_speed - explicit_task) > tolerance:
        reasons.append("clean_reference_misses_external_task_speed")
    reasons.extend(reason for reason in clean_reasons if reason not in reasons)
    pair = ProductionBenchmarkPair(
        clean=prepared_clean,
        corrupted=prepared_corrupted,
        task_target_speed=explicit_task,
        clean_window_reference_speed=clean_speed,
        parent_clip_speed=parent_speed,
        target_speed=target,
        target_speed_tolerance=tolerance,
        target_speed_source=source,
        eligible=not reasons,
        ineligibility_reasons=tuple(reasons),
        clean_endpoint_constraint_reasons=clean_reasons,
    )
    pair.assert_clean_endpoint_feasible()
    return pair
