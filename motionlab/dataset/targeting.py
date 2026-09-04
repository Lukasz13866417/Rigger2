"""Observed-symptom severity bins and deterministic corruptor-parameter search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from motionlab.corruptions.base import CorruptionResult, IneffectiveCorruptionError
from motionlab.motion.clip import MotionClip

TargetGenerator = Callable[[MotionClip, float, int, bool], CorruptionResult]


class SeverityTargetingError(ValueError):
    """Raised when bounded search cannot populate an observed-severity bin."""


@dataclass(frozen=True)
class ObservedSeverityBin:
    """One family/metric-specific half-open interval of measured symptom delta."""

    label: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("severity-bin label must be nonempty")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("severity-bin bounds must be finite")
        if self.lower < 0.0 or self.upper <= self.lower:
            raise ValueError("severity-bin bounds must satisfy 0 <= lower < upper")

    @property
    def target(self) -> float:
        """Return the midpoint used by the bounded parameter search."""
        return 0.5 * (self.lower + self.upper)

    def contains(self, measured_delta: float) -> bool:
        """Return whether a measured symptom delta belongs to this bin."""
        return self.lower <= measured_delta < self.upper


@dataclass(frozen=True)
class ObservedSeverityScale:
    """Five ordered bins tied to one physical diagnostic metric."""

    metric_name: str
    bins: tuple[ObservedSeverityBin, ...]

    def __post_init__(self) -> None:
        if not self.metric_name.strip():
            raise ValueError("observed-severity metric name must be nonempty")
        if tuple(item.label for item in self.bins) != (
            "near_threshold",
            "subtle",
            "moderate",
            "clear",
            "severe",
        ):
            raise ValueError("observed-severity scales require the five fixed ordered labels")
        for previous, current in zip(self.bins, self.bins[1:], strict=False):
            if not np.isclose(previous.upper, current.lower, atol=1.0e-12, rtol=0.0):
                raise ValueError("observed-severity bins must be contiguous")


@dataclass(frozen=True)
class TargetedCorruption:
    """A generated hard negative and transparent bounded-search diagnostics."""

    result: CorruptionResult
    severity_bin: ObservedSeverityBin
    selected_parameter: float
    evaluated_parameters: tuple[float, ...]


def _candidate(
    clip: MotionClip,
    generator: TargetGenerator,
    parameter: float,
    seed: int,
) -> CorruptionResult | None:
    try:
        return generator(clip, parameter, seed, False)
    except IneffectiveCorruptionError:
        return None


def _with_targeting_metadata(
    result: CorruptionResult,
    *,
    severity_bin: ObservedSeverityBin,
    selected_parameter: float,
    evaluated_parameters: tuple[float, ...],
    strategy: str,
) -> CorruptionResult:
    targeting = {
        "definition": "measured_postcondition_delta",
        "metric_name": result.postcondition.metric_name,
        "bin_label": severity_bin.label,
        "bin_lower_inclusive": severity_bin.lower,
        "bin_upper_exclusive": severity_bin.upper,
        "target_midpoint": severity_bin.target,
        "observed_delta": result.measured_severity,
        "selected_corruptor_parameter": selected_parameter,
        "search_strategy": strategy,
        "evaluated_parameters": list(evaluated_parameters),
    }
    return replace(
        result,
        generation_parameters={
            **dict(result.generation_parameters),
            "severity_targeting": targeting,
        },
        schema_metadata={**dict(result.schema_metadata), "severity_targeting": targeting},
    )


def target_observed_severity(
    clip: MotionClip,
    *,
    generator: TargetGenerator,
    severity_scale: ObservedSeverityScale,
    severity_bin: ObservedSeverityBin,
    parameter_bounds: tuple[float, float],
    seed: int,
    maximum_evaluations: int = 14,
    strategy: Literal["monotone_binary", "bounded_grid"] = "monotone_binary",
    parameter_quantum: float | None = None,
) -> TargetedCorruption:
    """Tune a mechanism parameter until its independently measured delta is in the target bin."""
    lower_parameter, upper_parameter = parameter_bounds
    if (
        not np.isfinite(lower_parameter)
        or not np.isfinite(upper_parameter)
        or lower_parameter < 0.0
        or upper_parameter <= lower_parameter
    ):
        raise ValueError("parameter bounds must satisfy 0 <= lower < upper")
    if severity_bin not in severity_scale.bins:
        raise ValueError("severity bin does not belong to the supplied scale")
    if maximum_evaluations < 4:
        raise ValueError("observed-severity search requires at least four evaluations")
    if parameter_quantum is not None and (
        not np.isfinite(parameter_quantum) or parameter_quantum <= 0.0
    ):
        raise ValueError("parameter_quantum must be finite and positive")

    evaluated: dict[float, CorruptionResult | None] = {}

    def evaluate(raw_parameter: float) -> CorruptionResult | None:
        parameter = float(np.clip(raw_parameter, lower_parameter, upper_parameter))
        if parameter_quantum is not None:
            parameter = float(round(parameter / parameter_quantum) * parameter_quantum)
            parameter = float(np.clip(parameter, lower_parameter, upper_parameter))
        if parameter not in evaluated and len(evaluated) < maximum_evaluations:
            evaluated[parameter] = _candidate(clip, generator, parameter, seed)
        return evaluated.get(parameter)

    if strategy == "bounded_grid":
        for parameter in np.linspace(lower_parameter, upper_parameter, maximum_evaluations):
            evaluate(float(parameter))
    else:
        low = lower_parameter
        high = upper_parameter
        evaluate(low)
        evaluate(high)
        while len(evaluated) < maximum_evaluations:
            midpoint = 0.5 * (low + high)
            before_count = len(evaluated)
            result = evaluate(midpoint)
            if len(evaluated) == before_count:
                break
            measured = 0.0 if result is None else result.measured_severity
            if measured < severity_bin.target:
                low = midpoint
            else:
                high = midpoint

    matching = [
        (parameter, result)
        for parameter, result in evaluated.items()
        if result is not None
        and result.postcondition.hard_negative
        and result.postcondition.metric_name == severity_scale.metric_name
        and severity_bin.contains(result.measured_severity)
    ]
    if not matching:
        observed = sorted(
            0.0 if result is None else result.measured_severity for result in evaluated.values()
        )
        raise SeverityTargetingError(
            f"{severity_scale.metric_name} bin {severity_bin.label!r} "
            f"[{severity_bin.lower:g}, {severity_bin.upper:g}) was not reached; "
            f"observed={observed}"
        )
    selected_parameter, selected_result = min(
        matching,
        key=lambda item: (abs(item[1].measured_severity - severity_bin.target), item[0]),
    )
    parameters = tuple(sorted(evaluated))
    result = _with_targeting_metadata(
        selected_result,
        severity_bin=severity_bin,
        selected_parameter=selected_parameter,
        evaluated_parameters=parameters,
        strategy=strategy,
    )
    return TargetedCorruption(
        result=result,
        severity_bin=severity_bin,
        selected_parameter=selected_parameter,
        evaluated_parameters=parameters,
    )
