"""CMA-ES driver for one typed motion parameter block or a concatenation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cma
import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.io.npz import save_motion_npz
from motionlab.motion.clip import MotionClip
from motionlab.optimization.parameter_blocks import MotionParameterBlock

BLOCK_CMA_VERSION = "motionlab.block_cma.v1"


@dataclass(frozen=True)
class BlockCMAResult:
    """Best feasible candidate and complete optimizer evidence."""

    motion: MotionClip
    values: NDArray[np.float64]
    objective: float
    evaluations: int
    feasible_evaluations: int
    report_path: Path


def optimize_parameter_block_with_cma(
    original: MotionClip,
    parameterization: MotionParameterBlock,
    objective: Callable[[MotionClip], float],
    output_directory: Path,
    *,
    constraint_reasons: Callable[[MotionClip], list[str]] | None = None,
    seed: int = 4701,
    maximum_evaluations: int = 2_000,
    initial_sigma: float = 0.25,
) -> BlockCMAResult:
    """Optimize any common-interface block while rejecting infeasible candidates."""
    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive")
    initial = np.asarray(parameterization.initial_values, dtype=np.float64)
    lower = np.asarray(parameterization.lower_bounds, dtype=np.float64)
    upper = np.asarray(parameterization.upper_bounds, dtype=np.float64)
    expected = len(parameterization.parameter_names)
    if initial.shape != (expected,) or lower.shape != (expected,) or upper.shape != (expected,):
        raise ValueError("parameter block vectors must match parameter_names")
    if not np.all(np.isfinite(initial)) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("parameter block initial values must be finite and within bounds")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    save_motion_npz(output / "original.npz", original)
    reject = constraint_reasons or (lambda _candidate: [])
    evaluations = 0
    feasible_evaluations = 0
    rejection_counts: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    best_values = initial.copy()
    best_motion = parameterization.apply(original, initial)
    initial_reasons = reject(best_motion)
    best_objective = float("inf") if initial_reasons else float(objective(best_motion))
    if np.isfinite(best_objective):
        feasible_evaluations = 1

    def evaluate(raw_values: ArrayLike) -> float:
        nonlocal evaluations, feasible_evaluations, best_values, best_motion, best_objective
        evaluations += 1
        try:
            candidate = parameterization.apply(original, raw_values)
            reasons = reject(candidate)
        except (ValueError, FloatingPointError) as exc:
            reasons = [f"invalid:{type(exc).__name__}"]
            candidate = original
        if reasons:
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            return 1.0e12 + len(reasons)
        score = float(objective(candidate))
        if not np.isfinite(score):
            rejection_counts["nonfinite_objective"] = (
                rejection_counts.get("nonfinite_objective", 0) + 1
            )
            return 1.0e12
        feasible_evaluations += 1
        if score < best_objective:
            best_values = np.asarray(raw_values, dtype=np.float64).copy()
            best_motion = candidate
            best_objective = score
        return score

    def callback(strategy: Any) -> None:
        history.append(
            {
                "generation": int(strategy.countiter),
                "evaluations": evaluations,
                "best_objective": None if not np.isfinite(best_objective) else best_objective,
                "feasible_evaluations": feasible_evaluations,
                "sigma": float(strategy.sigma),
            }
        )

    cma.fmin2(
        evaluate,
        initial,
        initial_sigma,
        {
            "bounds": [lower.tolist(), upper.tolist()],
            "maxfevals": maximum_evaluations,
            "seed": seed,
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
        },
        eval_initial_x=True,
        callback=callback,
    )
    save_motion_npz(output / "best.npz", best_motion)
    np.savez_compressed(
        output / "best_parameters.npz",
        names=np.asarray(parameterization.parameter_names, dtype=np.str_),
        values=best_values,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    report = {
        "format_version": BLOCK_CMA_VERSION,
        "parameterization": parameterization.name,
        "parameter_names": list(parameterization.parameter_names),
        "parameter_count": expected,
        "seed": seed,
        "maximum_evaluations": maximum_evaluations,
        "evaluations": evaluations,
        "feasible_evaluations": feasible_evaluations,
        "best_objective": None if not np.isfinite(best_objective) else best_objective,
        "best_values": best_values.tolist(),
        "rejection_counts": rejection_counts,
        "history": history,
        "artifacts": {
            "original": str(output / "original.npz"),
            "best": str(output / "best.npz"),
            "parameters": str(output / "best_parameters.npz"),
        },
    }
    report_path = output / "optimization.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BlockCMAResult(
        motion=best_motion,
        values=best_values,
        objective=best_objective,
        evaluations=evaluations,
        feasible_evaluations=feasible_evaluations,
        report_path=report_path,
    )
