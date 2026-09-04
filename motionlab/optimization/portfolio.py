"""Method portfolios for deterministic repair without privileging generic CMA."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from motionlab.motion.clip import MotionClip
from motionlab.optimization.acceptance import (
    ProductionThresholds,
    RankedCandidate,
    rank_repair_candidates,
)


@dataclass(frozen=True)
class RepairMethodSpec:
    """One declared specialized, semantic-CMA, or generic fallback method."""

    name: str
    kind: str
    cheap_to_evaluate: bool


REPAIR_METHOD_PORTFOLIO: dict[str, tuple[RepairMethodSpec, ...]] = {
    "foot_slide": (
        RepairMethodSpec("specialized_foot_lock_ik", "specialized", True),
        RepairMethodSpec("contact_target_cma", "semantic_cma", False),
        RepairMethodSpec("adaptive_so3_cma_fallback", "generic_cma", False),
    ),
    "joint_pop": (
        RepairMethodSpec("specialized_local_smoothing", "specialized", True),
        RepairMethodSpec("smoothing_block_cma", "semantic_cma", False),
        RepairMethodSpec("adaptive_so3_cma_fallback", "generic_cma", False),
    ),
    "joint_jitter": (
        RepairMethodSpec("specialized_local_smoothing", "specialized", True),
        RepairMethodSpec("local_spectral_cma", "semantic_cma", False),
        RepairMethodSpec("adaptive_so3_cma_fallback", "generic_cma", False),
    ),
    "loop_seam": (
        RepairMethodSpec("specialized_loop_closure", "specialized", True),
        RepairMethodSpec("loop_parameter_block_cma", "semantic_cma", False),
        RepairMethodSpec("adaptive_so3_cma_fallback", "generic_cma", False),
    ),
}


@dataclass(frozen=True)
class PortfolioDecision:
    """Selected method and the complete post-repair evidence for every candidate."""

    selected: RankedCandidate
    ranked: tuple[RankedCandidate, ...]

    def report(self) -> dict[str, Any]:
        rows = []
        for candidate in self.ranked:
            rows.append(
                {
                    "method": candidate.method,
                    "selected": candidate is self.selected,
                    "hard_invalid_reasons": list(candidate.acceptance.hard_invalid_reasons),
                    "all_required_thresholds_satisfied": (
                        candidate.acceptance.all_required_satisfied
                    ),
                    "satisficing_penalty": candidate.acceptance.satisficing_penalty,
                    "raw_deterministic_metrics": candidate.acceptance.raw_metrics,
                    "deterministic_constraints": candidate.acceptance.constraints,
                    "perceptual_score": candidate.perceptual_score,
                    "edit_magnitude": candidate.edit_magnitude,
                    "collateral_change": candidate.collateral_change,
                    "ordering_key": list(candidate.ordering_key),
                }
            )
        return {
            "selection_policy": [
                "reject hard-invalid candidates",
                "prefer candidates satisfying every required deterministic threshold",
                "among feasible candidates maximize perceptual quality",
                "break ties by collateral change and edit magnitude",
            ],
            "selected_method": self.selected.method,
            "candidates": rows,
        }


def declared_repair_methods(family: str) -> tuple[RepairMethodSpec, ...]:
    """Return the preserved method portfolio for a known deterministic family."""
    if family not in REPAIR_METHOD_PORTFOLIO:
        raise ValueError(f"no repair portfolio is declared for {family!r}")
    return REPAIR_METHOD_PORTFOLIO[family]


def select_repair_from_portfolio(
    reference: MotionClip,
    candidates: Mapping[str, MotionClip],
    *,
    perceptual_scorer: Callable[[MotionClip], float] | None = None,
    thresholds: ProductionThresholds | None = None,
) -> PortfolioDecision:
    """Evaluate actual outputs and select lexicographically, independent of method type."""
    if not candidates:
        raise ValueError("repair portfolio requires at least one candidate")
    evaluated = []
    for method, motion in candidates.items():
        score = None if perceptual_scorer is None else float(perceptual_scorer(motion))
        evaluated.append((method, motion, score))
    ranked = rank_repair_candidates(reference, evaluated, thresholds=thresholds)
    return PortfolioDecision(ranked[0], tuple(ranked))
