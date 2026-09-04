"""Satisficing deterministic production gates and lexicographic candidate ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from motionlab.critic.forensics import motion_deterministic_metrics, motion_distances
from motionlab.metrics.speed import speed_metric
from motionlab.motion.clip import MotionClip
from motionlab.optimization.cmaes import CMAOptimizationConfig, production_constraint_reasons

ACCEPTANCE_VERSION = "motionlab.production_acceptance.v1"


class ProductionThresholds(BaseModel):
    """Fixed-rig acceptable bands; values inside a band receive no extra reward."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_stance_slip_cm: float = Field(default=2.5, ge=0.0)
    maximum_ground_penetration_cm: float = Field(default=2.0, ge=0.0)
    maximum_floating_contact_cm: float = Field(default=5.0, ge=0.0)
    maximum_loop_seam: float = Field(default=0.2, ge=0.0)
    maximum_target_speed_error_mps: float = Field(default=0.05, ge=0.0)
    maximum_joint_limit_excess_deg: float = Field(default=0.0, ge=0.0)


@dataclass(frozen=True)
class ProductionAcceptance:
    """Raw deterministic evidence and its thresholded production interpretation."""

    raw_metrics: dict[str, float | None]
    constraints: dict[str, dict[str, Any]]
    hard_invalid_reasons: tuple[str, ...]
    all_required_satisfied: bool
    satisficing_penalty: float


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate evaluated under deterministic and perceptual ordering."""

    method: str
    motion: MotionClip
    acceptance: ProductionAcceptance
    perceptual_score: float | None
    edit_magnitude: float
    collateral_change: float

    @property
    def ordering_key(self) -> tuple[float, float, float, float, float, float]:
        """Higher is better under hard validity, satisficing, quality, then edit size."""
        hard_valid = not self.acceptance.hard_invalid_reasons
        perceptual = 0.0 if self.perceptual_score is None else self.perceptual_score
        return (
            float(hard_valid),
            float(self.acceptance.all_required_satisfied),
            -self.acceptance.satisficing_penalty,
            perceptual if self.acceptance.all_required_satisfied else 0.0,
            -self.collateral_change,
            -self.edit_magnitude,
        )


def declare_reference_speed_task(
    reference: MotionClip,
    candidate: MotionClip,
    *,
    tolerance_mps: float = 0.05,
) -> tuple[MotionClip, MotionClip]:
    """Attach a measured reference-speed task without inheriting a parent-clip command."""
    measured = speed_metric(reference).clip_value
    if measured is None:
        raise ValueError("reference speed is unavailable")
    target = float(measured)
    if not np.isfinite(tolerance_mps) or tolerance_mps < 0.0:
        raise ValueError("speed tolerance must be finite and nonnegative")

    def prepared(clip: MotionClip) -> MotionClip:
        return clip.with_updates(
            metadata={
                **dict(clip.metadata),
                "task_target_speed": None,
                "clean_window_reference_speed": target,
                "parent_clip_speed": clip.metadata.get("expected_speed_mps"),
                "target_speed": target,
                "target_speed_mps": target,
                "expected_speed_mps": target,
                "target_speed_tolerance": tolerance_mps,
                "target_speed_source": "measured_pair_reference_speed",
            }
        )

    return prepared(reference), prepared(candidate)


def assess_production_acceptance(
    reference: MotionClip,
    candidate: MotionClip,
    *,
    thresholds: ProductionThresholds | None = None,
) -> ProductionAcceptance:
    """Evaluate hard validity plus required metric bands without minimizing inside them."""
    limits = ProductionThresholds() if thresholds is None else thresholds
    metrics = motion_deterministic_metrics(candidate)
    definitions: tuple[tuple[str, str, float, bool], ...] = (
        (
            "foot_slide",
            "maximum_stance_slip_cm",
            limits.maximum_stance_slip_cm,
            True,
        ),
        (
            "ground_penetration",
            "maximum_ground_penetration_cm",
            limits.maximum_ground_penetration_cm,
            True,
        ),
        (
            "floating_contact",
            "maximum_floating_contact_cm",
            limits.maximum_floating_contact_cm,
            True,
        ),
        (
            "target_speed_error",
            "absolute_target_speed_error_mps",
            limits.maximum_target_speed_error_mps,
            True,
        ),
        (
            "loop_error",
            "loop_seam",
            limits.maximum_loop_seam,
            bool(reference.metadata.get("loop_kind")),
        ),
        (
            "joint_limit_excess",
            "maximum_joint_limit_violation_deg",
            limits.maximum_joint_limit_excess_deg,
            bool(reference.skeleton.joint_limits),
        ),
    )
    constraints: dict[str, dict[str, Any]] = {}
    penalty = 0.0
    band_failures: list[str] = []
    for name, metric, maximum, required in definitions:
        raw = metrics.get(metric)
        satisfied = not required or (raw is not None and float(raw) <= maximum + 1.0e-12)
        excess = 0.0 if not required or raw is None else max(0.0, float(raw) - maximum)
        if required and not satisfied:
            band_failures.append(name)
            penalty += 1.0 if raw is None else excess / max(maximum, 1.0e-6)
        constraints[name] = {
            "metric": metric,
            "raw_value": raw,
            "maximum": maximum,
            "required": required,
            "satisfied": satisfied,
            "excess": excess,
        }
    config = CMAOptimizationConfig(
        mode="production",
        objective_source="deterministic_target",
    )
    hard_reasons = production_constraint_reasons(reference, candidate, config)
    unique_reasons = tuple(dict.fromkeys(hard_reasons))
    return ProductionAcceptance(
        raw_metrics=metrics,
        constraints=constraints,
        hard_invalid_reasons=unique_reasons,
        all_required_satisfied=not band_failures and not unique_reasons,
        satisficing_penalty=penalty,
    )


def rank_repair_candidates(
    reference: MotionClip,
    candidates: list[tuple[str, MotionClip, float | None]],
    *,
    thresholds: ProductionThresholds | None = None,
) -> list[RankedCandidate]:
    """Order candidate methods using hard gates, perceptual quality, then small edits."""
    reference_metrics = motion_deterministic_metrics(reference)
    ranked: list[RankedCandidate] = []
    for method, motion, perceptual_score in candidates:
        acceptance = assess_production_acceptance(reference, motion, thresholds=thresholds)
        distances = motion_distances(reference, motion)
        edit = (
            distances["root_translation_rms_m"]
            + distances["joint_world_position_rms_m"]
            + distances["local_rotation_geodesic_rms_deg"] / 30.0
        )
        collateral = 0.0
        for key, raw in acceptance.raw_metrics.items():
            old = reference_metrics.get(key)
            if raw is not None and old is not None:
                collateral += max(0.0, float(raw) - float(old))
        ranked.append(
            RankedCandidate(
                method=method,
                motion=motion,
                acceptance=acceptance,
                perceptual_score=perceptual_score,
                edit_magnitude=edit,
                collateral_change=collateral,
            )
        )
    return sorted(ranked, key=lambda item: item.ordering_key, reverse=True)
