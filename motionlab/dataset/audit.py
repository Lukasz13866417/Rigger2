"""Collateral-defect policies and validation for generated single-defect samples."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from motionlab.corruptions._supervision import joint_part, measure_contact_supervision, part_slot
from motionlab.corruptions.base import CORRUPTION_DEFECT_NAMES, CorruptionResult
from motionlab.metrics.foot_sliding import foot_sliding_metric
from motionlab.metrics.ground import ground_metrics
from motionlab.metrics.joint_limits import joint_limit_metric
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.metrics.speed import speed_metric
from motionlab.processing.contacts import detect_foot_contacts
from motionlab.rigging.semantic_roles import FunctionalPart


class CollateralDefectError(ValueError):
    """Raised when a single-defect candidate creates a disallowed collateral symptom."""


@dataclass(frozen=True)
class CollateralRule:
    """Maximum tolerated increase for one independently measured collateral symptom."""

    defect_family: str
    metric_name: str
    maximum_delta: float
    action: Literal["reject", "label"] = "reject"

    def __post_init__(self) -> None:
        if self.defect_family not in CORRUPTION_DEFECT_NAMES:
            raise ValueError(f"unknown collateral defect family {self.defect_family!r}")
        if not self.metric_name.strip():
            raise ValueError("collateral metric name must be nonempty")
        if not np.isfinite(self.maximum_delta) or self.maximum_delta < 0.0:
            raise ValueError("collateral maximum_delta must be finite and nonnegative")


@dataclass(frozen=True)
class SingleDefectPolicy:
    """Primary postcondition plus all declared collateral bounds for one candidate."""

    primary_family: str
    primary_metric: str
    required_primary_delta: float
    collateral_rules: tuple[CollateralRule, ...]


@dataclass(frozen=True)
class CollateralFinding:
    """One collateral threshold exceeded by a generated candidate."""

    defect_family: str
    metric_name: str
    before: float
    after: float
    delta: float
    allowed_delta: float
    action: Literal["reject", "label"]


_METRICS = (
    ("foot_slide", "foot_sliding_cm_per_contact_interval", 1.0),
    ("ground_penetration", "maximum_ground_penetration_cm", 0.2),
    ("floating_contact", "maximum_floating_contact_cm", 0.5),
    ("joint_pop", "angular_smoothness_p99_rad_s3", 5000.0),
    ("loop_seam", "weighted_loop_seam", 0.03),
    ("joint_limit", "maximum_joint_limit_excess_deg", 1.0),
    ("speed_inconsistency", "root_speed_change_mps", 0.05),
)


def single_defect_policy(
    *,
    primary_family: str,
    primary_metric: str,
    required_primary_delta: float,
) -> SingleDefectPolicy:
    """Build the explicit default policy used by every automatic hard mechanism."""
    if primary_family not in CORRUPTION_DEFECT_NAMES:
        raise ValueError(f"unknown primary corruption family {primary_family!r}")
    if not np.isfinite(required_primary_delta) or required_primary_delta < 0.0:
        raise ValueError("required primary delta must be finite and nonnegative")
    equivalent_primary = {
        "joint_pop": {"joint_pop", "joint_jitter"},
        "joint_jitter": {"joint_pop", "joint_jitter"},
    }.get(primary_family, {primary_family})
    rules: list[CollateralRule] = []
    for family, metric_name, threshold in _METRICS:
        if family in equivalent_primary:
            continue
        action: Literal["reject", "label"] = "reject"
        if primary_family == "speed_inconsistency" and family in {
            "foot_slide",
            "ground_penetration",
        }:
            action = "label"
        if primary_family == "loop_seam" and family == "foot_slide":
            action = "label"
        if primary_family in {"joint_limit", "foot_clearance"} and family == "joint_pop":
            action = "label"
        rules.append(
            CollateralRule(
                defect_family=family,
                metric_name=metric_name,
                maximum_delta=threshold,
                action=action,
            )
        )
    return SingleDefectPolicy(
        primary_family=primary_family,
        primary_metric=primary_metric,
        required_primary_delta=required_primary_delta,
        collateral_rules=tuple(rules),
    )


def _metric_values(result: CorruptionResult) -> dict[str, tuple[float, float]]:
    clean = result.clean_motion
    corrupted = result.corrupted_motion
    _, fixed_clean, fixed_corrupted = measure_contact_supervision(clean, corrupted)
    clean_penetration, clean_floating = ground_metrics(fixed_clean)
    corrupted_penetration, corrupted_floating = ground_metrics(fixed_corrupted)
    clean_limit = joint_limit_metric(clean).clip_value
    corrupted_limit = joint_limit_metric(corrupted).clip_value
    clean_speed = float(speed_metric(clean).clip_value or 0.0)
    corrupted_speed = float(speed_metric(corrupted).clip_value or 0.0)
    return {
        "foot_sliding_cm_per_contact_interval": (
            float(foot_sliding_metric(clean, fixed_clean).clip_value or 0.0),
            float(foot_sliding_metric(corrupted, fixed_corrupted).clip_value or 0.0),
        ),
        "maximum_ground_penetration_cm": (
            float(clean_penetration.clip_value or 0.0),
            float(corrupted_penetration.clip_value or 0.0),
        ),
        "maximum_floating_contact_cm": (
            float(clean_floating.clip_value or 0.0),
            float(corrupted_floating.clip_value or 0.0),
        ),
        "angular_smoothness_p99_rad_s3": (
            float(smoothness_metric(clean).clip_value or 0.0),
            float(smoothness_metric(corrupted).clip_value or 0.0),
        ),
        "weighted_loop_seam": (
            float(loop_seam_metric(clean, fixed_clean).clip_value or 0.0),
            float(loop_seam_metric(corrupted, fixed_corrupted).clip_value or 0.0),
        ),
        "maximum_joint_limit_excess_deg": (
            0.0 if clean_limit is None else float(clean_limit),
            0.0 if corrupted_limit is None else float(corrupted_limit),
        ),
        "root_speed_change_mps": (0.0, abs(corrupted_speed - clean_speed)),
    }


def _label_finding(
    result: CorruptionResult,
    finding: CollateralFinding,
    symptom: np.ndarray,
    responsibility: np.ndarray,
    confidence: np.ndarray,
) -> None:
    defect = CORRUPTION_DEFECT_NAMES.index(finding.defect_family)
    support = np.any(result.intervention_mask, axis=(1, 2))
    if not np.any(support):
        support[:] = True
    if finding.defect_family == "loop_seam":
        support[:] = False
        support[max(0, result.clean_motion.num_frames - 3) :] = True

    intervention_parts = {
        joint_part(result.clean_motion.skeleton, int(joint))
        for joint in np.flatnonzero(np.any(result.intervention_mask, axis=(0, 2)))
    }
    symptom_parts = set(intervention_parts)
    if finding.defect_family in {"foot_slide", "ground_penetration", "floating_contact"}:
        symptom_parts = {FunctionalPart.LEFT_LEG, FunctionalPart.RIGHT_LEG}
    elif finding.defect_family == "speed_inconsistency":
        symptom_parts.update(
            (FunctionalPart.ROOT_PELVIS, FunctionalPart.LEFT_LEG, FunctionalPart.RIGHT_LEG)
        )
    elif finding.defect_family == "loop_seam":
        symptom_parts.update((FunctionalPart.ROOT_PELVIS, FunctionalPart.TRUNK_SPINE))
    if not symptom_parts:
        symptom_parts.add(FunctionalPart.OTHER_ACCESSORY)
    responsibility_parts = intervention_parts | symptom_parts

    if finding.defect_family == "foot_slide":
        contacts = result.contact_supervision
        if contacts is None:
            inferred_contacts = detect_foot_contacts(result.clean_motion)
            marker_names = inferred_contacts.marker_names
            truth_hard = inferred_contacts.hard
        else:
            marker_names = contacts.marker_names
            truth_hard = contacts.truth_hard
        for side, part in (
            ("left", FunctionalPart.LEFT_LEG),
            ("right", FunctionalPart.RIGHT_LEG),
        ):
            indices = [
                index for index, marker in enumerate(marker_names) if marker.startswith(f"{side}_")
            ]
            frames = support & np.any(truth_hard[:, indices], axis=1)
            symptom[frames, part_slot(part), defect] = True

    if finding.defect_family != "foot_slide":
        for part in symptom_parts:
            symptom[support, part_slot(part), defect] = True
    for part in responsibility_parts:
        slot = part_slot(part)
        responsibility[support, slot, defect] = np.maximum(
            responsibility[support, slot, defect], 0.5
        )
        confidence[support, slot, defect] = np.maximum(confidence[support, slot, defect], 0.5)


def validate_single_defect(
    result: CorruptionResult,
    policy: SingleDefectPolicy,
) -> tuple[CorruptionResult, tuple[CollateralFinding, ...]]:
    """Reject or multi-label collateral symptoms under a fully declared policy."""
    if result.corruption_family != policy.primary_family:
        raise ValueError("single-defect policy family does not match the corruption result")
    if result.postcondition.metric_name != policy.primary_metric:
        raise ValueError("single-defect policy metric does not match the corruption result")
    if result.measured_severity < policy.required_primary_delta:
        raise ValueError("corruption does not meet the policy's required primary delta")

    values = _metric_values(result)
    findings: list[CollateralFinding] = []
    for rule in policy.collateral_rules:
        if rule.defect_family == "loop_seam" and result.clean_motion.metadata.get(
            "loop_kind"
        ) not in {"root_motion", "in_place"}:
            continue
        before, after = values[rule.metric_name]
        delta = max(0.0, after - before)
        if delta > rule.maximum_delta:
            findings.append(
                CollateralFinding(
                    defect_family=rule.defect_family,
                    metric_name=rule.metric_name,
                    before=before,
                    after=after,
                    delta=delta,
                    allowed_delta=rule.maximum_delta,
                    action=rule.action,
                )
            )

    rejected = [finding for finding in findings if finding.action == "reject"]
    if rejected:
        details = ", ".join(
            f"{item.metric_name} +{item.delta:g} > {item.allowed_delta:g}" for item in rejected
        )
        raise CollateralDefectError(f"disallowed collateral defect(s): {details}")

    symptom = result.symptom_mask.copy()
    responsibility = result.responsibility_target.copy()
    responsibility_confidence = result.responsibility_confidence.copy()
    for finding in findings:
        _label_finding(result, finding, symptom, responsibility, responsibility_confidence)
    policy_record = {
        "primary_family": policy.primary_family,
        "primary_metric": policy.primary_metric,
        "required_primary_delta": policy.required_primary_delta,
        "collateral_rules": [
            {
                "defect_family": rule.defect_family,
                "metric_name": rule.metric_name,
                "maximum_delta": rule.maximum_delta,
                "action": rule.action,
            }
            for rule in policy.collateral_rules
        ],
        "findings": [
            {
                "defect_family": finding.defect_family,
                "metric_name": finding.metric_name,
                "before": finding.before,
                "after": finding.after,
                "delta": finding.delta,
                "allowed_delta": finding.allowed_delta,
                "action": finding.action,
            }
            for finding in findings
        ],
        "single_defect": not findings,
    }
    return (
        replace(
            result,
            symptom_mask=symptom,
            responsibility_target=responsibility,
            responsibility_confidence=responsibility_confidence,
            measured_metrics_before={
                **dict(result.measured_metrics_before),
                **{f"collateral_check:{name}": pair[0] for name, pair in values.items()},
            },
            measured_metrics_after={
                **dict(result.measured_metrics_after),
                **{f"collateral_check:{name}": pair[1] for name, pair in values.items()},
            },
            schema_metadata={
                **dict(result.schema_metadata),
                "single_defect_validation": policy_record,
            },
        ),
        tuple(findings),
    )
