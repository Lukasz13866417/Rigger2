"""Ground penetration and implausible floating-contact diagnostics."""

from __future__ import annotations

import numpy as np

from motionlab.metrics.base import DiagnosticEvent, MetricResult
from motionlab.processing._signals import true_intervals
from motionlab.processing.contacts import ContactResult


def ground_metrics(
    contacts: ContactResult,
    *,
    penetration_tolerance_m: float = 0.005,
    floating_tolerance_m: float = 0.04,
) -> tuple[MetricResult, MetricResult]:
    """Measure below-plane depth and high markers during declared contact."""
    if penetration_tolerance_m < 0.0 or floating_tolerance_m < 0.0:
        raise ValueError("ground tolerances must be nonnegative")
    penetration_m = np.maximum(0.0, -contacts.height_m)
    floating_m = np.where(contacts.hard, np.maximum(0.0, contacts.height_m), 0.0)
    penetration_events: list[DiagnosticEvent] = []
    floating_events: list[DiagnosticEvent] = []
    for marker_index, marker_name in enumerate(contacts.marker_names):
        for start, stop in true_intervals(penetration_m[:, marker_index] > penetration_tolerance_m):
            depth_cm = float(100.0 * np.max(penetration_m[start:stop, marker_index]))
            penetration_events.append(
                DiagnosticEvent(
                    type="ground_penetration",
                    frames=(start, stop),
                    joints=(marker_name,),
                    severity=depth_cm,
                    confidence=contacts.ground_confidence,
                    evidence={"marker": marker_name, "maximum_depth_cm": depth_cm},
                    suggested_operators=("adjust_ground_height", "lock_foot"),
                )
            )
        for start, stop in true_intervals(floating_m[:, marker_index] > floating_tolerance_m):
            height_cm = float(100.0 * np.max(floating_m[start:stop, marker_index]))
            floating_events.append(
                DiagnosticEvent(
                    type="floating_contact",
                    frames=(start, stop),
                    joints=(marker_name,),
                    severity=height_cm,
                    confidence=float(np.mean(contacts.confidence[start:stop, marker_index])),
                    evidence={"marker": marker_name, "maximum_height_cm": height_cm},
                    suggested_operators=("lock_foot",),
                )
            )
    penetration = MetricResult(
        name="ground_penetration",
        clip_value=float(100.0 * np.max(penetration_m, initial=0.0)),
        frame_values=(100.0 * np.max(penetration_m, axis=1)).astype(float).tolist(),
        events=tuple(penetration_events),
        units="cm",
        confidence=contacts.ground_confidence,
        metadata={"tolerance_m": penetration_tolerance_m},
    )
    floating = MetricResult(
        name="floating_contact",
        clip_value=float(100.0 * np.max(floating_m, initial=0.0)),
        frame_values=(100.0 * np.max(floating_m, axis=1)).astype(float).tolist(),
        events=tuple(floating_events),
        units="cm",
        confidence=float(np.mean(contacts.confidence)) * contacts.ground_confidence,
        metadata={"tolerance_m": floating_tolerance_m},
    )
    return penetration, floating
