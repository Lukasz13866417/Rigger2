"""Plane-relative heel/toe sliding diagnostics during detected contacts."""

from __future__ import annotations

import numpy as np

from motionlab.kinematics.fk import marker_world_positions
from motionlab.metrics.base import DiagnosticEvent, MetricResult
from motionlab.motion.clip import MotionClip
from motionlab.processing._signals import true_intervals
from motionlab.processing.contacts import ContactResult


def foot_sliding_metric(clip: MotionClip, contacts: ContactResult) -> MetricResult:
    """Measure confidence-weighted tangential marker displacement per stance interval."""
    if contacts.hard.shape[0] != clip.num_frames:
        raise ValueError("contact frame count must match clip")
    markers = marker_world_positions(clip, contacts.marker_names)
    normal = contacts.ground_plane[:3].astype(np.float64)
    frame_speed = np.where(
        contacts.hard,
        contacts.tangential_speed_mps * contacts.confidence,
        0.0,
    )
    events: list[DiagnosticEvent] = []
    interval_slips_cm: list[float] = []

    for marker_index, marker_name in enumerate(contacts.marker_names):
        position = markers[marker_name]
        delta = np.diff(position, axis=0)
        tangential_delta = delta - np.sum(delta * normal, axis=-1, keepdims=True) * normal
        distance = np.linalg.norm(tangential_delta, axis=-1)
        for start, stop in true_intervals(contacts.hard[:, marker_index]):
            if stop - start < 2:
                slip_m = 0.0
            else:
                interval_weight = np.minimum(
                    contacts.confidence[start : stop - 1, marker_index],
                    contacts.confidence[start + 1 : stop, marker_index],
                )
                slip_m = float(np.sum(distance[start : stop - 1] * interval_weight))
            slip_cm = 100.0 * slip_m
            interval_slips_cm.append(slip_cm)
            if slip_cm > 0.05:
                side = "left" if marker_name.startswith("left") else "right"
                events.append(
                    DiagnosticEvent(
                        type=f"{side}_foot_slide",
                        frames=(start, stop),
                        joints=(marker_name,),
                        severity=slip_cm,
                        confidence=float(np.mean(contacts.confidence[start:stop, marker_index])),
                        evidence={
                            "marker": marker_name,
                            "stance_slip_cm": slip_cm,
                            "mean_tangential_speed_mps": float(
                                np.mean(contacts.tangential_speed_mps[start:stop, marker_index])
                            ),
                            "max_tangential_speed_mps": float(
                                np.max(contacts.tangential_speed_mps[start:stop, marker_index])
                            ),
                        },
                        suggested_operators=("lock_foot",),
                    )
                )
    maximum_slip = max(interval_slips_cm, default=0.0)
    return MetricResult(
        name="foot_sliding",
        clip_value=maximum_slip,
        frame_values=np.max(frame_speed, axis=1).astype(float).tolist(),
        events=tuple(events),
        units="cm_per_contact_interval",
        confidence=float(np.mean(contacts.confidence)) * contacts.ground_confidence,
        metadata={
            "frame_value_units": "m/s_weighted",
            "interval_slips_cm": interval_slips_cm,
            "contact_source": contacts.source,
        },
    )
