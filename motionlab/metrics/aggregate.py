"""Aggregate deterministic diagnostics without collapsing physical evidence."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from motionlab.metrics.base import DeterministicReport
from motionlab.metrics.foot_sliding import foot_sliding_metric
from motionlab.metrics.gait import gait_metrics
from motionlab.metrics.ground import ground_metrics
from motionlab.metrics.joint_limits import joint_limit_metric
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.metrics.speed import speed_metric
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactConfig, detect_foot_contacts
from motionlab.processing.phase import estimate_gait_phase


def grade_motion_deterministic(
    clip: MotionClip,
    *,
    ground_plane: ArrayLike = (0.0, 1.0, 0.0, 0.0),
    target_speed_mps: float | None = None,
    contact_config: ContactConfig | None = None,
) -> DeterministicReport:
    """Run deterministic contact, gait, limit, ground, speed, smoothness, and seam metrics."""
    contacts = detect_foot_contacts(
        clip,
        ground_plane=ground_plane,
        config=contact_config,
    )
    phase = estimate_gait_phase(contacts)
    sliding = foot_sliding_metric(clip, contacts)
    gait = gait_metrics(clip, contacts)
    penetration, floating = ground_metrics(contacts)
    speed = speed_metric(
        clip,
        ground_plane=contacts.ground_plane,
        target_speed_mps=target_speed_mps,
    )
    smoothness = smoothness_metric(clip)
    seam = loop_seam_metric(clip, contacts)
    limits = joint_limit_metric(clip)
    metrics = {
        metric.name: metric
        for metric in (
            sliding,
            penetration,
            floating,
            speed,
            smoothness,
            seam,
            *gait,
            limits,
        )
    }
    rig_name = str(clip.skeleton.metadata.get("name", clip.skeleton.content_hash[:19]))
    measured = {
        "average_speed_mps": speed.clip_value,
        "maximum_stance_slip_cm": sliding.clip_value,
        "maximum_ground_penetration_cm": penetration.clip_value,
        "maximum_floating_contact_cm": floating.clip_value,
        "loop_seam": seam.clip_value,
        "cadence_steps_per_minute": gait[0].clip_value,
        "stride_time_s": gait[1].clip_value,
        "stride_length_m": gait[2].clip_value,
        "step_width_m": gait[3].clip_value,
        "gait_asymmetry_percent": gait[4].clip_value,
        "maximum_joint_limit_violation_deg": limits.clip_value,
    }
    return DeterministicReport(
        motion_id=clip.content_hash,
        rig=rig_name,
        fps=clip.fps,
        num_frames=clip.num_frames,
        duration_s=clip.duration_s,
        measured=measured,
        metrics=metrics,
        warnings=phase.warnings,
        confidence={
            "contact": float(contacts.confidence.mean()),
            "ground": contacts.ground_confidence,
            "gait_phase": phase.confidence,
            "gait_geometry": float(np.mean([metric.confidence for metric in gait])),
            "joint_limits": limits.confidence,
        },
        metadata={
            "contact_source": contacts.source,
            "contact_parameters": dict(contacts.metadata),
            "ground_plane": contacts.ground_plane.astype(float).tolist(),
            "gait_events": [
                {
                    "kind": event.kind,
                    "side": event.side,
                    "frame": event.frame,
                    "confidence": event.confidence,
                }
                for event in phase.events
            ],
        },
    )
