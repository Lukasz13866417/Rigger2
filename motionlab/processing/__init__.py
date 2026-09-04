"""Coordinate and temporal processing for validated motion clips."""

from motionlab.processing.canonicalize import canonicalize_origin_and_facing
from motionlab.processing.contacts import ContactConfig, ContactResult, detect_foot_contacts
from motionlab.processing.cycles import extract_best_cycle, find_cycle_candidates
from motionlab.processing.phase import estimate_gait_phase, extract_gait_events
from motionlab.processing.resample import resample_motion

__all__ = [
    "ContactConfig",
    "ContactResult",
    "canonicalize_origin_and_facing",
    "detect_foot_contacts",
    "estimate_gait_phase",
    "extract_best_cycle",
    "extract_gait_events",
    "find_cycle_candidates",
    "resample_motion",
]
