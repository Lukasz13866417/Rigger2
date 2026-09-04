"""Gait-cycle candidate extraction from same-side heel-strike intervals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.finite_difference import finite_difference
from motionlab.math.quaternion import quaternion_geodesic_distance
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult
from motionlab.processing.phase import GaitEvent, extract_gait_events


@dataclass(frozen=True)
class CycleCandidate:
    """A half-open candidate cycle interval and transparent compatibility components."""

    start_frame: int
    stop_frame: int
    side: str
    score: float
    pose_error_rad: float
    position_error_m: float
    velocity_error_mps: float
    contact_mismatch: float


def _same_side_pairs(events: tuple[GaitEvent, ...], side: str) -> list[tuple[int, int]]:
    strikes = [
        event.frame for event in events if event.kind == "heel_strike" and event.side == side
    ]
    return list(pairwise(strikes))


def find_cycle_candidates(
    clip: MotionClip,
    contacts: ContactResult,
    *,
    minimum_frames: int = 8,
) -> tuple[CycleCandidate, ...]:
    """Rank same-side heel-strike intervals by pose, velocity, and contact compatibility."""
    if contacts.hard.shape[0] != clip.num_frames:
        raise ValueError("contact frame count must match clip")
    if minimum_frames < 2:
        raise ValueError("minimum_frames must be at least two")
    events = extract_gait_events(contacts)
    position, _ = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    root = clip.skeleton.root_index
    position_relative = position - position[:, root : root + 1]
    root_velocity = finite_difference(clip.root_translation_m, 1.0 / clip.fps)
    candidates: list[CycleCandidate] = []
    for side in ("left", "right"):
        for start, stop in _same_side_pairs(events, side):
            if stop - start < minimum_frames or stop >= clip.num_frames:
                continue
            pose_error = float(
                np.mean(
                    quaternion_geodesic_distance(
                        clip.local_quat_wxyz[start],
                        clip.local_quat_wxyz[stop],
                    )
                )
            )
            position_error = float(
                np.mean(np.linalg.norm(position_relative[start] - position_relative[stop], axis=-1))
            )
            velocity_error = float(np.linalg.norm(root_velocity[start] - root_velocity[stop]))
            contact_mismatch = float(np.mean(contacts.hard[start] != contacts.hard[stop]))
            score = pose_error + 4.0 * position_error + 0.25 * velocity_error + contact_mismatch
            candidates.append(
                CycleCandidate(
                    start_frame=start,
                    stop_frame=stop,
                    side=side,
                    score=score,
                    pose_error_rad=pose_error,
                    position_error_m=position_error,
                    velocity_error_mps=velocity_error,
                    contact_mismatch=contact_mismatch,
                )
            )
    return tuple(sorted(candidates, key=lambda candidate: candidate.score))


def extract_best_cycle(clip: MotionClip, contacts: ContactResult) -> MotionClip:
    """Return the best half-open gait cycle without duplicating the closing strike frame."""
    candidates = find_cycle_candidates(clip, contacts)
    if not candidates:
        raise ValueError("no complete same-side heel-strike cycle is available")
    best = candidates[0]
    cycle = clip.slice_frames(best.start_frame, best.stop_frame)
    metadata = dict(cycle.metadata)
    metadata.update(
        {
            "cycle_side": best.side,
            "cycle_candidate_score": best.score,
            "cycle_source_interval": [best.start_frame, best.stop_frame],
        }
    )
    return cycle.with_updates(metadata=metadata)
