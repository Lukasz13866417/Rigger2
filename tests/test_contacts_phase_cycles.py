from __future__ import annotations

import numpy as np
import pytest

from motionlab.processing._signals import clean_contact_mask, true_intervals
from motionlab.processing.contacts import ContactConfig, ContactResult, detect_foot_contacts
from motionlab.processing.cycles import extract_best_cycle, find_cycle_candidates
from motionlab.processing.phase import estimate_gait_phase, extract_gait_events
from motionlab.testing.synthetic import make_synthetic_contact_walk, make_synthetic_walk


def make_manual_contacts(frame_count: int = 41) -> ContactResult:
    hard = np.zeros((frame_count, 4), dtype=np.bool_)
    for start in (0, 20, 40):
        hard[start : min(start + 5, frame_count), 0:2] = True
    for start in (10, 30):
        hard[start : min(start + 5, frame_count), 2:4] = True
    return ContactResult(
        marker_names=("left_heel", "left_toe", "right_heel", "right_toe"),
        hard=hard,
        confidence=np.ones((frame_count, 4), dtype=np.float32),
        height_m=np.zeros((frame_count, 4), dtype=np.float32),
        tangential_speed_mps=np.zeros((frame_count, 4), dtype=np.float32),
        ground_plane=np.array([0.0, 1.0, 0.0, 0.04], dtype=np.float32),
        source="test",
        ground_confidence=1.0,
    )


def test_boolean_signal_cleanup_closes_gaps_and_removes_islands() -> None:
    mask = np.array([False, True, True, False, True, True, False, True, False])
    cleaned = clean_contact_mask(mask, min_true_frames=2, max_false_gap_frames=1)
    np.testing.assert_array_equal(
        cleaned,
        [False, True, True, True, True, True, False, False, False],
    )
    assert true_intervals(cleaned) == ((1, 6),)


def test_stationary_grounded_markers_contact_and_raised_markers_do_not() -> None:
    clip = make_synthetic_walk(num_frames=12)
    rotations = np.broadcast_to(
        clip.skeleton.rest_local_quat_wxyz,
        (clip.num_frames, clip.num_joints, 4),
    )
    root = clip.root_translation_m.copy()
    root[:, [0, 2]] = 0.0
    root[:, 1] = 0.93
    stationary = clip.with_updates(local_quat_wxyz=rotations, root_translation_m=root)
    contacts = detect_foot_contacts(stationary, ground_plane=(0.0, 1.0, 0.0, 0.04))
    assert contacts.hard.all()

    root[:, 1] += 0.20
    raised = detect_foot_contacts(
        stationary.with_updates(root_translation_m=root),
        ground_plane=(0.0, 1.0, 0.0, 0.04),
    )
    assert not raised.hard.any()


def test_contact_thresholds_validate_hysteresis_direction() -> None:
    with pytest.raises(ValueError, match="height_exit"):
        ContactConfig(height_enter_m=0.05, height_exit_m=0.02)


def test_gait_events_and_phase_follow_alternating_strikes() -> None:
    contacts = make_manual_contacts()
    events = extract_gait_events(contacts)
    strikes = [event for event in events if event.kind == "heel_strike"]
    assert [(event.frame, event.side) for event in strikes] == [
        (0, "left"),
        (10, "right"),
        (20, "left"),
        (30, "right"),
        (40, "left"),
    ]
    phase = estimate_gait_phase(contacts)
    assert phase.valid.all()
    assert phase.phase_rad[0] == pytest.approx(0.0)
    assert phase.phase_rad[10] == pytest.approx(np.pi)
    assert phase.phase_rad[20] == pytest.approx(2.0 * np.pi)
    assert phase.confidence == pytest.approx(1.0)


def test_missing_strikes_return_invalid_phase_with_warning() -> None:
    contacts = make_manual_contacts(frame_count=5)
    phase = estimate_gait_phase(contacts)
    assert not phase.valid.any()
    assert phase.confidence == 0.0
    assert phase.warnings


def test_cycle_candidates_use_same_side_strikes_and_exclude_closing_frame() -> None:
    clip = make_synthetic_walk(num_frames=41, fps=20.0)
    contacts = make_manual_contacts()
    candidates = find_cycle_candidates(clip, contacts)
    assert candidates
    assert all(candidate.stop_frame - candidate.start_frame == 20 for candidate in candidates)
    cycle = extract_best_cycle(clip, contacts)
    assert cycle.num_frames == 20
    assert cycle.metadata["cycle_source_interval"] in ([0, 20], [20, 40])


def test_contact_authored_fixture_has_alternating_low_slip_stances() -> None:
    clip = make_synthetic_contact_walk(num_cycles=2, fps=60.0)
    contacts = detect_foot_contacts(clip)
    events = extract_gait_events(contacts)
    strikes = [(event.frame, event.side) for event in events if event.kind == "heel_strike"]
    assert strikes == [(0, "left"), (31, "right"), (61, "left"), (91, "right")]
    assert contacts.hard.sum(axis=0).min() >= 55
    candidates = find_cycle_candidates(clip, contacts)
    assert candidates
    assert candidates[0].stop_frame - candidates[0].start_frame in {60, 61}
