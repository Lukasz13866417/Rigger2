from __future__ import annotations

import numpy as np

from motionlab.corruptions.foot_slide import corrupt_foot_slide_root_drift
from motionlab.math.quaternion import axis_angle_to_quaternion
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.motion.clip import MotionClip
from motionlab.repair.close_loop import close_loop
from motionlab.repair.foot_lock import lock_foot
from motionlab.testing.synthetic import make_synthetic_humanoid


def make_stationary_walk(*, frames: int = 41, fps: float = 60.0) -> MotionClip:
    skeleton = make_synthetic_humanoid()
    rotations = np.broadcast_to(
        skeleton.rest_local_quat_wxyz,
        (frames, skeleton.num_joints, 4),
    )
    root = np.zeros((frames, 3), dtype=np.float32)
    root[:, 1] = 0.93
    return MotionClip(
        skeleton,
        rotations,
        root,
        fps,
        metadata={"fixture": "stationary_contact"},
    )


def test_foot_slide_corruption_is_seed_deterministic_and_boundary_smooth() -> None:
    clip = make_stationary_walk()
    first = corrupt_foot_slide_root_drift(
        clip,
        start_frame=8,
        stop_frame=33,
        distance_m=0.08,
        seed=17,
    )
    second = corrupt_foot_slide_root_drift(
        clip,
        start_frame=8,
        stop_frame=33,
        distance_m=0.08,
        seed=17,
    )
    np.testing.assert_array_equal(
        first.corrupted.root_translation_m, second.corrupted.root_translation_m
    )
    np.testing.assert_array_equal(first.affected_frames, second.affected_frames)
    assert first.corrupted.content_hash == second.corrupted.content_hash
    np.testing.assert_allclose(first.corrupted.root_translation_m[:8], clip.root_translation_m[:8])
    np.testing.assert_allclose(
        first.corrupted.root_translation_m[33:], clip.root_translation_m[33:]
    )
    np.testing.assert_allclose(first.corrupted.root_translation_m[8], clip.root_translation_m[8])
    np.testing.assert_allclose(first.corrupted.root_translation_m[32], clip.root_translation_m[32])


def test_foot_lock_reduces_synthetic_stance_slip_by_at_least_ninety_percent() -> None:
    clip = make_stationary_walk()
    corruption = corrupt_foot_slide_root_drift(
        clip,
        start_frame=8,
        stop_frame=33,
        distance_m=0.05,
    )
    result = lock_foot(
        corruption.corrupted,
        side="left",
        start_frame=8,
        stop_frame=33,
        anchor_mode="first",
        blend_in_frames=2,
        blend_out_frames=2,
        ground_plane=(0.0, 1.0, 0.0, 0.04),
    )
    assert result.before_slip_cm > 1.0
    assert result.after_slip_cm <= 0.1 * result.before_slip_cm
    assert result.maximum_ik_residual_m < 1.0e-4
    np.testing.assert_allclose(
        result.repaired.local_quat_wxyz[:8],
        corruption.corrupted.local_quat_wxyz[:8],
    )
    np.testing.assert_allclose(
        result.repaired.local_quat_wxyz[33:],
        corruption.corrupted.local_quat_wxyz[33:],
    )


def test_foot_lock_noop_on_clean_stationary_contact_is_small() -> None:
    clip = make_stationary_walk()
    result = lock_foot(
        clip,
        side="right",
        start_frame=5,
        stop_frame=35,
        ground_plane=(0.0, 1.0, 0.0, 0.04),
    )
    assert result.before_slip_cm == 0.0
    assert result.after_slip_cm < 1.0e-5
    assert np.max(np.abs(result.repaired.local_quat_wxyz - clip.local_quat_wxyz)) < 1.0e-6
    assert smoothness_metric(result.repaired).clip_value == 0.0


def test_loop_closure_reduces_pose_seam_by_at_least_eighty_percent() -> None:
    clip = make_stationary_walk(frames=31, fps=30.0)
    root = clip.root_translation_m.copy()
    root[:, 2] = np.arange(clip.num_frames, dtype=np.float32) / clip.fps
    rotations = clip.local_quat_wxyz.copy()
    rotations[-1, 10] = axis_angle_to_quaternion([0.35, 0.0, 0.0])
    broken = clip.with_updates(local_quat_wxyz=rotations, root_translation_m=root)
    before = loop_seam_metric(broken)
    result = close_loop(broken, seam_window_frames=8)
    after = loop_seam_metric(result.repaired)
    assert before.clip_value is not None and after.clip_value is not None
    assert result.reduction_fraction >= 0.80
    assert after.clip_value <= 0.20 * before.clip_value
    np.testing.assert_allclose(result.repaired.local_quat_wxyz[:-8], broken.local_quat_wxyz[:-8])
