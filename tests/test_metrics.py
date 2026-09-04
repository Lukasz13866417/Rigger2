from __future__ import annotations

import numpy as np
import pytest

from motionlab.math.quaternion import axis_angle_to_quaternion
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.metrics.foot_sliding import foot_sliding_metric
from motionlab.metrics.gait import gait_metrics
from motionlab.metrics.ground import ground_metrics
from motionlab.metrics.joint_limits import joint_limit_metric
from motionlab.metrics.loop_seam import loop_seam_metric
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.metrics.speed import speed_metric
from motionlab.motion.clip import MotionClip
from motionlab.motion.markers import JointLimitSpec
from motionlab.motion.skeleton import Skeleton
from motionlab.processing.contacts import ContactConfig, ContactResult, detect_foot_contacts
from motionlab.testing.synthetic import (
    make_synthetic_contact_walk,
    make_synthetic_humanoid,
    make_synthetic_walk,
)


def grounded_rest_clip(*, frames: int = 11, fps: float = 10.0) -> MotionClip:
    skeleton = make_synthetic_humanoid()
    rotations = np.broadcast_to(
        skeleton.rest_local_quat_wxyz,
        (frames, skeleton.num_joints, 4),
    )
    root = np.zeros((frames, 3), dtype=np.float32)
    root[:, 1] = 0.93
    return MotionClip(skeleton, rotations, root, fps)


def all_contact_result(clip: MotionClip, *, height_m: float = 0.0) -> ContactResult:
    shape = (clip.num_frames, 4)
    return ContactResult(
        marker_names=("left_heel", "left_toe", "right_heel", "right_toe"),
        hard=np.ones(shape, dtype=np.bool_),
        confidence=np.ones(shape, dtype=np.float32),
        height_m=np.full(shape, height_m, dtype=np.float32),
        tangential_speed_mps=np.zeros(shape, dtype=np.float32),
        ground_plane=np.array([0.0, 1.0, 0.0, 0.04], dtype=np.float32),
        source="test",
        ground_confidence=1.0,
    )


def test_stationary_contacts_have_zero_slip_and_known_drift_is_measured() -> None:
    clean = grounded_rest_clip()
    clean_metric = foot_sliding_metric(clean, all_contact_result(clean))
    assert clean_metric.clip_value == pytest.approx(0.0)

    root = clean.root_translation_m.copy()
    root[:, 0] = np.linspace(0.0, 0.10, clean.num_frames)
    sliding = clean.with_updates(root_translation_m=root)
    metric = foot_sliding_metric(sliding, all_contact_result(sliding))
    assert metric.clip_value == pytest.approx(10.0, abs=1.0e-5)
    assert metric.events
    assert metric.events[0].frames == (0, clean.num_frames)


def test_ground_metrics_measure_known_penetration_and_floating_contact() -> None:
    clip = grounded_rest_clip()
    lowered_root = clip.root_translation_m.copy()
    lowered_root[:, 1] -= 0.02
    lowered = clip.with_updates(root_translation_m=lowered_root)
    lowered_contacts = detect_foot_contacts(
        lowered,
        ground_plane=(0.0, 1.0, 0.0, 0.04),
    )
    penetration, _ = ground_metrics(lowered_contacts)
    assert penetration.clip_value == pytest.approx(2.0, abs=1.0e-4)
    assert penetration.events

    raised_root = clip.root_translation_m.copy()
    raised_root[:, 1] += 0.05
    raised = clip.with_updates(root_translation_m=raised_root)
    raised_contacts = detect_foot_contacts(
        raised,
        ground_plane=(0.0, 1.0, 0.0, 0.04),
        config=ContactConfig(height_enter_m=0.10, height_exit_m=0.11),
    )
    _, floating = ground_metrics(raised_contacts)
    assert floating.clip_value == pytest.approx(5.0, abs=1.0e-4)
    assert floating.events


def test_speed_metric_uses_physical_time_and_ignores_vertical_bob() -> None:
    clip = make_synthetic_walk(num_frames=61, fps=60.0)
    metric = speed_metric(clip, target_speed_mps=1.2)
    assert metric.clip_value == pytest.approx(1.2, abs=1.0e-5)
    assert metric.metadata["relative_target_error"] == pytest.approx(0.0, abs=1.0e-5)


def test_joint_pulse_increases_smoothness_metric_and_localizes_joint() -> None:
    clean = grounded_rest_clip(frames=25, fps=60.0)
    rotations = clean.local_quat_wxyz.copy()
    rotations[12, 11] = axis_angle_to_quaternion([0.0, 0.0, 0.5])
    corrupted = clean.with_updates(local_quat_wxyz=rotations)
    clean_metric = smoothness_metric(clean)
    corrupt_metric = smoothness_metric(corrupted)
    assert corrupt_metric.clip_value is not None
    assert clean_metric.clip_value is not None
    assert corrupt_metric.clip_value > clean_metric.clip_value
    assert any("LeftForeArm" in event.joints for event in corrupt_metric.events)


def test_perfect_constant_velocity_loop_has_zero_seam_and_pose_mismatch_increases_it() -> None:
    clip = grounded_rest_clip(frames=20, fps=20.0)
    root = clip.root_translation_m.copy()
    root[:, 2] = np.arange(clip.num_frames) / clip.fps
    loop = clip.with_updates(root_translation_m=root)
    contacts = all_contact_result(loop)
    perfect = loop_seam_metric(loop, contacts)
    assert perfect.clip_value == pytest.approx(0.0, abs=1.0e-8)

    rotations = loop.local_quat_wxyz.copy()
    rotations[-1, 10] = axis_angle_to_quaternion([0.2, 0.0, 0.0])
    mismatched = loop.with_updates(local_quat_wxyz=rotations)
    bad = loop_seam_metric(mismatched, contacts)
    assert bad.clip_value is not None
    assert bad.clip_value > 0.0
    assert bad.events


def test_aggregate_report_serializes_physical_metrics() -> None:
    clip = grounded_rest_clip(frames=21, fps=20.0)
    report = grade_motion_deterministic(
        clip,
        ground_plane=(0.0, 1.0, 0.0, 0.04),
        target_speed_mps=0.0,
    )
    payload = report.model_dump(mode="json")
    assert payload["motion_id"] == clip.content_hash
    assert set(payload["metrics"]) == {
        "foot_sliding",
        "ground_penetration",
        "floating_contact",
        "root_speed",
        "angular_smoothness",
        "loop_seam",
        "cadence",
        "stride_time",
        "stride_length",
        "step_width",
        "gait_asymmetry",
        "joint_limits",
    }
    assert payload["measured"]["average_speed_mps"] == pytest.approx(0.0)
    assert payload["metrics"]["joint_limits"]["clip_value"] is None
    assert payload["metrics"]["joint_limits"]["metadata"]["valid"] is False


@pytest.mark.parametrize("fps", [30.0, 60.0, 120.0])
def test_gait_metrics_use_physical_time_and_metric_geometry(fps: float) -> None:
    clip = make_synthetic_contact_walk(num_cycles=2, fps=fps)
    contacts = detect_foot_contacts(
        clip,
        config=ContactConfig(speed_enter_mps=0.6, speed_exit_mps=0.7),
    )
    cadence, stride_time, stride_length, step_width, asymmetry = gait_metrics(clip, contacts)
    # Contact transitions are quantized to sampled frames; 30 fps may move a strike by one frame.
    assert cadence.clip_value == pytest.approx(120.0, abs=3.0)
    assert stride_time.clip_value == pytest.approx(1.0, abs=1.0 / fps)
    assert stride_length.clip_value == pytest.approx(1.0, abs=0.005)
    assert step_width.clip_value == pytest.approx(0.2, abs=1.0e-5)
    assert asymmetry.clip_value == pytest.approx(0.0, abs=1.5)
    assert all(metric.confidence > 0.0 for metric in (cadence, stride_time, stride_length))


def _skeleton_with_left_arm_limit(spec: JointLimitSpec) -> Skeleton:
    source = make_synthetic_humanoid()
    return Skeleton(
        joint_names=source.joint_names,
        parents=source.parents,
        rest_offsets_m=source.rest_offsets_m,
        rest_local_quat_wxyz=source.rest_local_quat_wxyz,
        roles=source.roles,
        markers=source.markers,
        joint_limits={source.roles["left_shoulder"]: spec},
        metadata={**source.metadata, "local_axis_confidence": 0.75},
    )


def test_joint_limit_metric_reports_excess_integral_and_confidence() -> None:
    skeleton = _skeleton_with_left_arm_limit(
        JointLimitSpec(
            representation="euler_xyz",
            x_deg=(-20.0, 20.0),
            y_deg=(-10.0, 10.0),
            z_deg=(-10.0, 10.0),
            confidence=0.8,
            source="test_rig",
        )
    )
    frames = 10
    rotations = np.broadcast_to(
        skeleton.rest_local_quat_wxyz, (frames, skeleton.num_joints, 4)
    ).copy()
    rotations[4:7, skeleton.roles["left_shoulder"]] = axis_angle_to_quaternion(
        [np.deg2rad(35.0), 0.0, 0.0]
    )
    clip = MotionClip(skeleton, rotations, np.zeros((frames, 3)), fps=10.0)
    metric = joint_limit_metric(clip)
    assert metric.clip_value == pytest.approx(15.0, abs=1.0e-5)
    assert metric.confidence == pytest.approx(0.6)
    assert metric.metadata["integrated_violation_degree_seconds"] == pytest.approx(4.5)
    assert len(metric.events) == 1
    assert metric.events[0].frames == (4, 7)
    assert metric.events[0].joints == ("LeftArm",)


def test_incomplete_or_disabled_joint_limit_is_explicitly_unavailable() -> None:
    skeleton = _skeleton_with_left_arm_limit(
        JointLimitSpec(representation="swing_twist", valid=False, confidence=1.0)
    )
    clip = MotionClip(
        skeleton,
        np.broadcast_to(skeleton.rest_local_quat_wxyz, (2, skeleton.num_joints, 4)),
        np.zeros((2, 3)),
        fps=30.0,
    )
    metric = joint_limit_metric(clip)
    assert metric.clip_value is None
    assert metric.confidence == 0.0
    assert metric.metadata["configured_joint_count"] == 1
    assert metric.metadata["invalid_joint_count"] == 1


def test_swing_twist_joint_limits_are_quaternion_sign_invariant() -> None:
    skeleton = _skeleton_with_left_arm_limit(
        JointLimitSpec(
            representation="swing_twist",
            swing_x_deg=(-10.0, 10.0),
            swing_z_deg=(-10.0, 10.0),
            twist_deg=(-15.0, 15.0),
        )
    )
    rotations = np.broadcast_to(
        skeleton.rest_local_quat_wxyz,
        (3, skeleton.num_joints, 4),
    ).copy()
    joint = skeleton.roles["left_shoulder"]
    rotations[1, joint] = axis_angle_to_quaternion([0.0, np.deg2rad(30.0), 0.0])
    clip = MotionClip(skeleton, rotations, np.zeros((3, 3)), fps=30.0)
    metric = joint_limit_metric(clip)
    assert metric.clip_value == pytest.approx(15.0, abs=1.0e-5)

    signed = rotations.copy()
    signed[:, joint] *= -1.0
    signed_metric = joint_limit_metric(MotionClip(skeleton, signed, np.zeros((3, 3)), fps=30.0))
    assert signed_metric.clip_value == pytest.approx(metric.clip_value)
    assert signed_metric.joint_frame_values == metric.joint_frame_values
