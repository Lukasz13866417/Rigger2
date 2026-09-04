from __future__ import annotations

import numpy as np
import torch

from motionlab.kinematics.fk import (
    forward_kinematics_numpy,
    forward_kinematics_torch,
    marker_world_positions,
)
from motionlab.math.quaternion import axis_angle_to_quaternion
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton
from motionlab.testing.synthetic import make_synthetic_walk


def make_chain() -> Skeleton:
    return Skeleton(
        joint_names=("root", "middle", "tip"),
        parents=np.array([-1, 0, 1], dtype=np.int32),
        rest_offsets_m=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        rest_local_quat_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0]] * 3,
            dtype=np.float32,
        ),
    )


def test_exact_three_link_chain() -> None:
    skeleton = make_chain()
    local = skeleton.rest_local_quat_wxyz.copy()
    local[1] = axis_angle_to_quaternion([0.0, 0.0, np.pi / 2.0])
    positions, rotations = forward_kinematics_numpy(skeleton, local, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        positions,
        [[2.0, 3.0, 4.0], [3.0, 3.0, 4.0], [3.0, 4.0, 4.0]],
        atol=1e-12,
    )
    assert rotations.shape == (3, 4)


def test_numpy_batch_and_torch_agree_and_torch_backpropagates() -> None:
    skeleton = make_chain()
    local = np.broadcast_to(skeleton.rest_local_quat_wxyz, (4, 3, 4)).copy()
    local[:, 0] = axis_angle_to_quaternion(
        np.stack((np.zeros(4), np.linspace(0.0, 0.5, 4), np.zeros(4)), axis=-1)
    )
    root = np.arange(12, dtype=np.float64).reshape(4, 3) * 0.1
    numpy_position, numpy_rotation = forward_kinematics_numpy(skeleton, local, root)

    torch_local = torch.tensor(local, dtype=torch.float64, requires_grad=True)
    torch_root = torch.tensor(root, dtype=torch.float64, requires_grad=True)
    torch_position, torch_rotation = forward_kinematics_torch(
        skeleton,
        torch_local,
        torch_root,
    )
    np.testing.assert_allclose(torch_position.detach().numpy(), numpy_position, atol=1e-12)
    np.testing.assert_allclose(torch_rotation.detach().numpy(), numpy_rotation, atol=1e-12)
    (torch_position.square().sum() + torch_rotation.square().sum()).backward()
    assert torch_local.grad is not None
    assert torch_root.grad is not None
    assert torch.isfinite(torch_local.grad).all()


def test_marker_positions_match_fk_attachment() -> None:
    clip = make_synthetic_walk(num_frames=8)
    markers = marker_world_positions(clip)
    assert set(markers) == {"left_heel", "left_toe", "right_heel", "right_toe"}
    assert all(value.shape == (8, 3) for value in markers.values())
    assert all(np.isfinite(value).all() for value in markers.values())


def test_clip_rest_pose_root_translation() -> None:
    skeleton = make_chain()
    clip = MotionClip(
        skeleton=skeleton,
        local_quat_wxyz=np.broadcast_to(skeleton.rest_local_quat_wxyz, (2, 3, 4)),
        root_translation_m=np.array([[0.0, 0.0, 0.0], [5.0, -2.0, 1.0]], dtype=np.float32),
        fps=60.0,
    )
    positions, _ = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    expected_delta = np.broadcast_to([5.0, -2.0, 1.0], (3, 3))
    np.testing.assert_allclose(positions[1] - positions[0], expected_delta, atol=1e-12)
