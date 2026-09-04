"""Vectorized forward kinematics for NumPy and Torch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from motionlab.math.quaternion import quaternion_multiply, quaternion_rotate_vector
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton


def forward_kinematics_numpy(
    skeleton: Skeleton,
    local_quat_wxyz: ArrayLike,
    root_translation_m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate global positions and rotations for arbitrary leading batch dimensions.

    Args:
        skeleton: Hierarchy and parent-relative rest offsets.
        local_quat_wxyz: Absolute local orientations with shape ``[..., J, 4]``.
        root_translation_m: World root positions with shape ``[..., 3]``.

    Returns:
        A tuple ``(global_position_m, global_quat_wxyz)`` with shapes ``[..., J, 3]`` and
        ``[..., J, 4]``.
    """
    local = np.asarray(local_quat_wxyz, dtype=np.float64)
    root_translation = np.asarray(root_translation_m, dtype=np.float64)
    if local.ndim < 2 or local.shape[-2:] != (skeleton.num_joints, 4):
        raise ValueError(
            f"local_quat_wxyz must end in ({skeleton.num_joints}, 4), got {local.shape}"
        )
    if root_translation.ndim < 1 or root_translation.shape[-1] != 3:
        raise ValueError(f"root_translation_m must end in (3,), got {root_translation.shape}")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(root_translation)):
        raise ValueError("FK inputs must be finite")

    leading_shape = np.broadcast_shapes(local.shape[:-2], root_translation.shape[:-1])
    local = np.broadcast_to(local, (*leading_shape, skeleton.num_joints, 4))
    root_translation = np.broadcast_to(root_translation, (*leading_shape, 3))
    positions = np.empty((*leading_shape, skeleton.num_joints, 3), dtype=np.float64)
    rotations = np.empty((*leading_shape, skeleton.num_joints, 4), dtype=np.float64)

    for joint in skeleton.topological_order:
        parent = int(skeleton.parents[joint])
        if parent == -1:
            positions[..., joint, :] = root_translation
            rotations[..., joint, :] = local[..., joint, :]
        else:
            parent_rotation = rotations[..., parent, :]
            positions[..., joint, :] = positions[..., parent, :] + quaternion_rotate_vector(
                parent_rotation,
                skeleton.rest_offsets_m[joint],
            )
            rotations[..., joint, :] = quaternion_multiply(
                parent_rotation,
                local[..., joint, :],
            )
    return positions, rotations


def _torch_quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _torch_quaternion_rotate_vector(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )
    vector_part = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(vector_part, vector, dim=-1)
    return cast(
        torch.Tensor,
        vector
        + quaternion[..., :1] * twice_cross
        + torch.linalg.cross(vector_part, twice_cross, dim=-1),
    )


def forward_kinematics_torch(
    skeleton: Skeleton,
    local_quat_wxyz: torch.Tensor,
    root_translation_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably calculate global positions and rotations with Torch tensors."""
    if local_quat_wxyz.ndim < 2 or local_quat_wxyz.shape[-2:] != (
        skeleton.num_joints,
        4,
    ):
        raise ValueError(
            "local_quat_wxyz must end in "
            f"({skeleton.num_joints}, 4), got {tuple(local_quat_wxyz.shape)}"
        )
    if root_translation_m.ndim < 1 or root_translation_m.shape[-1] != 3:
        raise ValueError(
            f"root_translation_m must end in (3,), got {tuple(root_translation_m.shape)}"
        )
    if local_quat_wxyz.device != root_translation_m.device:
        raise ValueError("local rotations and root translations must share a device")
    if local_quat_wxyz.dtype != root_translation_m.dtype:
        raise ValueError("local rotations and root translations must share a dtype")

    leading_shape = torch.broadcast_shapes(
        local_quat_wxyz.shape[:-2],
        root_translation_m.shape[:-1],
    )
    local = torch.broadcast_to(
        local_quat_wxyz,
        (*leading_shape, skeleton.num_joints, 4),
    )
    root_translation = torch.broadcast_to(root_translation_m, (*leading_shape, 3))
    offset = torch.as_tensor(
        skeleton.rest_offsets_m.copy(),
        dtype=local.dtype,
        device=local.device,
    )
    positions: dict[int, torch.Tensor] = {}
    rotations: dict[int, torch.Tensor] = {}

    for joint in skeleton.topological_order:
        parent = int(skeleton.parents[joint])
        if parent == -1:
            positions[joint] = root_translation
            rotations[joint] = local[..., joint, :]
        else:
            parent_rotation = rotations[parent]
            joint_offset = offset[joint].expand((*leading_shape, 3))
            positions[joint] = positions[parent] + _torch_quaternion_rotate_vector(
                parent_rotation,
                joint_offset,
            )
            rotations[joint] = _torch_quaternion_multiply(
                parent_rotation,
                local[..., joint, :],
            )

    return (
        torch.stack([positions[index] for index in range(skeleton.num_joints)], dim=-2),
        torch.stack([rotations[index] for index in range(skeleton.num_joints)], dim=-2),
    )


def marker_world_positions(
    clip: MotionClip,
    marker_names: tuple[str, ...] | None = None,
) -> Mapping[str, NDArray[np.float64]]:
    """Evaluate configured virtual markers in world space for every clip frame."""
    positions, rotations = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    selected = tuple(clip.skeleton.markers) if marker_names is None else marker_names
    result: dict[str, NDArray[np.float64]] = {}
    for marker_name in selected:
        try:
            marker = clip.skeleton.markers[marker_name]
        except KeyError as exc:
            raise ValueError(f"unknown marker {marker_name!r}") from exc
        joint = clip.skeleton.resolve_joint(marker.joint)
        result[marker_name] = positions[:, joint] + quaternion_rotate_vector(
            rotations[:, joint],
            marker.local_offset_m,
        )
    return result
