from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from motionlab.io.bvh import BvhImportOptions, bvh_to_motion_clip, parse_bvh
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_rotate_vector,
)
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton
from motionlab.rigging.graphs import (
    DYNAMIC_RIG_CONTRACT,
    DYNAMIC_UNIVERSAL_CONTRACT,
    GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT,
    LOCAL_BASIS_CONVENTION,
    STATIC_RIG_CONTRACT,
    STATIC_UNIVERSAL_CONTRACT,
    batch_motion_graphs,
    build_motion_graph,
)
from motionlab.rigging.semantic_roles import FunctionalPart, JointSide
from motionlab.testing.synthetic import make_synthetic_walk


def _permuted_clip(clip: MotionClip, order: Sequence[int]) -> MotionClip:
    permutation = np.asarray(order, dtype=np.int32)
    inverse = np.empty(clip.num_joints, dtype=np.int32)
    inverse[permutation] = np.arange(clip.num_joints, dtype=np.int32)
    skeleton = clip.skeleton
    parents = np.asarray(
        [
            -1 if skeleton.parents[joint] < 0 else inverse[skeleton.parents[joint]]
            for joint in permutation
        ],
        dtype=np.int32,
    )
    remapped = Skeleton(
        joint_names=tuple(skeleton.joint_names[joint] for joint in permutation),
        parents=parents,
        rest_offsets_m=skeleton.rest_offsets_m[permutation],
        rest_local_quat_wxyz=skeleton.rest_local_quat_wxyz[permutation],
        roles={role: int(inverse[joint]) for role, joint in skeleton.roles.items()},
        markers=skeleton.markers,
        joint_limits={int(inverse[joint]): limit for joint, limit in skeleton.joint_limits.items()},
        metadata=skeleton.metadata,
    )
    return MotionClip(
        skeleton=remapped,
        local_quat_wxyz=clip.local_quat_wxyz[:, permutation],
        root_translation_m=clip.root_translation_m,
        fps=clip.fps,
        metadata=clip.metadata,
    )


def _short_upper_body_clip(clip: MotionClip) -> MotionClip:
    selected = np.asarray([0, 1, 2, 3], dtype=np.int32)
    skeleton = Skeleton(
        joint_names=tuple(clip.skeleton.joint_names[index] for index in selected),
        parents=np.asarray([-1, 0, 1, 2], dtype=np.int32),
        rest_offsets_m=clip.skeleton.rest_offsets_m[selected],
        rest_local_quat_wxyz=clip.skeleton.rest_local_quat_wxyz[selected],
        roles={
            role: int(np.flatnonzero(selected == joint)[0])
            for role, joint in clip.skeleton.roles.items()
            if joint in selected
        },
        metadata={"fixture": "short_upper_body_variable_rig"},
    )
    return MotionClip(
        skeleton=skeleton,
        local_quat_wxyz=clip.local_quat_wxyz[:, selected],
        root_translation_m=clip.root_translation_m,
        fps=clip.fps,
        metadata=clip.metadata,
    )


def _active_rows(graph: object, values: np.ndarray) -> dict[str, np.ndarray]:
    skeleton = graph.skeleton  # type: ignore[attr-defined]
    return {
        key: values[..., index, :]
        for index, key in enumerate(skeleton.node_keys)
        if skeleton.model_joint_mask[index]
    }


def test_joint_permutation_canonicalizes_static_dynamic_and_topology() -> None:
    clip = make_synthetic_walk(num_frames=23, fps=30.0)
    order = np.random.default_rng(81).permutation(clip.num_joints)
    original = build_motion_graph(clip)
    permuted = build_motion_graph(_permuted_clip(clip, order))

    assert permuted.skeleton.node_keys == original.skeleton.node_keys
    assert permuted.skeleton.anatomical_edge_keys == original.skeleton.anatomical_edge_keys
    np.testing.assert_array_equal(permuted.skeleton.parents, original.skeleton.parents)
    np.testing.assert_allclose(
        permuted.skeleton.static_universal, original.skeleton.static_universal
    )
    np.testing.assert_allclose(permuted.skeleton.static_rig, original.skeleton.static_rig)
    np.testing.assert_allclose(permuted.dynamic_universal, original.dynamic_universal)
    np.testing.assert_allclose(permuted.dynamic_rig, original.dynamic_rig)
    assert permuted.skeleton.content_hash == original.skeleton.content_hash


def test_variable_joint_batch_padding_is_zero_and_fully_masked() -> None:
    full_clip = make_synthetic_walk(num_frames=19, fps=30.0)
    short_clip = _short_upper_body_clip(full_clip).slice_frames(0, 11)
    short_graph = build_motion_graph(short_clip)
    full_graph = build_motion_graph(full_clip)
    batch = batch_motion_graphs((short_graph, full_graph))

    assert batch.dynamic_universal.shape == (
        2,
        19,
        16,
        DYNAMIC_UNIVERSAL_CONTRACT.width,
    )
    assert np.all(batch.joint_mask[0, :4])
    assert not np.any(batch.joint_mask[0, 4:])
    assert np.all(batch.frame_mask[0, :11])
    assert not np.any(batch.frame_mask[0, 11:])
    assert not np.any(batch.node_frame_mask[0, 11:, :])
    assert not np.any(batch.node_frame_mask[0, :, 4:])
    np.testing.assert_array_equal(
        batch.dynamic_universal[0, :11, :4], short_graph.dynamic_universal
    )
    assert np.count_nonzero(batch.dynamic_universal[0, 11:]) == 0
    assert np.count_nonzero(batch.dynamic_universal[0, :, 4:]) == 0
    assert np.count_nonzero(batch.rotation_confidence[0, :, 4:]) == 0
    assert np.all(batch.edge_index[0, :, ~batch.edge_mask[0]] == -1)
    assert not batch.static_universal.flags.writeable


def _with_identity_helper(clip: MotionClip) -> MotionClip:
    source = clip.skeleton
    helper = source.num_joints
    head = source.roles["head"]
    chest = int(source.parents[head])
    names = (*source.joint_names, "HeadIdentityHelper")
    parents = np.concatenate((source.parents, np.asarray([chest], dtype=np.int32)))
    parents[head] = helper
    offsets = np.concatenate(
        (source.rest_offsets_m, np.zeros((1, 3), dtype=np.float32)),
        axis=0,
    )
    rest = np.concatenate(
        (
            source.rest_local_quat_wxyz,
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        ),
        axis=0,
    )
    skeleton = Skeleton(
        joint_names=names,
        parents=parents,
        rest_offsets_m=offsets,
        rest_local_quat_wxyz=rest,
        roles=source.roles,
        markers=source.markers,
        joint_limits=source.joint_limits,
        metadata={
            **source.metadata,
            "joint_flags": {
                "HeadIdentityHelper": {
                    "is_helper_joint": True,
                    "is_deform_joint": False,
                    "critic_joint_mask": False,
                    "repair_joint_mask": False,
                }
            },
        },
    )
    identity = np.broadcast_to(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (clip.num_frames, 1, 4),
    )
    return MotionClip(
        skeleton=skeleton,
        local_quat_wxyz=np.concatenate((clip.local_quat_wxyz, identity), axis=1),
        root_translation_m=clip.root_translation_m,
        fps=clip.fps,
        metadata=clip.metadata,
    )


def test_identity_helper_is_explicitly_masked_and_anatomically_collapsed() -> None:
    original = build_motion_graph(make_synthetic_walk(num_frames=17))
    with_helper = build_motion_graph(_with_identity_helper(make_synthetic_walk(num_frames=17)))
    helper_index = with_helper.skeleton.node_keys.index("joint:headidentityhelper")

    assert with_helper.skeleton.is_helper_joint[helper_index]
    assert not with_helper.skeleton.model_joint_mask[helper_index]
    assert with_helper.skeleton.functional_part[helper_index] == FunctionalPart.HEAD_NECK
    assert with_helper.skeleton.side[helper_index] == JointSide.CENTER
    assert with_helper.skeleton.active_node_keys == original.skeleton.active_node_keys
    assert with_helper.skeleton.anatomical_edge_keys == original.skeleton.anatomical_edge_keys
    original_universal = _active_rows(original, original.dynamic_universal)
    helper_universal = _active_rows(with_helper, with_helper.dynamic_universal)
    original_rig = _active_rows(original, original.dynamic_rig)
    helper_rig = _active_rows(with_helper, with_helper.dynamic_rig)
    for key in original.skeleton.active_node_keys:
        np.testing.assert_allclose(helper_universal[key], original_universal[key])
        np.testing.assert_allclose(helper_rig[key], original_rig[key])


def _local_axis_reparameterization(clip: MotionClip) -> MotionClip:
    joint_count = clip.num_joints
    tangent = np.zeros((joint_count, 3), dtype=np.float64)
    tangent[:, 1] = np.linspace(-0.31, 0.29, joint_count)
    tangent[:, 2] = np.linspace(0.17, -0.13, joint_count)
    frame_change = axis_angle_to_quaternion(tangent)
    declared_to_reference = quaternion_inverse(frame_change)
    local = np.empty_like(clip.local_quat_wxyz)
    rest = np.empty_like(clip.skeleton.rest_local_quat_wxyz)
    offsets = clip.skeleton.rest_offsets_m.copy()
    for joint, parent in enumerate(clip.skeleton.parents):
        if parent < 0:
            local[:, joint] = quaternion_multiply(
                clip.local_quat_wxyz[:, joint], frame_change[joint]
            )
            rest[joint] = quaternion_multiply(
                clip.skeleton.rest_local_quat_wxyz[joint], frame_change[joint]
            )
        else:
            parent_change_inverse = quaternion_inverse(frame_change[int(parent)])
            local[:, joint] = quaternion_multiply(
                quaternion_multiply(parent_change_inverse, clip.local_quat_wxyz[:, joint]),
                frame_change[joint],
            )
            rest[joint] = quaternion_multiply(
                quaternion_multiply(
                    parent_change_inverse,
                    clip.skeleton.rest_local_quat_wxyz[joint],
                ),
                frame_change[joint],
            )
            offsets[joint] = quaternion_rotate_vector(parent_change_inverse, offsets[joint])
    source = clip.skeleton
    skeleton = Skeleton(
        joint_names=source.joint_names,
        parents=source.parents,
        rest_offsets_m=offsets,
        rest_local_quat_wxyz=rest,
        roles=source.roles,
        markers=source.markers,
        joint_limits=source.joint_limits,
        metadata={
            **source.metadata,
            "local_axis_basis": {
                "convention": LOCAL_BASIS_CONVENTION,
                "declared_to_reference_quat_wxyz": declared_to_reference.tolist(),
                "confidence": 1.0,
                "source": "test_gauge_transform",
            },
        },
    )
    return MotionClip(
        skeleton=skeleton,
        local_quat_wxyz=local,
        root_translation_m=clip.root_translation_m,
        fps=clip.fps,
        metadata=clip.metadata,
    )


def test_local_axis_reparameterization_preserves_both_normalized_streams() -> None:
    clip = make_synthetic_walk(num_frames=29, fps=30.0)
    original = build_motion_graph(clip)
    changed = build_motion_graph(_local_axis_reparameterization(clip))

    assert changed.skeleton.node_keys == original.skeleton.node_keys
    np.testing.assert_allclose(
        changed.skeleton.static_universal, original.skeleton.static_universal, atol=2e-6
    )
    np.testing.assert_allclose(changed.skeleton.static_rig, original.skeleton.static_rig, atol=2e-6)
    np.testing.assert_allclose(changed.dynamic_universal, original.dynamic_universal, atol=2e-5)
    np.testing.assert_allclose(changed.dynamic_rig, original.dynamic_rig, atol=2e-5)
    assert changed.skeleton.basis.source == "test_gauge_transform"
    assert changed.skeleton.basis.convention == LOCAL_BASIS_CONVENTION


def _two_joint_bvh(*, offset: str, frames: Sequence[str]) -> str:
    frame_lines = "\n".join(frames)
    return f"""\
HIERARCHY
ROOT Root
{{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
  JOINT Head
  {{
    OFFSET {offset}
    CHANNELS 3 Xrotation Yrotation Zrotation
    End Site
    {{
      OFFSET 0 0 0
    }}
  }}
}}
MOTION
Frames: 3
Frame Time: 0.0333333333333333
{frame_lines}
"""


def test_unit_and_axis_import_canonicalization_produces_same_graph_features() -> None:
    canonical = bvh_to_motion_clip(
        parse_bvh(
            _two_joint_bvh(
                offset="0 1 0",
                frames=(
                    "0 1 0 10 0 0 5 0 0",
                    "0 1 0.5 10 0 0 5 0 0",
                    "0 1 1 10 0 0 5 0 0",
                ),
            )
        ),
        options=BvhImportOptions(
            handedness="right",
            up_axis="Y",
            forward_axis="Z",
            unit_scale_to_m=1.0,
        ),
        role_names={"root": "Root", "head": "Head"},
    )
    centimeters_z_up = bvh_to_motion_clip(
        parse_bvh(
            _two_joint_bvh(
                offset="0 0 100",
                frames=(
                    "0 0 100 -10 0 0 -5 0 0",
                    "0 50 100 -10 0 0 -5 0 0",
                    "0 100 100 -10 0 0 -5 0 0",
                ),
            )
        ),
        options=BvhImportOptions(
            handedness="right",
            up_axis="Z",
            forward_axis="Y",
            unit_scale_to_m=0.01,
        ),
        role_names={"root": "Root", "head": "Head"},
    )
    first = build_motion_graph(canonical)
    second = build_motion_graph(centimeters_z_up)

    assert first.skeleton.node_keys == second.skeleton.node_keys
    np.testing.assert_allclose(second.skeleton.static_universal, first.skeleton.static_universal)
    np.testing.assert_allclose(second.skeleton.static_rig, first.skeleton.static_rig)
    np.testing.assert_allclose(second.dynamic_universal, first.dynamic_universal)
    np.testing.assert_allclose(second.dynamic_rig, first.dynamic_rig)
    np.testing.assert_allclose(second.global_dynamic_universal, first.global_dynamic_universal)
    assert second.skeleton.basis.coordinates.source_unit_scale_to_m == 0.01
    assert second.skeleton.basis.coordinates.source_up_axis == "Z"
    assert second.skeleton.basis.coordinates.canonical_unit == "meter"


def test_feature_contracts_keep_static_dynamic_and_coordinate_streams_distinct() -> None:
    graph = build_motion_graph(make_synthetic_walk(num_frames=7))
    assert graph.skeleton.static_universal.shape[-1] == STATIC_UNIVERSAL_CONTRACT.width
    assert graph.skeleton.static_rig.shape[-1] == STATIC_RIG_CONTRACT.width
    assert graph.dynamic_universal.shape[-1] == DYNAMIC_UNIVERSAL_CONTRACT.width
    assert graph.dynamic_rig.shape[-1] == DYNAMIC_RIG_CONTRACT.width
    assert graph.global_dynamic_universal.shape[-1] == GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT.width
    assert STATIC_UNIVERSAL_CONTRACT.field_slice("bone_length_m").stop == 7
    assert DYNAMIC_RIG_CONTRACT.time_varying
    assert not STATIC_RIG_CONTRACT.time_varying
    assert not graph.dynamic_universal.flags.writeable
    assert not graph.skeleton.static_rig.flags.writeable
