from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motionlab.core.clip_metadata import adapt_motion_clip
from motionlab.core.coordinate_contract import CoordinateContract
from motionlab.core.provenance import assign_source_lineage, provenance_from_clip
from motionlab.corruptions.foot_slide import corrupt_foot_slide_root_drift
from motionlab.io.bvh import BvhImportOptions, bvh_to_motion_clip, import_bvh, parse_bvh
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.kinematics.retarget import retarget_rest_relative
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton
from motionlab.processing.canonicalize import canonicalize_origin_and_facing
from motionlab.processing.contacts import ContactConfig, detect_foot_contacts
from motionlab.processing.resample import resample_motion
from motionlab.repair.close_loop import close_loop
from motionlab.repair.foot_lock import lock_foot
from motionlab.rigging.semantic_roles import FunctionalPart, JointSide
from motionlab.testing.synthetic import (
    make_nonidentity_target_humanoid,
    make_synthetic_contact_walk,
    make_synthetic_humanoid,
    make_synthetic_walk,
)

FIXTURE = Path("tests/fixtures/minimal_zxy.bvh")


def test_adapter_is_lossless_and_exposes_fixed_anatomical_metadata() -> None:
    clip = make_synthetic_contact_walk()
    adapter = adapt_motion_clip(clip)
    restored = adapter.to_motion_clip()
    assert restored.content_hash == clip.content_hash
    np.testing.assert_array_equal(restored.local_quat_wxyz, clip.local_quat_wxyz)
    np.testing.assert_array_equal(restored.root_translation_m, clip.root_translation_m)

    pelvis = clip.skeleton.roles["pelvis"]
    left_arm = clip.skeleton.roles["left_shoulder"]
    assert adapter.joints.semantic_role[pelvis] == "pelvis"
    assert adapter.joints.role_aliases[pelvis] == ("pelvis", "root")
    assert adapter.joints.functional_part[pelvis] == FunctionalPart.ROOT_PELVIS
    assert adapter.joints.functional_part[left_arm] == FunctionalPart.LEFT_ARM
    assert adapter.joints.side[left_arm] == JointSide.LEFT
    assert np.all(adapter.joints.is_deform_joint)
    assert np.all(adapter.joints.critic_joint_mask)
    assert adapter.coordinates.source == "motionlab_internal_contract"
    assert adapter.coordinates.canonical_unit == "meter"
    assert adapter.ground.source == "clip_metadata"
    assert np.all(adapter.ground.valid)
    assert not adapter.provenance.lineage_complete
    assert adapter.metadata["neural_input_excludes_provenance"] is True
    with pytest.raises(ValueError):
        adapter.joints.role_confidence[0] = 0.0


def test_coordinate_contract_round_trip_and_bvh_axis_confidence() -> None:
    options = BvhImportOptions(
        handedness="left",
        up_axis="Z",
        forward_axis="Y",
        unit_scale_to_m=0.01,
    )
    clip = import_bvh(
        FIXTURE,
        options=options,
        source_dataset="fixture_dataset",
        source_clip_id="minimal",
        source_take_id="take_001",
        split="test",
    )
    adapter = adapt_motion_clip(clip)
    vector = np.array([3.0, -2.0, 7.0])
    canonical = adapter.coordinates.source_to_canonical(vector)
    np.testing.assert_allclose(adapter.coordinates.canonical_to_source(canonical), vector)
    assert adapter.coordinates.source_handedness == "left"
    assert adapter.coordinates.source_up_axis == "Z"
    assert adapter.coordinates.source_forward_axis == "Y"
    assert np.all(adapter.joints.axis_confidence == pytest.approx(0.25))
    assert adapter.pose_reference.rest_kind == "bvh_zero_channel_rest"
    assert adapter.pose_reference.rest_confidence == pytest.approx(0.25)
    assert adapter.provenance.source_dataset == "fixture_dataset"
    assert adapter.provenance.source_take_id == "take_001"
    assert adapter.provenance.split == "test"
    assert adapter.provenance.lineage_complete


def test_bvh_missing_rotation_channels_have_explicit_invalid_mask() -> None:
    text = """\
HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
  JOINT StaticChild
  {
    OFFSET 0 1 0
    CHANNELS 0
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333333
0 0 0 0 0 0
0 0 1 0 0 0
"""
    clip = bvh_to_motion_clip(parse_bvh(text))
    adapter = adapt_motion_clip(clip)
    assert np.all(adapter.modalities.rotation_valid[:, 0])
    assert not np.any(adapter.modalities.rotation_valid[:, 1])
    assert np.all(adapter.modalities.rotation_confidence[:, 1] == 0.0)
    assert np.all(adapter.modalities.position_valid)
    assert adapter.joints.semantic_role[1] == "UNKNOWN"
    assert adapter.joints.functional_part[1] == FunctionalPart.OTHER_ACCESSORY
    assert adapter.joints.side[1] == JointSide.UNKNOWN
    assert adapter.joints.role_confidence[1] == 0.0
    assert not np.any(adapter.ground.valid)
    assert adapter.ground.source == "unavailable"


def test_contact_and_ground_confidence_remain_separate_modalities() -> None:
    clip = make_synthetic_contact_walk()
    contacts = detect_foot_contacts(
        clip,
        config=ContactConfig(
            speed_enter_mps=0.6,
            speed_exit_mps=0.7,
            ground_confidence=0.7,
        ),
    )
    adapter = adapt_motion_clip(clip, contacts=contacts)
    assert adapter.modalities.contact_source == "heuristic_height_speed"
    assert adapter.modalities.contact_valid is not None
    assert adapter.modalities.contact_confidence is not None
    assert np.all(adapter.modalities.contact_valid)
    np.testing.assert_array_equal(adapter.modalities.contact_confidence, contacts.confidence)
    assert adapter.ground.source == "contact_result:heuristic_height_speed"
    assert np.all(adapter.ground.confidence == pytest.approx(0.7))


def test_invalid_modality_is_zero_confidence_and_bad_joint_flags_are_rejected() -> None:
    clip = make_synthetic_contact_walk()
    valid = np.ones(clip.num_joints, dtype=np.bool_)
    valid[clip.skeleton.roles["left_wrist"]] = False
    adapter = adapt_motion_clip(clip, rotation_valid=valid)
    assert np.all(
        adapter.modalities.rotation_confidence[:, clip.skeleton.roles["left_wrist"]] == 0.0
    )

    source = make_synthetic_humanoid()
    skeleton = Skeleton(
        joint_names=source.joint_names,
        parents=source.parents,
        rest_offsets_m=source.rest_offsets_m,
        rest_local_quat_wxyz=source.rest_local_quat_wxyz,
        roles=source.roles,
        markers=source.markers,
        metadata={
            **source.metadata,
            "joint_flags": {"LeftHand": {"is_controller": True, "is_deform_joint": True}},
        },
    )
    bad_clip = MotionClip(
        skeleton,
        np.broadcast_to(skeleton.rest_local_quat_wxyz, (2, skeleton.num_joints, 4)),
        np.zeros((2, 3)),
        fps=30.0,
    )
    with pytest.raises(ValueError, match="controller joints"):
        adapt_motion_clip(bad_clip)


def test_per_frame_ground_planes_and_confidence_are_normalized_and_preserved() -> None:
    clip = make_synthetic_contact_walk()
    planes = np.zeros((clip.num_frames, 4), dtype=np.float32)
    planes[:, 1] = 2.0
    planes[:, 3] = np.linspace(0.0, -0.1, clip.num_frames)
    confidence = np.linspace(0.5, 0.9, clip.num_frames, dtype=np.float32)
    adapter = adapt_motion_clip(
        clip,
        ground_plane=planes,
        ground_source="per_frame_fixture",
        ground_confidence=confidence,
    )
    assert adapter.ground.plane is not None
    assert adapter.ground.plane.shape == (clip.num_frames, 4)
    np.testing.assert_allclose(
        np.linalg.norm(adapter.ground.plane[:, :3], axis=-1),
        1.0,
    )
    np.testing.assert_allclose(adapter.ground.confidence, confidence)
    assert adapter.ground.source == "per_frame_fixture"


def test_optional_bind_and_pose_anchor_data_are_distinct_and_validated() -> None:
    source = make_synthetic_humanoid()
    anchor = np.cumsum(source.rest_offsets_m, axis=0)
    skeleton = Skeleton(
        joint_names=source.joint_names,
        parents=source.parents,
        rest_offsets_m=source.rest_offsets_m,
        rest_local_quat_wxyz=source.rest_local_quat_wxyz,
        roles=source.roles,
        markers=source.markers,
        metadata={
            **source.metadata,
            "pose_reference": {
                "rest_kind": "authored_rest",
                "rest_confidence": 0.9,
                "bind_local_quat_wxyz": source.rest_local_quat_wxyz.tolist(),
                "bind_confidence": 0.8,
                "pose_anchor_kind": "neutral_reference",
                "pose_anchor_confidence": 0.7,
                "pose_anchor_position_m": anchor.tolist(),
            },
        },
    )
    clip = MotionClip(
        skeleton,
        np.broadcast_to(skeleton.rest_local_quat_wxyz, (2, skeleton.num_joints, 4)),
        np.zeros((2, 3)),
        fps=30.0,
    )
    pose = adapt_motion_clip(clip).pose_reference
    assert pose.rest_kind == "authored_rest"
    assert pose.bind_available
    assert pose.bind_confidence == pytest.approx(0.8)
    assert pose.bind_local_quat_wxyz is not None
    assert pose.pose_anchor_kind == "neutral_reference"
    assert pose.pose_anchor_position_m is not None


def test_source_split_lineage_survives_corruption_repair_retarget_and_resample(
    tmp_path: Path,
) -> None:
    original = make_synthetic_contact_walk()
    source = assign_source_lineage(
        original,
        source_dataset="generated_fixture",
        source_clip_id="walk_family_01",
        source_take_id="take_01",
        split="train",
        importer_version="fixture-1",
        clean_confidence=0.95,
        metadata={"nested": {"immutable": True}},
    )
    with pytest.raises(ValueError, match="already been assigned"):
        assign_source_lineage(
            source,
            source_dataset="other",
            source_clip_id="other",
            source_take_id="other",
        )

    corruption = corrupt_foot_slide_root_drift(
        source,
        start_frame=5,
        stop_frame=25,
        distance_m=0.04,
        seed=19,
    )
    repaired = lock_foot(
        corruption.corrupted,
        side="left",
        start_frame=5,
        stop_frame=25,
    ).repaired
    retargeted = retarget_rest_relative(repaired, make_nonidentity_target_humanoid())
    resampled = resample_motion(retargeted, 30.0)
    sliced = resampled.slice_frames(2, 20)
    provenance = provenance_from_clip(sliced)

    assert provenance.source_dataset == "generated_fixture"
    assert provenance.source_take_id == "take_01"
    assert provenance.split == "train"
    assert provenance.clean_confidence == pytest.approx(0.95)
    assert provenance.lineage_complete
    assert [step.kind for step in provenance.operations] == [
        "source",
        "corruption",
        "repair",
        "retarget",
        "transform",
        "slice",
    ]
    assert provenance.operations[1].parent_motion_id == source.content_hash
    assert provenance.operations[2].parent_motion_id == corruption.corrupted.content_hash
    assert provenance.operations[-1].parent_motion_id == resampled.content_hash
    assert len(provenance.corruption_lineage) == 1
    assert provenance.corruption_lineage[0].seed == 19
    assert provenance.retargeter_version == "motionlab-0.1.0"
    with pytest.raises(TypeError):
        provenance.metadata["nested"]["immutable"] = False

    path = tmp_path / "lineaged.npz"
    save_motion_npz(path, sliced)
    loaded = load_motion_npz(path)
    assert loaded.content_hash == sliced.content_hash
    loaded_provenance = provenance_from_clip(loaded)
    assert loaded_provenance.split_lineage_id == provenance.split_lineage_id
    assert [step.to_json() for step in loaded_provenance.operations] == [
        step.to_json() for step in provenance.operations
    ]


def test_coordinate_contract_rejects_non_orthonormal_transform() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        CoordinateContract(
            source_handedness="right",
            source_up_axis="Y",
            source_forward_axis="Z",
            source_unit_scale_to_m=1.0,
            source_to_canonical_matrix=np.diag([2.0, 1.0, 1.0]),
            source="test",
        )


def test_canonicalize_and_loop_repair_append_typed_lineage() -> None:
    source = assign_source_lineage(
        make_synthetic_walk(num_frames=31, fps=30.0),
        source_dataset="generated_fixture",
        source_clip_id="walk",
        source_take_id="walk_02",
        split="validation",
    )
    canonical = canonicalize_origin_and_facing(source, forward_world=(1.0, 0.0, 0.0))
    repaired = close_loop(canonical, seam_window_frames=6).repaired
    provenance = provenance_from_clip(repaired)
    assert [step.kind for step in provenance.operations] == ["source", "transform", "repair"]
    assert provenance.operations[1].operation == "canonicalize_origin_and_facing"
    assert provenance.operations[2].operation == "close_loop"
    assert provenance.operations[1].parent_motion_id == source.content_hash
    assert provenance.operations[2].parent_motion_id == canonical.content_hash
