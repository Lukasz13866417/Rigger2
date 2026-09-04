from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motionlab.io.bvh import (
    BvhImportOptions,
    BvhParseError,
    bvh_to_motion_clip,
    load_bvh,
    parse_bvh,
)
from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.quaternion import (
    axis_angle_to_quaternion,
    quaternion_geodesic_distance,
    quaternion_multiply,
    quaternion_rotate_vector,
)
from motionlab.motion.markers import JointLimitSpec

FIXTURE = Path("tests/fixtures/minimal_zxy.bvh")


def test_parser_preserves_hierarchy_channel_order_and_end_site() -> None:
    data = load_bvh(FIXTURE)
    assert data.joint_names == ("Hips", "Knee")
    np.testing.assert_array_equal(data.parents, [-1, 0])
    np.testing.assert_allclose(data.offsets, [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert data.channels[0] == (
        "Xposition",
        "Yposition",
        "Zposition",
        "Zrotation",
        "Xrotation",
        "Yrotation",
    )
    assert data.frame_time_s == pytest.approx(0.1)
    assert data.num_frames == 2
    assert data.num_channels == 9
    assert data.end_sites[0].parent == 1
    assert data.end_sites[0].offset == (0.0, -1.0, 0.0)


def test_bvh_conversion_respects_root_translation_and_zxy_rotation() -> None:
    clip = bvh_to_motion_clip(load_bvh(FIXTURE))
    assert clip.fps == pytest.approx(10.0)
    np.testing.assert_allclose(clip.root_translation_m[1], [1.0, 2.0, 3.0])
    positions, _ = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    np.testing.assert_allclose(positions[1, 1], [2.0, 2.0, 3.0], atol=1.0e-6)
    assert clip.skeleton.metadata["bvh_channels"][0][3:] == [
        "Zrotation",
        "Xrotation",
        "Yrotation",
    ]


def test_import_options_map_axes_and_units_explicitly() -> None:
    options = BvhImportOptions(
        handedness="right",
        up_axis="Z",
        forward_axis="Y",
        unit_scale_to_m=0.01,
    )
    matrix = options.source_to_canonical_matrix
    np.testing.assert_allclose(matrix @ [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(matrix @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    clip = bvh_to_motion_clip(load_bvh(FIXTURE), options=options)
    np.testing.assert_allclose(clip.root_translation_m[1], [-0.01, 0.03, 0.02])

    source_vector = np.array([0.3, -0.4, 0.5])
    canonical_vector = matrix @ source_vector
    np.testing.assert_allclose(matrix.T @ canonical_vector, source_vector)


def test_malformed_bvh_reports_context() -> None:
    malformed = FIXTURE.read_text().replace("Frames: 2", "Frames: 3")
    with pytest.raises(BvhParseError, match="motion values"):
        parse_bvh(malformed, source="broken.bvh")


def test_nonroot_translation_channels_are_rejected() -> None:
    text = FIXTURE.read_text().replace(
        "CHANNELS 3 Zrotation Xrotation Yrotation",
        "CHANNELS 4 Xposition Zrotation Xrotation Yrotation",
    )
    text = text.replace(
        "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
        "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
    )
    text = text.replace(
        "1.0 2.0 3.0 90.0 0.0 0.0 0.0 0.0 0.0",
        "1.0 2.0 3.0 90.0 0.0 0.0 0.0 0.0 0.0 0.0",
    )
    with pytest.raises(ValueError, match="non-root position"):
        bvh_to_motion_clip(parse_bvh(text))


def test_quaternion_output_rotates_as_expected() -> None:
    clip = bvh_to_motion_clip(load_bvh(FIXTURE))
    rotated = quaternion_rotate_vector(clip.local_quat_wxyz[1, 0], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(rotated, [1.0, 0.0, 0.0], atol=1.0e-6)


def test_nondefault_xyz_euler_order_is_preserved() -> None:
    text = """\
HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
}
MOTION
Frames: 1
Frame Time: 0.0166666667
0 0 0 20 30 40
"""
    clip = bvh_to_motion_clip(parse_bvh(text))
    angles = np.deg2rad([20.0, 30.0, 40.0])
    rotation_x = axis_angle_to_quaternion([angles[0], 0.0, 0.0])
    rotation_y = axis_angle_to_quaternion([0.0, angles[1], 0.0])
    rotation_z = axis_angle_to_quaternion([0.0, 0.0, angles[2]])
    expected = quaternion_multiply(quaternion_multiply(rotation_x, rotation_y), rotation_z)
    distance = quaternion_geodesic_distance(clip.local_quat_wxyz[0, 0], expected)
    assert distance == pytest.approx(0.0, abs=1.0e-7)


def test_bvh_adapter_maps_named_joint_limits_to_indices() -> None:
    limit = JointLimitSpec(
        representation="euler_xyz",
        x_deg=(-30.0, 30.0),
        y_deg=(-10.0, 10.0),
        z_deg=(-10.0, 10.0),
        confidence=0.9,
        source="fixture",
    )
    clip = bvh_to_motion_clip(load_bvh(FIXTURE), joint_limits={"Knee": limit})
    assert clip.skeleton.joint_limits[1] == limit
    with pytest.raises(ValueError, match="unknown BVH joint"):
        bvh_to_motion_clip(load_bvh(FIXTURE), joint_limits={"Missing": limit})
