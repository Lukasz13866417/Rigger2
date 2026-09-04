from __future__ import annotations

import numpy as np
import pytest

from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton
from motionlab.testing.synthetic import make_synthetic_humanoid, make_synthetic_walk


def test_skeleton_and_clip_arrays_are_defensively_copied_and_read_only() -> None:
    skeleton = make_synthetic_humanoid()
    clip = make_synthetic_walk(num_frames=12)
    assert not skeleton.parents.flags.writeable
    assert not skeleton.rest_offsets_m.flags.writeable
    assert not clip.local_quat_wxyz.flags.writeable
    assert not clip.root_translation_m.flags.writeable
    with pytest.raises(ValueError):
        clip.root_translation_m[0, 0] = 2.0

    external_root = np.zeros((2, 3), dtype=np.float32)
    external_rotations = np.broadcast_to(
        skeleton.rest_local_quat_wxyz,
        (2, skeleton.num_joints, 4),
    ).copy()
    detached = MotionClip(skeleton, external_rotations, external_root, 60.0)
    external_root[0, 0] = 9.0
    external_rotations[0, 0, 0] = -1.0
    assert detached.root_translation_m[0, 0] == 0.0
    assert detached.local_quat_wxyz[0, 0, 0] == 1.0


def test_skeleton_rejects_cycle_and_multiple_roots() -> None:
    rest = np.array([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32)
    offsets = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly one root"):
        Skeleton(("a", "b"), np.array([-1, -1]), offsets, rest)
    with pytest.raises(ValueError, match=r"exactly one root|cycle"):
        Skeleton(("a", "b"), np.array([1, 0]), offsets, rest)


def test_clip_rejects_nonunit_rotations_and_bad_shapes() -> None:
    skeleton = make_synthetic_humanoid()
    rotations = np.zeros((2, skeleton.num_joints, 4), dtype=np.float32)
    translations = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="unit quaternions"):
        MotionClip(skeleton, rotations, translations, 60.0)
    with pytest.raises(ValueError, match="root_translation_m"):
        MotionClip(
            skeleton,
            np.broadcast_to(skeleton.rest_local_quat_wxyz, (2, skeleton.num_joints, 4)),
            np.zeros((3, 3), dtype=np.float32),
            60.0,
        )


def test_clip_slice_tracks_parent_and_timing() -> None:
    clip = make_synthetic_walk(num_frames=21, fps=10.0)
    sliced = clip.slice_frames(4, 14)
    assert sliced.num_frames == 10
    assert sliced.duration_s == pytest.approx(0.9)
    assert sliced.metadata["parent_motion_id"] == clip.content_hash
    assert sliced.metadata["source_frame_slice"] == [4, 14]


def test_content_hash_is_stable_and_sensitive_to_samples() -> None:
    first = make_synthetic_walk(num_frames=10)
    second = make_synthetic_walk(num_frames=10)
    assert first.content_hash == second.content_hash
    translated = first.root_translation_m.copy()
    translated[3, 0] = 0.01
    modified = first.with_updates(root_translation_m=translated)
    assert modified.content_hash != first.content_hash
