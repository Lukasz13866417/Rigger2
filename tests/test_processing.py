from __future__ import annotations

import numpy as np
import pytest

from motionlab.io.bvh import bvh_to_motion_clip, load_bvh
from motionlab.math.quaternion import quaternion_geodesic_distance
from motionlab.processing.canonicalize import canonicalize_origin_and_facing
from motionlab.processing.resample import resample_motion
from motionlab.testing.synthetic import make_synthetic_walk


def test_resampling_uses_linear_translation_and_slerp() -> None:
    clip = bvh_to_motion_clip(load_bvh("tests/fixtures/minimal_zxy.bvh"))
    resampled = resample_motion(clip, 20.0)
    assert resampled.num_frames == 3
    assert resampled.fps == 20.0
    np.testing.assert_allclose(resampled.root_translation_m[1], [0.5, 1.0, 1.5])
    start = clip.local_quat_wxyz[0, 0]
    middle = resampled.local_quat_wxyz[1, 0]
    end = clip.local_quat_wxyz[1, 0]
    assert quaternion_geodesic_distance(start, middle) == pytest.approx(np.pi / 4.0, abs=1e-6)
    assert quaternion_geodesic_distance(middle, end) == pytest.approx(np.pi / 4.0, abs=1e-6)


def test_resampling_drops_partial_final_interval_instead_of_mislabeling_time() -> None:
    clip = make_synthetic_walk(num_frames=4, fps=50.0)  # duration 0.06 s
    resampled = resample_motion(clip, 20.0)
    assert resampled.num_frames == 2
    np.testing.assert_allclose(resampled.timestamps_s, [0.0, 0.05])
    expected_z = np.interp(0.05, clip.timestamps_s, clip.root_translation_m[:, 2])
    assert resampled.root_translation_m[-1, 2] == pytest.approx(expected_z)


def test_canonicalization_preserves_height_and_aligns_motion_to_positive_z() -> None:
    clip = make_synthetic_walk(num_frames=11, fps=10.0)
    root = clip.root_translation_m.copy()
    root[:, 0] = np.linspace(5.0, 6.2, clip.num_frames)
    root[:, 2] = 7.0
    moved = clip.with_updates(root_translation_m=root)
    canonical = canonicalize_origin_and_facing(moved)
    np.testing.assert_allclose(canonical.root_translation_m[:, 1], root[:, 1])
    np.testing.assert_allclose(canonical.root_translation_m[0, [0, 2]], [0.0, 0.0], atol=1e-6)
    assert canonical.root_translation_m[-1, 2] == pytest.approx(1.2, abs=1e-6)
    assert canonical.root_translation_m[-1, 0] == pytest.approx(0.0, abs=1e-6)


def test_canonicalization_rejects_unknown_stationary_facing() -> None:
    clip = make_synthetic_walk(num_frames=4)
    root = clip.root_translation_m.copy()
    root[:, [0, 2]] = 0.0
    with pytest.raises(ValueError, match="cannot infer facing"):
        canonicalize_origin_and_facing(clip.with_updates(root_translation_m=root))
