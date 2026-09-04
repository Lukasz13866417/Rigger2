from __future__ import annotations

import numpy as np

from motionlab.corruptions.temporal import corrupt_joint_jitter
from motionlab.critic.forensics import motion_distances
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.optimization.jitter_blocks import (
    JitterSmoothingBlock,
    fit_oracle_spectral_jitter_projection,
)
from motionlab.testing.synthetic import make_synthetic_contact_walk


def test_full_local_spectral_basis_reconstructs_jitter_clean_endpoint() -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    result = corrupt_joint_jitter(
        clean,
        joints=("left_shoulder", "right_shoulder"),
        start_frame=24,
        stop_frame=92,
        amplitude_rad=0.015,
        seed=51,
    )
    projection = fit_oracle_spectral_jitter_projection(
        result.corrupted,
        clean,
        energy_fraction=1.0,
    )
    distance = motion_distances(clean, projection.candidate)
    assert projection.captured_energy_fraction > 1.0 - 1.0e-12
    assert distance["local_rotation_geodesic_rms_deg"] < 1.0e-5
    assert len(projection.block.modes) <= 2 * 3 * (92 - 24)


def test_low_dimensional_smoothing_block_reduces_jitter_and_has_identity_start() -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    joints = (clean.skeleton.roles["left_shoulder"], clean.skeleton.roles["right_shoulder"])
    result = corrupt_joint_jitter(
        clean,
        joints=joints,
        start_frame=24,
        stop_frame=92,
        amplitude_rad=0.02,
        seed=52,
    )
    block = JitterSmoothingBlock(joints, 20, 96)
    identity = block.apply(result.corrupted, block.initial_values)
    np.testing.assert_allclose(
        identity.local_quat_wxyz,
        result.corrupted.local_quat_wxyz,
        atol=1e-7,
    )
    values = block.initial_values
    values[0] = 5.0
    values[1] = 1.0
    smoothed = block.apply(result.corrupted, values)
    before = np.percentile(
        np.asarray(smoothness_metric(result.corrupted).joint_frame_values)[20:96, joints],
        95.0,
    )
    after = np.percentile(
        np.asarray(smoothness_metric(smoothed).joint_frame_values)[20:96, joints],
        95.0,
    )
    assert after < before
    assert len(block.parameter_names) == 6
