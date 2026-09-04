from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.corruptions.physical import corrupt_joint_pop_rotation_pulse
from motionlab.motion.clip import MotionClip
from motionlab.optimization.adaptive import (
    AdaptiveSplineBlock,
    active_subspace_report,
    build_multiresolution_projections,
    exact_residual_field,
    production_active_region,
)
from motionlab.optimization.block_cma import optimize_parameter_block_with_cma
from motionlab.optimization.parameter_blocks import (
    CompositeMotionParameterization,
    ContactIKBlock,
    LoopSeamBlock,
    SpeedCadenceBlock,
)
from motionlab.testing.synthetic import make_synthetic_contact_walk


def test_active_subspace_and_multiresolution_recover_local_joint_pop() -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    corruption = corrupt_joint_pop_rotation_pulse(
        clean,
        joint="left_elbow",
        frame=55,
        angle_rad=0.3,
    )
    field = exact_residual_field(corruption.corrupted, clean)
    channels, report = active_subspace_report(field, corruption.corrupted)
    assert [channel.label for channel in channels] == ["rotation:LeftForeArm:x"]
    assert report["captured_energy_fraction"] >= 0.95
    assert report["time_interval_with_halo"] == [51, 60]

    hierarchy, projections = build_multiresolution_projections(corruption.corrupted, clean)
    assert hierarchy["different_temporal_resolutions_by_channel"] is True
    errors = [
        float(np.sqrt(np.mean(projection.error.rotation_rad**2))) for projection in projections
    ]
    assert errors[1] < errors[0]
    assert errors[2] < 1.0e-7
    assert projections[2].layout.parameter_count < clean.num_frames
    block = AdaptiveSplineBlock(projections[2].layout)
    reproduced = block.apply(corruption.corrupted, projections[2].coefficients)
    np.testing.assert_allclose(
        reproduced.local_quat_wxyz,
        projections[2].candidate.local_quat_wxyz,
        atol=1e-7,
    )


def test_production_localization_uses_intervention_and_joint_halos() -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    corruption = corrupt_joint_pop_rotation_pulse(
        clean,
        joint="left_elbow",
        frame=55,
        angle_rad=0.3,
    )
    region = production_active_region(corruption.corrupted, "joint_pop")
    assert region["source"] == "deterministic_authored_intervention_localization"
    assert region["time_interval"] == [47, 64]
    assert "LeftForeArm" in region["joint_names_with_halo"]
    assert len(region["joint_indices_with_halo"]) < clean.num_joints


def test_defect_specific_blocks_share_identity_and_composition_contract() -> None:
    clip = make_synthetic_contact_walk(num_cycles=2)
    blocks = (
        ContactIKBlock("left", 5, 30),
        SpeedCadenceBlock(),
        LoopSeamBlock(),
    )
    for block in blocks:
        result = block.apply(clip, block.initial_values)
        np.testing.assert_allclose(result.root_translation_m, clip.root_translation_m, atol=1e-7)
        np.testing.assert_allclose(result.local_quat_wxyz, clip.local_quat_wxyz, atol=1e-7)
    composite = CompositeMotionParameterization(blocks)
    assert len(composite.parameter_names) == 21
    result = composite.apply(clip, composite.initial_values)
    np.testing.assert_allclose(result.root_translation_m, clip.root_translation_m, atol=1e-7)

    contact = blocks[0]
    trajectory_values = contact.initial_values
    trajectory_values[3] = 0.02
    trajectory_values[6] = 1.0
    trajectory_values[7:9] = 2.0
    trajectory_values[9] = 0.03
    trajectory = contact.apply(clip, trajectory_values)
    assert not np.allclose(trajectory.local_quat_wxyz, clip.local_quat_wxyz)
    assert trajectory.metadata["operator_parameters"]["anchor_offset_end_m"] == [0.02, 0.0, 0.0]


@dataclass(frozen=True)
class _RootHeightBlock:
    name: str = "root_height"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("root_height:offset_m",)

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.asarray([-0.1])

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.asarray([0.1])

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.zeros(1)

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        value = float(np.asarray(values)[0])
        root = clip.root_translation_m.copy()
        root[:, 1] += value
        return clip.with_updates(root_translation_m=root)


def test_common_block_interface_is_optimized_directly_by_cma(tmp_path: Path) -> None:
    clip = make_synthetic_contact_walk(num_cycles=1)
    target_height = clip.root_translation_m[:, 1] + 0.04
    result = optimize_parameter_block_with_cma(
        clip,
        _RootHeightBlock(),
        lambda candidate: float(np.mean((candidate.root_translation_m[:, 1] - target_height) ** 2)),
        tmp_path,
        seed=19,
        maximum_evaluations=80,
        initial_sigma=0.03,
    )
    assert result.objective < 1.0e-8
    assert abs(result.values[0] - 0.04) < 1.0e-3
    assert result.report_path.exists()
