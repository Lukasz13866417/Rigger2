from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from motionlab.core.provenance import assign_source_lineage
from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.model import FIRST_CRITIC_DEFECTS, FixedRigFlatTCN, FixedRigTCNConfig
from motionlab.critic.training import (
    CriticTrainingConfig,
    _balanced_epoch_indices,
    critic_losses,
)
from motionlab.dataset.generation import DatasetGenerationConfig, generate_corruption_dataset
from motionlab.io.npz import save_motion_npz
from motionlab.optimization.cmaes import CMAOptimizationConfig
from motionlab.optimization.diagnostics import exact_clean_residual, interpolate_repair_path
from motionlab.optimization.spline import (
    SplineResidualLayout,
    apply_spline_residual,
    cubic_bspline_basis,
    refine_coefficients,
)
from motionlab.testing.synthetic import make_synthetic_contact_walk


def test_equivalent_controls_and_balanced_task_sampling(tmp_path: Path) -> None:
    source = assign_source_lineage(
        make_synthetic_contact_walk(num_cycles=3),
        source_dataset="optimization-test",
        source_clip_id="walk-train",
        source_take_id="take-train",
        split="train",
        clean_confidence=0.95,
    )
    source_path = tmp_path / "source.npz"
    save_motion_npz(source_path, source)
    root = tmp_path / "dataset"
    generate_corruption_dataset(
        [source_path],
        root,
        config=DatasetGenerationConfig(
            mechanisms=("speed_inconsistency/root_progression_scale",),
            include_heldout_mechanisms=False,
            include_soft_corruptions=False,
            composition_fraction=0.0,
            maximum_windows_per_source=1,
            include_sham_controls=False,
            equivalent_global_yaw_degrees=(-35.0, 35.0),
        ),
    )
    dataset = FixedRigSampleDataset(root, regime="train")
    equivalents = [
        record for record in dataset.records if record["sample_role"] == "equivalent_control"
    ]
    assert len(equivalents) == 2
    assert all(record["no_target_defect"] for record in equivalents)
    first = _balanced_epoch_indices(dataset, seed=71, epoch=2)
    second = _balanced_epoch_indices(dataset, seed=71, epoch=2)
    assert first == second
    roles = [dataset.records[index]["sample_role"] for index in first]
    assert sum(role in {"clean", "equivalent_control", "sham_control"} for role in roles) == (
        len(dataset) // 2
    )


def test_cubic_basis_and_four_to_eight_refinement_preserve_curve() -> None:
    basis = cubic_bspline_basis(101, 4)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(basis[0], (1.0, 0.0, 0.0, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(basis[-1], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12)

    clip = make_synthetic_contact_walk(num_cycles=2)
    coarse = SplineResidualLayout.for_skeleton(clip.skeleton, control_points=4)
    refined = SplineResidualLayout.for_skeleton(clip.skeleton, control_points=8)
    values = np.linspace(-0.2, 0.2, coarse.parameter_count)
    lifted = refine_coefficients(values, coarse, refined)
    coarse_curve = cubic_bspline_basis(101, 4) @ coarse.coefficient_matrix(values).T
    refined_curve = cubic_bspline_basis(101, 8) @ refined.coefficient_matrix(lifted).T
    np.testing.assert_allclose(refined_curve, coarse_curve, atol=1.0e-10)


def test_zero_spline_residual_preserves_motion_and_nonzero_residual_is_smooth() -> None:
    clip = make_synthetic_contact_walk(num_cycles=2)
    layout = SplineResidualLayout.for_skeleton(clip.skeleton, control_points=4)
    zero = apply_spline_residual(
        clip,
        layout,
        np.zeros(layout.parameter_count),
        rotation_scale_rad=0.35,
        root_translation_scale_m=0.12,
        root_yaw_scale_rad=0.35,
    )
    np.testing.assert_allclose(zero.local_quat_wxyz, clip.local_quat_wxyz, atol=1.0e-7)
    np.testing.assert_allclose(zero.root_translation_m, clip.root_translation_m, atol=1.0e-7)

    coefficients = np.zeros((layout.channel_count, layout.control_points))
    coefficients[-4] = (0.0, 0.2, -0.2, 0.0)
    edited = apply_spline_residual(
        clip,
        layout,
        coefficients.reshape(-1),
        rotation_scale_rad=0.35,
        root_translation_scale_m=0.12,
        root_yaw_scale_rad=0.35,
    )
    residual = edited.root_translation_m[:, 0] - clip.root_translation_m[:, 0]
    assert np.max(np.abs(np.diff(residual, n=2))) < 0.001


def test_ranking_loss_ignores_inactive_families_and_config_rejects_global_objective() -> None:
    clip = make_synthetic_contact_walk(num_cycles=2)
    config = FixedRigTCNConfig(
        num_joints=clip.num_joints,
        hidden_channels=8,
        dilations=(1, 2, 4, 8),
    )
    model = FixedRigFlatTCN(config)
    frames = 32
    output = model(torch.zeros((1, frames, config.input_channels)))
    defect_count = len(FIRST_CRITIC_DEFECTS)
    clip_target = torch.zeros((1, defect_count))
    clip_target[0, 0] = 1.0
    batch = {
        "frame_target": torch.zeros((1, frames, defect_count)),
        "part_target": torch.zeros((1, frames, 8, defect_count)),
        "clip_target": clip_target,
        "severity_target": clip_target * 0.6,
    }
    settings = CriticTrainingConfig(epochs=1)
    baseline = critic_losses(output, batch, settings)["ranking"]
    changed_rank = output.family_ranking_score.clone()
    changed_rank[:, 1:] = 10000.0
    changed = critic_losses(replace(output, family_ranking_score=changed_rank), batch, settings)[
        "ranking"
    ]
    torch.testing.assert_close(changed, baseline)
    with pytest.raises(ValueError, match="unknown objective"):
        CMAOptimizationConfig(objective_weights={"global_quality": 1.0})
    with pytest.raises(ValueError, match="4-to-8"):
        CMAOptimizationConfig(coarse_control_points=5)
    assert (
        CMAOptimizationConfig(mode="production").resolved_objective_source == "deterministic_target"
    )
    with pytest.raises(ValueError, match="production cannot optimize learned"):
        CMAOptimizationConfig(mode="production", objective_source="learned_severity")


def test_per_frame_layer_norm_is_consistent_for_aligned_interior_windows() -> None:
    torch.manual_seed(19)
    config = FixedRigTCNConfig(
        num_joints=2,
        joint_features=2,
        global_features=3,
        hidden_channels=8,
        dilations=(1, 2, 4, 8),
        normalization="layer_norm",
    )
    model = FixedRigFlatTCN(config).eval()
    features = torch.randn(1, 96, config.input_channels)
    offset = 8
    with torch.no_grad():
        first = model(features[:, :80])
        second = model(features[:, offset : offset + 80])
    radius = sum(dilation * (config.kernel_size - 1) for dilation in config.dilations)
    global_start = offset + radius
    global_stop = 80 - radius
    torch.testing.assert_close(
        first.frame_logits[:, global_start:global_stop],
        second.frame_logits[:, global_start - offset : global_stop - offset],
        atol=2.0e-6,
        rtol=0.0,
    )


def test_exact_repair_path_reaches_clean_pair() -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    corrupted = clean.with_updates(
        root_translation_m=(
            clean.root_translation_m + np.asarray([0.04, 0.02, -0.03], dtype=np.float32)[None]
        )
    )
    residual = exact_clean_residual(corrupted, clean)
    np.testing.assert_allclose(
        residual.root_translation_m,
        clean.root_translation_m - corrupted.root_translation_m,
    )
    repaired = interpolate_repair_path(corrupted, clean, 1.0)
    np.testing.assert_allclose(repaired.root_translation_m, clean.root_translation_m, atol=1e-7)
    np.testing.assert_allclose(repaired.local_quat_wxyz, clean.local_quat_wxyz, atol=1e-6)
