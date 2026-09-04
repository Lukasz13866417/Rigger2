from __future__ import annotations

from pathlib import Path

import torch

from motionlab.core.provenance import assign_source_lineage
from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.model import FIRST_CRITIC_DEFECTS, FixedRigFlatTCN, FixedRigTCNConfig
from motionlab.critic.training import (
    CriticTrainingConfig,
    critic_losses,
    load_checkpoint,
    train_fixed_rig_tcn,
)
from motionlab.dataset.generation import DatasetGenerationConfig, generate_corruption_dataset
from motionlab.io.npz import save_motion_npz
from motionlab.testing.synthetic import make_synthetic_contact_walk


def _critic_dataset(tmp_path: Path) -> Path:
    sources: list[Path] = []
    for index, split in enumerate(("train", "validation", "test")):
        clip = assign_source_lineage(
            make_synthetic_contact_walk(num_cycles=3),
            source_dataset="critic-test",
            source_clip_id=f"walk-{split}",
            source_take_id=f"take-{index}",
            split=split,  # type: ignore[arg-type]
            clean_confidence=0.95,
        )
        path = tmp_path / f"{split}.npz"
        save_motion_npz(path, clip)
        sources.append(path)
    output = tmp_path / "dataset"
    generate_corruption_dataset(
        sources,
        output,
        config=DatasetGenerationConfig(
            mechanisms=(
                "speed_inconsistency/root_progression_scale",
                "speed_inconsistency/leg_pose_amplitude_scale",
            ),
            include_soft_corruptions=False,
            composition_fraction=0.0,
            maximum_windows_per_source=1,
            include_sham_controls=False,
        ),
    )
    return output


def test_fixed_rig_tensorization_model_heads_and_loss(tmp_path: Path) -> None:
    root = _critic_dataset(tmp_path)
    train = FixedRigSampleDataset(root, regime="train")
    validation = FixedRigSampleDataset(root, regime="validation")
    known = FixedRigSampleDataset(root, regime="known_test")
    heldout = FixedRigSampleDataset(root, regime="heldout_mechanism")
    assert len(train) == len(validation) == len(known) == 6
    assert len(heldout) == 5
    item = train[0]
    assert item["frame_features"].shape == (160, train.num_joints * 15 + 21)
    config = FixedRigTCNConfig(
        num_joints=train.num_joints,
        hidden_channels=8,
        dilations=(1, 2, 4, 8),
    )
    model = FixedRigFlatTCN(config)
    features = item["frame_features"].unsqueeze(0)
    output = model(features)
    assert output.frame_logits.shape == (1, 160, len(FIRST_CRITIC_DEFECTS))
    assert output.frame_severity.shape == output.frame_logits.shape
    assert output.part_logits.shape == (1, 160, 8, len(FIRST_CRITIC_DEFECTS))
    assert output.clip_logits.shape == (1, len(FIRST_CRITIC_DEFECTS))
    assert output.family_ranking_score.shape == (1, len(FIRST_CRITIC_DEFECTS))
    assert output.representation.shape == (1, 8)
    batch = {
        key: item[key].unsqueeze(0)
        for key in ("frame_target", "part_target", "clip_target", "severity_target")
    }
    losses = critic_losses(output, batch, CriticTrainingConfig(epochs=1))
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_epoch_boundary_resume_is_bit_exact(tmp_path: Path) -> None:
    root = _critic_dataset(tmp_path)
    dataset = FixedRigSampleDataset(root, regime="train", cache_samples=False)
    model_config = FixedRigTCNConfig(
        num_joints=dataset.num_joints,
        hidden_channels=8,
        dilations=(1, 2, 4, 8),
    )
    training_config = CriticTrainingConfig(
        epochs=2,
        batch_size=6,
        learning_rate=1.0e-3,
        early_stopping_patience=3,
    )
    uninterrupted = train_fixed_rig_tcn(
        root,
        tmp_path / "uninterrupted",
        training_config=training_config,
        model_config=model_config,
    )
    first = train_fixed_rig_tcn(
        root,
        tmp_path / "resumed",
        training_config=training_config,
        model_config=model_config,
        maximum_epochs_this_run=1,
    )
    resumed = train_fixed_rig_tcn(
        root,
        tmp_path / "resumed",
        training_config=training_config,
        model_config=model_config,
        resume_checkpoint=first.checkpoint_path,
    )
    uninterrupted_payload = load_checkpoint(uninterrupted.checkpoint_path)
    resumed_payload = load_checkpoint(resumed.checkpoint_path)
    assert uninterrupted_payload["next_epoch"] == resumed_payload["next_epoch"] == 2
    assert uninterrupted_payload["history"] == resumed_payload["history"]
    for name, value in uninterrupted_payload["model_state"].items():
        assert torch.equal(value, resumed_payload["model_state"][name])
