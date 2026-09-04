"""Deterministic CPU training and tiny-overfit gates for the fixed-rig TCN."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr
from torch import nn

from motionlab._version import __version__
from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.model import (
    FIRST_CRITIC_DEFECTS,
    CriticOutput,
    FixedRigFlatTCN,
    FixedRigTCNConfig,
)
from motionlab.dataset.io import file_sha256

CHECKPOINT_VERSION = "motionlab.fixed_rig_flat_tcn.checkpoint.v1"


class CriticTrainingConfig(BaseModel):
    """Frozen optimization settings persisted in every checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 2027
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1.0e-3, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    gradient_clip_norm: float = Field(default=5.0, gt=0.0)
    early_stopping_patience: int = Field(default=8, ge=1)
    device: str = "cpu"
    frame_loss_weight: float = Field(default=1.0, ge=0.0)
    part_loss_weight: float = Field(default=0.75, ge=0.0)
    clip_loss_weight: float = Field(default=1.0, ge=0.0)
    severity_loss_weight: float = Field(default=0.5, ge=0.0)
    ranking_loss_weight: float = Field(default=0.25, ge=0.0)
    balanced_task_sampling: bool = True


@dataclass(frozen=True)
class TrainingResult:
    """Paths and scalar evidence from one complete or resumed training run."""

    checkpoint_path: Path
    best_checkpoint_path: Path
    history: tuple[Mapping[str, float | int], ...]
    stopped_epoch: int


def seed_everything(seed: int) -> None:
    """Seed all local stochastic sources and request deterministic Torch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _positive_weight(target: torch.Tensor) -> torch.Tensor:
    reduce = tuple(range(target.ndim - 1))
    positive = target.sum(dim=reduce)
    total = float(np.prod(target.shape[:-1]))
    negative = total - positive
    return torch.clamp(negative / torch.clamp(positive, min=1.0), min=1.0, max=30.0)


def critic_losses(
    output: CriticOutput,
    batch: Mapping[str, torch.Tensor],
    config: CriticTrainingConfig,
) -> dict[str, torch.Tensor]:
    """Compute normalized strong symptom, part, clip, severity, and ordinal losses."""
    frame_target = batch["frame_target"]
    part_target = batch["part_target"]
    clip_target = batch["clip_target"]
    severity_target = batch["severity_target"]
    frame_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.frame_logits,
        frame_target,
        pos_weight=_positive_weight(frame_target),
    )
    part_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.part_logits,
        part_target,
        pos_weight=_positive_weight(part_target),
    )
    clip_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.clip_logits,
        clip_target,
        pos_weight=_positive_weight(clip_target),
    )
    temporal_severity_target = frame_target * severity_target[:, None, :]
    severity_loss = torch.nn.functional.smooth_l1_loss(
        output.clip_severity,
        severity_target,
    ) + torch.nn.functional.smooth_l1_loss(output.frame_severity, temporal_severity_target)
    # Synthetic labels establish only within-family order.  Inactive families are masked so the
    # loss cannot imply any cross-family or global quality ordering.
    active = clip_target > 0.5
    ranking_error = torch.nn.functional.smooth_l1_loss(
        output.family_ranking_score,
        severity_target,
        reduction="none",
    )
    ranking_loss = (ranking_error * active).sum() / torch.clamp(active.sum(), min=1)
    total = (
        config.frame_loss_weight * frame_loss
        + config.part_loss_weight * part_loss
        + config.clip_loss_weight * clip_loss
        + config.severity_loss_weight * severity_loss
        + config.ranking_loss_weight * ranking_loss
    )
    return {
        "total": total,
        "frame": frame_loss,
        "part": part_loss,
        "clip": clip_loss,
        "severity": severity_loss,
        "ranking": ranking_loss,
    }


def _collate(items: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    tensor_keys = (
        "frame_features",
        "summary_features",
        "frame_target",
        "part_target",
        "clip_target",
        "severity_target",
    )
    result: dict[str, Any] = {
        key: torch.stack([item[key] for item in items]).to(device) for key in tensor_keys
    }
    for key in items[0]:
        if key not in tensor_keys:
            result[key] = [item[key] for item in items]
    return result


def _epoch_indices(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def _balanced_epoch_indices(
    dataset: FixedRigSampleDataset,
    seed: int,
    epoch: int,
) -> list[int]:
    """Draw 50% controls and 50% family-balanced defects with deterministic replacement."""
    negative_roles = {"clean", "equivalent_control", "sham_control"}
    controls: dict[str, list[int]] = {}
    positives: dict[str, list[int]] = {}
    for index, record in enumerate(dataset.records):
        role = str(record.get("sample_role"))
        if role in negative_roles:
            controls.setdefault(role, []).append(index)
            continue
        family = str(record.get("corruption_family") or "")
        if family in FIRST_CRITIC_DEFECTS:
            positives.setdefault(family, []).append(index)
    if not controls or not positives:
        return _epoch_indices(len(dataset), seed, epoch)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)

    def draw(groups: Mapping[str, Sequence[int]], count: int) -> list[int]:
        names = sorted(groups)
        result: list[int] = []
        for offset in range(count):
            values = groups[names[offset % len(names)]]
            selected = int(torch.randint(len(values), (1,), generator=generator).item())
            result.append(values[selected])
        return result

    negative_count = len(dataset) // 2
    selected = draw(controls, negative_count) + draw(positives, len(dataset) - negative_count)
    order = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[index] for index in order]


def _run_epoch(
    model: FixedRigFlatTCN,
    dataset: FixedRigSampleDataset,
    config: CriticTrainingConfig,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    indices = list(range(len(dataset)))
    if training:
        indices = (
            _balanced_epoch_indices(dataset, config.seed, epoch)
            if config.balanced_task_sampling
            else _epoch_indices(len(dataset), config.seed, epoch)
        )
    totals = {name: 0.0 for name in ("total", "frame", "part", "clip", "severity", "ranking")}
    batches = 0
    device = torch.device(config.device)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(indices), config.batch_size):
            batch = _collate(
                [dataset[index] for index in indices[start : start + config.batch_size]],
                device,
            )
            output = model(batch["frame_features"])
            losses = critic_losses(output, batch, config)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu())
            batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load and minimally validate a trusted local training checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported critic checkpoint")
    return payload


def model_from_checkpoint(path: Path, *, device: str = "cpu") -> FixedRigFlatTCN:
    """Recreate an evaluation model from its complete architecture declaration."""
    payload = load_checkpoint(path, device=device)
    model = FixedRigFlatTCN(FixedRigTCNConfig.model_validate(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def _checkpoint_payload(
    model: FixedRigFlatTCN,
    optimizer: torch.optim.Optimizer,
    training_config: CriticTrainingConfig,
    *,
    next_epoch: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    history: Sequence[Mapping[str, float | int]],
    dataset_directory: Path,
) -> dict[str, Any]:
    root = Path(dataset_directory)
    return {
        "format_version": CHECKPOINT_VERSION,
        "code_version": __version__,
        "model_config": model.config.model_dump(mode="json"),
        "training_config": training_config.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": [dict(record) for record in history],
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "dataset_manifest_sha256": file_sha256(root / "manifest.jsonl"),
        "normalization_manifest_sha256": file_sha256(root / "normalization.json"),
    }


def train_fixed_rig_tcn(
    dataset_directory: Path,
    output_directory: Path,
    *,
    training_config: CriticTrainingConfig | None = None,
    model_config: FixedRigTCNConfig | None = None,
    resume_checkpoint: Path | None = None,
    maximum_epochs_this_run: int | None = None,
) -> TrainingResult:
    """Train or exactly resume the fixed-rig baseline at an epoch boundary."""
    settings = CriticTrainingConfig() if training_config is None else training_config
    if settings.device != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"requested unavailable device {settings.device!r}")
    seed_everything(settings.seed)
    train_dataset = FixedRigSampleDataset(dataset_directory, regime="train")
    validation_dataset = FixedRigSampleDataset(dataset_directory, regime="validation")
    architecture = (
        FixedRigTCNConfig(num_joints=train_dataset.num_joints)
        if model_config is None
        else model_config
    )
    if architecture.num_joints != train_dataset.num_joints:
        raise ValueError("model joint count does not match the fixed-rig dataset")
    model = FixedRigFlatTCN(architecture).to(settings.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    start_epoch = 0
    best_loss = float("inf")
    history: list[Mapping[str, float | int]] = []
    epochs_without_improvement = 0
    if resume_checkpoint is not None:
        payload = load_checkpoint(resume_checkpoint, device=settings.device)
        if payload["model_config"] != architecture.model_dump(mode="json"):
            raise ValueError("resume checkpoint architecture differs")
        if payload["training_config"] != settings.model_dump(mode="json"):
            raise ValueError("resume checkpoint training configuration differs")
        root = Path(dataset_directory)
        if payload["dataset_manifest_sha256"] != file_sha256(root / "manifest.jsonl"):
            raise ValueError("resume checkpoint dataset manifest differs")
        if payload["normalization_manifest_sha256"] != file_sha256(root / "normalization.json"):
            raise ValueError("resume checkpoint normalization differs")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["next_epoch"])
        best_loss = float(payload["best_validation_loss"])
        history = list(payload["history"])
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    stopped_epoch = start_epoch
    stop_epoch = settings.epochs
    if maximum_epochs_this_run is not None:
        if maximum_epochs_this_run < 1:
            raise ValueError("maximum_epochs_this_run must be positive")
        stop_epoch = min(stop_epoch, start_epoch + maximum_epochs_this_run)
    for epoch in range(start_epoch, stop_epoch):
        train_metrics = _run_epoch(
            model,
            train_dataset,
            settings,
            epoch=epoch,
            optimizer=optimizer,
        )
        validation_metrics = _run_epoch(
            model,
            validation_dataset,
            settings,
            epoch=epoch,
            optimizer=None,
        )
        record: dict[str, float | int] = {"epoch": epoch + 1}
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(record)
        improved = validation_metrics["total"] < best_loss
        if improved:
            best_loss = validation_metrics["total"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = _checkpoint_payload(
            model,
            optimizer,
            settings,
            next_epoch=epoch + 1,
            best_validation_loss=best_loss,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            dataset_directory=dataset_directory,
        )
        _atomic_torch_save(last_path, payload)
        if improved:
            _atomic_torch_save(best_path, payload)
        stopped_epoch = epoch + 1
        if epochs_without_improvement >= settings.early_stopping_patience:
            break
    history_path = output / "metrics.jsonl"
    history_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in history
    )
    history_path.write_text(history_text, encoding="utf-8")
    return TrainingResult(last_path, best_path, tuple(history), stopped_epoch)


def _binary_f1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted = predicted.bool()
    target = target.bool()
    true_positive = int(torch.sum(predicted & target))
    false_positive = int(torch.sum(predicted & ~target))
    false_negative = int(torch.sum(~predicted & target))
    return 2.0 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)


def run_tiny_overfit_gate(
    dataset_directory: Path,
    output_directory: Path,
    *,
    seed: int = 2027,
    epochs: int = 250,
    maximum_samples: int = 18,
) -> dict[str, Any]:
    """Deliberately memorize a small, diverse source-window subset before real training."""
    seed_everything(seed)
    dataset = FixedRigSampleDataset(dataset_directory, regime="train")
    selected: list[int] = []

    def add_first(
        *, sample_role: str, family: str | None = None, severity: str | None = None
    ) -> None:
        if len(selected) >= maximum_samples:
            return
        for index, record in enumerate(dataset.records):
            if index in selected or record.get("sample_role") != sample_role:
                continue
            if family is not None and record.get("corruption_family") != family:
                continue
            if severity is not None and record.get("severity_label") != severity:
                continue
            selected.append(index)
            return

    # Deliberately include negative controls, then low/high examples of every family before
    # spending remaining capacity on intermediate severities. Manifest hash order must not decide
    # whether the memorization set contains a usable ordinal chain.
    add_first(sample_role="clean")
    add_first(sample_role="sham_control")
    for severity in ("near_threshold", "severe", "moderate", "subtle", "clear"):
        for family in FIRST_CRITIC_DEFECTS:
            add_first(sample_role="hard_corruption", family=family, severity=severity)
    if len(selected) < maximum_samples:
        for index in range(len(dataset.records)):
            if index not in selected:
                selected.append(index)
            if len(selected) >= maximum_samples:
                break
    if len(selected) < 4:
        raise ValueError("tiny-overfit gate requires at least four selected samples")
    per_family: dict[str, int] = {}
    per_role: dict[str, int] = {}
    for index in selected:
        record = dataset.records[index]
        family = str(record.get("corruption_family") or "clean")
        role = str(record.get("sample_role"))
        per_family[family] = per_family.get(family, 0) + 1
        per_role[role] = per_role.get(role, 0) + 1
    model_config = FixedRigTCNConfig(
        num_joints=dataset.num_joints,
        hidden_channels=128,
        dilations=(1, 2, 4, 8),
    )
    model = FixedRigFlatTCN(model_config)
    settings = CriticTrainingConfig(
        seed=seed,
        epochs=epochs,
        batch_size=len(selected),
        learning_rate=5.0e-4,
        weight_decay=0.0,
        early_stopping_patience=epochs,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    items = [dataset[index] for index in selected]
    batch = _collate(items, torch.device("cpu"))
    curve: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        output = model(batch["frame_features"])
        losses = critic_losses(output, batch, settings)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            curve.append({"epoch": epoch + 1, "loss": float(losses["total"].detach())})
    model.eval()
    with torch.no_grad():
        output = model(batch["frame_features"])
    clip_predictions = torch.sigmoid(output.clip_logits) >= 0.5
    clip_target = batch["clip_target"].bool()
    clip_exact = float(torch.mean(torch.all(clip_predictions == clip_target, dim=-1).float()))
    frame_f1 = _binary_f1(torch.sigmoid(output.frame_logits) >= 0.5, batch["frame_target"])
    correlations: list[float] = []
    for family_index in range(len(FIRST_CRITIC_DEFECTS)):
        active = batch["clip_target"][:, family_index].bool().numpy()
        target_rank = batch["severity_target"][:, family_index].numpy()[active]
        predicted_rank = output.family_ranking_score[:, family_index].numpy()[active]
        if len(target_rank) >= 2 and len(np.unique(target_rank)) >= 2:
            correlation = float(spearmanr(target_rank, predicted_rank).statistic)
            if np.isfinite(correlation):
                correlations.append(correlation)
    correlation = float(np.mean(correlations)) if correlations else 0.0
    passed = clip_exact >= 0.95 and frame_f1 >= 0.85 and correlation >= 0.9
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "code_version": __version__,
        "model_config": model_config.model_dump(mode="json"),
        "training_config": settings.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "next_epoch": epochs,
        "best_validation_loss": float(losses["total"].detach()),
        "epochs_without_improvement": 0,
        "history": curve,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "dataset_manifest_sha256": file_sha256(Path(dataset_directory) / "manifest.jsonl"),
        "normalization_manifest_sha256": file_sha256(
            Path(dataset_directory) / "normalization.json"
        ),
    }
    checkpoint_path = output_root / "tiny_overfit.pt"
    _atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "format_version": "motionlab.tiny_overfit.v1",
        "passed": passed,
        "sample_count": len(items),
        "family_counts": per_family,
        "sample_role_counts": per_role,
        "clip_exact_match": clip_exact,
        "temporal_localization_f1": frame_f1,
        "severity_rank_spearman": correlation,
        "initial_loss": curve[0]["loss"],
        "final_loss": float(losses["total"].detach()),
        "curve": curve,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": "sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    (output_root / "tiny_overfit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
