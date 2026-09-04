"""Training and pre-optimization evaluation for fixed-rig perceptual preferences."""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from motionlab.critic.data import encode_motion_clip
from motionlab.critic.training import model_from_checkpoint, seed_everything
from motionlab.dataset.io import file_sha256, load_normalization_statistics
from motionlab.io.npz import load_motion_npz
from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.perceptual.labeling import apply_human_labels
from motionlab.perceptual.model import FixedRigPerceptualResidual

PERCEPTUAL_CHECKPOINT_VERSION = "motionlab.fixed_rig_perceptual_residual.v1"
PERCEPTUAL_EVALUATION_VERSION = "motionlab.fixed_rig_perceptual_evaluation.v1"


class PerceptualTrainingConfig(BaseModel):
    """Small-head training controls; the baseline GroupNorm backbone stays frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 8201
    epochs: int = Field(default=240, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=2.0e-3, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    auxiliary_loss_weight: float = Field(default=0.25, ge=0.0)
    early_stopping_patience: int = Field(default=40, ge=1)
    representation_channels: int = Field(default=64, ge=8)
    freeze_backbone: bool = True


@dataclass(frozen=True)
class PerceptualTrainingResult:
    """Trained residual critic and its pre-optimization evaluation evidence."""

    checkpoint_path: Path
    evaluation_path: Path
    optimization_gate_passed: bool


def _pair_subset(record: dict[str, Any]) -> str:
    if record["adversarial_origin"]:
        return "adversarial"
    if record["mechanism_partition"] == "heldout" and record["source_split"] == "test":
        return "heldout_mechanism"
    if record["source_split"] == "test" and record["mechanism_partition"] == "train":
        return "heldout_source"
    if record["source_split"] == "validation" and record["mechanism_partition"] == "train":
        return "validation"
    if record["source_split"] == "train" and record["mechanism_partition"] == "train":
        return "train"
    return "other"


def _supervised(record: dict[str, Any]) -> bool:
    return record["supervision_category"] in {"CERTAIN", "HUMAN_LABELED"} and (
        record["preference"] is not None
    )


def _target(preference: str) -> float:
    return {
        "a_better": 1.0,
        "b_better": 0.0,
        "approximately_equal": 0.5,
    }[preference]


def _dimension_for_tag(tag: str) -> str | None:
    return {
        "naturalness": "naturalness",
        "coordination": "coordination",
        "rigidity/smoothing": "rigidity_smoothing",
    }.get(tag)


def _load_backbone_representations(
    pairs: list[dict[str, Any]],
    pair_directory: Path,
    source_dataset_directory: Path,
    model: FixedRigPerceptualResidual,
) -> dict[str, torch.Tensor]:
    normalization = load_normalization_statistics(source_dataset_directory)
    paths = sorted({str(record[field]) for record in pairs for field in ("motion_a", "motion_b")})
    cache: dict[str, torch.Tensor] = {}
    model.backbone.eval()
    with torch.no_grad():
        for relative in paths:
            motion = load_motion_npz(pair_directory / relative)
            features = encode_motion_clip(motion, normalization)
            cache[relative] = model.backbone(features[None]).representation[0].detach().cpu()
    return cache


def _expanded_training_examples(
    records: list[dict[str, Any]],
    representations: dict[str, torch.Tensor],
    *,
    subsets: tuple[str, ...] = ("train",),
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        if not _supervised(record) or _pair_subset(record) not in subsets:
            continue
        target = _target(str(record["preference"]))
        example = {
            "a": representations[str(record["motion_a"])],
            "b": representations[str(record["motion_b"])],
            "target": target,
            "dimensions": tuple(
                dimension
                for tag in record["reason_tags"]
                if (dimension := _dimension_for_tag(str(tag))) is not None
            ),
        }
        examples.append(example)
        if target != 0.5:
            examples.append(
                {
                    "a": example["b"],
                    "b": example["a"],
                    "target": 1.0 - target,
                    "dimensions": example["dimensions"],
                }
            )
    if not examples:
        raise ValueError("perceptual training has no supervised train pairs")
    return examples


def _batch_loss(
    model: FixedRigPerceptualResidual,
    examples: list[dict[str, Any]],
    auxiliary_weight: float,
) -> torch.Tensor:
    a = torch.stack([example["a"] for example in examples])
    b = torch.stack([example["b"] for example in examples])
    target = torch.tensor([example["target"] for example in examples], dtype=torch.float32)
    output_a = model.score_from_backbone_representation(a)
    output_b = model.score_from_backbone_representation(b)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output_a.score - output_b.score,
        target,
    )
    auxiliary_losses = []
    for row, example in enumerate(examples):
        if example["target"] == 0.5:
            continue
        for dimension in example["dimensions"]:
            difference = (
                output_a.auxiliary_scores[dimension][row]
                - output_b.auxiliary_scores[dimension][row]
            )
            auxiliary_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    difference,
                    target[row],
                )
            )
    if auxiliary_losses:
        loss = loss + auxiliary_weight * torch.stack(auxiliary_losses).mean()
    return loss


def _validation_loss(
    model: FixedRigPerceptualResidual,
    records: list[dict[str, Any]],
    representations: dict[str, torch.Tensor],
) -> float:
    selected = [
        record for record in records if _supervised(record) and _pair_subset(record) == "validation"
    ]
    if not selected:
        raise ValueError("perceptual evaluation requires validation pairs")
    with torch.no_grad():
        losses = []
        for record in selected:
            a = representations[str(record["motion_a"])][None]
            b = representations[str(record["motion_b"])][None]
            difference = (
                model.score_from_backbone_representation(a).score
                - model.score_from_backbone_representation(b).score
            )
            target = torch.tensor([_target(str(record["preference"]))])
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(difference, target))
    return float(torch.stack(losses).mean())


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 5) -> float | None:
    if probabilities.size == 0:
        return None
    total = probabilities.size
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        if np.any(selected):
            result += (
                float(np.sum(selected))
                / total
                * abs(float(np.mean(probabilities[selected])) - float(np.mean(targets[selected])))
            )
    return result


def _evaluate_records(
    model: FixedRigPerceptualResidual,
    records: list[dict[str, Any]],
    representations: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for record in records:
            a = model.score_from_backbone_representation(
                representations[str(record["motion_a"])][None]
            )
            b = model.score_from_backbone_representation(
                representations[str(record["motion_b"])][None]
            )
            score_a = float(a.score.item())
            score_b = float(b.score.item())
            probability = 1.0 / (1.0 + math.exp(-(score_a - score_b)))
            preference = record["preference"]
            correct: bool | None = None
            if preference == "a_better":
                correct = score_a > score_b
            elif preference == "b_better":
                correct = score_b > score_a
            elif preference == "approximately_equal":
                correct = abs(probability - 0.5) <= 0.1
            rows.append(
                {
                    "pair_id": record["pair_id"],
                    "subset": _pair_subset(record),
                    "mechanism": record["perturbation_mechanism"],
                    "supervision_category": record["supervision_category"],
                    "preference": preference,
                    "score_a": score_a,
                    "score_b": score_b,
                    "probability_a_better": probability,
                    "confidence": abs(probability - 0.5) * 2.0,
                    "correct": correct,
                    "adversarial_origin": record["adversarial_origin"],
                }
            )

    subsets: dict[str, Any] = {}
    for subset in ("train", "validation", "heldout_source", "heldout_mechanism", "adversarial"):
        directional = [
            row
            for row in rows
            if row["subset"] == subset
            and row["preference"] in {"a_better", "b_better"}
            and row["correct"] is not None
        ]
        probabilities = np.asarray(
            [
                row["probability_a_better"]
                if row["preference"] == "a_better"
                else 1.0 - row["probability_a_better"]
                for row in directional
            ],
            dtype=np.float64,
        )
        targets = np.ones(len(directional), dtype=np.float64)
        subsets[subset] = {
            "directional_pair_count": len(directional),
            "pairwise_accuracy": (
                None
                if not directional
                else sum(bool(row["correct"]) for row in directional) / len(directional)
            ),
            "mean_correct_direction_probability": (
                None if not directional else float(np.mean(probabilities))
            ),
            "expected_calibration_error": _ece(probabilities, targets),
        }
    equivalent = [row for row in rows if row["preference"] == "approximately_equal"]
    equivalent_false = [row for row in equivalent if not row["correct"]]
    human = [
        row
        for row in rows
        if row["supervision_category"] == "HUMAN_LABELED" and row["correct"] is not None
    ]
    report = {
        "subsets": subsets,
        "equivalent_control_count": len(equivalent),
        "equivalent_control_false_preference_count": len(equivalent_false),
        "equivalent_control_false_preference_rate": (
            None if not equivalent else len(equivalent_false) / len(equivalent)
        ),
        "human_labeled_pair_count": len(human),
        "pairwise_human_agreement": (
            None if not human else sum(bool(row["correct"]) for row in human) / len(human)
        ),
    }
    return report, rows


def train_and_evaluate_perceptual_residual(
    pair_dataset_directory: Path,
    source_dataset_directory: Path,
    backbone_checkpoint: Path,
    output_directory: Path,
    *,
    human_labels_path: Path | None = None,
    training_config: PerceptualTrainingConfig | None = None,
) -> PerceptualTrainingResult:
    """Train the frozen-backbone residual head and decide whether CMA may proceed."""
    settings = PerceptualTrainingConfig() if training_config is None else training_config
    if not settings.freeze_backbone:
        raise ValueError("the first perceptual experiment must keep the backbone frozen")
    seed_everything(settings.seed)
    pairs = load_perceptual_pairs(pair_dataset_directory)
    if human_labels_path is not None:
        pairs = apply_human_labels(pairs, human_labels_path)
    backbone = model_from_checkpoint(backbone_checkpoint)
    if backbone.config.normalization != "group_norm":
        raise ValueError("the perceptual baseline must use the current GroupNorm checkpoint")
    model = FixedRigPerceptualResidual(
        backbone,
        representation_channels=settings.representation_channels,
        freeze_backbone=True,
    )
    initial_state = copy.deepcopy(model.state_dict())
    representations = _load_backbone_representations(
        pairs,
        Path(pair_dataset_directory),
        source_dataset_directory,
        model,
    )
    examples = _expanded_training_examples(pairs, representations)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(settings.epochs):
        model.train()
        order = list(range(len(examples)))
        random.Random(settings.seed + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), settings.batch_size):
            batch = [examples[index] for index in order[start : start + settings.batch_size]]
            loss = _batch_loss(model, batch, settings.auxiliary_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = _validation_loss(model, pairs, representations)
        training = float(np.mean(losses))
        history.append(
            {"epoch": epoch + 1, "training_pair_loss": training, "validation_pair_loss": validation}
        )
        if validation < best_validation - 1.0e-7:
            best_validation = validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= settings.early_stopping_patience:
            break
    best_epoch = int(np.argmin([row["validation_pair_loss"] for row in history])) + 1
    # Standard final fit: after selecting the epoch count without touching test sources, restart
    # the same head and train on train+validation sources for that fixed number of epochs.
    model.load_state_dict(initial_state)
    final_examples = _expanded_training_examples(
        pairs,
        representations,
        subsets=("train", "validation"),
    )
    final_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    final_history: list[dict[str, float | int]] = []
    for epoch in range(best_epoch):
        model.train()
        order = list(range(len(final_examples)))
        random.Random(settings.seed + 100_000 + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), settings.batch_size):
            batch = [final_examples[index] for index in order[start : start + settings.batch_size]]
            loss = _batch_loss(model, batch, settings.auxiliary_loss_weight)
            final_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            final_optimizer.step()
            losses.append(float(loss.detach()))
        final_history.append({"epoch": epoch + 1, "training_pair_loss": float(np.mean(losses))})
    model.eval()
    evaluation, rows = _evaluate_records(model, pairs, representations)
    heldout_source = evaluation["subsets"]["heldout_source"]
    heldout_mechanism = evaluation["subsets"]["heldout_mechanism"]
    gate_reasons = []
    for label, result in (
        ("heldout_source", heldout_source),
        ("heldout_mechanism", heldout_mechanism),
    ):
        if result["directional_pair_count"] < 4:
            gate_reasons.append(f"{label}:insufficient_pairs")
        elif float(result["pairwise_accuracy"]) < 0.65:
            gate_reasons.append(f"{label}:near_chance_ranking")
    equivalent_rate = evaluation["equivalent_control_false_preference_rate"]
    if equivalent_rate is None or float(equivalent_rate) > 0.25:
        gate_reasons.append("equivalent_controls:false_preference_rate")
    gate_passed = not gate_reasons
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "perceptual_residual.pt"
    torch.save(
        {
            "format_version": PERCEPTUAL_CHECKPOINT_VERSION,
            "training_config": settings.model_dump(mode="json"),
            "backbone_checkpoint": str(backbone_checkpoint),
            "backbone_checkpoint_sha256": file_sha256(backbone_checkpoint),
            "backbone_model_config": backbone.config.model_dump(mode="json"),
            "perceptual_state": {
                key: value
                for key, value in model.state_dict().items()
                if not key.startswith("backbone.")
            },
            "supported_auxiliary_dimensions": [
                "naturalness",
                "coordination",
                "rigidity_smoothing",
            ],
            "unsupported_unlabeled_dimensions": ["style_coherence", "weight_transfer"],
            "human_labels_path": None if human_labels_path is None else str(human_labels_path),
            "human_labels_sha256": (
                None if human_labels_path is None else file_sha256(human_labels_path)
            ),
        },
        checkpoint_path,
    )
    evaluation_report = {
        "format_version": PERCEPTUAL_EVALUATION_VERSION,
        "architecture": "frozen current GroupNorm TCN plus separate perceptual residual head",
        "normalization_reopened": False,
        "backbone_frozen": True,
        "human_labels_path": None if human_labels_path is None else str(human_labels_path),
        "pair_count": len(pairs),
        "labeled_pair_count": sum(_supervised(record) for record in pairs),
        "unordered_pair_count": sum(not _supervised(record) for record in pairs),
        "training_example_count_after_direction_balancing": len(examples),
        "training_history": history,
        "selected_epoch_count": best_epoch,
        "final_refit_sources": ["train", "validation"],
        "final_refit_history": final_history,
        "best_validation_loss": best_validation,
        **evaluation,
        "optimization_gate": {
            "passed": gate_passed,
            "reasons": gate_reasons,
            "minimum_pairwise_accuracy": 0.65,
            "maximum_equivalent_false_preference_rate": 0.25,
        },
        "checkpoint": str(checkpoint_path),
        "pair_scores": rows,
        "supervision_counts": dict(
            Counter(str(record["supervision_category"]) for record in pairs)
        ),
    }
    evaluation_path = output / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(evaluation_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return PerceptualTrainingResult(checkpoint_path, evaluation_path, gate_passed)


def load_perceptual_model(
    checkpoint_path: Path,
) -> tuple[FixedRigPerceptualResidual, dict[str, Any]]:
    """Load a trusted local residual-head checkpoint and its frozen backbone."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != PERCEPTUAL_CHECKPOINT_VERSION:
        raise ValueError("unsupported perceptual checkpoint")
    backbone_path = Path(str(payload["backbone_checkpoint"]))
    if file_sha256(backbone_path) != payload["backbone_checkpoint_sha256"]:
        raise ValueError("perceptual backbone checkpoint checksum mismatch")
    backbone = model_from_checkpoint(backbone_path)
    config = PerceptualTrainingConfig.model_validate(payload["training_config"])
    model = FixedRigPerceptualResidual(
        backbone,
        representation_channels=config.representation_channels,
        freeze_backbone=True,
    )
    state = model.state_dict()
    state.update(payload["perceptual_state"])
    model.load_state_dict(state)
    model.eval()
    return model, payload
