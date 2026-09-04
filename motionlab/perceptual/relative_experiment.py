"""Human-audited fixed-rig relative-comparator ablation experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
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
from motionlab.perceptual.dataset import (
    load_perceptual_pairs,
    objective_equivalence_control,
)
from motionlab.perceptual.human_audit import apply_audited_supervision_policy
from motionlab.perceptual.relative_model import (
    FixedRigRelativeComparator,
    RelativeComparatorConfig,
)

RELATIVE_CHECKPOINT_VERSION = "motionlab.fixed_rig_relative_comparator.v1"
RELATIVE_EXPERIMENT_VERSION = "motionlab.relative_comparator_experiment.v3"
PREFERENCE_CLASSES = ("a_better", "b_better", "approximately_equal")
RELATIVE_VARIANTS = (
    "relative_comparator_frozen",
    "relative_comparator_finetuned",
)


def _motioncritic_assessment() -> dict[str, Any]:
    return {
        "status": "deferred",
        "reason": (
            "No human-labeled pairs exist yet, and the public checkpoint expects 60-frame "
            "sequences of 24 SMPL local axis-angle joints plus root XYZ. The current 23-joint "
            "rig cannot be mapped defensibly without an explicit SMPL retarget/rest-frame "
            "conversion and licensed SMPL body assets. That is substantial unrelated engineering."
        ),
        "expected_input": "[batch, 60, 25, 3]: 24 SMPL axis-angle joints plus root XYZ",
        "mapping_attempted": False,
        "pretrained_pairwise_accuracy": None,
        "upstream_repository": "https://github.com/ou524u/MotionCritic",
        "upstream_project": "https://motioncritic.github.io/",
    }


class RelativeExperimentConfig(BaseModel):
    """Matched frozen/fine-tuned training controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 8401
    epochs: int = Field(default=120, ge=1)
    batch_size: int = Field(default=8, ge=1)
    comparator_learning_rate: float = Field(default=2.0e-3, gt=0.0)
    backbone_learning_rate: float = Field(default=2.0e-5, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    patience: int = Field(default=24, ge=1)
    comparison_channels: int = Field(default=64, ge=8)
    minimum_human_audit_labels: int = Field(default=50, ge=1)
    heldout_source_accuracy_gate: float = Field(default=0.65, ge=0.0, le=1.0)
    maximum_equivalent_false_preference_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    maximum_swap_consistency_error: float = Field(default=1.0e-6, ge=0.0)


@dataclass(frozen=True)
class RelativeExperimentResult:
    report_path: Path
    optimization_gate_passed: bool
    selected_checkpoint: Path | None


def _subset(record: dict[str, Any]) -> str:
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
    return record.get("preference") in PREFERENCE_CLASSES and record.get(
        "supervision_category"
    ) in {"CERTAIN", "HUMAN_LABELED"}


def _target(record: dict[str, Any]) -> int:
    return PREFERENCE_CLASSES.index(str(record["preference"]))


def relative_selection_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy the immutable train/validation partition before applying held-out audit evidence."""
    return [
        copy.deepcopy(record) for record in records if _subset(record) in {"train", "validation"}
    ]


def relative_selection_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe the exact records and targets allowed to fit/select the two ablations."""
    selected = relative_selection_records(records)
    supervised = sorted(
        (
            {
                "pair_id": str(record["pair_id"]),
                "subset": _subset(record),
                "preference": str(record["preference"]),
                "target_index": _target(record),
                "supervision_category": str(record["supervision_category"]),
            }
            for record in selected
            if _supervised(record)
        ),
        key=lambda row: (row["subset"], row["pair_id"]),
    )
    encoded = json.dumps(
        supervised,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "policy": "immutable_pair_dataset_train_and_validation_supervision",
        "records": supervised,
        "record_count": len(supervised),
        "train_record_count": sum(row["subset"] == "train" for row in supervised),
        "validation_record_count": sum(row["subset"] == "validation" for row in supervised),
        "records_and_targets_sha256": hashlib.sha256(encoded).hexdigest(),
        "heldout_human_labels_used": False,
        "heldout_human_audit_policy_used": False,
        "heldout_source_records_used": False,
        "heldout_mechanism_records_used": False,
        "adversarial_records_used": False,
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _load_features(
    records: list[dict[str, Any]],
    pair_directory: Path,
    source_dataset_directory: Path,
) -> dict[str, torch.Tensor]:
    normalization = load_normalization_statistics(source_dataset_directory)
    paths = sorted({str(record[field]) for record in records for field in ("motion_a", "motion_b")})
    return {
        relative: encode_motion_clip(load_motion_npz(pair_directory / relative), normalization)
        .detach()
        .cpu()
        for relative in paths
    }


def _batch(
    records: list[dict[str, Any]],
    features: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.stack([features[str(record["motion_a"])] for record in records])
    b = torch.stack([features[str(record["motion_b"])] for record in records])
    target = torch.tensor([_target(record) for record in records], dtype=torch.long)
    return a, b, target


def _loss(
    model: FixedRigRelativeComparator,
    a: torch.Tensor,
    b: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    original = model(a, b).logits
    swapped = model(b, a).logits
    swapped_target = torch.where(target == 0, 1, torch.where(target == 1, 0, 2))
    return 0.5 * (
        torch.nn.functional.cross_entropy(original, target)
        + torch.nn.functional.cross_entropy(swapped, swapped_target)
    )


def _validation_loss(
    model: FixedRigRelativeComparator,
    records: list[dict[str, Any]],
    features: dict[str, torch.Tensor],
) -> float:
    selected = [
        record for record in records if _supervised(record) and _subset(record) == "validation"
    ]
    if not selected:
        raise ValueError("relative comparator requires immutable validation supervision")
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(selected), 16):
            a, b, target = _batch(selected[start : start + 16], features)
            values.append(float(_loss(model, a, b, target)))
    return float(np.mean(values))


def select_relative_comparator_variant(
    variants: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Select an ablation from validation loss without consulting held-out evidence."""
    missing = [name for name in RELATIVE_VARIANTS if name not in variants]
    if missing:
        raise ValueError(f"relative comparator selection is missing variants: {missing}")

    validation_losses: dict[str, float] = {}
    for name in RELATIVE_VARIANTS:
        if "best_validation_loss" not in variants[name]:
            raise ValueError(f"relative comparator selection lacks validation loss: {name}")
        loss = float(variants[name]["best_validation_loss"])
        if not math.isfinite(loss):
            raise ValueError(f"relative comparator validation loss is not finite: {name}")
        validation_losses[name] = loss

    # Tuple order provides a stable, declared tie break without looking at any test metric.
    selected = min(
        RELATIVE_VARIANTS,
        key=lambda name: (validation_losses[name], RELATIVE_VARIANTS.index(name)),
    )
    return selected, {
        "policy": "minimum_best_validation_cross_entropy",
        "evidence_subset": "validation",
        "validation_losses": validation_losses,
        "tie_break_order": list(RELATIVE_VARIANTS),
        "heldout_source_used_for_selection": False,
        "heldout_mechanism_used_for_selection": False,
        "adversarial_pairs_used_for_selection": False,
    }


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["correct"] is not None]
    correct = sum(bool(row["correct"]) for row in selected)
    return {
        "pair_count": len(selected),
        "accuracy": None if not selected else correct / len(selected),
        "accuracy_95_percent_wilson_interval": _wilson_interval(correct, len(selected)),
    }


def _evaluate(
    model: FixedRigRelativeComparator,
    records: list[dict[str, Any]],
    features: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows = []
    swap_errors = []
    model.eval()
    with torch.no_grad():
        for record in records:
            a = features[str(record["motion_a"])][None]
            b = features[str(record["motion_b"])][None]
            output = model(a, b)
            swapped = model(b, a)
            probabilities = output.probabilities[0].cpu().numpy()
            swapped_probabilities = swapped.probabilities[0].cpu().numpy()
            swap_error = float(np.max(np.abs(probabilities - swapped_probabilities[[1, 0, 2]])))
            swap_errors.append(swap_error)
            prediction_index = int(np.argmax(probabilities))
            prediction = PREFERENCE_CLASSES[prediction_index]
            preference = record.get("preference")
            is_equivalent, equivalence_provenance = objective_equivalence_control(record)
            rows.append(
                {
                    "pair_id": record["pair_id"],
                    "subset": _subset(record),
                    "source_clip_id": record["source_clip_id"],
                    "perturbation_family": record["perturbation_mechanism"],
                    "supervision_category": record["supervision_category"],
                    "generated_supervision_category": record.get(
                        "generated_supervision_category", record["supervision_category"]
                    ),
                    "preference": preference,
                    "prediction": prediction,
                    "probabilities": {
                        label: float(probabilities[index])
                        for index, label in enumerate(PREFERENCE_CLASSES)
                    },
                    "swapped_probabilities": {
                        label: float(swapped_probabilities[index])
                        for index, label in enumerate(PREFERENCE_CLASSES)
                    },
                    "correct": None if preference is None else prediction == preference,
                    "swap_consistency_error": swap_error,
                    "adversarial_origin": record["adversarial_origin"],
                    "objective_equivalence_control": is_equivalent,
                    "objective_equivalence_control_provenance": equivalence_provenance,
                    "human_confidence": record.get("human_confidence"),
                    "reason_tags": record.get("reason_tags", []),
                }
            )

    heldout_source = [row for row in rows if row["subset"] == "heldout_source"]
    heldout_mechanism = [row for row in rows if row["subset"] == "heldout_mechanism"]
    human = [row for row in rows if row["supervision_category"] == "HUMAN_LABELED"]
    synthetic = [row for row in rows if row["supervision_category"] == "CERTAIN"]
    equivalent = [row for row in rows if row["objective_equivalence_control"]]
    adversarial = [row for row in rows if row["adversarial_origin"]]
    by_family: dict[str, Any] = {}
    for family in sorted({str(row["perturbation_family"]) for row in rows}):
        by_family[family] = _accuracy([row for row in rows if row["perturbation_family"] == family])
    by_confidence: dict[str, Any] = {}
    for confidence in sorted(
        {str(row["human_confidence"]) for row in human if row["human_confidence"] is not None}
    ):
        by_confidence[confidence] = _accuracy(
            [row for row in human if str(row["human_confidence"]) == confidence]
        )
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in human:
        for reason in row["reason_tags"]:
            by_reason[str(reason)].append(row)
    human_by_subset = {
        subset: _accuracy([row for row in human if row["subset"] == subset])
        for subset in ("train", "validation", "heldout_source", "heldout_mechanism", "adversarial")
    }
    synthetic_by_subset = {
        subset: _accuracy([row for row in synthetic if row["subset"] == subset])
        for subset in ("train", "validation", "heldout_source", "heldout_mechanism", "adversarial")
    }
    return {
        "heldout_source": _accuracy(heldout_source),
        "heldout_mechanism": _accuracy(heldout_mechanism),
        "human_labeled": {**_accuracy(human), "by_subset": human_by_subset},
        "synthetic_certain": {**_accuracy(synthetic), "by_subset": synthetic_by_subset},
        "equivalent_controls": {
            **_accuracy(equivalent),
            "false_preference_rate": (
                None
                if not equivalent
                else sum(row["prediction"] != "approximately_equal" for row in equivalent)
                / len(equivalent)
            ),
            "false_preference_rate_95_percent_wilson_interval": _wilson_interval(
                sum(row["prediction"] != "approximately_equal" for row in equivalent),
                len(equivalent),
            ),
            "selection_uses_explicit_objective_equivalence_flag": True,
        },
        "adversarial": _accuracy(adversarial),
        "swap_consistency": {
            "maximum_probability_error": max(swap_errors, default=0.0),
            "mean_probability_error": float(np.mean(swap_errors)) if swap_errors else 0.0,
        },
        "by_perturbation_family": by_family,
        "by_human_confidence": by_confidence,
        "by_human_reason_tag": {
            reason: _accuracy(reason_rows) for reason, reason_rows in sorted(by_reason.items())
        },
        "pair_scores": rows,
    }


def _train_variant(
    variant: str,
    records: list[dict[str, Any]],
    features: dict[str, torch.Tensor],
    backbone_checkpoint: Path,
    output_directory: Path,
    settings: RelativeExperimentConfig,
) -> tuple[dict[str, Any], Path]:
    fine_tune = variant == "relative_comparator_finetuned"
    seed_everything(settings.seed)
    backbone = model_from_checkpoint(backbone_checkpoint)
    if backbone.config.normalization != "group_norm":
        raise ValueError("relative comparator ablations require the unchanged GroupNorm baseline")
    comparator_config = RelativeComparatorConfig(
        comparison_channels=settings.comparison_channels,
        fine_tune_backbone=fine_tune,
    )
    model = FixedRigRelativeComparator(backbone, comparator_config)
    comparator_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": comparator_parameters, "lr": settings.comparator_learning_rate}
    ]
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    ]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": settings.backbone_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=settings.weight_decay)
    training = [record for record in records if _supervised(record) and _subset(record) == "train"]
    if not training:
        raise ValueError("relative comparator has no immutable training pairs")
    history = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(settings.epochs):
        model.train()
        order = list(range(len(training)))
        random.Random(settings.seed + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), settings.batch_size):
            selected = [training[index] for index in order[start : start + settings.batch_size]]
            a, b, target = _batch(selected, features)
            loss = _loss(model, a, b, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _validation_loss(model, records, features)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": float(np.mean(losses)),
                "validation_loss": validation,
            }
        )
        if validation < best_loss - 1.0e-7:
            best_loss = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= settings.patience:
            break
    model.load_state_dict(best_state)
    checkpoint = output_directory / f"{variant}.pt"
    torch.save(
        {
            "format_version": RELATIVE_CHECKPOINT_VERSION,
            "variant": variant,
            "comparator_config": comparator_config.model_dump(mode="json"),
            "experiment_config": settings.model_dump(mode="json"),
            "backbone_checkpoint": str(backbone_checkpoint),
            "backbone_checkpoint_sha256": file_sha256(backbone_checkpoint),
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    return {
        "variant": variant,
        "backbone_frozen": not fine_tune,
        "comparator_learning_rate": settings.comparator_learning_rate,
        "backbone_learning_rate": None if not fine_tune else settings.backbone_learning_rate,
        "selected_epoch": int(np.argmin([row["validation_loss"] for row in history])) + 1,
        "best_validation_loss": best_loss,
        "history": history,
        "final_evaluation_status": "withheld_until_variant_selection",
        "metrics": None,
        "checkpoint": str(checkpoint),
    }, checkpoint


def run_relative_comparator_experiment(
    pair_dataset_directory: Path,
    source_dataset_directory: Path,
    backbone_checkpoint: Path,
    human_labels_path: Path,
    human_audit_report_path: Path,
    absolute_evaluation_path: Path,
    output_directory: Path,
    *,
    config: RelativeExperimentConfig | None = None,
) -> RelativeExperimentResult:
    """Run matched ablations only after the human audit prerequisite is satisfied."""
    settings = RelativeExperimentConfig() if config is None else config
    audit = json.loads(Path(human_audit_report_path).read_text(encoding="utf-8"))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    prerequisite_reasons = []
    if audit["status"] != "complete":
        prerequisite_reasons.append("human_audit_incomplete")
    if int(audit["human_labeled_required_pair_count"]) < settings.minimum_human_audit_labels:
        prerequisite_reasons.append("insufficient_human_audit_labels")
    if not Path(human_labels_path).is_file():
        prerequisite_reasons.append("human_label_file_missing")
    elif audit.get("human_labels_sha256") != file_sha256(human_labels_path):
        prerequisite_reasons.append("human_audit_report_is_stale")
    if prerequisite_reasons:
        absolute = json.loads(Path(absolute_evaluation_path).read_text(encoding="utf-8"))
        report = {
            "format_version": RELATIVE_EXPERIMENT_VERSION,
            "status": "not_run_human_audit_required",
            "prerequisite_reasons": prerequisite_reasons,
            "human_audit": audit,
            "ablations": {
                "absolute_q": {
                    "status": "preserved",
                    "evaluation": str(absolute_evaluation_path),
                    "heldout_source": absolute["subsets"]["heldout_source"],
                },
                "relative_comparator_frozen": {"status": "not_run"},
                "relative_comparator_finetuned": {"status": "not_run"},
            },
            "variant_selection": {
                "status": "not_run",
                "policy": "minimum_best_validation_cross_entropy",
                "evidence_subset": "validation",
                "tie_break_order": list(RELATIVE_VARIANTS),
                "heldout_source_used_for_selection": False,
                "heldout_human_labels_used_for_selection": False,
                "heldout_human_audit_policy_used_for_selection": False,
                "heldout_source_evaluated_after_selection": False,
            },
            "motioncritic_baseline": _motioncritic_assessment(),
            "optimization_gate": {
                "passed": False,
                "reasons": prerequisite_reasons,
                "perceptual_cma_invoked": False,
            },
            "normalization_reopened": False,
            "graph_attention_started": False,
        }
        report_path = output / "relative_comparator_experiment.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return RelativeExperimentResult(report_path, False, None)

    # Freeze fitting and ablation-selection evidence before overlaying any held-out human
    # judgments or the policy derived from those judgments.  The held-out audit remains a
    # prerequisite and a final-evaluation target, but cannot change selection membership/targets.
    base_records = load_perceptual_pairs(pair_dataset_directory)
    selection_records = relative_selection_records(base_records)
    selection_contract = relative_selection_contract(base_records)
    selection_features = _load_features(
        selection_records,
        Path(pair_dataset_directory),
        source_dataset_directory,
    )
    variants: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, Path] = {}
    for variant in RELATIVE_VARIANTS:
        variants[variant], checkpoints[variant] = _train_variant(
            variant,
            selection_records,
            selection_features,
            backbone_checkpoint,
            output,
            settings,
        )
        variants[variant]["selection_records_and_targets_sha256"] = selection_contract[
            "records_and_targets_sha256"
        ]
    selected_variant, variant_selection = select_relative_comparator_variant(variants)
    records = apply_audited_supervision_policy(
        base_records,
        audit,
        human_labels_path,
    )
    absolute = json.loads(Path(absolute_evaluation_path).read_text(encoding="utf-8"))
    final_features = _load_features(
        records,
        Path(pair_dataset_directory),
        source_dataset_directory,
    )
    for variant in RELATIVE_VARIANTS:
        final_model, _ = load_relative_comparator(checkpoints[variant])
        variants[variant]["metrics"] = _evaluate(final_model, records, final_features)
        variants[variant]["final_evaluation_status"] = "complete_after_validation_only_selection"
    metrics = variants[selected_variant]["metrics"]
    gate_reasons = []
    heldout_accuracy = metrics["heldout_source"]["accuracy"]
    if heldout_accuracy is None or float(heldout_accuracy) < settings.heldout_source_accuracy_gate:
        gate_reasons.append("heldout_source:below_65_percent")
    equivalent_rate = metrics["equivalent_controls"]["false_preference_rate"]
    if equivalent_rate is None or float(equivalent_rate) > (
        settings.maximum_equivalent_false_preference_rate
    ):
        gate_reasons.append("equivalent_controls:false_preference_rate")
    if metrics["swap_consistency"]["maximum_probability_error"] > (
        settings.maximum_swap_consistency_error
    ):
        gate_reasons.append("swap_consistency")
    gate_passed = not gate_reasons
    report = {
        "format_version": RELATIVE_EXPERIMENT_VERSION,
        "status": "complete",
        "human_audit": audit,
        "split_policy": "all derivatives of a source/style remain in its source split",
        "architecture": (
            "shared fixed-rig temporal encoder; ordered [EA, EB, EB-EA, abs(EB-EA), EA*EB] "
            "head; symmetric equality head; exactly swap-equivariant three-way probabilities"
        ),
        "selection_supervision": selection_contract,
        "ablations": {
            "absolute_q": {
                "status": "preserved",
                "evaluation": str(absolute_evaluation_path),
                "heldout_source": absolute["subsets"]["heldout_source"],
            },
            **variants,
        },
        "selected_variant": selected_variant,
        "variant_selection": {
            **variant_selection,
            "selected_variant": selected_variant,
            "heldout_source_metadata_loaded_before_selection": True,
            "heldout_source_motion_features_loaded_before_selection": False,
            "heldout_source_evaluated_after_selection": True,
            "heldout_human_labels_used_for_selection": False,
            "heldout_human_audit_policy_used_for_selection": False,
            "all_variant_final_evaluations_started_after_selection": True,
            "optimization_gate_uses_selected_variant_only": True,
        },
        "motioncritic_baseline": _motioncritic_assessment(),
        "optimization_gate": {
            "passed": gate_passed,
            "reasons": gate_reasons,
            "minimum_heldout_source_accuracy": settings.heldout_source_accuracy_gate,
            "heldout_source_accuracy_95_percent_wilson_interval": metrics["heldout_source"][
                "accuracy_95_percent_wilson_interval"
            ],
            "perceptual_cma_invoked": False,
        },
        "normalization_reopened": False,
        "graph_attention_started": False,
    }
    report_path = output / "relative_comparator_experiment.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return RelativeExperimentResult(report_path, gate_passed, checkpoints[selected_variant])


def load_relative_comparator(
    checkpoint_path: Path,
) -> tuple[FixedRigRelativeComparator, dict[str, Any]]:
    """Load one frozen or fine-tuned relative-comparator artifact."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != RELATIVE_CHECKPOINT_VERSION:
        raise ValueError("unsupported relative-comparator checkpoint")
    backbone_path = Path(str(payload["backbone_checkpoint"]))
    if file_sha256(backbone_path) != payload["backbone_checkpoint_sha256"]:
        raise ValueError("relative comparator backbone checkpoint checksum mismatch")
    model = FixedRigRelativeComparator(
        model_from_checkpoint(backbone_path),
        RelativeComparatorConfig.model_validate(payload["comparator_config"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
