"""Separated fixed-rig critic evaluations, shortcut probes, and simple baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from motionlab.critic.data import (
    RELEVANT_SUMMARY_METRIC,
    SUMMARY_METRIC_NAMES,
    FixedRigSampleDataset,
)
from motionlab.critic.forensics import audit_sham_controls
from motionlab.critic.model import FIRST_CRITIC_DEFECTS, FixedRigFlatTCN
from motionlab.critic.training import model_from_checkpoint, seed_everything

EVALUATION_VERSION = "motionlab.fixed_rig_flat_tcn.evaluation.v3"


def binary_auroc(
    target: np.ndarray[Any, np.dtype[Any]], score: np.ndarray[Any, np.dtype[Any]]
) -> float | None:
    """Compute tie-aware binary AUROC without an external metrics dependency."""
    target = np.asarray(target, dtype=np.bool_)
    score = np.asarray(score, dtype=np.float64)
    positive = int(np.sum(target))
    negative = int(np.sum(~target))
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(score, method="average")
    positive_rank_sum = float(np.sum(ranks[target]))
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def average_precision(
    target: np.ndarray[Any, np.dtype[Any]], score: np.ndarray[Any, np.dtype[Any]]
) -> float | None:
    target = np.asarray(target, dtype=np.bool_)
    if not np.any(target):
        return None
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    ordered = target[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / np.sum(ordered))


def _binary_localization(
    target: np.ndarray[Any, np.dtype[Any]], predicted: np.ndarray[Any, np.dtype[Any]]
) -> tuple[float, float]:
    target = np.asarray(target, dtype=np.bool_)
    predicted = np.asarray(predicted, dtype=np.bool_)
    true_positive = int(np.sum(target & predicted))
    false_positive = int(np.sum(~target & predicted))
    false_negative = int(np.sum(target & ~predicted))
    f1 = 2.0 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return f1, iou


def _ece(
    target: np.ndarray[Any, np.dtype[Any]], probability: np.ndarray[Any, np.dtype[Any]]
) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    result = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (probability >= lower) & (probability < lower + 0.1)
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected])) - float(np.mean(target[selected]))
            )
    return result


def _predict(
    model: FixedRigFlatTCN,
    dataset: FixedRigSampleDataset,
    *,
    batch_size: int = 16,
) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray[Any, np.dtype[Any]]]] = {
        "clip_probability": [],
        "clip_severity": [],
        "family_ranking_score": [],
        "frame_probability": [],
        "part_probability": [],
        "representation": [],
        "clip_target": [],
        "severity_target": [],
        "frame_target": [],
        "part_target": [],
        "summary_features": [],
    }
    metadata: list[dict[str, Any]] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [
                dataset[index] for index in range(start, min(start + batch_size, len(dataset)))
            ]
            frame_features = torch.stack([item["frame_features"] for item in items]).to(device)
            output = model(frame_features)
            values = {
                "clip_probability": torch.sigmoid(output.clip_logits),
                "clip_severity": output.clip_severity,
                "family_ranking_score": output.family_ranking_score,
                "frame_probability": torch.sigmoid(output.frame_logits),
                "part_probability": torch.sigmoid(output.part_logits),
                "representation": output.representation,
                "clip_target": torch.stack([item["clip_target"] for item in items]),
                "severity_target": torch.stack([item["severity_target"] for item in items]),
                "frame_target": torch.stack([item["frame_target"] for item in items]),
                "part_target": torch.stack([item["part_target"] for item in items]),
                "summary_features": torch.stack([item["summary_features"] for item in items]),
            }
            for key, value in values.items():
                arrays[key].append(value.detach().cpu().numpy())
            for item in items:
                metadata.append(
                    {
                        key: item[key]
                        for key in (
                            "sample_id",
                            "source_clip_id",
                            "source_split",
                            "sample_role",
                            "corruption_family",
                            "corruption_mechanism",
                            "severity_label",
                            "ordinal_chain_id",
                            "ordinal_rank",
                            "measured_severity",
                            "sample_path",
                            "source_path",
                        )
                    }
                )
    return {key: np.concatenate(value, axis=0) for key, value in arrays.items()} | {
        "metadata": metadata
    }


def _per_family_metrics(
    prediction: Mapping[str, Any], *, primary_mechanisms_only: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, family in enumerate(FIRST_CRITIC_DEFECTS):
        family_target = prediction["clip_target"][:, index]
        sample_indices = np.arange(len(family_target))
        if primary_mechanisms_only:
            primary = np.asarray(
                [record["corruption_family"] == family for record in prediction["metadata"]],
                dtype=np.bool_,
            )
            # Other mechanisms carrying this family as an explicit collateral label are neither
            # positives for the primary-mechanism test nor clean negatives for that family.
            sample_indices = sample_indices[primary | (family_target < 0.5)]
        clip_target = family_target[sample_indices]
        clip_probability = prediction["clip_probability"][sample_indices, index]
        frame_target = prediction["frame_target"][sample_indices, :, index]
        frame_predicted = prediction["frame_probability"][sample_indices, :, index] >= 0.5
        part_target = prediction["part_target"][sample_indices, :, :, index]
        part_predicted = prediction["part_probability"][sample_indices, :, :, index] >= 0.5
        severity_target = prediction["severity_target"][sample_indices, index]
        severity_prediction = prediction["clip_severity"][sample_indices, index]
        family_rank_prediction = prediction["family_ranking_score"][sample_indices, index]
        frame_f1, frame_iou = _binary_localization(frame_target, frame_predicted)
        part_f1, part_iou = _binary_localization(part_target, part_predicted)
        selected = clip_target > 0.5
        severity_correlation: float | None = None
        family_rank_correlation: float | None = None
        if np.sum(selected) >= 2 and len(np.unique(severity_target[selected])) >= 2:
            value = spearmanr(
                severity_target[selected],
                severity_prediction[selected],
            ).statistic
            severity_correlation = float(value) if np.isfinite(value) else None
            family_rank_value = spearmanr(
                severity_target[selected],
                family_rank_prediction[selected],
            ).statistic
            family_rank_correlation = (
                float(family_rank_value) if np.isfinite(family_rank_value) else None
            )
        result[family] = {
            "sample_count": len(clip_target),
            "positive_count": int(np.sum(selected)),
            "defect_auroc": binary_auroc(clip_target, clip_probability),
            "defect_average_precision": average_precision(clip_target, clip_probability),
            "temporal_localization_f1": frame_f1,
            "temporal_localization_iou": frame_iou,
            "anatomical_part_localization_f1": part_f1,
            "anatomical_part_localization_iou": part_iou,
            "severity_rank_spearman": severity_correlation,
            "family_ranking_head_spearman": family_rank_correlation,
            "calibration_ece": _ece(clip_target, clip_probability),
        }
    return result


def _severity_bin_metrics(prediction: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metadata = prediction["metadata"]
    for family_index, family in enumerate(FIRST_CRITIC_DEFECTS):
        family_result: dict[str, Any] = {}
        for severity in ("near_threshold", "subtle", "moderate", "clear", "severe"):
            indices = [
                index
                for index, record in enumerate(metadata)
                if record["corruption_family"] == family and record["severity_label"] == severity
            ]
            if not indices:
                family_result[severity] = {"count": 0, "recall": None, "mean_probability": None}
                continue
            probability = prediction["clip_probability"][indices, family_index]
            family_result[severity] = {
                "count": len(indices),
                "recall": float(np.mean(probability >= 0.5)),
                "mean_probability": float(np.mean(probability)),
            }
        result[family] = family_result
    return result


def _pairwise_ranking(
    dataset_directory: Path,
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score_by_id: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    family_by_id: dict[str, str] = {}
    for prediction in predictions:
        for index, record in enumerate(prediction["metadata"]):
            score_by_id[record["sample_id"]] = prediction["family_ranking_score"][index]
            family_by_id[record["sample_id"]] = record["corruption_family"]
    preferences = [
        json.loads(line)
        for line in (Path(dataset_directory) / "preferences.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    correct = 0
    total = 0
    inversions: list[dict[str, Any]] = []
    by_family: dict[str, list[bool]] = {}
    for record in preferences:
        better = str(record["better_sample_id"])
        worse = str(record["worse_sample_id"])
        if better not in score_by_id or worse not in score_by_id:
            continue
        family = family_by_id[worse]
        if family not in FIRST_CRITIC_DEFECTS:
            continue
        family_index = FIRST_CRITIC_DEFECTS.index(family)
        margin = float(score_by_id[worse][family_index] - score_by_id[better][family_index])
        is_correct = margin > 0.0
        correct += int(is_correct)
        total += 1
        by_family.setdefault(family, []).append(is_correct)
        if not is_correct:
            inversions.append(
                {
                    "preference_id": record["preference_id"],
                    "family": family,
                    "better_sample_id": better,
                    "worse_sample_id": worse,
                    "predicted_margin": margin,
                }
            )
    return {
        "pair_count": total,
        "accuracy": correct / total if total else None,
        "by_family": {
            family: {"count": len(values), "accuracy": float(np.mean(values))}
            for family, values in sorted(by_family.items())
        },
        "inversions": sorted(inversions, key=lambda value: value["predicted_margin"])[:10],
    }


def _cross_mechanism_consistency(
    dataset_directory: Path, predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    score_by_id: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for prediction in predictions:
        for index, record in enumerate(prediction["metadata"]):
            score_by_id[record["sample_id"]] = prediction["family_ranking_score"][index]
    records = [
        json.loads(line)
        for line in (Path(dataset_directory) / "consistency.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    differences: dict[str, list[float]] = {}
    for record in records:
        if record.get("target_type") != "per_defect_severity_equal":
            continue
        sample_a = str(record["sample_a_id"])
        sample_b = str(record["sample_b_id"])
        family = str(record["family"])
        if sample_a not in score_by_id or sample_b not in score_by_id:
            continue
        if family not in FIRST_CRITIC_DEFECTS:
            continue
        index = FIRST_CRITIC_DEFECTS.index(family)
        differences.setdefault(family, []).append(
            abs(float(score_by_id[sample_a][index] - score_by_id[sample_b][index]))
        )
    return {
        family: {
            "pair_count": len(values),
            "mean_absolute_predicted_severity_difference": float(np.mean(values)),
            "maximum_absolute_predicted_severity_difference": float(np.max(values)),
        }
        for family, values in sorted(differences.items())
    }


def _mechanism_probe(
    train: Mapping[str, Any], test: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    train_indices = [
        index
        for index, record in enumerate(train["metadata"])
        if record["sample_role"] == "hard_corruption"
    ]
    test_indices = [
        index
        for index, record in enumerate(test["metadata"])
        if record["sample_role"] == "hard_corruption"
    ]
    mechanisms = sorted(
        {train["metadata"][index]["corruption_mechanism"] for index in train_indices}
    )
    test_indices = [
        index
        for index in test_indices
        if test["metadata"][index]["corruption_mechanism"] in mechanisms
    ]
    if len(mechanisms) < 2 or not test_indices:
        return {"accuracy": None, "chance_accuracy": None, "mechanism_count": len(mechanisms)}
    x_train = torch.from_numpy(train["representation"][train_indices]).float()
    x_test = torch.from_numpy(test["representation"][test_indices]).float()
    mean = x_train.mean(dim=0, keepdim=True)
    scale = torch.clamp(x_train.std(dim=0, keepdim=True), min=1.0e-6)
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    y_train = torch.tensor(
        [
            mechanisms.index(train["metadata"][index]["corruption_mechanism"])
            for index in train_indices
        ]
    )
    y_test = torch.tensor(
        [
            mechanisms.index(test["metadata"][index]["corruption_mechanism"])
            for index in test_indices
        ]
    )
    seed_everything(seed)
    classifier = torch.nn.Linear(x_train.shape[1], len(mechanisms))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.03)
    for _ in range(250):
        loss = torch.nn.functional.cross_entropy(classifier(x_train), y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = float(torch.mean((classifier(x_test).argmax(dim=-1) == y_test).float()))
    return {
        "accuracy": accuracy,
        "chance_accuracy": 1.0 / len(mechanisms),
        "mechanism_count": len(mechanisms),
        "train_sample_count": len(train_indices),
        "test_sample_count": len(test_indices),
        "mechanisms": mechanisms,
        "interpretation_requires_heldout_performance": True,
    }


def _summary_baselines(
    train: Mapping[str, Any], test: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    train_x = np.asarray(train["summary_features"], dtype=np.float32)
    test_x = np.asarray(test["summary_features"], dtype=np.float32)
    train_y = np.asarray(train["clip_target"], dtype=np.float32)
    test_y = np.asarray(test["clip_target"], dtype=np.float32)
    metric_result: dict[str, Any] = {}
    mlp_test_indices: dict[str, np.ndarray[Any, np.dtype[np.int64]]] = {}
    for family_index, family in enumerate(FIRST_CRITIC_DEFECTS):
        metric_index = RELEVANT_SUMMARY_METRIC[family]
        family_target = test_y[:, family_index]
        primary = np.asarray(
            [record["corruption_family"] == family for record in test["metadata"]],
            dtype=np.bool_,
        )
        indices = np.flatnonzero(primary | (family_target < 0.5))
        mlp_test_indices[family] = indices
        metric_result[family] = {
            "summary_metric_index": metric_index,
            "summary_metric_name": SUMMARY_METRIC_NAMES[metric_index],
            "auroc": binary_auroc(test_y[indices, family_index], test_x[indices, metric_index]),
            "average_precision": average_precision(
                test_y[indices, family_index], test_x[indices, metric_index]
            ),
        }
    mean = train_x.mean(axis=0, keepdims=True)
    scale = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-6)
    x_train = torch.from_numpy((train_x - mean) / scale)
    x_test = torch.from_numpy((test_x - mean) / scale)
    y_train = torch.from_numpy(train_y)
    seed_everything(seed)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, len(FIRST_CRITIC_DEFECTS)),
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for _ in range(400):
        logits = mlp(x_train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probability = torch.sigmoid(mlp(x_test)).numpy()
    mlp_result = {
        family: {
            "auroc": binary_auroc(
                test_y[mlp_test_indices[family], index],
                probability[mlp_test_indices[family], index],
            ),
            "average_precision": average_precision(
                test_y[mlp_test_indices[family], index],
                probability[mlp_test_indices[family], index],
            ),
        }
        for index, family in enumerate(FIRST_CRITIC_DEFECTS)
    }
    return {"deterministic_metric": metric_result, "summary_mlp": mlp_result}


def _failure_examples(
    known: Mapping[str, Any],
    heldout: Mapping[str, Any],
    all_prediction: Mapping[str, Any],
    ranking: Mapping[str, Any],
) -> dict[str, Any]:
    def examples(prediction: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for sample_index, record in enumerate(prediction["metadata"]):
            for family_index, family in enumerate(FIRST_CRITIC_DEFECTS):
                target = bool(prediction["clip_target"][sample_index, family_index])
                probability = float(prediction["clip_probability"][sample_index, family_index])
                failure_score = probability if not target else 1.0 - probability
                if mode == "sham" and record["sample_role"] != "sham_control":
                    continue
                if mode == "error" and (probability >= 0.5) == target:
                    continue
                values.append(
                    {
                        "sample_id": record["sample_id"],
                        "source_clip_id": record["source_clip_id"],
                        "mechanism": record["corruption_mechanism"],
                        "family": family,
                        "target": target,
                        "probability": probability,
                        "failure_score": failure_score,
                    }
                )
        return sorted(values, key=lambda value: value["failure_score"], reverse=True)[:10]

    return {
        "known_mechanism_errors": examples(known, "error"),
        "sham_false_positives": examples(all_prediction, "sham"),
        "heldout_mechanism_errors": examples(heldout, "error"),
        "subtle_ranking_inversions": ranking["inversions"],
    }


def evaluate_fixed_rig_tcn(
    dataset_directory: Path,
    checkpoint_path: Path,
    output_directory: Path,
    *,
    seed: int = 2027,
) -> dict[str, Any]:
    """Run all required known, held-out, sham, ranking, probe, and baseline regimes."""
    model = model_from_checkpoint(checkpoint_path)
    train_dataset = FixedRigSampleDataset(dataset_directory, regime="train")
    known_dataset = FixedRigSampleDataset(dataset_directory, regime="known_test")
    heldout_dataset = FixedRigSampleDataset(dataset_directory, regime="heldout_mechanism")
    all_dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    train = _predict(model, train_dataset)
    known = _predict(model, known_dataset)
    heldout = _predict(model, heldout_dataset)
    all_prediction = _predict(model, all_dataset)
    ranking = _pairwise_ranking(dataset_directory, (all_prediction,))
    sham_forensics = audit_sham_controls(
        all_dataset,
        all_prediction,
        Path(output_directory) / "sham_forensics",
    )
    report = {
        "format_version": EVALUATION_VERSION,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_directory),
        "known_mechanism_unseen_source": _per_family_metrics(known, primary_mechanisms_only=True),
        "heldout_mechanism": _per_family_metrics(heldout, primary_mechanisms_only=True),
        "all_labeled_symptom_context": {
            "known_mechanism_unseen_source": _per_family_metrics(known),
            "heldout_mechanism": _per_family_metrics(heldout),
        },
        "by_severity_bin_known_mechanism": _severity_bin_metrics(known),
        "by_severity_bin_heldout_mechanism": _severity_bin_metrics(heldout),
        "sham": {key: value for key, value in sham_forensics.items() if key != "records"},
        "subtle_and_ordinal_ranking": ranking,
        "cross_mechanism_severity_consistency": _cross_mechanism_consistency(
            dataset_directory, (known, heldout)
        ),
        "mechanism_id_linear_probe": _mechanism_probe(train, known, seed),
        "trivial_baselines": _summary_baselines(train, known, seed),
    }
    report["failure_examples"] = _failure_examples(known, heldout, all_prediction, ranking)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    all_predictions = (train, known, heldout)
    np.savez_compressed(
        output / "representations.npz",
        sample_ids=np.asarray(
            [
                record["sample_id"]
                for prediction in all_predictions
                for record in prediction["metadata"]
            ],
            dtype=np.str_,
        ),
        regimes=np.asarray(
            [
                regime
                for regime, prediction in zip(
                    ("train", "known_test", "heldout_mechanism"), all_predictions, strict=True
                )
                for _ in prediction["metadata"]
            ],
            dtype=np.str_,
        ),
        representation=np.concatenate(
            [prediction["representation"] for prediction in all_predictions], axis=0
        ).astype(np.float32),
        clip_probability=np.concatenate(
            [prediction["clip_probability"] for prediction in all_predictions], axis=0
        ).astype(np.float32),
        clip_severity=np.concatenate(
            [prediction["clip_severity"] for prediction in all_predictions], axis=0
        ).astype(np.float32),
        family_ranking_score=np.concatenate(
            [prediction["family_ranking_score"] for prediction in all_predictions], axis=0
        ).astype(np.float32),
    )
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "failure_examples.json").write_text(
        json.dumps(report["failure_examples"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
