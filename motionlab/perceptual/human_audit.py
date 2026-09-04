"""Stratified human audit and synthetic-supervision agreement policy."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from motionlab.dataset.io import file_sha256
from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.perceptual.labeling import apply_human_labels, load_latest_human_labels

HUMAN_AUDIT_VERSION = "motionlab.perceptual_human_audit.v1"
HUMAN_AUDIT_MINIMUM_AGREEMENT = 0.70


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


def _difficulty(record: dict[str, Any], family_strengths: dict[str, list[float]]) -> str:
    if record["preference"] == "approximately_equal":
        return "equivalent"
    if record["supervision_category"] == "UNORDERED":
        return "subtle"
    if record["adversarial_origin"]:
        return "clear"
    strength = record.get("perturbation_strength")
    values = family_strengths[str(record["perturbation_mechanism"])]
    if strength is None or len(values) < 2:
        return "medium"
    return "medium" if float(strength) == min(values) else "clear"


def prepare_perceptual_human_audit(
    pair_dataset_directory: Path,
    absolute_evaluation_path: Path,
    absolute_cma_report_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Select all held-out-source/style pairs plus preserved admissible exploits."""
    pairs = load_perceptual_pairs(pair_dataset_directory)
    evaluation = json.loads(Path(absolute_evaluation_path).read_text(encoding="utf-8"))
    cma = json.loads(Path(absolute_cma_report_path).read_text(encoding="utf-8"))
    scores = {str(row["pair_id"]): row for row in evaluation["pair_scores"]}
    mandatory = [str(row["pair_id"]) for row in cma.get("representative_critic_exploits", [])[:5]]
    selected = [
        record
        for record in pairs
        if record["source_split"] == "test" or record["adversarial_origin"]
    ]
    selected_ids = {str(record["pair_id"]) for record in selected}
    missing = [pair_id for pair_id in mandatory if pair_id not in selected_ids]
    if missing:
        raise ValueError(f"confident misrankings are absent from audit candidates: {missing}")
    family_strengths: dict[str, list[float]] = defaultdict(list)
    for record in pairs:
        strength = record.get("perturbation_strength")
        if strength is not None:
            family_strengths[str(record["perturbation_mechanism"])].append(float(strength))
    for family in family_strengths:
        family_strengths[family] = sorted(set(family_strengths[family]))

    mandatory_order = {pair_id: index for index, pair_id in enumerate(mandatory)}
    selected.sort(
        key=lambda record: (
            0 if record["pair_id"] in mandatory_order else 1,
            mandatory_order.get(str(record["pair_id"]), 0),
            str(record["source_clip_id"]),
            str(record["perturbation_mechanism"]),
            float(record.get("perturbation_strength") or 0.0),
            str(record["pair_id"]),
        )
    )
    queue_rows = []
    for rank, record in enumerate(selected, start=1):
        pair_id = str(record["pair_id"])
        score = scores.get(pair_id)
        reasons = []
        if pair_id in mandatory_order:
            reasons.append("confident_heldout_source_misranking")
        if record["source_split"] == "test":
            reasons.append("heldout_source_style")
        if record["adversarial_origin"]:
            reasons.append("adversarial_origin")
        if record["preference"] == "approximately_equal":
            reasons.append("equivalent_control")
        queue_rows.append(
            {
                "audit_rank": rank,
                "pair_id": pair_id,
                "source_clip_id": record["source_clip_id"],
                "perturbation_family": record["perturbation_mechanism"],
                "perturbation_strength": record.get("perturbation_strength"),
                "difficulty": _difficulty(record, family_strengths),
                "generated_supervision_category": record["supervision_category"],
                "generated_preference": record["preference"],
                "absolute_critic_probability_a_better": (
                    None if score is None else score["probability_a_better"]
                ),
                "absolute_critic_correct": None if score is None else score["correct"],
                "selection_reasons": reasons,
            }
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    queue_path = output / "human_audit_queue.jsonl"
    queue_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in queue_rows),
        encoding="utf-8",
    )
    summary = {
        "format_version": HUMAN_AUDIT_VERSION,
        "status": "ready_for_human_labeling",
        "queue": str(queue_path),
        "required_pair_count": len(queue_rows),
        "confident_misranking_count": sum(
            "confident_heldout_source_misranking" in row["selection_reasons"] for row in queue_rows
        ),
        "source_counts": dict(sorted(Counter(row["source_clip_id"] for row in queue_rows).items())),
        "family_counts": dict(
            sorted(Counter(row["perturbation_family"] for row in queue_rows).items())
        ),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in queue_rows).items())),
        "selection_policy": (
            "all fixed-rig test-source/style pairs, both admissible adversarial pairs, and the "
            "five confident absolute-critic misrankings first"
        ),
        "human_labels_created_by_this_command": 0,
    }
    (output / "human_audit_plan.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def analyze_perceptual_human_audit(
    pair_dataset_directory: Path,
    audit_queue_path: Path,
    labels_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Measure human/synthetic agreement and suspend unverified automatic directions."""
    pairs = load_perceptual_pairs(pair_dataset_directory)
    by_id = {str(record["pair_id"]): record for record in pairs}
    queue = [
        json.loads(line)
        for line in Path(audit_queue_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = load_latest_human_labels(labels_path)
    unknown = sorted(set(labels).difference(by_id))
    if unknown:
        raise ValueError(f"human labels reference unknown pairs: {unknown}")
    required_ids = {str(row["pair_id"]) for row in queue}
    completed_ids = required_ids.intersection(labels)
    families = sorted({str(row["perturbation_family"]) for row in queue})
    family_reports: dict[str, Any] = {}
    supervision_policy: dict[str, Any] = {}
    audit_complete = completed_ids == required_ids
    for family in families:
        family_ids = {str(row["pair_id"]) for row in queue if row["perturbation_family"] == family}
        family_labels = [labels[pair_id] for pair_id in sorted(family_ids.intersection(labels))]
        comparisons = []
        for pair_id in sorted(family_ids.intersection(labels)):
            generated = by_id[pair_id]["preference"]
            if generated is not None:
                comparisons.append(labels[pair_id]["preference"] == generated)
        agreements = sum(comparisons)
        agreement = None if not comparisons else agreements / len(comparisons)
        family_reports[family] = {
            "required_pair_count": len(family_ids),
            "human_labeled_pair_count": len(family_labels),
            "synthetic_comparison_count": len(comparisons),
            "agreement": agreement,
            "agreement_95_percent_wilson_interval": _wilson_interval(agreements, len(comparisons)),
            "human_preference_counts": dict(
                sorted(Counter(str(label["preference"]) for label in family_labels).items())
            ),
            "human_confidence_counts": dict(
                sorted(
                    Counter(
                        str(label.get("confidence", "unreported")) for label in family_labels
                    ).items()
                )
            ),
        }
        generated_categories = {
            str(by_id[pair_id]["supervision_category"]) for pair_id in family_ids
        }
        exact_equivalence = all(
            by_id[pair_id]["preference"] == "approximately_equal" for pair_id in family_ids
        )
        if exact_equivalence:
            category = "CERTAIN"
            reason = "exact equivalence construction"
        elif (
            audit_complete
            and comparisons
            and agreement is not None
            and agreement >= HUMAN_AUDIT_MINIMUM_AGREEMENT
            and "CERTAIN" in generated_categories
        ):
            category = "CERTAIN"
            reason = "human audit confirms automatic ordering"
        else:
            category = "HUMAN_REQUIRED"
            reason = (
                "human audit incomplete"
                if not audit_complete
                else "automatic ordering is absent or below the human-agreement threshold"
            )
        supervision_policy[family] = {
            "automatic_supervision": category,
            "minimum_human_agreement": HUMAN_AUDIT_MINIMUM_AGREEMENT,
            "reason": reason,
        }
    report = {
        "format_version": HUMAN_AUDIT_VERSION,
        "status": "complete" if audit_complete else "pending_human_labels",
        "required_pair_count": len(required_ids),
        "human_labeled_required_pair_count": len(completed_ids),
        "remaining_pair_count": len(required_ids - completed_ids),
        "extraneous_human_label_count": len(set(labels).difference(required_ids)),
        "human_labels_are_append_only": True,
        "human_labels_path": str(labels_path),
        "human_labels_sha256": (file_sha256(labels_path) if Path(labels_path).is_file() else None),
        "family_agreement": family_reports,
        "supervision_policy": supervision_policy,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def apply_audited_supervision_policy(
    records: list[dict[str, Any]],
    audit_report: dict[str, Any],
    labels_path: Path,
) -> list[dict[str, Any]]:
    """Apply human labels, then remove every unverified automatic direction."""
    merged = apply_human_labels(records, labels_path)
    policy = audit_report["supervision_policy"]
    result = []
    for source in merged:
        record = dict(source)
        if record["supervision_category"] != "HUMAN_LABELED":
            family = str(record["perturbation_mechanism"])
            family_policy = policy.get(family, {"automatic_supervision": "HUMAN_REQUIRED"})
            if family_policy["automatic_supervision"] != "CERTAIN":
                record["generated_preference"] = record["preference"]
                record["preference"] = None
                record["supervision_category"] = "HUMAN_REQUIRED"
                record["label_basis"] = None
        result.append(record)
    return result
