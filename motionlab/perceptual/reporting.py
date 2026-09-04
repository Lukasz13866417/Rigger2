"""Reproducible milestone summary for fixed-rig perceptual-quality experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PERCEPTUAL_MILESTONE_VERSION = "motionlab.perceptual_quality_milestone.v1"


def compile_perceptual_milestone_report(
    dataset_summary_path: Path,
    evaluation_path: Path,
    cma_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Combine dataset, ranking-gate, and optimization evidence without hiding nulls."""
    dataset = json.loads(Path(dataset_summary_path).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    cma = json.loads(Path(cma_report_path).read_text(encoding="utf-8"))

    independent_subsets = ("heldout_source", "heldout_mechanism", "adversarial")
    subset_metrics = evaluation["subsets"]
    independent_count = sum(
        int(subset_metrics[name]["directional_pair_count"]) for name in independent_subsets
    )
    independent_correct = sum(
        float(subset_metrics[name]["pairwise_accuracy"])
        * int(subset_metrics[name]["directional_pair_count"])
        for name in independent_subsets
    )
    heldout_subtle_count = sum(
        1
        for row in evaluation["pair_scores"]
        if row["supervision_category"] == "UNORDERED"
        and row["subset"] in {"heldout_source", "heldout_mechanism"}
    )
    report: dict[str, Any] = {
        "format_version": PERCEPTUAL_MILESTONE_VERSION,
        "decision": {
            "perceptual_critic_ready_for_optimization": bool(
                evaluation["optimization_gate"]["passed"]
            ),
            "next_priority": "improve_perceptual_supervision_and_critic",
            "variable_rig_graph_attention_started": False,
            "normalization_reopened": False,
            "deterministic_repair_stack_status": "frozen_except_concrete_bugs",
        },
        "dataset": {
            "hard_feasible_pair_count": dataset["hard_feasible_pair_count"],
            "certain_pair_count": dataset["certain_pair_count"],
            "human_labeled_pair_count": dataset["human_labeled_pair_count"],
            "unordered_pair_count": dataset["unordered_pair_count"],
            "equivalent_control_count": dataset["equivalent_control_count"],
            "hard_feasible_adversarial_pair_count": dataset["hard_feasible_adversarial_pair_count"],
            "red_team_only_adversarial_pair_count": dataset["red_team_only_adversarial_pair_count"],
            "mechanism_counts": dataset["mechanism_counts"],
        },
        "preference_evaluation": {
            "independent_directional_pair_count": independent_count,
            "independent_pairwise_accuracy": (
                independent_correct / independent_count if independent_count else None
            ),
            "heldout_source": subset_metrics["heldout_source"],
            "heldout_mechanism": subset_metrics["heldout_mechanism"],
            "adversarial": subset_metrics["adversarial"],
            "heldout_subtle_unordered_pair_count": heldout_subtle_count,
            "heldout_subtle_accuracy": None,
            "heldout_subtle_note": (
                "No automatic order is asserted; these pairs are queued for human labels."
            ),
            "equivalent_false_preference_rate": evaluation[
                "equivalent_control_false_preference_rate"
            ],
            "calibration_by_subset": {
                name: metrics["expected_calibration_error"]
                for name, metrics in subset_metrics.items()
            },
            "human_pairwise_agreement": evaluation["pairwise_human_agreement"],
            "gate": evaluation["optimization_gate"],
        },
        "perceptual_cma": {
            "status": cma["status"],
            "cma_es_invoked": cma.get("cma_es_invoked", True),
            "success_rate": cma.get("perceptual_cma_success_rate"),
            "exploit_rate": cma.get("perceptual_cma_exploit_rate"),
            "representative_successful_refinements": cma.get(
                "representative_successful_refinements", []
            ),
            "representative_critic_exploits": cma.get("representative_critic_exploits", []),
        },
        "artifacts": {
            "dataset_summary": str(dataset_summary_path),
            "evaluation": str(evaluation_path),
            "cma_report": str(cma_report_path),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
