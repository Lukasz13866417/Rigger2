from __future__ import annotations

import inspect

from motionlab.optimization.production_benchmark import (
    _optimize_without_clean_reference,
    inference_available_motion,
    select_repair_family,
)
from motionlab.testing.synthetic import make_synthetic_contact_walk


def test_inference_contract_removes_synthetic_labels_but_keeps_task_speed() -> None:
    clip = make_synthetic_contact_walk(num_cycles=2).with_updates(
        metadata={
            **dict(make_synthetic_contact_walk(num_cycles=2).metadata),
            "corruption_family": "foot_slide",
            "generation_parameters": {"hidden": True},
            "source_clip_id": "must-not-reach-system",
            "metadata_only_fields": ["source_clip_id"],
            "hard_negative": True,
            "provenance": {"hidden_corruption_history": True},
            "clean_window_reference_speed": 0.73,
            "target_speed": 0.73,
            "target_speed_tolerance": 0.05,
        }
    )
    inference = inference_available_motion(clip)
    assert "corruption_family" not in inference.metadata
    assert "generation_parameters" not in inference.metadata
    assert "source_clip_id" not in inference.metadata
    assert "hard_negative" not in inference.metadata
    assert "provenance" not in inference.metadata
    assert inference.metadata["target_speed"] == 0.73
    assert inference.metadata["target_speed_tolerance"] == 0.05


def test_task_objective_selects_block_without_clean_motion() -> None:
    clip = inference_available_motion(make_synthetic_contact_walk(num_cycles=2))
    family, evidence = select_repair_family(
        clip,
        {"primary_objective_metric": "maximum_floating_contact_cm"},
    )
    assert family == "floating_contact"
    assert evidence["source"] == "declared_task_objective"
    assert "diagnostic_recommendation" in evidence
    assert "clean" not in inspect.signature(_optimize_without_clean_reference).parameters
