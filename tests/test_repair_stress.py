from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import motionlab.optimization.repair_stress as repair_stress
from motionlab.corruptions.temporal import (
    corrupt_joint_jitter_white,
    corrupt_speed_root_progression,
)
from motionlab.optimization.repair_stress import (
    STRESS_FAMILIES,
    _optimize_methods_without_clean_reference,
    _prior_source_exclusions,
    _require_fresh_output,
    deterministic_stress_objective,
    run_deterministic_stress_case,
    select_specialized_first,
)
from motionlab.testing.synthetic import make_synthetic_contact_walk


def test_jitter_stress_uses_global_deterministic_metric_with_localized_block() -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    corruption = corrupt_joint_jitter_white(
        clean,
        joints=("left_elbow",),
        start_frame=30,
        stop_frame=90,
        amplitude_rad=0.08,
        seed=41,
    )
    objective = deterministic_stress_objective(corruption.corrupted, "joint_jitter")
    assert objective.metric_name == "angular_smoothness_rad_s3_p99"
    assert not objective.use_localized_angular_peak
    assert objective.joint_indices
    assert objective.time_interval is not None
    assert objective.value(corruption.corrupted) > objective.value(clean) + 100.0


def test_specialized_first_policy_falls_back_only_when_needed() -> None:
    rows = [
        {
            "method": "specialized",
            "kind": "specialized",
            "diagnostic_success": True,
            "production_success": True,
            "non_target_constraint_reasons": [],
            "objective": {"after": 2.0},
        },
        {
            "method": "cma",
            "kind": "generic_cma",
            "diagnostic_success": True,
            "production_success": True,
            "non_target_constraint_reasons": [],
            "objective": {"after": 1.0},
        },
    ]
    selected, reason = select_specialized_first(rows, production_claim_eligible=True)
    assert selected == "specialized"
    assert reason == "successful_specialized_operator"

    rows[0]["production_success"] = False
    selected, reason = select_specialized_first(rows, production_claim_eligible=True)
    assert selected == "cma"
    assert "cma_fallback" in reason

    rows[1]["portfolio_selection_eligible"] = False
    selected, reason = select_specialized_first(rows, production_claim_eligible=True)
    assert selected == "specialized"
    assert reason == "no_method_met_success_gate_best_deterministic_candidate"


def test_prior_benchmark_styles_are_excluded_globally(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "selected_cases": [
                    {"family": "foot_slide", "source_clip_id": "StyleA"},
                    {"family": "joint_pop", "source_clip_id": "StyleB"},
                ]
            }
        ),
        encoding="utf-8",
    )
    exclusions = _prior_source_exclusions(prior)
    assert set(exclusions) == set(STRESS_FAMILIES)
    assert all(values == {"StyleA", "StyleB"} for values in exclusions.values())


def test_prior_clean_artifact_cannot_enter_a_new_repair_run(tmp_path: Path) -> None:
    (tmp_path / "clean_evaluation_only.npz").write_bytes(b"privileged")
    with pytest.raises(FileExistsError, match="new or empty"):
        _require_fresh_output(tmp_path)


def test_speed_stress_case_is_label_independent_and_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    corruption = corrupt_speed_root_progression(clean, speed_scale=1.2, seed=17)
    original_optimizer = repair_stress._optimize_methods_without_clean_reference

    def guarded_optimizer(*args: object, **kwargs: object) -> object:
        assert not (tmp_path / "clean_evaluation_only.npz").exists()
        return original_optimizer(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(
        repair_stress,
        "_optimize_methods_without_clean_reference",
        guarded_optimizer,
    )
    report = run_deterministic_stress_case(
        clean,
        corruption.corrupted,
        "speed_inconsistency",
        tmp_path,
        seed=73,
        maximum_evaluations=32,
    )
    assert "clean" not in inspect.signature(_optimize_methods_without_clean_reference).parameters
    assert report["clean_reference_available_during_repair"] is False
    assert report["learned_perceptual_objective_used"] is False
    assert report["speed_semantics"]["target_speed_source"] == "measured_clean_window_speed"
    assert [row["kind"] for row in report["methods"]] == ["specialized", "generic_cma"]
    specialized = report["methods"][0]
    assert specialized["objective"]["after"] < specialized["objective"]["before"]
    assert specialized["objective"]["after"] < 1.0e-4
    assert report["portfolio"]["selected_method"] in {
        "specialized_direct_speed_rescale",
        "generic_deterministic_cma_fallback",
    }
    assert (tmp_path / "stress_case.json").exists()


def test_successful_specialized_repair_short_circuits_production_but_compares_cma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_synthetic_contact_walk(num_cycles=2)
    clean = source.with_updates(
        metadata={key: value for key, value in source.metadata.items() if key != "loop_kind"}
    )
    root = clean.root_translation_m.copy()
    root[:, 1] -= 0.03
    corrupted = clean.with_updates(root_translation_m=root)
    original_generic = repair_stress._generic_cma_candidate
    generic_calls = 0

    def guarded_generic(*args: object, **kwargs: object) -> object:
        nonlocal generic_calls
        generic_calls += 1
        assert not (tmp_path / "clean_evaluation_only.npz").exists()
        return original_generic(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(repair_stress, "_generic_cma_candidate", guarded_generic)
    report = run_deterministic_stress_case(
        clean,
        corrupted,
        "ground_penetration",
        tmp_path,
        seed=81,
        maximum_evaluations=8,
    )
    assert report["repair_routing"]["route"] == "specialized_success_short_circuit"
    assert report["repair_routing"]["generic_cma_invoked"] is True
    assert report["repair_routing"]["generic_cma_invoked_for_production"] is False
    assert report["repair_routing"]["generic_cma_evaluation_only_invoked"] is True
    assert report["repair_routing"]["selection_frozen_before_evaluation_only_comparison"]
    assert [row["kind"] for row in report["methods"]] == [
        "specialized",
        "generic_cma",
    ]
    assert report["portfolio"]["selected_method"] == "specialized_contact_height_alignment"
    assert report["portfolio"]["selection_frozen_without_clean_reference"] is True
    assert report["specialized_vs_generic_cma"]["available"] is True
    assert report["specialized_vs_generic_cma"]["production_selection_affected"] is False
    generic = report["methods"][1]
    assert generic["execution_purpose"] == "evaluation_only_head_to_head"
    assert generic["portfolio_selection_eligible"] is False
    assert generic_calls == 1
    assert not (tmp_path / "cma_fallback").exists()
    assert (tmp_path / "cma_evaluation_only").exists()
    np.testing.assert_allclose(
        report["methods"][0]["objective"]["after"],
        0.0,
        atol=1.0e-6,
    )
