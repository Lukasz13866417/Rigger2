from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from motionlab.perceptual.comparator_readiness import (
    AUTHENTICATED_COMPARATOR_PAIRS_VERSION,
    COMPARATOR_VARIANTS,
    build_aligned_comparator_predictions,
    build_comparator_readiness_report,
    prepare_comparator_readiness_report,
    prepare_post_label_comparator_evaluation,
    wilson_interval,
)
from motionlab.perceptual.motioncritic_baseline import (
    assess_motioncritic_compatibility,
    motioncritic_pairwise_agreement,
    prepare_motioncritic_baseline_report,
)


def _pairs() -> list[dict[str, Any]]:
    specifications = (
        ("train", "train", "first_better", False, "phase"),
        ("validation", "train", "second_better", False, "phase"),
        ("test", "train", "first_better", False, "rigidity"),
        ("test", "heldout", "second_better", False, "coordination"),
        ("validation", "train", "effectively_equal", False, "equivalent"),
        ("test", "train", "first_better", True, "adversarial_cma"),
    )
    rows = []
    for index, (source_split, mechanism, outcome, adversarial, family) in enumerate(specifications):
        rows.append(
            {
                "pair_id": f"human-{index}",
                "pair_outcome_canonical": outcome,
                "supervision_category": "HUMAN_LABELED",
                "source_split": source_split,
                "mechanism_partition": mechanism,
                "perturbation_mechanism": family,
                "objective_equivalence_control": outcome == "effectively_equal",
                "adversarial_origin": adversarial,
            }
        )
    rows.extend(
        [
            {
                "pair_id": "synthetic-heldout-source",
                "preference": "b_better",
                "supervision_category": "CERTAIN",
                "source_split": "test",
                "mechanism_partition": "train",
                "perturbation_mechanism": "pop",
                "adversarial_origin": False,
            },
            {
                "pair_id": "synthetic-train",
                "preference": "a_better",
                "supervision_category": "CERTAIN",
                "source_split": "train",
                "mechanism_partition": "train",
                "perturbation_mechanism": "slide",
                "adversarial_origin": False,
            },
        ]
    )
    return rows


def _canonical(value: str) -> str:
    return {
        "first_better": "a_better",
        "second_better": "b_better",
        "effectively_equal": "approximately_equal",
    }.get(value, value)


def _probabilities(label: str) -> dict[str, float]:
    return {
        value: 0.8 if value == label else 0.1
        for value in ("a_better", "b_better", "approximately_equal")
    }


def _predictions(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = {
        "absolute_q": {"human-2", "human-3", "human-5"},
        "relative_comparator_frozen": {"human-3"},
        "relative_comparator_finetuned": set(),
    }
    variants = {}
    for variant in COMPARATOR_VARIANTS:
        rows = []
        for pair in pairs:
            target = _canonical(str(pair.get("pair_outcome_canonical", pair.get("preference"))))
            if pair["pair_id"] in wrong[variant]:
                target = "b_better" if target != "b_better" else "a_better"
            forward = _probabilities(target)
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "probabilities": forward,
                    "swapped_probabilities": {
                        "a_better": forward["b_better"],
                        "b_better": forward["a_better"],
                        "approximately_equal": forward["approximately_equal"],
                    },
                }
            )
        variants[variant] = {"predictions": rows}
    return {"variants": variants}


def test_comparator_readiness_covers_every_planned_evaluation_without_training() -> None:
    pairs = _pairs()
    report = build_comparator_readiness_report(
        pairs,
        _predictions(pairs),
        evidence_origin="synthetic_fixture",
    )

    assert report["status"] == "evaluation_machinery_ready"
    assert report["training_invoked"] is False
    assert report["critic_parameters_updated"] is False
    assert report["synthetic_fixture_results_are_not_substantive_perceptual_findings"] is True
    assert tuple(report["required_variant_order"]) == COMPARATOR_VARIANTS
    assert all(report["evaluation_contract"].values())
    fine_tuned = report["variants"]["relative_comparator_finetuned"]
    assert fine_tuned["heldout_source"]["pair_count"] == 2
    assert fine_tuned["heldout_mechanism"]["pair_count"] == 1
    assert fine_tuned["direct_human_only"]["pair_count"] == 6
    assert fine_tuned["direct_human_only"]["accuracy"] == 1.0
    assert fine_tuned["equivalent_controls"]["false_preference_rate"] == 0.0
    assert fine_tuned["adversarial"]["pair_count"] == 1
    assert fine_tuned["swap_consistency"]["exact_zero_error_rate"] == 1.0
    interval = fine_tuned["heldout_source"]["accuracy_95_percent_wilson_interval"]
    assert interval is not None and interval[0] <= 1.0 <= interval[1]
    assert (
        report["variants"]["absolute_q"]["direct_human_only"]["accuracy"]
        < fine_tuned["direct_human_only"]["accuracy"]
    )


def test_native_three_ablation_outputs_align_to_authenticated_human_observations() -> None:
    pairs = []
    for row in _pairs()[:6]:
        pairs.append(
            {
                **row,
                "format_version": AUTHENTICATED_COMPARATOR_PAIRS_VERSION,
                "pair_id": "observation-" + str(row["pair_id"]),
                "observation_id": "observation-" + str(row["pair_id"]),
                "model_pair_id": "model-" + str(row["pair_id"]),
                "human_preference": _canonical(str(row["pair_outcome_canonical"])),
                "supervision_origin": "human",
                "evidence_authentication": {
                    "current_protocol_validated": True,
                    "freeze_id": "freeze:test",
                },
            }
        )
    pair_export = {
        "format_version": AUTHENTICATED_COMPARATOR_PAIRS_VERSION,
        "pairs": pairs,
    }
    absolute = {
        "format_version": "motionlab.fixed_rig_perceptual_evaluation.v1",
        "pair_scores": [
            {
                "pair_id": row["model_pair_id"],
                "score_a": 0.0,
                "score_b": 0.0,
                "probability_a_better": 0.5,
            }
            for row in pairs
        ],
    }
    ablations = {}
    for variant in ("relative_comparator_frozen", "relative_comparator_finetuned"):
        scores = []
        for row in pairs:
            forward = _probabilities(str(row["human_preference"]))
            scores.append(
                {
                    "pair_id": row["model_pair_id"],
                    "probabilities": forward,
                    "swapped_probabilities": {
                        "a_better": forward["b_better"],
                        "b_better": forward["a_better"],
                        "approximately_equal": forward["approximately_equal"],
                    },
                }
            )
        ablations[variant] = {
            "selection_records_and_targets_sha256": "a" * 64,
            "metrics": {"pair_scores": scores},
        }
    relative = {
        "format_version": "motionlab.relative_comparator_experiment.v3",
        "status": "complete",
        "variant_selection": {
            "heldout_human_labels_used_for_selection": False,
            "heldout_human_audit_policy_used_for_selection": False,
        },
        "selection_supervision": {
            "heldout_human_labels_used": False,
            "heldout_human_audit_policy_used": False,
            "records_and_targets_sha256": "a" * 64,
        },
        "ablations": ablations,
    }

    aligned = build_aligned_comparator_predictions(pair_export, absolute, relative)
    report = build_comparator_readiness_report(
        pairs,
        aligned,
        evidence_origin="human_pilot",
    )

    assert aligned["training_invoked"] is False
    assert report["status"] == "evaluation_machinery_ready"
    assert report["variants"]["absolute_q"]["swap_consistency"]["exact_zero_error_rate"] == 1.0
    for variant in ("relative_comparator_frozen", "relative_comparator_finetuned"):
        assert report["variants"][variant]["direct_human_only"]["accuracy"] == 1.0


def test_post_label_three_ablation_path_reauthenticates_and_exports_common_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specifications = (
        ("train", "train", False, False, "phase"),
        ("test", "train", False, True, "equivalent"),
        ("test", "heldout", False, False, "coordination"),
        ("adversarial", "adversarial", True, False, "critic_exploit"),
    )
    records = []
    stimuli = []
    observations = []
    for index, (source_split, mechanism_partition, adversarial, equivalent, family) in enumerate(
        specifications
    ):
        model_pair_id = f"model-pair-{index}"
        records.append(
            {
                "pair_id": model_pair_id,
                "motion_a": f"a-{index}.npz",
                "motion_b": f"b-{index}.npz",
                "source_clip_id": f"source-{index}",
                "source_split": source_split,
                "mechanism_partition": mechanism_partition,
                "perturbation_mechanism": family,
                "adversarial_origin": adversarial,
                "objective_equivalence_control": equivalent,
            }
        )
        stimulus_ids = [f"stimulus-{index}-a", f"stimulus-{index}-b"]
        stimuli.extend(
            [
                {
                    "stimulus_id": stimulus_ids[0],
                    "motion_content_hash": f"hash-{index}-a",
                },
                {
                    "stimulus_id": stimulus_ids[1],
                    "motion_content_hash": f"hash-{index}-b",
                },
            ]
        )
        observation_stimulus_ids = list(reversed(stimulus_ids)) if index == 2 else stimulus_ids
        observations.append(
            {
                "observation_id": f"observation-{index}",
                "task_type": "pair_comparison",
                "pair_outcome_canonical": ("effectively_equal" if equivalent else "first_better"),
                "pair_id": model_pair_id,
                "stimulus_ids": observation_stimulus_ids,
                "rater_id": "synthetic-rater",
                "session_id": "synthetic-session",
                "trial_id": f"trial-{index}",
                "confidence": 5,
                "reason_tags": [],
            }
        )

    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    pilot = pilot_root / "pilot_manifest.json"
    pilot.write_text(
        json.dumps(
            {
                "protocol_version": "motionlab.subjective_protocol.v3",
                "stimuli": stimuli,
                "sessions": [
                    {
                        "trials": [
                            {"trial_id": observation["trial_id"]} for observation in observations
                        ]
                    }
                ],
                "adaptive_policy": {"enabled": False},
                "adaptive_comparison_pool": [],
            }
        ),
        encoding="utf-8",
    )
    observation_path = pilot_root / "raw_observations.jsonl"
    observation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
    )
    served = pilot_root / "served_playlist.jsonl"
    served.write_text("{}\n", encoding="utf-8")
    freeze = pilot_root / "protocol_freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    pilot_report = tmp_path / "analysis" / "pilot_report.json"
    pilot_report.parent.mkdir()
    pilot_report.write_text(
        json.dumps(
            {
                "format_version": "motionlab.subjective_analysis.v3",
                "status": "pilot_complete_ready_for_inspection",
                "pilot_completion": {"complete": True},
                "observation_validation": {
                    "protocol_freeze_validated": True,
                    "served_trial_linkage_validated": True,
                    "freeze_id": "freeze:synthetic",
                    "validated_observation_count": len(observations),
                },
                "critic_retraining_started": False,
            }
        ),
        encoding="utf-8",
    )
    authentication_calls = []
    monkeypatch.setattr(
        "motionlab.perceptual.comparator_readiness.resolve_frozen_evidence_paths",
        lambda *args, **kwargs: (
            served,
            {"protocol_id": "protocol:synthetic", "freeze_id": "freeze:synthetic"},
        ),
    )
    monkeypatch.setattr(
        "motionlab.perceptual.comparator_readiness.validate_current_protocol_observations",
        lambda *args: authentication_calls.append(args),
    )
    monkeypatch.setattr(
        "motionlab.perceptual.comparator_readiness.load_perceptual_pairs",
        lambda _: records,
    )
    monkeypatch.setattr(
        "motionlab.perceptual.comparator_readiness._pair_motion_hashes",
        lambda _root, row: (
            f"hash-{row['pair_id'].rsplit('-', 1)[1]}-a",
            f"hash-{row['pair_id'].rsplit('-', 1)[1]}-b",
        ),
    )

    absolute_path = tmp_path / "absolute.json"
    absolute_path.write_text(
        json.dumps(
            {
                "format_version": "motionlab.fixed_rig_perceptual_evaluation.v1",
                "pair_scores": [
                    {
                        "pair_id": row["pair_id"],
                        "score_a": 0.0,
                        "score_b": 0.0,
                        "probability_a_better": 0.5,
                    }
                    for row in records
                ],
            }
        ),
        encoding="utf-8",
    )
    relative_rows = []
    for index, (row, observation) in enumerate(zip(records, observations, strict=True)):
        target = (
            "b_better" if index == 2 else _canonical(str(observation["pair_outcome_canonical"]))
        )
        forward = _probabilities(target)
        relative_rows.append(
            {
                "pair_id": row["pair_id"],
                "probabilities": forward,
                "swapped_probabilities": {
                    "a_better": forward["b_better"],
                    "b_better": forward["a_better"],
                    "approximately_equal": forward["approximately_equal"],
                },
            }
        )
    relative_path = tmp_path / "relative.json"
    relative_path.write_text(
        json.dumps(
            {
                "format_version": "motionlab.relative_comparator_experiment.v3",
                "status": "complete",
                "variant_selection": {
                    "heldout_human_labels_used_for_selection": False,
                    "heldout_human_audit_policy_used_for_selection": False,
                },
                "selection_supervision": {
                    "heldout_human_labels_used": False,
                    "heldout_human_audit_policy_used": False,
                    "records_and_targets_sha256": "a" * 64,
                },
                "ablations": {
                    variant: {
                        "selection_records_and_targets_sha256": "a" * 64,
                        "metrics": {"pair_scores": relative_rows},
                    }
                    for variant in (
                        "relative_comparator_frozen",
                        "relative_comparator_finetuned",
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "post-label-evaluation"
    report = prepare_post_label_comparator_evaluation(
        tmp_path / "pair-dataset",
        pilot,
        observation_path,
        freeze,
        pilot_report,
        absolute_path,
        relative_path,
        output,
        served_playlist_path=served,
    )

    assert len(authentication_calls) == 1
    assert report["status"] == "evaluation_machinery_ready"
    assert report["input_coverage_counts"] == {
        "heldout_source": 1,
        "heldout_mechanism": 1,
        "direct_human_only": 4,
        "equivalent_controls": 1,
        "adversarial_pairs": 1,
    }
    assert (output / "authenticated_direct_human_pairs.json").is_file()
    exported_pairs = json.loads((output / "authenticated_direct_human_pairs.json").read_text())[
        "pairs"
    ]
    assert exported_pairs[2]["model_orientation"] == "observation_order_matches_model_b_a"
    assert exported_pairs[2]["human_preference"] == "b_better"
    assert (output / "aligned_ablation_predictions.json").is_file()
    assert (
        json.loads((output / "comparator_evaluation.json").read_text())["training_invoked"] is False
    )


def test_comparator_readiness_file_api_is_safe_and_reports_incomplete_inputs(
    tmp_path: Path,
) -> None:
    pairs = _pairs()
    predictions = _predictions(pairs)
    predictions["variants"]["relative_comparator_frozen"]["predictions"].pop()
    pairs_path = tmp_path / "pairs.jsonl"
    predictions_path = tmp_path / "predictions.json"
    output_path = tmp_path / "readiness.json"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    before = pairs_path.read_bytes()

    report = prepare_comparator_readiness_report(
        pairs_path,
        predictions_path,
        output_path,
        evidence_origin="synthetic_fixture",
    )

    assert report["status"] == "evaluation_inputs_incomplete"
    assert report["variants"]["relative_comparator_frozen"]["missing_pair_ids"] == [
        "synthetic-train"
    ]
    assert pairs_path.read_bytes() == before
    with pytest.raises(ValueError, match="must not overwrite"):
        prepare_comparator_readiness_report(
            pairs_path,
            predictions_path,
            pairs_path,
            evidence_origin="synthetic_fixture",
        )
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError, match="0 <= successes"):
        wilson_interval(2, 1)


def test_comparator_readiness_treats_only_declared_objective_equivalence_as_controls() -> None:
    pairs = _pairs()
    pairs[4]["objective_equivalence_control"] = False
    pairs[0]["objective_equivalence_control"] = True

    report = build_comparator_readiness_report(
        pairs,
        _predictions(pairs),
        evidence_origin="synthetic_fixture",
    )

    evaluated = report["variants"]["relative_comparator_finetuned"]
    assert evaluated["equivalent_controls"]["pair_count"] == 1
    controls = [row["pair_id"] for row in evaluated["pair_scores"] if row["equivalent_control"]]
    assert controls == ["human-0"]
    tied = next(row for row in evaluated["pair_scores"] if row["pair_id"] == "human-4")
    assert tied["reference_preference"] == "approximately_equal"
    assert tied["equivalent_control"] is False


def test_comparator_readiness_requires_all_evaluation_subsets() -> None:
    pairs = [_pairs()[0]]
    report = build_comparator_readiness_report(
        pairs,
        _predictions(pairs),
        evidence_origin="synthetic_fixture",
    )

    assert report["prediction_alignment_complete"] is True
    assert report["status"] == "evaluation_inputs_incomplete"
    assert report["input_coverage_counts"] == {
        "heldout_source": 0,
        "heldout_mechanism": 0,
        "direct_human_only": 1,
        "equivalent_controls": 0,
        "adversarial_pairs": 0,
    }
    assert report["supported_capabilities"]["heldout_source"] is True
    assert report["evaluation_contract"]["heldout_source"] is False
    assert report["evaluation_contract"]["heldout_mechanism"] is False
    assert report["evaluation_contract"]["equivalent_controls"] is False
    assert report["evaluation_contract"]["adversarial_pairs"] is False


def test_comparator_readiness_rejects_stale_unexpected_prediction_ids() -> None:
    pairs = _pairs()
    predictions = _predictions(pairs)
    stale = {
        "pair_id": "stale-pair",
        "probabilities": _probabilities("a_better"),
        "swapped_probabilities": _probabilities("b_better"),
    }
    for variant in COMPARATOR_VARIANTS:
        predictions["variants"][variant]["predictions"].append(stale)

    report = build_comparator_readiness_report(
        pairs,
        predictions,
        evidence_origin="synthetic_fixture",
    )

    assert report["status"] == "evaluation_inputs_incomplete"
    assert report["prediction_alignment_complete"] is False
    for variant in COMPARATOR_VARIANTS:
        assert report["variants"][variant]["unexpected_prediction_pair_ids"] == ["stale-pair"]


def test_human_pilot_origin_requires_direct_human_provenance_without_synthetic_fallback() -> None:
    mixed_pairs = _pairs()
    with pytest.raises(ValueError, match="without direct-human provenance"):
        build_comparator_readiness_report(
            mixed_pairs,
            _predictions(mixed_pairs),
            evidence_origin="human_pilot",
        )

    disguised_synthetic = {
        "pair_id": "not-actually-human",
        "format_version": AUTHENTICATED_COMPARATOR_PAIRS_VERSION,
        "observation_id": "observation:sha256:" + "a" * 64,
        "pair_outcome_canonical": "not_sure",
        "preference": "a_better",
        "model_pair_id": "model-not-actually-human",
        "supervision_category": "HUMAN_LABELED",
        "supervision_origin": "human",
        "evidence_authentication": {
            "current_protocol_validated": True,
            "freeze_id": "freeze:test",
        },
        "source_split": "test",
        "mechanism_partition": "heldout",
        "objective_equivalence_control": True,
        "adversarial_origin": True,
    }
    with pytest.raises(ValueError, match="must not fall back to synthetic preferences"):
        build_comparator_readiness_report(
            [disguised_synthetic],
            _predictions([disguised_synthetic]),
            evidence_origin="human_pilot",
        )

    human_pairs = []
    for row in mixed_pairs[:6]:
        human_pairs.append(
            {
                **row,
                "format_version": AUTHENTICATED_COMPARATOR_PAIRS_VERSION,
                "model_pair_id": "model-" + str(row["pair_id"]),
                "supervision_origin": "human",
                "evidence_authentication": {
                    "current_protocol_validated": True,
                    "freeze_id": "freeze:test",
                },
            }
        )
    report = build_comparator_readiness_report(
        human_pairs,
        _predictions(human_pairs),
        evidence_origin="human_pilot",
    )
    assert report["status"] == "evaluation_machinery_ready"
    for variant in COMPARATOR_VARIANTS:
        evaluated = report["variants"][variant]
        assert evaluated["overall"]["pair_count"] == 6
        assert evaluated["direct_human_only"]["pair_count"] == 6


def test_motioncritic_is_deferred_for_current_rig_and_never_adapts_labels(
    tmp_path: Path,
) -> None:
    current_contract = {
        "num_joints": 23,
        "skeleton_convention": "MotionLabHuman",
        "rotation_format": "local_quaternion",
        "root_representation": "XYZ",
        "num_frames": 121,
        "verified_smpl_joint_mapping": False,
        "smpl_assets_available": False,
        "validated_resampling_to_60_frames": False,
    }
    compatibility = assess_motioncritic_compatibility(current_contract)
    assert compatibility["status"] == "deferred_incompatible_representation"
    assert any(value.startswith("joint_count_mismatch") for value in compatibility["blockers"])
    assert "verified_smpl_joint_mapping_missing" in compatibility["blockers"]
    assert "required_smpl_assets_unavailable" in compatibility["blockers"]
    assert compatibility["mapping_attempted"] is False
    assert compatibility["checkpoint_loaded"] is False
    agreement = motioncritic_pairwise_agreement([], [], compatibility=compatibility)
    assert agreement["status"] == "not_run_incompatible_representation"
    assert agreement["labels_adapted_or_used_for_fitting"] is False

    contract_path = tmp_path / "rig.json"
    output_path = tmp_path / "motioncritic_assessment.json"
    contract_path.write_text(json.dumps(current_contract), encoding="utf-8")
    report = prepare_motioncritic_baseline_report(contract_path, output_path)
    assert report == json.loads(output_path.read_text())
    assert report["status"] == "deferred_incompatible_representation"


def test_motioncritic_agreement_path_uses_direct_human_pairs_only() -> None:
    compatible = assess_motioncritic_compatibility(
        {
            "joint_count": 24,
            "skeleton_model": "SMPL",
            "rotation_representation": "local_axis_angle",
            "root_representation": "XYZ",
            "sequence_frames": 60,
            "verified_smpl_joint_mapping": True,
            "smpl_assets_available": True,
            "validated_resampling_to_60_frames": True,
        }
    )
    assert compatible["status"] == "ready_for_evaluation_only"
    human_pairs = [
        {
            "pair_id": "human-a",
            "pair_outcome_canonical": "first_better",
            "family": "phase",
        },
        {
            "pair_id": "human-b",
            "pair_outcome_canonical": "second_better",
            "family": "rigidity",
        },
        {
            "pair_id": "synthetic-not-human",
            "preference": "a_better",
            "supervision_category": "CERTAIN",
            "family": "synthetic",
        },
    ]
    predictions = [
        {"pair_id": "human-a", "prediction": "a_better"},
        {"pair_id": "human-b", "prediction": "a_better"},
        {"pair_id": "synthetic-not-human", "prediction": "a_better"},
    ]
    report = motioncritic_pairwise_agreement(
        human_pairs,
        predictions,
        compatibility=compatible,
    )
    metric = report["pairwise_agreement_with_human_judgments"]
    assert report["status"] == "complete"
    assert metric["pair_count"] == 2
    assert metric["agreement"] == pytest.approx(0.5)
    assert metric["agreement_95_percent_wilson_interval"] is not None
    assert report["labels_adapted_or_used_for_fitting"] is False
    assert report["checkpoint_or_model_updated"] is False
