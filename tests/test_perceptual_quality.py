from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from motionlab.critic.model import FixedRigFlatTCN, FixedRigTCNConfig
from motionlab.dataset.io import file_sha256
from motionlab.optimization.acceptance import (
    assess_production_acceptance,
    declare_reference_speed_task,
)
from motionlab.optimization.portfolio import (
    declared_repair_methods,
    select_repair_from_portfolio,
)
from motionlab.perceptual.dataset import objective_equivalence_control
from motionlab.perceptual.human_audit import analyze_perceptual_human_audit
from motionlab.perceptual.labeling import apply_human_labels, record_human_label
from motionlab.perceptual.model import FixedRigPerceptualResidual
from motionlab.perceptual.optimization import run_gated_perceptual_cma
from motionlab.perceptual.perturbations import PERTURBATION_SPECS
from motionlab.perceptual.relative_experiment import (
    RelativeExperimentConfig,
    run_relative_comparator_experiment,
    select_relative_comparator_variant,
)
from motionlab.perceptual.relative_model import (
    FixedRigRelativeComparator,
    RelativeComparatorConfig,
)
from motionlab.perceptual.relative_optimization import run_gated_relative_perceptual_cma
from motionlab.testing.synthetic import make_synthetic_contact_walk


def test_acceptance_records_raw_metrics_and_satisfies_instead_of_minimizing() -> None:
    raw = make_synthetic_contact_walk(num_cycles=2)
    raw = raw.with_updates(metadata={**dict(raw.metadata), "loop_kind": None})
    reference, candidate = declare_reference_speed_task(raw, raw)
    acceptance = assess_production_acceptance(reference, candidate)
    assert acceptance.all_required_satisfied
    assert acceptance.satisficing_penalty == 0.0
    assert acceptance.constraints["foot_slide"]["raw_value"] is not None
    assert acceptance.constraints["foot_slide"]["satisfied"]


def test_portfolio_rejects_high_scoring_infeasible_candidate() -> None:
    raw = make_synthetic_contact_walk(num_cycles=2)
    raw = raw.with_updates(metadata={**dict(raw.metadata), "loop_kind": None})
    reference, valid = declare_reference_speed_task(raw, raw)
    root = valid.root_translation_m.copy()
    root[:, 2] *= 2.0
    invalid = valid.with_updates(root_translation_m=root)
    decision = select_repair_from_portfolio(
        reference,
        {"specialized": valid, "generic_cma": invalid},
        perceptual_scorer=lambda motion: 100.0 if motion is invalid else 0.0,
    )
    assert decision.selected.method == "specialized"
    assert not decision.ranked[-1].acceptance.all_required_satisfied
    assert [method.kind for method in declared_repair_methods("foot_slide")] == [
        "specialized",
        "semantic_cma",
        "generic_cma",
    ]


def test_subjective_perturbations_are_not_automatically_ordered() -> None:
    unordered = [spec for spec in PERTURBATION_SPECS if spec.supervision_category == "UNORDERED"]
    certain = [spec for spec in PERTURBATION_SPECS if spec.supervision_category == "CERTAIN"]
    assert unordered and all(spec.preference is None for spec in unordered)
    assert certain and all(spec.preference is not None and spec.label_basis for spec in certain)


def test_objective_equivalence_is_not_inferred_from_a_human_tie() -> None:
    human_tie = {
        "preference": "approximately_equal",
        "supervision_category": "HUMAN_LABELED",
        "perturbation_mechanism": "subjective_pair",
        "label_basis": "direct human judgment",
    }
    exact_legacy_control = {
        "preference": "approximately_equal",
        "supervision_category": "CERTAIN",
        "perturbation_mechanism": "equivalent_origin_shift",
        "label_basis": "exact world-origin equivalence",
    }
    assert objective_equivalence_control(human_tie)[0] is False
    assert objective_equivalence_control(exact_legacy_control) == (
        True,
        "legacy_exact_world_origin_construction",
    )
    assert (
        objective_equivalence_control(
            {**exact_legacy_control, "objective_equivalence_control": False}
        )[0]
        is False
    )


def test_perceptual_head_keeps_groupnorm_backbone_frozen() -> None:
    backbone = FixedRigFlatTCN(FixedRigTCNConfig(num_joints=3, hidden_channels=16))
    model = FixedRigPerceptualResidual(backbone, representation_channels=8)
    features = torch.zeros((2, 20, backbone.config.input_channels))
    output = model(features)
    assert output.score.shape == (2,)
    assert set(output.auxiliary_scores) == {
        "naturalness",
        "coordination",
        "rigidity_smoothing",
    }
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_human_label_record_is_separate_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    record = record_human_label(
        path,
        pair_id="perceptual-pair:sha256:test",
        preference="approximately_equal",
        reason_tags=["coordination", "coordination"],
        confidence=4,
    )
    assert record["supervision_category"] == "HUMAN_LABELED"
    stored = json.loads(path.read_text())
    assert stored["reason_tags"] == ["coordination"]
    assert stored["confidence"] == 4
    merged = apply_human_labels(
        [
            {
                "pair_id": "perceptual-pair:sha256:test",
                "supervision_category": "UNORDERED",
                "preference": None,
                "reason_tags": [],
            }
        ],
        path,
    )
    assert merged[0]["supervision_category"] == "HUMAN_LABELED"
    assert merged[0]["preference"] == "approximately_equal"
    assert merged[0]["generated_supervision_category"] == "UNORDERED"
    assert merged[0]["human_confidence"] == 4


def test_relative_comparator_is_exactly_swap_consistent() -> None:
    backbone = FixedRigFlatTCN(FixedRigTCNConfig(num_joints=3, hidden_channels=16))
    model = FixedRigRelativeComparator(
        backbone,
        RelativeComparatorConfig(comparison_channels=8),
    )
    a = torch.randn((2, 20, backbone.config.input_channels))
    b = torch.randn((2, 20, backbone.config.input_channels))
    forward = model(a, b).probabilities
    swapped = model(b, a).probabilities
    assert torch.allclose(forward, swapped[:, [1, 0, 2]], atol=1.0e-7)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_relative_variant_selection_is_invariant_to_heldout_evidence() -> None:
    variants = {
        "relative_comparator_frozen": {
            "best_validation_loss": 0.31,
            "metrics": {
                "heldout_source": {
                    "accuracy": 0.0,
                    "pair_scores": [
                        {"preference": "a_better", "prediction": "b_better"},
                    ],
                }
            },
        },
        "relative_comparator_finetuned": {
            "best_validation_loss": 0.44,
            "metrics": {
                "heldout_source": {
                    "accuracy": 1.0,
                    "pair_scores": [
                        {"preference": "a_better", "prediction": "a_better"},
                    ],
                }
            },
        },
    }
    selected_before, evidence_before = select_relative_comparator_variant(variants)

    changed_heldout = copy.deepcopy(variants)
    changed_heldout["relative_comparator_frozen"]["metrics"]["heldout_source"] = {
        "accuracy": 1.0,
        "pair_scores": [{"preference": "b_better", "prediction": "b_better"}],
    }
    changed_heldout["relative_comparator_finetuned"]["metrics"]["heldout_source"] = {
        "accuracy": 0.0,
        "pair_scores": [{"preference": "b_better", "prediction": "a_better"}],
    }
    selected_after, evidence_after = select_relative_comparator_variant(changed_heldout)

    assert selected_before == selected_after == "relative_comparator_frozen"
    assert evidence_before == evidence_after
    assert evidence_after["evidence_subset"] == "validation"
    assert evidence_after["heldout_source_used_for_selection"] is False


def test_relative_experiment_selection_records_ignore_heldout_labels_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_records = [
        {
            "pair_id": "train-pair",
            "motion_a": "train-a.npz",
            "motion_b": "train-b.npz",
            "source_clip_id": "train-source",
            "source_split": "train",
            "mechanism_partition": "train",
            "perturbation_mechanism": "shared-family",
            "supervision_category": "CERTAIN",
            "preference": "a_better",
            "adversarial_origin": False,
            "objective_equivalence_control": False,
        },
        {
            "pair_id": "validation-pair",
            "motion_a": "validation-a.npz",
            "motion_b": "validation-b.npz",
            "source_clip_id": "validation-source",
            "source_split": "validation",
            "mechanism_partition": "train",
            "perturbation_mechanism": "shared-family",
            "supervision_category": "CERTAIN",
            "preference": "b_better",
            "adversarial_origin": False,
            "objective_equivalence_control": False,
        },
        {
            "pair_id": "heldout-pair",
            "motion_a": "heldout-a.npz",
            "motion_b": "heldout-b.npz",
            "source_clip_id": "heldout-source",
            "source_split": "test",
            "mechanism_partition": "train",
            "perturbation_mechanism": "shared-family",
            "supervision_category": "UNORDERED",
            "preference": None,
            "adversarial_origin": False,
            "objective_equivalence_control": False,
        },
    ]
    monkeypatch.setattr(
        "motionlab.perceptual.relative_experiment.load_perceptual_pairs",
        lambda _: copy.deepcopy(base_records),
    )
    monkeypatch.setattr(
        "motionlab.perceptual.relative_experiment._load_features",
        lambda *args, **kwargs: {},
    )
    captured: list[list[tuple[str, str | None, str]]] = []

    def fake_train(
        variant: str,
        records: list[dict[str, object]],
        *args: object,
    ) -> tuple[dict[str, object], Path]:
        captured.append(
            [
                (
                    str(row["pair_id"]),
                    None if row["preference"] is None else str(row["preference"]),
                    str(row["supervision_category"]),
                )
                for row in records
            ]
        )
        checkpoint = tmp_path / f"{variant}-{len(captured)}.pt"
        return {
            "variant": variant,
            "best_validation_loss": (0.2 if variant == "relative_comparator_frozen" else 0.3),
            "metrics": None,
        }, checkpoint

    monkeypatch.setattr("motionlab.perceptual.relative_experiment._train_variant", fake_train)
    monkeypatch.setattr(
        "motionlab.perceptual.relative_experiment.load_relative_comparator",
        lambda _: (object(), {}),
    )
    monkeypatch.setattr(
        "motionlab.perceptual.relative_experiment._evaluate",
        lambda *args: {
            "heldout_source": {
                "pair_count": 1,
                "accuracy": 1.0,
                "accuracy_95_percent_wilson_interval": [0.2, 1.0],
            },
            "heldout_mechanism": {
                "pair_count": 0,
                "accuracy": None,
                "accuracy_95_percent_wilson_interval": None,
            },
            "equivalent_controls": {"false_preference_rate": 0.0},
            "swap_consistency": {"maximum_probability_error": 0.0},
            "pair_scores": [],
        },
    )
    absolute = tmp_path / "absolute.json"
    absolute.write_text(
        json.dumps({"subsets": {"heldout_source": {"pairwise_accuracy": 0.5}}}),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.jsonl"
    audit_path = tmp_path / "audit.json"

    reports = []
    for index, (preference, automatic_supervision) in enumerate(
        (("a_better", "CERTAIN"), ("b_better", "HUMAN_REQUIRED"))
    ):
        labels.unlink(missing_ok=True)
        record_human_label(
            labels,
            pair_id="heldout-pair",
            preference=preference,
            reason_tags=[],
            confidence=5,
        )
        audit_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "human_labeled_required_pair_count": 1,
                    "human_labels_sha256": file_sha256(labels),
                    "supervision_policy": {
                        "shared-family": {
                            "automatic_supervision": automatic_supervision,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        result = run_relative_comparator_experiment(
            tmp_path / "pairs",
            tmp_path / "source",
            tmp_path / "backbone.pt",
            labels,
            audit_path,
            absolute,
            tmp_path / f"run-{index}",
            config=RelativeExperimentConfig(minimum_human_audit_labels=1),
        )
        reports.append(json.loads(result.report_path.read_text(encoding="utf-8")))

    first_run_records = captured[:2]
    second_run_records = captured[2:]
    assert first_run_records == second_run_records
    assert all(
        rows
        == [
            ("train-pair", "a_better", "CERTAIN"),
            ("validation-pair", "b_better", "CERTAIN"),
        ]
        for rows in captured
    )
    assert reports[0]["selection_supervision"] == reports[1]["selection_supervision"]
    assert reports[0]["selected_variant"] == reports[1]["selected_variant"]
    assert reports[0]["variant_selection"]["heldout_human_labels_used_for_selection"] is False
    assert reports[0]["variant_selection"]["heldout_human_audit_policy_used_for_selection"] is False


def test_relative_comparator_finetunes_only_temporal_backbone() -> None:
    backbone = FixedRigFlatTCN(FixedRigTCNConfig(num_joints=3, hidden_channels=16))
    model = FixedRigRelativeComparator(
        backbone,
        RelativeComparatorConfig(comparison_channels=8, fine_tune_backbone=True),
    )
    assert any(parameter.requires_grad for parameter in model.backbone.blocks.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.backbone.clip_logits_head.parameters()
    )


def test_human_audit_downgrades_a_low_agreement_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "pair_id": f"pair-{index}",
            "preference": "a_better",
            "supervision_category": "CERTAIN",
            "perturbation_mechanism": "excessive_smoothing",
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        "motionlab.perceptual.human_audit.load_perceptual_pairs",
        lambda _: records,
    )
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "".join(
            json.dumps({"pair_id": record["pair_id"], "perturbation_family": "excessive_smoothing"})
            + "\n"
            for record in records
        )
    )
    labels = tmp_path / "labels.jsonl"
    record_human_label(
        labels,
        pair_id="pair-0",
        preference="a_better",
        reason_tags=["rigidity/smoothing"],
        confidence=5,
    )
    record_human_label(
        labels,
        pair_id="pair-1",
        preference="b_better",
        reason_tags=["rigidity/smoothing"],
        confidence=5,
    )
    report = analyze_perceptual_human_audit(
        tmp_path / "pairs",
        queue,
        labels,
        tmp_path / "agreement.json",
    )
    assert report["family_agreement"]["excessive_smoothing"]["agreement"] == 0.5
    assert (
        report["supervision_policy"]["excessive_smoothing"]["automatic_supervision"]
        == "HUMAN_REQUIRED"
    )


def test_perceptual_cma_is_not_invoked_when_ranking_gate_fails(tmp_path: Path) -> None:
    root = tmp_path
    evaluation = root / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "optimization_gate": {
                    "passed": False,
                    "reasons": ["heldout_source:near_chance_ranking"],
                }
            }
        )
    )
    report = run_gated_perceptual_cma(
        root / "missing_pairs",
        root / "missing_source",
        root / "missing_checkpoint.pt",
        evaluation,
        root / "output",
    )
    assert report["cma_es_invoked"] is False
    assert report["perceptual_cma_exploit_rate"] is None


def test_relative_experiment_and_cma_wait_for_human_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pending_human_labels",
                "human_labeled_required_pair_count": 0,
            }
        )
    )
    absolute = tmp_path / "absolute.json"
    absolute.write_text(json.dumps({"subsets": {"heldout_source": {"pairwise_accuracy": 0.4}}}))
    result = run_relative_comparator_experiment(
        tmp_path / "missing_pairs",
        tmp_path / "missing_source",
        tmp_path / "missing_backbone.pt",
        tmp_path / "missing_labels.jsonl",
        audit,
        absolute,
        tmp_path / "relative",
    )
    assert not result.optimization_gate_passed
    report = json.loads(result.report_path.read_text())
    assert report["status"] == "not_run_human_audit_required"

    cma = run_gated_relative_perceptual_cma(
        tmp_path / "missing_pairs",
        tmp_path / "missing_source",
        tmp_path / "missing_comparator.pt",
        result.report_path,
        tmp_path / "relative_cma",
    )
    assert cma["cma_es_invoked"] is False
    assert cma["global_absolute_quality_assumed"] is False
