from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from motionlab.cli import app
from motionlab.core.provenance import assign_source_lineage
from motionlab.corruptions import (
    CORRUPTION_CATALOG,
    CORRUPTION_DEFECT_NAMES,
    compose_corruptions,
    corrupt_floating_contact_root_offset,
    corrupt_foot_clearance_ik_lowering,
    corrupt_foot_slide_hip_rotation_drift,
    corrupt_joint_jitter_band_limited,
    corrupt_joint_jitter_white,
    corrupt_joint_limit_excess,
    corrupt_joint_pop_smooth_pulse,
    corrupt_leg_stride_amplitude,
    corrupt_limb_phase_shift,
    corrupt_loop_seam_root_velocity,
    corrupt_pelvis_vertical_curve,
    corrupt_speed_root_progression,
    corrupt_style_region_blend,
    corrupt_time_warp,
)
from motionlab.dataset import (
    AUTOMATIC_MECHANISMS,
    NEURAL_INPUT_FIELDS,
    SEVERITY_LABELS,
    CollateralDefectError,
    DatasetGenerationConfig,
    corruption_training_sample,
    generate_corruption_dataset,
    load_normalization_statistics,
    load_training_sample,
    single_defect_policy,
    validate_corruption_dataset,
    validate_single_defect,
)
from motionlab.io.npz import save_motion_npz
from motionlab.motion.clip import MotionClip
from motionlab.motion.markers import JointLimitSpec
from motionlab.motion.skeleton import Skeleton
from motionlab.rigging.semantic_roles import FunctionalPart
from motionlab.testing.synthetic import make_synthetic_contact_walk

runner = CliRunner()


def _lineaged_walk(*, cycles: int = 3, split: str = "train") -> MotionClip:
    return assign_source_lineage(
        make_synthetic_contact_walk(num_cycles=cycles),
        source_dataset="synthetic",
        source_clip_id=f"walk-{split}",
        source_take_id=f"take-{split}",
        split=split,  # type: ignore[arg-type]
        clean_confidence=0.95,
    )


def _with_knee_limit(clip: MotionClip) -> MotionClip:
    skeleton = clip.skeleton
    knee = skeleton.roles["left_knee"]
    limited = Skeleton(
        joint_names=skeleton.joint_names,
        parents=skeleton.parents,
        rest_offsets_m=skeleton.rest_offsets_m,
        rest_local_quat_wxyz=skeleton.rest_local_quat_wxyz,
        roles=skeleton.roles,
        markers=skeleton.markers,
        joint_limits={
            knee: JointLimitSpec(
                representation="euler_xyz",
                x_deg=(-120.0, 120.0),
                y_deg=(-20.0, 20.0),
                z_deg=(-20.0, 20.0),
                source="test",
            )
        },
        metadata=skeleton.metadata,
    )
    return MotionClip(
        limited,
        clip.local_quat_wxyz,
        clip.root_translation_m,
        clip.fps,
        clip.metadata,
    )


def test_seeded_jitter_is_deterministic_localized_and_has_heldout_variant() -> None:
    clip = make_synthetic_contact_walk(num_cycles=3)
    arguments = {
        "joints": ["chest", "left_shoulder"],
        "start_frame": 20,
        "stop_frame": 70,
        "amplitude_rad": 0.025,
        "seed": 91,
    }
    first = corrupt_joint_jitter_white(clip, **arguments)
    second = corrupt_joint_jitter_white(clip, **arguments)
    different = corrupt_joint_jitter_white(clip, **{**arguments, "seed": 92})
    heldout = corrupt_joint_jitter_band_limited(clip, **arguments)
    np.testing.assert_array_equal(
        first.corrupted_motion.local_quat_wxyz,
        second.corrupted_motion.local_quat_wxyz,
    )
    assert not np.array_equal(
        first.corrupted_motion.local_quat_wxyz,
        different.corrupted_motion.local_quat_wxyz,
    )
    np.testing.assert_array_equal(
        first.corrupted_motion.local_quat_wxyz[:20], clip.local_quat_wxyz[:20]
    )
    np.testing.assert_array_equal(
        first.corrupted_motion.local_quat_wxyz[70:], clip.local_quat_wxyz[70:]
    )
    np.testing.assert_allclose(
        np.linalg.norm(first.corrupted_motion.local_quat_wxyz, axis=-1), 1.0, atol=1.0e-6
    )
    assert first.postcondition.hard_negative
    assert heldout.postcondition.hard_negative
    assert first.catalog_partition == "train"
    assert heldout.catalog_partition == "heldout_eval"


def test_remaining_temporal_catalog_distinguishes_hard_and_soft_preferences() -> None:
    clip = make_synthetic_contact_walk(num_cycles=3)
    hard_results = (
        corrupt_limb_phase_shift(
            clip,
            start_frame=10,
            stop_frame=130,
            shift_frames=10,
        ),
        corrupt_time_warp(
            clip,
            start_frame=10,
            stop_frame=150,
            strength=0.45,
        ),
        corrupt_speed_root_progression(clip, speed_scale=1.2),
        corrupt_leg_stride_amplitude(clip, amplitude_scale=0.75),
        corrupt_loop_seam_root_velocity(
            clip,
            seam_window_frames=12,
            velocity_delta_mps=0.4,
        ),
    )
    assert all(result.postcondition.hard_negative for result in hard_results)
    assert all(result.measured_severity > 0.0 for result in hard_results)
    soft = corrupt_pelvis_vertical_curve(clip, amplitude_scale=1.8)
    assert soft.measured_severity > 0.0
    assert not soft.postcondition.hard_negative
    assert soft.preference_confidence == 0.0

    donor = corrupt_limb_phase_shift(
        clip,
        start_frame=10,
        stop_frame=130,
        shift_frames=12,
    ).corrupted_motion
    style = corrupt_style_region_blend(
        clip,
        donor,
        parts=[FunctionalPart.LEFT_ARM],
        blend_weight=0.6,
    )
    assert style.corruption_family == "style_incoherence"
    assert not style.postcondition.hard_negative


def test_contact_clearance_floating_pop_and_authored_limit_are_measured() -> None:
    clip = make_synthetic_contact_walk(num_cycles=3)
    results = (
        corrupt_floating_contact_root_offset(
            clip,
            start_frame=5,
            stop_frame=25,
            height_m=0.05,
        ),
        corrupt_foot_clearance_ik_lowering(
            clip,
            side="left",
            start_frame=25,
            stop_frame=55,
            clearance_loss_m=0.04,
        ),
        corrupt_joint_pop_smooth_pulse(
            clip,
            joint="left_knee",
            start_frame=17,
            stop_frame=24,
            angle_rad=0.3,
        ),
    )
    assert all(result.postcondition.hard_negative for result in results)
    limited = _with_knee_limit(clip)
    limit = corrupt_joint_limit_excess(
        limited,
        joint="left_knee",
        start_frame=20,
        stop_frame=31,
        excess_deg=15.0,
    )
    assert limit.postcondition.hard_negative
    assert limit.postcondition.after == pytest.approx(15.0, abs=1.0e-4)


def test_composition_remeasures_constituents_and_unions_per_defect_labels() -> None:
    clip = make_synthetic_contact_walk(num_cycles=3)

    def speed(value: MotionClip):
        return corrupt_speed_root_progression(value, speed_scale=1.15, seed=1)

    def jitter(value: MotionClip):
        return corrupt_joint_jitter_white(
            value,
            joints=["chest"],
            start_frame=20,
            stop_frame=80,
            amplitude_rad=0.02,
            seed=2,
        )

    result = compose_corruptions(clip, (speed, jitter), seed=3)
    assert result.corruption_family == "composite"
    assert result.postcondition.hard_negative
    assert len(result.generation_parameters["constituents"]) == 2
    assert result.schema_metadata["constituents_remeasured_on_final_composition"] is True
    speed_index = CORRUPTION_DEFECT_NAMES.index("speed_inconsistency")
    jitter_index = CORRUPTION_DEFECT_NAMES.index("joint_jitter")
    assert np.any(result.symptom_mask[:, :, speed_index])
    assert np.any(result.symptom_mask[:, :, jitter_index])

    def heldout(value: MotionClip):
        return corrupt_leg_stride_amplitude(value, amplitude_scale=0.75, seed=4)

    with pytest.raises(ValueError, match="held-out"):
        compose_corruptions(clip, (speed, heldout), seed=5)


def test_training_sample_has_fixed_features_and_no_privileged_or_metadata_leakage() -> None:
    clip = _lineaged_walk().slice_frames(0, 160)
    result = corrupt_speed_root_progression(clip, speed_scale=1.2, seed=11)
    sample = corruption_training_sample(result, source_window_start=0)
    assert sample.arrays["local_rot6d"].shape == (160, clip.num_joints, 6)
    assert sample.arrays["joint_position_rel"].shape == (160, clip.num_joints, 3)
    assert sample.arrays["symptom_mask"].shape[-1] == len(CORRUPTION_DEFECT_NAMES)
    assert set(NEURAL_INPUT_FIELDS).issubset(sample.arrays)
    assert set(sample.neural_inputs) == set(NEURAL_INPUT_FIELDS)
    assert "clean_local_quat" not in sample.neural_inputs
    assert "clean_local_quat" in sample.targets
    assert "contact_truth_hard" not in sample.arrays
    for forbidden in sample.metadata["metadata_only_fields"]:
        assert forbidden not in sample.arrays
    assert sample.metadata["corruption_mechanism"] == "root_progression_scale"
    assert sample.arrays["preference_valid"].item()
    assert not sample.arrays["no_target_defect"].item()
    assert not sample.arrays["approximately_equal_quality"].item()
    assert sample.arrays["quality_equivalence_valid"].item()


def test_dataset_generation_is_resumable_split_safe_and_train_normalized(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    save_motion_npz(source_path, _lineaged_walk(cycles=4))
    output = tmp_path / "dataset"
    config = DatasetGenerationConfig(
        mechanisms=(
            "speed_inconsistency/root_progression_scale",
            "speed_inconsistency/leg_pose_amplitude_scale",
        ),
        include_soft_corruptions=False,
        composition_fraction=0.0,
    )
    first = generate_corruption_dataset([source_path], output, config=config)
    summary_bytes = (output / "summary.json").read_bytes()
    second = generate_corruption_dataset([source_path], output, config=config)
    assert first["sample_count"] == 33  # 3 windows * (clean + five train + five held-out)
    assert second["resumed_existing_sample_count"] == first["sample_count"]
    assert (output / "summary.json").read_bytes() == summary_bytes
    validated = validate_corruption_dataset(output)
    assert validated["sample_count"] == first["sample_count"]

    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert not any(
        record["data_split"] == "train" and record["catalog_partition"] == "heldout_eval"
        for record in records
    )
    heldout = [record for record in records if record["catalog_partition"] == "heldout_eval"]
    assert heldout and all(record["data_split"] == "heldout_corruptor" for record in heldout)
    stats = load_normalization_statistics(output)
    training_ids = {
        record["sample_id"]
        for record in records
        if record["source_split"] == "train"
        and record["catalog_partition"] in {"clean", "train", "sham"}
    }
    assert set(stats.training_sample_ids) == training_ids

    first_sample = load_training_sample(output / records[0]["path"])
    assert first_sample.arrays["local_rot6d"].shape[0] == 160
    with np.load(output / records[0]["path"], allow_pickle=False) as archive:
        assert all(archive[name].dtype != np.dtype(object) for name in archive.files)


def test_observed_bins_shams_counterfactuals_and_consistency_targets(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "dataset"
    save_motion_npz(source, _lineaged_walk())
    summary = generate_corruption_dataset(
        [source],
        output,
        config=DatasetGenerationConfig(
            mechanisms=(
                "foot_slide/root_drift",
                "foot_slide/hip_rotation_drift",
            ),
            include_soft_corruptions=False,
            composition_fraction=0.0,
            maximum_windows_per_source=1,
        ),
    )
    assert summary["sample_count"] == 16  # clean + two five-bin mechanisms + five shams
    assert summary["sham_control_count"] == 5
    assert summary["preference_pair_count"] == 10
    assert summary["consistency_target_count"] >= 15
    diagnostics = summary["dataset_diagnostics"]["mechanisms"]
    for mechanism in (
        "foot_slide/root_drift",
        "foot_slide/hip_rotation_drift",
    ):
        assert diagnostics[mechanism]["all_severity_bins_populated"]
        assert diagnostics[mechanism]["postcondition_pass_rate"] == 1.0
        assert set(diagnostics[mechanism]["severity_bin_counts"]) == set(SEVERITY_LABELS)
    assert diagnostics["foot_slide/root_drift"]["sham_control_count"] == 5

    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len({record["counterfactual_group_id"] for record in records}) == 1
    shams = [record for record in records if record["sample_role"] == "sham_control"]
    assert len(shams) == 5
    assert all(record["no_target_defect"] for record in shams)
    assert all(record["approximately_equal_quality"] for record in shams)
    assert all(f"{record['target_side']}_stance" in record["gait_event_labels"] for record in shams)
    for record in records:
        if record["sample_role"] != "hard_corruption":
            continue
        severity_bin = record["severity_bin"]
        assert severity_bin["bin_lower_inclusive"] <= record["measured_severity"]
        assert record["measured_severity"] < severity_bin["bin_upper_exclusive"]

    consistency = [
        json.loads(line) for line in (output / "consistency.jsonl").read_text().splitlines()
    ]
    target_types = {record["target_type"] for record in consistency}
    assert {
        "per_defect_severity_equal",
        "approximately_equal_quality",
        "matched_sham_defect_contrast",
    }.issubset(target_types)
    validate_corruption_dataset(output)


def test_collateral_validation_rejects_or_labels_significant_secondary_defects() -> None:
    clip = make_synthetic_contact_walk(num_cycles=3).slice_frames(0, 160)
    speed = corrupt_speed_root_progression(clip, speed_scale=1.3)
    validated, findings = validate_single_defect(
        speed,
        single_defect_policy(
            primary_family=speed.corruption_family,
            primary_metric=speed.postcondition.metric_name,
            required_primary_delta=0.1,
        ),
    )
    assert any(
        finding.defect_family == "foot_slide" and finding.action == "label" for finding in findings
    )
    foot_slide = CORRUPTION_DEFECT_NAMES.index("foot_slide")
    assert np.any(validated.symptom_mask[:, :, foot_slide])
    assert validated.schema_metadata["single_defect_validation"]["single_defect"] is False

    pitched_hip = corrupt_foot_slide_hip_rotation_drift(
        clip,
        side="left",
        start_frame=61,
        stop_frame=90,
        angle_rad=0.05,
        axis_local=(1.0, 0.0, 0.0),
    )
    with pytest.raises(CollateralDefectError, match="ground_penetration"):
        validate_single_defect(
            pitched_hip,
            single_defect_policy(
                primary_family=pitched_hip.corruption_family,
                primary_metric=pitched_hip.postcondition.metric_name,
                required_primary_delta=0.1,
            ),
        )


def test_dataset_state_rejects_silent_configuration_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    save_motion_npz(source, _lineaged_walk())
    output = tmp_path / "dataset"
    base = DatasetGenerationConfig(
        mechanisms=("speed_inconsistency/root_progression_scale",),
        include_heldout_mechanisms=False,
        include_soft_corruptions=False,
        composition_fraction=0.0,
        maximum_windows_per_source=1,
    )
    generate_corruption_dataset([source], output, config=base)
    changed = base.model_copy(update={"stride_frames": 16})
    with pytest.raises(ValueError, match="generation state differs"):
        generate_corruption_dataset([source], output, config=changed)


def test_phase6_catalog_coverage_and_cli_dataset_round_trip(tmp_path: Path) -> None:
    automatic_families = {mechanism.key.split("/", 1)[0] for mechanism in AUTOMATIC_MECHANISMS}
    assert {
        "foot_slide",
        "ground_penetration",
        "floating_contact",
        "joint_pop",
        "joint_jitter",
        "limb_phase_mismatch",
        "cadence_inconsistency",
        "speed_inconsistency",
        "loop_seam",
        "pelvis_curve",
        "foot_clearance",
        "joint_limit",
    }.issubset(automatic_families)
    assert any(entry.family == "style_incoherence" for entry in CORRUPTION_CATALOG)

    source = tmp_path / "source.npz"
    output = tmp_path / "dataset"
    save_motion_npz(source, _lineaged_walk())
    generated = runner.invoke(
        app,
        [
            "generate-dataset",
            str(source),
            "--output",
            str(output),
            "--mechanism",
            "speed_inconsistency/root_progression_scale",
            "--no-heldout-mechanisms",
            "--no-soft-corruptions",
            "--composition-fraction",
            "0",
            "--maximum-windows-per-source",
            "1",
        ],
    )
    assert generated.exit_code == 0
    generated_payload = json.loads(generated.stdout)
    assert generated_payload["sample_count"] == 6
    assert generated_payload["consistency_target_count"] == 0
    validated = runner.invoke(app, ["validate-dataset", str(output)])
    assert validated.exit_code == 0
    assert json.loads(validated.stdout)["sample_count"] == 6
