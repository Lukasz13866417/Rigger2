from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from motionlab.perceptual.calibration_bank import (
    CALIBRATION_ONLY,
    CALIBRATION_SEVERITIES,
    EVALUATION_ANCHOR,
    HARD_FEASIBLE_PERCEPTUAL,
    prepare_mannequin_subjective_pilot,
    prepare_subjective_calibration_bank,
    validate_calibration_bank_manifest,
    validate_mannequin_pilot_manifest,
)
from motionlab.perceptual.subjective import MANNEQUIN_VIEWING_SETTINGS, _public_trial

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE8_DATASET = PROJECT_ROOT / "artifacts/phase8_critic_hardening/dataset"
PAIR_DATASET = PROJECT_ROOT / "artifacts/phase12_perceptual_quality/dataset"
AUDIT_QUEUE = (
    PROJECT_ROOT / "artifacts/phase13_relative_comparator/human_audit/human_audit_queue.jsonl"
)


def _require_source_artifacts() -> None:
    required = (
        PHASE8_DATASET / "manifest.jsonl",
        PAIR_DATASET / "pairs.jsonl",
        AUDIT_QUEUE,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"source research artifacts are unavailable: {missing}")


def test_calibration_bank_and_revised_pilot_are_purpose_separated(
    tmp_path: Path,
) -> None:
    _require_source_artifacts()
    bank_root = tmp_path / "bank"
    bank_summary = prepare_subjective_calibration_bank(PHASE8_DATASET, bank_root)
    bank = json.loads((bank_root / "calibration_bank.json").read_text(encoding="utf-8"))

    assert bank_summary["ladder_count"] == 13
    assert bank_summary["stimulus_count"] == 65
    assert bank_summary["measured_ladder_count"] == 6
    assert bank_summary["preset_ladder_count"] == 7
    validate_calibration_bank_manifest(bank, bank_root)
    assert all(
        ladder["ordered_severities"] == list(CALIBRATION_SEVERITIES)
        and len(ladder["stimulus_ids"]) == 5
        for ladder in bank["ladders"]
    )
    assert {
        "upper_body_phase_mismatch",
        "excessive_rigidification",
        "excessive_smoothing",
        "torso_counter_rotation_mismatch",
        "asymmetric_limb_timing",
        "reduced_pelvis_bounce",
        "loop_seam",
    }.issubset({ladder["calibration_family"] for ladder in bank["ladders"]})
    assert all(
        item["stimulus_population"] == CALIBRATION_ONLY
        and item["measurement_cohort"] == "calibration_easy"
        and item["critic_training_eligible"] is False
        and item["quality_evaluation_eligible"] is False
        for item in bank["stimuli"]
    )
    assert all(
        item["rendered_difference_from_clean"]["representable"] is True
        for item in bank["stimuli"]
        if item["calibration_severity"] != "clean"
    )
    assert all(
        item["rendered_difference_from_clean"]
        == {
            "global_rotation_max_geodesic_deg": 0.0,
            "representable": False,
            "tracked_position_max_abs_m": 0.0,
        }
        for item in bank["stimuli"]
        if item["calibration_severity"] == "clean"
    )
    for ladder in bank["ladders"]:
        rungs = {
            stimulus["stimulus_id"]: stimulus
            for stimulus in bank["stimuli"]
            if stimulus["stimulus_id"] in ladder["stimulus_ids"]
        }
        assert len({rungs[value]["rendered_geometry_hash"] for value in rungs}) == 5
        assert len({rungs[value]["visual_content_hash"] for value in rungs}) == 5

    phase14 = PROJECT_ROOT / "artifacts/phase14_subjective_pilot"
    phase14_snapshot = (
        {
            path.relative_to(phase14): path.read_bytes()
            for path in phase14.rglob("*")
            if path.is_file()
        }
        if phase14.is_dir()
        else {}
    )

    pilot_root = tmp_path / "pilot"
    summary = prepare_mannequin_subjective_pilot(
        PAIR_DATASET,
        AUDIT_QUEUE,
        PHASE8_DATASET,
        bank_root,
        pilot_root,
    )
    manifest = json.loads((pilot_root / "pilot_manifest.json").read_text(encoding="utf-8"))
    validate_mannequin_pilot_manifest(manifest, pilot_root)

    assert summary["scored_stimulus_count"] == 40
    assert summary["hard_feasible_stimulus_count"] == 32
    assert summary["hidden_anchor_count"] == 8
    assert summary["hidden_anchor_repeat_trial_count"] == 4
    assert summary["hard_feasible_repeat_trial_count"] == 4
    assert summary["calibration_tutorial_ladder_count"] == 3
    assert summary["adaptive_toggle_comparison_count"] == 12
    assert summary["session_base_trial_counts"] == [24, 24]
    assert manifest["viewing_settings"] == MANNEQUIN_VIEWING_SETTINGS
    assert Path(manifest["stimulus_directory"]).resolve() == pilot_root.resolve()
    assert Path(manifest["calibration_bank"]).is_file()

    stimuli = {item["stimulus_id"]: item for item in manifest["stimuli"]}
    scored = [stimuli[stimulus_id] for stimulus_id in manifest["scored_stimulus_ids"]]
    assert Counter(item["stimulus_population"] for item in scored) == Counter(
        {HARD_FEASIBLE_PERCEPTUAL: 32, EVALUATION_ANCHOR: 8}
    )
    assert all(
        item["critic_training_eligible"]
        is (item["stimulus_population"] == HARD_FEASIBLE_PERCEPTUAL)
        for item in manifest["stimuli"]
    )
    tutorial_ids = {
        stimulus_id
        for tutorial in manifest["calibration_tutorial"]
        for stimulus_id in tutorial["stimulus_ids"]
    }
    assert tutorial_ids.isdisjoint(manifest["scored_stimulus_ids"])
    assert all(
        stimuli[stimulus_id]["stimulus_population"] == CALIBRATION_ONLY
        for stimulus_id in tutorial_ids
    )
    scored_visual_hashes = {
        stimuli[stimulus_id]["visual_content_hash"]
        for stimulus_id in manifest["scored_stimulus_ids"]
    }
    assert scored_visual_hashes.isdisjoint(
        {stimuli[stimulus_id]["visual_content_hash"] for stimulus_id in tutorial_ids}
    )
    scored_source_windows = {
        (stimuli[value]["source_id"], stimuli[value]["source_window_start"])
        for value in manifest["scored_stimulus_ids"]
        if stimuli[value].get("source_window_start") is not None
    }
    tutorial_source_windows = {
        (stimuli[value]["source_id"], stimuli[value]["source_window_start"])
        for value in tutorial_ids
        if stimuli[value].get("source_window_start") is not None
    }
    assert scored_source_windows.isdisjoint(tutorial_source_windows)

    base_trials = [trial for session in manifest["sessions"] for trial in session["trials"]]
    naturalness_counts = Counter(
        trial["stimulus_ids"][0] for trial in base_trials if trial["task_type"] == "naturalness"
    )
    repeated_anchor_ids = set(manifest["hidden_repeated_anchor_stimulus_ids"])
    repeated_hfp_ids = set(manifest["hidden_repeated_hard_feasible_stimulus_ids"])
    assert len(repeated_anchor_ids) == 4
    assert len(repeated_hfp_ids) == 4
    assert all(naturalness_counts[value] == 2 for value in repeated_anchor_ids)
    assert all(naturalness_counts[value] == 2 for value in repeated_hfp_ids)
    assert all(
        naturalness_counts[value] == 1
        for value in manifest["hidden_anchor_stimulus_ids"]
        if value not in repeated_anchor_ids
    )
    repeat_positions: dict[str, list[int]] = {}
    for index, trial in enumerate(base_trials):
        for group in trial["hidden_repeat_group_ids"]:
            repeat_positions.setdefault(group, []).append(index)
    repeat_gaps = [
        positions[1] - positions[0]
        for positions in repeat_positions.values()
        if len(positions) == 2
    ]
    assert len(repeat_gaps) == 8
    assert min(repeat_gaps) >= 12
    assert manifest["repeat_validation"]["minimum_global_trial_index_distance"] == min(repeat_gaps)
    assert len(manifest["adaptive_comparison_pool"]) == 12
    assert all(
        trial["comparison_mode"] == "same_viewport_toggle"
        and trial["stimulus_populations"] == [HARD_FEASIBLE_PERCEPTUAL, HARD_FEASIBLE_PERCEPTUAL]
        and trial["critic_training_eligible"] is True
        for trial in manifest["adaptive_comparison_pool"]
    )
    odd_manifest = deepcopy(manifest)
    odd_manifest["adaptive_comparison_pool"] = odd_manifest["adaptive_comparison_pool"][:11]
    odd_manifest["adaptive_policy"]["maximum_followups_per_session"] = 6
    odd_manifest["adaptive_policy"]["maximum_followups_total"] = 11
    odd_manifest["adaptive_policy"]["followup_pool_counts_by_session"] = [6, 5]
    validate_mannequin_pilot_manifest(odd_manifest, pilot_root)
    invalid_policy = deepcopy(manifest)
    invalid_policy["adaptive_policy"]["maximum_followups_total"] = 11
    with pytest.raises(ValueError, match="adaptive total cap"):
        validate_mannequin_pilot_manifest(invalid_policy, pilot_root)
    invalid_mode = deepcopy(manifest)
    invalid_mode["adaptive_policy"]["required_comparison_mode"] = "sequential_neutral_gap"
    with pytest.raises(ValueError, match="toggle-only contract"):
        validate_mannequin_pilot_manifest(invalid_mode, pilot_root)

    scored_public = _public_trial(base_trials[0], manifest)
    assert {
        "source_ids",
        "families",
        "hidden_anchor",
        "stimulus_populations",
        "measurement_cohorts",
        "critic_training_eligible",
    }.isdisjoint(scored_public)
    tutorial_public = _public_trial(manifest["calibration_tutorial"][0], manifest)
    assert tutorial_public["is_tutorial"] is True
    assert tutorial_public["tutorial_disclosure"]["ordered_severities"] == list(
        CALIBRATION_SEVERITIES
    )

    assert not (pilot_root / "raw_observations.jsonl").exists()
    assert not (pilot_root / "served_playlist.jsonl").exists()
    assert not (pilot_root / "PILOT_PAUSED.json").exists()
    assert (
        phase14_snapshot
        == {
            path.relative_to(phase14): path.read_bytes()
            for path in phase14.rglob("*")
            if path.is_file()
        }
        if phase14.is_dir()
        else {}
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_subjective_calibration_bank(PHASE8_DATASET, bank_root)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_mannequin_subjective_pilot(
            PAIR_DATASET,
            AUDIT_QUEUE,
            PHASE8_DATASET,
            bank_root,
            pilot_root,
        )


@pytest.mark.parametrize(
    "protected_name",
    ["raw_observations.jsonl", "served_playlist.jsonl", "PILOT_PAUSED.json"],
)
def test_revised_pilot_refuses_human_evidence_or_pause_sentinel(
    tmp_path: Path,
    protected_name: str,
) -> None:
    output = tmp_path / "existing-pilot"
    output.mkdir()
    protected = output / protected_name
    protected.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="human evidence or pause sentinel"):
        prepare_mannequin_subjective_pilot(
            tmp_path / "missing-pairs",
            tmp_path / "missing-audit.jsonl",
            tmp_path / "missing-corruptions",
            tmp_path / "missing-bank.json",
            output,
        )
    assert protected.read_text(encoding="utf-8") == "keep me\n"


def test_calibration_validator_rejects_training_eligible_rung(tmp_path: Path) -> None:
    _require_source_artifacts()
    root = tmp_path / "bank"
    prepare_subjective_calibration_bank(PHASE8_DATASET, root)
    manifest = json.loads((root / "calibration_bank.json").read_text(encoding="utf-8"))
    manifest["stimuli"][0]["critic_training_eligible"] = True

    with pytest.raises(ValueError, match="purpose metadata"):
        validate_calibration_bank_manifest(manifest, root)
