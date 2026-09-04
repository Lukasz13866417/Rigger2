from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from motionlab.perceptual.subjective_analysis import analyze_subjective_pilot

SEVERITIES = ("clean", "mild", "medium", "strong", "severe")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _telemetry(*, exercised_controls: bool) -> dict[str, Any]:
    return {
        "format_version": "motionlab.inspection_telemetry.v1",
        "replay_count": [2 if exercised_controls else 0],
        "speed_change_events": (
            [
                {"elapsed_ms": 1000, "from_rate": 1.0, "to_rate": 0.5},
                {"elapsed_ms": 3000, "from_rate": 0.5, "to_rate": 1.0},
            ]
            if exercised_controls
            else []
        ),
        "playback_wall_time_by_speed_ms": {
            "0.25": 0,
            "0.5": 2000 if exercised_controls else 0,
            "1": 1000,
            "1.5": 0,
            "2": 0,
        },
        "camera_change_events": (
            [
                {"elapsed_ms": 1100, "kind": "zoom", "from": 1.0, "to": 1.25},
                {
                    "elapsed_ms": 1200,
                    "kind": "view",
                    "from": "three_quarter",
                    "to": "side",
                },
                {"elapsed_ms": 1300, "kind": "reset", "from": "side", "to": "default"},
                {"elapsed_ms": 1400, "kind": "orbit", "from": 25, "to": 35},
                {"elapsed_ms": 1500, "kind": "overlay", "from": False, "to": True},
            ]
            if exercised_controls
            else []
        ),
        "zoom_change_count": 1 if exercised_controls else 0,
        "view_change_count": 1 if exercised_controls else 0,
        "overlay_change_count": 1 if exercised_controls else 0,
        "total_inspection_time_ms": 7000 if exercised_controls else 1000,
        "final_playback_rate": 1.0,
        "final_camera_state": {
            "preset": "side" if exercised_controls else "three_quarter",
            "yaw_degrees": 90 if exercised_controls else 25,
            "pitch_degrees": 0,
            "zoom": 1.25 if exercised_controls else 1.0,
            "skeleton_overlay": exercised_controls,
        },
    }


def _observation(
    *,
    index: int,
    stimulus: dict[str, Any],
    rating: int,
    repeat: int,
    exercised_controls: bool = False,
) -> dict[str, Any]:
    population = str(stimulus["stimulus_population"])
    cohort = str(stimulus["measurement_cohort"])
    eligible = bool(stimulus["critic_training_eligible"])
    telemetry = _telemetry(exercised_controls=exercised_controls)
    return {
        "format_version": "motionlab.subjective_observation.v2",
        "observation_id": f"observation-{index}",
        "render_protocol_hash": "render:revised-mannequin",
        "task_type": "naturalness",
        "rating": rating,
        "pair_outcome_canonical": None,
        "stimulus_ids": [stimulus["stimulus_id"]],
        "stimulus_populations": [population],
        "measurement_cohorts": [cohort],
        "critic_training_eligible": eligible,
        "source_ids": [stimulus["source_id"]],
        "rater_id": "rater-1",
        "session_id": "session-1",
        "session_index": 0,
        "trial_id": f"trial-{index}",
        "trial_index": index,
        "timestamp_utc": f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
        "hidden_repeat_group_ids": [f"repeat-{stimulus['stimulus_id']}"],
        "anchor_status": (
            "hidden_session_anchor" if population == "EVALUATION_ANCHOR" else "not_anchor"
        ),
        "confidence": 3 if cohort == "hard_feasible" else 5,
        "judgment_difficulty": "difficult" if cohort == "hard_feasible" else "easy",
        "reason_tags": [],
        "families": [stimulus["family"]],
        "styles": [stimulus["source_id"]],
        "response_time_ms": 10_000 + index,
        "replay_count": telemetry["replay_count"],
        "inspection_telemetry": telemetry,
        "comparison_mode": None,
        "equivalence_control": False,
        "viewing_settings": {"protocol": "motionlab.fixed_mannequin_canvas.v1"},
        "repeat_occurrence_index": repeat,
    }


def test_revised_dashboard_separates_measurement_and_training_populations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = [
        {
            "stimulus_id": f"calibration-{severity}",
            "source_id": "calibration-source",
            "family": "foot_slide",
            "stimulus_population": "CALIBRATION_ONLY",
            "measurement_cohort": "calibration_easy",
            "calibration_family": "foot_slide",
            "calibration_severity": severity,
            "critic_training_eligible": False,
        }
        for severity in SEVERITIES
    ]
    hard_feasible = [
        {
            "stimulus_id": f"hard-{index}",
            "source_id": f"hard-source-{index}",
            "family": "subtle_phase_mismatch",
            "stimulus_population": "HARD_FEASIBLE_PERCEPTUAL",
            "measurement_cohort": "hard_feasible",
            "calibration_family": None,
            "calibration_severity": None,
            "critic_training_eligible": True,
        }
        for index in range(3)
    ]
    anchors = [
        {
            "stimulus_id": "easy-anchor",
            "source_id": "anchor-source",
            "family": "clean_reference",
            "stimulus_population": "EVALUATION_ANCHOR",
            "measurement_cohort": "calibration_easy",
            "calibration_family": None,
            "calibration_severity": None,
            "critic_training_eligible": False,
        }
    ]
    stimuli = calibration + hard_feasible + anchors
    manifest = {
        "format_version": "motionlab.subjective_pilot.v2",
        "render_protocol_hash": "render:revised-mannequin",
        "viewing_settings": {"protocol": "motionlab.fixed_mannequin_canvas.v1"},
        "stimuli": stimuli,
    }
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    index = 0
    for stimulus, rating in zip(calibration, (7, 6, 5, 3, 1), strict=True):
        for repeat in range(2):
            rows.append(
                _observation(
                    index=index,
                    stimulus=stimulus,
                    rating=rating,
                    repeat=repeat,
                    exercised_controls=index == 0,
                )
            )
            index += 1
    for stimulus, ratings in zip(
        hard_feasible,
        ((4, 4), (5, 4), (6, 6)),
        strict=True,
    ):
        for repeat, rating in enumerate(ratings):
            rows.append(
                _observation(
                    index=index,
                    stimulus=stimulus,
                    rating=rating,
                    repeat=repeat,
                )
            )
            index += 1
    for repeat in range(2):
        rows.append(
            _observation(
                index=index,
                stimulus=anchors[0],
                rating=7,
                repeat=repeat,
            )
        )
        index += 1
    # A contradictory client-side flag can never make a calibration stimulus trainable.
    rows[0]["critic_training_eligible"] = True
    observations_path = tmp_path / "observations.jsonl"
    _write_jsonl(observations_path, rows)
    monkeypatch.setattr(
        "motionlab.perceptual.subjective_analysis.load_perceptual_pairs",
        lambda _: [],
    )

    output = tmp_path / "analysis"
    report = analyze_subjective_pilot(
        tmp_path / "pair-dataset",
        manifest_path,
        observations_path,
        output,
    )
    dashboard = json.loads((output / "reliability_dashboard.json").read_text())
    evidence = [
        json.loads(line) for line in (output / "training_evidence.jsonl").read_text().splitlines()
    ]

    assert report["latent_model_observation_count"] == 8
    assert dashboard["rating_distribution"]["overall"]["ordinal_observation_count"] == 18
    assert (
        dashboard["rating_distribution"]["by_measurement_cohort"]["calibration_easy"][
            "ordinal_observation_count"
        ]
        == 12
    )
    assert (
        dashboard["repeat_agreement_by_measurement_cohort"]["hard_feasible"][
            "ordinal_repeat_pair_count"
        ]
        == 3
    )
    inspection = dashboard["inspection_usage"]
    assert inspection["telemetry_observation_count"] == 18
    assert inspection["speed_controls"]["speed_change_count"] == 2
    assert inspection["speed_controls"]["playback_wall_time_seconds_by_speed"][
        "0.5"
    ] == pytest.approx(2.0)
    assert inspection["camera_controls"]["zoom_change_count"] == 1
    assert inspection["camera_controls"]["view_change_count"] == 1
    assert inspection["camera_controls"]["reset_count"] == 1
    assert inspection["camera_controls"]["orbit_change_count"] == 1
    assert dashboard["judgment_difficulty"]["overall"]["counts"] == {
        "difficult": 6,
        "easy": 12,
        "moderate": 0,
    }

    perceptibility = dashboard["family_perceptibility"]
    assert perceptibility["families"]["foot_slide"]["status"] == "sufficient_for_ranking"
    assert perceptibility["families"]["foot_slide"][
        "severity_order_accuracy_ties_half_credit"
    ] == pytest.approx(1.0)
    assert perceptibility["families"]["subtle_phase_mismatch"]["status"] == "insufficient_data"
    assert perceptibility["ranked_easiest_to_hardest"] == [
        "foot_slide",
        "subtle_phase_mismatch",
    ]
    assert perceptibility["severity_ladder_ranked_easiest_to_hardest"] == ["foot_slide"]
    assert perceptibility["ranking_is_exploratory_not_perceptual_accuracy"] is True
    assert (
        dashboard["mannequin_and_inspection_measurement_quality"]["status"]
        == "mannequin_absolute_only_no_matched_skeleton_baseline"
    )

    eligibility = dashboard["training_eligibility"]
    assert eligibility["eligible_observation_count"] == 6
    assert eligibility["excluded_observation_count"] == 12
    assert eligibility["excluded_observation_counts_by_population"] == {
        "CALIBRATION_ONLY": 10,
        "EVALUATION_ANCHOR": 2,
    }
    assert sum(row["evidence_kind"] == "raw_human_ordinal" for row in evidence) == 6
    assert sum(row["evidence_kind"] == "fitted_human_latent_quality" for row in evidence) == 3
    assert all(row["critic_training_eligible"] is True for row in evidence)
    assert {stimulus_id for row in evidence for stimulus_id in row["stimulus_ids"]} == {
        item["stimulus_id"] for item in hard_feasible
    }
