from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from motionlab.perceptual.observation_auth import resolve_frozen_evidence_paths
from motionlab.perceptual.subjective_analysis import (
    _anchor_drift,
    _pilot_completion_audit,
    _quality_standard_error,
    _validate_completed_observations,
    analyze_subjective_pilot,
)


def test_anchor_drift_matches_the_same_anchor_across_sessions() -> None:
    def anchor(
        stimulus_id: str,
        session_index: int,
        rating: int,
    ) -> dict[str, Any]:
        return {
            "anchor_status": "hidden_session_anchor",
            "rating": rating,
            "rater_id": "rater-a",
            "stimulus_ids": [stimulus_id],
            "session_index": session_index,
            "session_id": f"session-{session_index}",
            "trial_index": session_index,
        }

    report = _anchor_drift(
        [
            anchor("anchor-a", 0, 3),
            anchor("anchor-b", 0, 7),
            anchor("anchor-a", 1, 4),
        ]
    )

    # Pooled means move down because session 1 omits anchor-b, while the matched anchor moves up.
    assert [row["mean_rating_diagnostic_only"] for row in report["session_anchor_drift"]] == [
        5.0,
        4.0,
    ]
    matched = report["matched_anchor_cross_session_drift"]
    assert len(matched) == 1
    assert matched[0]["stimulus_id"] == "anchor-a"
    assert matched[0]["later_session_comparisons"][0]["delta_from_first_matched_session"] == 1.0
    assert report["matched_anchor_cross_session_summary"] == {
        "matched_rater_anchor_count": 1,
        "comparison_count": 1,
        "mean_signed_rating_delta": 1.0,
        "mean_absolute_rating_delta": 1.0,
    }


def _canonical_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _current_evidence_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    viewing_settings = {
        "protocol": "motionlab.fixed_mannequin_canvas.v1",
        "playback_rates": [0.5, 1.0],
        "zoom_limits": {"minimum": 0.75, "maximum": 1.5},
        "orbit": {"maximum_pitch_degrees": 35.0},
        "default_camera": {
            "preset": "three_quarter",
            "yaw_degrees": 25.0,
            "pitch_degrees": -7.0,
            "zoom": 1.0,
            "skeleton_overlay": False,
        },
    }
    stimulus = {
        "stimulus_id": "stimulus-a",
        "source_id": "source-a",
        "variant_id": "variant-a",
        "family": "subtle-coordination",
        "style": "neutral",
        "stimulus_population": "HARD_FEASIBLE_PERCEPTUAL",
        "measurement_cohort": "hard_feasible",
    }
    trial = {
        "trial_id": "trial-a",
        "trial_index": 0,
        "task_type": "naturalness",
        "stimulus_ids": ["stimulus-a"],
        "source_ids": ["source-a"],
        "families": ["subtle-coordination"],
        "styles": ["neutral"],
        "side_classes": ["bilateral"],
        "hidden_repeat_group_ids": ["repeat-a"],
        "hidden_anchor": False,
        "comparison_mode": None,
        "presentation_order": ["stimulus-a"],
        "view_mirror_x": False,
        "render_hashes": ["presented-render:a"],
        "pair_id": None,
        "schedule_reason": "hard_feasible_single_stimulus",
        "session_index": 0,
        "seed": 10,
        "stimulus_populations": ["HARD_FEASIBLE_PERCEPTUAL"],
        "measurement_cohorts": ["hard_feasible"],
        "critic_training_eligible": True,
    }
    manifest = {
        "format_version": "motionlab.subjective_pilot.v2",
        "protocol_version": "motionlab.subjective_protocol.v3",
        "seed": 10,
        "render_protocol_hash": "render:test",
        "viewing_settings": viewing_settings,
        "stimuli": [stimulus],
        "sessions": [{"session_index": 0, "trials": [trial]}],
        "adaptive_comparison_pool": [],
    }
    session_id = _canonical_id("session", {"rater": "rater-a", "index": 0, "pilot_seed": 10})
    camera = dict(viewing_settings["default_camera"])
    observation = {
        "format_version": "motionlab.subjective_observation.v2",
        "observation_id": "observation:sha256:" + "a" * 64,
        "render_protocol_hash": "render:test",
        "render_hashes": ["presented-render:a"],
        "task_type": "naturalness",
        "rating": 4,
        "pair_outcome_display_order": None,
        "pair_outcome_canonical": None,
        "stimulus_ids": ["stimulus-a"],
        "stimulus_populations": ["HARD_FEASIBLE_PERCEPTUAL"],
        "measurement_cohorts": ["hard_feasible"],
        "critic_training_eligible": True,
        "source_ids": ["source-a"],
        "variant_ids": ["variant-a"],
        "rater_id": "rater-a",
        "session_id": session_id,
        "session_index": 0,
        "trial_id": "trial-a",
        "trial_index": 0,
        "playlist_seed": 10,
        "timestamp_utc": "2026-01-01T00:00:01+00:00",
        "presentation_order": ["stimulus-a"],
        "schedule_reason": "hard_feasible_single_stimulus",
        "hidden_repeat_group_ids": ["repeat-a"],
        "anchor_status": "not_anchor",
        "confidence": 4,
        "judgment_difficulty": "moderate",
        "reason_tags": ["naturalness"],
        "families": ["subtle-coordination"],
        "styles": ["neutral"],
        "response_time_ms": 3000.0,
        "replay_count": [0],
        "marked_interval_s": None,
        "comparison_mode": None,
        "equivalence_control": False,
        "pair_id": None,
        "viewing_settings": viewing_settings,
        "view_mirror_x": False,
        "presented_side_classes": ["bilateral"],
        "inspection_telemetry": {
            "format_version": "motionlab.inspection_telemetry.v1",
            "replay_count": [0],
            "speed_change_events": [],
            "playback_wall_time_by_speed_ms": {"0.5": 0.0, "1": 1000.0},
            "camera_change_events": [],
            "zoom_change_count": 0,
            "view_change_count": 0,
            "overlay_change_count": 0,
            "total_inspection_time_ms": 1000.0,
            "final_playback_rate": 1.0,
            "final_camera_state": camera,
        },
    }
    served = {
        "format_version": "motionlab.subjective_protocol.v3",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "rater_id": "rater-a",
        "session_id": session_id,
        "session_index": 0,
        "trial_id": "trial-a",
        "trial_index": 0,
        "seed": 10,
        "task_type": "naturalness",
        "presentation_order": ["stimulus-a"],
        "schedule_reason": "hard_feasible_single_stimulus",
    }
    return manifest, observation, served


def _row(
    index: int,
    stimulus_ids: list[str],
    *,
    rating: int | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    return {
        "format_version": "motionlab.subjective_observation.v2",
        "observation_id": f"observation-{index:03d}",
        "render_protocol_hash": "render:synthetic-readiness",
        "task_type": "naturalness" if rating is not None else "pair_comparison",
        "rating": rating,
        "pair_outcome_canonical": outcome,
        "stimulus_ids": stimulus_ids,
        "stimulus_populations": ["HARD_FEASIBLE_PERCEPTUAL"] * len(stimulus_ids),
        "measurement_cohorts": ["hard_feasible"] * len(stimulus_ids),
        "critic_training_eligible": True,
        "source_ids": [f"source-{int(value[-1]) % 2}" for value in stimulus_ids],
        "rater_id": f"synthetic-rater-{index % 2}",
        "session_id": f"synthetic-session-{index % 2}",
        "session_index": index % 2,
        "trial_id": f"trial-{index:03d}",
        "trial_index": index,
        "timestamp_utc": f"2026-01-01T00:00:{index:02d}+00:00",
        "hidden_repeat_group_ids": [],
        "anchor_status": "not_anchor",
        "confidence": 4,
        "judgment_difficulty": "moderate",
        "reason_tags": [],
        "families": ["synthetic_subtle"],
        "styles": ["synthetic-style"],
        "response_time_ms": 3000 + index,
        "replay_count": [0] * len(stimulus_ids),
        "comparison_mode": "same_viewport_toggle" if outcome is not None else None,
        "equivalence_control": False,
        "pair_id": f"pair-{index:03d}" if outcome is not None else None,
    }


def test_direct_pair_evidence_is_held_out_from_ordinal_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stimuli = [
        {
            "stimulus_id": f"stimulus-{index}",
            "source_id": f"source-{index % 2}",
            "family": "synthetic_subtle",
            "stimulus_population": "HARD_FEASIBLE_PERCEPTUAL",
            "measurement_cohort": "hard_feasible",
            "critic_training_eligible": True,
        }
        for index in range(5)
    ]
    manifest = {
        "format_version": "motionlab.subjective_pilot.v2",
        "render_protocol_hash": "render:synthetic-readiness",
        "viewing_settings": {"protocol": "synthetic-test-only"},
        "stimuli": stimuli,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    index = 0
    for stimulus_index, rating in enumerate((7, 5, 3, 1)):
        for _ in range(4):
            rows.append(_row(index, [f"stimulus-{stimulus_index}"], rating=rating))
            index += 1
    rows.extend(
        [
            _row(index, ["stimulus-0", "stimulus-3"], outcome="first_better"),
            _row(index + 1, ["stimulus-2", "stimulus-1"], outcome="second_better"),
            _row(index + 2, ["stimulus-0", "stimulus-1"], outcome="effectively_equal"),
            # stimulus-4 deliberately has no ordinal rating, exercising explicit coverage.
            _row(index + 3, ["stimulus-0", "stimulus-4"], outcome="first_better"),
        ]
    )
    for row in rows[:3]:
        row["anchor_status"] = "hidden_session_anchor"
    observations_path = tmp_path / "dummy_observations.jsonl"
    observations_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "motionlab.perceptual.subjective_analysis.load_perceptual_pairs",
        lambda _: [
            {
                "pair_id": "pair-016",
                "preference": "a_better",
                "perturbation_mechanism": "synthetic_subtle",
            }
        ],
    )

    output = tmp_path / "analysis"
    report = analyze_subjective_pilot(
        tmp_path / "dummy-pairs",
        manifest_path,
        observations_path,
        output,
    )
    evidence = json.loads((output / "direct_pair_vs_ordinal_evidence.json").read_text())
    dashboard = json.loads((output / "reliability_dashboard.json").read_text())
    agreement = json.loads((output / "human_synthetic_agreement.json").read_text())

    assert report["format_version"] == "motionlab.subjective_analysis.v3"
    assert report["outputs"]["direct_pair_vs_ordinal_evidence"].endswith(
        "direct_pair_vs_ordinal_evidence.json"
    )
    assert evidence["status"] == "ready"
    assert evidence["ordinal_model_excludes_direct_pairs"] is True
    assert evidence["direct_pair_observation_count"] == 4
    assert evidence["ordinal_covered_direct_pair_count"] == 3
    assert evidence["uncovered_direct_pair_count"] == 1
    assert evidence["decisive_comparison_count"] == 2
    assert evidence["directional_agreement"] == pytest.approx(1.0)
    assert evidence["by_family"]["synthetic_subtle"]["direct_pair_observation_count"] == 4
    assert evidence["by_family"]["synthetic_subtle"]["directional_agreement"] == pytest.approx(1.0)
    assert evidence["records"][-1]["ordinal_model_coverage"] is False
    assert dashboard["direct_pair_vs_ordinal_evidence"] == evidence
    assert dashboard["anchor_calibration"]["session_anchor_drift"]
    assert any(
        row["stimulus_id"] == "stimulus-4" for row in dashboard["items_needing_more_judgments"]
    )
    assert agreement["comparison_count"] == 1
    assert agreement["by_family"]["synthetic_subtle"] == {
        "count": 1,
        "agreement": 1.0,
    }
    # Dummy fixtures prove machinery only; the command still does no critic training.
    assert report["critic_retraining_started"] is False
    with pytest.raises(ValueError, match="analysis output must be fresh"):
        analyze_subjective_pilot(
            tmp_path / "dummy-pairs",
            manifest_path,
            observations_path,
            output,
        )


def test_pilot_completion_requires_all_manifest_and_adaptive_trials() -> None:
    manifest = {
        "sessions": [
            {"trials": [{"trial_id": "base-0"}]},
            {"trials": [{"trial_id": "base-1"}]},
        ],
        "adaptive_policy": {"enabled": True, "maximum_followups_total": 2},
    }
    partial = [
        {"rater_id": "rater", "trial_id": "base-0", "schedule_reason": "base"},
        {"rater_id": "rater", "trial_id": "base-1", "schedule_reason": "base"},
        {"rater_id": "rater", "trial_id": "followup-0", "schedule_reason": "adaptive:x"},
    ]
    audit = _pilot_completion_audit(partial, manifest)
    assert audit["complete"] is False
    assert audit["expected_total_trial_count_per_rater"] == 4
    complete = _pilot_completion_audit(
        [
            *partial,
            {
                "rater_id": "rater",
                "trial_id": "followup-1",
                "schedule_reason": "adaptive:y",
            },
        ],
        manifest,
    )
    assert complete["complete"] is True
    assert complete["completed_rater_count"] == 1


def test_quality_interval_uses_full_covariance_and_rejects_invalid_variance() -> None:
    covariance = np.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    assert _quality_standard_error(covariance, 0, 1) == pytest.approx(math.sqrt(3.0))
    covariance[0, 1] = covariance[1, 0] = -2.0
    assert _quality_standard_error(covariance, 0, 1) is None
    covariance[0, 1] = covariance[1, 0] = np.nan
    assert _quality_standard_error(covariance, 0, 1) is None


def test_completed_observation_validation_rejects_forged_or_malformed_rows() -> None:
    manifest, valid, served_row = _current_evidence_fixture()
    served = [served_row]
    _validate_completed_observations(
        [valid],
        manifest,
        require_current_protocol_fields=True,
        served_playlist=served,
    )
    with pytest.raises(ValueError, match="duplicate subjective observation_id"):
        _validate_completed_observations(
            [valid, dict(valid)],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=served,
        )
    with pytest.raises(ValueError, match="unknown pilot stimuli"):
        _validate_completed_observations(
            [{**valid, "stimulus_ids": ["unknown"]}],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=served,
        )
    with pytest.raises(ValueError, match="no valid completed rating"):
        _validate_completed_observations(
            [{**valid, "rating": None}],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=served,
        )
    with pytest.raises(ValueError, match="not linked to a served trial"):
        _validate_completed_observations(
            [valid],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=[],
        )


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("source_ids", ["forged-source"]),
        ("variant_ids", ["forged-variant"]),
        ("render_protocol_hash", "render:forged"),
        ("render_hashes", ["forged-render"]),
        ("families", ["forged-family"]),
        ("styles", ["forged-style"]),
        ("stimulus_populations", ["CALIBRATION_ONLY"]),
        ("measurement_cohorts", ["calibration_easy"]),
        ("critic_training_eligible", False),
        ("pair_id", "forged-pair"),
        ("comparison_mode", "same_viewport_toggle"),
        ("equivalence_control", True),
        ("viewing_settings", {"protocol": "forged"}),
        ("view_mirror_x", True),
        ("presented_side_classes", ["left"]),
        ("hidden_repeat_group_ids", ["forged-repeat"]),
        ("anchor_status", "hidden_session_anchor"),
    ],
)
def test_current_observation_authenticates_all_analysis_metadata(
    field: str,
    forged: Any,
) -> None:
    manifest, valid, served = _current_evidence_fixture()
    with pytest.raises(ValueError, match=field):
        _validate_completed_observations(
            [{**valid, field: forged}],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=[served],
        )


def test_current_observation_requires_exact_served_presentation_and_session_identity() -> None:
    manifest, valid, served = _current_evidence_fixture()
    with pytest.raises(ValueError, match="exact presentation metadata"):
        _validate_completed_observations(
            [{**valid, "presentation_order": ["forged-stimulus"]}],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=[served],
        )
    with pytest.raises(ValueError, match="session identity"):
        _validate_completed_observations(
            [{**valid, "session_id": "session:sha256:" + "f" * 64}],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=[{**served, "session_id": "session:sha256:" + "f" * 64}],
        )


def test_current_adaptive_observation_authenticates_realized_schedule() -> None:
    manifest, observation, served = _current_evidence_fixture()
    adaptive_trial = dict(manifest["sessions"][0]["trials"][0])
    adaptive_trial["schedule_reason"] = "adaptive:precomputed_information_stratified_toggle_pool"
    manifest["sessions"][0]["trials"][0] = {
        **adaptive_trial,
        "trial_id": "unobserved-base-trial",
        "schedule_reason": "hard_feasible_single_stimulus",
    }
    manifest["adaptive_comparison_pool"] = [adaptive_trial]
    realized = "adaptive:response_adaptive_toggle_pool+underrepresented_family"
    observation.update({"trial_index": 1, "schedule_reason": realized})
    served.update({"trial_index": 1, "schedule_reason": realized})
    _validate_completed_observations(
        [observation],
        manifest,
        require_current_protocol_fields=True,
        served_playlist=[served],
    )
    invalid = "adaptive:response_adaptive_toggle_pool+forged_reason"
    with pytest.raises(ValueError, match="invalid adaptive reason"):
        _validate_completed_observations(
            [{**observation, "schedule_reason": invalid}],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=[{**served, "schedule_reason": invalid}],
        )


def test_current_pair_outcome_must_match_display_order() -> None:
    manifest, first, served = _current_evidence_fixture()
    second_stimulus = {
        **manifest["stimuli"][0],
        "stimulus_id": "stimulus-b",
        "variant_id": "variant-b",
    }
    manifest["stimuli"].append(second_stimulus)
    trial = manifest["sessions"][0]["trials"][0]
    trial.update(
        {
            "task_type": "pair_comparison",
            "stimulus_ids": ["stimulus-a", "stimulus-b"],
            "source_ids": ["source-a"],
            "families": ["subtle-coordination"],
            "styles": ["neutral"],
            "side_classes": ["bilateral"],
            "comparison_mode": "same_viewport_toggle",
            "presentation_order": ["stimulus-b", "stimulus-a"],
            "render_hashes": ["presented-render:a", "presented-render:b"],
            "pair_id": "pair-a",
            "stimulus_populations": [
                "HARD_FEASIBLE_PERCEPTUAL",
                "HARD_FEASIBLE_PERCEPTUAL",
            ],
            "measurement_cohorts": ["hard_feasible", "hard_feasible"],
        }
    )
    pair = {
        **first,
        "task_type": "pair_comparison",
        "stimulus_ids": ["stimulus-a", "stimulus-b"],
        "source_ids": ["source-a"],
        "variant_ids": ["variant-a", "variant-b"],
        "families": ["subtle-coordination"],
        "styles": ["neutral"],
        "render_hashes": ["presented-render:a", "presented-render:b"],
        "presentation_order": ["stimulus-b", "stimulus-a"],
        "comparison_mode": "same_viewport_toggle",
        "pair_id": "pair-a",
        "rating": None,
        "pair_outcome_display_order": "a_better",
        # Display A is canonical stimulus-b, so this must be second_better.
        "pair_outcome_canonical": "second_better",
        "stimulus_populations": [
            "HARD_FEASIBLE_PERCEPTUAL",
            "HARD_FEASIBLE_PERCEPTUAL",
        ],
        "measurement_cohorts": ["hard_feasible", "hard_feasible"],
        "replay_count": [0, 0],
        "inspection_telemetry": {
            **first["inspection_telemetry"],
            "replay_count": [0, 0],
        },
    }
    pair_served = {
        **served,
        "task_type": "pair_comparison",
        "presentation_order": ["stimulus-b", "stimulus-a"],
    }
    _validate_completed_observations(
        [pair],
        manifest,
        require_current_protocol_fields=True,
        served_playlist=[pair_served],
    )
    with pytest.raises(ValueError, match="valid completed comparison"):
        _validate_completed_observations(
            [{**pair, "pair_outcome_canonical": "first_better"}],
            manifest,
            require_current_protocol_fields=True,
            served_playlist=[pair_served],
        )


def test_current_analysis_rejects_noncanonical_frozen_evidence_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = tmp_path / "pilot_manifest.json"
    observations = tmp_path / "raw_observations.jsonl"
    served = tmp_path / "served_playlist.jsonl"
    copied = tmp_path / "copied_observations.jsonl"
    freeze = tmp_path / "protocol_freeze.json"
    pilot.write_text("{}", encoding="utf-8")
    observations.write_text("", encoding="utf-8")
    copied.write_text("", encoding="utf-8")
    served.write_text("", encoding="utf-8")
    freeze.write_text(
        json.dumps(
            {
                "pilot_manifest": pilot.name,
                "append_only_evidence_prefixes": {
                    "observations": {"path": observations.name},
                    "served_playlist": {"path": served.name},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "motionlab.perceptual.observation_auth.validate_active_evaluation_protocol",
        lambda *args, **kwargs: {"freeze_id": "freeze:test"},
    )
    resolved, validation = resolve_frozen_evidence_paths(pilot, observations, freeze)
    assert resolved == served.resolve()
    assert validation == {"freeze_id": "freeze:test"}
    with pytest.raises(ValueError, match="canonical frozen observation log"):
        resolve_frozen_evidence_paths(pilot, copied, freeze)
    with pytest.raises(ValueError, match="canonical frozen served-playlist log"):
        resolve_frozen_evidence_paths(
            pilot,
            observations,
            freeze,
            served_playlist_path=tmp_path / "copied_playlist.jsonl",
        )
