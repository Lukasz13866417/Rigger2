from __future__ import annotations

import json
from pathlib import Path

import pytest

from motionlab.perceptual import protocol_freeze, subjective, subjective_v3
from motionlab.perceptual.calibration_bank import EVALUATION_ANCHOR


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _telemetry() -> dict[str, object]:
    return {
        "camera_change_events": [],
        "final_camera_state": {
            "pitch_degrees": -7.0,
            "preset": "three_quarter",
            "skeleton_overlay": False,
            "yaw_degrees": 25.0,
            "zoom": 1.0,
        },
        "final_playback_rate": 1.0,
        "format_version": subjective.INSPECTION_TELEMETRY_VERSION,
        "overlay_change_count": 0,
        "playback_wall_time_by_speed_ms": {
            "0.25": 0.0,
            "0.5": 0.0,
            "1": 1000.0,
            "1.5": 0.0,
            "2": 0.0,
        },
        "replay_count": [0],
        "speed_change_events": [],
        "total_inspection_time_ms": 1000.0,
        "view_change_count": 0,
        "zoom_change_count": 0,
    }


def _observation(seed: int, *, observation_id: str) -> dict[str, object]:
    return {
        "anchor_status": "hidden_session_anchor",
        "comparison_mode": None,
        "confidence": 4,
        "critic_training_eligible": False,
        "equivalence_control": False,
        "families": ["clean_reference"],
        "format_version": subjective.RAW_OBSERVATION_VERSION,
        "hidden_repeat_group_ids": ["repeat:test"],
        "inspection_telemetry": _telemetry(),
        "judgment_difficulty": "easy",
        "marked_interval_s": None,
        "measurement_cohorts": ["calibration_easy"],
        "observation_id": observation_id,
        "pair_id": None,
        "pair_outcome_canonical": None,
        "pair_outcome_display_order": None,
        "playlist_seed": seed,
        "presentation_order": ["stimulus:test"],
        "presented_side_classes": ["neutral"],
        "rater_id": "fixture-rater",
        "rating": 6,
        "reason_tags": [],
        "render_hashes": ["presented-render:test"],
        "render_protocol_hash": subjective.render_protocol_hash(),
        "replay_count": [0],
        "response_time_ms": 5000.0,
        "schedule_reason": "hidden_broad_range_anchor",
        "session_id": "session:test",
        "session_index": 0,
        "source_ids": ["source"],
        "stimulus_ids": ["stimulus:test"],
        "stimulus_populations": [EVALUATION_ANCHOR],
        "styles": ["style"],
        "task_type": "naturalness",
        "timestamp_utc": "2026-09-04T00:00:00+00:00",
        "trial_id": "trial:test",
        "trial_index": 0,
        "variant_ids": ["variant"],
        "view_mirror_x": False,
        "viewing_settings": subjective.MANNEQUIN_VIEWING_SETTINGS,
    }


def _playlist(seed: int, *, trial_id: str) -> dict[str, object]:
    return {
        "format_version": subjective.SUBJECTIVE_PROTOCOL_VERSION,
        "presentation_order": ["stimulus:test"],
        "rater_id": "fixture-rater",
        "schedule_reason": "hidden_broad_range_anchor",
        "seed": seed,
        "session_id": "session:test",
        "session_index": 0,
        "task_type": "naturalness",
        "timestamp_utc": "2026-09-04T00:00:00+00:00",
        "trial_id": trial_id,
        "trial_index": 0,
    }


def _active_pilot(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "pilot"
    root.mkdir()
    motion = root / "motions" / "anchor.npz"
    bank_motion = root / "calibration_bank" / "motions" / "calibration.npz"
    motion.parent.mkdir()
    bank_motion.parent.mkdir(parents=True)
    motion.write_bytes(b"immutable active motion")
    bank_motion.write_bytes(b"immutable calibration motion")

    anchor = {
        "stimulus_id": "stimulus:test",
        "stimulus_population": EVALUATION_ANCHOR,
        "motion": "motions/anchor.npz",
        "phase_reference_motion": "motions/anchor.npz",
        "render_hash": "presented-render:test",
        "render_protocol_hash": subjective.render_protocol_hash(),
        "motion_content_hash": "sha256:motion",
        "visual_content_hash": "visual-motion:test",
        "family": "clean_reference",
        "calibration_severity": "clean",
        "source_id": "source",
        "variant_id": "variant",
    }
    bank = {
        "format_version": "motionlab.subjective_calibration_bank.v1",
        "stimulus_directory": str((root / "calibration_bank").resolve()),
        "viewing_settings": subjective.MANNEQUIN_VIEWING_SETTINGS,
        "render_protocol_hash": subjective.render_protocol_hash(),
        "stimuli": [
            {
                "motion": "motions/calibration.npz",
                "phase_reference_motion": "motions/calibration.npz",
            }
        ],
    }
    bank_path = root / "calibration_bank" / "calibration_bank.json"
    _write_json(bank_path, bank)
    _write_json(root / "calibration_bank" / "calibration_bank_preparation.json", {"ok": True})

    seed = 9502
    manifest = {
        "format_version": subjective.PILOT_MANIFEST_VERSION,
        "protocol_version": subjective.SUBJECTIVE_PROTOCOL_VERSION,
        "seed": seed,
        "viewing_settings": subjective.MANNEQUIN_VIEWING_SETTINGS,
        "render_protocol_hash": subjective.render_protocol_hash(),
        "scale_anchors": {
            "naturalness": list(subjective.NATURALNESS_SCALE),
            "style_adherence": list(subjective.STYLE_SCALE),
        },
        "stimuli": [anchor],
        "hidden_anchor_stimulus_ids": ["stimulus:test"],
        "hidden_repeated_anchor_stimulus_ids": ["stimulus:test"],
        "hidden_repeated_hard_feasible_stimulus_ids": [],
        "minimum_intervening_trials_for_repeat": 12,
        "repeat_validation": {
            "all_repeats_cross_session": True,
            "hidden_repeat_count": 1,
            "minimum_global_trial_index_distance": 15,
        },
        "session_count": 1,
        "sessions": [
            {"session_index": 0, "trial_count": 1, "trials": [{"trial_id": "trial:test"}]}
        ],
        "calibration_tutorial": [],
        "adaptive_comparison_pool": [],
        "adaptive_policy": {
            "enabled": False,
            "selection_mode": "response_adaptive_toggle_pool",
        },
        "calibration_bank": str(bank_path.resolve()),
    }
    manifest_path = root / "pilot_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(root / "pilot_preparation.json", {"status": "ready"})
    observations = root / "raw_observations.jsonl"
    playlist = root / "served_playlist.jsonl"
    observations.write_text(
        json.dumps(_observation(seed, observation_id="observation:one"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    playlist.write_text(
        json.dumps(_playlist(seed, trial_id="trial:one"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, observations, playlist


def test_freeze_captures_exact_protocol_and_allows_valid_appends(tmp_path: Path) -> None:
    manifest, observations, playlist = _active_pilot(tmp_path)
    before_manifest = manifest.read_bytes()
    before_observations = observations.read_bytes()
    before_playlist = playlist.read_bytes()

    frozen = protocol_freeze.freeze_active_evaluation_protocol(manifest)

    assert manifest.read_bytes() == before_manifest
    assert observations.read_bytes() == before_observations
    assert playlist.read_bytes() == before_playlist
    assert frozen["protocol_id"].startswith("human-evaluation-protocol:sha256:")
    assert frozen["immutable_snapshot_id"].startswith("active-pilot-snapshot:sha256:")
    contract = frozen["contract"]
    assert contract["mannequin_render"]["renderer_revision"] == "canvas2d_oriented_solids.v1"
    assert contract["camera_and_playback"]["first_pass"]["playback_rate"] == 1.0
    assert (
        contract["rating_semantics"]["single_stimulus"]["naturalness_question"]
        == "Ignoring whether you personally like the style, how natural and internally "
        "coherent is this walk?"
    )
    assert contract["hidden_repeat_policy"]["minimum_intervening_trials"] == 12
    assert contract["anchor_set"][0]["stimulus_id"] == "stimulus:test"
    assert contract["scheduler_randomization"]["seed"] == 9502
    assert contract["observation_schema"]["format_version"] == subjective.RAW_OBSERVATION_VERSION

    with observations.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(_observation(9502, observation_id="observation:two"), sort_keys=True) + "\n"
        )
    with playlist.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_playlist(9502, trial_id="trial:two"), sort_keys=True) + "\n")
    report = protocol_freeze.validate_active_evaluation_protocol(
        manifest.parent / "protocol_freeze.json"
    )
    assert report["status"] == "valid"
    assert report["observation_prefix_record_count"] == 1
    assert report["current_observation_record_count"] == 2
    assert report["current_served_playlist_record_count"] == 2


def test_freeze_fails_closed_when_an_active_asset_drifts(tmp_path: Path) -> None:
    manifest, _, _ = _active_pilot(tmp_path)
    protocol_freeze.freeze_active_evaluation_protocol(manifest)
    (manifest.parent / "motions" / "anchor.npz").write_bytes(b"changed")

    with pytest.raises(ValueError, match="immutable assets drifted"):
        protocol_freeze.validate_active_evaluation_protocol(
            manifest.parent / "protocol_freeze.json"
        )


def test_freeze_fails_closed_when_collected_evidence_is_rewritten(tmp_path: Path) -> None:
    manifest, observations, _ = _active_pilot(tmp_path)
    protocol_freeze.freeze_active_evaluation_protocol(manifest)
    observations.write_bytes(observations.read_bytes().replace(b"fixture-rater", b"edited-rater!"))

    with pytest.raises(ValueError, match="append-only evidence prefix drifted"):
        protocol_freeze.validate_active_evaluation_protocol(
            manifest.parent / "protocol_freeze.json"
        )


def test_freeze_rejects_contract_drift_and_refuses_overwrite(tmp_path: Path) -> None:
    manifest, _, _ = _active_pilot(tmp_path)
    freeze_path = manifest.parent / "protocol_freeze.json"
    protocol_freeze.freeze_active_evaluation_protocol(manifest)
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        protocol_freeze.freeze_active_evaluation_protocol(manifest)

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["seed"] = 9503
    _write_json(manifest, value)
    with pytest.raises(ValueError, match="protocol contract drifted"):
        protocol_freeze.validate_active_evaluation_protocol(freeze_path)


def test_freeze_rejects_new_observations_with_a_different_schema(tmp_path: Path) -> None:
    manifest, observations, _ = _active_pilot(tmp_path)
    protocol_freeze.freeze_active_evaluation_protocol(manifest)
    with observations.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    with pytest.raises(ValueError, match="wrong schema"):
        protocol_freeze.validate_active_evaluation_protocol(
            manifest.parent / "protocol_freeze.json"
        )


def test_render_ui_change_requires_a_new_render_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _ = _active_pilot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(subjective_v3, "_HTML", subjective_v3._HTML + "\n<!-- changed -->")

    with pytest.raises(ValueError, match="changed without a new render protocol"):
        protocol_freeze.build_active_evaluation_contract(manifest)


def test_historical_freeze_uses_archived_implementation_after_ui_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _ = _active_pilot(tmp_path)
    frozen = protocol_freeze.freeze_active_evaluation_protocol(manifest_path)
    monkeypatch.setattr(subjective, "_HTML", "new camera implementation")

    report = protocol_freeze.validate_active_evaluation_protocol(
        manifest_path.parent / "protocol_freeze.json"
    )

    assert report["status"] == "valid"
    assert report["protocol_id"] == frozen["protocol_id"]
    assert (
        frozen["contract"]["observation_schema"]["served_playlist"]["format_version"]
        == subjective_v3.SUBJECTIVE_PROTOCOL_VERSION
    )


def test_continuation_history_is_an_immutable_input(tmp_path: Path) -> None:
    manifest_path, _, _ = _active_pilot(tmp_path)
    history = manifest_path.parent / "continuation_history.json"
    _write_json(history, {"completed_trials": []})
    frozen = protocol_freeze.freeze_active_evaluation_protocol(manifest_path)

    assert "continuation_history.json" in {item["path"] for item in frozen["immutable_assets"]}
    _write_json(history, {"completed_trials": ["trial:test"]})

    with pytest.raises(ValueError, match="immutable assets drifted"):
        protocol_freeze.validate_active_evaluation_protocol(
            manifest_path.parent / "protocol_freeze.json"
        )


def test_declared_continuation_history_must_be_the_frozen_sibling(tmp_path: Path) -> None:
    manifest_path, _, _ = _active_pilot(tmp_path)
    history = manifest_path.parent / "other_history.json"
    _write_json(history, {"completed_trials": []})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["camera_continuation"] = {"history_path": str(history.resolve())}
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="must be the frozen sibling"):
        protocol_freeze.freeze_active_evaluation_protocol(manifest_path)


@pytest.mark.parametrize("collection_version", ["v3", "v4"])
def test_freeze_rejects_mixed_collection_and_render_versions(
    tmp_path: Path, collection_version: str
) -> None:
    manifest_path, _, _ = _active_pilot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if collection_version == "v3":
        manifest["viewing_settings"] = subjective.FREE_CAMERA_VIEWING_SETTINGS
    else:
        manifest["protocol_version"] = subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION

    with pytest.raises(ValueError, match="viewing settings differ"):
        protocol_freeze.build_active_evaluation_contract(manifest)


def test_free_camera_freeze_selects_new_implementation_and_playlist_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, observations_path, playlist_path = _active_pilot(tmp_path)
    version = subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION
    settings = subjective.FREE_CAMERA_VIEWING_SETTINGS
    render_hash = subjective.render_protocol_hash(settings)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        protocol_version=version,
        viewing_settings=settings,
        render_protocol_hash=render_hash,
    )
    for stimulus in manifest["stimuli"]:
        stimulus["render_protocol_hash"] = render_hash
    _write_json(manifest_path, manifest)
    bank_path = Path(manifest["calibration_bank"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank.update(viewing_settings=settings, render_protocol_hash=render_hash)
    _write_json(bank_path, bank)
    observation = _observation(9502, observation_id="observation:one")
    observation.update(viewing_settings=settings, render_protocol_hash=render_hash)
    observation["inspection_telemetry"] = {
        **_telemetry(),
        "final_camera_state": subjective._default_camera_state(settings),
    }
    observations_path.write_text(json.dumps(observation) + "\n", encoding="utf-8")
    playlist = _playlist(9502, trial_id="trial:one")
    playlist["format_version"] = version
    playlist_path.write_text(json.dumps(playlist) + "\n", encoding="utf-8")

    # Register only this test's implementation; production registrations stay immutable.
    monkeypatch.setattr(subjective_v3, "_HTML", "archived UI differs from current UI")
    hashes = protocol_freeze._implementation_hashes(version)
    monkeypatch.setitem(
        protocol_freeze._FROZEN_RENDER_IMPLEMENTATIONS,
        render_hash,
        {key: hashes[key] for key in ("browser_ui_bundle_sha256", "motion_payload_source_sha256")},
    )
    monkeypatch.setitem(
        protocol_freeze._FROZEN_COLLECTION_IMPLEMENTATIONS,
        version,
        {
            key: hashes[key]
            for key in ("collection_server_source_sha256", "public_trial_source_sha256")
        },
    )
    frozen = protocol_freeze.freeze_active_evaluation_protocol(manifest_path)
    contract = frozen["contract"]
    assert contract["implementation_hashes"] == hashes
    assert contract["observation_schema"]["served_playlist"]["format_version"] == version
    assert contract["camera_and_playback"]["pan"] == settings["pan"]
    assert (
        protocol_freeze.validate_active_evaluation_protocol(
            manifest_path.parent / "protocol_freeze.json"
        )["status"]
        == "valid"
    )

    with playlist_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_playlist(9502, trial_id="trial:two")) + "\n")
    with pytest.raises(ValueError, match="wrong schema"):
        protocol_freeze.validate_active_evaluation_protocol(
            manifest_path.parent / "protocol_freeze.json"
        )


def test_historical_server_dispatches_to_archived_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, observations_path, playlist_path = _active_pilot(tmp_path)
    calls: list[tuple[object, ...]] = []

    def archived_server(*args: object, **kwargs: object) -> None:
        calls.append((*args, kwargs))

    monkeypatch.setattr(subjective_v3, "serve_subjective_evaluator", archived_server)
    pair_root = tmp_path / "pair_dataset"
    subjective.serve_subjective_evaluator(
        pair_root,
        manifest_path,
        observations_path,
        served_playlist_path=playlist_path,
        port=9876,
    )

    assert calls == [
        (
            pair_root,
            manifest_path,
            observations_path,
            {"served_playlist_path": playlist_path, "host": "127.0.0.1", "port": 9876},
        )
    ]
