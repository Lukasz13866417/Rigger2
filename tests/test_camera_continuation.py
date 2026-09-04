from __future__ import annotations

import copy
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from motionlab.dataset.io import file_sha256
from motionlab.perceptual import camera_continuation, protocol_freeze, subjective
from motionlab.perceptual.observation_auth import validate_current_protocol_observations

_SOURCE = Path(__file__).resolve().parents[1] / "artifacts/phase15_mannequin_pilot"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_response(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    trial = manifest["sessions"][0]["trials"][0]
    stimuli = {item["stimulus_id"]: item for item in manifest["stimuli"]}
    session_id = subjective._canonical_hash(
        "session", {"rater": "continuation-test", "index": 0, "pilot_seed": manifest["seed"]}
    )
    row = {
        "format_version": subjective.RAW_OBSERVATION_VERSION,
        "observation_id": "observation:sha256:" + "a" * 64,
        "timestamp_utc": "2026-09-04T01:00:01+00:00",
        "rater_id": "continuation-test",
        "session_id": session_id,
        "session_index": 0,
        "trial_id": trial["trial_id"],
        "trial_index": trial["trial_index"],
        "playlist_seed": manifest["seed"],
        "rating": 4,
        "pair_outcome_display_order": None,
        "pair_outcome_canonical": None,
        "pair_id": trial.get("pair_id"),
        "variant_ids": [stimuli[value]["variant_id"] for value in trial["stimulus_ids"]],
        "render_protocol_hash": manifest["render_protocol_hash"],
        "anchor_status": "hidden_session_anchor" if trial["hidden_anchor"] else "not_anchor",
        "viewing_settings": manifest["viewing_settings"],
        "presented_side_classes": trial["side_classes"],
        "equivalence_control": bool(trial.get("equivalence_control", False)),
        "confidence": None,
        "reason_tags": [],
        "marked_interval_s": None,
        "judgment_difficulty": None,
        "response_time_ms": 2000.0,
        "replay_count": [0],
        "inspection_telemetry": {
            "format_version": subjective.INSPECTION_TELEMETRY_VERSION,
            "replay_count": [0],
            "speed_change_events": [],
            "camera_change_events": [],
            "zoom_change_count": 0,
            "view_change_count": 0,
            "overlay_change_count": 0,
            "total_inspection_time_ms": 1000.0,
            "final_playback_rate": 1.0,
            "final_camera_state": manifest["viewing_settings"]["default_camera"],
            "playback_wall_time_by_speed_ms": {
                f"{rate:g}": 1000.0 if rate == 1 else 0.0
                for rate in manifest["viewing_settings"]["playback_rates"]
            },
        },
    }
    for key in (
        "task_type",
        "stimulus_ids",
        "source_ids",
        "families",
        "styles",
        "render_hashes",
        "presentation_order",
        "comparison_mode",
        "hidden_repeat_group_ids",
        "view_mirror_x",
        "schedule_reason",
        "stimulus_populations",
        "measurement_cohorts",
        "critic_training_eligible",
    ):
        row[key] = trial[key]
    served = {
        "format_version": subjective.SUBJECTIVE_PROTOCOL_VERSION,
        "timestamp_utc": "2026-09-04T01:00:00+00:00",
        "seed": manifest["seed"],
        **{
            key: row[key]
            for key in (
                "rater_id",
                "session_id",
                "session_index",
                "trial_id",
                "trial_index",
                "task_type",
                "presentation_order",
                "schedule_reason",
            )
        },
    }
    return row, served


@pytest.fixture
def predecessor(tmp_path: Path) -> Path:
    if not (_SOURCE / "pilot_manifest.json").is_file():
        pytest.skip("prepared mannequin assets are unavailable")
    root = tmp_path / "original"
    source = json.loads((_SOURCE / "pilot_manifest.json").read_text())
    root.mkdir()
    # Copy immutable motion assets only. Never read the user's collected responses.
    for motion in _SOURCE.rglob("*.npz"):
        target = root / motion.relative_to(_SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(motion, target)
    bank = json.loads(Path(source["calibration_bank"]).read_text())
    bank_path = root / "calibration_bank/calibration_bank.json"
    bank["stimulus_directory"] = str(bank_path.parent)
    source["stimulus_directory"] = str(root)
    source["calibration_bank"] = str(bank_path)
    _write(bank_path, bank)
    pilot_path = root / "pilot_manifest.json"
    _write(pilot_path, source)
    response, served = _synthetic_response(source)
    validate_current_protocol_observations([response], source, [served])
    (root / "raw_observations.jsonl").write_text(json.dumps(response) + "\n")
    (root / "served_playlist.jsonl").write_text(json.dumps(served) + "\n")
    protocol_freeze.freeze_active_evaluation_protocol(pilot_path)
    return pilot_path


def test_continuation_preserves_assets_and_progress_without_fabricating_labels(
    predecessor: Path, tmp_path: Path
) -> None:
    original_files = {
        str(path): file_sha256(path) for path in predecessor.parent.rglob("*") if path.is_file()
    }
    original = json.loads(predecessor.read_text())
    result = camera_continuation.prepare_camera_continuation(predecessor, tmp_path / "new")
    manifest = json.loads(Path(result["pilot_manifest"]).read_text())
    assert manifest["protocol_version"] == subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION
    assert manifest["viewing_settings"] == subjective.FREE_CAMERA_VIEWING_SETTINGS
    assert manifest["seed"] == original["seed"]
    assert result["new_human_observations_created"] == 0
    assert not (tmp_path / "new/raw_observations.jsonl").exists()
    assert not (tmp_path / "new/served_playlist.jsonl").exists()
    assert original_files == {
        str(path): file_sha256(path) for path in predecessor.parent.rglob("*") if path.is_file()
    }
    for before, after in zip(original["stimuli"], manifest["stimuli"], strict=True):
        assert before["stimulus_id"] != after["stimulus_id"]
        assert before["render_hash"] != after["render_hash"]
        assert before["motion_content_hash"] == after["motion_content_hash"]
        assert before["cycle_window"] == after["cycle_window"]
        assert file_sha256(predecessor.parent / before["motion"]) == file_sha256(
            tmp_path / "new" / after["motion"]
        )
    history = camera_continuation.load_camera_continuation_history(manifest)
    assert len(history) == 1
    assert history[0]["_scheduling_only"] is True
    assert "observation_id" not in history[0]
    assert "render_protocol_hash" not in history[0]
    assert history[0]["trial_id"] == manifest["sessions"][0]["trials"][0]["trial_id"]
    assert history[0]["stimulus_ids"] == manifest["sessions"][0]["trials"][0]["stimulus_ids"]
    assert history[0]["source_trial_id"] == original["sessions"][0]["trials"][0]["trial_id"]
    for old_session, new_session in zip(original["sessions"], manifest["sessions"], strict=True):
        assert old_session["trial_count"] == new_session["trial_count"]
        for before, after in zip(old_session["trials"], new_session["trials"], strict=True):
            assert before["trial_index"] == after["trial_index"]
            assert before["view_mirror_x"] == after["view_mirror_x"]
            assert before["task_type"] == after["task_type"]
            assert before["trial_id"] != after["trial_id"]
    assert (
        original["calibration_tutorial"][0]["tutorial_id"]
        != manifest["calibration_tutorial"][0]["tutorial_id"]
    )
    bank = json.loads(Path(manifest["calibration_bank"]).read_text())
    bank_ids = {item["stimulus_id"] for item in bank["stimuli"]}
    assert all(set(ladder["stimulus_ids"]) <= bank_ids for ladder in bank["ladders"])
    assert bank["render_protocol_hash"] == manifest["render_protocol_hash"]
    # Tampering cannot manufacture completion progress, even if the original response
    # logs have legitimately acquired more rows after this immutable snapshot.
    history_path = Path(manifest["camera_continuation"]["history_path"])
    record = json.loads(history_path.read_text())
    record["scheduling_rows"][0]["trial_index"] = 999
    _write(history_path, record)
    with pytest.raises(ValueError, match="history hash drifted"):
        camera_continuation.load_camera_continuation_history(manifest)


def test_continuation_refuses_existing_output_and_source_descendant(
    predecessor: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="not already exist"):
        camera_continuation.prepare_camera_continuation(predecessor, tmp_path)
    with pytest.raises(ValueError, match="separate from the original"):
        camera_continuation.prepare_camera_continuation(predecessor, predecessor.parent / "new")


def test_continuation_rejects_unauthenticated_or_incomplete_evidence(
    predecessor: Path, tmp_path: Path
) -> None:
    log = predecessor.with_name("raw_observations.jsonl")
    original = log.read_text()
    row = json.loads(original)
    row["observation_id"] = "observation:sha256:" + "b" * 64
    row["trial_id"] = "unknown-trial"
    row["trial_index"] = 999
    log.write_text(original + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unknown pilot trial"):
        camera_continuation.prepare_camera_continuation(predecessor, tmp_path / "unserved")
    assert not (tmp_path / "unserved").exists()
    log.write_text(original + '{"incomplete":')
    with pytest.raises((ValueError, RuntimeError)):
        camera_continuation.prepare_camera_continuation(predecessor, tmp_path / "partial")
    assert not (tmp_path / "partial").exists()


def test_history_is_absent_for_original_manifest_and_rejects_path_escape() -> None:
    assert camera_continuation.load_camera_continuation_history({}) == []
    manifest = {
        "protocol_version": subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION,
        "stimulus_directory": "/tmp/camera-history-test",
        "camera_continuation": {"history_path": "../outside.json"},
    }
    with pytest.raises(ValueError, match="escapes"):
        camera_continuation.load_camera_continuation_history(copy.deepcopy(manifest))


def test_frozen_continuation_server_resumes_only_the_authenticated_rater(
    predecessor: Path, tmp_path: Path
) -> None:
    result = camera_continuation.prepare_camera_continuation(predecessor, tmp_path / "resume")
    pilot = Path(result["pilot_manifest"])
    manifest = json.loads(pilot.read_text())
    protocol_freeze.freeze_active_evaluation_protocol(pilot)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from motionlab.perceptual.subjective import serve_subjective_evaluator; "
            "serve_subjective_evaluator(Path(sys.argv[1]), Path(sys.argv[2]), "
            "Path(sys.argv[3]), port=int(sys.argv[4]))",
            str(manifest["pair_dataset_directory"]),
            str(pilot),
            str(pilot.with_name("raw_observations.jsonl")),
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                pytest.fail(f"continuation server failed: {process.communicate()[0]}")
            try:
                with urlopen(base_url, timeout=0.5):
                    break
            except URLError:
                if time.monotonic() >= deadline:
                    pytest.fail("continuation server did not become ready")
                time.sleep(0.05)

        def next_trial(rater: str) -> dict[str, Any]:
            request = Request(
                base_url + "/api/session",
                data=json.dumps({"rater_id": rater, "session_index": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=5) as response:
                token = json.load(response)["session_token"]
            with urlopen(base_url + f"/api/next?session_token={token}", timeout=5) as response:
                return dict(json.load(response)["trial"])

        resumed = next_trial("continuation-test")
        assert resumed["trial_id"] == manifest["sessions"][0]["trials"][1]["trial_id"]
        assert resumed["is_tutorial"] is False
        new_rater = next_trial("unseen-rater")
        assert new_rater["trial_id"] == manifest["calibration_tutorial"][0]["tutorial_id"]
        assert new_rater["is_tutorial"] is True
        assert not pilot.with_name("raw_observations.jsonl").exists()
        served = [
            json.loads(line)
            for line in pilot.with_name("served_playlist.jsonl").read_text().splitlines()
        ]
        assert len(served) == 2
        assert all(row["format_version"] == manifest["protocol_version"] for row in served)
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
