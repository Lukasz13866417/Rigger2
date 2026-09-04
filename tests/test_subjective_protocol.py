from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import motionlab.perceptual.subjective as subjective
import motionlab.perceptual.subjective_v3 as subjective_v3
from motionlab.perceptual.subjective import (
    PILOT_MANIFEST_VERSION,
    RAW_OBSERVATION_VERSION,
    _canonical_pair_outcome,
    _public_trial,
    prepare_subjective_pilot,
)
from motionlab.perceptual.subjective_analysis import analyze_subjective_pilot
from motionlab.testing.synthetic import make_synthetic_walk


def _inspection_telemetry(
    replay_count: list[int],
    *,
    settings: dict[str, object] | None = None,
) -> dict[str, object]:
    viewing = subjective.VIEWING_SETTINGS if settings is None else settings
    rates = [float(rate) for rate in viewing.get("playback_rates", [1.0])]  # type: ignore[union-attr]
    return {
        "format_version": subjective.INSPECTION_TELEMETRY_VERSION,
        "replay_count": replay_count,
        "speed_change_events": [],
        "playback_wall_time_by_speed_ms": {f"{rate:g}": 0.0 for rate in rates},
        "camera_change_events": [],
        "zoom_change_count": 0,
        "view_change_count": 0,
        "overlay_change_count": 0,
        "total_inspection_time_ms": 0.0,
        "final_playback_rate": 1.0,
        "final_camera_state": subjective._default_camera_state(viewing),
    }


@contextmanager
def _running_protocol_server(
    *,
    monkeypatch: pytest.MonkeyPatch,
    pair_root: Path,
    manifest_path: Path,
    observations_path: Path,
    served_path: Path,
) -> Iterator[int]:
    server_ready = threading.Event()
    servers: list[subjective_v3.ThreadingHTTPServer] = []
    server_class = subjective_v3.ThreadingHTTPServer

    class CapturingServer(server_class):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            servers.append(self)
            server_ready.set()

    # The public entrypoint dispatches historical v3 collection to its frozen module.
    monkeypatch.setattr(subjective_v3, "ThreadingHTTPServer", CapturingServer)
    server_errors: list[BaseException] = []

    def run_server() -> None:
        try:
            subjective.serve_subjective_evaluator(
                pair_root,
                manifest_path,
                observations_path,
                served_playlist_path=served_path,
                port=0,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            server_errors.append(exc)
            server_ready.set()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    assert server_ready.wait(timeout=5)
    assert servers and not server_errors
    try:
        yield int(servers[0].server_address[1])
    finally:
        servers[0].shutdown()
        server_thread.join(timeout=5)


def _post_json(port: int, path: str, payload: dict[str, object]) -> tuple[int, str]:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def _get_json(port: int, path: str) -> tuple[int, str]:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def _pair(index: int) -> dict[str, object]:
    source = f"source-{index % 4}"
    family = (
        "excessive_smoothing",
        "excessive_rigidification",
        "asymmetric_limb_timing",
        "equivalent_origin_shift",
        "torso_counter_rotation_mismatch",
    )[index % 5]
    return {
        "pair_id": f"pair-{index}",
        "motion_a": f"motions/clean-{index % 4}.npz",
        "motion_b": f"motions/candidate-{index}.npz",
        "source_clip_id": source,
        "perturbation_mechanism": family,
        "preference": "approximately_equal" if family == "equivalent_origin_shift" else "a_better",
        "adversarial_origin": index in {0, 1},
    }


def test_render_protocol_versions_preserve_legacy_identity() -> None:
    legacy_hash = subjective.render_protocol_hash(subjective.LEGACY_SKELETON_VIEWING_SETTINGS)
    assert legacy_hash == (
        "render:sha256:68b9052a8345ebafc8dc761d02bd4ff54c37f675060a30231e51fa84fcbf81e5"
    )
    assert subjective.render_protocol_hash() != legacy_hash
    motion = "motions/example.npz"
    assert subjective._public_stimulus_id(
        motion,
        subjective.LEGACY_SKELETON_VIEWING_SETTINGS,
    ) != subjective._public_stimulus_id(motion)


def test_mannequin_payload_includes_orientation_and_semantic_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motion = make_synthetic_walk(num_frames=121)
    monkeypatch.setattr(subjective, "load_motion_npz", lambda _path: motion)
    monkeypatch.setattr(subjective, "detect_foot_contacts", lambda _motion: object())
    monkeypatch.setattr(
        subjective,
        "estimate_gait_phase",
        lambda _contacts: SimpleNamespace(
            phase_rad=np.linspace(0.0, 4.0 * np.pi, motion.num_frames)
        ),
    )
    payload = subjective._two_cycle_payload(Path("unused.npz"))
    assert payload["renderer_protocol"] == "motionlab.fixed_mannequin_canvas.v1"
    assert payload["roles"]["pelvis"] == motion.skeleton.roles["pelvis"]
    assert payload["rest_offsets_m"] == motion.skeleton.rest_offsets_m.round(5).tolist()
    assert np.asarray(payload["positions"]).shape[-2:] == (
        motion.skeleton.num_joints,
        3,
    )
    rotations = np.asarray(payload["global_quat_wxyz"])
    assert rotations.shape[-2:] == (motion.skeleton.num_joints, 4)
    assert np.allclose(np.linalg.norm(rotations, axis=-1), 1.0, atol=1.0e-4)


def test_pilot_preparation_refuses_collected_or_paused_directory(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    output.mkdir()
    (output / "raw_observations.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collected or paused"):
        prepare_subjective_pilot(tmp_path / "pairs", tmp_path / "audit", output)


def test_legacy_pilot_builder_cannot_emit_current_mannequin_protocol(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="legacy pilot builder only supports"):
        prepare_subjective_pilot(
            tmp_path / "pairs",
            tmp_path / "audit",
            tmp_path / "pilot",
            viewing_settings=subjective.MANNEQUIN_VIEWING_SETTINGS,
        )


def test_paused_pilot_cannot_be_served(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot_manifest.json"
    (tmp_path / "PILOT_PAUSED.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collection for this pilot is paused"):
        subjective.serve_subjective_evaluator(
            tmp_path / "pairs",
            manifest,
            tmp_path / "observations.jsonl",
            port=0,
        )


def test_tutorial_completion_bypasses_scored_playback_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_root = tmp_path / "pairs"
    pair_root.mkdir()
    tutorial_stimulus_id = "tutorial-stimulus"
    scored_stimulus_id = "scored-stimulus"
    scored_trial_id = "scored-trial"
    manifest = {
        "format_version": PILOT_MANIFEST_VERSION,
        "protocol_version": subjective.SUBJECTIVE_PROTOCOL_VERSION,
        "pair_dataset_directory": str(pair_root),
        "seed": 23,
        "render_protocol_hash": subjective.render_protocol_hash(),
        "viewing_settings": subjective.MANNEQUIN_VIEWING_SETTINGS,
        "scale_anchors": {
            "naturalness": [str(value) for value in range(1, 8)],
        },
        "hidden_anchor_stimulus_ids": [],
        "adaptive_policy": {"enabled": False},
        "stimuli": [
            {"stimulus_id": tutorial_stimulus_id},
            {"stimulus_id": scored_stimulus_id},
        ],
        "calibration_tutorial": [
            {
                "tutorial_id": "tutorial-trial",
                "tutorial_index": 0,
                "task_type": "calibration",
                "presentation_order": [tutorial_stimulus_id],
                "tutorial_disclosure": {
                    "calibration_family": "foot_slide",
                    "ordered_severities": ["clean"],
                    "instruction": "Inspect the calibration example.",
                },
            }
        ],
        "sessions": [
            {
                "trials": [
                    {
                        "trial_id": scored_trial_id,
                        "trial_index": 0,
                        "task_type": "naturalness",
                        "comparison_mode": None,
                        "presentation_order": [scored_stimulus_id],
                        "schedule_reason": "test",
                        "seed": 23,
                    }
                ]
            }
        ],
    }
    manifest_path = tmp_path / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observations_path = tmp_path / "observations.jsonl"
    served_path = tmp_path / "served.jsonl"

    with _running_protocol_server(
        monkeypatch=monkeypatch,
        pair_root=pair_root,
        manifest_path=manifest_path,
        observations_path=observations_path,
        served_path=served_path,
    ) as port:
        status, body = _post_json(
            port,
            "/api/session",
            {"rater_id": "tutorial-gate-rater", "session_index": 0},
        )
        assert status == 200
        token = json.loads(body)["session_token"]

        status, body = _get_json(port, f"/api/next?session_token={token}")
        assert status == 200
        assert json.loads(body)["trial"]["is_tutorial"] is True

        status, body = _post_json(
            port,
            "/api/tutorial-complete",
            {"session_token": token, "trial_id": "tutorial-trial"},
        )
        assert status == 200, body

        status, body = _get_json(port, f"/api/next?session_token={token}")
        assert status == 200
        assert json.loads(body)["trial"]["trial_id"] == scored_trial_id

        status, body = _post_json(
            port,
            "/api/observation",
            {
                "session_token": token,
                "trial_id": scored_trial_id,
                "response": 4,
            },
        )
        assert status == 400
        assert "complete every required full-speed playback" in body
        assert not observations_path.exists()


def test_response_adaptive_toggle_pool_selects_unserved_close_pair() -> None:
    def template(pair_id: str, first: str, second: str, family: str) -> dict[str, object]:
        return {
            "trial_id": f"trial-{pair_id}",
            "task_type": "pair_comparison",
            "comparison_mode": "same_viewport_toggle",
            "eligible_session_indices": [0],
            "pair_id": pair_id,
            "stimulus_ids": [first, second],
            "presentation_order": [first, second],
            "families": [family],
            "hidden_repeat_group_ids": [f"repeat-{pair_id}"],
        }

    manifest = {
        "seed": 17,
        "adaptive_policy": {
            "enabled": True,
            "selection_mode": "response_adaptive_toggle_pool",
            "maximum_followups_per_session": 6,
            "required_comparison_mode": "same_viewport_toggle",
        },
        "adaptive_comparison_pool": [
            template("already-used", "a", "b", "common"),
            template("close", "a", "b", "rare"),
            template("far", "a", "c", "common"),
        ],
        "sessions": [{"trials": []}],
    }
    observations = [
        {
            "rater_id": "rater",
            "session_index": 0,
            "task_type": "pair_comparison",
            "pair_id": "already-used",
            "schedule_reason": "base",
            "families": ["common"],
        },
        {
            "rater_id": "rater",
            "session_index": 0,
            "task_type": "naturalness",
            "rating": 5,
            "stimulus_ids": ["a"],
            "hidden_repeat_group_ids": ["repeat-a"],
            "families": ["common"],
            "schedule_reason": "base",
        },
        {
            "rater_id": "rater",
            "session_index": 0,
            "task_type": "naturalness",
            "rating": 5,
            "stimulus_ids": ["b"],
            "hidden_repeat_group_ids": ["repeat-b"],
            "families": ["common"],
            "schedule_reason": "base",
        },
        {
            "rater_id": "rater",
            "session_index": 0,
            "task_type": "naturalness",
            "rating": 1,
            "stimulus_ids": ["c"],
            "hidden_repeat_group_ids": ["repeat-c"],
            "families": ["common"],
            "schedule_reason": "base",
        },
    ]
    selected = subjective._adaptive_candidate(
        manifest,
        observations,
        rater_id="rater",
        session_index=0,
    )
    assert selected is not None
    assert selected["pair_id"] == "close"
    assert selected["comparison_mode"] == "same_viewport_toggle"
    assert selected["schedule_reason"].startswith("adaptive:response_adaptive_toggle_pool")


def test_inspection_telemetry_counts_must_match_events() -> None:
    telemetry = _inspection_telemetry([0])
    telemetry["zoom_change_count"] = 1
    with pytest.raises(ValueError, match="counts do not match"):
        subjective._validate_inspection_telemetry(
            telemetry,
            stimulus_count=1,
            viewing_settings=subjective.MANNEQUIN_VIEWING_SETTINGS,
        )


def test_pilot_is_blinded_balanced_and_repeat_spaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = [_pair(index) for index in range(40)]
    monkeypatch.setattr("motionlab.perceptual.subjective.load_perceptual_pairs", lambda _: pairs)
    monkeypatch.setattr(
        "motionlab.perceptual.subjective._two_cycle_payload",
        lambda path, **_: {
            "render_hash": f"rendered:{path.name}",
            "cycle_window": {
                "target_cycles": 2.0,
                "selection_method": "detected_phase_exact_two_cycles",
            },
        },
    )
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        "".join(
            json.dumps(
                {
                    "pair_id": row["pair_id"],
                    "difficulty": "clear" if index < 20 else "subtle",
                    "selection_reasons": (
                        ["confident_heldout_source_misranking"] if index < 6 else []
                    ),
                }
            )
            + "\n"
            for index, row in enumerate(pairs)
        ),
        encoding="utf-8",
    )
    summary = prepare_subjective_pilot(
        tmp_path / "pairs",
        audit,
        tmp_path / "pilot",
        unique_stimulus_count=30,
    )
    manifest = json.loads((tmp_path / "pilot" / "pilot_manifest.json").read_text())
    assert summary["unique_animation_count"] == 30
    assert manifest["format_version"] == subjective.LEGACY_PILOT_MANIFEST_VERSION
    assert manifest["protocol_version"] == subjective.LEGACY_SUBJECTIVE_PROTOCOL_VERSION
    assert manifest["viewing_settings"] == subjective.LEGACY_SKELETON_VIEWING_SETTINGS
    assert manifest["render_protocol_hash"] == subjective.render_protocol_hash(
        subjective.LEGACY_SKELETON_VIEWING_SETTINGS
    )
    assert manifest["scale_anchors"]["naturalness"] == [
        "Clearly broken / highly unnatural",
        "Very unnatural",
        "Noticeably unnatural",
        "Acceptable but visibly mediocre",
        "Good and mostly natural",
        "Very good with only subtle issues",
        "Excellent / difficult to improve",
    ]
    assert len(manifest["hidden_anchor_stimulus_ids"]) == 6
    assert all(session["trial_count"] >= 25 for session in manifest["sessions"])
    by_stimulus = {row["stimulus_id"]: row for row in manifest["stimuli"]}
    comparison_trials = [
        trial
        for session in manifest["sessions"]
        for trial in session["trials"]
        if trial["task_type"] == "pair_comparison"
    ]
    assert (
        sum(
            by_stimulus[trial["presentation_order"][0]]["variant_kind"] == "clean_reference"
            for trial in comparison_trials
        )
        == 6
    )
    assert (
        abs(
            manifest["presented_side_counts"].get("left", 0)
            - manifest["presented_side_counts"].get("right", 0)
        )
        <= 1
    )
    chronological_singles = [
        trial
        for session in manifest["sessions"]
        for trial in session["trials"]
        if trial["task_type"] != "pair_comparison"
    ]
    occurrence_by_group: dict[str, list[int]] = defaultdict(list)
    for trial in chronological_singles:
        occurrence_by_group[trial["hidden_repeat_group_ids"][0]].append(
            trial["repeat_occurrence_index"]
        )
    assert all(values == list(range(len(values))) for values in occurrence_by_group.values())
    for session in manifest["sessions"]:
        positions: dict[str, list[int]] = defaultdict(list)
        for index, trial in enumerate(session["trials"]):
            for group in trial["hidden_repeat_group_ids"]:
                positions[group].append(index)
        assert all(
            right - left >= 12 for values in positions.values() for left, right in pairwise(values)
        )
        assert (
            Counter(trial["comparison_mode"] for trial in session["trials"])["same_viewport_toggle"]
            == 3
        )
        assert sum(trial["hidden_anchor"] for trial in session["trials"]) == 6

    private = manifest["sessions"][0]["trials"][0]
    public = _public_trial(private, manifest)
    encoded = json.dumps(public)
    assert "source_ids" not in public
    assert "families" not in public
    assert "hidden_anchor" not in public
    assert "Proud" not in encoded


def test_pair_outcome_is_stored_in_canonical_order() -> None:
    canonical = ["clean", "candidate"]
    assert _canonical_pair_outcome("a_better", canonical, canonical) == "first_better"
    assert (
        _canonical_pair_outcome("a_better", canonical, list(reversed(canonical))) == "second_better"
    )
    assert _canonical_pair_outcome("not_sure", canonical, canonical) == "not_sure"


def test_concurrent_duplicate_observation_is_committed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_root = tmp_path / "pairs"
    pair_root.mkdir()
    observations_path = tmp_path / "observations.jsonl"
    served_path = tmp_path / "served.jsonl"
    stimulus_id = "stimulus-1"
    trial_id = "trial-1"
    manifest = {
        "format_version": PILOT_MANIFEST_VERSION,
        "protocol_version": subjective.SUBJECTIVE_PROTOCOL_VERSION,
        "pair_dataset_directory": str(pair_root),
        "seed": 91,
        "render_protocol_hash": subjective.render_protocol_hash(),
        "viewing_settings": subjective.VIEWING_SETTINGS,
        "scale_anchors": {"naturalness": [str(value) for value in range(1, 8)]},
        "hidden_anchor_stimulus_ids": [],
        "adaptive_policy": {"enabled": False},
        "stimuli": [
            {
                "stimulus_id": stimulus_id,
                "motion": "motions/stimulus.npz",
                "phase_reference_motion": "motions/reference.npz",
                "render_hash": "render-1",
                "variant_id": "variant-1",
            }
        ],
        "sessions": [
            {
                "trials": [
                    {
                        "trial_id": trial_id,
                        "trial_index": 0,
                        "task_type": "naturalness",
                        "stimulus_ids": [stimulus_id],
                        "source_ids": ["source-1"],
                        "families": ["clean_reference"],
                        "styles": ["neutral"],
                        "side_classes": ["bilateral"],
                        "hidden_repeat_group_ids": ["repeat-1"],
                        "hidden_anchor": False,
                        "comparison_mode": None,
                        "presentation_order": [stimulus_id],
                        "view_mirror_x": False,
                        "render_hashes": ["render-1"],
                        "pair_id": "pair-1",
                        "schedule_reason": "test",
                        "session_index": 0,
                        "seed": 91,
                    }
                ]
            }
        ],
    }
    first_trial = manifest["sessions"][0]["trials"][0]
    manifest["sessions"][0]["trials"].append(
        {
            **first_trial,
            "trial_id": "trial-2",
            "trial_index": 1,
            "hidden_repeat_group_ids": ["repeat-2"],
            "pair_id": "pair-2",
        }
    )
    manifest_path = tmp_path / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        subjective_v3,
        "_two_cycle_payload",
        lambda *_args, **_kwargs: {
            "duration_s": 0.0,
            "render_hash": "render-1",
        },
    )
    original_append = subjective_v3._append_jsonl

    def delayed_observation_append(path: Path, value: dict[str, object]) -> None:
        if Path(path) == observations_path:
            time.sleep(0.15)
        original_append(path, value)

    monkeypatch.setattr(subjective_v3, "_append_jsonl", delayed_observation_append)

    server_ready = threading.Event()
    servers: list[subjective_v3.ThreadingHTTPServer] = []
    server_class = subjective_v3.ThreadingHTTPServer

    class CapturingServer(server_class):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            servers.append(self)
            server_ready.set()

    monkeypatch.setattr(subjective_v3, "ThreadingHTTPServer", CapturingServer)
    server_errors: list[BaseException] = []

    def run_server() -> None:
        try:
            subjective.serve_subjective_evaluator(
                pair_root,
                manifest_path,
                observations_path,
                served_playlist_path=served_path,
                port=0,
            )
        except BaseException as exc:  # pragma: no cover - reported by the assertion below
            server_errors.append(exc)
            server_ready.set()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    assert server_ready.wait(timeout=5)
    assert servers and not server_errors
    port = int(servers[0].server_address[1])

    def post(path: str, payload: dict[str, object]) -> tuple[int, str]:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(
                "POST",
                path,
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def get(path: str) -> tuple[int, str]:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    try:
        session_status, session_body = post(
            "/api/session",
            {"rater_id": "concurrent-rater", "session_index": 0},
        )
        assert session_status == 200
        token = json.loads(session_body)["session_token"]
        next_status, next_body = get(f"/api/next?session_token={token}")
        assert next_status == 200
        assert json.loads(next_body)["trial"]["trial_id"] == trial_id
        playback = {
            "session_token": token,
            "trial_id": trial_id,
            "stimulus_id": stimulus_id,
            "playback_rate": 1.0,
            "camera_state": subjective._default_camera_state(subjective.VIEWING_SETTINGS),
        }
        assert (
            post(
                "/api/playback-start",
                {**playback, "playback_rate": 0.5},
            )[0]
            == 400
        )
        assert (
            post(
                "/api/playback-start",
                {
                    **playback,
                    "camera_state": {
                        **playback["camera_state"],
                        "zoom": 1.15,
                    },
                },
            )[0]
            == 400
        )
        assert post("/api/playback-start", playback)[0] == 200
        assert post("/api/playback-complete", playback)[0] == 200
        observation = {
            "session_token": token,
            "trial_id": trial_id,
            "response": 4,
            "confidence": 3,
            "reason_tags": [],
            "replay_count": [0],
            "marked_interval_s": None,
            "judgment_difficulty": None,
            "inspection_telemetry": _inspection_telemetry([0]),
        }
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: post("/api/observation", observation), range(8)))
        served_rows = [json.loads(line) for line in served_path.read_text().splitlines()]
        assert len(served_rows) == 1
        next_status, next_body = get(f"/api/next?session_token={token}")
        assert next_status == 200
        assert json.loads(next_body)["trial"]["trial_id"] == "trial-2"
        (manifest_path.with_name("PILOT_PAUSED.json")).write_text(
            "{}\n",
            encoding="utf-8",
        )
        paused_status, paused_body = get(f"/api/next?session_token={token}")
        assert paused_status == 400
        assert "running server will not serve or record trials" in paused_body
    finally:
        servers[0].shutdown()
        server_thread.join(timeout=5)

    statuses = [status for status, _ in results]
    assert statuses.count(200) == 1
    assert statuses.count(400) == 7
    rows = [json.loads(line) for line in observations_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["trial_id"] == trial_id
    assert not server_errors


def test_analysis_keeps_raw_judgments_and_latent_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stimuli = [
        {
            "stimulus_id": f"stimulus-{index}",
            "source_id": f"source-{index % 2}",
            "family": "clean_reference" if index < 2 else "test_family",
        }
        for index in range(4)
    ]
    manifest = {
        "format_version": PILOT_MANIFEST_VERSION,
        "render_protocol_hash": "render:test",
        "stimuli": stimuli,
    }
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observations = []
    for index in range(12):
        stimulus = stimuli[index % 4]
        observations.append(
            {
                "format_version": RAW_OBSERVATION_VERSION,
                "observation_id": f"observation-{index}",
                "render_protocol_hash": "render:test",
                "task_type": "naturalness",
                "rating": 3 + (index % 4),
                "pair_outcome_canonical": None,
                "stimulus_ids": [stimulus["stimulus_id"]],
                "source_ids": [stimulus["source_id"]],
                "rater_id": f"rater-{index % 2}",
                "session_id": f"session-{index % 2}",
                "session_index": index % 2,
                "trial_id": f"trial-{index}",
                "trial_index": index,
                "timestamp_utc": f"2026-01-01T00:00:{index:02d}+00:00",
                "hidden_repeat_group_ids": [f"repeat-{index % 4}"],
                "anchor_status": ("hidden_session_anchor" if index % 4 == 0 else "not_anchor"),
                "confidence": 3,
                "reason_tags": [],
                "families": [stimulus["family"]],
                "styles": [stimulus["source_id"]],
                "response_time_ms": 3000 + index,
                "replay_count": [0],
                "comparison_mode": None,
                "equivalence_control": False,
            }
        )
    observations_path = tmp_path / "raw.jsonl"
    observations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in observations),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "motionlab.perceptual.subjective_analysis.load_perceptual_pairs",
        lambda _: [],
    )
    report = analyze_subjective_pilot(
        tmp_path / "pairs",
        manifest_path,
        observations_path,
        tmp_path / "analysis",
    )
    latent = json.loads((tmp_path / "analysis" / "latent_quality.json").read_text())
    evidence = [
        json.loads(line)
        for line in (tmp_path / "analysis" / "training_evidence.jsonl").read_text().splitlines()
    ]
    assert report["status"] == "pilot_in_progress_analysis_available"
    assert report["pilot_completion"]["completion_contract_available"] is False
    assert not report["raw_mean_used_as_canonical_score"]
    assert latent["joint_single_and_pair_fit"]
    assert all(row["quality_95_percent_interval"] for row in latent["item_quality"])
    assert evidence == []
    assert report["training_eligibility"]["eligible_observation_count"] == 0
    assert report["training_eligibility"]["excluded_observation_counts_by_population"] == {
        "LEGACY_UNCLASSIFIED": 12
    }
    assert not report["critic_retraining_started"]
