from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from playwright.sync_api import expect, sync_playwright
from test_subjective_ui import _launch_browser

from motionlab.critic.model import FixedRigFlatTCN, FixedRigTCNConfig
from motionlab.dataset.io import file_sha256
from motionlab.io.npz import save_motion_npz
from motionlab.noticeability import migration
from motionlab.noticeability.analysis import analyze, detector_metrics, latent_detectability
from motionlab.noticeability.legacy import read_registry
from motionlab.noticeability.model import (
    FixedRigNoticeability,
    choose_operating_point,
    clip_noticeability_loss,
    production_noticeability,
)
from motionlab.noticeability.pilot import build_staircase_pilot
from motionlab.noticeability.protocol import (
    PROTOCOL,
    PrimaryObservation,
    append_event,
    authenticated_observations,
    freeze_pilot,
    identity,
    read_json,
    read_rows,
    validate_pilot,
    write_json,
)
from motionlab.noticeability.server import make_server
from motionlab.noticeability.staircase import select_interleaved, staircase_step
from motionlab.noticeability.workflow import export_training, fixed_validation_plan
from motionlab.perceptual import subjective
from motionlab.testing.synthetic import make_synthetic_walk


@pytest.fixture(params=["mannequin_v1", "free_camera_v2"])
def pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Path:
    motion = make_synthetic_walk(num_frames=61)
    save_motion_npz(tmp_path / "motion.npz", motion)
    monkeypatch.setattr(subjective, "detect_foot_contacts", lambda _: object())
    monkeypatch.setattr(
        subjective,
        "estimate_gait_phase",
        lambda _: SimpleNamespace(
            phase_rad=np.linspace(0, 4 * np.pi, motion.num_frames),
        ),
    )
    settings = (
        subjective.MANNEQUIN_VIEWING_SETTINGS
        if request.param == "mannequin_v1"
        else subjective.FREE_CAMERA_VIEWING_SETTINGS
    )
    payload = subjective._two_cycle_payload(tmp_path / "motion.npz", viewing_settings=settings)
    item = {
        "stimulus_id": "stimulus_test",
        "variant_id": "test_variant",
        "source_id": "source_test",
        "render_hash": subjective._canonical_hash(
            "presented-render", {"base_render_hash": payload["render_hash"], "mirror_x": False}
        ),
        "base_render_hash": payload["render_hash"],
        "render_protocol_hash": payload["render_protocol_hash"],
        "motion": "motion.npz",
        "phase_reference_motion": "motion.npz",
        "viewing_settings": settings,
        "view_mirror_x": False,
        "duration_s": payload["duration_s"],
        "fps": payload["fps"],
        "population": "CLEAN_SHAM_CONTROL",
        "family": "clean_reference",
        "style": "Neutral",
        "evaluation_goal": "Notice unintended motion problems",
        "style_context": "Ordinary walking",
        "physical_measurements": {},
        "cycle_window": payload["cycle_window"],
    }
    trials = [
        {
            "trial_id": f"t_{i}",
            "trial_index": i,
            "stimulus_id": item["stimulus_id"],
            "prior_exposure": i > 0,
        }
        for i in range(3)
    ]
    manifest = {"protocol_version": PROTOCOL, "stimuli": [item], "trials": trials}
    write_json(tmp_path / "pilot_manifest.json", manifest)
    freeze_pilot(tmp_path / "pilot_manifest.json")
    return tmp_path / "pilot_manifest.json"


@contextmanager
def running(pilot: Path) -> Iterator[str]:
    server = make_server(pilot, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def direct_fixture(pilot: Path, notice: str = "YES") -> dict[str, Any]:
    manifest = read_json(pilot)
    item = manifest["stimuli"][0]
    timestamp = datetime.now(UTC)
    for event, dt in [("served", -4.0), ("started", -3.0), ("completed", -0.5)]:
        row = {
            "event_type": event,
            "rater_id": "test",
            "session_id": "s1",
            "trial_id": "t_0",
            "trial_index": 0,
            "timestamp_utc": (timestamp + timedelta(seconds=dt)).isoformat(),
        }
        row["serve_id"] = identity("notice-serve", row)
        append_event(pilot.parent / "served_playlist.jsonl", row)
    row = PrimaryObservation(
        observation_id="pending",
        **{
            k: item[k]
            for k in (
                "stimulus_id",
                "variant_id",
                "source_id",
                "render_hash",
                "render_protocol_hash",
                "evaluation_goal",
                "style_context",
            )
        },
        rater_id="test",
        session_id="s1",
        trial_id="t_0",
        trial_index=0,
        spontaneous_notice=notice,
        spontaneous_response_time_ms=300.0,
        initial_presentation_seconds=item["duration_s"],
        prior_exposure=False,
        timestamp_utc=timestamp.isoformat(),
    ).model_dump(exclude={"observation_id"})
    return append_event(pilot.parent / "observations.jsonl", row)


def test_frozen_assets_and_immutable_primary(pilot: Path) -> None:
    direct_fixture(pilot)
    rows = authenticated_observations(pilot)
    assert len(rows) == 1 and rows[0]["spontaneous_notice"] == "YES"
    assert rows[0]["inspected_notice"] is None
    primary = read_rows(pilot.parent / "observations.jsonl")[0]
    with pytest.raises(ValueError):
        PrimaryObservation.model_validate({**primary, "replay_count": 1})
    with pytest.raises(ValueError):
        PrimaryObservation.model_validate({**primary, "optional_reason_tags": ["foot/contact"]})
    assert freeze_pilot(pilot) == read_json(pilot.parent / "protocol_freeze.json")
    (pilot.parent / "motion.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset changed"):
        validate_pilot(pilot)


def test_server_and_browser_primary_precedes_inspection(pilot: Path) -> None:
    with running(pilot) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1100, "height": 1200})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_visible()
        response = page.request.post(
            url + "/api/spontaneous", data={"trial_id": "t_0", "spontaneous_notice": "YES"}
        )
        assert response.status == 400
        response = page.request.post(
            url + "/api/inspection", data={"primary_observation_id": "fake"}
        )
        assert response.status == 400
        page.locator("#watch").click()
        expect(page.locator("#primary")).to_be_visible(timeout=10000)
        expect(page.locator("#inspection")).to_be_hidden()
        expect(page.locator("#view")).to_be_hidden()
        assert not (pilot.parent / "observations.jsonl").exists()
        page.locator('#primaryAnswers [data-notice="YES"]').click()
        expect(page.locator("#inspection")).to_be_visible()
        raw_before = (pilot.parent / "observations.jsonl").read_bytes()
        assert len(raw_before.splitlines()) == 1
        assert read_rows(pilot.parent / "observations.jsonl")[0]["spontaneous_notice"] == "YES"
        response = page.request.post(
            url + "/api/spontaneous", data={"trial_id": "t_0", "spontaneous_notice": "NO"}
        )
        assert response.status == 400
        page.locator("#replay").click()
        page.locator("#speed").select_option(".5")
        page.locator("#zoomOut").click()
        page.locator("#preset").select_option("side")
        if page.evaluate("'pan_x_m' in window.__noticeabilityState().camera"):
            page.locator("#view").scroll_into_view_if_needed()
            box = page.locator("#view").bounding_box()
            assert box is not None
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down(button="right")
            page.mouse.move(x + 40, y + 20)
            page.mouse.up(button="right")
            assert page.evaluate("window.__noticeabilityState().camera.pan_x_m") > 0
            page.mouse.wheel(0, 200)
            page.wait_for_function("window.__noticeabilityState().camera.zoom < 0.8")
        page.locator('#inspectedAnswers [data-notice="NO"]').click()
        page.locator("#next").click()
        expect(page.locator("#watch")).to_be_visible()
        raw_after = (pilot.parent / "observations.jsonl").read_bytes()
        assert raw_after.startswith(raw_before)
        rows = authenticated_observations(pilot)
        assert len(rows) == 1
        assert rows[0]["spontaneous_notice"] == "YES" and rows[0]["inspected_notice"] == "NO"
        assert rows[0]["replay_count"] == 1
        assert rows[0]["optional_reason_tags"] == []
        assert rows[0]["inspection_telemetry"]["zoom_change_count"] >= 1
        assert not errors
        # A reload after primary commit must not ask that same primary question again.
        page.reload()
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_visible()
        assert page.evaluate("window.__noticeabilityState().trial.trial_id") == "t_1"
        browser.close()


def test_interrupted_initial_view_is_not_silently_replayed(pilot: Path) -> None:
    with running(pilot) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        page.goto(url)
        page.locator("#begin").click()
        page.locator("#watch").click()
        page.evaluate("window.dispatchEvent(new Event('blur'))")
        expect(page.locator("#skip")).to_be_visible()
        assert not (pilot.parent / "observations.jsonl").exists()
        page.locator("#skip").click()
        expect(page.locator("#watch")).to_be_visible()
        assert page.evaluate("window.__noticeabilityState().trial.trial_id") == "t_1"
        browser.close()


def test_analysis_and_export_keep_constructs_separate(pilot: Path) -> None:
    assert analyze(pilot)["spontaneous"]["rates"]["YES"] is None
    direct_fixture(pilot, "MAYBE")
    report = analyze(pilot)
    assert report["spontaneous"]["counts"] == {"NO": 0, "MAYBE": 1, "YES": 0}
    assert report["spontaneous_inspected_disagreement"] is None
    assert report["production_tau"] is None
    export = export_training(pilot, pilot.parent / "export.json")
    assert export["records"][0]["target_level"] == "clip_event"
    assert not export["critic_retraining_permitted"]
    with pytest.raises(ValueError, match="no supported human threshold"):
        fixed_validation_plan(pilot, pilot.parent / "validation.json")


def test_ordinal_monotonicity_and_evidence_gate() -> None:
    x = [1, 2, 3, 4, 5, 6, 7] * 4
    y = [2, 2, 1, 1, 0, 0, 0] * 4
    p = migration.ordinal_probabilities(migration.fit_ordinal(x, y), np.arange(1, 8))
    assert np.allclose(p.sum(1), 1)
    assert (np.diff(p[:, 2]) <= 0).all()
    assert (p > 0).all()  # Probabilistic, never hard-converted from old scores.
    assert not migration.calibrate_matches([])["proxy_gate_passed"]
    matches = [
        {
            "render": "r1",
            "source": f"s{i % 4}",
            "family": "f",
            "style": "ordinary",
            "score": x[i],
            "label": y[i],
            "original_observation_id": str(i),
        }
        for i in range(28)
    ]
    result = migration.calibrate_matches(matches)
    assert result["render_strata"]["r1"]["held_out_source"]["count"] == 28
    assert not result["proxy_gate_passed"]  # Fewer than 30 unique overlap ratings.


def test_reversible_idempotent_migration_and_direct_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records, matches = [], []
    for source in range(4):
        for score, label in [(1, 2), (4, 1), (7, 0)]:
            for repeat in range(3):
                oid = f"old:{source}:{score}:{repeat}"
                raw = {
                    "render_hashes": [oid],
                    "render_protocol_hash": "r1",
                    "rater_id": "test",
                    "rating": score,
                }
                records.append(
                    {
                        "original_observation_id": oid,
                        "raw_observation": raw,
                        "original_row_hash": identity("legacy-row", raw),
                        "legacy_quality_ordinal": score,
                    }
                )
                matches.append(
                    {
                        "original_observation_id": oid,
                        "render": "r1",
                        "source": str(source),
                        "family": "f",
                        "style": "ordinary",
                        "score": score,
                        "label": label,
                    }
                )
    raw_pair = {"pair_outcome": "a_better"}
    records.append(
        {
            "original_observation_id": "pair",
            "raw_observation": raw_pair,
            "original_row_hash": identity("legacy-row", raw_pair),
            "legacy_pairwise_quality": "a_better",
        }
    )
    raw = tmp_path / "old.jsonl"
    raw.write_text("".join(json.dumps(r["raw_observation"]) + "\n" for r in records))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "raw.jsonl").write_bytes(raw.read_bytes())
    before = raw.read_bytes()
    registry = {
        "records": records,
        "sources": [
            {
                "snapshot_directory": str(snapshot),
                "files": {"raw.jsonl": file_sha256(raw)},
                "original_observations": str(raw),
                "raw_sha256": file_sha256(raw),
            }
        ],
    }
    registry["registry_id"] = identity("legacy-registry", registry)
    registry_path = tmp_path / "registry.json"
    write_json(registry_path, registry)
    assert read_registry(registry_path) == registry
    monkeypatch.setattr(migration, "matched_overlap", lambda *args: matches)
    model = {
        **migration.calibrate_matches(matches),
        "registry_id": registry["registry_id"],
        "overlap_evidence": matches,
        "noticeability_manifest": "unit-test-placeholder",
    }
    assert model["proxy_gate_passed"]
    model["migration_model_hash"] = identity("notice-migration-model", model)
    model_path, output = tmp_path / "model.json", tmp_path / "proxies.json"
    write_json(model_path, model)
    first = migration.migrate(registry_path, model_path, output)
    assert len(first["derived_records"]) == 36
    assert migration.migrate(registry_path, model_path, output) == first
    assert migration.validate_migration(registry_path, model_path, output)["valid"]
    assert all(r["original_observation_id"] != "pair" for r in first["derived_records"])
    assert all(
        r["migration_model_hash"] == model["migration_model_hash"]
        and r["label_strength"] == "WEAK_PROXY"
        for r in first["derived_records"]
    )
    direct = [
        {
            "render_hash": records[0]["raw_observation"]["render_hashes"][0],
            "rater_id": "test",
            "provenance": "DIRECT_HUMAN",
        }
    ]
    merged = migration.prefer_direct(direct, first["derived_records"])
    assert len(merged) == 36 and merged[0] == direct[0]
    output.unlink()  # Only a temporary derived test artifact, never the raw evidence.
    assert migration.migrate(registry_path, model_path, output) == first
    assert raw.read_bytes() == before


def test_render_mismatch_blocks_direct_calibration(
    pilot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = direct_fixture(pilot)
    manifest = read_json(pilot)
    manifest["trials"][0]["original_observation_id"] = "old"
    manifest["legacy_registry"] = "registry.json"
    # Unit-test the matching boundary independently of the stronger frozen-asset guard.
    monkeypatch.setattr(migration, "authenticated_observations", lambda _: [row])
    registry = {
        "registry_id": "test_registry",
        "records": [
            {
                "original_observation_id": "old",
                "legacy_quality_ordinal": 3,
                "raw_observation": {
                    "render_hashes": ["wrong-render"],
                    "render_protocol_hash": row["render_protocol_hash"],
                    "rater_id": row["rater_id"],
                },
            }
        ],
    }
    write_json(pilot.parent / "registry.json", registry)
    new_manifest = pilot.parent / "matching-test.json"
    write_json(new_manifest, manifest)
    with pytest.raises(ValueError, match="mismatch"):
        migration.matched_overlap(registry, new_manifest)


def test_interleaved_staircases_and_clip_head() -> None:
    assert [staircase_step(2, response, 5) for response in ("NO", "MAYBE", "YES")] == [3, 2, 1]
    tracks = {"a": ["a0", "a1", "a2"], "b": ["b0", "b1", "b2"]}
    first = select_interleaved(tracks, [], seed=1)
    history = [{"track": first[0], "spontaneous_notice": "YES"}]
    assert select_interleaved(tracks, history, seed=1)[0] != first[0]
    backbone = FixedRigFlatTCN(FixedRigTCNConfig(num_joints=2, hidden_channels=8))
    model = FixedRigNoticeability(backbone, ("ordinary", "hands_in_pockets"))
    result = model(torch.zeros(2, 17, 51), torch.tensor([0, 1]))
    assert result["noticeability_clip_logits"].shape == (2, 3)
    assert torch.allclose(
        result["p_notice_spontaneous"], result["noticeability_probabilities"][:, 2]
    )
    loss = clip_noticeability_loss(result["noticeability_clip_logits"], torch.tensor([0, 2]))
    loss.backward()
    with pytest.raises(ValueError, match="never replicated"):
        clip_noticeability_loss(torch.zeros(2, 3, 17), torch.zeros(2, 17, dtype=torch.long))


def test_frozen_interleaved_scheduler_is_runnable(pilot: Path) -> None:
    manifest = read_json(pilot)
    base = manifest["stimuli"][0]
    stimuli = [dict(base, control_kind="clean")]
    for track in ("phase", "rigidity"):
        for level in range(3):
            stimuli.append(
                {
                    **base,
                    "stimulus_id": f"{track}_{level}",
                    "ladder_id": track,
                    "physical_strength": float(level + 1),
                }
            )
    manifest.update(stimuli=stimuli, seed=1, legacy_registry="legacy_registry.json")
    write_json(pilot.parent / "legacy_registry.json", {})
    # A different manifest is separately frozen; the original fixture is not overwritten.
    parent = pilot.parent / "parent"
    parent.mkdir()
    (parent / "motion.npz").write_bytes((pilot.parent / "motion.npz").read_bytes())
    write_json(parent / "legacy_registry.json", {})
    write_json(parent / "pilot_manifest.json", manifest)
    freeze_pilot(parent / "pilot_manifest.json")
    output = pilot.parent / "adaptive"
    derived = build_staircase_pilot(parent / "pilot_manifest.json", output, steps=10, seed=4)
    assert derived["scheduler"]["steps"] == 10
    assert len([t for t in derived["trials"] if t["track"] == "control"]) == 2
    freeze_pilot(output / "pilot_manifest.json")
    with running(output / "pilot_manifest.json") as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        page.goto(url)
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_visible()
        trial = page.evaluate("window.__noticeabilityState().trial")
        assert trial["total"] == 10
        selected = next(t for t in derived["trials"] if t["trial_id"] == trial["trial_id"])
        expected_track, _, sid = select_interleaved(derived["scheduler"]["tracks"], [], seed=4)
        assert selected["track"] == expected_track and selected["stimulus_id"] == sid
        browser.close()


def test_detection_and_production_satisficing() -> None:
    metrics = detector_metrics(
        ["NO", "MAYBE", "YES"], [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]]
    )
    assert metrics["auroc_yes_vs_no"] == 1 and metrics["maybe_excluded_from_binary"] == 1
    assert choose_operating_point([], [], split_role="held_out_validation")["tau"] is None
    with pytest.raises(ValueError, match="held-out"):
        choose_operating_point([0.1], [0.9], split_role="test")
    point = choose_operating_point([0.12] * 40, [0.8] * 20, split_role="held_out_validation")
    assert point["tau"] != 0.5 and point["recall"] == 1
    assert (
        production_noticeability(0.1, tau=0.2, deterministic_feasible=True)["perceptual_penalty"]
        == 0
    )
    assert (
        production_noticeability(0.01, tau=0.2, deterministic_feasible=True)["perceptual_penalty"]
        == 0
    )
    assert not production_noticeability(0.01, tau=0.2, deterministic_feasible=False)["feasible"]
    assert latent_detectability([])["status"] == "awaiting_human_labels"
