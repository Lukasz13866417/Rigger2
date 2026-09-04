from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page, expect, sync_playwright
from test_subjective_ui import (
    _begin_session,
    _isolated_manifest,
    _launch_browser,
    _read_jsonl,
    _running_evaluator,
)

from motionlab.perceptual import subjective
from motionlab.perceptual.observation_auth import validate_current_protocol_observations


def _free_camera_manifest(
    tmp_path: Path,
    *,
    task_type: str = "naturalness",
    comparison_mode: str | None = None,
) -> tuple[dict[str, Any], Path]:
    manifest, path = _isolated_manifest(
        tmp_path, task_type=task_type, comparison_mode=comparison_mode
    )
    settings = subjective.FREE_CAMERA_VIEWING_SETTINGS
    manifest["sessions"] = manifest["sessions"][:1]
    manifest["calibration_tutorial"] = []
    identifiers = {}
    for item in manifest["stimuli"]:
        previous_id = item["stimulus_id"]
        item["stimulus_id"] = subjective._public_stimulus_id(item["motion"], settings)
        identifiers[previous_id] = item["stimulus_id"]
        payload = subjective._two_cycle_payload(
            Path(manifest["stimulus_directory"]) / item["motion"],
            phase_reference_path=Path(manifest["stimulus_directory"])
            / item["phase_reference_motion"],
            viewing_settings=settings,
        )
        item["render_protocol_hash"] = subjective.render_protocol_hash(settings)
        item["render_hash"] = payload["render_hash"]
        item["stimulus_population"] = "HARD_FEASIBLE_PERCEPTUAL"
        item["measurement_cohort"] = "hard_feasible"
    by_id = {item["stimulus_id"]: item for item in manifest["stimuli"]}
    for trial in manifest["sessions"][0]["trials"]:
        for key in ("stimulus_ids", "presentation_order"):
            trial[key] = [identifiers[value] for value in trial[key]]
        trial["render_hashes"] = [
            subjective._canonical_hash(
                "presented-render",
                {
                    "base_render_hash": by_id[value]["render_hash"],
                    "mirror_x": trial["view_mirror_x"],
                },
            )
            for value in trial["stimulus_ids"]
        ]
    manifest["hidden_anchor_stimulus_ids"] = [
        identifiers[value] for value in manifest["hidden_anchor_stimulus_ids"]
    ]
    manifest["protocol_version"] = subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION
    manifest["viewing_settings"] = settings
    manifest["render_protocol_hash"] = subjective.render_protocol_hash(settings)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, path


def _inspection(page: Page) -> dict[str, Any]:
    return page.evaluate("window.__motionlabInspectionState()")  # type: ignore[no-any-return]


def _drag(
    page: Page,
    *,
    dx: float,
    dy: float,
    button: str = "left",
    shift: bool = False,
) -> None:
    box = page.locator("#view").bounding_box()
    assert box is not None
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    if shift:
        page.keyboard.down("Shift")
    page.mouse.down(button=button)  # type: ignore[arg-type]
    page.mouse.move(x + dx, y + dy, steps=5)
    page.mouse.up(button=button)  # type: ignore[arg-type]
    if shift:
        page.keyboard.up("Shift")


def test_free_camera_settings_preserve_historical_protocol() -> None:
    old = subjective.MANNEQUIN_VIEWING_SETTINGS
    new = subjective.FREE_CAMERA_VIEWING_SETTINGS
    assert old["zoom_limits"] == {"minimum": 0.65, "maximum": 1.8, "step": 0.15}
    assert old["orbit"] == {"enabled": True, "maximum_pitch_degrees": 35.0}
    assert "pan_x_m" not in old["default_camera"]
    assert new["zoom_limits"]["minimum"] == 0.1
    assert new["zoom_limits"]["maximum"] == 5.0
    assert subjective.render_protocol_hash(new) != subjective.render_protocol_hash(old)
    assert (
        subjective._resolve_viewing_settings(
            {"viewing_settings": new, "render_protocol_hash": subjective.render_protocol_hash(new)}
        )
        == new
    )
    state = subjective._default_camera_state(new)
    assert subjective._camera_is_default(state, new)
    assert subjective._validate_camera_state(state, new) == state
    state["pan_x_m"] = 0.1
    assert not subjective._camera_is_default(state, new)
    with pytest.raises(ValueError, match="unsupported fields"):
        subjective._validate_camera_state(state, old)
    with pytest.raises(ValueError, match="unsupported fields"):
        subjective._validate_camera_state(subjective._default_camera_state(old), new)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pan_x_m", 20.01),
        ("pan_y_m", -20.01),
        ("pan_x_m", float("nan")),
        ("pan_y_m", True),
        ("zoom", 0.09),
        ("zoom", 5.01),
        ("pitch_degrees", 89.01),
    ],
)
def test_free_camera_rejects_invalid_states(field: str, value: Any) -> None:
    settings = subjective.FREE_CAMERA_VIEWING_SETTINGS
    state = subjective._default_camera_state(settings)
    state[field] = value
    with pytest.raises(ValueError):
        subjective._validate_camera_state(state, settings)


def _empty_telemetry(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": subjective.INSPECTION_TELEMETRY_VERSION,
        "replay_count": [0],
        "speed_change_events": [],
        "playback_wall_time_by_speed_ms": {f"{rate:g}": 0 for rate in settings["playback_rates"]},
        "camera_change_events": [],
        "zoom_change_count": 0,
        "view_change_count": 0,
        "overlay_change_count": 0,
        "total_inspection_time_ms": 100,
        "final_playback_rate": 1,
        "final_camera_state": subjective._default_camera_state(settings),
    }


def test_pan_telemetry_is_versioned_and_validates_event_states() -> None:
    settings = subjective.FREE_CAMERA_VIEWING_SETTINGS
    telemetry = _empty_telemetry(settings)
    after = {**telemetry["final_camera_state"], "pan_x_m": 1.0}
    telemetry["camera_change_events"] = [
        {"elapsed_ms": 50, "kind": "pan", "from": telemetry["final_camera_state"], "to": after}
    ]
    telemetry["final_camera_state"] = after
    assert (
        subjective._validate_inspection_telemetry(
            telemetry, stimulus_count=1, viewing_settings=settings
        )["final_camera_state"]
        == after
    )
    bad = deepcopy(telemetry)
    bad["camera_change_events"][0]["from"]["pan_y_m"] = 21
    with pytest.raises(ValueError, match="camera pan"):
        subjective._validate_inspection_telemetry(bad, stimulus_count=1, viewing_settings=settings)
    with pytest.raises(ValueError, match="event kind"):
        subjective._validate_inspection_telemetry(
            telemetry, stimulus_count=1, viewing_settings=subjective.MANNEQUIN_VIEWING_SETTINGS
        )


def test_free_camera_browser_controls_locks_and_saved_telemetry(tmp_path: Path) -> None:
    manifest, path = _free_camera_manifest(tmp_path)
    settings = subjective.FREE_CAMERA_VIEWING_SETTINGS
    default = subjective._default_camera_state(settings)
    with (
        _running_evaluator(manifest, path, tmp_path) as (url, observations, served),
        sync_playwright() as playwright,
    ):
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport={"width": 1120, "height": 1100})
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            _begin_session(page, url, "free-camera-rater")
            expect(page.locator("#zoomOut")).to_be_disabled()
            page.locator("#view").hover()
            page.mouse.wheel(0, 700)
            _drag(page, dx=60, dy=30, shift=True)
            _drag(page, dx=60, dy=-30)
            assert _inspection(page)["camera"] == default
            assert _inspection(page)["telemetry"]["camera_change_events"] == []
            expect(page.locator("#playbackNotice")).to_have_text(
                "Full viewing complete. You may rate or replay.", timeout=15_000
            )
            # An actual browser wheel gesture zooms well below the previous .65 minimum,
            # without scrolling the surrounding page.
            page.locator("#view").hover()
            scroll_before = page.evaluate("window.scrollY")
            page.mouse.wheel(0, 500)
            page.wait_for_function("window.__motionlabInspectionState().camera.zoom < .65")
            assert page.evaluate("window.scrollY") == scroll_before
            zoomed = _inspection(page)["camera"]
            assert zoomed["zoom"] == pytest.approx(math.exp(-0.75))
            _drag(page, dx=120, dy=75, shift=True)
            panned = _inspection(page)["camera"]
            assert panned["pan_x_m"] > 0
            assert panned["pan_y_m"] < 0
            assert panned["yaw_degrees"] == default["yaw_degrees"]
            _drag(page, dx=-40, dy=0, button="middle")
            assert _inspection(page)["camera"]["pan_x_m"] < panned["pan_x_m"]
            _drag(page, dx=0, dy=-40, button="right")
            assert _inspection(page)["camera"]["pan_y_m"] > panned["pan_y_m"]
            _drag(page, dx=380, dy=-380)
            orbited = _inspection(page)["camera"]
            assert orbited["pitch_degrees"] == 89
            assert orbited["yaw_degrees"] > 150
            assert orbited["pan_x_m"] != 0
            page.locator("#resetCamera").click()
            assert _inspection(page)["camera"] == default

            # Delta modes are normalized, and a long high-resolution trackpad gesture
            # creates one event instead of exhausting the 512-event telemetry limit.
            for mode, amount, expected in ((1, 10, 160), (0, 160, 160)):
                page.locator("#resetCamera").click()
                page.locator("#view").dispatch_event("wheel", {"deltaY": amount, "deltaMode": mode})
                assert _inspection(page)["camera"]["zoom"] == pytest.approx(
                    math.exp(-expected * 0.0015)
                )
            page.locator("#resetCamera").click()
            before = _inspection(page)["telemetry"]["zoom_change_count"]
            page.evaluate(
                """() => { for (let i=0; i<1000; i++) document.getElementById('view')
                    .dispatchEvent(new WheelEvent('wheel', {deltaY:.1, deltaMode:0})); }"""
            )
            state = _inspection(page)
            assert state["camera"]["zoom"] == pytest.approx(math.exp(-0.15))
            assert state["telemetry"]["zoom_change_count"] == before + 1
            page.locator("#resetCamera").click()
            for _ in range(12):
                page.locator("#zoomOut").click()
            assert _inspection(page)["camera"]["zoom"] == 0.1
            for _ in range(20):
                page.locator("#zoomIn").click()
            assert _inspection(page)["camera"]["zoom"] == 5
            page.locator("#resetCamera").click()
            _drag(page, dx=80, dy=30, shift=True)
            page.locator('#scale button[data-value="4"]').click()
            page.locator("#continueRating").click()
            locked = _inspection(page)
            page.locator("#view").hover()
            page.mouse.wheel(0, -500)
            _drag(page, dx=60, dy=-60, button="right")
            _drag(page, dx=60, dy=-60)
            assert _inspection(page) == locked
            page.locator("#submit").click()
            expect(page.get_by_role("heading", name="Session complete")).to_be_visible(
                timeout=10_000
            )
            assert errors == []
        finally:
            browser.close()
    row = _read_jsonl(observations)[0]
    assert row["render_protocol_hash"] == subjective.render_protocol_hash(settings)
    telemetry = row["inspection_telemetry"]
    assert telemetry["final_camera_state"] == locked["camera"]
    assert telemetry["final_camera_state"]["pan_x_m"] > 0
    assert any(event["kind"] == "pan" for event in telemetry["camera_change_events"])
    subjective._validate_inspection_telemetry(
        telemetry, stimulus_count=1, viewing_settings=settings
    )
    validate_current_protocol_observations([row], manifest, _read_jsonl(served))


def test_free_camera_pair_has_shared_view_and_cancelled_drags_are_recorded(tmp_path: Path) -> None:
    manifest, path = _free_camera_manifest(
        tmp_path, task_type="pair_comparison", comparison_mode="same_viewport_toggle"
    )
    with (
        _running_evaluator(manifest, path, tmp_path) as (url, observations, served),
        sync_playwright() as playwright,
    ):
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport={"width": 1120, "height": 1100})
            _begin_session(page, url, "free-camera-pair-rater")
            expect(page.locator("#playbackNotice")).to_have_text(
                "Full viewing complete. You may rate or replay.", timeout=20_000
            )
            page.locator("#view").hover()
            page.mouse.wheel(0, 600)
            page.wait_for_function("window.__motionlabInspectionState().camera.zoom < .5")
            _drag(page, dx=90, dy=40, shift=True)
            _drag(page, dx=100, dy=-180)
            camera = _inspection(page)["camera"]
            page.locator("#toggleA").click()
            assert _inspection(page)["camera"] == camera
            page.locator("#toggleB").click()
            assert _inspection(page)["camera"] == camera
            # Pointer cancellation ends the gesture rather than losing its last state.
            page.locator("#view").hover()
            page.mouse.down(button="right")
            box = page.locator("#view").bounding_box()
            assert box is not None
            page.mouse.move(box["x"] + box["width"] / 2 + 60, box["y"] + box["height"] / 2)
            page.locator("#view").dispatch_event("pointercancel", {"pointerId": 1})
            page.mouse.up(button="right")
            state = _inspection(page)
            assert state["telemetry"]["camera_change_events"][-1]["kind"] == "pan"
            assert state["telemetry"]["camera_change_events"][-1]["to"] == state["camera"]
            page.locator('[data-value="effectively_equal"]').click()
            page.locator("#continueRating").click()
            page.locator("#submit").click()
            expect(page.get_by_role("heading", name="Session complete")).to_be_visible(
                timeout=10_000
            )
        finally:
            browser.close()
    telemetry = _read_jsonl(observations)[0]["inspection_telemetry"]
    assert telemetry["final_camera_state"] == state["camera"]
    subjective._validate_inspection_telemetry(
        telemetry, stimulus_count=2, viewing_settings=subjective.FREE_CAMERA_VIEWING_SETTINGS
    )
    validate_current_protocol_observations(_read_jsonl(observations), manifest, _read_jsonl(served))
