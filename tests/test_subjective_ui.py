from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from playwright.sync_api import Browser, Error, Page, Playwright, expect, sync_playwright

from motionlab.perceptual import subjective

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = PROJECT_ROOT / "artifacts/phase14_subjective_pilot/pilot_manifest.json"
PAIR_PRESENTATION_MODES = ("same_viewport_toggle", "sequential_neutral_gap")


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(process: subprocess.Popen[str], url: str) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.communicate()[0]
            raise AssertionError(f"subjective evaluator exited during startup:\n{output}")
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(0.05)
    raise AssertionError("subjective evaluator did not become ready within 15 seconds")


def _stop_server(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def _browser_required() -> bool:
    return os.environ.get("MOTIONLAB_REQUIRE_E2E_BROWSER", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _launch_browser(playwright: Playwright) -> Browser:
    requested = os.environ.get("MOTIONLAB_E2E_BROWSER")
    commands = (
        requested,
        shutil.which("brave"),
        shutil.which("brave-browser"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        playwright.chromium.executable_path,
    )
    executable = next(
        (Path(command) for command in commands if command is not None and Path(command).is_file()),
        None,
    )
    if executable is None:
        message = (
            "no Chromium-family browser found; install Playwright Chromium or set "
            "MOTIONLAB_E2E_BROWSER"
        )
        if _browser_required():
            pytest.fail(message)
        pytest.skip(message)
    try:
        return playwright.chromium.launch(
            executable_path=str(executable),
            headless=True,
            args=["--disable-dev-shm-usage"],
        )
    except Error as exc:
        pytest.fail(f"could not launch headless browser {executable}: {exc}")


def _isolated_manifest(
    tmp_path: Path,
    *,
    task_type: str,
    comparison_mode: str | None = None,
) -> tuple[dict[str, Any], Path]:
    if not PILOT_PATH.is_file():
        message = "prepared Phase 14 pilot artifacts are not present"
        # Research artifacts and human evidence are intentionally not published. Requiring
        # a browser must not require private/local datasets on an otherwise clean checkout.
        pytest.skip(message)

    manifest: dict[str, Any] = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    trial = next(
        (
            candidate
            for candidate in manifest["sessions"][0]["trials"]
            if candidate["task_type"] == task_type
            and (comparison_mode is None or candidate["comparison_mode"] == comparison_mode)
        ),
        None,
    )
    if trial is None:
        pytest.fail(
            f"real Phase 14 pilot has no {task_type!r} trial with "
            f"comparison mode {comparison_mode!r}"
        )
    stimulus_root = Path(manifest.get("stimulus_directory", manifest["pair_dataset_directory"]))
    by_id = {item["stimulus_id"]: item for item in manifest["stimuli"]}
    identifier_map: dict[str, str] = {}
    converted_stimuli: list[dict[str, Any]] = []
    for old_id in trial["stimulus_ids"]:
        item = dict(by_id[old_id])
        new_id = subjective._public_stimulus_id(
            item["motion"],
            subjective.MANNEQUIN_VIEWING_SETTINGS,
        )
        payload = subjective._two_cycle_payload(
            stimulus_root / item["motion"],
            phase_reference_path=stimulus_root / item["phase_reference_motion"],
            viewing_settings=subjective.MANNEQUIN_VIEWING_SETTINGS,
        )
        identifier_map[old_id] = new_id
        item.update(
            {
                "stimulus_id": new_id,
                "render_protocol_hash": subjective.render_protocol_hash(),
                "render_hash": payload["render_hash"],
                "cycle_window": payload["cycle_window"],
            }
        )
        converted_stimuli.append(item)
    trial = dict(trial)
    trial["stimulus_ids"] = [identifier_map[value] for value in trial["stimulus_ids"]]
    trial["presentation_order"] = [identifier_map[value] for value in trial["presentation_order"]]
    trial["render_hashes"] = [
        subjective._canonical_hash(
            "presented-render",
            {
                "base_render_hash": item["render_hash"],
                "mirror_x": trial["view_mirror_x"],
            },
        )
        for item in converted_stimuli
    ]
    trial["stimulus_populations"] = ["HARD_FEASIBLE_PERCEPTUAL" for _ in trial["stimulus_ids"]]
    trial["measurement_cohorts"] = ["hard_feasible" for _ in trial["stimulus_ids"]]
    trial["critic_training_eligible"] = True
    manifest["format_version"] = subjective.PILOT_MANIFEST_VERSION
    manifest["protocol_version"] = subjective.SUBJECTIVE_PROTOCOL_VERSION
    manifest["viewing_settings"] = subjective.MANNEQUIN_VIEWING_SETTINGS
    manifest["render_protocol_hash"] = subjective.render_protocol_hash()
    manifest["stimulus_directory"] = str(stimulus_root)
    manifest["stimuli"] = converted_stimuli
    manifest["hidden_anchor_stimulus_ids"] = [
        identifier_map[value]
        for value in manifest["hidden_anchor_stimulus_ids"]
        if value in identifier_map
    ]
    manifest["sessions"][0]["trials"] = [trial]
    manifest["sessions"][0]["trial_count"] = 1
    manifest["adaptive_policy"]["enabled"] = False
    path = tmp_path / "pilot_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, path


def test_missing_local_pilot_skips_even_when_browser_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOTIONLAB_REQUIRE_E2E_BROWSER", "1")
    monkeypatch.setattr(sys.modules[__name__], "PILOT_PATH", tmp_path / "absent_pilot.json")
    with pytest.raises(pytest.skip.Exception, match="pilot artifacts are not present"):
        _isolated_manifest(tmp_path, task_type="naturalness")


@contextmanager
def _running_evaluator(
    manifest: dict[str, Any],
    manifest_path: Path,
    tmp_path: Path,
) -> Iterator[tuple[str, Path, Path]]:
    observations = tmp_path / "raw_observations.jsonl"
    served_playlist = tmp_path / "served_playlist.jsonl"
    assert observations.parent == tmp_path
    assert served_playlist.parent == tmp_path
    port = _available_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "motionlab",
            "label-perceptual-pairs",
            manifest["pair_dataset_directory"],
            "--pilot",
            str(manifest_path),
            "--observations",
            str(observations),
            "--served-playlist",
            str(served_playlist),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(process, base_url)
        yield base_url, observations, served_playlist
    finally:
        _stop_server(process)


def _begin_session(page: Page, base_url: str, rater_id: str) -> dict[str, Any]:
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.get_by_role("heading", name="Motion quality study")).to_be_visible()
    assert page.locator("#view").evaluate("canvas => [canvas.width, canvas.height]") == [
        960,
        640,
    ]
    page.locator("#rater").fill(rater_id)
    with page.expect_response(lambda response: "/api/next?" in response.url) as response:
        page.get_by_role("button", name="Begin session").click()
    public_trial = response.value.json()["trial"]
    page.wait_for_function("window.__motionlabRendererReady === true", timeout=15_000)
    return public_trial


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_subjective_evaluator_headless_flow_is_blinded_and_isolated(
    tmp_path: Path,
) -> None:
    manifest, isolated_manifest = _isolated_manifest(
        tmp_path,
        task_type="naturalness",
    )
    with (
        _running_evaluator(
            manifest,
            isolated_manifest,
            tmp_path,
        ) as (base_url, observations, served_playlist),
        sync_playwright() as playwright,
    ):
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport={"width": 1120, "height": 900})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            public_trial = _begin_session(page, base_url, "e2e-headless-rater")
            assert public_trial["task_type"] == "naturalness"
            assert public_trial["scale_anchors"] == [
                "Clearly broken / highly unnatural",
                "Very unnatural",
                "Noticeably unnatural",
                "Acceptable but visibly mediocre",
                "Good and mostly natural",
                "Very good with only subtle issues",
                "Excellent / difficult to improve",
            ]
            assert {
                "source_ids",
                "families",
                "hidden_anchor",
                "hidden_repeat_group_ids",
                "pair_id",
            }.isdisjoint(public_trial)

            scale = page.locator("#scale button[data-value]")
            expect(page.locator("#question")).to_have_text(
                "Ignoring whether you personally like the style, how natural and internally "
                "coherent is this walk?"
            )
            expect(scale).to_have_count(7)
            assert all(scale.nth(index).is_disabled() for index in range(7))
            expect(page.locator("#submit")).to_be_disabled()
            expect(page.locator('[data-speed="0.5"]')).to_be_disabled()
            expect(page.get_by_role("button", name="Front")).to_be_disabled()
            expect(page.get_by_role("button", name="Zoom in")).to_be_disabled()
            expect(page.locator("#skeletonOverlay")).to_be_disabled()

            page.wait_for_timeout(200)
            mannequin_pixels = page.locator("#view").evaluate(
                """canvas => {
                        const data = canvas.getContext('2d').getImageData(
                            0, 0, canvas.width, canvas.height
                        ).data;
                        let count = 0;
                        for (let i = 0; i < data.length; i += 4) {
                            if (data[i] > 55 && data[i + 1] > 65 && data[i + 2] > 80) {
                                count += 1;
                            }
                        }
                        return count;
                    }"""
            )
            assert mannequin_pixels > 1_000

            expect(page.locator("#playbackNotice")).to_have_text(
                "Full viewing complete. You may rate or replay.",
                timeout=15_000,
            )
            assert all(scale.nth(index).is_enabled() for index in range(7))
            page.locator('[data-speed="0.5"]').click()
            page.get_by_role("button", name="Front").click()
            page.get_by_role("button", name="Zoom in").click()
            page.locator("#skeletonOverlay").check()
            page.locator('#scale button[data-value="4"]').click()
            page.locator("#confidence").select_option("3")
            page.locator('#reasons input[value="naturalness"]').check()
            interval_button = page.get_by_role("button", name="Mark interval point")
            interval_button.click()
            interval_button.click()
            expect(page.locator("#difficulty")).to_be_hidden()
            page.locator("#continueRating").click()
            expect(page.locator("#difficulty")).to_be_visible()
            expect(page.locator("#play")).to_be_disabled()
            expect(page.locator('[data-speed="0.5"]')).to_be_disabled()
            expect(page.get_by_role("button", name="Front")).to_be_disabled()
            expect(page.locator("#skeletonOverlay")).to_be_disabled()
            page.locator('[data-difficulty="moderate"]').click()
            page.screenshot(path=tmp_path / "subjective-evaluator.png", full_page=True)
            page.locator("#submit").click()
            expect(page.get_by_role("heading", name="Session complete")).to_be_visible(
                timeout=10_000
            )
            assert page_errors == []
        finally:
            browser.close()

    recorded = _read_jsonl(observations)
    served = _read_jsonl(served_playlist)
    assert len(recorded) == 1
    assert len(served) == 1
    observation = recorded[0]
    assert observation["rater_id"] == "e2e-headless-rater"
    assert observation["rating"] == 4
    assert observation["confidence"] == 3
    assert observation["reason_tags"] == ["naturalness"]
    assert len(observation["marked_interval_s"]) == 2
    assert observation["judgment_difficulty"] == "moderate"
    telemetry = observation["inspection_telemetry"]
    assert telemetry["final_playback_rate"] == 0.5
    assert len(telemetry["speed_change_events"]) == 1
    assert telemetry["zoom_change_count"] == 1
    assert telemetry["view_change_count"] == 1
    assert telemetry["overlay_change_count"] == 1
    assert telemetry["final_camera_state"]["skeleton_overlay"]
    assert observation["render_protocol_hash"] == manifest["render_protocol_hash"]
    assert observation["trial_id"] == served[0]["trial_id"]


def test_calibration_tutorial_precedes_scoring_without_observation(
    tmp_path: Path,
) -> None:
    manifest, isolated_manifest = _isolated_manifest(
        tmp_path,
        task_type="naturalness",
    )
    base = manifest["stimuli"][0]
    tutorial_ids = [base["stimulus_id"]] + [
        f"{base['stimulus_id']}:tutorial-{index}" for index in range(1, 5)
    ]
    manifest["stimuli"] = [base] + [
        {
            **base,
            "stimulus_id": tutorial_ids[index],
            "variant_id": f"tutorial-variant-{index}",
        }
        for index in range(1, 5)
    ]
    manifest["calibration_tutorial"] = [
        {
            "tutorial_id": "tutorial-001",
            "tutorial_index": 0,
            "task_type": "calibration",
            "stimulus_ids": tutorial_ids,
            "presentation_order": tutorial_ids,
            "render_hashes": [base["render_hash"] for _ in tutorial_ids],
            "stimulus_populations": ["CALIBRATION_ONLY" for _ in tutorial_ids],
            "measurement_cohorts": ["calibration_easy" for _ in tutorial_ids],
            "tutorial_disclosure": {
                "calibration_family": "foot_slide",
                "ordered_severities": [
                    "clean",
                    "mild",
                    "medium",
                    "strong",
                    "severe",
                ],
                "instruction": "Watch how foot sliding increases.",
            },
        }
    ]
    isolated_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with (
        _running_evaluator(
            manifest,
            isolated_manifest,
            tmp_path,
        ) as (base_url, observations, served_playlist),
        sync_playwright() as playwright,
    ):
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport={"width": 1120, "height": 900})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            public_trial = _begin_session(page, base_url, "e2e-tutorial-rater")
            assert public_trial["is_tutorial"]
            assert public_trial["tutorial_disclosure"] == {
                "calibration_family": "foot_slide",
                "ordered_severities": [
                    "clean",
                    "mild",
                    "medium",
                    "strong",
                    "severe",
                ],
                "instruction": "Watch how foot sliding increases.",
            }
            assert {
                "stimulus_populations",
                "measurement_cohorts",
                "families",
            }.isdisjoint(public_trial)

            # Calibration is explicitly an unrestricted learning mode. The standardized
            # default-camera/1x full-play lock applies only to scored stimuli.
            expect(page.locator("#continueTutorial")).to_be_enabled(timeout=500)
            expect(page.locator("#play")).to_be_enabled(timeout=500)
            for severity in ("Clean", "Mild", "Medium", "Strong", "Severe"):
                expect(page.get_by_role("button", name=severity, exact=True)).to_be_enabled(
                    timeout=500
                )
            expect(page.locator('[data-speed="0.25"]')).to_be_enabled(timeout=500)
            expect(page.get_by_role("button", name="Side")).to_be_enabled(timeout=500)
            expect(page.get_by_role("button", name="Zoom in")).to_be_enabled(timeout=500)

            page.get_by_role("button", name="Medium", exact=True).click()
            expect(page.locator("#clipBadge")).to_have_text("Medium")
            page.locator('[data-speed="0.25"]').click()
            page.get_by_role("button", name="Side").click()
            page.get_by_role("button", name="Zoom in").click()
            inspection = page.evaluate("window.__motionlabInspectionState()")
            assert inspection["current"] == 2
            assert inspection["playbackRate"] == 0.25
            assert inspection["camera"]["preset"] == "side"
            assert inspection["camera"]["zoom"] > 1.0

            page.locator("#continueTutorial").click()
            expect(page.locator("#taskTitle")).to_have_text("Naturalness", timeout=10_000)
            assert not observations.exists()
            assert page_errors == []
        finally:
            browser.close()

    served = _read_jsonl(served_playlist)
    assert [row["task_type"] for row in served] == ["calibration", "naturalness"]


@pytest.mark.parametrize("comparison_mode", PAIR_PRESENTATION_MODES)
def test_pair_presentations_keep_responses_locked_until_final_server_ack(
    tmp_path: Path,
    comparison_mode: str,
) -> None:
    manifest, isolated_manifest = _isolated_manifest(
        tmp_path,
        task_type="pair_comparison",
        comparison_mode=comparison_mode,
    )
    rater_id = f"e2e-{comparison_mode}"
    with (
        _running_evaluator(
            manifest,
            isolated_manifest,
            tmp_path,
        ) as (base_url, observations, served_playlist),
        sync_playwright() as playwright,
    ):
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport={"width": 1120, "height": 900})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.add_init_script(
                """(() => {
                        const realFetch = window.fetch.bind(window);
                        let releaseFinalAck;
                        const finalAckGate = new Promise(resolve => {
                            releaseFinalAck = resolve;
                        });
                        window.__releaseFinalPlaybackAck = () => releaseFinalAck();
                        window.__finalPlaybackAckPending = false;
                        window.__finalPlaybackAckResponseOk = false;
                        window.__playbackCompleteCount = 0;
                        window.fetch = async (...args) => {
                            const target = typeof args[0] === 'string'
                                ? args[0]
                                : args[0].url;
                            const response = await realFetch(...args);
                            if (target.includes('/api/playback-complete')) {
                                window.__playbackCompleteCount += 1;
                                if (window.__playbackCompleteCount === 2) {
                                    window.__finalPlaybackAckResponseOk = response.ok;
                                    window.__finalPlaybackAckPending = true;
                                    await finalAckGate;
                                    window.__finalPlaybackAckPending = false;
                                }
                            }
                            return response;
                        };
                    })()"""
            )
            public_trial = _begin_session(page, base_url, rater_id)
            assert public_trial["task_type"] == "pair_comparison"
            assert public_trial["comparison_mode"] == comparison_mode
            assert [item["label"] for item in public_trial["stimuli"]] == [
                "A",
                "B",
            ]
            assert {
                "source_ids",
                "families",
                "hidden_anchor",
                "hidden_repeat_group_ids",
                "pair_id",
            }.isdisjoint(public_trial)

            choices = page.locator(".pairChoices button[data-value]")
            expect(page.locator("#taskTitle")).to_have_text("Motion comparison")
            expect(choices).to_have_count(4)
            assert all(choices.nth(index).is_disabled() for index in range(4))
            expect(page.locator("#submit")).to_be_disabled()
            if comparison_mode == "same_viewport_toggle":
                expect(page.locator("#toggleA")).to_be_visible()
                expect(page.locator("#toggleB")).to_be_visible()
                expect(page.locator("#toggleA")).to_be_disabled()
                expect(page.locator("#toggleB")).to_be_disabled()
                expect(page.locator("#keyboard")).to_be_visible()
            else:
                expect(page.locator("#toggleA")).to_be_hidden()
                expect(page.locator("#toggleB")).to_be_hidden()
                expect(page.locator("#keyboard")).to_be_hidden()
                expect(page.locator("#neutral")).to_be_visible(timeout=15_000)

            page.wait_for_function(
                "window.__finalPlaybackAckPending === true",
                timeout=30_000,
            )
            assert page.evaluate("window.__finalPlaybackAckResponseOk") is True
            expect(page.locator("#clipBadge")).to_have_text("B")
            expect(page.locator("#completion")).to_have_text("1 / 2 full viewings")
            expect(page.locator("#playbackNotice")).to_have_text("Confirming full viewing…")
            assert all(choices.nth(index).is_disabled() for index in range(4))
            expect(page.locator("#submit")).to_be_disabled()
            expect(page.locator("#play")).to_be_disabled()
            if comparison_mode == "same_viewport_toggle":
                expect(page.locator("#toggleA")).to_be_disabled()
                expect(page.locator("#toggleB")).to_be_disabled()

            page.evaluate("window.__releaseFinalPlaybackAck()")
            expect(page.locator("#playbackNotice")).to_have_text(
                "Full viewing complete. You may rate or replay.",
                timeout=10_000,
            )
            assert all(choices.nth(index).is_enabled() for index in range(4))
            if comparison_mode == "same_viewport_toggle":
                expect(page.locator("#toggleA")).to_be_enabled()
                expect(page.locator("#toggleB")).to_be_enabled()
                page.locator('[data-speed="1.5"]').click()
                page.get_by_role("button", name="Side").click()
                page.get_by_role("button", name="Zoom out").click()
                page.keyboard.press("a")
                expect(page.locator("#clipBadge")).to_have_text("A")
                before_toggle = page.evaluate("window.__motionlabInspectionState()")
                page.keyboard.press("b")
                expect(page.locator("#clipBadge")).to_have_text("B")
                after_toggle = page.evaluate("window.__motionlabInspectionState()")
                assert before_toggle["frame"] == after_toggle["frame"]
                assert before_toggle["phaseTime"] == after_toggle["phaseTime"]
                assert before_toggle["playbackRate"] == after_toggle["playbackRate"]
                assert before_toggle["camera"] == after_toggle["camera"]

            page.locator('.pairChoices button[data-value="a_better"]').click()
            page.locator("#confidence").select_option("4")
            page.locator('#reasons input[value="coordination"]').check()
            page.locator("#continueRating").click()
            expect(page.locator("#difficulty")).to_be_visible()
            page.locator("#submit").click()
            expect(page.get_by_role("heading", name="Session complete")).to_be_visible(
                timeout=10_000
            )
            assert page_errors == []
        finally:
            browser.close()

    recorded = _read_jsonl(observations)
    served = _read_jsonl(served_playlist)
    assert len(recorded) == 1
    assert len(served) == 1
    observation = recorded[0]
    assert observation["rater_id"] == rater_id
    assert observation["task_type"] == "pair_comparison"
    assert observation["comparison_mode"] == comparison_mode
    assert observation["pair_outcome_display_order"] == "a_better"
    assert observation["confidence"] == 4
    assert observation["reason_tags"] == ["coordination"]
    assert observation["replay_count"] == [0, 0]
    assert observation["judgment_difficulty"] is None
    assert observation["inspection_telemetry"]["replay_count"] == [0, 0]
    assert observation["critic_training_eligible"]
    assert observation["trial_id"] == served[0]["trial_id"]
