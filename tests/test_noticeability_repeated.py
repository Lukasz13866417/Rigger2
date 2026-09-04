from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright
from test_noticeability import direct_fixture
from test_noticeability import pilot as pilot
from test_subjective_ui import _launch_browser

from motionlab.noticeability import protocol, repeated_protocol
from motionlab.noticeability.migration import matched_overlap
from motionlab.noticeability.repeated_continuation import (
    analyze_repeated,
    prepare_repeated_view_pilot,
)
from motionlab.noticeability.repeated_server import make_server
from motionlab.noticeability.workflow import export_training


@pytest.fixture
def repeated_pilot(pilot: Path) -> Path:
    before = {
        name: pilot.with_name(name).read_bytes()
        for name in ("pilot_manifest.json", "protocol_freeze.json", "motion.npz")
    }
    protocol.write_json(pilot.with_name("PILOT_PAUSED.json"), {"reason": "test continuation"})
    output = pilot.parent / "repeated"
    prepare_repeated_view_pilot(pilot, output)
    repeated_protocol.freeze_pilot(output / "pilot_manifest.json")
    assert all(pilot.with_name(name).read_bytes() == raw for name, raw in before.items())
    protocol.validate_pilot(pilot)
    return output / "pilot_manifest.json"


@contextmanager
def running(pilot_path: Path) -> Iterator[str]:
    server = make_server(pilot_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_three_viewings_mild_speed_and_separate_immutable_target(repeated_pilot: Path) -> None:
    with running(repeated_pilot) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1100, "height": 1200})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_enabled()
        assert page.locator("#preSpeed option").evaluate_all(
            "options => options.map(x => x.value)"
        ) == ["0.85", "0.9", "0.95", "1"]
        for invalid in (0.5, 0.84, 1.05, True, "0.85"):
            response = page.request.post(
                url + "/api/start", data={"trial_id": "t_0", "playback_rate": invalid}
            )
            assert response.status == 400
        response = page.request.post(
            url + "/api/notice", data={"trial_id": "t_0", "repeated_view_notice": "YES"}
        )
        assert response.status == 400
        for index, rate in enumerate(("1", "0.85", "0.9"), start=1):
            page.locator("#preSpeed").select_option(rate)
            page.locator("#watch").click()
            expect(page.locator("#preSpeed")).to_be_disabled()
            expect(page.locator("#primary")).to_be_hidden()
            expect(page.locator("#inspection")).to_be_hidden()
            response = page.request.post(url + "/api/complete", data={"trial_id": "t_0"})
            assert response.status == 400  # Server checks the speed-adjusted wall time.
            expect(page.locator("#primary")).to_be_visible(timeout=10000)
            expect(page.locator("#viewCount")).to_have_text(f"{index} of 3 viewings completed")
            expect(page.locator('#primaryAnswers [data-notice="YES"]')).to_be_enabled()
        expect(page.locator("#watch")).to_be_disabled()
        response = page.request.post(
            url + "/api/start", data={"trial_id": "t_0", "playback_rate": 1}
        )
        assert response.status == 400
        page.locator('#primaryAnswers [data-notice="YES"]').click()
        expect(page.locator("#inspection")).to_be_visible()
        expect(page.locator("#preControls")).to_be_hidden()
        primary_bytes = repeated_pilot.with_name("observations.jsonl").read_bytes()
        rows = repeated_protocol.authenticated_observations(repeated_pilot)
        assert len(rows) == 1 and rows[0]["repeated_view_notice"] == "YES"
        assert rows[0]["spontaneous_notice"] is None
        assert [v["playback_rate"] for v in rows[0]["pre_answer_viewings"]] == [1, 0.85, 0.9]
        assert all(
            v["wall_duration_s"] >= (v["content_duration_s"] - 2 / 60) / v["playback_rate"]
            for v in rows[0]["pre_answer_viewings"]
        )
        response = page.request.post(
            url + "/api/notice", data={"trial_id": "t_0", "repeated_view_notice": "NO"}
        )
        assert response.status == 400
        page.locator('#inspectedAnswers [data-notice="MAYBE"]').click()
        page.locator("#next").click()
        expect(page.locator("#watch")).to_be_enabled()
        assert repeated_pilot.with_name("observations.jsonl").read_bytes().startswith(primary_bytes)
        assert (
            repeated_protocol.authenticated_observations(repeated_pilot)[0]["inspected_notice"]
            == "MAYBE"
        )
        assert not errors
        browser.close()
    report = analyze_repeated(repeated_pilot, repeated_pilot.with_name("analysis.json"))
    assert report["pre_answer_viewing_count_distribution"] == {3: 1}
    assert report["target"] == "repeated_view_notice"
    exported = export_training(repeated_pilot, repeated_pilot.with_name("export.json"))
    assert exported["records"][0]["target"] == "repeated_view_notice"
    assert not exported["critic_retraining_permitted"]
    with pytest.raises(ValueError, match="cannot enter spontaneous"):
        matched_overlap({}, repeated_pilot)


def test_can_answer_after_one_view_and_resume_after_commit(repeated_pilot: Path) -> None:
    with running(repeated_pilot) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        page.goto(url)
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        page.locator("#preSpeed").select_option("0.85")
        page.locator("#watch").click()
        expect(page.locator("#primary")).to_be_visible(timeout=10000)
        page.locator('#primaryAnswers [data-notice="NO"]').click()
        expect(page.locator("#inspection")).to_be_visible()
        assert (
            len(
                repeated_protocol.authenticated_observations(repeated_pilot)[0][
                    "pre_answer_viewings"
                ]
            )
            == 1
        )
        page.reload()
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_enabled()
        assert page.evaluate("window.__noticeabilityState().trial.trial_id") == "t_1"
        browser.close()


def test_completed_single_view_answers_are_preserved_and_skipped(pilot: Path) -> None:
    direct_fixture(pilot)
    original = pilot.with_name("observations.jsonl").read_bytes()
    protocol.write_json(pilot.with_name("PILOT_PAUSED.json"), {"reason": "test continuation"})
    output = pilot.parent / "new_repeated"
    prepare_repeated_view_pilot(pilot, output)
    repeated_protocol.freeze_pilot(output / "pilot_manifest.json")
    with running(output / "pilot_manifest.json") as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        page.goto(url)
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_enabled()
        assert page.evaluate("window.__noticeabilityState().trial.trial_id") == "t_1"
        assert repeated_protocol.authenticated_observations(output / "pilot_manifest.json") == []
        assert pilot.with_name("observations.jsonl").read_bytes() == original
        assert protocol.authenticated_observations(pilot)[0]["spontaneous_notice"] == "YES"
        browser.close()


@pytest.mark.parametrize("rate", [0.5, 1.1, 0.86, float("nan"), True])
def test_schema_rejects_unpermitted_pre_answer_speeds(rate: float) -> None:
    with pytest.raises(ValueError):
        repeated_protocol.PreAnswerViewing(
            viewing_index=1,
            playback_rate=rate,
            content_duration_s=1.0,
            wall_duration_s=2.0,
            started_utc="2026-09-05T00:00:00+00:00",
            completed_utc="2026-09-05T00:00:02+00:00",
        )
