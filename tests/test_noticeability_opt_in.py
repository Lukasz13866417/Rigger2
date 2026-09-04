from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright
from test_noticeability import direct_fixture
from test_noticeability import pilot as pilot
from test_noticeability_repeated import repeated_pilot as repeated_pilot
from test_noticeability_repeated import running as running_repeated
from test_subjective_ui import _launch_browser

from motionlab.noticeability import opt_in_inspection, protocol, repeated_protocol
from motionlab.noticeability.repeated_continuation import prepare_repeated_view_pilot


@pytest.fixture
def opt_in_pilot(repeated_pilot: Path) -> Path:
    original = repeated_pilot.read_bytes()
    protocol.write_json(repeated_pilot.with_name("PILOT_PAUSED.json"), {"reason": "test opt-in"})
    output = repeated_pilot.parent / "opt_in"
    opt_in_inspection.prepare_pilot(repeated_pilot, output)
    repeated_protocol.freeze_pilot(output / "pilot_manifest.json")
    assert repeated_pilot.read_bytes() == original
    repeated_protocol.validate_pilot(repeated_pilot)
    return output / "pilot_manifest.json"


@contextmanager
def running(pilot_path: Path) -> Iterator[str]:
    server = opt_in_inspection.make_server(pilot_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_continue_never_opens_or_records_inspection(opt_in_pilot: Path) -> None:
    with running(opt_in_pilot) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1100, "height": 1200})
        page.goto(url)
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_enabled()
        expect(page.locator("#openInspection")).to_be_hidden()
        page.evaluate("document.getElementById('openInspection').click()")
        expect(page.locator("#inspection")).to_be_hidden()
        page.locator("#preSpeed").select_option("0.85")
        page.locator("#watch").click()
        expect(page.locator("#primary")).to_be_visible(timeout=10000)
        page.locator('#primaryAnswers [data-notice="YES"]').click()
        expect(page.locator("#continue")).to_be_visible()
        expect(page.locator("#openInspection")).to_be_enabled()
        expect(page.locator("#inspection")).to_be_hidden()
        expect(page.locator("#viewportWrap")).to_be_hidden()
        assert page.evaluate("inspectStart") == 0
        before = opt_in_pilot.with_name("observations.jsonl").read_bytes()
        assert len(before.splitlines()) == 1
        page.locator("#continue").click()
        expect(page.locator("#watch")).to_be_enabled()
        assert page.evaluate("window.__noticeabilityState().trial.trial_id") == "t_1"
        assert opt_in_pilot.with_name("observations.jsonl").read_bytes() == before
        row = repeated_protocol.authenticated_observations(opt_in_pilot)[0]
        assert row["repeated_view_notice"] == "YES"
        assert row["inspected_notice"] is None and row["inspection_observation_id"] is None
        assert row["inspection_telemetry"] is None
        browser.close()


def test_inspect_explicitly_opens_and_timer_starts_on_open(opt_in_pilot: Path) -> None:
    with running(opt_in_pilot) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1100, "height": 1200})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.locator("#begin").click()
        page.locator("#watch").click()
        expect(page.locator("#primary")).to_be_visible(timeout=10000)
        page.locator('#primaryAnswers [data-notice="MAYBE"]').click()
        expect(page.locator("#continue")).to_be_visible()
        expect(page.locator("#inspection")).to_be_hidden()
        before = opt_in_pilot.with_name("observations.jsonl").read_bytes()
        assert page.evaluate("inspectStart") == 0
        before_open = page.evaluate("performance.now()")
        page.locator("#openInspection").click()
        expect(page.locator("#inspection")).to_be_visible()
        expect(page.locator("#continue")).to_be_hidden()
        expect(page.locator("#view")).to_be_visible()
        assert page.evaluate("inspectStart") >= before_open
        page.locator("#replay").click()
        page.locator("#speed").select_option(".5")
        page.locator("#zoomOut").click()
        page.locator('#inspectedAnswers [data-notice="YES"]').click()
        page.locator("#next").click()
        expect(page.locator("#watch")).to_be_enabled()
        assert opt_in_pilot.with_name("observations.jsonl").read_bytes().startswith(before)
        row = repeated_protocol.authenticated_observations(opt_in_pilot)[0]
        assert row["repeated_view_notice"] == "MAYBE" and row["inspected_notice"] == "YES"
        assert row["replay_count"] == 1
        assert row["inspection_telemetry"]["zoom_change_count"] == 1
        assert not errors
        browser.close()


def test_cutover_preserves_progress_across_both_prior_protocols(pilot: Path) -> None:
    direct_fixture(pilot)
    original = pilot.with_name("observations.jsonl").read_bytes()
    protocol.write_json(pilot.with_name("PILOT_PAUSED.json"), {"reason": "test"})
    repeated_root = pilot.parent / "prior_repeated"
    prepare_repeated_view_pilot(pilot, repeated_root)
    repeated = repeated_root / "pilot_manifest.json"
    repeated_protocol.freeze_pilot(repeated)
    with running_repeated(repeated) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        page.goto(url)
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        page.locator("#watch").click()
        expect(page.locator("#primary")).to_be_visible(timeout=10000)
        page.locator('#primaryAnswers [data-notice="NO"]').click()
        expect(page.locator("#inspection")).to_be_visible()
        browser.close()
    repeated_bytes = repeated.with_name("observations.jsonl").read_bytes()
    protocol.write_json(repeated.with_name("PILOT_PAUSED.json"), {"reason": "test"})
    output = pilot.parent / "opt_in_continuation"
    opt_in_inspection.prepare_pilot(repeated, output)
    path = output / "pilot_manifest.json"
    repeated_protocol.freeze_pilot(path)
    history = protocol.read_json(output / "continuation_history.json")
    assert history["completed_trial_ids_by_rater"]["test"] == ["t_0", "t_1"]
    with running(path) as url, sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        page.goto(url)
        page.locator("#rater").fill("test")
        page.locator("#begin").click()
        expect(page.locator("#watch")).to_be_enabled()
        assert page.evaluate("window.__noticeabilityState().trial.trial_id") == "t_2"
        assert repeated_protocol.authenticated_observations(path) == []
        browser.close()
    assert pilot.with_name("observations.jsonl").read_bytes() == original
    assert repeated.with_name("observations.jsonl").read_bytes() == repeated_bytes
