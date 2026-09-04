"""Local-only append-only noticeability collector with server-side first-answer gates."""

from __future__ import annotations

import json
import secrets
import threading
import time
from datetime import UTC, datetime
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from motionlab.noticeability.protocol import (
    InspectionObservation,
    PrimaryObservation,
    append_event,
    authenticated_observations,
    identity,
    inside,
    now,
    read_rows,
    validate_optional,
    validate_pilot,
)
from motionlab.noticeability.staircase import select_interleaved
from motionlab.noticeability.ui import HTML
from motionlab.perceptual import subjective


def make_server(manifest_path: Path, *, port: int = 8765) -> ThreadingHTTPServer:
    manifest_path = manifest_path.resolve()
    manifest = validate_pilot(manifest_path)
    authenticated_observations(manifest_path)
    root = manifest_path.parent
    items = {s["stimulus_id"]: s for s in manifest["stimuli"]}
    sessions: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def reply(self, value: Any, status: int = 200, cookie: str | None = None) -> None:
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if cookie:
                self.send_header(
                    "Set-Cookie", f"notice_session={cookie}; HttpOnly; SameSite=Strict; Path=/"
                )
            self.end_headers()
            self.wfile.write(data)

        def local(self) -> None:
            expected = {
                f"localhost:{cast(ThreadingHTTPServer, self.server).server_port}",
                f"127.0.0.1:{cast(ThreadingHTTPServer, self.server).server_port}",
            }
            if self.headers.get("Host") not in expected:
                raise ValueError("local host required")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + x for x in expected}:
                raise ValueError("cross-origin requests are not permitted")
            if (root / "PILOT_PAUSED.json").exists():
                raise ValueError("noticeability pilot is paused")

        def do_GET(self) -> None:
            try:
                self.local()
                if self.path != "/":
                    self.reply({"error": "not found"}, 404)
                    return
                data = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except ValueError as exc:
                self.reply({"error": str(exc)}, 400)

        def do_POST(self) -> None:
            with lock:
                try:
                    self.local()
                    if self.headers.get("Content-Type") != "application/json":
                        raise ValueError("JSON required")
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 500_000:
                        raise ValueError("invalid request length")
                    value = json.loads(self.rfile.read(size))
                    if not isinstance(value, dict):
                        raise ValueError("object required")
                    route = self.path.removeprefix("/api/")
                    if route == "login":
                        rater = value.get("rater_id")
                        if not isinstance(rater, str) or not rater.strip() or len(rater) > 80:
                            raise ValueError("rater ID is required")
                        cookie = secrets.token_urlsafe(32)
                        # One live session per rater: prevent two tabs presenting the same trial.
                        for key in list(sessions):
                            if sessions[key]["rater_id"] == rater.strip():
                                del sessions[key]
                        sessions[cookie] = {
                            "rater_id": rater.strip(),
                            "session_id": identity("notice-session", cookie),
                            "active": None,
                        }
                        self.reply({"ok": True}, cookie=cookie)
                        return
                    cookies = SimpleCookie(self.headers.get("Cookie", ""))
                    token = cookies.get("notice_session")
                    if token is None or token.value not in sessions:
                        raise ValueError("session expired; reload and sign in")
                    state = sessions[token.value]
                    self.reply(self.dispatch(route, value, state))
                except (ValueError, KeyError, TypeError) as exc:
                    self.reply({"error": str(exc)}, 400)

        def dispatch(
            self, route: str, value: dict[str, Any], state: dict[str, Any]
        ) -> dict[str, Any]:
            if route == "next":
                if state["active"] and state["active"]["stage"] not in {
                    "inspection",
                    "done",
                    "interrupted",
                }:
                    raise ValueError("finish or explicitly interrupt the active trial first")
                events = read_rows(root / "observations.jsonl")
                done = {
                    r["trial_id"]
                    for r in events
                    if r["event_type"] == "spontaneous_notice"
                    and r["rater_id"] == state["rater_id"]
                }
                served = read_rows(root / "served_playlist.jsonl")
                # A crash/reload after exposure must not masquerade as a new initial viewing.
                exposed = {
                    r["trial_id"]
                    for r in served
                    if r["rater_id"] == state["rater_id"]
                    and r["event_type"] in {"started", "interrupted"}
                }
                eligible = [
                    t
                    for t in manifest["trials"]
                    if t.get("eligible_rater_id", state["rater_id"]) == state["rater_id"]
                    and t["trial_id"] not in done | exposed
                ]
                available = [
                    t
                    for t in eligible
                    if not t.get("not_before_utc")
                    or datetime.fromisoformat(t["not_before_utc"]) <= datetime.now(UTC)
                ]
                scheduler = manifest.get("scheduler")
                if scheduler:
                    trial_index = {t["trial_id"]: t for t in manifest["trials"]}
                    attempted = [trial_index[tid]["trial_index"] for tid in done | exposed]
                    step = max(attempted, default=-1) + 1
                    available = [t for t in available if t["trial_index"] == step]
                    if available and available[0]["track"] != "control":
                        history = [
                            {
                                "track": trial_index[r["trial_id"]]["track"],
                                "spontaneous_notice": r["spontaneous_notice"],
                            }
                            for r in events
                            if r["event_type"] == "spontaneous_notice"
                            and r["rater_id"] == state["rater_id"]
                            and trial_index[r["trial_id"]]["track"] != "control"
                        ]
                        track, _, sid = select_interleaved(
                            scheduler["tracks"], history, seed=manifest["seed"]
                        )
                        available = [
                            t for t in available if t["track"] == track and t["stimulus_id"] == sid
                        ]
                    # Unchosen frozen candidates are not unfinished trials.
                    eligible = available
                if not available:
                    return {
                        "trial": None,
                        "message": (
                            "Available trials complete. Overlap trials unlock at "
                            + min(t["not_before_utc"] for t in eligible)
                        )
                        if eligible
                        else (
                            "Pilot complete. Stop here for analysis and review; "
                            "no critic training will start automatically."
                        ),
                    }
                trial = available[0]
                item = items[trial["stimulus_id"]]
                payload = subjective._two_cycle_payload(
                    inside(root, item["motion"]),
                    phase_reference_path=inside(root, item["phase_reference_motion"]),
                    viewing_settings=item["viewing_settings"],
                )
                if payload["render_hash"] != item["base_render_hash"]:
                    raise ValueError("render differs from frozen stimulus")
                state["active"] = {"trial": trial, "item": item, "stage": "ready"}
                self.evidence(state, "served")
                return {
                    "trial": {
                        "trial_id": trial["trial_id"],
                        "display_index": len(done) + 1,
                        "total": scheduler["steps"] if scheduler else len(manifest["trials"]),
                        **{
                            k: item[k]
                            for k in (
                                "viewing_settings",
                                "view_mirror_x",
                                "duration_s",
                                "evaluation_goal",
                                "style_context",
                            )
                        },
                    },
                    "motion": payload,
                }
            active = state["active"]
            if active is None:
                raise ValueError("no active trial")
            trial, item = active["trial"], active["item"]
            if route != "inspection" and value.get("trial_id") != trial["trial_id"]:
                raise ValueError("wrong active trial")
            if route == "start":
                if active["stage"] != "ready":
                    raise ValueError("initial presentation may start only once")
                active.update(stage="initial", started=time.monotonic())
                self.evidence(state, "started")
            elif route == "complete":
                if (
                    active["stage"] != "initial"
                    or time.monotonic() - active["started"] < item["duration_s"] - 1 / item["fps"]
                ):
                    raise ValueError("full normal-speed presentation is required")
                active.update(stage="question", completed=time.monotonic())
                self.evidence(state, "completed")
            elif route == "interrupt":
                if active["stage"] in {"inspection", "done"}:
                    raise ValueError("cannot interrupt a committed answer")
                active["stage"] = "interrupted"
                self.evidence(state, "interrupted")
            elif route == "spontaneous":
                if active["stage"] != "question":
                    raise ValueError(
                        "primary answer requires a complete initial viewing and is immutable"
                    )
                row = PrimaryObservation(
                    observation_id="pending",
                    **{
                        k: item[k]
                        for k in (
                            "stimulus_id",
                            "render_hash",
                            "render_protocol_hash",
                            "source_id",
                            "variant_id",
                            "evaluation_goal",
                            "style_context",
                        )
                    },
                    rater_id=state["rater_id"],
                    session_id=state["session_id"],
                    trial_id=trial["trial_id"],
                    trial_index=trial["trial_index"],
                    spontaneous_notice=value["spontaneous_notice"],
                    spontaneous_response_time_ms=(time.monotonic() - active["completed"]) * 1000,
                    initial_presentation_seconds=float(item["duration_s"]),
                    prior_exposure=bool(trial.get("prior_exposure", False)),
                    timestamp_utc=now(),
                ).model_dump(exclude={"observation_id"})
                saved = append_event(root / "observations.jsonl", row)
                active.update(stage="inspection", primary_id=saved["observation_id"])
                return saved
            elif route == "inspection":
                if (
                    active["stage"] != "inspection"
                    or value.get("primary_observation_id") != active["primary_id"]
                ):
                    raise ValueError("inspection requires an already committed primary answer")
                validate_optional(value, item["duration_s"])
                subjective._validate_inspection_telemetry(
                    value["inspection_telemetry"],
                    stimulus_count=1,
                    viewing_settings=item["viewing_settings"],
                )
                row = InspectionObservation(
                    observation_id="pending", timestamp_utc=now(), **value
                ).model_dump(exclude={"observation_id"})
                saved = append_event(root / "observations.jsonl", row)
                active["stage"] = "done"
                return saved
            else:
                raise ValueError("unknown endpoint")
            return {"ok": True}

        def evidence(self, state: dict[str, Any], event: str) -> None:
            trial = state["active"]["trial"]
            row = {
                "event_type": event,
                "rater_id": state["rater_id"],
                "session_id": state["session_id"],
                "trial_id": trial["trial_id"],
                "trial_index": trial["trial_index"],
                "timestamp_utc": now(),
            }
            row["serve_id"] = identity("notice-serve", row)
            append_event(root / "served_playlist.jsonl", row)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_pilot(manifest_path: Path, *, port: int = 8765) -> None:
    with make_server(manifest_path, port=port) as server:
        print(f"Noticeability pilot: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()
