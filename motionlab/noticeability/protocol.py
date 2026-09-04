"""Immutable noticeability contracts, evidence schemas, and content-addressed assets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from motionlab.dataset.io import file_sha256
from motionlab.perceptual import subjective

PROTOCOL = "motionlab.noticeability.v1"
NOTICE_VALUES = ("NO", "MAYBE", "YES")
Notice = Literal["NO", "MAYBE", "YES"]
POPULATIONS = {
    "CALIBRATION_ONLY",
    "NOTICEABILITY_THRESHOLD",
    "DEPLOYMENT_CANDIDATE",
    "ADVERSARIAL_CRITIC",
    "CLEAN_SHAM_CONTROL",
}
PRIMARY_QUESTION = (
    "During that normal viewing, did you notice anything that looked unintentionally "
    "wrong or unnatural?"
)
INSPECTION_QUESTION = "After inspecting it, can you identify an unintended problem?"
REASON_TAGS = (
    "foot/contact",
    "timing/rhythm",
    "knee/leg motion",
    "arm/leg coordination",
    "pelvis/torso coordination",
    "stiffness/rigidity",
    "over-smoothing",
    "anatomy/pose",
    "asymmetry",
    "style inconsistency",
    "other",
    "cannot identify",
)
BODY_PARTS = ("feet", "legs", "arms", "pelvis", "torso", "head", "whole body", "cannot identify")


def identity(prefix: str, value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"{prefix}:sha256:{hashlib.sha256(data).hexdigest()}"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Idempotent derived output: never overwrite a different artifact."""
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"incomplete append-only log: {path}")
    rows = [json.loads(line) for line in data.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("evidence rows must be objects")
    return rows


def append_event(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row["observation_id"] = identity("notice-observation", row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return row


class PrimaryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    protocol_version: Literal["motionlab.noticeability.v1"] = PROTOCOL  # type: ignore[assignment]
    event_type: Literal["spontaneous_notice"] = "spontaneous_notice"
    observation_id: str
    stimulus_id: str
    render_hash: str
    render_protocol_hash: str
    source_id: str
    variant_id: str
    rater_id: str
    session_id: str
    trial_id: str
    trial_index: int = Field(ge=0)
    spontaneous_notice: Notice
    spontaneous_response_time_ms: float = Field(ge=0, allow_inf_nan=False)
    initial_presentation_seconds: float = Field(ge=0, allow_inf_nan=False)
    inspected_notice: None = None
    replay_count: Literal[0] = 0
    inspection_telemetry: None = None
    optional_reason_tags: list[str] = Field(default_factory=list, max_length=0)
    optional_body_part: None = None
    optional_interval: None = None
    optional_confidence: None = None
    evaluation_goal: str
    style_context: str
    prior_exposure: bool
    timestamp_utc: str
    provenance: Literal["DIRECT_HUMAN"] = "DIRECT_HUMAN"


class InspectionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    protocol_version: Literal["motionlab.noticeability.v1"] = PROTOCOL  # type: ignore[assignment]
    event_type: Literal["inspection"] = "inspection"
    observation_id: str
    primary_observation_id: str
    inspected_notice: Notice | None
    inspection_telemetry: dict[str, Any]
    optional_reason_tags: list[str]
    optional_body_part: str | None
    optional_interval: list[float] | None
    optional_confidence: int | None
    timestamp_utc: str
    provenance: Literal["DIRECT_HUMAN"] = "DIRECT_HUMAN"


def inside(root: Path, name: str) -> Path:
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("asset/evidence path escapes pilot")
    return result


def collection_identity() -> dict[str, str]:
    # Analysis/model revisions do not silently change a running collection protocol.
    root = Path(__file__).parent
    names = ("protocol.py", "ui.py", "server.py", "staircase.py")
    return {
        **{name: file_sha256(root / name) for name in names},
        "original_renderer_and_camera": identity("renderer", subjective._HTML),
    }


def freeze_pilot(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    if manifest.get("protocol_version") != PROTOCOL:
        raise ValueError("not a noticeability pilot")
    assets = {manifest_path.name}
    for item in manifest["stimuli"]:
        if item["population"] not in POPULATIONS:
            raise ValueError("unknown noticeability population")
        subjective._resolve_viewing_settings(item)
        assets.update((item["motion"], item["phase_reference_motion"]))
    if manifest.get("legacy_registry"):
        assets.add(manifest["legacy_registry"])
    record = {
        "format_version": "motionlab.noticeability_freeze.v1",
        "protocol_version": PROTOCOL,
        "questions": {"spontaneous": PRIMARY_QUESTION, "inspected": INSPECTION_QUESTION},
        "responses": list(NOTICE_VALUES),
        "primary_before_inspection": True,
        "implementation": collection_identity(),
        "assets": {name: file_sha256(inside(root, name)) for name in sorted(assets)},
    }
    record["protocol_id"] = identity("notice-protocol", record)
    write_json(root / "protocol_freeze.json", record)
    return record


def validate_pilot(manifest_path: Path) -> dict[str, Any]:
    root = manifest_path.parent
    freeze = read_json(root / "protocol_freeze.json")
    if freeze["protocol_id"] != identity(
        "notice-protocol", {k: v for k, v in freeze.items() if k != "protocol_id"}
    ):
        raise ValueError("noticeability freeze changed")
    if freeze["implementation"] != collection_identity():
        raise ValueError("collection code changed; create a new protocol freeze/version")
    for name, digest in freeze["assets"].items():
        if file_sha256(inside(root, name)) != digest:
            raise ValueError(f"frozen pilot asset changed: {name}")
    if manifest_path.name not in freeze["assets"]:
        raise ValueError("manifest not part of freeze")
    return read_json(manifest_path)


def authenticated_observations(manifest_path: Path) -> list[dict[str, Any]]:
    """Join append-only optional inspection amendments without rewriting primary labels."""
    manifest = validate_pilot(manifest_path)
    root = manifest_path.parent
    stimuli = {r["stimulus_id"]: r for r in manifest["stimuli"]}
    trials = {r["trial_id"]: r for r in manifest["trials"]}
    served = {r["serve_id"]: r for r in read_rows(root / "served_playlist.jsonl")}
    for event in served.values():
        core = {k: v for k, v in event.items() if k not in {"serve_id", "observation_id"}}
        if event["serve_id"] != identity("notice-serve", core):
            raise ValueError("served evidence hash mismatch")
    primary: dict[str, dict[str, Any]] = {}
    completed: set[tuple[str, str]] = set()
    amended: set[str] = set()
    for row in read_rows(root / "observations.jsonl"):
        digest = identity(
            "notice-observation", {k: v for k, v in row.items() if k != "observation_id"}
        )
        if row.get("observation_id") != digest:
            raise ValueError("noticeability event hash mismatch")
        if row.get("event_type") == "spontaneous_notice":
            checked = PrimaryObservation.model_validate(row).model_dump()
            item = stimuli[checked["stimulus_id"]]
            trial = trials[checked["trial_id"]]
            key = (checked["rater_id"], checked["trial_id"])
            if key in completed or trial["stimulus_id"] != item["stimulus_id"]:
                raise ValueError("duplicate or wrong noticeability trial")
            matches = [
                s
                for s in served.values()
                if all(
                    s.get(k) == checked[k]
                    for k in ("rater_id", "session_id", "trial_id", "trial_index")
                )
            ]
            if {s["event_type"] for s in matches} != {"served", "started", "completed"} or checked[
                "trial_index"
            ] != trial["trial_index"]:
                raise ValueError("noticeability observation was not served")
            times = {s["event_type"]: datetime.fromisoformat(s["timestamp_utc"]) for s in matches}
            if (
                not times["served"]
                <= times["started"]
                <= times["completed"]
                <= datetime.fromisoformat(checked["timestamp_utc"])
            ):
                raise ValueError("invalid presentation event order")
            if (times["completed"] - times["started"]).total_seconds() < item[
                "duration_s"
            ] - 2 / item["fps"]:
                raise ValueError("server presentation duration incomplete")
            if trial.get("eligible_rater_id", checked["rater_id"]) != checked["rater_id"]:
                raise ValueError("overlap response belongs to a different rater")
            for key_name in (
                "render_hash",
                "render_protocol_hash",
                "source_id",
                "variant_id",
                "evaluation_goal",
                "style_context",
            ):
                if checked[key_name] != item[key_name]:
                    raise ValueError(f"noticeability evidence mismatch: {key_name}")
            if checked["initial_presentation_seconds"] < item["duration_s"] - 2 / item["fps"]:
                raise ValueError("initial presentation was incomplete")
            if checked["prior_exposure"] != bool(trial.get("prior_exposure", False)):
                raise ValueError("prior exposure mismatch")
            completed.add((checked["rater_id"], checked["trial_id"]))
            primary[digest] = {**checked, "inspection_observation_id": None}
        elif row.get("event_type") == "inspection":
            amendment = InspectionObservation.model_validate(row).model_dump()
            parent = amendment["primary_observation_id"]
            if parent not in primary or parent in amended:
                raise ValueError("inspection must follow exactly one primary judgment")
            item = stimuli[primary[parent]["stimulus_id"]]
            telemetry = subjective._validate_inspection_telemetry(
                amendment["inspection_telemetry"],
                stimulus_count=1,
                viewing_settings=item["viewing_settings"],
            )
            validate_optional(amendment, item["duration_s"])
            primary[parent].update(
                {
                    **{
                        k: amendment[k]
                        for k in (
                            "inspected_notice",
                            "optional_reason_tags",
                            "optional_body_part",
                            "optional_interval",
                            "optional_confidence",
                        )
                    },
                    "inspection_telemetry": telemetry,
                    "replay_count": telemetry["replay_count"][0],
                    "inspection_observation_id": digest,
                }
            )
            amended.add(parent)
        else:
            raise ValueError("unknown noticeability event type")
    return list(primary.values())


def validate_optional(value: dict[str, Any], duration: float) -> None:
    tags = value.get("optional_reason_tags", [])
    if (
        not isinstance(tags, list)
        or any(t not in REASON_TAGS for t in tags)
        or len(set(tags)) != len(tags)
    ):
        raise ValueError("invalid diagnostic reason tags")
    if value.get("optional_body_part") not in (*BODY_PARTS, None):
        raise ValueError("invalid body part")
    confidence = value.get("optional_confidence")
    if confidence is not None and (type(confidence) is not int or not 1 <= confidence <= 5):
        raise ValueError("confidence must be an integer 1-5 or missing")
    interval = value.get("optional_interval")
    if interval is not None and (
        not isinstance(interval, list)
        or not 1 <= len(interval) <= 2
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= duration
            for x in interval
        )
        or interval != sorted(interval)
    ):
        raise ValueError("invalid diagnostic time interval")
