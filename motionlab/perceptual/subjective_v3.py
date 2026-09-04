"""Calibrated, blinded single-stimulus and same-viewport motion evaluation."""

# ruff: noqa: E501 -- the embedded application is intentionally self-contained.

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from motionlab.io.npz import load_motion_npz
from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.processing.contacts import detect_foot_contacts
from motionlab.processing.phase import estimate_gait_phase

LEGACY_SUBJECTIVE_PROTOCOL_VERSION = "motionlab.subjective_protocol.v2"
SUBJECTIVE_PROTOCOL_VERSION = "motionlab.subjective_protocol.v3"
SUPPORTED_SUBJECTIVE_PROTOCOL_VERSIONS = {
    LEGACY_SUBJECTIVE_PROTOCOL_VERSION,
    SUBJECTIVE_PROTOCOL_VERSION,
}
LEGACY_RAW_OBSERVATION_VERSION = "motionlab.subjective_observation.v1"
RAW_OBSERVATION_VERSION = "motionlab.subjective_observation.v2"
SUPPORTED_RAW_OBSERVATION_VERSIONS = {
    LEGACY_RAW_OBSERVATION_VERSION,
    RAW_OBSERVATION_VERSION,
}
LEGACY_PILOT_MANIFEST_VERSION = "motionlab.subjective_pilot.v1"
PILOT_MANIFEST_VERSION = "motionlab.subjective_pilot.v2"
SUPPORTED_PILOT_MANIFEST_VERSIONS = {
    LEGACY_PILOT_MANIFEST_VERSION,
    PILOT_MANIFEST_VERSION,
}
INSPECTION_TELEMETRY_VERSION = "motionlab.inspection_telemetry.v1"
JUDGMENT_DIFFICULTIES = {"easy", "moderate", "difficult"}

NATURALNESS_SCALE = (
    "Clearly broken / highly unnatural",
    "Very unnatural",
    "Noticeably unnatural",
    "Acceptable but visibly mediocre",
    "Good and mostly natural",
    "Very good with only subtle issues",
    "Excellent / difficult to improve",
)
STYLE_SCALE = (
    "Clearly contradicts the requested style",
    "Very poor style match",
    "Weak style match",
    "Ambiguous style match",
    "Mostly matches the style",
    "Very strong style match",
    "Exceptional style match",
)
PAIR_OUTCOMES = {"a_better", "b_better", "effectively_equal", "not_sure"}
REASON_TAGS = {
    "naturalness",
    "coordination",
    "style",
    "weight_transfer",
    "rigidity_smoothing",
    "anatomical_plausibility",
    "timing",
    "other",
}

LEGACY_SKELETON_VIEWING_SETTINGS: dict[str, Any] = {
    "protocol": "motionlab.fixed_skeleton_canvas.v2",
    "character": "fixed_rig_neutral_skeleton_v1",
    "viewport_width_px": 960,
    "viewport_height_px": 640,
    "resolution_css": "960x640",
    "frame_rate_hz": 60.0,
    "playback_rate": 1.0,
    "gait_cycles": 2.0,
    "camera_sequence": [{"normalized_time": 0.0, "yaw_degrees": 25.0}],
    "camera_projection": "orthographic",
    "pixels_per_meter": 285.0,
    "ground": "y=0_grid_0.25m",
    "lighting": "constant_high_contrast_skeleton_v1",
    "root_horizontal_tracking": True,
    "background_rgb": [18, 21, 27],
}

MANNEQUIN_VIEWING_SETTINGS: dict[str, Any] = {
    "protocol": "motionlab.fixed_mannequin_canvas.v1",
    "renderer_revision": "canvas2d_oriented_solids.v1",
    "character": "fixed_rig_neutral_mannequin_v1",
    "geometry_revision": "neutral_anatomical_solids.v1",
    "viewport_width_px": 960,
    "viewport_height_px": 640,
    "resolution_css": "960x640",
    "frame_rate_hz": 60.0,
    "playback_rate": 1.0,
    "playback_rates": [0.25, 0.5, 1.0, 1.5, 2.0],
    "gait_cycles": 2.0,
    "default_camera": {
        "preset": "three_quarter",
        "yaw_degrees": 25.0,
        "pitch_degrees": -7.0,
        "zoom": 1.0,
        "skeleton_overlay": False,
    },
    "camera_presets": {
        "front": {"yaw_degrees": 0.0, "pitch_degrees": -7.0},
        "three_quarter": {"yaw_degrees": 25.0, "pitch_degrees": -7.0},
        "side": {"yaw_degrees": 90.0, "pitch_degrees": -7.0},
    },
    "camera_projection": "orthographic",
    "pixels_per_meter": 285.0,
    "zoom_limits": {"minimum": 0.65, "maximum": 1.8, "step": 0.15},
    "orbit": {"enabled": True, "maximum_pitch_degrees": 35.0},
    "ground": "y=0_projected_grid_0.25m",
    "lighting": "constant_neutral_three_face_shading_v1",
    "material": "fixed_blue_gray_orientation_palette_v1",
    "root_horizontal_tracking": True,
    "background_rgb": [18, 21, 27],
    "skeleton_overlay": {"available": True, "default": False, "locked_until_first_pass": True},
    "first_pass": {
        "playback_rate": 1.0,
        "camera": "default_camera",
        "require_complete_server_ack": True,
    },
    "post_rating_difficulty": True,
}

# The current preparation default. The legacy settings remain independently addressable and
# byte-for-byte hash-compatible with every already-served skeleton observation.
VIEWING_SETTINGS = MANNEQUIN_VIEWING_SETTINGS
VIEWING_PROTOCOLS: dict[str, dict[str, Any]] = {
    LEGACY_SKELETON_VIEWING_SETTINGS["protocol"]: LEGACY_SKELETON_VIEWING_SETTINGS,
    MANNEQUIN_VIEWING_SETTINGS["protocol"]: MANNEQUIN_VIEWING_SETTINGS,
}


def _canonical_hash(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def render_protocol_hash(viewing_settings: dict[str, Any] | None = None) -> str:
    """Return the immutable rendering-protocol identifier stored with every judgment."""
    return _canonical_hash(
        "render",
        VIEWING_SETTINGS if viewing_settings is None else viewing_settings,
    )


def _public_stimulus_id(
    motion_path: str,
    viewing_settings: dict[str, Any] | None = None,
) -> str:
    return _canonical_hash(
        "stimulus",
        {
            "motion": motion_path,
            "render": render_protocol_hash(viewing_settings),
        },
    )


def _resolve_viewing_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    raw = manifest.get("viewing_settings")
    if not isinstance(raw, dict):
        raise ValueError("subjective pilot has invalid viewing settings")
    protocol = str(raw.get("protocol", ""))
    canonical = VIEWING_PROTOCOLS.get(protocol)
    if canonical is None or raw != canonical:
        raise ValueError("subjective pilot uses an unknown or stale rendering protocol")
    if manifest.get("render_protocol_hash") != render_protocol_hash(canonical):
        raise ValueError("subjective pilot rendering hash does not match its protocol")
    return canonical


def _default_camera_state(viewing_settings: dict[str, Any]) -> dict[str, Any]:
    raw = viewing_settings.get("default_camera")
    if not isinstance(raw, dict):
        return {
            "preset": "legacy",
            "yaw_degrees": 25.0,
            "pitch_degrees": 0.0,
            "zoom": 1.0,
            "skeleton_overlay": False,
        }
    return {
        "preset": str(raw["preset"]),
        "yaw_degrees": float(raw["yaw_degrees"]),
        "pitch_degrees": float(raw["pitch_degrees"]),
        "zoom": float(raw["zoom"]),
        "skeleton_overlay": bool(raw["skeleton_overlay"]),
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_camera_state(
    value: Any,
    viewing_settings: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("camera state must be an object")
    required = {
        "preset",
        "yaw_degrees",
        "pitch_degrees",
        "zoom",
        "skeleton_overlay",
    }
    if set(value) != required:
        raise ValueError("camera state has unsupported fields")
    if not isinstance(value["preset"], str) or not value["preset"]:
        raise ValueError("camera preset must be a nonempty string")
    if not all(_finite_number(value[key]) for key in ("yaw_degrees", "pitch_degrees", "zoom")):
        raise ValueError("camera angles and zoom must be finite numbers")
    if not isinstance(value["skeleton_overlay"], bool):
        raise ValueError("camera skeleton overlay flag must be boolean")
    zoom = float(value["zoom"])
    limits = viewing_settings.get("zoom_limits", {"minimum": 1.0, "maximum": 1.0})
    if not float(limits["minimum"]) <= zoom <= float(limits["maximum"]):
        raise ValueError("camera zoom is outside the permitted range")
    maximum_pitch = float(viewing_settings.get("orbit", {}).get("maximum_pitch_degrees", 0.0))
    if abs(float(value["pitch_degrees"])) > max(
        maximum_pitch, abs(_default_camera_state(viewing_settings)["pitch_degrees"])
    ):
        raise ValueError("camera pitch is outside the permitted range")
    return {
        "preset": value["preset"],
        "yaw_degrees": float(value["yaw_degrees"]),
        "pitch_degrees": float(value["pitch_degrees"]),
        "zoom": zoom,
        "skeleton_overlay": value["skeleton_overlay"],
    }


def _camera_is_default(
    value: dict[str, Any],
    viewing_settings: dict[str, Any],
) -> bool:
    expected = _default_camera_state(viewing_settings)
    return (
        value["preset"] == expected["preset"]
        and math.isclose(value["yaw_degrees"], expected["yaw_degrees"], abs_tol=1.0e-9)
        and math.isclose(value["pitch_degrees"], expected["pitch_degrees"], abs_tol=1.0e-9)
        and math.isclose(value["zoom"], expected["zoom"], abs_tol=1.0e-9)
        and value["skeleton_overlay"] is expected["skeleton_overlay"]
    )


def _validate_inspection_telemetry(
    value: Any,
    *,
    stimulus_count: int,
    viewing_settings: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("inspection telemetry must be an object")
    required = {
        "format_version",
        "replay_count",
        "speed_change_events",
        "playback_wall_time_by_speed_ms",
        "camera_change_events",
        "zoom_change_count",
        "view_change_count",
        "overlay_change_count",
        "total_inspection_time_ms",
        "final_playback_rate",
        "final_camera_state",
    }
    if set(value) != required:
        raise ValueError("inspection telemetry has unsupported fields")
    if value["format_version"] != INSPECTION_TELEMETRY_VERSION:
        raise ValueError("unsupported inspection telemetry version")
    replay_count = value["replay_count"]
    if (
        not isinstance(replay_count, list)
        or len(replay_count) != stimulus_count
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in replay_count
        )
    ):
        raise ValueError("inspection replay count is invalid")
    permitted_rates = {float(rate) for rate in viewing_settings.get("playback_rates", [1.0])}
    speed_events = value["speed_change_events"]
    if not isinstance(speed_events, list) or len(speed_events) > 512:
        raise ValueError("speed change events must be a bounded list")
    normalized_speed_events: list[dict[str, float]] = []
    for event in speed_events:
        if not isinstance(event, dict) or set(event) != {"elapsed_ms", "from_rate", "to_rate"}:
            raise ValueError("speed change event has unsupported fields")
        if not all(_finite_number(event[key]) for key in event):
            raise ValueError("speed change event values must be finite")
        if float(event["elapsed_ms"]) < 0.0:
            raise ValueError("speed change event elapsed time must be nonnegative")
        if (
            float(event["from_rate"]) not in permitted_rates
            or float(event["to_rate"]) not in permitted_rates
        ):
            raise ValueError("speed change event uses an unsupported playback rate")
        normalized_speed_events.append({key: float(event[key]) for key in event})
    time_by_speed = value["playback_wall_time_by_speed_ms"]
    expected_rate_keys = {f"{rate:g}" for rate in sorted(permitted_rates)}
    if (
        not isinstance(time_by_speed, dict)
        or set(time_by_speed) != expected_rate_keys
        or any(not _finite_number(item) or float(item) < 0.0 for item in time_by_speed.values())
    ):
        raise ValueError("playback time by speed is invalid")
    camera_events = value["camera_change_events"]
    permitted_camera_events = {"zoom", "view", "reset", "orbit", "overlay"}
    if not isinstance(camera_events, list) or len(camera_events) > 512:
        raise ValueError("camera change events must be a bounded list")
    for event in camera_events:
        if not isinstance(event, dict) or set(event) != {"elapsed_ms", "kind", "from", "to"}:
            raise ValueError("camera change event has unsupported fields")
        if not _finite_number(event["elapsed_ms"]) or float(event["elapsed_ms"]) < 0.0:
            raise ValueError("camera change event elapsed time must be nonnegative")
        if event["kind"] not in permitted_camera_events:
            raise ValueError("unsupported camera change event kind")
    counts: dict[str, int] = {}
    for key in ("zoom_change_count", "view_change_count", "overlay_change_count"):
        raw_count = value[key]
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        counts[key] = raw_count
    expected_counts = {
        "zoom_change_count": sum(event["kind"] == "zoom" for event in camera_events),
        "view_change_count": sum(event["kind"] == "view" for event in camera_events),
        "overlay_change_count": sum(event["kind"] == "overlay" for event in camera_events),
    }
    if counts != expected_counts:
        raise ValueError("camera change counts do not match camera event kinds")
    if (
        not _finite_number(value["total_inspection_time_ms"])
        or float(value["total_inspection_time_ms"]) < 0.0
    ):
        raise ValueError("total inspection time must be nonnegative")
    if (
        not _finite_number(value["final_playback_rate"])
        or float(value["final_playback_rate"]) not in permitted_rates
    ):
        raise ValueError("final playback rate is unsupported")
    return {
        "format_version": INSPECTION_TELEMETRY_VERSION,
        "replay_count": list(replay_count),
        "speed_change_events": normalized_speed_events,
        "playback_wall_time_by_speed_ms": {
            key: float(time_by_speed[key]) for key in sorted(expected_rate_keys)
        },
        "camera_change_events": camera_events,
        **counts,
        "total_inspection_time_ms": float(value["total_inspection_time_ms"]),
        "final_playback_rate": float(value["final_playback_rate"]),
        "final_camera_state": _validate_camera_state(
            value["final_camera_state"],
            viewing_settings,
        ),
    }


def _side_class(family: str) -> str:
    if "asymmetric" in family:
        return "left"
    if family in {"equivalent_origin_shift", "preserved_critic_exploit"}:
        return "neutral"
    return "bilateral"


def _quality_stratum(audit: dict[str, Any], pair: dict[str, Any], variant_kind: str) -> str:
    if variant_kind == "clean_reference":
        return "excellent_candidate"
    if pair.get("preference") == "approximately_equal":
        return "equivalent_candidate"
    difficulty = str(audit.get("difficulty", "medium"))
    return {
        "clear": "poor_candidate",
        "medium": "mid_candidate",
        "subtle": "near_reference_candidate",
    }.get(difficulty, "mid_candidate")


def _stimulus_record(
    pair: dict[str, Any],
    audit: dict[str, Any],
    *,
    motion_key: str,
    variant_kind: str,
    viewing_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = VIEWING_SETTINGS if viewing_settings is None else viewing_settings
    motion_path = str(pair[motion_key])
    stimulus_id = _public_stimulus_id(motion_path, settings)
    return {
        "stimulus_id": stimulus_id,
        "source_id": str(pair["source_clip_id"]),
        "variant_id": f"{pair['pair_id']}:{'a' if motion_key == 'motion_a' else 'b'}",
        "motion": motion_path,
        "phase_reference_motion": str(pair["motion_a"]),
        "variant_kind": variant_kind,
        "family": "clean_reference"
        if variant_kind == "clean_reference"
        else str(pair["perturbation_mechanism"]),
        "style": str(pair["source_clip_id"]),
        "side_class": (
            "neutral"
            if variant_kind == "clean_reference"
            else _side_class(str(pair["perturbation_mechanism"]))
        ),
        "expected_quality_stratum": _quality_stratum(audit, pair, variant_kind),
        "hidden_repeat_group_id": _canonical_hash("repeat", {"stimulus_id": stimulus_id}),
        "render_protocol_hash": render_protocol_hash(settings),
        "selection_reasons": list(audit.get("selection_reasons", [])),
        "optimization_output": bool(pair.get("adversarial_origin", False)),
    }


def _balanced_candidate_selection(
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
    count: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for audit, pair in candidates:
        buckets[str(pair["perturbation_mechanism"])].append((audit, pair))
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    family_order = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        rng.shuffle(family_order)
        for family in family_order:
            if buckets[family] and len(selected) < count:
                # Prefer a source different from the immediately preceding source.
                choices = buckets[family]
                position = 0
                if selected:
                    previous_source = selected[-1][1]["source_clip_id"]
                    position = next(
                        (
                            i
                            for i, value in enumerate(choices)
                            if value[1]["source_clip_id"] != previous_source
                        ),
                        0,
                    )
                selected.append(choices.pop(position))
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} eligible candidates exist for a {count}-candidate pilot"
        )
    return selected


def _constrained_order(
    trials: list[dict[str, Any]],
    *,
    rng: random.Random,
    minimum_repeat_gap: int,
    previously_seen_repeat_groups: set[str] | None = None,
) -> list[dict[str, Any]]:
    # Restarted randomized construction avoids a late repeat-spacing dead end while keeping the
    # exact result reproducible from the caller's seed.
    prior_groups = set() if previously_seen_repeat_groups is None else previously_seen_repeat_groups
    for _ in range(500):
        remaining = trials[:]
        rng.shuffle(remaining)
        ordered: list[dict[str, Any]] = []
        last_repeat_position: dict[str, int] = {}
        while remaining:
            repeat_valid = [
                trial
                for trial in remaining
                if all(
                    len(ordered) - last_repeat_position.get(group, -10_000) >= minimum_repeat_gap
                    for group in trial["hidden_repeat_group_ids"]
                )
                and (
                    int(trial.get("repeat_occurrence_index", 0)) == 0
                    or any(
                        group in prior_groups or group in last_repeat_position
                        for group in trial["hidden_repeat_group_ids"]
                    )
                )
            ]
            if not repeat_valid:
                break
            constrained = [
                trial
                for trial in repeat_valid
                if (not ordered or trial["source_ids"] != ordered[-1]["source_ids"])
                and not (
                    len(ordered) >= 2
                    and trial["families"] == ordered[-1]["families"]
                    and trial["families"] == ordered[-2]["families"]
                )
            ]
            valid = constrained or repeat_valid
            quality_counts = Counter(
                tuple(item["expected_quality_strata"]) for item in ordered[-8:]
            )
            minimum_quality_count = min(
                quality_counts[tuple(item["expected_quality_strata"])] for item in valid
            )
            preferred = [
                item
                for item in valid
                if quality_counts[tuple(item["expected_quality_strata"])] == minimum_quality_count
            ]
            chosen = rng.choice(preferred)
            remaining.remove(chosen)
            position = len(ordered)
            for group in chosen["hidden_repeat_group_ids"]:
                last_repeat_position[group] = position
            ordered.append(chosen)
        if not remaining:
            return ordered
    raise ValueError("cannot meet the declared hidden-repeat spacing after 500 schedules")


def prepare_subjective_pilot(
    pair_dataset_directory: Path,
    audit_queue_path: Path,
    output_directory: Path,
    *,
    seed: int = 9401,
    unique_stimulus_count: int = 42,
    session_count: int = 2,
    minimum_repeat_gap: int = 12,
    viewing_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare the frozen legacy skeleton pilot without creating human evidence.

    The revised mannequin protocol has a different population and tutorial contract and must be
    built with :func:`prepare_mannequin_subjective_pilot`.  Keeping this entry point explicitly
    legacy prevents an old schedule from being mislabeled as a current mannequin pilot.
    """
    output = Path(output_directory)
    protected = [
        output / "raw_observations.jsonl",
        output / "served_playlist.jsonl",
        output / "PILOT_PAUSED.json",
    ]
    existing_protected = [path.name for path in protected if path.exists()]
    if existing_protected:
        raise ValueError(
            "refusing to prepare into a collected or paused pilot directory: "
            + ", ".join(existing_protected)
        )
    settings = LEGACY_SKELETON_VIEWING_SETTINGS if viewing_settings is None else viewing_settings
    canonical_settings = VIEWING_PROTOCOLS.get(str(settings.get("protocol", "")))
    if canonical_settings is None or settings != canonical_settings:
        raise ValueError("pilot preparation requires a registered rendering protocol")
    if settings != LEGACY_SKELETON_VIEWING_SETTINGS:
        raise ValueError(
            "the legacy pilot builder only supports the frozen skeleton protocol; "
            "use prepare_mannequin_subjective_pilot for mannequin collection"
        )
    if not 30 <= unique_stimulus_count <= 50:
        raise ValueError("pilot must contain 30 to 50 unique animations")
    if session_count != 2:
        raise ValueError("the first pilot is intentionally fixed to two sessions")
    rng = random.Random(seed)
    pairs = load_perceptual_pairs(pair_dataset_directory)
    by_pair = {str(row["pair_id"]): row for row in pairs}
    audit_rows = _read_jsonl(audit_queue_path)
    candidates = [
        (row, by_pair[str(row["pair_id"])]) for row in audit_rows if str(row["pair_id"]) in by_pair
    ]
    if not candidates:
        raise ValueError("audit queue contains no known perceptual pairs")

    # Every available source gets one neutral reference; the remaining slots are family-balanced variants.
    first_by_clean_path: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for audit, pair in candidates:
        first_by_clean_path.setdefault(str(pair["motion_a"]), (audit, pair))
    clean_items = list(first_by_clean_path.values())
    rng.shuffle(clean_items)
    if len(clean_items) >= unique_stimulus_count:
        raise ValueError("pilot needs room for both reference and variant animations")
    selected_candidates = _balanced_candidate_selection(
        candidates, unique_stimulus_count - len(clean_items), rng
    )

    stimuli: dict[str, dict[str, Any]] = {}
    stimulus_pair: dict[str, str] = {}
    for audit, pair in clean_items:
        item = _stimulus_record(
            pair,
            audit,
            motion_key="motion_a",
            variant_kind="clean_reference",
            viewing_settings=settings,
        )
        stimuli[item["stimulus_id"]] = item
        stimulus_pair[item["stimulus_id"]] = str(pair["pair_id"])
    for audit, pair in selected_candidates:
        item = _stimulus_record(
            pair,
            audit,
            motion_key="motion_b",
            variant_kind="candidate",
            viewing_settings=settings,
        )
        stimuli[item["stimulus_id"]] = item
        stimulus_pair[item["stimulus_id"]] = str(pair["pair_id"])
    if len(stimuli) != unique_stimulus_count:
        raise ValueError("selected pilot motions were not unique")
    unilateral_items = sorted(
        (item for item in stimuli.values() if item["side_class"] == "left"),
        key=lambda item: item["stimulus_id"],
    )
    for index, item in enumerate(unilateral_items):
        item["display_mirror_x"] = index % 2 == 1
    for item in stimuli.values():
        item.setdefault("display_mirror_x", False)

    excellent = [item for item in stimuli.values() if item["variant_kind"] == "clean_reference"]
    poor = [
        item for item in stimuli.values() if item["expected_quality_stratum"] == "poor_candidate"
    ]
    rng.shuffle(excellent)
    rng.shuffle(poor)
    anchors = excellent[:3] + poor[:3]
    if len(anchors) < 6:
        raise ValueError("pilot lacks six hidden anchor candidates spanning the expected range")
    anchor_ids = {item["stimulus_id"] for item in anchors}

    unique_items = list(stimuli.values())
    rng.shuffle(unique_items)
    # Three anchors originate in each session; the rest are greedily balanced by source,
    # family, and expected quality before the presentation-order constraints are applied.
    session_unique: list[list[dict[str, Any]]] = [[], []]
    capacities = [(unique_stimulus_count + 1) // 2, unique_stimulus_count // 2]
    for index, anchor in enumerate(anchors):
        session_unique[index % 2].append(anchor)
    for item in [value for value in unique_items if value not in anchors]:
        eligible = [index for index in range(2) if len(session_unique[index]) < capacities[index]]
        chosen_session = min(
            eligible,
            key=lambda index: (
                sum(value["source_id"] == item["source_id"] for value in session_unique[index]),
                sum(value["family"] == item["family"] for value in session_unique[index]),
                sum(
                    value["expected_quality_stratum"] == item["expected_quality_stratum"]
                    for value in session_unique[index]
                ),
                len(session_unique[index]),
                rng.random(),
            ),
        )
        session_unique[chosen_session].append(item)

    def single_trial(item: dict[str, Any], session_index: int, occurrence: int) -> dict[str, Any]:
        trial_key = {
            "stimulus": item["stimulus_id"],
            "session": session_index,
            "occurrence": occurrence,
            "seed": seed,
        }
        return {
            "trial_id": _canonical_hash("trial", trial_key),
            "task_type": "naturalness",
            "stimulus_ids": [item["stimulus_id"]],
            "source_ids": [item["source_id"]],
            "families": [item["family"]],
            "styles": [item["style"]],
            "side_classes": [
                "right"
                if item["side_class"] == "left" and item["display_mirror_x"]
                else item["side_class"]
            ],
            "expected_quality_strata": [item["expected_quality_stratum"]],
            "hidden_repeat_group_ids": [item["hidden_repeat_group_id"]],
            "hidden_anchor": item["stimulus_id"] in anchor_ids,
            "comparison_mode": None,
            "presentation_order": [item["stimulus_id"]],
            "view_mirror_x": item["display_mirror_x"],
            "pair_id": stimulus_pair[item["stimulus_id"]],
            "schedule_reason": "broad_single_stimulus"
            if occurrence == 0
            else "hidden_exact_repeat",
            "repeat_occurrence_index": occurrence,
            "session_index": session_index,
            "seed": seed,
        }

    session_trials: list[list[dict[str, Any]]] = [[], []]
    occurrence_count: Counter[str] = Counter()
    for session_index, items in enumerate(session_unique):
        for item in items:
            session_trials[session_index].append(
                single_trial(item, session_index, occurrence_count[item["stimulus_id"]])
            )
            occurrence_count[item["stimulus_id"]] += 1
    for anchor in anchors:
        original_session = 0 if anchor in session_unique[0] else 1
        target_session = 1 - original_session
        session_trials[target_session].append(
            single_trial(anchor, target_session, occurrence_count[anchor["stimulus_id"]])
        )
        occurrence_count[anchor["stimulus_id"]] += 1
    # Add one high-information, non-anchor within-session repeat to each session.
    for session_index in range(2):
        important = next(
            item
            for item in reversed(session_unique[session_index])
            if item["stimulus_id"] not in anchor_ids
            and (item["selection_reasons"] or item["optimization_output"])
        )
        session_trials[session_index].append(
            single_trial(important, session_index, occurrence_count[important["stimulus_id"]])
        )
        occurrence_count[important["stimulus_id"]] += 1

    selected_by_pair = {str(pair["pair_id"]): (audit, pair) for audit, pair in selected_candidates}
    comparison_pool = list(selected_by_pair.values())
    comparison_pool.sort(
        key=lambda value: (
            0 if value[1].get("preference") == "approximately_equal" else 1,
            0 if value[0].get("selection_reasons") else 1,
            str(value[1]["perturbation_mechanism"]),
        )
    )
    chosen_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_families: Counter[str] = Counter()
    for _ in range(12):
        available = [value for value in comparison_pool if value not in chosen_pairs]
        available.sort(
            key=lambda value: (used_families[str(value[1]["perturbation_mechanism"])], rng.random())
        )
        chosen = available[0]
        chosen_pairs.append(chosen)
        used_families[str(chosen[1]["perturbation_mechanism"])] += 1
    clean_first_flags = [True] * 6 + [False] * 6
    rng.shuffle(clean_first_flags)
    asymmetric_pair_count = 0
    for index, (audit, pair) in enumerate(chosen_pairs):
        clean_item = _stimulus_record(
            pair,
            audit,
            motion_key="motion_a",
            variant_kind="clean_reference",
            viewing_settings=settings,
        )
        candidate_item = _stimulus_record(
            pair,
            audit,
            motion_key="motion_b",
            variant_kind="candidate",
            viewing_settings=settings,
        )
        stimuli.setdefault(clean_item["stimulus_id"], clean_item)
        stimuli.setdefault(candidate_item["stimulus_id"], candidate_item)
        display_order = [clean_item["stimulus_id"], candidate_item["stimulus_id"]]
        if not clean_first_flags[index]:
            display_order.reverse()
        session_index = index % 2
        mode = "same_viewport_toggle" if index % 4 in {0, 3} else "sequential_neutral_gap"
        asymmetric = str(pair["perturbation_mechanism"]) == "asymmetric_limb_timing"
        view_mirror_x = asymmetric and asymmetric_pair_count % 2 == 1
        if asymmetric:
            asymmetric_pair_count += 1
        trial_key = {"pair": pair["pair_id"], "session": session_index, "mode": mode, "seed": seed}
        session_trials[session_index].append(
            {
                "trial_id": _canonical_hash("trial", trial_key),
                "task_type": "pair_comparison",
                "stimulus_ids": [clean_item["stimulus_id"], candidate_item["stimulus_id"]],
                "source_ids": [str(pair["source_clip_id"])],
                "families": [str(pair["perturbation_mechanism"])],
                "styles": [str(pair["source_clip_id"])],
                "side_classes": [
                    "right"
                    if _side_class(str(pair["perturbation_mechanism"])) == "left" and view_mirror_x
                    else _side_class(str(pair["perturbation_mechanism"]))
                ],
                "expected_quality_strata": [
                    _quality_stratum(audit, pair, "clean_reference"),
                    _quality_stratum(audit, pair, "candidate"),
                ],
                "hidden_repeat_group_ids": [
                    _canonical_hash("pair-repeat", {"pair": pair["pair_id"], "mode": mode})
                ],
                "hidden_anchor": False,
                "comparison_mode": mode,
                "presentation_order": display_order,
                "view_mirror_x": view_mirror_x,
                "pair_id": str(pair["pair_id"]),
                "equivalence_control": pair.get("preference") == "approximately_equal",
                "schedule_reason": "matched_same_source_comparison",
                "repeat_occurrence_index": 0,
                "session_index": session_index,
                "seed": seed,
            }
        )

    # Define occurrence order by session chronology before randomizing within each block.
    global_occurrence: Counter[str] = Counter()
    for trials in session_trials:
        for trial in trials:
            if trial["task_type"] == "pair_comparison":
                continue
            group = str(trial["hidden_repeat_group_ids"][0])
            occurrence = global_occurrence[group]
            trial["repeat_occurrence_index"] = occurrence
            trial["schedule_reason"] = (
                "broad_single_stimulus" if occurrence == 0 else "hidden_exact_repeat"
            )
            global_occurrence[group] += 1
    previously_seen: set[str] = set()
    for session_index in range(2):
        session_trials[session_index] = _constrained_order(
            session_trials[session_index],
            rng=rng,
            minimum_repeat_gap=minimum_repeat_gap,
            previously_seen_repeat_groups=previously_seen,
        )
        for trial_index, trial in enumerate(session_trials[session_index]):
            trial["trial_index"] = trial_index
            previously_seen.update(str(group) for group in trial["hidden_repeat_group_ids"])

    for item in stimuli.values():
        render = _two_cycle_payload(
            Path(pair_dataset_directory) / item["motion"],
            phase_reference_path=(Path(pair_dataset_directory) / item["phase_reference_motion"]),
            viewing_settings=settings,
        )
        item["render_hash"] = render["render_hash"]
        item["cycle_window"] = render["cycle_window"]
    for trials in session_trials:
        for trial in trials:
            trial["render_hashes"] = [
                _canonical_hash(
                    "presented-render",
                    {
                        "base_render_hash": stimuli[stimulus_id]["render_hash"],
                        "mirror_x": trial["view_mirror_x"],
                    },
                )
                for stimulus_id in trial["stimulus_ids"]
            ]

    manifest = {
        "format_version": LEGACY_PILOT_MANIFEST_VERSION,
        "protocol_version": LEGACY_SUBJECTIVE_PROTOCOL_VERSION,
        "seed": seed,
        "created_utc": datetime.now(UTC).isoformat(),
        "pair_dataset_directory": str(Path(pair_dataset_directory).resolve()),
        "stimulus_directory": str(Path(pair_dataset_directory).resolve()),
        "audit_queue": str(Path(audit_queue_path).resolve()),
        "render_protocol_hash": render_protocol_hash(settings),
        "viewing_settings": settings,
        "scale_anchors": {
            "naturalness": list(NATURALNESS_SCALE),
            "style_adherence": list(STYLE_SCALE),
        },
        "unique_stimulus_count": unique_stimulus_count,
        "session_count": session_count,
        "minimum_intervening_trials_for_repeat": minimum_repeat_gap,
        "render_validation": {
            "all_stimuli_use_exactly_two_detected_gait_cycles": True,
            "all_stimuli_use_fixed_frame_rate_hz": settings["frame_rate_hz"],
            "validated_stimulus_count": len(stimuli),
        },
        "presented_side_counts": dict(
            sorted(
                Counter(
                    side
                    for trials in session_trials
                    for trial in trials
                    for side in trial["side_classes"]
                ).items()
            )
        ),
        "stimuli": sorted(stimuli.values(), key=lambda item: item["stimulus_id"]),
        "hidden_anchor_stimulus_ids": sorted(anchor_ids),
        "sessions": [
            {"session_index": index, "trial_count": len(trials), "trials": trials}
            for index, trials in enumerate(session_trials)
        ],
        "adaptive_policy": {
            "enabled": True,
            "maximum_followups_per_session": 4,
            "priorities": [
                "wide_latent_uncertainty",
                "within_rater_repeat_disagreement",
                "model_human_error",
                "close_pair_score",
                "optimization_output",
                "underrepresented_source_family",
            ],
            "comparison_restriction": "same source and same editing goal",
            "pair_generation": "targeted only; no all-pairs matrix",
            "realized_order_log": "served_playlist.jsonl",
        },
        "pilot_status": "ready_for_human_collection",
        "human_observation_count_created": 0,
        "critic_retraining_permitted": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "pilot_manifest.json"
    _write_json(manifest_path, manifest)
    summary = {
        "format_version": LEGACY_PILOT_MANIFEST_VERSION,
        "status": "ready_for_human_collection",
        "manifest": str(manifest_path),
        "unique_animation_count": unique_stimulus_count,
        "single_stimulus_trial_count": sum(
            trial["task_type"] != "pair_comparison" for trials in session_trials for trial in trials
        ),
        "comparison_trial_count": sum(
            trial["task_type"] == "pair_comparison" for trials in session_trials for trial in trials
        ),
        "hidden_anchor_count": len(anchor_ids),
        "hidden_repeat_trial_count": sum(max(0, count - 1) for count in occurrence_count.values()),
        "session_trial_counts": [len(trials) for trials in session_trials],
        "human_observations": 0,
        "critic_retraining_started": False,
    }
    _write_json(output / "pilot_preparation.json", summary)
    return summary


_MANNEQUIN_REQUIRED_ROLES = (
    "pelvis",
    "chest",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)


def _two_cycle_payload(
    path: Path,
    *,
    phase_reference_path: Path | None = None,
    viewing_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = VIEWING_SETTINGS if viewing_settings is None else viewing_settings
    motion = load_motion_npz(path)
    if not math.isclose(motion.fps, float(settings["frame_rate_hz"]), abs_tol=1.0e-6):
        raise ValueError(f"subjective rendering requires {settings['frame_rate_hz']} Hz motion")
    if settings["protocol"] == MANNEQUIN_VIEWING_SETTINGS["protocol"]:
        missing_roles = [
            role for role in _MANNEQUIN_REQUIRED_ROLES if role not in motion.skeleton.roles
        ]
        if missing_roles:
            raise ValueError(
                "mannequin rendering requires semantic roles: " + ", ".join(missing_roles)
            )
    positions, global_rotations = forward_kinematics_numpy(
        motion.skeleton,
        motion.local_quat_wxyz,
        motion.root_translation_m,
    )
    phase_reference = (
        motion if phase_reference_path is None else load_motion_npz(phase_reference_path)
    )
    if phase_reference.num_frames != motion.num_frames or not math.isclose(
        phase_reference.fps, motion.fps, abs_tol=1.0e-6
    ):
        raise ValueError("phase reference must have the same frames and rate as the stimulus")
    phase_method = "detected_phase_exact_two_cycles"
    estimated_cycles: float | None = None
    try:
        phase = np.asarray(
            estimate_gait_phase(detect_foot_contacts(phase_reference)).phase_rad,
            dtype=np.float64,
        )
        span = float(phase[-1] - phase[0])
        estimated_cycles = span / (2.0 * math.pi)
        candidates: list[tuple[int, int, int]] = []
        for index in range(motion.num_frames):
            target = phase[index] + 4.0 * math.pi
            if target > phase[-1]:
                break
            end = int(np.searchsorted(phase, target, side="left"))
            if end > index + 2:
                candidates.append((abs((motion.num_frames - 1 - end) - index), index, end + 1))
        if candidates:
            _, start, stop = min(candidates)
        else:
            raise ValueError("motion does not contain two detected gait cycles")
    except ValueError as exc:
        raise ValueError(f"motion is ineligible for the fixed two-cycle render: {path}") from exc
    selected = np.asarray(positions[start:stop], dtype=np.float32)
    selected_rotations = np.asarray(global_rotations[start:stop], dtype=np.float32)
    root = motion.skeleton.root_index
    selected[:, :, 0] -= selected[:, root : root + 1, 0]
    selected[:, :, 2] -= selected[:, root : root + 1, 2]
    cycle_window = {
        "source_start_frame": start,
        "source_stop_frame": stop,
        "target_cycles": 2.0,
        "estimated_full_window_cycles": estimated_cycles,
        "selection_method": phase_method,
    }
    render_hash = _canonical_hash(
        "rendered-stimulus",
        {
            "motion_content_hash": motion.content_hash,
            "phase_reference_content_hash": phase_reference.content_hash,
            "cycle_window": cycle_window,
            "viewing_settings": settings,
        },
    )
    return {
        "fps": float(motion.fps),
        "joint_names": list(motion.skeleton.joint_names),
        "parents": motion.skeleton.parents.tolist(),
        "rest_offsets_m": motion.skeleton.rest_offsets_m.round(5).tolist(),
        "roles": {role: int(index) for role, index in sorted(motion.skeleton.roles.items())},
        "positions": selected.round(5).tolist(),
        "global_quat_wxyz": selected_rotations.round(6).tolist(),
        "duration_s": len(selected) / float(motion.fps),
        "render_protocol_hash": render_protocol_hash(settings),
        "render_hash": render_hash,
        "cycle_window": cycle_window,
        "renderer_protocol": settings["protocol"],
    }


def _public_trial(trial: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    order = list(trial["presentation_order"])
    is_tutorial = trial.get("task_type") == "calibration"
    disclosure = trial.get("tutorial_disclosure") if is_tutorial else None
    severity_labels = (
        list(disclosure.get("ordered_severities", [])) if isinstance(disclosure, dict) else []
    )
    if is_tutorial:
        public_stimuli = [
            {
                "label": (
                    str(severity_labels[index]).replace("_", " ").title()
                    if index < len(severity_labels)
                    else f"Example {index + 1}"
                ),
                "stimulus_id": stimulus_id,
            }
            for index, stimulus_id in enumerate(order)
        ]
    else:
        public_stimuli = (
            [{"label": "motion", "stimulus_id": order[0]}]
            if len(order) == 1
            else [
                {"label": "A", "stimulus_id": order[0]},
                {"label": "B", "stimulus_id": order[1]},
            ]
        )
    result: dict[str, Any] = {
        "trial_id": trial.get("trial_id", trial.get("tutorial_id")),
        "trial_number": int(trial.get("trial_index", trial.get("tutorial_index", 0))) + 1,
        "task_type": trial["task_type"],
        "comparison_mode": trial.get("comparison_mode"),
        "stimuli": public_stimuli,
        "response_options": (
            []
            if is_tutorial
            else (
                list(range(1, 8))
                if trial["task_type"] != "pair_comparison"
                else sorted(PAIR_OUTCOMES)
            )
        ),
        "scale_anchors": manifest["scale_anchors"].get(trial["task_type"]),
        "style_prompt": trial.get("style_prompt")
        if trial["task_type"] == "style_adherence"
        else None,
        "viewing_settings": manifest["viewing_settings"],
        "view_mirror_x": bool(trial.get("view_mirror_x", False)),
        "is_tutorial": is_tutorial,
        "tutorial_disclosure": disclosure,
        "ask_post_rating_difficulty": bool(
            not is_tutorial and manifest["viewing_settings"].get("post_rating_difficulty", False)
        ),
    }
    return result


def _adaptive_candidate(
    manifest: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    rater_id: str,
    session_index: int,
) -> dict[str, Any] | None:
    policy = manifest["adaptive_policy"]
    if not policy["enabled"]:
        return None
    rater_observations = [row for row in observations if row.get("rater_id") == rater_id]
    adaptive_done = sum(
        str(row.get("schedule_reason", "")).startswith("adaptive:")
        and row.get("session_index") == session_index
        for row in rater_observations
    )
    if adaptive_done >= int(policy["maximum_followups_per_session"]):
        return None
    if policy.get("selection_mode") == "response_adaptive_toggle_pool":
        required_mode = str(policy.get("required_comparison_mode", "same_viewport_toggle"))
        used_pair_ids = {
            str(row.get("pair_id"))
            for row in rater_observations
            if row.get("task_type") == "pair_comparison"
        }
        family_counts = Counter(
            str(family) for row in rater_observations for family in row.get("families", [])
        )
        single_ratings: dict[str, list[float]] = defaultdict(list)
        repeat_ratings: dict[str, list[float]] = defaultdict(list)
        for row in rater_observations:
            if row.get("rating") is None:
                continue
            stimulus_id = str(row["stimulus_ids"][0])
            single_ratings[stimulus_id].append(float(row["rating"]))
            for group in row.get("hidden_repeat_group_ids", []):
                repeat_ratings[str(group)].append(float(row["rating"]))
        candidates: list[tuple[float, str, dict[str, Any], list[str]]] = []
        for template in manifest.get("adaptive_comparison_pool", []):
            if template.get("task_type") != "pair_comparison":
                continue
            if template.get("comparison_mode") != required_mode:
                continue
            if session_index not in template.get("eligible_session_indices", []):
                continue
            pair_id = str(template.get("pair_id"))
            if pair_id in used_pair_ids:
                continue
            reasons = ["underrepresented_family"]
            families = [str(value) for value in template.get("families", [])]
            score = 1.0 / (1.0 + sum(family_counts[value] for value in families))
            ids = [str(value) for value in template.get("stimulus_ids", [])]
            if len(ids) == 2 and single_ratings[ids[0]] and single_ratings[ids[1]]:
                difference = abs(
                    float(np.mean(single_ratings[ids[0]])) - float(np.mean(single_ratings[ids[1]]))
                )
                score += max(0.0, 2.0 - difference)
                if difference <= 1.0:
                    reasons.append("close_single_ratings")
            repeat_groups = [str(value) for value in template.get("hidden_repeat_group_ids", [])]
            if any(
                len(repeat_ratings[group]) >= 2 and np.ptp(repeat_ratings[group]) >= 2.0
                for group in repeat_groups
            ):
                score += 1.5
                reasons.append("repeat_disagreement")
            candidates.append((score, pair_id, template, reasons))
        if not candidates:
            return None
        _, _, template, reasons = max(
            candidates,
            key=lambda value: (value[0], value[1]),
        )
        trial = dict(template)
        trial.update(
            {
                "trial_index": len(manifest["sessions"][session_index]["trials"]) + adaptive_done,
                "session_index": session_index,
                "seed": manifest["seed"],
                "schedule_reason": "adaptive:response_adaptive_toggle_pool+"
                + "+".join(sorted(set(reasons))),
            }
        )
        return trial
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row.get("rating") is not None:
            for group in row.get("hidden_repeat_group_ids", []):
                by_group[str(group)].append(row)
    rated_ids = Counter(value for row in observations for value in row.get("stimulus_ids", []))
    stimuli = {item["stimulus_id"]: item for item in manifest["stimuli"]}
    if adaptive_done % 2 == 1:
        pair_templates = [
            trial
            for session in manifest["sessions"]
            for trial in session["trials"]
            if trial["task_type"] == "pair_comparison"
        ]
        pair_counts = Counter(
            str(row.get("pair_id"))
            for row in observations
            if row.get("task_type") == "pair_comparison"
        )
        pair_scores: list[tuple[float, dict[str, Any], list[str]]] = []
        single_rating: dict[str, list[int]] = defaultdict(list)
        for row in observations:
            if row.get("rating") is not None:
                single_rating[str(row["stimulus_ids"][0])].append(int(row["rating"]))
        for template in pair_templates:
            pair_id = str(template["pair_id"])
            score = 1.0 / (1.0 + pair_counts[pair_id])
            reasons = ["underrepresented_source_family"]
            first, second = [str(value) for value in template["stimulus_ids"]]
            if single_rating[first] and single_rating[second]:
                distance = abs(
                    float(np.mean(single_rating[first])) - float(np.mean(single_rating[second]))
                )
                if distance <= 1.0:
                    score += 1.5
                    reasons.append("close_pair_score")
            item_meta = [stimuli[first], stimuli[second]]
            if any(
                "confident_heldout_source_misranking" in item["selection_reasons"]
                for item in item_meta
            ):
                score += 1.5
                reasons.append("model_human_error")
            if any(item["optimization_output"] for item in item_meta):
                score += 1.25
                reasons.append("optimization_output")
            pair_scores.append((score, template, reasons))
        if pair_scores:
            _, template, reasons = max(
                pair_scores, key=lambda value: (value[0], value[1]["trial_id"])
            )
            mode = (
                "sequential_neutral_gap"
                if template["comparison_mode"] == "same_viewport_toggle"
                else "same_viewport_toggle"
            )
            presentation = list(reversed(template["presentation_order"]))
            trial = dict(template)
            trial.update(
                {
                    "trial_id": _canonical_hash(
                        "trial",
                        {
                            "adaptive_pair": template["pair_id"],
                            "rater": rater_id,
                            "session": session_index,
                            "occurrence": pair_counts[str(template["pair_id"])],
                            "seed": manifest["seed"],
                        },
                    ),
                    "comparison_mode": mode,
                    "presentation_order": presentation,
                    "hidden_repeat_group_ids": [
                        _canonical_hash("pair-repeat", {"pair": template["pair_id"], "mode": mode})
                    ],
                    "schedule_reason": "adaptive:" + "+".join(sorted(set(reasons))),
                    "session_index": session_index,
                    "trial_index": len(manifest["sessions"][session_index]["trials"])
                    + adaptive_done,
                }
            )
            return trial
    scores: list[tuple[float, str, list[str]]] = []
    for stimulus_id, item in stimuli.items():
        single_reasons: list[str] = []
        score = 1.0 / (1.0 + rated_ids[stimulus_id])
        group_rows = by_group[item["hidden_repeat_group_id"]]
        if len(group_rows) >= 2:
            values = [float(row["rating"]) for row in group_rows]
            disagreement = max(values) - min(values)
            score += disagreement
            if disagreement >= 2:
                single_reasons.append("within_rater_repeat_disagreement")
        if "confident_heldout_source_misranking" in item["selection_reasons"]:
            score += 1.5
            single_reasons.append("model_human_error")
        if item["optimization_output"]:
            score += 1.25
            single_reasons.append("optimization_output")
        if rated_ids[stimulus_id] <= 1:
            single_reasons.append("wide_latent_uncertainty")
        single_reasons.append("underrepresented_source_family")
        scores.append((score, stimulus_id, single_reasons))
    if not scores:
        return None
    _, stimulus_id, reasons = max(scores, key=lambda value: (value[0], value[1]))
    item = stimuli[stimulus_id]
    occurrence = rated_ids[stimulus_id]
    trial = {
        "trial_id": _canonical_hash(
            "trial",
            {
                "adaptive": stimulus_id,
                "rater": rater_id,
                "session": session_index,
                "occurrence": occurrence,
                "seed": manifest["seed"],
            },
        ),
        "task_type": "naturalness",
        "stimulus_ids": [stimulus_id],
        "source_ids": [item["source_id"]],
        "families": [item["family"]],
        "styles": [item["style"]],
        "side_classes": [
            "right"
            if item["side_class"] == "left" and item["display_mirror_x"]
            else item["side_class"]
        ],
        "expected_quality_strata": [item["expected_quality_stratum"]],
        "hidden_repeat_group_ids": [item["hidden_repeat_group_id"]],
        "hidden_anchor": stimulus_id in set(manifest["hidden_anchor_stimulus_ids"]),
        "comparison_mode": None,
        "presentation_order": [stimulus_id],
        "view_mirror_x": item["display_mirror_x"],
        "render_hashes": [
            _canonical_hash(
                "presented-render",
                {
                    "base_render_hash": item["render_hash"],
                    "mirror_x": item["display_mirror_x"],
                },
            )
        ],
        "pair_id": None,
        "schedule_reason": "adaptive:" + "+".join(sorted(set(reasons))),
        "repeat_occurrence_index": len(by_group[item["hidden_repeat_group_id"]]),
        "session_index": session_index,
        "trial_index": len(manifest["sessions"][session_index]["trials"]) + adaptive_done,
        "seed": manifest["seed"],
    }
    return trial


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Motion quality study</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0d1016;color:#eef2f8}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#1b2638,#0d1016 48%)}main{width:min(1040px,calc(100% - 28px));margin:24px auto 56px}.panel{background:#151a24;border:1px solid #2b3444;border-radius:14px;padding:18px;box-shadow:0 18px 60px #0007}h1{font-size:24px;margin:0 0 6px}h2{font-size:19px}.muted{color:#9ca9bb}.hidden{display:none!important}input,select,button{font:inherit}input,select{background:#0e131c;color:#eef;border:1px solid #48556a;border-radius:7px;padding:9px}button{border:1px solid #53637b;border-radius:9px;background:#273247;color:#fff;padding:10px 14px;cursor:pointer}button:hover:not(:disabled){background:#354561}button:disabled{cursor:not-allowed;opacity:.38}.start{background:#4266a8}.topline{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}#viewportWrap{position:relative;background:#12151b;border:1px solid #313a48;border-radius:12px;overflow:hidden;aspect-ratio:3/2;touch-action:none}canvas{width:100%;height:100%;display:block}.clipBadge{position:absolute;left:16px;top:14px;padding:5px 9px;background:#05070acc;border:1px solid #526078;border-radius:7px;font-weight:700}.neutral{position:absolute;inset:0;background:#141820;display:grid;place-items:center;color:#aeb9ca;font-size:20px}.transport,.inspectionRow,.tutorialClips,.difficultyChoices{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:12px 0}.inspection{padding:10px 12px;border:1px solid #313a48;border-radius:10px;background:#10151e}.inspectionLabel{min-width:78px;color:#aeb9ca;font-size:13px}.active{outline:2px solid #72a5ff;background:#38598c!important}.tutorialDisclosure{padding:12px;border-left:4px solid #72a5ff;background:#101826;border-radius:7px;margin:10px 0}.question{font-size:20px;font-weight:650;margin:18px 0 11px}.scale{display:grid;grid-template-columns:repeat(7,1fr);gap:7px}.scale button{min-height:92px;padding:7px;font-size:13px}.scale strong{display:block;font-size:21px;margin-bottom:4px}.pairChoices{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.pairChoices button{min-height:62px}.selected{outline:3px solid #72a5ff;background:#38598c!important}.metadata{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.reasons{display:flex;flex-wrap:wrap;gap:9px 14px}.reasons label{white-space:nowrap}.submit{width:100%;margin-top:16px;background:#316f57;font-weight:700}.notice{min-height:24px;color:#f3cb75}.keyboard{color:#aeb9ca;font-size:13px}@media(max-width:760px){.scale{grid-template-columns:1fr}.scale button{min-height:48px}.pairChoices,.metadata{grid-template-columns:1fr}}
</style></head><body><main>
<section id="login" class="panel"><h1>Motion quality study</h1><p class="muted">Your session contains full-size, blinded motion judgments. Take breaks between sessions.</p><p><label>Participant ID <input id="rater" autocomplete="off" placeholder="assigned ID"></label></p><p><label>Session <select id="sessionIndex"><option value="0">1</option><option value="1">2</option></select></label></p><button class="start" onclick="startSession()">Begin session</button><p id="loginError" class="notice"></p></section>
<section id="study" class="panel hidden"><div class="topline"><div><h1 id="taskTitle">Motion quality</h1><span id="progress" class="muted"></span></div><span id="completion" class="muted"></span></div>
<div id="tutorialDisclosure" class="tutorialDisclosure hidden"></div>
<div id="viewportWrap"><canvas id="view" width="960" height="640"></canvas><div id="clipBadge" class="clipBadge hidden"></div><div id="neutral" class="neutral hidden">Neutral interval</div></div>
<div class="transport"><button id="play" onclick="playCurrent()">Play from start</button><button id="toggleA" class="hidden" onclick="switchClip(0)">A</button><button id="toggleB" class="hidden" onclick="switchClip(1)">B</button><span id="tutorialClips" class="tutorialClips hidden"></span><button id="markPoint" onclick="markPoint()" disabled>Mark interval point</button><span id="playbackNotice" class="notice">Watch the complete motion before rating.</span></div><div id="keyboard" class="keyboard hidden">A/B keys switch versions at the same frame and phase. Space alternates A and B.</div>
<div id="inspection" class="inspection hidden"><div class="inspectionRow"><span class="inspectionLabel">Speed</span><button data-speed="0.25" onclick="changeSpeed(.25)" disabled>0.25x</button><button data-speed="0.5" onclick="changeSpeed(.5)" disabled>0.5x</button><button data-speed="1" onclick="changeSpeed(1)" disabled>1x</button><button data-speed="1.5" onclick="changeSpeed(1.5)" disabled>1.5x</button><button data-speed="2" onclick="changeSpeed(2)" disabled>2x</button></div><div class="inspectionRow"><span class="inspectionLabel">Camera</span><button data-view="front" onclick="setView('front')" disabled>Front</button><button data-view="three_quarter" onclick="setView('three_quarter')" disabled>3/4</button><button data-view="side" onclick="setView('side')" disabled>Side</button><button id="zoomOut" onclick="changeZoom(-1)" disabled>Zoom out</button><button id="zoomIn" onclick="changeZoom(1)" disabled>Zoom in</button><button id="resetCamera" onclick="resetCamera()" disabled>Reset camera</button><label><input id="skeletonOverlay" type="checkbox" onchange="changeOverlay()" disabled> faint skeleton</label><span class="muted">Drag the viewport to orbit.</span></div></div>
<div id="singleResponse" class="hidden"><div id="question" class="question"></div><div id="scale" class="scale"></div></div>
<div id="pairResponse" class="hidden"><div class="question">Which motion looks better overall?</div><div class="pairChoices"><button data-value="a_better" disabled>A better</button><button data-value="b_better" disabled>B better</button><button data-value="effectively_equal" disabled>Effectively equal</button><button data-value="not_sure" disabled>Not sure</button></div></div>
<div id="metadata" class="metadata"><label>Confidence (optional)<select id="confidence"><option value="">Choose…</option><option value="1">1 — very unsure</option><option value="2">2</option><option value="3">3 — moderate</option><option value="4">4</option><option value="5">5 — very sure</option></select></label><div><span class="muted">Optional reasons</span><div id="reasons" class="reasons"></div></div></div>
<button id="continueRating" class="submit" onclick="lockRating()" disabled>Continue</button><div id="difficulty" class="hidden"><div class="question">How difficult was this judgment?</div><p class="muted">Optional — choose one or submit without answering.</p><div class="difficultyChoices"><button data-difficulty="easy" onclick="chooseDifficulty('easy',this)">Easy</button><button data-difficulty="moderate" onclick="chooseDifficulty('moderate',this)">Moderate</button><button data-difficulty="difficult" onclick="chooseDifficulty('difficult',this)">Difficult</button></div></div>
<button id="submit" class="submit hidden" onclick="submitResponse()" disabled>Submit judgment</button><button id="continueTutorial" class="submit hidden" onclick="completeTutorial()" disabled>Continue</button><p id="error" class="notice"></p></section>
</main><script>
const reasonNames=[['naturalness','Naturalness'],['coordination','Coordination'],['style','Style'],['weight_transfer','Weight transfer'],['rigidity_smoothing','Rigidity / smoothing'],['anatomical_plausibility','Anatomy'],['timing','Timing'],['other','Other']];
const telemetryVersion='motionlab.inspection_telemetry.v1';
let token=null,trial=null,motions=[],current=0,frame=0,phaseTime=0,playing=false,lastTs=0,completed=[],replays=[],selected=null,marked=[],submitting=false,ratingLocked=false,judgmentDifficulty=null,playbackRate=1,camera=null,inspectionStartMs=null,lockedInspectionTimeMs=null,playbackRequiresAck=false,speedEvents=[],cameraEvents=[],timeBySpeed={},zoomChanges=0,viewChanges=0,overlayChanges=0,orbitStart=null;
const $=id=>document.getElementById(id);reasonNames.forEach(([v,n])=>$('reasons').insertAdjacentHTML('beforeend',`<label><input type="checkbox" value="${v}"> ${n}</label>`));
async function api(path,body){let controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);try{const result=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined,signal:controller.signal});if(!result.ok)throw new Error(await result.text());return await result.json()}catch(e){if(e&&e.name==='AbortError')throw new Error('The evaluator server did not respond. Check that it is still running, then retry.');throw e}finally{clearTimeout(timer)}}
async function startSession(){try{let r=$('rater').value.trim();if(!r)throw new Error('Enter the assigned participant ID.');let data=await api('/api/session',{rater_id:r,session_index:Number($('sessionIndex').value)});token=data.session_token;$('login').classList.add('hidden');$('study').classList.remove('hidden');await loadNext()}catch(e){$('loginError').textContent=e.message}}
function supportsInspection(){return trial&&trial.viewing_settings.protocol==='motionlab.fixed_mannequin_canvas.v1'}
function cameraSnapshot(){return{preset:camera.preset,yaw_degrees:camera.yaw_degrees,pitch_degrees:camera.pitch_degrees,zoom:camera.zoom,skeleton_overlay:camera.skeleton_overlay}}
function defaultCamera(){let raw=trial.viewing_settings.default_camera||{preset:'legacy',yaw_degrees:25,pitch_degrees:0,zoom:1,skeleton_overlay:false};return{preset:String(raw.preset),yaw_degrees:Number(raw.yaw_degrees),pitch_degrees:Number(raw.pitch_degrees),zoom:Number(raw.zoom),skeleton_overlay:Boolean(raw.skeleton_overlay)}}
function resetTelemetry(){playbackRate=1;camera=defaultCamera();inspectionStartMs=null;lockedInspectionTimeMs=null;speedEvents=[];cameraEvents=[];zoomChanges=0;viewChanges=0;overlayChanges=0;timeBySpeed={};for(const rate of(trial.viewing_settings.playback_rates||[1]))timeBySpeed[String(Number(rate))]=0;$('skeletonOverlay').checked=false;updateControlHighlights()}
function elapsedInspection(){return inspectionStartMs===null?0:Math.max(0,(lockedInspectionTimeMs===null?performance.now():lockedInspectionTimeMs)-inspectionStartMs)}
function recordCamera(kind,from,to){cameraEvents.push({elapsed_ms:elapsedInspection(),kind:kind,from:from,to:to})}
function setInspectionEnabled(enabled){document.querySelectorAll('#inspection button,#inspection input').forEach(x=>x.disabled=!enabled);$('markPoint').disabled=!enabled||Boolean(trial&&trial.is_tutorial);document.querySelectorAll('#tutorialClips button').forEach(x=>x.disabled=!enabled);if(trial&&trial.comparison_mode==='same_viewport_toggle'){$('toggleA').disabled=!enabled;$('toggleB').disabled=!enabled}}
function updateControlHighlights(){document.querySelectorAll('[data-speed]').forEach(x=>x.classList.toggle('active',Number(x.dataset.speed)===playbackRate));document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x.dataset.view===camera.preset))}
async function loadNext(){window.__motionlabRendererReady=false;playing=false;selected=null;marked=[];ratingLocked=false;judgmentDifficulty=null;$('confidence').value='';document.querySelectorAll('#reasons input').forEach(x=>x.checked=false);document.querySelectorAll('[data-value],[data-difficulty]').forEach(x=>{x.classList.remove('selected');if(x.dataset.value)x.disabled=true});$('continueRating').disabled=true;$('continueRating').classList.add('hidden');$('submit').disabled=true;$('submit').classList.add('hidden');$('continueTutorial').disabled=true;$('continueTutorial').classList.add('hidden');$('difficulty').classList.add('hidden');$('error').textContent='';let data=await api('/api/next?session_token='+encodeURIComponent(token));if(data.complete){$('study').innerHTML='<h1>Session complete</h1><p>Thank you. Your raw judgments were saved.</p>';return}trial=data.trial;motions=[];for(const item of trial.stimuli)motions.push(await api('/api/stimulus/'+encodeURIComponent(item.stimulus_id)+'?session_token='+encodeURIComponent(token)));current=0;frame=0;phaseTime=0;completed=motions.map(()=>Boolean(trial.is_tutorial));replays=motions.map(()=>0);playing=false;playbackRequiresAck=false;resetTelemetry();$('inspection').classList.toggle('hidden',!supportsInspection());setInspectionEnabled(false);$('progress').textContent=trial.is_tutorial?'Calibration '+trial.trial_number:'Trial '+trial.trial_number;$('completion').textContent=trial.is_tutorial?'Explore '+motions.length+' severity levels':'0 / '+motions.length+' full viewings';$('taskTitle').textContent=trial.is_tutorial?'Calibration / learning':trial.task_type==='pair_comparison'?'Motion comparison':trial.task_type==='style_adherence'?'Style adherence':'Naturalness';$('singleResponse').classList.toggle('hidden',trial.is_tutorial||trial.task_type==='pair_comparison');$('pairResponse').classList.toggle('hidden',trial.is_tutorial||trial.task_type!=='pair_comparison');$('metadata').classList.toggle('hidden',trial.is_tutorial);$('toggleA').classList.toggle('hidden',trial.comparison_mode!=='same_viewport_toggle');$('toggleB').classList.toggle('hidden',trial.comparison_mode!=='same_viewport_toggle');$('keyboard').classList.toggle('hidden',trial.comparison_mode!=='same_viewport_toggle');$('tutorialClips').classList.toggle('hidden',!trial.is_tutorial);$('tutorialClips').innerHTML='';if(trial.is_tutorial){trial.stimuli.forEach((item,index)=>{let b=document.createElement('button');b.textContent=item.label;b.disabled=true;b.classList.toggle('active',index===0);b.onclick=()=>switchClip(index);$('tutorialClips').appendChild(b)})}$('clipBadge').classList.toggle('hidden',trial.stimuli.length===1);$('clipBadge').textContent=trial.stimuli[current].label;let disclosure=trial.tutorial_disclosure;$('tutorialDisclosure').classList.toggle('hidden',!trial.is_tutorial);$('tutorialDisclosure').innerHTML=trial.is_tutorial?'<strong>'+String(disclosure.calibration_family).replaceAll('_',' ')+'</strong><br>'+String(disclosure.instruction)+'<br><span class="muted">Order: '+disclosure.ordered_severities.join(' → ')+'</span>':'';$('question').textContent=trial.task_type==='style_adherence'?'How well does this motion adhere to the requested style?':'Ignoring whether you personally like the style, how natural and internally coherent is this walk?';buildScale();$('continueRating').classList.toggle('hidden',trial.is_tutorial);$('playbackNotice').textContent=trial.is_tutorial?'Choose any severity, speed, or view. Continue when ready.':trial.comparison_mode==='sequential_neutral_gap'?'A and B will play in sequence with a neutral interval.':'Watch the complete motion before rating.';draw();window.__motionlabRendererReady=true;if(trial.is_tutorial){inspectionStartMs=performance.now();enableInspectionAndResponse();$('play').disabled=false}else playCurrent()}
function buildScale(){let box=$('scale');box.innerHTML='';if(!trial.scale_anchors)return;trial.scale_anchors.forEach((text,i)=>{let b=document.createElement('button');b.dataset.value=String(i+1);b.disabled=true;b.innerHTML='<strong>'+(i+1)+'</strong>'+text;b.onclick=()=>choose(i+1,b);box.appendChild(b)})}
function choose(value,button){if(!allComplete()||ratingLocked)return;selected=value;document.querySelectorAll('[data-value]').forEach(x=>x.classList.remove('selected'));button.classList.add('selected');$('continueRating').disabled=false}
document.querySelectorAll('.pairChoices button').forEach(button=>button.onclick=()=>choose(button.dataset.value,button));
function allComplete(){return completed.every(Boolean)}
async function playCurrent(){if(!trial||playing||ratingLocked)return;playbackRequiresAck=!trial.is_tutorial&&!completed[current];if(!playbackRequiresAck)replays[current]++;frame=0;phaseTime=0;playing=true;$('play').disabled=true;$('error').textContent='';$('playbackNotice').textContent=trial.is_tutorial?'Playing '+trial.stimuli[current].label+' — choose another rung at any time.':playbackRequiresAck?'Playing required full viewing…':'Replaying motion…';try{if(playbackRequiresAck)await api('/api/playback-start',{session_token:token,trial_id:trial.trial_id,stimulus_id:trial.stimuli[current].stimulus_id,playback_rate:playbackRate,camera_state:cameraSnapshot()});lastTs=performance.now();requestAnimationFrame(tick)}catch(e){playing=false;$('play').disabled=false;$('playbackNotice').textContent='Playback could not start. Please retry.';$('error').textContent=e.message}}
function switchClip(index){if(!trial||ratingLocked||!allComplete()||index<0||index>=motions.length)return;if(!trial.is_tutorial&&trial.comparison_mode!=='same_viewport_toggle')return;current=index;$('clipBadge').textContent=trial.stimuli[current].label;if(trial.is_tutorial){document.querySelectorAll('#tutorialClips button').forEach((button,buttonIndex)=>button.classList.toggle('active',buttonIndex===index));$('playbackNotice').textContent=playing?'Playing '+trial.stimuli[current].label+' — choose another rung at any time.':'Selected '+trial.stimuli[current].label+'. Press Play from start to inspect it.'}frame=Math.min(motions[current].positions.length-1,Math.floor(phaseTime*motions[current].fps));draw()}
async function finishMandatory(finished){$('playbackNotice').textContent='Confirming full viewing…';await api('/api/playback-complete',{session_token:token,trial_id:trial.trial_id,stimulus_id:trial.stimuli[finished].stimulus_id,playback_rate:playbackRate,camera_state:cameraSnapshot()});completed[finished]=true;$('completion').textContent=completed.filter(Boolean).length+' / '+motions.length+' full viewings';if(!allComplete()){let next=completed.findIndex(value=>!value);if(trial.comparison_mode==='sequential_neutral_gap'&&finished===0){showNeutralThen(next);return}current=next;frame=0;phaseTime=0;$('clipBadge').textContent=trial.stimuli[current].label;setTimeout(playCurrent,350);return}$('play').disabled=false;inspectionStartMs=performance.now();$('playbackNotice').textContent='Full viewing complete. You may rate or replay.';enableInspectionAndResponse()}
async function tick(ts){if(!playing)return;let dt=Math.max(0,Math.min(.05,(ts-lastTs)/1000));lastTs=ts;if(inspectionStartMs!==null)timeBySpeed[String(playbackRate)]=(timeBySpeed[String(playbackRate)]||0)+dt*1000;phaseTime+=dt*playbackRate;frame=Math.min(motions[current].positions.length-1,Math.floor(phaseTime*motions[current].fps));draw();if(phaseTime>=motions[current].duration_s){playing=false;let finished=current;try{if(playbackRequiresAck)await finishMandatory(finished);else{$('play').disabled=false;$('playbackNotice').textContent=trial.is_tutorial?'Calibration controls are ready. Choose any level or continue.':'Replay complete. You may rate or replay.'}}catch(e){$('play').disabled=false;$('playbackNotice').textContent='Playback was not accepted. Please replay the motion.';$('error').textContent=e.message}return}requestAnimationFrame(tick)}
function showNeutralThen(index){$('neutral').classList.remove('hidden');setTimeout(()=>{$('neutral').classList.add('hidden');current=index;frame=0;phaseTime=0;$('clipBadge').textContent=trial.stimuli[current].label;playCurrent()},700)}
function enableInspectionAndResponse(){setInspectionEnabled(supportsInspection());$('markPoint').disabled=Boolean(trial.is_tutorial);if(trial.comparison_mode==='same_viewport_toggle'){$('toggleA').disabled=false;$('toggleB').disabled=false}if(trial.is_tutorial){$('continueTutorial').classList.remove('hidden');$('continueTutorial').disabled=false}else{document.querySelectorAll('[data-value]').forEach(x=>x.disabled=false)}updateControlHighlights()}
function markPoint(){if(!trial||!allComplete())return;let value=frame/motions[current].fps;marked.push(value);if(marked.length>2)marked=marked.slice(-2);$('playbackNotice').textContent=marked.length===1?'Interval start marked. Mark again for the end.':'Interval marked.'}
function changeSpeed(rate){if(!allComplete()||!(trial.viewing_settings.playback_rates||[]).includes(rate)||rate===playbackRate)return;let previous=playbackRate;playbackRate=rate;speedEvents.push({elapsed_ms:elapsedInspection(),from_rate:previous,to_rate:rate});updateControlHighlights()}
function setView(name){if(!allComplete())return;let preset=trial.viewing_settings.camera_presets[name];if(!preset)return;let previous=cameraSnapshot();camera.preset=name;camera.yaw_degrees=Number(preset.yaw_degrees);camera.pitch_degrees=Number(preset.pitch_degrees);viewChanges++;recordCamera('view',previous,cameraSnapshot());updateControlHighlights();draw()}
function changeZoom(direction){if(!allComplete())return;let limits=trial.viewing_settings.zoom_limits,previous=cameraSnapshot(),next=Math.max(Number(limits.minimum),Math.min(Number(limits.maximum),camera.zoom+direction*Number(limits.step)));if(next===camera.zoom)return;camera.zoom=Number(next.toFixed(4));zoomChanges++;recordCamera('zoom',previous,cameraSnapshot());draw()}
function resetCamera(){if(!allComplete())return;let previous=cameraSnapshot();camera=defaultCamera();$('skeletonOverlay').checked=camera.skeleton_overlay;recordCamera('reset',previous,cameraSnapshot());updateControlHighlights();draw()}
function changeOverlay(){if(!allComplete()){$('skeletonOverlay').checked=camera.skeleton_overlay;return}let previous=cameraSnapshot();camera.skeleton_overlay=$('skeletonOverlay').checked;overlayChanges++;recordCamera('overlay',previous,cameraSnapshot());draw()}
function lockRating(){if(selected===null||!allComplete()||ratingLocked)return;ratingLocked=true;playing=false;orbitStart=null;lockedInspectionTimeMs=performance.now();setInspectionEnabled(false);$('play').disabled=true;document.querySelectorAll('[data-value]').forEach(x=>x.disabled=true);$('continueRating').disabled=true;$('continueRating').classList.add('hidden');if(trial.ask_post_rating_difficulty)$('difficulty').classList.remove('hidden');$('submit').classList.remove('hidden');$('submit').disabled=false}
function chooseDifficulty(value,button){judgmentDifficulty=value;document.querySelectorAll('[data-difficulty]').forEach(x=>x.classList.remove('selected'));button.classList.add('selected')}
function telemetry(){return{format_version:telemetryVersion,replay_count:[...replays],speed_change_events:speedEvents,playback_wall_time_by_speed_ms:timeBySpeed,camera_change_events:cameraEvents,zoom_change_count:zoomChanges,view_change_count:viewChanges,overlay_change_count:overlayChanges,total_inspection_time_ms:elapsedInspection(),final_playback_rate:playbackRate,final_camera_state:cameraSnapshot()}}
async function completeTutorial(){if(!trial||!trial.is_tutorial||submitting)return;submitting=true;playing=false;$('continueTutorial').disabled=true;try{await api('/api/tutorial-complete',{session_token:token,trial_id:trial.trial_id});submitting=false;await loadNext()}catch(e){submitting=false;$('continueTutorial').disabled=false;$('error').textContent=e.message}}
async function submitResponse(){if(!allComplete()||selected===null||!ratingLocked||submitting)return;submitting=true;playing=false;$('submit').disabled=true;let confidence=$('confidence').value;let body={session_token:token,trial_id:trial.trial_id,response:selected,confidence:confidence===''?null:Number(confidence),reason_tags:[...document.querySelectorAll('#reasons input:checked')].map(x=>x.value),replay_count:replays,marked_interval_s:marked.length?marked:null,judgment_difficulty:judgmentDifficulty,inspection_telemetry:telemetry()};try{await api('/api/observation',body)}catch(e){submitting=false;$('submit').disabled=false;$('error').textContent=e.message;return}submitting=false;try{await loadNext()}catch(e){$('error').textContent='Judgment saved, but the next trial could not be loaded: '+e.message}}
function add(a,b){return[a[0]+b[0],a[1]+b[1],a[2]+b[2]]}function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}function mul(a,s){return[a[0]*s,a[1]*s,a[2]*s]}function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function norm(a){let n=Math.hypot(...a);return n<1e-8?[0,1,0]:mul(a,1/n)}
function qrot(q,v){let w=q[0],u=[q[1],q[2],q[3]],t=mul(cross(u,v),2);return add(v,add(mul(t,w),cross(u,t)))}
function projection(c){let yaw=camera.yaw_degrees*Math.PI/180,pitch=camera.pitch_degrees*Math.PI/180,right=[Math.cos(yaw),0,-Math.sin(yaw)],up=[-Math.sin(yaw)*Math.sin(pitch),Math.cos(pitch),-Math.cos(yaw)*Math.sin(pitch)],depth=[Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)],scale=Number(trial.viewing_settings.pixels_per_meter)*camera.zoom;return v=>{let p=[trial.view_mirror_x?-v[0]:v[0],v[1]-.82,v[2]];return[c.width/2+dot(p,right)*scale,c.height*.56-dot(p,up)*scale,dot(p,depth)]}}
function drawProjectedGrid(x,c,project){x.strokeStyle='#273140';x.lineWidth=1;for(let i=-8;i<=8;i++){let a=project([i*.25,0,-2]),b=project([i*.25,0,2]);x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke()}for(let i=-8;i<=8;i++){let a=project([-2,0,i*.25]),b=project([2,0,i*.25]);x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke()}}
function shade(hex,factor){let n=parseInt(hex.slice(1),16),r=Math.min(255,Math.round(((n>>16)&255)*factor)),g=Math.min(255,Math.round(((n>>8)&255)*factor)),b=Math.min(255,Math.round((n&255)*factor));return`rgb(${r},${g},${b})`}
function boxVertices(center,axes,half){let result=[];for(const z of[-1,1])for(const y of[-1,1])for(const x of[-1,1])result.push(add(center,add(mul(axes[0],x*half[0]),add(mul(axes[1],y*half[1]),mul(axes[2],z*half[2])))));return result}
const boxFaces=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]];
function pushBox(faces,vertices,color,accentFace=null){let light=norm([-.35,.8,-.45]);boxFaces.forEach((indices,index)=>{let points=indices.map(i=>vertices[i]),normal=norm(cross(sub(points[1],points[0]),sub(points[2],points[0]))),brightness=.58+.35*Math.abs(dot(normal,light));faces.push({points:points,color:index===accentFace?'#426f9f':shade(color,brightness)})})}
function orientedBox(faces,p,q,index,offset,half,color,accentFace=null){let axes=[qrot(q[index],[1,0,0]),qrot(q[index],[0,1,0]),qrot(q[index],[0,0,1])],center=add(p[index],qrot(q[index],offset));pushBox(faces,boxVertices(center,axes,half),color,accentFace)}
function boneBox(faces,p,q,start,end,width,depth,color,accentFace=null){if(start===undefined||end===undefined)return;let a=p[start],b=p[end],axis=norm(sub(b,a)),reference=qrot(q[start],[0,0,1]),side=norm(cross(axis,reference));if(Math.hypot(...cross(axis,reference))<1e-5)side=norm(cross(axis,qrot(q[start],[0,1,0])));let front=norm(cross(side,axis));pushBox(faces,boxVertices(mul(add(a,b),.5),[side,axis,front],[width*.5,Math.hypot(...sub(b,a))*.5,depth*.5]),color,accentFace)}
function drawSkeletonOverlay(x,motion,p,project,legacy=false){x.save();x.strokeStyle=legacy?'#e5ebf5':'#dce9f4aa';x.fillStyle=legacy?'#89b4ff':'#9fc4e5aa';x.lineWidth=legacy?4:1.5;x.lineCap='round';for(let j=0;j<p.length;j++){let parent=motion.parents[j];if(parent<0)continue;let u=project(p[j]),v=project(p[parent]);x.beginPath();x.moveTo(v[0],v[1]);x.lineTo(u[0],u[1]);x.stroke()}for(let j=0;j<p.length;j++){let u=project(p[j]);x.beginPath();x.arc(u[0],u[1],legacy?3.5:2,0,Math.PI*2);x.fill()}x.restore()}
function drawLegacy(x,c,motion,p){drawLegacyGrid(x,c);let yaw=25*Math.PI/180,cs=Math.cos(yaw),sn=Math.sin(yaw),s=285,map=v=>{let mx=trial.view_mirror_x?-v[0]:v[0],rx=mx*cs-v[2]*sn,rz=mx*sn+v[2]*cs;void rx;return[c.width/2+rz*s,c.height*.9-v[1]*s]};drawSkeletonOverlay(x,motion,p,map,true)}
function drawLegacyGrid(x,c){x.strokeStyle='#273140';x.lineWidth=1;for(let i=0;i<8;i++){let depth=i/8,y=c.height*.66+c.height*.24*depth*depth;x.beginPath();x.moveTo(0,y);x.lineTo(c.width,y);x.stroke()}for(let i=-12;i<=12;i++){let px=c.width/2+i*71.25;x.beginPath();x.moveTo(px,c.height*.9);x.lineTo(c.width/2+(px-c.width/2)*.25,c.height*.66);x.stroke()}x.beginPath();x.moveTo(0,c.height*.9);x.lineTo(c.width,c.height*.9);x.stroke()}
function drawMannequin(x,c,motion,p,q){let project=projection(c);drawProjectedGrid(x,c,project);let r=motion.roles,faces=[],idx=name=>r[name];boneBox(faces,p,q,idx('left_shoulder'),idx('left_elbow'),.09,.06,'#7894b5');boneBox(faces,p,q,idx('left_elbow'),idx('left_wrist'),.07,.045,'#718dad');boneBox(faces,p,q,idx('right_shoulder'),idx('right_elbow'),.09,.06,'#7894b5');boneBox(faces,p,q,idx('right_elbow'),idx('right_wrist'),.07,.045,'#718dad');boneBox(faces,p,q,idx('left_hip'),idx('left_knee'),.12,.09,'#647f9e');boneBox(faces,p,q,idx('left_knee'),idx('left_ankle'),.085,.06,'#607a98');boneBox(faces,p,q,idx('right_hip'),idx('right_knee'),.12,.09,'#647f9e');boneBox(faces,p,q,idx('right_knee'),idx('right_ankle'),.085,.06,'#607a98');orientedBox(faces,p,q,idx('pelvis'),[0,.035,0],[.17,.105,.12],'#526d8e',1);orientedBox(faces,p,q,idx('chest'),[0,-.09,0],[.205,.17,.105],'#7898ba',1);orientedBox(faces,p,q,idx('head'),[0,.065,0],[.105,.13,.105],'#91a9c1');orientedBox(faces,p,q,idx('head'),[0,.065,.118],[.06,.055,.018],'#547ba3',1);orientedBox(faces,p,q,idx('left_wrist'),[.07,0,0],[.095,.035,.06],'#839db8',1);orientedBox(faces,p,q,idx('right_wrist'),[-.07,0,0],[.095,.035,.06],'#839db8',1);for(const side of['left','right']){let ankle=idx(side+'_ankle'),toe=idx(side+'_toe');if(toe!==undefined){boneBox(faces,p,q,ankle,toe,.115,.065,'#4f6d8d',1);orientedBox(faces,p,q,toe,[0,0,.018],[.07,.045,.045],'#426f9f',1)}else orientedBox(faces,p,q,ankle,[0,-.02,.1],[.09,.055,.16],'#4f6d8d',1)}let projected=[];for(const face of faces){let points=face.points.map(project),normal=norm(cross(sub(face.points[1],face.points[0]),sub(face.points[2],face.points[0])));projected.push({points:points,depth:points.reduce((sum,v)=>sum+v[2],0)/points.length,color:face.color,normal:normal})}projected.sort((a,b)=>b.depth-a.depth);x.lineJoin='round';for(const face of projected){x.beginPath();x.moveTo(face.points[0][0],face.points[0][1]);for(let i=1;i<face.points.length;i++)x.lineTo(face.points[i][0],face.points[i][1]);x.closePath();x.fillStyle=face.color;x.fill();x.strokeStyle='#243344';x.lineWidth=1;x.stroke()}if(camera.skeleton_overlay)drawSkeletonOverlay(x,motion,p,project,false)}
function draw(){let c=$('view'),x=c.getContext('2d');x.fillStyle='#12151b';x.fillRect(0,0,c.width,c.height);if(!motions.length)return;let motion=motions[current],index=Math.min(frame,motion.positions.length-1),p=motion.positions[index];if(motion.renderer_protocol==='motionlab.fixed_mannequin_canvas.v1')drawMannequin(x,c,motion,p,motion.global_quat_wxyz[index]);else drawLegacy(x,c,motion,p)}
window.__motionlabInspectionState=()=>({current:current,frame:frame,phaseTime:phaseTime,playbackRate:playbackRate,camera:cameraSnapshot(),telemetry:telemetry()});
const canvas=$('view');canvas.addEventListener('pointerdown',event=>{if(!supportsInspection()||ratingLocked||!allComplete()||!trial.viewing_settings.orbit.enabled)return;orbitStart={x:event.clientX,y:event.clientY,state:cameraSnapshot()};canvas.setPointerCapture(event.pointerId)});canvas.addEventListener('pointermove',event=>{if(!orbitStart||ratingLocked)return;let maximum=Number(trial.viewing_settings.orbit.maximum_pitch_degrees);camera.preset='orbit';camera.yaw_degrees=orbitStart.state.yaw_degrees+(event.clientX-orbitStart.x)*.35;camera.pitch_degrees=Math.max(-maximum,Math.min(maximum,orbitStart.state.pitch_degrees-(event.clientY-orbitStart.y)*.25));updateControlHighlights();draw()});function finishOrbit(){if(!orbitStart)return;let previous=orbitStart.state;orbitStart=null;recordCamera('orbit',previous,cameraSnapshot())}canvas.addEventListener('pointerup',finishOrbit);canvas.addEventListener('pointercancel',finishOrbit);
document.addEventListener('keydown',e=>{if(!trial||trial.comparison_mode!=='same_viewport_toggle')return;if(e.key.toLowerCase()==='a')switchClip(0);if(e.key.toLowerCase()==='b')switchClip(1);if(e.code==='Space'){e.preventDefault();switchClip(1-current)}});
</script></body></html>"""


def serve_subjective_evaluator(
    pair_dataset_directory: Path,
    pilot_manifest_path: Path,
    observations_path: Path,
    *,
    served_playlist_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve the localhost-only calibrated evaluator and append immutable raw events."""
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("the subjective evaluator is restricted to localhost")
    pilot_path = Path(pilot_manifest_path)
    pause_marker = pilot_path.with_name("PILOT_PAUSED.json")
    if pause_marker.exists():
        raise ValueError(
            f"human collection for this pilot is paused ({pause_marker}); analysis remains available"
        )
    manifest = json.loads(pilot_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") not in SUPPORTED_PILOT_MANIFEST_VERSIONS:
        raise ValueError("unsupported subjective pilot manifest")
    if manifest.get("protocol_version") not in SUPPORTED_SUBJECTIVE_PROTOCOL_VERSIONS:
        raise ValueError("subjective pilot uses an unsupported collection protocol")
    viewing_settings = _resolve_viewing_settings(manifest)
    uses_mannequin = viewing_settings["protocol"] == MANNEQUIN_VIEWING_SETTINGS["protocol"]
    pair_root = Path(pair_dataset_directory)
    manifest_root = Path(manifest["pair_dataset_directory"])
    if pair_root.resolve() != manifest_root.resolve():
        raise ValueError("pilot manifest belongs to a different pair dataset")
    stimulus_root = Path(manifest.get("stimulus_directory", manifest["pair_dataset_directory"]))
    stimuli = {str(item["stimulus_id"]): item for item in manifest["stimuli"]}
    sessions: dict[str, dict[str, Any]] = {}
    payload_cache: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    served_path = (
        Path(served_playlist_path)
        if served_playlist_path is not None
        else pilot_path.with_name("served_playlist.jsonl")
    )

    def observations() -> list[dict[str, Any]]:
        return _read_jsonl(observations_path)

    def current_trial(state: dict[str, Any]) -> dict[str, Any] | None:
        if state.get("current") is not None:
            value = state["current"]
            if not isinstance(value, dict):
                raise ValueError("invalid in-memory trial state")
            return value
        tutorial: dict[str, Any] | None = None
        if state["session_index"] == 0:
            tutorial = next(
                (
                    raw
                    for raw in manifest.get("calibration_tutorial", [])
                    if str(raw["tutorial_id"]) not in state["completed_tutorial_ids"]
                ),
                None,
            )
        if tutorial is not None:
            trial = {
                **tutorial,
                "trial_id": str(tutorial["tutorial_id"]),
                "trial_index": int(tutorial["tutorial_index"]),
                "comparison_mode": None,
                "view_mirror_x": False,
                "schedule_reason": "calibration_tutorial",
                "seed": manifest["seed"],
            }
        else:
            trial = None
        complete_ids = {
            str(row["trial_id"])
            for row in observations()
            if row.get("rater_id") == state["rater_id"]
            and row.get("session_index") == state["session_index"]
        }
        base = manifest["sessions"][state["session_index"]]["trials"]
        if trial is None:
            trial = next(
                (row for row in base if str(row["trial_id"]) not in complete_ids),
                None,
            )
        if trial is None and tutorial is None:
            trial = _adaptive_candidate(
                manifest,
                observations(),
                rater_id=state["rater_id"],
                session_index=state["session_index"],
            )
            if trial is not None and trial["trial_id"] in complete_ids:
                trial = None
        if trial is None:
            return None
        state["current"] = trial
        state["started_monotonic"] = time.monotonic()
        state["completed_stimuli"] = set()
        state["playback_started"] = {}
        state["served_stimuli"] = set(trial["presentation_order"])
        _append_jsonl(
            served_path,
            {
                "format_version": manifest["protocol_version"],
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "rater_id": state["rater_id"],
                "session_id": state["session_id"],
                "session_index": state["session_index"],
                "trial_id": trial["trial_id"],
                "trial_index": trial["trial_index"],
                "seed": trial["seed"],
                "task_type": trial["task_type"],
                "presentation_order": trial["presentation_order"],
                "schedule_reason": trial["schedule_reason"],
            },
        )
        return trial

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, "application/json", json.dumps(value, allow_nan=False).encode())

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65_536:
                raise ValueError("invalid request length")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("JSON object required")
            return value

        def _state(self, token: str) -> dict[str, Any]:
            state = sessions.get(token)
            if state is None:
                raise ValueError("unknown or expired session")
            return state

        def _ensure_collection_active(self) -> None:
            if pause_marker.exists():
                raise ValueError(
                    "human collection for this pilot is paused; "
                    "the running server will not serve or record trials"
                )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._ensure_collection_active()
                if parsed.path == "/":
                    self._send(HTTPStatus.OK, "text/html; charset=utf-8", _HTML.encode())
                    return
                query = parse_qs(parsed.query)
                token = str(query.get("session_token", [""])[0])
                state = self._state(token)
                if parsed.path == "/api/next":
                    with lock:
                        trial = current_trial(state)
                    self._json(
                        {
                            "complete": trial is None,
                            "trial": None if trial is None else _public_trial(trial, manifest),
                        }
                    )
                    return
                prefix = "/api/stimulus/"
                if parsed.path.startswith(prefix):
                    trial = state.get("current")
                    if not isinstance(trial, dict):
                        raise ValueError("there is no active trial")
                    stimulus_id = unquote(parsed.path[len(prefix) :])
                    if stimulus_id not in state["served_stimuli"]:
                        raise ValueError("stimulus is not part of the current trial")
                    if stimulus_id not in payload_cache:
                        payload_cache[stimulus_id] = _two_cycle_payload(
                            stimulus_root / stimuli[stimulus_id]["motion"],
                            phase_reference_path=(
                                stimulus_root / stimuli[stimulus_id]["phase_reference_motion"]
                            ),
                            viewing_settings=viewing_settings,
                        )
                    if (
                        payload_cache[stimulus_id]["render_hash"]
                        != stimuli[stimulus_id]["render_hash"]
                    ):
                        raise ValueError("rendered stimulus no longer matches the pilot manifest")
                    self._json(payload_cache[stimulus_id])
                    return
                self._send(HTTPStatus.NOT_FOUND, "text/plain", b"not found")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, "text/plain", str(exc).encode())

        def do_POST(self) -> None:
            try:
                self._ensure_collection_active()
                payload = self._payload()
                request_path = urlparse(self.path).path
                if request_path == "/api/session":
                    rater_id = str(payload.get("rater_id", "")).strip()
                    session_index = int(payload.get("session_index", -1))
                    if not rater_id or len(rater_id) > 128:
                        raise ValueError("a valid assigned participant ID is required")
                    if not 0 <= session_index < len(manifest["sessions"]):
                        raise ValueError("unknown session index")
                    token = uuid.uuid4().hex
                    session_id = _canonical_hash(
                        "session",
                        {"rater": rater_id, "index": session_index, "pilot_seed": manifest["seed"]},
                    )
                    sessions[token] = {
                        "rater_id": rater_id,
                        "session_index": session_index,
                        "session_id": session_id,
                        "current": None,
                        "completed_tutorial_ids": set(),
                    }
                    self._json({"session_token": token, "session_id": session_id})
                    return
                token = str(payload.get("session_token", ""))
                state = self._state(token)
                active_trial = state.get("current")
                trial = active_trial if isinstance(active_trial, dict) else None
                if trial is None or str(payload.get("trial_id")) != str(trial["trial_id"]):
                    raise ValueError("response does not match the current trial")
                is_calibration_tutorial = trial.get("task_type") == "calibration"
                if request_path in {"/api/playback-start", "/api/playback-complete"}:
                    stimulus_id = str(payload.get("stimulus_id"))
                    if stimulus_id not in state["served_stimuli"]:
                        raise ValueError("unknown stimulus for current trial")
                    if stimulus_id not in payload_cache:
                        payload_cache[stimulus_id] = _two_cycle_payload(
                            stimulus_root / stimuli[stimulus_id]["motion"],
                            phase_reference_path=(
                                stimulus_root / stimuli[stimulus_id]["phase_reference_motion"]
                            ),
                            viewing_settings=viewing_settings,
                        )
                    if (
                        payload_cache[stimulus_id]["render_hash"]
                        != stimuli[stimulus_id]["render_hash"]
                    ):
                        raise ValueError("rendered stimulus no longer matches the pilot manifest")
                    if request_path == "/api/playback-start":
                        if uses_mannequin and not is_calibration_tutorial:
                            playback_rate = payload.get("playback_rate")
                            if not _finite_number(playback_rate):
                                raise ValueError(
                                    "the first complete playback must use normal 1x speed"
                                )
                            playback_rate_value = float(playback_rate)  # type: ignore[arg-type]
                            if not math.isclose(
                                playback_rate_value,
                                float(viewing_settings["first_pass"]["playback_rate"]),
                                abs_tol=1.0e-9,
                            ):
                                raise ValueError(
                                    "the first complete playback must use normal 1x speed"
                                )
                            camera_state = _validate_camera_state(
                                payload.get("camera_state"),
                                viewing_settings,
                            )
                            if not _camera_is_default(camera_state, viewing_settings):
                                raise ValueError(
                                    "the first complete playback must use the standardized default camera"
                                )
                        state["playback_started"][stimulus_id] = {
                            "started_monotonic": time.monotonic(),
                            "playback_rate": 1.0,
                        }
                        self._json({"ok": True})
                        return
                    if is_calibration_tutorial:
                        # Calibration is an unrestricted inspection mode, not a scored
                        # first-pass gate. Accept completion calls from already-open legacy
                        # pages so a server update cannot strand a tutorial in progress.
                        state["completed_stimuli"].add(stimulus_id)
                        self._json({"ok": True})
                        return
                    started = state["playback_started"].get(stimulus_id)
                    required_seconds = max(
                        0.0,
                        float(payload_cache[stimulus_id]["duration_s"])
                        - 2.0
                        / float(
                            payload_cache[stimulus_id].get("fps", viewing_settings["frame_rate_hz"])
                        ),
                    )
                    if (
                        not isinstance(started, dict)
                        or time.monotonic() - float(started["started_monotonic"]) < required_seconds
                    ):
                        raise ValueError("the full normal-speed playback has not completed")
                    if uses_mannequin:
                        completed_rate = payload.get("playback_rate")
                        completed_camera = _validate_camera_state(
                            payload.get("camera_state"),
                            viewing_settings,
                        )
                        if not _finite_number(completed_rate):
                            raise ValueError(
                                "the completed first playback must remain at 1x and the default camera"
                            )
                        completed_rate_value = float(completed_rate)  # type: ignore[arg-type]
                        if not math.isclose(
                            completed_rate_value, 1.0, abs_tol=1.0e-9
                        ) or not _camera_is_default(completed_camera, viewing_settings):
                            raise ValueError(
                                "the completed first playback must remain at 1x and the default camera"
                            )
                    state["completed_stimuli"].add(stimulus_id)
                    self._json({"ok": True})
                    return
                if request_path == "/api/tutorial-complete":
                    if trial.get("task_type") != "calibration":
                        raise ValueError("the active trial is not a calibration tutorial")
                    with lock:
                        active_trial = state.get("current")
                        if not isinstance(active_trial, dict) or str(
                            active_trial.get("trial_id")
                        ) != str(trial["trial_id"]):
                            raise ValueError("response does not match the current trial")
                        state["completed_tutorial_ids"].add(str(trial["trial_id"]))
                        state["current"] = None
                    self._json({"ok": True})
                    return
                if request_path != "/api/observation":
                    self._send(HTTPStatus.NOT_FOUND, "text/plain", b"not found")
                    return
                if trial.get("task_type") == "calibration":
                    raise ValueError("calibration tutorials never create scored observations")
                if state["completed_stimuli"] != state["served_stimuli"]:
                    raise ValueError(
                        "complete every required full-speed playback before responding"
                    )
                response = payload.get("response")
                rating: int | None = None
                outcome: str | None = None
                if trial["task_type"] == "pair_comparison":
                    outcome = str(response)
                    if outcome not in PAIR_OUTCOMES:
                        raise ValueError("unsupported comparison outcome")
                else:
                    if not isinstance(response, int) or isinstance(response, bool):
                        raise ValueError("rating must be an integer from 1 through 7")
                    rating = int(response)
                    if not 1 <= rating <= 7:
                        raise ValueError("rating must be an integer from 1 through 7")
                confidence = payload.get("confidence")
                if confidence is not None and (
                    not isinstance(confidence, int) or not 1 <= confidence <= 5
                ):
                    raise ValueError("confidence must be null or an integer from 1 through 5")
                reasons = sorted(set(str(value) for value in payload.get("reason_tags", [])))
                if any(value not in REASON_TAGS for value in reasons):
                    raise ValueError("unsupported reason tag")
                replay_count = payload.get("replay_count", [])
                if (
                    not isinstance(replay_count, list)
                    or len(replay_count) != len(trial["presentation_order"])
                    or any(
                        not isinstance(value, int) or isinstance(value, bool) or value < 0
                        for value in replay_count
                    )
                ):
                    raise ValueError(
                        "replay count must contain one nonnegative integer per displayed motion"
                    )
                marked = payload.get("marked_interval_s")
                if marked is not None and (
                    not isinstance(marked, list)
                    or not 1 <= len(marked) <= 2
                    or any(not isinstance(value, (int, float)) or value < 0 for value in marked)
                ):
                    raise ValueError("marked interval must be null or one/two nonnegative times")
                inspection_telemetry: dict[str, Any] | None = None
                judgment_difficulty: str | None = None
                if uses_mannequin:
                    inspection_telemetry = _validate_inspection_telemetry(
                        payload.get("inspection_telemetry"),
                        stimulus_count=len(trial["presentation_order"]),
                        viewing_settings=viewing_settings,
                    )
                    if inspection_telemetry["replay_count"] != replay_count:
                        raise ValueError("top-level and inspection replay counts must match")
                    raw_difficulty = payload.get("judgment_difficulty")
                    if raw_difficulty is not None:
                        judgment_difficulty = str(raw_difficulty)
                        if judgment_difficulty not in JUDGMENT_DIFFICULTIES:
                            raise ValueError("unsupported judgment difficulty")
                public_order = list(trial["presentation_order"])
                canonical_ids = list(trial["stimulus_ids"])
                observation = {
                    "format_version": (
                        RAW_OBSERVATION_VERSION
                        if uses_mannequin
                        else LEGACY_RAW_OBSERVATION_VERSION
                    ),
                    "observation_id": _canonical_hash(
                        "observation",
                        {
                            "session": state["session_id"],
                            "trial": trial["trial_id"],
                            "time": time.time_ns(),
                        },
                    ),
                    "stimulus_ids": canonical_ids,
                    "source_ids": list(trial["source_ids"]),
                    "variant_ids": [stimuli[value]["variant_id"] for value in canonical_ids],
                    "families": list(trial["families"]),
                    "styles": list(trial["styles"]),
                    "render_protocol_hash": manifest["render_protocol_hash"],
                    "render_hashes": list(trial["render_hashes"]),
                    "rater_id": state["rater_id"],
                    "session_id": state["session_id"],
                    "session_index": state["session_index"],
                    "trial_id": trial["trial_id"],
                    "trial_index": trial["trial_index"],
                    "pair_id": trial.get("pair_id"),
                    "playlist_seed": trial["seed"],
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "task_type": trial["task_type"],
                    "rating": rating,
                    "pair_outcome_display_order": outcome,
                    "pair_outcome_canonical": _canonical_pair_outcome(
                        outcome, canonical_ids, public_order
                    ),
                    "presentation_order": public_order,
                    "comparison_mode": trial["comparison_mode"],
                    "response_time_ms": round(
                        (time.monotonic() - state["started_monotonic"]) * 1000.0, 3
                    ),
                    "replay_count": replay_count,
                    "hidden_repeat_group_ids": list(trial["hidden_repeat_group_ids"]),
                    "anchor_status": "hidden_session_anchor"
                    if trial["hidden_anchor"]
                    else "not_anchor",
                    "confidence": confidence,
                    "reason_tags": reasons,
                    "marked_interval_s": marked,
                    "viewing_settings": manifest["viewing_settings"],
                    "view_mirror_x": bool(trial.get("view_mirror_x", False)),
                    "presented_side_classes": list(trial["side_classes"]),
                    "schedule_reason": trial["schedule_reason"],
                    "equivalence_control": bool(trial.get("equivalence_control", False)),
                }
                if uses_mannequin:
                    observation.update(
                        {
                            "inspection_telemetry": inspection_telemetry,
                            "judgment_difficulty": judgment_difficulty,
                            "stimulus_populations": list(trial.get("stimulus_populations", [])),
                            "measurement_cohorts": list(trial.get("measurement_cohorts", [])),
                            "critic_training_eligible": bool(
                                trial.get("critic_training_eligible", False)
                            ),
                        }
                    )
                with lock:
                    active_trial = state.get("current")
                    if not isinstance(active_trial, dict) or str(
                        active_trial.get("trial_id")
                    ) != str(trial["trial_id"]):
                        raise ValueError("response does not match the current trial")
                    _append_jsonl(observations_path, observation)
                    state["current"] = None
                self._json({"ok": True, "observation_id": observation["observation_id"]})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, "text/plain", str(exc).encode())

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _canonical_pair_outcome(
    display_outcome: str | None,
    canonical_ids: list[str],
    display_order: list[str],
) -> str | None:
    if display_outcome not in {"a_better", "b_better"}:
        return display_outcome
    preferred_display = display_order[0] if display_outcome == "a_better" else display_order[1]
    return "first_better" if preferred_display == canonical_ids[0] else "second_better"
