"""Content-addressed freezes for an active mannequin human-evaluation protocol.

The pilot manifest and motion assets are immutable inputs. Observation and served-playlist
logs remain writable during collection, so their already-existing byte prefixes are frozen
instead: later appends are permitted while edits or truncation of collected evidence fail.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from motionlab.dataset.io import file_sha256
from motionlab.perceptual import calibration_bank, subjective, subjective_v3

PROTOCOL_FREEZE_VERSION = "motionlab.subjective_protocol_freeze.v1"
MANNEQUIN_SCHEDULER_VERSION = "motionlab.phase15_mannequin_scheduler.v1"

# These identities make version bumps mandatory rather than advisory. A changed browser
# bundle or motion payload needs a new render protocol/hash; changed server trial semantics
# need a new collection protocol; and changed scheduling code needs a new scheduler version.
_FROZEN_RENDER_IMPLEMENTATIONS = {
    "render:sha256:cea7579f211ffb88143ee439f1de9fbdca3af6e4394d9b86a9dd78c69cec1f4b": {
        "browser_ui_bundle_sha256": (
            "sha256:280a24671ac5e9e3583b53edc1c63194aa1dd9a6c135a3be295a6ff9068dbb63"
        ),
        "motion_payload_source_sha256": (
            "sha256:ca9f080958efeb08ba7d7727067ce1777f5e15e2c3fdab62ee669b16aef1dbde"
        ),
    },
    "render:sha256:9304a3c1d2f62892bfa927527569fa6f800c2d85a46942d954a1dc02f2b2d99e": {
        "browser_ui_bundle_sha256": (
            "sha256:f11a552e46631f83c6b1a4c2a7295cd6e0735a93e03b0f9141e71a790890cc22"
        ),
        "motion_payload_source_sha256": (
            "sha256:6a20817934e9fc09559fdda1e85d28cf7517e83038973e2173eafbd9d7efba5b"
        ),
    },
}
_FROZEN_COLLECTION_IMPLEMENTATIONS = {
    "motionlab.subjective_protocol.v3": {
        "collection_server_source_sha256": (
            "sha256:000b56bdd7d4b8d8b1144754f3777dbbe5ce524fed7fa21016f0ccb6d49c8886"
        ),
        "public_trial_source_sha256": (
            "sha256:3b1ccdf9dbfa049e7a869e676216f3edb7944fb89e4e81a53bdbbb6d6eea017d"
        ),
    },
    "motionlab.subjective_protocol.v4": {
        "collection_server_source_sha256": (
            "sha256:f7289e1d980d703a8a78ae9b39315ab0083303abf12759452eaaba5d4eeff5d5"
        ),
        "public_trial_source_sha256": (
            "sha256:3b1ccdf9dbfa049e7a869e676216f3edb7944fb89e4e81a53bdbbb6d6eea017d"
        ),
    },
}
_FROZEN_SCHEDULER_IMPLEMENTATIONS = {
    "motionlab.phase15_mannequin_scheduler.v1": (
        "sha256:444f8c0c5813c994fc96f2f62f7d9e477a55d06ad8cb734406e3bece055e3f0c"
    )
}

_NATURALNESS_PROMPT = (
    "Ignoring whether you personally like the style, how natural and internally coherent "
    "is this walk?"
)
_STYLE_PROMPT = "How well does this motion adhere to the requested style?"
_PAIR_PROMPT = "Which motion looks better overall?"
_DIFFICULTY_PROMPT = "How difficult was this judgment?"

_OBSERVATION_FIELDS = (
    "anchor_status",
    "comparison_mode",
    "confidence",
    "critic_training_eligible",
    "equivalence_control",
    "families",
    "format_version",
    "hidden_repeat_group_ids",
    "inspection_telemetry",
    "judgment_difficulty",
    "marked_interval_s",
    "measurement_cohorts",
    "observation_id",
    "pair_id",
    "pair_outcome_canonical",
    "pair_outcome_display_order",
    "playlist_seed",
    "presentation_order",
    "presented_side_classes",
    "rater_id",
    "rating",
    "reason_tags",
    "render_hashes",
    "render_protocol_hash",
    "replay_count",
    "response_time_ms",
    "schedule_reason",
    "session_id",
    "session_index",
    "source_ids",
    "stimulus_ids",
    "stimulus_populations",
    "styles",
    "task_type",
    "timestamp_utc",
    "trial_id",
    "trial_index",
    "variant_ids",
    "view_mirror_x",
    "viewing_settings",
)

_TELEMETRY_FIELDS = (
    "camera_change_events",
    "final_camera_state",
    "final_playback_rate",
    "format_version",
    "overlay_change_count",
    "playback_wall_time_by_speed_ms",
    "replay_count",
    "speed_change_events",
    "total_inspection_time_ms",
    "view_change_count",
    "zoom_change_count",
)

_PLAYLIST_FIELDS = (
    "format_version",
    "presentation_order",
    "rater_id",
    "schedule_reason",
    "seed",
    "session_id",
    "session_index",
    "task_type",
    "timestamp_utc",
    "trial_id",
    "trial_index",
)


def _canonical_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load protocol JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"protocol JSON must contain an object: {path}")
    return value


def _source_sha256(value: Any) -> str:
    return _sha256_bytes(inspect.getsource(value).encode("utf-8"))


def _rating_semantics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    scale_anchors = manifest.get("scale_anchors")
    if not isinstance(scale_anchors, dict):
        raise ValueError("pilot manifest has no rating scale anchors")
    expected_scales = {
        "naturalness": list(subjective.NATURALNESS_SCALE),
        "style_adherence": list(subjective.STYLE_SCALE),
    }
    if scale_anchors != expected_scales:
        raise ValueError("pilot rating scales differ from the active mannequin protocol")
    return {
        "single_stimulus": {
            "naturalness_question": _NATURALNESS_PROMPT,
            "naturalness_scale": expected_scales["naturalness"],
            "style_adherence_question": _STYLE_PROMPT,
            "style_adherence_scale": expected_scales["style_adherence"],
            "response_values": list(range(1, 8)),
        },
        "pair_comparison": {
            "question": _PAIR_PROMPT,
            "display_order": [
                {"value": "a_better", "label": "A better"},
                {"value": "b_better", "label": "B better"},
                {"value": "effectively_equal", "label": "Effectively equal"},
                {"value": "not_sure", "label": "Not sure"},
            ],
        },
        "confidence": {
            "optional": True,
            "values": list(range(1, 6)),
            "endpoint_labels": {"1": "very unsure", "3": "moderate", "5": "very sure"},
        },
        "reason_tags": {
            "optional": True,
            "values": sorted(subjective.REASON_TAGS),
        },
        "post_rating_difficulty": {
            "question": _DIFFICULTY_PROMPT,
            "optional": True,
            "values": ["easy", "moderate", "difficult"],
            "rating_and_inspection_frozen_before_prompt": True,
        },
        "marked_interval_seconds": {"optional": True, "maximum_points": 2},
    }


def _observation_schema(collection_protocol_version: str) -> dict[str, Any]:
    return {
        "format_version": subjective.RAW_OBSERVATION_VERSION,
        "required_top_level_fields": list(_OBSERVATION_FIELDS),
        "task_contract": {
            "naturalness": {"rating": "integer_1_through_7", "pair_outcome": None},
            "style_adherence": {"rating": "integer_1_through_7", "pair_outcome": None},
            "pair_comparison": {
                "rating": None,
                "display_outcomes": sorted(subjective.PAIR_OUTCOMES),
                "canonical_directional_outcomes": ["first_better", "second_better"],
            },
        },
        "inspection_telemetry": {
            "format_version": subjective.INSPECTION_TELEMETRY_VERSION,
            "required_fields": list(_TELEMETRY_FIELDS),
        },
        "served_playlist": {
            "format_version": collection_protocol_version,
            "required_fields": list(_PLAYLIST_FIELDS),
        },
    }


def _implementation_hashes(
    collection_protocol_version: str = subjective.SUBJECTIVE_PROTOCOL_VERSION,
) -> dict[str, str]:
    """Hash the implementation that actually serves the selected collection version."""
    implementation = (
        subjective_v3
        if collection_protocol_version == subjective.SUBJECTIVE_PROTOCOL_VERSION
        else subjective
    )
    return {
        "browser_ui_bundle_sha256": _sha256_bytes(implementation._HTML.encode("utf-8")),
        "collection_server_source_sha256": _source_sha256(
            implementation.serve_subjective_evaluator
        ),
        "motion_payload_source_sha256": _source_sha256(implementation._two_cycle_payload),
        "public_trial_source_sha256": _source_sha256(implementation._public_trial),
        "scheduler_source_sha256": _source_sha256(
            calibration_bank.prepare_mannequin_subjective_pilot
        ),
    }


def _validate_registered_implementations(
    *,
    render_protocol_hash: str,
    collection_protocol_version: str,
    hashes: Mapping[str, str],
) -> None:
    render_identity = _FROZEN_RENDER_IMPLEMENTATIONS.get(render_protocol_hash)
    if render_identity is None:
        raise ValueError("render protocol has no registered immutable implementation identity")
    if any(hashes.get(key) != value for key, value in render_identity.items()):
        raise ValueError("renderer or UI changed without a new render protocol and hash")
    collection_identity = _FROZEN_COLLECTION_IMPLEMENTATIONS.get(collection_protocol_version)
    if collection_identity is None:
        raise ValueError("collection protocol has no registered immutable implementation identity")
    if any(hashes.get(key) != value for key, value in collection_identity.items()):
        raise ValueError("collection behavior changed without a new protocol version")
    expected_scheduler = _FROZEN_SCHEDULER_IMPLEMENTATIONS.get(MANNEQUIN_SCHEDULER_VERSION)
    if expected_scheduler is None or hashes.get("scheduler_source_sha256") != expected_scheduler:
        raise ValueError("scheduler changed without a new scheduler version")


def _required_list(manifest: Mapping[str, Any], name: str) -> list[Any]:
    value = manifest.get(name)
    if not isinstance(value, list):
        raise ValueError(f"pilot manifest field {name!r} must be a list")
    return value


def _required_dict(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = manifest.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"pilot manifest field {name!r} must be an object")
    return value


def _anchor_contract(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    stimuli = _required_list(manifest, "stimuli")
    by_id = {
        str(item["stimulus_id"]): item
        for item in stimuli
        if isinstance(item, dict) and "stimulus_id" in item
    }
    anchor_ids = [str(value) for value in _required_list(manifest, "hidden_anchor_stimulus_ids")]
    if len(by_id) != len(stimuli) or len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("pilot manifest has duplicate or malformed stimulus IDs")
    result: list[dict[str, Any]] = []
    for stimulus_id in sorted(anchor_ids):
        item = by_id.get(stimulus_id)
        if item is None:
            raise ValueError(f"hidden anchor is absent from the stimulus table: {stimulus_id}")
        if item.get("stimulus_population") != calibration_bank.EVALUATION_ANCHOR:
            raise ValueError(f"hidden anchor has the wrong population: {stimulus_id}")
        result.append(
            {
                key: item.get(key)
                for key in (
                    "stimulus_id",
                    "render_hash",
                    "render_protocol_hash",
                    "motion_content_hash",
                    "visual_content_hash",
                    "family",
                    "calibration_severity",
                    "source_id",
                    "variant_id",
                    "stimulus_population",
                )
            }
        )
    return result


def build_active_evaluation_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the stable semantic contract used to derive the protocol identifier."""
    if manifest.get("format_version") != subjective.PILOT_MANIFEST_VERSION:
        raise ValueError("only the current mannequin pilot format can be frozen")
    collection_version = str(manifest.get("protocol_version", ""))
    if collection_version == subjective.SUBJECTIVE_PROTOCOL_VERSION:
        canonical_settings = subjective_v3.MANNEQUIN_VIEWING_SETTINGS
    elif collection_version == subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION:
        canonical_settings = subjective.FREE_CAMERA_VIEWING_SETTINGS
    else:
        raise ValueError("only a registered mannequin collection protocol can be frozen")
    viewing_settings = manifest.get("viewing_settings")
    if viewing_settings != canonical_settings:
        raise ValueError("pilot viewing settings differ from its mannequin collection protocol")
    expected_render_hash = subjective.render_protocol_hash(canonical_settings)
    if manifest.get("render_protocol_hash") != expected_render_hash:
        raise ValueError("pilot render hash differs from its mannequin viewing settings")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("pilot seed must be an integer")

    sessions = _required_list(manifest, "sessions")
    tutorial = _required_list(manifest, "calibration_tutorial")
    adaptive_pool = _required_list(manifest, "adaptive_comparison_pool")
    adaptive_policy = _required_dict(manifest, "adaptive_policy")
    repeated_anchors = sorted(
        str(value) for value in _required_list(manifest, "hidden_repeated_anchor_stimulus_ids")
    )
    repeated_hfp = sorted(
        str(value)
        for value in _required_list(
            manifest,
            "hidden_repeated_hard_feasible_stimulus_ids",
        )
    )
    schedule_payload = {
        "sessions": sessions,
        "calibration_tutorial": tutorial,
        "adaptive_comparison_pool": adaptive_pool,
        "adaptive_policy": adaptive_policy,
    }
    implementation_hashes = _implementation_hashes(collection_version)
    _validate_registered_implementations(
        render_protocol_hash=expected_render_hash,
        collection_protocol_version=collection_version,
        hashes=implementation_hashes,
    )
    return {
        "contract_version": PROTOCOL_FREEZE_VERSION,
        "pilot_versions": {
            "manifest": manifest["format_version"],
            "collection": manifest["protocol_version"],
        },
        "mannequin_render": {
            "protocol": viewing_settings["protocol"],
            "renderer_revision": viewing_settings["renderer_revision"],
            "geometry_revision": viewing_settings["geometry_revision"],
            "character": viewing_settings["character"],
            "render_protocol_hash": expected_render_hash,
            "viewing_settings": viewing_settings,
        },
        "camera_and_playback": {
            key: viewing_settings[key]
            for key in (
                "viewport_width_px",
                "viewport_height_px",
                "frame_rate_hz",
                "playback_rate",
                "playback_rates",
                "gait_cycles",
                "default_camera",
                "camera_presets",
                "camera_projection",
                "pixels_per_meter",
                "zoom_limits",
                "orbit",
                "root_horizontal_tracking",
                "skeleton_overlay",
                "first_pass",
            )
            + (("pan",) if "pan" in viewing_settings else ())
        },
        "rating_semantics": _rating_semantics(manifest),
        "hidden_repeat_policy": {
            "minimum_intervening_trials": manifest.get("minimum_intervening_trials_for_repeat"),
            "validation": _required_dict(manifest, "repeat_validation"),
            "repeated_anchor_stimulus_ids": repeated_anchors,
            "repeated_hard_feasible_stimulus_ids": repeated_hfp,
        },
        "anchor_set": _anchor_contract(manifest),
        "scheduler_randomization": {
            "scheduler_version": MANNEQUIN_SCHEDULER_VERSION,
            "seed": seed,
            "session_count": manifest.get("session_count"),
            "base_trial_counts": [session.get("trial_count") for session in sessions],
            "adaptive_policy": adaptive_policy,
            "exact_realized_schedule_id": _canonical_id("schedule", schedule_payload),
        },
        "observation_schema": _observation_schema(collection_version),
        "implementation_hashes": implementation_hashes,
    }


def _within_root(root: Path, raw_path: str | Path, *, base: Path, label: str) -> Path:
    supplied = Path(raw_path)
    resolved = (supplied if supplied.is_absolute() else base / supplied).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the active pilot directory: {raw_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    return resolved


def _declared_motion_paths(
    records: Sequence[Any],
    *,
    root: Path,
    base: Path,
    label: str,
) -> set[Path]:
    result: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label} stimulus {index} must be an object")
        for field in ("motion", "phase_reference_motion"):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} stimulus {index} has no {field}")
            result.add(
                _within_root(
                    root,
                    value,
                    base=base,
                    label=f"{label} {field}",
                )
            )
    return result


def _immutable_asset_paths(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> list[Path]:
    root = manifest_path.parent.resolve()
    paths = {manifest_path.resolve()}
    preparation = root / "pilot_preparation.json"
    if preparation.is_file():
        paths.add(preparation.resolve())
    continuation_history = root / "continuation_history.json"
    continuation = manifest.get("camera_continuation")
    if continuation is not None:
        if not isinstance(continuation, dict) or not isinstance(
            continuation.get("history_path"), str
        ):
            raise ValueError("pilot has malformed camera continuation metadata")
        declared_history = _within_root(
            root,
            continuation["history_path"],
            base=root,
            label="camera continuation history",
        )
        if declared_history != continuation_history:
            raise ValueError("camera continuation history must be the frozen sibling file")
    if continuation_history.is_file():
        paths.add(continuation_history.resolve())
    paths.update(
        _declared_motion_paths(
            _required_list(manifest, "stimuli"),
            root=root,
            base=root,
            label="pilot",
        )
    )

    calibration_path_value = manifest.get("calibration_bank")
    if not isinstance(calibration_path_value, str) or not calibration_path_value:
        raise ValueError("pilot manifest has no calibration-bank manifest")
    bank_path = _within_root(
        root,
        calibration_path_value,
        base=root,
        label="calibration-bank manifest",
    )
    bank_root = bank_path.parent
    bank = _load_json_object(bank_path)
    if bank.get("viewing_settings") != manifest.get("viewing_settings") or bank.get(
        "render_protocol_hash"
    ) != manifest.get("render_protocol_hash"):
        raise ValueError("calibration bank does not use the frozen rendering protocol")
    paths.add(bank_path)
    bank_preparation = bank_root / "calibration_bank_preparation.json"
    if bank_preparation.is_file():
        paths.add(bank_preparation.resolve())
    paths.update(
        _declared_motion_paths(
            _required_list(bank, "stimuli"),
            root=root,
            base=bank_root,
            label="calibration-bank",
        )
    )
    # Also preserve dormant/reference NPZs already present in the active pilot tree.
    # They are not necessarily served by this manifest, but collection must never silently
    # reuse a mutated in-place asset later.
    paths.update(path.resolve() for path in root.rglob("*.npz") if path.is_file())
    return sorted(paths)


def _immutable_asset_snapshot(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = manifest_path.parent.resolve()
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "byte_count": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in _immutable_asset_paths(manifest_path, manifest)
    ]


def _stable_jsonl_prefix(path: Path, *, attempts: int = 8) -> dict[str, Any]:
    destination = Path(path)
    if not destination.exists():
        return {
            "path": destination.name,
            "existed_at_freeze": False,
            "prefix_byte_count": 0,
            "prefix_record_count": 0,
            "prefix_sha256": _sha256_bytes(b""),
        }
    if not destination.is_file():
        raise ValueError(f"append-only evidence path is not a file: {destination}")
    for _ in range(attempts):
        before = destination.stat()
        data = destination.read_bytes()
        after = destination.stat()
        if (
            before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or len(data) != after.st_size
        ):
            time.sleep(0.01)
            continue
        if data and not data.endswith(b"\n"):
            time.sleep(0.01)
            continue
        try:
            rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"append-only evidence is not complete JSONL: {destination}") from exc
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"append-only evidence contains a non-object row: {destination}")
        return {
            "path": destination.name,
            "existed_at_freeze": True,
            "prefix_byte_count": len(data),
            "prefix_record_count": len(rows),
            "prefix_sha256": _sha256_bytes(data),
        }
    raise RuntimeError(f"append-only evidence changed continuously while freezing: {destination}")


def _validate_live_log_rows(
    snapshot: Mapping[str, Any],
    path: Path,
    *,
    expected_version: str,
    manifest: Mapping[str, Any],
) -> None:
    byte_count = snapshot.get("prefix_byte_count")
    expected_digest = snapshot.get("prefix_sha256")
    if not isinstance(byte_count, int) or byte_count < 0 or not isinstance(expected_digest, str):
        raise ValueError("protocol freeze has a malformed append-only prefix record")
    if not path.is_file():
        if not snapshot.get("existed_at_freeze") and byte_count == 0:
            return
        raise ValueError(f"frozen append-only evidence is missing: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(byte_count)
    if len(prefix) != byte_count or _sha256_bytes(prefix) != expected_digest:
        raise ValueError(f"append-only evidence prefix drifted: {path}")
    try:
        rows = [json.loads(line) for line in prefix.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"frozen append-only prefix is invalid JSONL: {path}") from exc
    if len(rows) != snapshot.get("prefix_record_count"):
        raise ValueError(f"frozen append-only prefix count drifted: {path}")
    for row in rows:
        if not isinstance(row, dict) or row.get("format_version") != expected_version:
            raise ValueError(f"frozen append-only prefix uses the wrong schema: {path}")
        if expected_version == subjective.RAW_OBSERVATION_VERSION:
            if set(row) != set(_OBSERVATION_FIELDS):
                raise ValueError(f"frozen observation fields do not match the schema: {path}")
            telemetry = row.get("inspection_telemetry")
            if (
                not isinstance(telemetry, dict)
                or set(telemetry) != set(_TELEMETRY_FIELDS)
                or telemetry.get("format_version") != subjective.INSPECTION_TELEMETRY_VERSION
            ):
                raise ValueError(f"frozen observation telemetry does not match the schema: {path}")
            if row.get("render_protocol_hash") != manifest.get("render_protocol_hash"):
                raise ValueError(f"frozen observation uses the wrong render protocol: {path}")
            if row.get("playlist_seed") != manifest.get("seed"):
                raise ValueError(f"frozen observation uses the wrong pilot seed: {path}")
        else:
            if set(row) != set(_PLAYLIST_FIELDS):
                raise ValueError(f"frozen playlist fields do not match the schema: {path}")
            if row.get("seed") != manifest.get("seed"):
                raise ValueError(f"frozen playlist row uses the wrong pilot seed: {path}")


def _relative_evidence_path(root: Path, path: Path, *, label: str) -> Path:
    destination = Path(path).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the active pilot directory") from exc
    return destination


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def freeze_active_evaluation_protocol(
    pilot_manifest_path: Path,
    *,
    freeze_path: Path | None = None,
    observations_path: Path | None = None,
    served_playlist_path: Path | None = None,
) -> dict[str, Any]:
    """Create a new immutable protocol freeze without touching active pilot inputs."""
    manifest_path = Path(pilot_manifest_path).resolve()
    root = manifest_path.parent
    output = root / "protocol_freeze.json" if freeze_path is None else Path(freeze_path).resolve()
    if output.parent != root:
        raise ValueError("protocol freeze must be stored beside the active pilot manifest")
    if output.exists():
        raise FileExistsError(
            f"protocol freeze already exists and will not be overwritten: {output}"
        )
    manifest = _load_json_object(manifest_path)
    contract = build_active_evaluation_contract(manifest)
    assets = _immutable_asset_snapshot(manifest_path, manifest)
    observation_log = _relative_evidence_path(
        root,
        root / "raw_observations.jsonl" if observations_path is None else observations_path,
        label="observation log",
    )
    playlist_log = _relative_evidence_path(
        root,
        root / "served_playlist.jsonl" if served_playlist_path is None else served_playlist_path,
        label="served-playlist log",
    )
    observation_prefix = _stable_jsonl_prefix(observation_log)
    observation_prefix["path"] = observation_log.relative_to(root).as_posix()
    playlist_prefix = _stable_jsonl_prefix(playlist_log)
    playlist_prefix["path"] = playlist_log.relative_to(root).as_posix()
    prefixes = {
        "observations": observation_prefix,
        "served_playlist": playlist_prefix,
    }
    _validate_live_log_rows(
        prefixes["observations"],
        observation_log,
        expected_version=subjective.RAW_OBSERVATION_VERSION,
        manifest=manifest,
    )
    _validate_live_log_rows(
        prefixes["served_playlist"],
        playlist_log,
        expected_version=str(manifest["protocol_version"]),
        manifest=manifest,
    )
    protocol_id = _canonical_id("human-evaluation-protocol", contract)
    immutable_snapshot_id = _canonical_id("active-pilot-snapshot", assets)
    record: dict[str, Any] = {
        "format_version": PROTOCOL_FREEZE_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_active_collection_protocol",
        "pilot_manifest": manifest_path.name,
        "protocol_id": protocol_id,
        "immutable_snapshot_id": immutable_snapshot_id,
        "contract": contract,
        "immutable_assets": assets,
        "append_only_evidence_prefixes": prefixes,
        "mutation_policy": {
            "immutable_assets": "exact_bytes_required",
            "observations": "append_only_existing_prefix_required",
            "served_playlist": "append_only_existing_prefix_required",
            "future_render_or_ui_changes_require_new_protocol_and_render_hashes": True,
        },
    }
    record["freeze_id"] = _canonical_id("protocol-freeze", record)
    _atomic_write_json(output, record)
    return record


def validate_active_evaluation_protocol(
    freeze_path: Path,
    *,
    pilot_manifest_path: Path | None = None,
    observations_path: Path | None = None,
    served_playlist_path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed if a frozen contract, asset, or collected-evidence prefix drifted."""
    source = Path(freeze_path).resolve()
    root = source.parent
    freeze = _load_json_object(source)
    if freeze.get("format_version") != PROTOCOL_FREEZE_VERSION:
        raise ValueError("unsupported active-evaluation protocol freeze")
    expected_freeze_id = freeze.get("freeze_id")
    without_id = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if expected_freeze_id != _canonical_id("protocol-freeze", without_id):
        raise ValueError("protocol freeze record itself has drifted")

    manifest_name = freeze.get("pilot_manifest")
    if not isinstance(manifest_name, str) or not manifest_name:
        raise ValueError("protocol freeze has no pilot manifest path")
    manifest_path = (
        _within_root(root, manifest_name, base=root, label="pilot manifest")
        if pilot_manifest_path is None
        else Path(pilot_manifest_path).resolve()
    )
    if manifest_path.parent != root:
        raise ValueError("frozen pilot manifest must remain beside the freeze record")
    manifest = _load_json_object(manifest_path)
    contract = build_active_evaluation_contract(manifest)
    if freeze.get("contract") != contract:
        raise ValueError("active human-evaluation protocol contract drifted")
    if freeze.get("protocol_id") != _canonical_id("human-evaluation-protocol", contract):
        raise ValueError("active human-evaluation protocol identifier is invalid")

    assets = _immutable_asset_snapshot(manifest_path, manifest)
    if freeze.get("immutable_assets") != assets:
        raise ValueError("active pilot immutable assets drifted")
    if freeze.get("immutable_snapshot_id") != _canonical_id("active-pilot-snapshot", assets):
        raise ValueError("active pilot snapshot identifier is invalid")

    prefixes = freeze.get("append_only_evidence_prefixes")
    if not isinstance(prefixes, dict):
        raise ValueError("protocol freeze has no append-only evidence prefixes")
    observation_record = prefixes.get("observations")
    playlist_record = prefixes.get("served_playlist")
    if not isinstance(observation_record, dict) or not isinstance(playlist_record, dict):
        raise ValueError("protocol freeze has malformed append-only evidence prefixes")
    observation_log = _relative_evidence_path(
        root,
        root / str(observation_record.get("path"))
        if observations_path is None
        else observations_path,
        label="observation log",
    )
    playlist_log = _relative_evidence_path(
        root,
        root / str(playlist_record.get("path"))
        if served_playlist_path is None
        else served_playlist_path,
        label="served-playlist log",
    )
    _validate_live_log_rows(
        observation_record,
        observation_log,
        expected_version=subjective.RAW_OBSERVATION_VERSION,
        manifest=manifest,
    )
    _validate_live_log_rows(
        playlist_record,
        playlist_log,
        expected_version=str(manifest["protocol_version"]),
        manifest=manifest,
    )
    # The frozen records above protect history. A fresh stable snapshot also checks that
    # observations appended after the freeze continue to honor the same persisted schemas.
    current_observations = _stable_jsonl_prefix(observation_log)
    current_playlist = _stable_jsonl_prefix(playlist_log)
    _validate_live_log_rows(
        current_observations,
        observation_log,
        expected_version=subjective.RAW_OBSERVATION_VERSION,
        manifest=manifest,
    )
    _validate_live_log_rows(
        current_playlist,
        playlist_log,
        expected_version=str(manifest["protocol_version"]),
        manifest=manifest,
    )
    return {
        "status": "valid",
        "protocol_id": freeze["protocol_id"],
        "freeze_id": freeze["freeze_id"],
        "immutable_asset_count": len(assets),
        "observation_prefix_record_count": observation_record["prefix_record_count"],
        "served_playlist_prefix_record_count": playlist_record["prefix_record_count"],
        "current_observation_record_count": current_observations["prefix_record_count"],
        "current_served_playlist_record_count": current_playlist["prefix_record_count"],
        "append_only_growth_permitted": True,
    }
