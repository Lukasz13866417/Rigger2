"""Version a camera-only pilot continuation without relabelling historical evidence.

Inherited responses are authenticated once and reduced to private scheduler inputs. They
are never copied into the new observation or served-playlist logs, and do not become
observations of the new renderer merely because their completion status is carried over.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from motionlab.dataset.io import file_sha256
from motionlab.perceptual import subjective
from motionlab.perceptual.calibration_bank import _presented_render_hash, _rendered_geometry_hash
from motionlab.perceptual.observation_auth import (
    resolve_frozen_evidence_paths,
    validate_current_protocol_observations,
)

CAMERA_CONTINUATION_VERSION = "motionlab.camera_continuation.v1"
_HISTORY_FILENAME = "continuation_history.json"
_SCHEDULING_FIELDS = (
    "rater_id",
    "session_id",
    "session_index",
    "trial_index",
    "task_type",
    "pair_id",
    "rating",
    "families",
    "schedule_reason",
)


def _identity(prefix: str, value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"{prefix}:sha256:{hashlib.sha256(data).hexdigest()}"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read camera continuation object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"camera continuation requires a JSON object: {path}")
    return value


def _write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _inside(root: Path, raw: str | Path) -> Path:
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("camera continuation asset path escapes its pilot directory")
    return path


def _log_snapshot(path: Path) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    data = path.read_bytes() if path.exists() else b""
    if data and not data.endswith(b"\n"):
        raise ValueError(f"camera continuation refuses an incomplete log line: {path}")
    try:
        rows = [json.loads(line) for line in data.splitlines()]
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"camera continuation evidence is not valid JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("camera continuation evidence rows must be objects")
    return (
        rows,
        data,
        {
            "path": str(path),
            "prefix_byte_count": len(data),
            "prefix_record_count": len(rows),
            "prefix_sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        },
    )


def _new_identifier(old: str, render_hash: str) -> str:
    return _identity(old.split(":", 1)[0], {"predecessor_id": old, "render": render_hash})


def prepare_camera_continuation(
    pilot_manifest_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Prepare fresh immutable assets plus private progress; never write old evidence.

    Stop the predecessor collector before the actual cutover. A consistent authenticated
    prefix is captured here; later predecessor appends cannot silently migrate into a
    previously prepared continuation. A new protocol freeze must be made before serving.
    """
    pilot_path = Path(pilot_manifest_path).resolve()
    source_root = pilot_path.parent
    output = Path(output_directory).resolve()
    if output.exists():
        raise ValueError("camera continuation output directory must not already exist")
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("camera continuation output must be separate from the original pilot")
    source = _read_object(pilot_path)
    if (
        source.get("protocol_version") != subjective.SUBJECTIVE_PROTOCOL_VERSION
        or source.get("viewing_settings") != subjective.MANNEQUIN_VIEWING_SETTINGS
    ):
        raise ValueError("camera continuation requires the frozen v3 mannequin predecessor")
    freeze_path = source_root / "protocol_freeze.json"
    freeze = _read_object(freeze_path)
    observation_path = _inside(
        source_root, freeze["append_only_evidence_prefixes"]["observations"]["path"]
    )
    served_path, _ = resolve_frozen_evidence_paths(pilot_path, observation_path, freeze_path)
    observations, observation_bytes, observation_snapshot = _log_snapshot(observation_path)
    served, served_bytes, served_snapshot = _log_snapshot(served_path)
    validate_current_protocol_observations(observations, source, served)

    settings = copy.deepcopy(subjective.FREE_CAMERA_VIEWING_SETTINGS)
    render_hash = subjective.render_protocol_hash(settings)
    manifest = copy.deepcopy(source)
    bank_source_path = _inside(source_root, str(source["calibration_bank"]))
    bank_source = _read_object(bank_source_path)
    bank = copy.deepcopy(bank_source)
    bank_relative = bank_source_path.relative_to(source_root)
    bank_output_root = output / bank_relative.parent
    maps: dict[str, dict[str, str]] = {
        "stimulus_ids": {},
        "trial_ids": {},
        "repeat_group_ids": {},
        "tutorial_ids": {},
    }

    output.mkdir(parents=True)
    copied_motions: list[dict[str, Any]] = []
    for asset in freeze["immutable_assets"]:
        relative = str(asset["path"])
        if not relative.endswith(".npz"):
            continue
        original = _inside(source_root, relative)
        destination = _inside(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        if file_sha256(destination) != asset["sha256"]:
            raise ValueError("camera continuation motion changed while copying frozen assets")
        copied_motions.append({"path": relative, "sha256": asset["sha256"]})

    def remap_repeat(old: str) -> str:
        maps["repeat_group_ids"].setdefault(old, _new_identifier(old, render_hash))
        return maps["repeat_group_ids"][old]

    def convert_stimuli(items: list[dict[str, Any]], asset_root: Path) -> None:
        for item in items:
            old = str(item["stimulus_id"])
            maps["stimulus_ids"].setdefault(old, _new_identifier(old, render_hash))
            item["stimulus_id"] = maps["stimulus_ids"][old]
            motion = _inside(asset_root, str(item["motion"]))
            reference = _inside(asset_root, str(item.get("phase_reference_motion", item["motion"])))
            payload = subjective._two_cycle_payload(
                motion, phase_reference_path=reference, viewing_settings=settings
            )
            item.update(
                render_protocol_hash=render_hash,
                render_hash=payload["render_hash"],
                cycle_window=payload["cycle_window"],
            )
            if "rendered_geometry_hash" in item:
                item["rendered_geometry_hash"] = _rendered_geometry_hash(payload)
            if item.get("hidden_repeat_group_id") is not None:
                item["hidden_repeat_group_id"] = remap_repeat(str(item["hidden_repeat_group_id"]))

    convert_stimuli(manifest["stimuli"], output)
    convert_stimuli(bank["stimuli"], bank_output_root)
    stimuli = {item["stimulus_id"]: item for item in manifest["stimuli"]}

    def convert_trial(trial: dict[str, Any], *, tutorial: bool = False) -> None:
        field = "tutorial_id" if tutorial else "trial_id"
        map_name = "tutorial_ids" if tutorial else "trial_ids"
        old = str(trial[field])
        maps[map_name].setdefault(old, _new_identifier(old, render_hash))
        trial[field] = maps[map_name][old]
        for ids_field in ("stimulus_ids", "presentation_order"):
            trial[ids_field] = [maps["stimulus_ids"][value] for value in trial[ids_field]]
        if "hidden_repeat_group_ids" in trial:
            trial["hidden_repeat_group_ids"] = [
                remap_repeat(value) for value in trial["hidden_repeat_group_ids"]
            ]
        trial["render_hashes"] = [
            _presented_render_hash(stimuli[value], bool(trial.get("view_mirror_x", False)))
            for value in trial["stimulus_ids"]
        ]

    for session in manifest["sessions"]:
        for trial in session["trials"]:
            convert_trial(trial)
    for trial in manifest["adaptive_comparison_pool"]:
        convert_trial(trial)
    for tutorial in manifest["calibration_tutorial"]:
        convert_trial(tutorial, tutorial=True)
    for key in (
        "scored_stimulus_ids",
        "hidden_anchor_stimulus_ids",
        "hidden_repeated_anchor_stimulus_ids",
        "hidden_repeated_hard_feasible_stimulus_ids",
    ):
        if key in manifest:
            manifest[key] = [maps["stimulus_ids"][value] for value in manifest[key]]
    for ladder in bank.get("ladders", []):
        ladder["stimulus_ids"] = [maps["stimulus_ids"][value] for value in ladder["stimulus_ids"]]

    scheduling_rows = []
    for row in observations:
        inherited = {key: copy.deepcopy(row.get(key)) for key in _SCHEDULING_FIELDS}
        inherited.update(
            _scheduling_only=True,
            trial_id=maps["trial_ids"][row["trial_id"]],
            stimulus_ids=[maps["stimulus_ids"][value] for value in row["stimulus_ids"]],
            hidden_repeat_group_ids=[
                maps["repeat_group_ids"][value] for value in row["hidden_repeat_group_ids"]
            ],
            source_observation_id=row["observation_id"],
            source_trial_id=row["trial_id"],
        )
        scheduling_rows.append(inherited)
    history: dict[str, Any] = {
        "format_version": CAMERA_CONTINUATION_VERSION,
        "purpose": "private_scheduler_progress_only_not_new_protocol_observations",
        "source_pilot_manifest": str(pilot_path),
        "source_manifest_sha256": file_sha256(pilot_path),
        "source_protocol_freeze": str(freeze_path),
        "source_freeze_sha256": file_sha256(freeze_path),
        "source_protocol_id": freeze["protocol_id"],
        "source_freeze_id": freeze["freeze_id"],
        "source_immutable_snapshot_id": freeze["immutable_snapshot_id"],
        "source_render_protocol_hash": source["render_protocol_hash"],
        "destination_render_protocol_hash": render_hash,
        "evidence_snapshots": {
            "observations": observation_snapshot,
            "served_playlist": served_snapshot,
        },
        "identity_maps": maps,
        "copied_motion_assets": copied_motions,
        "scheduling_rows": scheduling_rows,
        "tutorial_completion_policy": "scored_session_zero_response_required_never_served_only",
        "new_human_observations_created": 0,
        "analysis_policy": "analyze_original_observations_under_original_protocol_only",
    }
    history["history_id"] = _identity("camera-continuation", history)
    history_path = output / _HISTORY_FILENAME
    manifest.update(
        protocol_version=subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION,
        viewing_settings=settings,
        render_protocol_hash=render_hash,
        stimulus_directory=str(output),
        calibration_bank=str(output / bank_relative),
        created_utc=datetime.now(UTC).isoformat(),
        pilot_status="camera_continuation_requires_new_protocol_freeze",
        camera_continuation={
            "history_path": str(history_path),
            "history_id": history["history_id"],
            "source_protocol_id": freeze["protocol_id"],
            "source_render_protocol_hash": source["render_protocol_hash"],
            "inherited_scored_trial_count": len(scheduling_rows),
            "inherited_rows_are_new_observations": False,
        },
    )
    if "render_validation" in manifest:
        manifest["render_validation"]["renderer_protocol"] = settings["protocol"]
    bank.update(
        viewing_settings=settings,
        render_protocol_hash=render_hash,
        stimulus_directory=str(bank_output_root),
        camera_continuation_source=str(bank_source_path),
    )
    # A second check closes immutable-asset races. Only additional complete log rows are
    # allowed; captured bytes must still be exact prefixes after asset preparation.
    resolve_frozen_evidence_paths(pilot_path, observation_path, freeze_path)
    for path, captured in ((observation_path, observation_bytes), (served_path, served_bytes)):
        current = path.read_bytes() if path.exists() else b""
        if not current.startswith(captured):
            raise ValueError("camera continuation source evidence prefix drifted while preparing")
    _write_object(history_path, history)
    _write_object(output / bank_relative, bank)
    manifest_output = output / pilot_path.name
    _write_object(manifest_output, manifest)
    report = {
        "format_version": CAMERA_CONTINUATION_VERSION,
        "status": "prepared_requires_freeze_before_serving",
        "pilot_manifest": str(manifest_output),
        "history_id": history["history_id"],
        "source_protocol_id": freeze["protocol_id"],
        "render_protocol_hash": render_hash,
        "inherited_scored_trial_count": len(scheduling_rows),
        "copied_motion_count": len(copied_motions),
        "new_human_observations_created": 0,
        "original_pilot_modified": False,
    }
    _write_object(output / "pilot_preparation.json", report)
    return report


def load_camera_continuation_history(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return authenticated scheduling-only progress, never analysis observations."""
    declaration = manifest.get("camera_continuation")
    if declaration is None:
        return []
    if not isinstance(declaration, dict):
        raise ValueError("camera continuation declaration must be an object")
    if manifest.get("protocol_version") != subjective.FREE_CAMERA_SUBJECTIVE_PROTOCOL_VERSION:
        raise ValueError("camera continuation requires the new collection protocol")
    root = Path(str(manifest["stimulus_directory"])).resolve()
    raw_path = declaration.get("history_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("camera continuation has no history path")
    path = _inside(root, raw_path)
    history = _read_object(path)
    without_id = {key: value for key, value in history.items() if key != "history_id"}
    expected_id = _identity("camera-continuation", without_id)
    if history.get("history_id") != expected_id or declaration.get("history_id") != expected_id:
        raise ValueError("camera continuation history hash drifted")
    if (
        history.get("format_version") != CAMERA_CONTINUATION_VERSION
        or history.get("destination_render_protocol_hash") != manifest.get("render_protocol_hash")
        or history.get("source_protocol_id") != declaration.get("source_protocol_id")
        or declaration.get("inherited_rows_are_new_observations") is not False
    ):
        raise ValueError("camera continuation provenance does not match its manifest")
    rows = history.get("scheduling_rows")
    if not isinstance(rows, list) or len(rows) != declaration.get("inherited_scored_trial_count"):
        raise ValueError("camera continuation has invalid scheduling rows")
    if any(
        not isinstance(row, dict)
        or row.get("_scheduling_only") is not True
        or "observation_id" in row
        or "format_version" in row
        for row in rows
    ):
        raise ValueError("camera continuation rows must not impersonate new observations")
    return copy.deepcopy(rows)
