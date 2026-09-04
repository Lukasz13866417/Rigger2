"""Authentication of frozen subjective-observation evidence.

The append-only protocol freeze protects bytes and schemas.  This module adds the
semantic half of the boundary: an observation must describe the exact manifest trial
that the server recorded as served, and analysis must read the evidence paths named by
the freeze rather than an arbitrary look-alike JSONL file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from motionlab.perceptual.protocol_freeze import validate_active_evaluation_protocol
from motionlab.perceptual.subjective import (
    INSPECTION_TELEMETRY_VERSION,
    JUDGMENT_DIFFICULTIES,
    PAIR_OUTCOMES,
    RAW_OBSERVATION_VERSION,
    REASON_TAGS,
    _canonical_pair_outcome,
    _validate_camera_state,
    _validate_inspection_telemetry,
)

_OBSERVATION_ID_PATTERN = re.compile(r"^observation:sha256:[0-9a-f]{64}$")
_PAIR_CANONICAL_OUTCOMES = {
    "first_better",
    "second_better",
    "effectively_equal",
    "not_sure",
}
_ADAPTIVE_REASON_PREFIX = "adaptive:response_adaptive_toggle_pool+"
_ADAPTIVE_REASONS = {
    "close_single_ratings",
    "repeat_disagreement",
    "underrepresented_family",
}


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load protocol freeze: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("protocol freeze must contain a JSON object")
    return value


def _frozen_path(root: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"protocol freeze has no canonical {label} record")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"protocol freeze has no canonical {label} path")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"frozen {label} path escapes the active pilot directory") from exc
    return path


def resolve_frozen_evidence_paths(
    pilot_manifest_path: Path,
    observations_path: Path,
    protocol_freeze_path: Path,
    *,
    served_playlist_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate the one canonical observation/playlist evidence pair.

    This is intentionally reusable by post-label tools.  Supplying a copied or filtered
    observation file is rejected even when it happens to live under the pilot directory.
    """
    pilot = Path(pilot_manifest_path).resolve()
    observations = Path(observations_path).resolve()
    freeze_path = Path(protocol_freeze_path).resolve()
    root = pilot.parent
    if freeze_path.parent != root:
        raise ValueError("protocol freeze must be stored beside the pilot manifest")
    freeze = _load_json_object(freeze_path)
    raw_manifest = freeze.get("pilot_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise ValueError("protocol freeze has no canonical pilot manifest path")
    frozen_manifest = (root / raw_manifest).resolve()
    if frozen_manifest != pilot:
        raise ValueError("analysis requires the canonical frozen pilot manifest")
    prefixes = freeze.get("append_only_evidence_prefixes")
    if not isinstance(prefixes, Mapping):
        raise ValueError("protocol freeze has no append-only evidence paths")
    frozen_observations = _frozen_path(
        root,
        prefixes.get("observations"),
        label="observation log",
    )
    frozen_playlist = _frozen_path(
        root,
        prefixes.get("served_playlist"),
        label="served-playlist log",
    )
    if observations != frozen_observations:
        raise ValueError("analysis requires the canonical frozen observation log")
    if served_playlist_path is not None and Path(served_playlist_path).resolve() != frozen_playlist:
        raise ValueError("analysis requires the canonical frozen served-playlist log")
    validation = validate_active_evaluation_protocol(
        freeze_path,
        pilot_manifest_path=pilot,
        observations_path=frozen_observations,
        served_playlist_path=frozen_playlist,
    )
    return frozen_playlist, validation


def served_observation_key(
    row: Mapping[str, Any],
    *,
    observation: bool,
) -> tuple[Any, ...]:
    """Return the fields persisted in both a served event and its response."""
    order = row.get("presentation_order")
    return (
        row.get("rater_id"),
        row.get("session_id"),
        row.get("session_index"),
        row.get("trial_id"),
        row.get("trial_index"),
        row.get("playlist_seed" if observation else "seed"),
        row.get("task_type"),
        tuple(order) if isinstance(order, list) else None,
        row.get("schedule_reason"),
    )


def _canonical_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp must include a timezone")
    return parsed


def _manifest_trial_index(
    manifest: Mapping[str, Any],
) -> tuple[dict[tuple[int, str], tuple[Mapping[str, Any], bool]], dict[int, int], dict[int, int]]:
    index: dict[tuple[int, str], tuple[Mapping[str, Any], bool]] = {}
    base_counts: dict[int, int] = {}
    adaptive_counts: dict[int, int] = {}

    def register(raw: Any, *, session_index: int, adaptive: bool) -> None:
        if not isinstance(raw, Mapping):
            raise ValueError("pilot contains a malformed scored trial")
        trial_id = raw.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("pilot contains a scored trial without a trial ID")
        key = (session_index, trial_id)
        if key in index:
            raise ValueError(f"pilot contains duplicate scored trial ID: {trial_id}")
        index[key] = (raw, adaptive)

    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("current pilot has no session trial contract")
    for fallback_index, session in enumerate(sessions):
        if not isinstance(session, Mapping) or not isinstance(session.get("trials"), list):
            raise ValueError("current pilot has a malformed session trial contract")
        session_index = session.get("session_index", fallback_index)
        if not isinstance(session_index, int) or isinstance(session_index, bool):
            raise ValueError("pilot session index is invalid")
        trials = session["trials"]
        base_counts[session_index] = len(trials)
        adaptive_counts.setdefault(session_index, 0)
        for trial in trials:
            register(trial, session_index=session_index, adaptive=False)
    adaptive_pool = manifest.get("adaptive_comparison_pool", [])
    if not isinstance(adaptive_pool, list):
        raise ValueError("current pilot has a malformed adaptive comparison pool")
    for trial in adaptive_pool:
        if not isinstance(trial, Mapping):
            raise ValueError("current pilot contains a malformed adaptive trial")
        session_index = trial.get("session_index")
        if not isinstance(session_index, int) or isinstance(session_index, bool):
            raise ValueError("adaptive trial session index is invalid")
        adaptive_counts[session_index] = adaptive_counts.get(session_index, 0) + 1
        register(trial, session_index=session_index, adaptive=True)
    return index, base_counts, adaptive_counts


def _expect_equal(
    row: Mapping[str, Any],
    field: str,
    expected: Any,
    *,
    observation_id: str,
) -> None:
    if row.get(field) != expected:
        raise ValueError(
            f"observation {observation_id} {field} disagrees with its frozen pilot trial"
        )


def _validate_response_semantics(
    row: Mapping[str, Any],
    *,
    observation_id: str,
    stimulus_ids: list[str],
) -> None:
    task_type = row.get("task_type")
    if task_type in {"naturalness", "style_adherence"}:
        rating = row.get("rating")
        if (
            len(stimulus_ids) != 1
            or not isinstance(rating, int)
            or isinstance(rating, bool)
            or not 1 <= rating <= 7
            or row.get("pair_outcome_display_order") is not None
            or row.get("pair_outcome_canonical") is not None
        ):
            raise ValueError(f"observation {observation_id} has no valid completed rating")
        return
    if task_type != "pair_comparison":
        raise ValueError(f"observation {observation_id} has an unsupported task type")
    display_outcome = row.get("pair_outcome_display_order")
    canonical_outcome = row.get("pair_outcome_canonical")
    presentation_order = row.get("presentation_order")
    if (
        len(stimulus_ids) != 2
        or len(set(stimulus_ids)) != 2
        or row.get("rating") is not None
        or display_outcome not in PAIR_OUTCOMES
        or canonical_outcome not in _PAIR_CANONICAL_OUTCOMES
        or not isinstance(presentation_order, list)
        or _canonical_pair_outcome(
            str(display_outcome),
            stimulus_ids,
            [str(value) for value in presentation_order],
        )
        != canonical_outcome
    ):
        raise ValueError(f"observation {observation_id} has no valid completed comparison")


def _validate_response_metadata(
    row: Mapping[str, Any],
    *,
    observation_id: str,
    stimulus_count: int,
    viewing_settings: Mapping[str, Any],
) -> None:
    confidence = row.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, int) or isinstance(confidence, bool) or not 1 <= confidence <= 5
    ):
        raise ValueError(f"observation {observation_id} confidence is invalid")
    reasons = row.get("reason_tags")
    if (
        not isinstance(reasons, list)
        or reasons != sorted(set(reasons))
        or any(not isinstance(value, str) or value not in REASON_TAGS for value in reasons)
    ):
        raise ValueError(f"observation {observation_id} reason tags are invalid")
    marked = row.get("marked_interval_s")
    if marked is not None and (
        not isinstance(marked, list)
        or not 1 <= len(marked) <= 2
        or any(not _finite_nonnegative(value) for value in marked)
    ):
        raise ValueError(f"observation {observation_id} marked interval is invalid")
    difficulty = row.get("judgment_difficulty")
    if difficulty is not None and difficulty not in JUDGMENT_DIFFICULTIES:
        raise ValueError(f"observation {observation_id} judgment difficulty is invalid")
    replay_count = row.get("replay_count")
    if (
        not isinstance(replay_count, list)
        or len(replay_count) != stimulus_count
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in replay_count
        )
    ):
        raise ValueError(f"observation {observation_id} replay counts are invalid")
    if not _finite_nonnegative(row.get("response_time_ms")):
        raise ValueError(f"observation {observation_id} response time is invalid")
    telemetry = row.get("inspection_telemetry")
    normalized = _validate_inspection_telemetry(
        telemetry,
        stimulus_count=stimulus_count,
        viewing_settings=dict(viewing_settings),
    )
    if normalized.get("format_version") != INSPECTION_TELEMETRY_VERSION:
        raise ValueError(f"observation {observation_id} inspection telemetry is invalid")
    if normalized["replay_count"] != replay_count:
        raise ValueError(
            f"observation {observation_id} top-level and inspection replay counts differ"
        )
    for event in normalized["camera_change_events"]:
        _validate_camera_state(event.get("from"), dict(viewing_settings))
        _validate_camera_state(event.get("to"), dict(viewing_settings))


def _adaptive_schedule_reason(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(_ADAPTIVE_REASON_PREFIX):
        return False
    reasons = value.removeprefix(_ADAPTIVE_REASON_PREFIX).split("+")
    return (
        len(reasons) == len(set(reasons))
        and reasons == sorted(reasons)
        and "underrepresented_family" in reasons
        and set(reasons).issubset(_ADAPTIVE_REASONS)
    )


def validate_current_protocol_observations(
    observations: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    served_playlist: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed unless every response is an exact authenticated current trial."""
    stimulus_meta = {
        str(row["stimulus_id"]): row
        for row in manifest.get("stimuli", [])
        if isinstance(row, Mapping) and row.get("stimulus_id") is not None
    }
    trial_index, base_counts, adaptive_counts = _manifest_trial_index(manifest)
    served_by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for served in served_playlist:
        key = served_observation_key(served, observation=False)
        served_by_key.setdefault(key, []).append(served)
    seen_observation_ids: set[str] = set()
    seen_trial_responses: set[tuple[str, str, str]] = set()
    seen_trial_positions: set[tuple[str, str, int]] = set()
    viewing_settings = manifest.get("viewing_settings")
    if not isinstance(viewing_settings, Mapping):
        raise ValueError("current pilot has no authoritative viewing settings")

    for row_index, row in enumerate(observations):
        observation_id = row.get("observation_id")
        if not isinstance(observation_id, str) or not _OBSERVATION_ID_PATTERN.fullmatch(
            observation_id
        ):
            raise ValueError(f"observation row {row_index} has an invalid observation_id")
        if observation_id in seen_observation_ids:
            raise ValueError(f"duplicate subjective observation_id: {observation_id}")
        seen_observation_ids.add(observation_id)
        if row.get("format_version") != RAW_OBSERVATION_VERSION:
            raise ValueError("current-protocol observation uses an unsupported schema")
        for field in ("rater_id", "session_id", "trial_id"):
            if not isinstance(row.get(field), str) or not row.get(field):
                raise ValueError(f"observation {observation_id} has no {field}")
        session_index = row.get("session_index")
        trial_position = row.get("trial_index")
        if (
            not isinstance(session_index, int)
            or isinstance(session_index, bool)
            or not isinstance(trial_position, int)
            or isinstance(trial_position, bool)
            or session_index < 0
            or trial_position < 0
        ):
            raise ValueError(f"observation {observation_id} has invalid trial coordinates")
        rater_id = str(row["rater_id"])
        session_id = str(row["session_id"])
        trial_id = str(row["trial_id"])
        expected_session_id = _canonical_id(
            "session",
            {
                "rater": rater_id,
                "index": session_index,
                "pilot_seed": manifest.get("seed"),
            },
        )
        if session_id != expected_session_id:
            raise ValueError(f"observation {observation_id} session identity is invalid")
        response_key = (rater_id, session_id, trial_id)
        position_key = (rater_id, session_id, trial_position)
        if response_key in seen_trial_responses:
            raise ValueError(f"duplicate response for served trial: {trial_id}")
        if position_key in seen_trial_positions:
            raise ValueError(
                f"duplicate response at served trial index: {session_id}/{trial_position}"
            )
        seen_trial_responses.add(response_key)
        seen_trial_positions.add(position_key)

        trial_record = trial_index.get((session_index, trial_id))
        if trial_record is None:
            raise ValueError(f"observation {observation_id} references an unknown pilot trial")
        trial, adaptive = trial_record
        served_matches = served_by_key.get(served_observation_key(row, observation=True), [])
        if not served_matches:
            raise ValueError(
                f"observation {observation_id} is not linked to a served trial "
                "with exact presentation metadata"
            )
        served_timestamp = min(
            _timestamp(value.get("timestamp_utc"), label="served event") for value in served_matches
        )
        observation_timestamp = _timestamp(
            row.get("timestamp_utc"), label=f"observation {observation_id}"
        )
        if observation_timestamp < served_timestamp:
            raise ValueError(f"observation {observation_id} predates its served event")

        stimulus_ids_raw = row.get("stimulus_ids")
        if not isinstance(stimulus_ids_raw, list):
            raise ValueError(f"observation {observation_id} has invalid stimulus IDs")
        stimulus_ids = [str(value) for value in stimulus_ids_raw]
        if stimulus_ids_raw != stimulus_ids:
            raise ValueError(f"observation {observation_id} has non-string stimulus IDs")
        unknown = [value for value in stimulus_ids if value not in stimulus_meta]
        if unknown:
            raise ValueError(f"observation references unknown pilot stimuli: {unknown}")

        expected_fields = {
            "task_type": trial.get("task_type"),
            "stimulus_ids": list(trial.get("stimulus_ids", [])),
            "source_ids": list(trial.get("source_ids", [])),
            "variant_ids": [stimulus_meta[value].get("variant_id") for value in stimulus_ids],
            "families": list(trial.get("families", [])),
            "styles": list(trial.get("styles", [])),
            "render_protocol_hash": manifest.get("render_protocol_hash"),
            "render_hashes": list(trial.get("render_hashes", [])),
            "presentation_order": list(trial.get("presentation_order", [])),
            "comparison_mode": trial.get("comparison_mode"),
            "pair_id": trial.get("pair_id"),
            "hidden_repeat_group_ids": list(trial.get("hidden_repeat_group_ids", [])),
            "anchor_status": (
                "hidden_session_anchor" if trial.get("hidden_anchor") is True else "not_anchor"
            ),
            "viewing_settings": dict(viewing_settings),
            "view_mirror_x": bool(trial.get("view_mirror_x", False)),
            "presented_side_classes": list(trial.get("side_classes", [])),
            "stimulus_populations": list(trial.get("stimulus_populations", [])),
            "measurement_cohorts": list(trial.get("measurement_cohorts", [])),
            "critic_training_eligible": bool(trial.get("critic_training_eligible", False)),
            "equivalence_control": bool(trial.get("equivalence_control", False)),
            "playlist_seed": manifest.get("seed"),
        }
        for field, expected in expected_fields.items():
            _expect_equal(row, field, expected, observation_id=observation_id)
        expected_populations = [
            stimulus_meta[value].get("stimulus_population") for value in stimulus_ids
        ]
        expected_cohorts = [
            stimulus_meta[value].get("measurement_cohort") for value in stimulus_ids
        ]
        if expected_fields["stimulus_populations"] != expected_populations:
            raise ValueError(f"pilot trial {trial_id} has inconsistent stimulus populations")
        if expected_fields["measurement_cohorts"] != expected_cohorts:
            raise ValueError(f"pilot trial {trial_id} has inconsistent measurement cohorts")
        expected_eligible = bool(expected_populations) and all(
            population == "HARD_FEASIBLE_PERCEPTUAL" for population in expected_populations
        )
        if expected_fields["critic_training_eligible"] is not expected_eligible:
            raise ValueError(f"pilot trial {trial_id} has inconsistent training eligibility")

        if adaptive:
            lower = base_counts.get(session_index, 0)
            upper = lower + adaptive_counts.get(session_index, 0)
            if not lower <= trial_position < upper:
                raise ValueError(f"observation {observation_id} has invalid adaptive trial index")
            if not _adaptive_schedule_reason(row.get("schedule_reason")):
                raise ValueError(f"observation {observation_id} has invalid adaptive reason")
        else:
            _expect_equal(
                row,
                "trial_index",
                trial.get("trial_index"),
                observation_id=observation_id,
            )
            _expect_equal(
                row,
                "schedule_reason",
                trial.get("schedule_reason"),
                observation_id=observation_id,
            )

        _validate_response_semantics(
            row,
            observation_id=observation_id,
            stimulus_ids=stimulus_ids,
        )
        _validate_response_metadata(
            row,
            observation_id=observation_id,
            stimulus_count=len(stimulus_ids),
            viewing_settings=viewing_settings,
        )
