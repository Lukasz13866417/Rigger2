"""Lossless cutover to bounded repeated-view labeling, never relabeling old evidence."""

from __future__ import annotations

import copy
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from motionlab.dataset.io import file_sha256
from motionlab.noticeability import protocol, repeated_protocol


def prepare_repeated_view_pilot(
    parent: Path,
    output: Path,
    *,
    inspection_presentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    version = protocol.read_json(parent)["protocol_version"]
    validator = repeated_protocol if version == repeated_protocol.PROTOCOL else protocol
    manifest: dict[str, Any] = validator.validate_pilot(parent)
    observations = validator.authenticated_observations(parent)
    if not parent.with_name("PILOT_PAUSED.json").is_file():
        raise ValueError("pause the old collector before continuing")
    if version not in {protocol.PROTOCOL, repeated_protocol.PROTOCOL} or manifest.get("scheduler"):
        raise ValueError("continuation requires a supported fixed noticeability pilot")
    if output.exists():
        raise ValueError("use a fresh continuation directory; never overwrite evidence")
    served = protocol.read_rows(parent.with_name("served_playlist.jsonl"))
    raw = protocol.read_rows(parent.with_name("observations.jsonl"))
    prior = (
        protocol.read_json(protocol.inside(parent.parent, manifest["continuation_history"]))
        if manifest.get("continuation_history")
        else {}
    )
    completed: dict[str, list[str]] = copy.deepcopy(prior.get("completed_trial_ids_by_rater", {}))
    for row in observations:
        ids = completed.setdefault(row["rater_id"], [])
        if row["trial_id"] not in ids:
            ids.append(row["trial_id"])
    exposed = {row["trial_id"] for row in served if row["event_type"] == "started"}
    history = {
        "format_version": "motionlab.repeated_view_continuation.v1",
        "original_manifest": str(parent.resolve()),
        "original_protocol_id": protocol.read_json(parent.with_name("protocol_freeze.json"))[
            "protocol_id"
        ],
        "original_evidence_hashes": {
            name: file_sha256(parent.with_name(name))
            for name in (
                "pilot_manifest.json",
                "protocol_freeze.json",
                "observations.jsonl",
                "served_playlist.jsonl",
            )
            if parent.with_name(name).exists()
        },
        "original_raw_observations": raw,
        "original_observation_count": len(observations),
        "completed_trial_ids_by_rater": completed,
        "exposed_trial_ids": sorted(exposed),
        "scheduling_only_not_new_labels": True,
        "inherited_continuation_history": prior,
    }
    derived = copy.deepcopy(manifest)
    derived.update(
        protocol_version=repeated_protocol.PROTOCOL,
        primary_viewing_policy=repeated_protocol.primary_viewing_policy(),
        continuation_history="continuation_history.json",
        pilot_role="bounded_repeated_view_noticeability_review_before_training",
    )
    if inspection_presentation is not None:
        derived["inspection_presentation"] = inspection_presentation
    for trial in derived["trials"]:
        if trial["trial_id"] in exposed:
            trial["prior_exposure"] = True
    output.mkdir(parents=True)
    assets = {i[key] for i in manifest["stimuli"] for key in ("motion", "phase_reference_motion")}
    if manifest.get("legacy_registry"):
        assets.add(manifest["legacy_registry"])
    for name in assets:
        target = protocol.inside(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(protocol.inside(parent.parent, name), target)
    protocol.write_json(output / "continuation_history.json", history)
    protocol.write_json(output / "pilot_manifest.json", derived)
    return derived


def analyze_repeated(manifest_path: Path, output: Path) -> dict[str, Any]:
    from motionlab.noticeability.analysis import rate_summary

    rows = repeated_protocol.authenticated_observations(manifest_path)
    inspected = [r for r in rows if r["inspected_notice"] is not None]
    report = {
        "format_version": "motionlab.repeated_view_analysis.v1",
        "protocol_version": repeated_protocol.PROTOCOL,
        "target": "repeated_view_notice",
        "not_interchangeable_with": "spontaneous_notice",
        "direct_observation_count": len(rows),
        "repeated_view": rate_summary([r["repeated_view_notice"] for r in rows]),
        "pre_answer_viewing_count_distribution": dict(
            Counter(len(r["pre_answer_viewings"]) for r in rows)
        ),
        "pre_answer_speed_usage": dict(
            Counter(v["playback_rate"] for r in rows for v in r["pre_answer_viewings"])
        ),
        "inspected": rate_summary([r["inspected_notice"] for r in inspected]),
        "repeated_view_inspected_disagreement": sum(
            r["repeated_view_notice"] != r["inspected_notice"] for r in inspected
        )
        / len(inspected)
        if inspected
        else None,
        "critic_retraining_permitted": False,
        "legacy_spontaneous_calibration_permitted": False,
    }
    protocol.write_json(output, report)
    return report
