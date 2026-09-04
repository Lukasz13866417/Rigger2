"""Review-gated exports and independent follow-up sampling plans."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from motionlab.noticeability import repeated_protocol
from motionlab.noticeability.analysis import analyze
from motionlab.noticeability.protocol import (
    authenticated_observations,
    identity,
    read_json,
    write_json,
)


def export_training(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Export direct clip-level targets; no proxy admission or train/test split by random row."""
    manifest = read_json(manifest_path)
    items = {i["stimulus_id"]: i for i in manifest["stimuli"]}
    repeated = manifest["protocol_version"] == repeated_protocol.PROTOCOL
    rows = (
        repeated_protocol.authenticated_observations(manifest_path)
        if repeated
        else authenticated_observations(manifest_path)
    )
    records = []
    excluded = []
    for row in rows:
        item = items[row["stimulus_id"]]
        if item["population"] == "CALIBRATION_ONLY":
            excluded.append(row["observation_id"])
            continue
        records.append(
            {
                **row,
                "motion": str((manifest_path.parent / item["motion"]).resolve()),
                "rendered_cycle_window": item["cycle_window"],
                "view_mirror_x": item["view_mirror_x"],
                "input_window_policy": "Use only displayed cycle-window frames, not unseen frames.",
                "population": item["population"],
                "physical_measurements": item["physical_measurements"],
                "family": item["family"],
                "source_group": item["source_id"],
                "context_key": identity(
                    "notice-context", [row["evaluation_goal"], row["style_context"]]
                ),
                "target": "repeated_view_notice" if repeated else "spontaneous_notice",
                "target_level": "clip_event",
                "label_strength": "DIRECT_HUMAN",
                "loss_weight": 1.0,
                "inspected_target_is_separate": True,
            }
        )
    result = {
        "format_version": "motionlab.repeated_view_training_export.v1"
        if repeated
        else "motionlab.noticeability_training_export.v1",
        "protocol_version": manifest["protocol_version"],
        "records": records,
        "excluded_calibration_only_observation_ids": excluded,
        "weak_proxies_included": False,
        "source_manifest": str(manifest_path.resolve()),
        "split_policy": (
            "Group by original source across all renders, severities, repeats and migration "
            "versions. Hold out mechanisms separately. Never split observations of one source "
            "across train/validation/test."
        ),
        "review_before_training": True,
        "critic_retraining_permitted": False,
        "production_objective_enabled": False,
    }
    result["export_hash"] = identity("notice-export", result)
    write_json(output, result)
    return result


def fixed_validation_plan(manifest_path: Path, output: Path, *, seed: int = 9611) -> dict[str, Any]:
    """Independent fixed design only after repeated human data identifies a crossing."""
    report = analyze(manifest_path)
    plans: list[dict[str, Any]] = []
    for track, fit in report["psychometric"].items():
        if fit["notice_threshold"] is None:
            continue
        threshold, jnd = fit["notice_threshold"], fit["jnd"]
        lower, upper = fit["levels"][0]["physical_strength"], fit["levels"][-1]["physical_strength"]
        levels = sorted(
            {max(lower, min(upper, threshold + step * jnd)) for step in (-1.5, -0.75, 0, 0.75, 1.5)}
        )
        plans.extend(
            {
                "track": track,
                "physical_strength": strength,
                "replicate": repeat,
                "perceptual_label": None,
                "population": "NOTICEABILITY_THRESHOLD",
            }
            for strength in levels
            for repeat in range(3)
        )
    if not plans:
        raise ValueError(
            "no supported human threshold yet; complete/repeat the exploratory screen first"
        )
    random.Random(seed).shuffle(plans)
    result = {
        "format_version": "motionlab.noticeability_fixed_validation_plan.v1",
        "seed": seed,
        "trials": plans,
        "role": "independent_fixed_validation_not_training",
        "requires_new_rendered_batch": True,
        "additional_controls": "Interleave at least 20% clean/sham and retain hidden repeats.",
        "source_analysis_hash": identity("notice-analysis", report),
    }
    write_json(output, result)
    return result
