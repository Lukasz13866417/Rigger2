"""Blinded, content-addressed overlap and broad-to-threshold collection pilots."""

from __future__ import annotations

import random
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.noticeability.legacy import read_registry, select_overlap
from motionlab.noticeability.protocol import PROTOCOL, identity, inside, validate_pilot, write_json
from motionlab.perceptual import subjective
from motionlab.perceptual.perturbations import PerturbationSpec, apply_perceptual_perturbation

# These are construction parameters, NOT asserted human perceptual labels.
LADDERS = {
    "upper_body_phase_mismatch": ([1.0, 3.0, 6.0, 10.0, 16.0, 24.0], "frames"),
    "excessive_rigidification": ([0.03, 0.12, 0.28, 0.48, 0.72, 0.95], "fraction"),
    "torso_counter_rotation_mismatch": ([0.005, 0.02, 0.05, 0.10, 0.20, 0.38], "radians"),
}
GOAL = "Notice unintended motion problems; do not penalize the explicitly intended walking style."


def context(style: str) -> str:
    return (
        "Intended walking style: "
        + style.removesuffix("_FW")
        + ". Preserve its intentional characteristics."
    )


def build_pilot(
    registry_path: Path,
    output: Path,
    *,
    overlap_count: int = 40,
    include_threshold: bool = True,
    source_count: int = 2,
    seed: int = 9601,
) -> dict[str, Any]:
    if not 0 <= overlap_count <= 60 or not 1 <= source_count <= 12:
        raise ValueError("use 0-60 overlap stimuli and 1-12 source clips")
    if output.exists():
        raise ValueError("use a fresh output directory; existing evidence is never overwritten")
    registry = read_registry(registry_path)
    selection = select_overlap(registry, count=overlap_count, seed=seed)
    records = {r["original_observation_id"]: r for r in registry["records"]}
    stimuli: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    write_json(output / "legacy_registry.json", registry)
    write_json(output / "overlap_selection.json", selection)

    def asset(path: Path) -> str:
        clip = load_motion_npz(path)
        name = "motions/" + clip.content_hash.split(":")[-1] + ".npz"
        target = output / name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        return name

    def add(
        motion: str,
        reference: str,
        *,
        source: str,
        style: str,
        family: str,
        population: str,
        settings: dict[str, Any],
        mirror: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = subjective._two_cycle_payload(
            output / motion,
            phase_reference_path=output / reference,
            viewing_settings=settings,
        )
        reference_clip = load_motion_npz(output / reference)
        source_lineage = str(reference_clip.metadata.get("source_clip_id", source))
        intended_style = (
            source_lineage if any(x in style for x in ("exploit", "red_team")) else style
        )
        presented = subjective._canonical_hash(
            "presented-render",
            {
                "base_render_hash": payload["render_hash"],
                "mirror_x": mirror,
            },
        )
        item = {
            "stimulus_id": identity("notice-stimulus", [presented, GOAL, context(intended_style)]),
            "render_hash": presented,
            "base_render_hash": payload["render_hash"],
            "render_protocol_hash": payload["render_protocol_hash"],
            "source_id": source_lineage,
            "legacy_source_id": source,
            "style": intended_style,
            "variant_id": identity("notice-variant", [motion, family]),
            "motion": motion,
            "phase_reference_motion": reference,
            "family": family,
            "population": population,
            "evaluation_goal": GOAL,
            "style_context": context(intended_style),
            "viewing_settings": settings,
            "view_mirror_x": mirror,
            "duration_s": payload["duration_s"],
            "fps": payload["fps"],
            "cycle_window": payload["cycle_window"],
            "physical_measurements": dict(
                grade_motion_deterministic(load_motion_npz(output / motion)).measured
            ),
            "human_noticeability": None,
            **(extra or {}),
        }
        if not any(s["stimulus_id"] == item["stimulus_id"] for s in stimuli):
            stimuli.append(item)
        return item

    latest = max(
        (
            datetime.fromisoformat(r["raw_observation"]["timestamp_utc"])
            for r in registry["records"]
        ),
        default=datetime(2000, 1, 1, tzinfo=UTC),
    )
    for oid in selection["selected_original_observation_ids"]:
        record = records[oid]
        old, row = record["stimuli"][0], record["raw_observation"]
        root = Path(record["original_stimulus_directory"])
        control = old["family"] in {"clean_reference", "equivalent_origin_shift"}
        population = (
            "CLEAN_SHAM_CONTROL"
            if control
            else (
                "CALIBRATION_ONLY"
                if not old.get("critic_training_eligible", True)
                else "NOTICEABILITY_THRESHOLD"
            )
        )
        if "exploit" in old["source_id"] or old["family"] == "preserved_critic_exploit":
            population = "ADVERSARIAL_CRITIC"
        item = add(
            asset(root / old["motion"]),
            asset(root / old["phase_reference_motion"]),
            source=old["source_id"],
            style=old["style"],
            family=old["family"],
            population=population,
            settings=record["viewing_settings"],
            mirror=row["view_mirror_x"],
            extra={"legacy_overlap": True},
        )
        if item["render_hash"] != row["render_hashes"][0]:
            raise ValueError("exact legacy overlap render could not be reproduced")
        trials.append(
            {
                "stimulus_id": item["stimulus_id"],
                "prior_exposure": True,
                "original_observation_id": oid,
                "eligible_rater_id": row["rater_id"],
                "not_before_utc": (latest + timedelta(hours=24)).isoformat(),
                "cohort": "legacy_overlap",
            }
        )

    if include_threshold:
        sources: dict[str, tuple[dict[str, Any], Path]] = {}
        for record in registry["records"]:
            for old in record["stimuli"]:
                sources.setdefault(
                    old["source_id"], (old, Path(record["original_stimulus_directory"]))
                )
        # Include sources with articulated arm swing; constrained-arm styles can make an
        # entire phase-offset ladder nearly invisible for reasons unrelated to the threshold.
        priorities = {"Neutral_FW": 0, "Proud_FW": 1, "Elated_FW": 2}
        ordinary_first = sorted(sources, key=lambda s: (priorities.get(s, 3), "exploit" in s, s))
        for source in ordinary_first[:source_count]:
            old, root = sources[source]
            reference = asset(root / old["phase_reference_motion"])
            clean = load_motion_npz(output / reference)
            settings = subjective.FREE_CAMERA_VIEWING_SETTINGS
            for kind, shift in [("clean_reference", 0.0), ("equivalent_origin_shift", 0.12)]:
                clip = (
                    clean
                    if shift == 0
                    else apply_perceptual_perturbation(
                        clean,
                        PerturbationSpec(
                            kind,
                            shift,
                            "train",
                            "UNORDERED",
                            None,
                            (),
                            None,
                        ),
                    )
                )
                name = "motions/" + clip.content_hash.split(":")[-1] + ".npz"
                if not (output / name).exists():
                    save_motion_npz(output / name, clip)
                item = add(
                    name,
                    reference,
                    source=source,
                    style=old["style"],
                    family=kind,
                    population="CLEAN_SHAM_CONTROL",
                    settings=settings,
                    extra={"control_kind": "clean" if shift == 0 else "sham"},
                )
                trials.append(
                    {
                        "stimulus_id": item["stimulus_id"],
                        "cohort": "fixed_screen",
                        "prior_exposure": True,
                    }
                )  # Source may have appeared in the old pilot.
            for family, (levels, unit) in LADDERS.items():
                for index, strength in enumerate(levels):
                    clip = apply_perceptual_perturbation(
                        clean,
                        PerturbationSpec(
                            family,
                            strength,
                            "train",
                            "UNORDERED",
                            None,
                            (),
                            None,
                        ),
                    )
                    name = "motions/" + clip.content_hash.split(":")[-1] + ".npz"
                    if not (output / name).exists():
                        save_motion_npz(output / name, clip)
                    item = add(
                        name,
                        reference,
                        source=source,
                        style=old["style"],
                        family=family,
                        population="NOTICEABILITY_THRESHOLD",
                        settings=settings,
                        extra={
                            "ladder_id": source + ":" + family,
                            "severity_index": index,
                            "physical_strength": strength,
                            "physical_strength_unit": unit,
                            "intended_range": [
                                "very_subtle",
                                "subtle",
                                "threshold_search",
                                "structured_uncanny",
                                "strong",
                                "intended_obvious",
                            ][index],
                            "perceptual_range_validated": False,
                        },
                    )
                    trials.append(
                        {
                            "stimulus_id": item["stimulus_id"],
                            "prior_exposure": False,
                            "cohort": "fixed_screen",
                        }
                    )
    rng = random.Random(seed)
    rng.shuffle(trials)
    # Four hidden repeats, widely separated; never average their raw observations.
    repeats = [
        dict(t, prior_exposure=True, cohort="hidden_repeat")
        for t in [t for t in trials if t.get("cohort") != "legacy_overlap"][:4]
    ]
    trials.extend(repeats)
    for index, trial in enumerate(trials):
        trial["trial_index"] = index
        trial["trial_id"] = identity("notice-trial", [seed, index, trial])
    manifest = {
        "format_version": "motionlab.noticeability_pilot.v1",
        "protocol_version": PROTOCOL,
        "style_policy": "explicit_style_and_goal_for_human_and_critic",
        "seed": seed,
        "stimuli": stimuli,
        "trials": trials,
        "legacy_registry": "legacy_registry.json",
        "pilot_role": "exploratory_overlap_and_fixed_threshold_screen_not_final_validation",
        "critic_retraining_permitted": False,
        "overlap_selection": selection,
        "minimum_overlap_delay_hours": 24,
    }
    write_json(output / "pilot_manifest.json", manifest)
    return manifest


def build_staircase_pilot(
    parent: Path,
    output: Path,
    *,
    steps: int = 30,
    seed: int = 9621,
) -> dict[str, Any]:
    """Freeze candidate assets for a new interleaved, response-adaptive session."""
    manifest = validate_pilot(parent)
    if output.exists() or not 10 <= steps <= 120:
        raise ValueError("fresh output directory and 10-120 staircase steps required")
    tracks: dict[str, list[dict[str, Any]]] = {}
    for item in manifest["stimuli"]:
        if item.get("ladder_id"):
            tracks.setdefault(item["ladder_id"], []).append(item)
    if len(tracks) < 2:
        raise ValueError("at least two independent ladder tracks required")
    ordered = {
        track: [i["stimulus_id"] for i in sorted(items, key=lambda x: x["physical_strength"])]
        for track, items in tracks.items()
    }
    controls = [i["stimulus_id"] for i in manifest["stimuli"] if i.get("control_kind")]
    if not controls:
        raise ValueError("interleaved staircases require clean/sham catch trials")
    trials = []
    for step in range(steps):
        candidates = (
            [("control", controls[(step // 5) % len(controls)])]
            if step % 5 == 4
            else [(track, sid) for track, ids in ordered.items() for sid in ids]
        )
        for track, sid in candidates:
            trials.append(
                {
                    "trial_id": identity("notice-staircase-trial", [seed, step, sid]),
                    "trial_index": step,
                    "stimulus_id": sid,
                    "track": track,
                    "prior_exposure": True,
                    "cohort": "adaptive_staircase",
                }
            )
    output.mkdir(parents=True)
    assets = {manifest["legacy_registry"]}
    for item in manifest["stimuli"]:
        assets.update((item["motion"], item["phase_reference_motion"]))
    for name in assets:
        target = inside(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(parent.parent, name), target)
    derived = {
        **manifest,
        "trials": trials,
        "seed": seed,
        "scheduler": {
            "mode": "interleaved_staircase",
            "steps": steps,
            "tracks": ordered,
            "controls_every": 5,
            "maybe_policy": "hold",
            "interrupted_policy": "advance_without_label",
        },
        "pilot_role": "adaptive_threshold_search_not_final_validation",
        "parent_manifest_hash": identity("notice-parent", manifest),
    }
    write_json(output / "pilot_manifest.json", derived)
    return derived
