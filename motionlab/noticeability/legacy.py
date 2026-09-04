"""Lossless legacy snapshots and blinded overlap selection; no invented notice labels."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from motionlab.dataset.io import file_sha256
from motionlab.noticeability.protocol import identity, read_json, read_rows, write_json
from motionlab.perceptual.observation_auth import (
    resolve_frozen_evidence_paths,
    validate_current_protocol_observations,
)


def freeze_legacy_protocols(pilots: list[Path], output: Path) -> dict[str, Any]:
    """Snapshot exact v3/v4 raw bytes plus derived compatibility records outside old pilots."""
    if output.exists():
        existing = read_registry(output / "legacy_registry.json")
        if [str(p.resolve()) for p in pilots] != [
            s["original_manifest"] for s in existing["sources"]
        ]:
            raise ValueError("existing legacy snapshot belongs to different pilots")
        return existing
    snapshots = []
    records = []
    seen: set[str] = set()
    for index, supplied in enumerate(pilots):
        pilot = supplied.resolve()
        if output.resolve().is_relative_to(pilot.parent):
            raise ValueError("legacy snapshot must remain outside original pilot")
        manifest = read_json(pilot)
        freeze = read_json(pilot.with_name("protocol_freeze.json"))
        observation_path = (
            pilot.parent / freeze["append_only_evidence_prefixes"]["observations"]["path"]
        )
        served_path, _ = resolve_frozen_evidence_paths(
            pilot, observation_path, pilot.with_name("protocol_freeze.json")
        )
        raw = observation_path.read_bytes() if observation_path.exists() else b""
        rows = read_rows(observation_path)
        served = read_rows(served_path)
        validate_current_protocol_observations(rows, manifest, served)
        # A paused collector is required: capture consistency, not a moving live suffix.
        if not pilot.with_name("PILOT_PAUSED.json").exists():
            raise ValueError("pause the old collector before freezing its final evidence")
        if raw != (observation_path.read_bytes() if observation_path.exists() else b""):
            raise ValueError("legacy evidence changed during snapshot")
        snapshot_dir = output / f"source_{index}"
        snapshot_dir.mkdir(parents=True)
        byte_files = {
            "pilot_manifest.json": pilot.read_bytes(),
            "raw_observations.jsonl": raw,
            "served_playlist.jsonl": served_path.read_bytes() if served_path.exists() else b"",
            "protocol_freeze.json": pilot.with_name("protocol_freeze.json").read_bytes(),
        }
        for name, data in byte_files.items():
            with (snapshot_dir / name).open("xb") as stream:
                stream.write(data)
        snapshots.append(
            {
                "original_manifest": str(pilot),
                "original_observations": str(observation_path),
                "snapshot_directory": str(snapshot_dir.resolve()),
                "protocol_version": manifest["protocol_version"],
                "protocol_alias": "perceptual_quality_v1",
                "protocol_id": freeze["protocol_id"],
                "raw_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "files": {name: file_sha256(snapshot_dir / name) for name in byte_files},
            }
        )
        stimuli = {r["stimulus_id"]: r for r in manifest["stimuli"]}
        for row in rows:
            oid = row["observation_id"]
            if oid in seen:
                raise ValueError("legacy observations duplicated across input logs")
            seen.add(oid)
            record = {
                "original_observation_id": oid,
                "original_row_hash": identity("legacy-row", row),
                "source_index": index,
                "derived": True,
                "provenance": "LEGACY_HUMAN_QUALITY",
                "raw_observation": row,
                "original_manifest": str(pilot),
                "stimuli": [stimuli[s] for s in row["stimulus_ids"]],
                "original_stimulus_directory": manifest["stimulus_directory"],
                "viewing_settings": manifest["viewing_settings"],
                "question_and_scale": manifest["scale_anchors"],
            }
            if row["task_type"] == "pair_comparison":
                record["legacy_pairwise_quality"] = row["pair_outcome_canonical"]
            elif row["task_type"] == "naturalness":
                record["legacy_quality_ordinal"] = row["rating"]
            else:
                record["legacy_style_ordinal"] = row["rating"]
            records.append(record)
    registry = {
        "format_version": "motionlab.legacy_quality_registry.v1",
        "sources": snapshots,
        "records": records,
        "raw_observation_count": len(records),
        "old_score_distribution": dict(
            sorted(
                Counter(
                    str(r["legacy_quality_ordinal"])
                    for r in records
                    if "legacy_quality_ordinal" in r
                ).items()
            )
        ),
        "pairwise_observation_count": sum("legacy_pairwise_quality" in r for r in records),
        "automatic_noticeability_conversion": False,
    }
    registry["registry_id"] = identity("legacy-registry", registry)
    write_json(output / "legacy_registry.json", registry)
    return registry


def read_registry(path: Path) -> dict[str, Any]:
    registry = read_json(path)
    if registry["registry_id"] != identity(
        "legacy-registry", {k: v for k, v in registry.items() if k != "registry_id"}
    ):
        raise ValueError("legacy registry hash mismatch")
    for source in registry["sources"]:
        root = Path(source["snapshot_directory"])
        for name, digest in source["files"].items():
            if file_sha256(root / name) != digest:
                raise ValueError("legacy snapshot has changed")
        original = Path(source["original_observations"])
        if file_sha256(original) != source["raw_sha256"]:
            raise ValueError("original legacy observations changed after freeze")
    for record in registry["records"]:
        if identity("legacy-row", record["raw_observation"]) != record["original_row_hash"]:
            raise ValueError("legacy compatibility record changed")
    return registry


def select_overlap(
    registry: dict[str, Any], *, count: int = 40, seed: int = 9601
) -> dict[str, Any]:
    """Select only already-rated single motions; never use pair outcomes as detection labels."""
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in registry["records"]:
        if "legacy_quality_ordinal" not in record:
            continue
        row = record["raw_observation"]
        key = (row["rater_id"], row["render_hashes"][0])
        # One old observation is the calibration reference. Preserve repeats separately in registry.
        unique.setdefault(key, record)
    remaining = list(unique.values())
    random.Random(seed).shuffle(remaining)
    counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []

    def strata(record: dict[str, Any]) -> list[tuple[str, str]]:
        row, item = record["raw_observation"], record["stimuli"][0]
        return [
            ("score", str(record["legacy_quality_ordinal"])),
            ("source", item["source_id"]),
            ("style", item["style"]),
            ("family", item["family"]),
            ("confidence", str(row.get("confidence"))),
            ("difficulty", str(row.get("judgment_difficulty"))),
            (
                "critic_disagreement",
                str(
                    any(
                        "misrank" in s or "disagreement" in s
                        for s in item.get("selection_reasons", [])
                    )
                ),
            ),
            (
                "clean_sham",
                str(
                    item.get("variant_kind") == "clean_reference"
                    or row.get("equivalence_control", False)
                ),
            ),
            ("render", row["render_protocol_hash"]),
        ]

    while remaining and len(selected) < count:
        chosen = max(remaining, key=lambda r: sum(1 / (1 + counts[s]) for s in strata(r)))
        selected.append(chosen)
        counts.update(strata(chosen))
        remaining.remove(chosen)
    return {
        "format_version": "motionlab.noticeability_overlap_selection.v1",
        "registry_id": registry["registry_id"],
        "seed": seed,
        "requested_count": count,
        "available_unique_rated_stimuli": len(unique),
        "selected_count": len(selected),
        "selected_original_observation_ids": [r["original_observation_id"] for r in selected],
        "status": "ready" if len(selected) >= 30 else "limited_by_existing_single_stimulus_labels",
        "strata": {f"{k}:{v}": n for (k, v), n in sorted(counts.items())},
    }
