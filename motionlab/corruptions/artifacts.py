"""Pickle-free corruption artifacts with split and privilege safeguards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motionlab.core.provenance import provenance_from_clip
from motionlab.corruptions.base import (
    CORRUPTION_CHANNEL_NAMES,
    CORRUPTION_DEFECT_NAMES,
    CORRUPTION_PART_NAMES,
    CorruptionResult,
)
from motionlab.corruptions.catalog import CORRUPTION_CATALOG_VERSION, catalog_entry
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.motion.clip import MotionClip

CORRUPTION_ARTIFACT_VERSION = "motionlab.corruption_artifact.v1"
_LABELS_FILENAME = "labels.npz"
_MANIFEST_FILENAME = "manifest.json"
_CLEAN_FILENAME = "clean_motion.npz"
_CORRUPTED_FILENAME = "corrupted_motion.npz"


@dataclass(frozen=True)
class CorruptionArtifact:
    """Loaded corruption motions, dense arrays, and compact manifest."""

    clean_motion: MotionClip
    corrupted_motion: MotionClip
    arrays: Mapping[str, NDArray[Any]]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        arrays: dict[str, NDArray[Any]] = {}
        for name, raw in self.arrays.items():
            value = np.array(raw, copy=True, order="C")
            value.setflags(write=False)
            arrays[name] = value
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _dense_arrays(result: CorruptionResult) -> dict[str, NDArray[Any]]:
    arrays: dict[str, NDArray[Any]] = {
        "intervention_mask": result.intervention_mask,
        "symptom_mask": result.symptom_mask,
        "responsibility_target": result.responsibility_target,
        "responsibility_confidence": result.responsibility_confidence,
    }
    contacts = result.contact_supervision
    if contacts is not None:
        arrays.update(
            {
                "contact_truth_hard": contacts.truth_hard,
                "contact_truth_confidence": contacts.truth_confidence,
                "contact_inferred_clean_hard": contacts.inferred_clean_hard,
                "contact_inferred_clean_confidence": contacts.inferred_clean_confidence,
                "contact_inferred_corrupted_hard": contacts.inferred_corrupted_hard,
                "contact_inferred_corrupted_confidence": contacts.inferred_corrupted_confidence,
                "ground_plane": contacts.ground_plane,
            }
        )
    return arrays


def _save_labels_npz(
    path: Path,
    *,
    result: CorruptionResult,
    arrays: Mapping[str, NDArray[Any]],
    declaration: Mapping[str, Any],
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            payload: dict[str, Any] = {
                "format_version": np.asarray(CORRUPTION_ARTIFACT_VERSION),
                "source_motion_id": np.asarray(result.source_motion_id),
                "corrupted_motion_id": np.asarray(result.corrupted_motion.content_hash),
                "split_lineage_id": np.asarray(result.split_lineage_id or ""),
                "declaration_json": np.asarray(
                    json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False)
                ),
            }
            payload.update(arrays)
            np.savez_compressed(stream, **payload)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_corruption_artifact(
    directory: Path,
    result: CorruptionResult,
    *,
    require_complete_lineage: bool = True,
    require_hard_negative: bool = True,
) -> Path:
    """Write a compact manifest, two motions, and dense labels into one directory."""
    output = Path(directory)
    clean_provenance = provenance_from_clip(result.clean_motion)
    corrupted_provenance = provenance_from_clip(result.corrupted_motion)
    if require_complete_lineage and (
        not clean_provenance.lineage_complete
        or clean_provenance.split_lineage_id is None
        or not corrupted_provenance.lineage_complete
    ):
        raise ValueError("training artifacts require complete source and split lineage")
    if result.split_lineage_id != clean_provenance.split_lineage_id:
        raise ValueError("corruption result split lineage does not match its clean source")
    if corrupted_provenance.split_lineage_id != result.split_lineage_id:
        raise ValueError("corrupted motion did not preserve the source split lineage")
    if require_hard_negative and not result.postcondition.hard_negative:
        raise ValueError("training artifacts require a measured, validated hard negative")
    if result.corruption_family == "composite":
        if not result.generation_parameters.get("constituents"):
            raise ValueError("composite corruption is missing its constituent declaration")
    else:
        entry = catalog_entry(result.corruption_family, result.corruption_mechanism)
        if entry.partition != result.catalog_partition:
            raise ValueError("corruption result partition disagrees with the catalog")

    output.mkdir(parents=True, exist_ok=True)
    clean_path = output / _CLEAN_FILENAME
    corrupted_path = output / _CORRUPTED_FILENAME
    labels_path = output / _LABELS_FILENAME
    save_motion_npz(clean_path, result.clean_motion)
    save_motion_npz(corrupted_path, result.corrupted_motion)
    arrays = _dense_arrays(result)
    declaration = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    _save_labels_npz(
        labels_path,
        result=result,
        arrays=arrays,
        declaration=declaration,
    )
    contacts = result.contact_supervision
    manifest: dict[str, Any] = {
        "format_version": CORRUPTION_ARTIFACT_VERSION,
        "catalog_version": CORRUPTION_CATALOG_VERSION,
        "source_motion_id": result.source_motion_id,
        "corrupted_motion_id": result.corrupted_motion.content_hash,
        "corruption_family": result.corruption_family,
        "corruption_mechanism": result.corruption_mechanism,
        "catalog_partition": result.catalog_partition,
        "corruption_seed": result.corruption_seed,
        "split_lineage_id": result.split_lineage_id,
        "requested_severity_parameter": result.requested_severity_parameter,
        "generation_parameters": dict(result.generation_parameters),
        "measured_metrics_before": dict(result.measured_metrics_before),
        "measured_metrics_after": dict(result.measured_metrics_after),
        "measured_severity": result.measured_severity,
        "preference_confidence": result.preference_confidence,
        "postcondition": {
            "metric_name": result.postcondition.metric_name,
            "before": result.postcondition.before,
            "after": result.postcondition.after,
            "delta": result.postcondition.delta,
            "minimum_worsening": result.postcondition.minimum_worsening,
            "passed": result.postcondition.passed,
            "hard_negative": result.postcondition.hard_negative,
        },
        "source": {
            "dataset": clean_provenance.source_dataset,
            "clip_id": clean_provenance.source_clip_id,
            "take_id": clean_provenance.source_take_id,
            "split": clean_provenance.split,
            "clean_confidence": clean_provenance.clean_confidence,
        },
        "vocabulary": {
            "intervention_channels": list(CORRUPTION_CHANNEL_NAMES),
            "functional_parts": list(CORRUPTION_PART_NAMES),
            "defects": list(CORRUPTION_DEFECT_NAMES),
        },
        "dense_arrays": declaration,
        "contact_supervision": (
            None
            if contacts is None
            else {
                "marker_names": list(contacts.marker_names),
                "truth_source": contacts.truth_source,
                "inference_source": contacts.inference_source,
                "ground_confidence": contacts.ground_confidence,
            }
        ),
        "schema_metadata": dict(result.schema_metadata),
        "privileged_dense_fields": ["contact_truth_hard", "contact_truth_confidence"],
        "neural_input_exclusions": [
            "source",
            "source_motion_id",
            "corruption_family",
            "corruption_mechanism",
            "corruption_seed",
            "requested_severity_parameter",
            "generation_parameters",
            "split_lineage_id",
            "contact_truth_hard",
            "contact_truth_confidence",
        ],
        "files": {
            "clean_motion": {"path": _CLEAN_FILENAME, "sha256": _sha256(clean_path)},
            "corrupted_motion": {
                "path": _CORRUPTED_FILENAME,
                "sha256": _sha256(corrupted_path),
            },
            "dense_labels": {"path": _LABELS_FILENAME, "sha256": _sha256(labels_path)},
        },
    }
    manifest_path = output / _MANIFEST_FILENAME
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _artifact_path(directory: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact file path must be a nonempty string")
    root = directory.resolve()
    candidate = (directory / relative_path).resolve()
    if candidate.parent != root:
        raise ValueError("artifact file path must name a direct child of its directory")
    return candidate


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read corruption artifact manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("corruption artifact manifest must be a JSON object")
    if value.get("format_version") != CORRUPTION_ARTIFACT_VERSION:
        raise ValueError("unsupported corruption artifact version")
    return value


def load_corruption_artifact(directory: Path) -> CorruptionArtifact:
    """Load and verify a corruption artifact without enabling pickle."""
    root = Path(directory)
    manifest = _load_manifest(root / _MANIFEST_FILENAME)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("corruption artifact files declaration must be an object")
    resolved: dict[str, Path] = {}
    for key in ("clean_motion", "corrupted_motion", "dense_labels"):
        record = files.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"corruption artifact is missing file declaration {key!r}")
        path = _artifact_path(root, record.get("path"))
        checksum = record.get("sha256")
        if not isinstance(checksum, str) or _sha256(path) != checksum:
            raise ValueError(f"corruption artifact checksum mismatch for {key!r}")
        resolved[key] = path
    clean = load_motion_npz(resolved["clean_motion"])
    corrupted = load_motion_npz(resolved["corrupted_motion"])
    if clean.content_hash != manifest.get("source_motion_id"):
        raise ValueError("clean motion ID does not match the corruption manifest")
    if corrupted.content_hash != manifest.get("corrupted_motion_id"):
        raise ValueError("corrupted motion ID does not match the corruption manifest")

    declaration = manifest.get("dense_arrays")
    if not isinstance(declaration, dict):
        raise ValueError("dense array declaration must be an object")
    try:
        context = np.load(resolved["dense_labels"], allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("unable to read dense corruption labels") from exc
    arrays: dict[str, NDArray[Any]] = {}
    with context as archive:
        required = {
            "format_version",
            "source_motion_id",
            "corrupted_motion_id",
            "split_lineage_id",
            "declaration_json",
        }
        if required.difference(archive.files):
            raise ValueError("dense corruption labels are missing required fields")
        if str(archive["format_version"].item()) != CORRUPTION_ARTIFACT_VERSION:
            raise ValueError("dense corruption label version does not match")
        scalar_links = {
            "source_motion_id": manifest.get("source_motion_id"),
            "corrupted_motion_id": manifest.get("corrupted_motion_id"),
            "split_lineage_id": manifest.get("split_lineage_id") or "",
        }
        for key, expected in scalar_links.items():
            if str(archive[key].item()) != expected:
                raise ValueError(f"dense corruption label {key} does not match the manifest")
        try:
            embedded_declaration = json.loads(str(archive["declaration_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("dense corruption label declaration is invalid") from exc
        if embedded_declaration != declaration:
            raise ValueError("dense corruption label declaration does not match the manifest")
        declared_keys = set(declaration)
        unexpected = set(archive.files).difference(required).difference(declared_keys)
        if unexpected:
            raise ValueError(
                f"dense corruption labels contain undeclared arrays: {sorted(unexpected)}"
            )
        missing = declared_keys.difference(archive.files)
        if missing:
            raise ValueError(f"dense corruption labels are missing arrays: {sorted(missing)}")
        for key, raw_declaration in declaration.items():
            if not isinstance(raw_declaration, dict):
                raise ValueError("dense corruption array declarations must be objects")
            raw = archive[key]
            if list(raw.shape) != raw_declaration.get("shape") or str(
                raw.dtype
            ) != raw_declaration.get("dtype"):
                raise ValueError(f"dense corruption array {key!r} does not match its declaration")
            arrays[key] = np.array(raw, copy=True)
    return CorruptionArtifact(
        clean_motion=clean,
        corrupted_motion=corrupted,
        arrays=arrays,
        manifest=manifest,
    )
