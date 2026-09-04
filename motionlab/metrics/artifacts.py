"""Compact JSON and dense pickle-free NPZ persistence for metric reports."""

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

from motionlab.metrics.base import DeterministicReport, MetricResult

DENSE_METRIC_FORMAT_VERSION = "motionlab.metric_arrays.v1"
_DENSE_FIELDS = ("frame_values", "joint_frame_values")


@dataclass(frozen=True)
class DenseMetricArtifact:
    """Validated dense diagnostic arrays linked to one motion ID."""

    motion_id: str
    arrays: Mapping[str, NDArray[np.float64]]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen: dict[str, NDArray[np.float64]] = {}
        for name, raw in self.arrays.items():
            value = np.array(raw, dtype=np.float64, copy=True, order="C")
            value.setflags(write=False)
            frozen[name] = value
        object.__setattr__(self, "arrays", MappingProxyType(frozen))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _split_dense_values(
    report: DeterministicReport,
) -> tuple[dict[str, MetricResult], dict[str, NDArray[np.float64]], dict[str, Any]]:
    compact_metrics: dict[str, MetricResult] = {}
    arrays: dict[str, NDArray[np.float64]] = {}
    manifest_metrics: dict[str, dict[str, Any]] = {}
    for metric_index, (metric_name, metric) in enumerate(report.metrics.items()):
        dense_fields: dict[str, str] = {}
        for field_name in _DENSE_FIELDS:
            raw = getattr(metric, field_name)
            if raw is None:
                continue
            key = f"metric_{metric_index:03d}_{field_name}"
            value = np.asarray(raw, dtype=np.float64)
            arrays[key] = value
            dense_fields[field_name] = key
        metadata = dict(metric.metadata)
        if dense_fields:
            metadata["dense_fields"] = dense_fields
        compact_metrics[metric_name] = metric.model_copy(
            update={
                "frame_values": None,
                "joint_frame_values": None,
                "metadata": metadata,
            }
        )
        manifest_metrics[metric_name] = {
            field_name: {
                "key": key,
                "shape": list(arrays[key].shape),
                "dtype": str(arrays[key].dtype),
            }
            for field_name, key in dense_fields.items()
        }
    manifest = {
        "format_version": DENSE_METRIC_FORMAT_VERSION,
        "motion_id": report.motion_id,
        "metrics": manifest_metrics,
    }
    return compact_metrics, arrays, manifest


def _save_dense_npz(
    path: Path,
    *,
    motion_id: str,
    arrays: Mapping[str, NDArray[np.float64]],
    manifest: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "format_version": np.asarray(DENSE_METRIC_FORMAT_VERSION),
                "motion_id": np.asarray(motion_id),
                "manifest_json": np.asarray(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
                ),
            }
            payload.update(arrays)
            np.savez_compressed(stream, **payload)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_deterministic_report(
    json_path: Path,
    report: DeterministicReport,
    *,
    dense_path: Path | None = None,
) -> tuple[Path, Path]:
    """Atomically save compact JSON and linked dense NPZ artifacts."""
    summary_path = Path(json_path)
    artifact_path = (
        summary_path.with_name(f"{summary_path.stem}.dense.npz")
        if dense_path is None
        else Path(dense_path)
    )
    if summary_path.resolve() == artifact_path.resolve():
        raise ValueError("summary and dense artifact paths must be different")
    compact_metrics, arrays, manifest = _split_dense_values(report)
    _save_dense_npz(
        artifact_path,
        motion_id=report.motion_id,
        arrays=arrays,
        manifest=manifest,
    )
    relative_artifact = os.path.relpath(artifact_path, start=summary_path.parent)
    metadata = {
        **report.metadata,
        "dense_artifact": {
            "format_version": DENSE_METRIC_FORMAT_VERSION,
            "path": relative_artifact,
            "sha256": _sha256(artifact_path),
            "array_count": len(arrays),
        },
    }
    compact = report.model_copy(update={"metrics": compact_metrics, "metadata": metadata})
    payload = json.dumps(
        compact.model_dump(mode="json", exclude_none=True),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=summary_path.parent,
            prefix=f".{summary_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.write("\n")
        os.replace(temporary_path, summary_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return summary_path, artifact_path


def load_dense_metric_artifact(
    path: Path,
    *,
    expected_motion_id: str | None = None,
    expected_sha256: str | None = None,
) -> DenseMetricArtifact:
    """Load and validate a dense metric artifact without enabling pickle."""
    source = Path(path)
    if expected_sha256 is not None and _sha256(source) != expected_sha256:
        raise ValueError("dense metric artifact checksum does not match the summary")
    try:
        context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read dense metric artifact: {source}") from exc
    with context as archive:
        required = {"format_version", "motion_id", "manifest_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"dense metric artifact is missing fields: {sorted(missing)}")
        version = str(archive["format_version"].item())
        if version != DENSE_METRIC_FORMAT_VERSION:
            raise ValueError(f"unsupported dense metric artifact version {version!r}")
        motion_id = str(archive["motion_id"].item())
        try:
            manifest = json.loads(str(archive["manifest_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("dense metric artifact has an invalid manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("motion_id") != motion_id:
            raise ValueError("dense metric manifest motion ID does not match the artifact")
        if expected_motion_id is not None and motion_id != expected_motion_id:
            raise ValueError("dense metric artifact motion ID does not match the expected motion")
        if manifest.get("format_version") != DENSE_METRIC_FORMAT_VERSION:
            raise ValueError("dense metric manifest format version does not match the artifact")
        manifest_metrics = manifest.get("metrics")
        if not isinstance(manifest_metrics, dict):
            raise ValueError("dense metric manifest metrics must be an object")
        declarations: list[tuple[str, list[int], str]] = []
        for fields in manifest_metrics.values():
            if not isinstance(fields, dict):
                raise ValueError("dense metric manifest fields must be objects")
            for declaration in fields.values():
                if not isinstance(declaration, dict):
                    raise ValueError("dense metric array declaration must be an object")
                key = declaration.get("key")
                shape = declaration.get("shape")
                dtype = declaration.get("dtype")
                if (
                    not isinstance(key, str)
                    or not isinstance(shape, list)
                    or not all(isinstance(size, int) and size >= 0 for size in shape)
                    or not isinstance(dtype, str)
                ):
                    raise ValueError("dense metric array declaration is invalid")
                declarations.append((key, shape, dtype))
        declared_keys = {key for key, _, _ in declarations}
        if len(declared_keys) != len(declarations):
            raise ValueError("dense metric manifest contains duplicate array keys")
        missing_arrays = declared_keys.difference(archive.files)
        if missing_arrays:
            raise ValueError(f"dense metric artifact is missing arrays: {sorted(missing_arrays)}")
        unexpected_arrays = set(archive.files).difference(required).difference(declared_keys)
        if unexpected_arrays:
            raise ValueError(
                f"dense metric artifact contains undeclared arrays: {sorted(unexpected_arrays)}"
            )
        arrays: dict[str, NDArray[np.float64]] = {}
        for key, shape, dtype in declarations:
            raw = archive[key]
            if list(raw.shape) != shape or str(raw.dtype) != dtype:
                raise ValueError(f"dense metric array {key!r} does not match its declaration")
            arrays[key] = np.asarray(raw, dtype=np.float64)
    return DenseMetricArtifact(motion_id=motion_id, arrays=arrays, manifest=manifest)
