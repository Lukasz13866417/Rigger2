"""Atomic pickle-free training sample and normalization persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motionlab.dataset.features import DATASET_SAMPLE_VERSION, TrainingSample

NORMALIZATION_VERSION = "motionlab.fixed_rig_normalization.v2"
NORMALIZED_FIELDS = (
    "root_translation",
    "root_velocity",
    "root_angular_velocity",
    "joint_position_rel",
    "joint_velocity_rel",
    "joint_angular_velocity",
    "target_speed",
)


def file_sha256(path: Path) -> str:
    """Return a prefixed SHA-256 digest for one persisted file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            stream.write(text)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_training_sample(path: Path, sample: TrainingSample) -> Path:
    """Atomically persist one declared sample NPZ without object arrays."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    declaration = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sample.arrays.items()
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            payload: dict[str, Any] = {
                "format_version": np.asarray(DATASET_SAMPLE_VERSION),
                "sample_id": np.asarray(sample.sample_id),
                "metadata_json": np.asarray(_json(sample.metadata)),
                "declaration_json": np.asarray(_json(declaration)),
            }
            payload.update(sample.arrays)
            np.savez_compressed(stream, **payload)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_training_sample(path: Path) -> TrainingSample:
    """Load and validate one training sample without enabling pickle."""
    source = Path(path)
    try:
        context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read training sample: {source}") from exc
    with context as archive:
        required = {"format_version", "sample_id", "metadata_json", "declaration_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"training sample is missing fields: {sorted(missing)}")
        if str(archive["format_version"].item()) != DATASET_SAMPLE_VERSION:
            raise ValueError("unsupported training sample version")
        sample_id = str(archive["sample_id"].item())
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
            declaration = json.loads(str(archive["declaration_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("training sample contains invalid JSON") from exc
        if not isinstance(metadata, dict) or not isinstance(declaration, dict):
            raise ValueError("training sample JSON fields must be objects")
        declared = set(declaration)
        missing_arrays = declared.difference(archive.files)
        unexpected = set(archive.files).difference(required).difference(declared)
        if missing_arrays or unexpected:
            raise ValueError(
                f"training sample arrays mismatch; missing={sorted(missing_arrays)}, "
                f"unexpected={sorted(unexpected)}"
            )
        arrays: dict[str, NDArray[Any]] = {}
        for name, raw_declaration in declaration.items():
            if not isinstance(raw_declaration, dict):
                raise ValueError("training sample array declarations must be objects")
            raw = archive[name]
            if list(raw.shape) != raw_declaration.get("shape") or str(
                raw.dtype
            ) != raw_declaration.get("dtype"):
                raise ValueError(f"training sample array {name!r} violates its declaration")
            arrays[name] = np.array(raw, copy=True)
    return TrainingSample(sample_id=sample_id, arrays=arrays, metadata=metadata)


@dataclass(frozen=True)
class NormalizationStatistics:
    """Per-channel train-only mean/scale statistics."""

    means: Mapping[str, NDArray[np.float32]]
    scales: Mapping[str, NDArray[np.float32]]
    counts: Mapping[str, int]
    training_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.means) != set(NORMALIZED_FIELDS) or set(self.scales) != set(NORMALIZED_FIELDS):
            raise ValueError("normalization fields do not match the fixed schema")
        means: dict[str, NDArray[np.float32]] = {}
        scales: dict[str, NDArray[np.float32]] = {}
        for name in NORMALIZED_FIELDS:
            mean = np.array(self.means[name], dtype=np.float32, copy=True)
            scale = np.array(self.scales[name], dtype=np.float32, copy=True)
            if mean.shape != scale.shape or np.any(scale <= 0.0):
                raise ValueError("normalization mean/scale shapes must match with positive scale")
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
                raise ValueError("normalization statistics must be finite")
            mean.setflags(write=False)
            scale.setflags(write=False)
            means[name] = mean
            scales[name] = scale
            if self.counts.get(name, 0) <= 0:
                raise ValueError("normalization counts must be positive")
        object.__setattr__(self, "means", MappingProxyType(means))
        object.__setattr__(self, "scales", MappingProxyType(scales))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "training_sample_ids", tuple(self.training_sample_ids))


def fit_training_normalization(samples: Sequence[TrainingSample]) -> NormalizationStatistics:
    """Fit translation/velocity statistics from training-partition samples only."""
    training = [
        sample
        for sample in samples
        if sample.metadata.get("split") == "train"
        and sample.metadata.get("catalog_partition") in {"clean", "equivalent", "train", "sham"}
    ]
    if not training:
        raise ValueError("normalization requires at least one training sample")
    sums: dict[str, NDArray[np.float64]] = {}
    squared: dict[str, NDArray[np.float64]] = {}
    counts: dict[str, int] = {}
    for sample in training:
        for name in NORMALIZED_FIELDS:
            value = np.asarray(sample.arrays[name], dtype=np.float64)
            flattened = value.reshape((-1, value.shape[-1]))
            sums[name] = sums.get(name, np.zeros(value.shape[-1], dtype=np.float64)) + np.sum(
                flattened, axis=0
            )
            squared[name] = squared.get(name, np.zeros(value.shape[-1], dtype=np.float64)) + np.sum(
                flattened * flattened, axis=0
            )
            counts[name] = counts.get(name, 0) + flattened.shape[0]
    means: dict[str, NDArray[np.float32]] = {}
    scales: dict[str, NDArray[np.float32]] = {}
    for name in NORMALIZED_FIELDS:
        mean = sums[name] / counts[name]
        variance = np.maximum(squared[name] / counts[name] - mean * mean, 0.0)
        means[name] = mean.astype(np.float32)
        scales[name] = np.maximum(np.sqrt(variance), 1.0e-6).astype(np.float32)
    return NormalizationStatistics(
        means=means,
        scales=scales,
        counts=counts,
        training_sample_ids=tuple(sorted(sample.sample_id for sample in training)),
    )


def save_normalization_statistics(directory: Path, stats: NormalizationStatistics) -> Path:
    """Persist normalization arrays and a checksummed train-only manifest."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "normalization.npz"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output,
            prefix=".normalization.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            payload: dict[str, Any] = {
                "format_version": np.asarray(NORMALIZATION_VERSION),
                "training_sample_ids": np.asarray(stats.training_sample_ids, dtype=np.str_),
            }
            for name in NORMALIZED_FIELDS:
                payload[f"{name}_mean"] = stats.means[name]
                payload[f"{name}_scale"] = stats.scales[name]
                payload[f"{name}_count"] = np.asarray(stats.counts[name], dtype=np.int64)
            np.savez_compressed(stream, **payload)
        os.replace(temporary_path, arrays_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    manifest = {
        "format_version": NORMALIZATION_VERSION,
        "fit_partition": {
            "split": "train",
            "catalog_partitions": ["clean", "equivalent", "train", "sham"],
        },
        "excluded_fields": ["local_rot6d", "contact_soft", "phase_sincos", "all_targets"],
        "fields": {
            name: {
                "mean_shape": list(stats.means[name].shape),
                "scale_shape": list(stats.scales[name].shape),
                "count": stats.counts[name],
            }
            for name in NORMALIZED_FIELDS
        },
        "training_sample_count": len(stats.training_sample_ids),
        "training_sample_ids_sha256": (
            "sha256:" + hashlib.sha256("\n".join(stats.training_sample_ids).encode()).hexdigest()
        ),
        "arrays": {"path": arrays_path.name, "sha256": file_sha256(arrays_path)},
    }
    manifest_path = output / "normalization.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_normalization_statistics(directory: Path) -> NormalizationStatistics:
    """Load and verify versioned normalization statistics without pickle."""
    root = Path(directory)
    manifest_path = root / "normalization.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read normalization manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != NORMALIZATION_VERSION:
        raise ValueError("unsupported normalization manifest")
    arrays_record = manifest.get("arrays")
    if not isinstance(arrays_record, dict) or arrays_record.get("path") != "normalization.npz":
        raise ValueError("normalization arrays declaration is invalid")
    arrays_path = root / "normalization.npz"
    if file_sha256(arrays_path) != arrays_record.get("sha256"):
        raise ValueError("normalization array checksum mismatch")
    try:
        context = np.load(arrays_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("unable to read normalization arrays") from exc
    means: dict[str, NDArray[np.float32]] = {}
    scales: dict[str, NDArray[np.float32]] = {}
    counts: dict[str, int] = {}
    with context as archive:
        if str(archive["format_version"].item()) != NORMALIZATION_VERSION:
            raise ValueError("normalization array version does not match")
        training_ids = tuple(str(value) for value in archive["training_sample_ids"].tolist())
        for name in NORMALIZED_FIELDS:
            means[name] = np.asarray(archive[f"{name}_mean"], dtype=np.float32)
            scales[name] = np.asarray(archive[f"{name}_scale"], dtype=np.float32)
            counts[name] = int(archive[f"{name}_count"].item())
    if len(training_ids) != manifest.get("training_sample_count"):
        raise ValueError("normalization training sample count does not match")
    return NormalizationStatistics(
        means=means,
        scales=scales,
        counts=counts,
        training_sample_ids=training_ids,
    )
