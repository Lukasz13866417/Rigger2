"""Immutable source, split, and operation lineage for derived motion clips."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from motionlab.motion.clip import MotionClip

PROVENANCE_FORMAT_VERSION = "motionlab.provenance.v1"
OperationKind = Literal["source", "transform", "retarget", "corruption", "repair", "slice"]
SplitName = Literal["train", "validation", "test"]


def _json_copy(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(_deep_thaw(value), sort_keys=True, allow_nan=False)
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be an object")
    return result


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _nonempty_optional(value: str | None, *, name: str) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"{name} must be nonempty when supplied")


@dataclass(frozen=True)
class LineageStep:
    """One typed derivation from a content-addressed parent motion."""

    operation: str
    kind: OperationKind
    parent_motion_id: str | None
    version: str | None = None
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("lineage operation must be nonempty")
        _nonempty_optional(self.parent_motion_id, name="parent_motion_id")
        _nonempty_optional(self.version, name="operation version")
        if self.kind not in {"source", "transform", "retarget", "corruption", "repair", "slice"}:
            raise ValueError(f"unsupported lineage operation kind {self.kind!r}")
        parameters = _json_copy(self.parameters, name="lineage parameters")
        object.__setattr__(self, "parameters", _deep_freeze(parameters))

    def to_json(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "kind": self.kind,
            "parent_motion_id": self.parent_motion_id,
            "version": self.version,
            "seed": self.seed,
            "parameters": _deep_thaw(self.parameters),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> LineageStep:
        raw_parameters = value.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("lineage step parameters must be an object")
        return cls(
            operation=str(value["operation"]),
            kind=value["kind"],
            parent_motion_id=(
                None if value.get("parent_motion_id") is None else str(value["parent_motion_id"])
            ),
            version=None if value.get("version") is None else str(value["version"]),
            seed=None if value.get("seed") is None else int(value["seed"]),
            parameters=raw_parameters,
        )


@dataclass(frozen=True)
class Provenance:
    """Source identity and immutable split lineage, separate from neural inputs."""

    motion_id: str
    source_dataset: str | None
    source_clip_id: str | None
    source_take_id: str | None
    split_lineage_id: str | None
    split: SplitName | None = None
    importer_version: str | None = None
    retargeter_version: str | None = None
    clean_confidence: float | None = None
    operations: tuple[LineageStep, ...] = ()
    lineage_complete: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.motion_id.strip():
            raise ValueError("motion_id must be nonempty")
        if self.split not in {None, "train", "validation", "test"}:
            raise ValueError(f"unsupported split {self.split!r}")
        for name in (
            "source_dataset",
            "source_clip_id",
            "source_take_id",
            "split_lineage_id",
            "importer_version",
            "retargeter_version",
        ):
            _nonempty_optional(getattr(self, name), name=name)
        if self.clean_confidence is not None and (
            not np.isfinite(self.clean_confidence) or not 0.0 <= self.clean_confidence <= 1.0
        ):
            raise ValueError("clean_confidence must be finite and within [0,1]")
        operations = tuple(self.operations)
        if not all(isinstance(step, LineageStep) for step in operations):
            raise TypeError("operations must contain LineageStep values")
        metadata = _json_copy(self.metadata, name="provenance metadata")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "metadata", _deep_freeze(metadata))

    @property
    def corruption_lineage(self) -> tuple[LineageStep, ...]:
        return tuple(step for step in self.operations if step.kind == "corruption")

    def to_embedded_json(self) -> dict[str, Any]:
        """Serialize lineage without the current ID, avoiding a hash dependency cycle."""
        return {
            "format_version": PROVENANCE_FORMAT_VERSION,
            "source_dataset": self.source_dataset,
            "source_clip_id": self.source_clip_id,
            "source_take_id": self.source_take_id,
            "split_lineage_id": self.split_lineage_id,
            "split": self.split,
            "importer_version": self.importer_version,
            "retargeter_version": self.retargeter_version,
            "clean_confidence": self.clean_confidence,
            "operations": [step.to_json() for step in self.operations],
            "lineage_complete": self.lineage_complete,
            "metadata": _deep_thaw(self.metadata),
        }

    @classmethod
    def from_embedded_json(
        cls,
        value: Mapping[str, Any],
        *,
        motion_id: str,
    ) -> Provenance:
        if value.get("format_version") != PROVENANCE_FORMAT_VERSION:
            raise ValueError("unsupported or missing embedded provenance format version")
        raw_operations = value.get("operations", [])
        if not isinstance(raw_operations, list) or not all(
            isinstance(step, dict) for step in raw_operations
        ):
            raise ValueError("embedded provenance operations must be a list of objects")
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("embedded provenance metadata must be an object")
        raw_clean_confidence = value.get("clean_confidence")
        return cls(
            motion_id=motion_id,
            source_dataset=_optional_string(value.get("source_dataset")),
            source_clip_id=_optional_string(value.get("source_clip_id")),
            source_take_id=_optional_string(value.get("source_take_id")),
            split_lineage_id=_optional_string(value.get("split_lineage_id")),
            split=value.get("split"),
            importer_version=_optional_string(value.get("importer_version")),
            retargeter_version=_optional_string(value.get("retargeter_version")),
            clean_confidence=(
                None if raw_clean_confidence is None else float(raw_clean_confidence)
            ),
            operations=tuple(LineageStep.from_json(step) for step in raw_operations),
            lineage_complete=bool(value.get("lineage_complete", False)),
            metadata=raw_metadata,
        )


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def make_split_lineage_id(source_dataset: str, source_take_id: str) -> str:
    """Derive a stable split-family ID from global source identity."""
    if not source_dataset.strip() or not source_take_id.strip():
        raise ValueError("source dataset and take IDs must be nonempty")
    digest = hashlib.sha256()
    digest.update(source_dataset.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_take_id.encode("utf-8"))
    return f"split:sha256:{digest.hexdigest()}"


def assign_source_lineage(
    clip: MotionClip,
    *,
    source_dataset: str,
    source_clip_id: str,
    source_take_id: str,
    split: SplitName | None = None,
    split_lineage_id: str | None = None,
    importer_version: str | None = None,
    clean_confidence: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MotionClip:
    """Assign source identity exactly once, before creating any descendants."""
    if "provenance" in clip.metadata:
        raise ValueError("source provenance is immutable and has already been assigned")
    lineage_id = (
        make_split_lineage_id(source_dataset, source_take_id)
        if split_lineage_id is None
        else split_lineage_id
    )
    provenance = Provenance(
        motion_id=clip.content_hash,
        source_dataset=source_dataset,
        source_clip_id=source_clip_id,
        source_take_id=source_take_id,
        split_lineage_id=lineage_id,
        split=split,
        importer_version=importer_version,
        clean_confidence=clean_confidence,
        operations=(LineageStep("source", "source", None, version=importer_version),),
        lineage_complete=True,
        metadata={} if metadata is None else metadata,
    )
    clip_metadata = dict(clip.metadata)
    clip_metadata["provenance"] = provenance.to_embedded_json()
    return clip.with_updates(metadata=clip_metadata)


def append_operation_lineage(
    clip: MotionClip,
    output_metadata: Mapping[str, Any],
    *,
    operation: str,
    kind: OperationKind,
    parameters: Mapping[str, Any] | None = None,
    version: str | None = None,
    seed: int | None = None,
    retargeter_version: str | None = None,
) -> dict[str, Any]:
    """Append a step when lineage is present; leave legacy metadata byte-for-byte unchanged."""
    result = dict(output_metadata)
    raw = clip.metadata.get("provenance")
    if raw is None:
        return result
    if not isinstance(raw, dict):
        raise ValueError("embedded provenance must be an object")
    provenance = Provenance.from_embedded_json(raw, motion_id=clip.content_hash)
    updated = Provenance(
        motion_id=clip.content_hash,
        source_dataset=provenance.source_dataset,
        source_clip_id=provenance.source_clip_id,
        source_take_id=provenance.source_take_id,
        split_lineage_id=provenance.split_lineage_id,
        split=provenance.split,
        importer_version=provenance.importer_version,
        retargeter_version=(
            retargeter_version if retargeter_version is not None else provenance.retargeter_version
        ),
        clean_confidence=provenance.clean_confidence,
        operations=(
            *provenance.operations,
            LineageStep(
                operation=operation,
                kind=kind,
                parent_motion_id=clip.content_hash,
                version=version,
                seed=seed,
                parameters={} if parameters is None else parameters,
            ),
        ),
        lineage_complete=provenance.lineage_complete,
        metadata=provenance.metadata,
    )
    result["provenance"] = updated.to_embedded_json()
    return result


def provenance_from_clip(clip: MotionClip) -> Provenance:
    """Read embedded provenance or explicitly expose incomplete legacy metadata."""
    raw = clip.metadata.get("provenance")
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("embedded provenance must be an object")
        return Provenance.from_embedded_json(raw, motion_id=clip.content_hash)

    operation = clip.metadata.get("operation")
    steps: tuple[LineageStep, ...] = ()
    if isinstance(operation, str) and operation:
        kind: OperationKind = (
            "corruption"
            if operation.startswith("corrupt_")
            else "retarget"
            if operation.startswith("retarget_")
            else "repair"
            if operation in {"lock_foot", "close_loop"}
            else "transform"
        )
        parent = clip.metadata.get("parent_motion_id")
        steps = (
            LineageStep(
                operation=operation,
                kind=kind,
                parent_motion_id=parent if isinstance(parent, str) else None,
                seed=(
                    int(clip.metadata["corruption_seed"])
                    if "corruption_seed" in clip.metadata
                    else None
                ),
                parameters=(
                    clip.metadata["operator_parameters"]
                    if isinstance(clip.metadata.get("operator_parameters"), dict)
                    else {}
                ),
            ),
        )
    return Provenance(
        motion_id=clip.content_hash,
        source_dataset=(
            str(clip.metadata["source_dataset"])
            if clip.metadata.get("source_dataset") is not None
            else None
        ),
        source_clip_id=(
            str(clip.metadata["source_clip_id"])
            if clip.metadata.get("source_clip_id") is not None
            else None
        ),
        source_take_id=(
            str(clip.metadata["source_take_id"])
            if clip.metadata.get("source_take_id") is not None
            else None
        ),
        split_lineage_id=(
            str(clip.metadata["split_lineage_id"])
            if clip.metadata.get("split_lineage_id") is not None
            else None
        ),
        operations=steps,
        lineage_complete=False,
        metadata={"reason": "legacy_clip_without_embedded_provenance"},
    )
