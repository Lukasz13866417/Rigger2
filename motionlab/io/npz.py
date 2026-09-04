"""Stable, versioned, pickle-free NPZ serialization for fixed-rig clips."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motionlab.constants import NPZ_FORMAT_VERSION
from motionlab.motion.clip import MotionClip
from motionlab.motion.markers import JointLimitSpec, MarkerSpec
from motionlab.motion.skeleton import Skeleton

_REQUIRED_FIELDS = frozenset(
    {
        "format_version",
        "joint_names",
        "parents",
        "rest_offsets_m",
        "rest_local_quat_wxyz",
        "local_quat_wxyz",
        "root_translation_m",
        "fps",
        "metadata_json",
        "roles_json",
        "markers_json",
        "joint_limits_json",
    }
)


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_object(raw: NDArray[Any], *, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw.item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object in NPZ field {field!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"NPZ field {field!r} must contain a JSON object")
    return parsed


def save_motion_npz(path: Path, clip: MotionClip) -> None:
    """Atomically write a validated motion clip to a compressed NPZ file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "clip": dict(clip.metadata),
        "skeleton": dict(clip.skeleton.metadata),
    }
    marker_payload = {
        name: marker.model_dump(mode="json") for name, marker in clip.skeleton.markers.items()
    }
    limit_payload = {
        str(index): limit.model_dump(mode="json")
        for index, limit in clip.skeleton.joint_limits.items()
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        np.savez_compressed(
            temporary_path,
            format_version=np.asarray(NPZ_FORMAT_VERSION),
            joint_names=np.asarray(clip.skeleton.joint_names, dtype=np.str_),
            parents=clip.skeleton.parents,
            rest_offsets_m=clip.skeleton.rest_offsets_m,
            rest_local_quat_wxyz=clip.skeleton.rest_local_quat_wxyz,
            local_quat_wxyz=clip.local_quat_wxyz,
            root_translation_m=clip.root_translation_m,
            fps=np.asarray(clip.fps, dtype=np.float64),
            metadata_json=np.asarray(_json_dump(metadata)),
            roles_json=np.asarray(_json_dump(dict(clip.skeleton.roles))),
            markers_json=np.asarray(_json_dump(marker_payload)),
            joint_limits_json=np.asarray(_json_dump(limit_payload)),
        )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_motion_npz(path: Path) -> MotionClip:
    """Load and validate a MotionLab motion clip without enabling pickle."""
    source = Path(path)
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read MotionLab NPZ: {source}") from exc

    with archive_context as archive:
        missing = _REQUIRED_FIELDS.difference(archive.files)
        if missing:
            raise ValueError(f"MotionLab NPZ is missing required fields: {sorted(missing)}")
        version = str(archive["format_version"].item())
        if version != NPZ_FORMAT_VERSION:
            raise ValueError(
                f"unsupported MotionLab NPZ version {version!r}; expected {NPZ_FORMAT_VERSION!r}"
            )
        metadata = _json_object(archive["metadata_json"], field="metadata_json")
        roles = _json_object(archive["roles_json"], field="roles_json")
        markers_raw = _json_object(archive["markers_json"], field="markers_json")
        limits_raw = _json_object(archive["joint_limits_json"], field="joint_limits_json")
        markers = {name: MarkerSpec.model_validate(marker) for name, marker in markers_raw.items()}
        limits = {
            int(index): JointLimitSpec.model_validate(limit) for index, limit in limits_raw.items()
        }
        joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
        skeleton = Skeleton(
            joint_names=joint_names,
            parents=archive["parents"],
            rest_offsets_m=archive["rest_offsets_m"],
            rest_local_quat_wxyz=archive["rest_local_quat_wxyz"],
            roles={str(role): int(index) for role, index in roles.items()},
            markers=markers,
            joint_limits=limits,
            metadata=metadata.get("skeleton", {}),
        )
        return MotionClip(
            skeleton=skeleton,
            local_quat_wxyz=archive["local_quat_wxyz"],
            root_translation_m=archive["root_translation_m"],
            fps=float(archive["fps"].item()),
            metadata=metadata.get("clip", {}),
        )
