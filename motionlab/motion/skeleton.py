"""Validated immutable fixed-rig skeleton representation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motionlab.constants import QUATERNION_NORM_TOLERANCE
from motionlab.motion._validation import frozen_array, frozen_json_mapping
from motionlab.motion.markers import JointLimitSpec, MarkerSpec


@dataclass(frozen=True)
class Skeleton:
    """A single-root joint hierarchy with rest transforms and semantic metadata.

    Arrays have shapes ``parents [J]``, ``rest_offsets_m [J,3]``, and
    ``rest_local_quat_wxyz [J,4]``. They are defensively copied and made read-only.
    """

    joint_names: tuple[str, ...]
    parents: NDArray[np.int32]
    rest_offsets_m: NDArray[np.float32]
    rest_local_quat_wxyz: NDArray[np.float32]
    roles: Mapping[str, int] = field(default_factory=dict)
    markers: Mapping[str, MarkerSpec] = field(default_factory=dict)
    joint_limits: Mapping[int, JointLimitSpec] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        if not names:
            raise ValueError("skeleton must contain at least one joint")
        if any(not name or not isinstance(name, str) for name in names):
            raise ValueError("every joint name must be a non-empty string")
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique")

        parents = frozen_array(self.parents, dtype=np.dtype(np.int32), name="parents")
        offsets = frozen_array(
            self.rest_offsets_m,
            dtype=np.dtype(np.float32),
            name="rest_offsets_m",
        )
        rest_quaternions = frozen_array(
            self.rest_local_quat_wxyz,
            dtype=np.dtype(np.float32),
            name="rest_local_quat_wxyz",
        )
        joint_count = len(names)
        if parents.shape != (joint_count,):
            raise ValueError(f"parents must have shape ({joint_count},), got {parents.shape}")
        if offsets.shape != (joint_count, 3):
            raise ValueError(
                f"rest_offsets_m must have shape ({joint_count}, 3), got {offsets.shape}"
            )
        if rest_quaternions.shape != (joint_count, 4):
            raise ValueError(
                "rest_local_quat_wxyz must have shape "
                f"({joint_count}, 4), got {rest_quaternions.shape}"
            )

        roots = np.flatnonzero(parents == -1)
        if roots.size != 1:
            raise ValueError(f"skeleton must have exactly one root, found {roots.size}")
        if np.any((parents < -1) | (parents >= joint_count)):
            raise ValueError("parent indices must be -1 or valid joint indices")
        if np.any(parents == np.arange(joint_count, dtype=np.int32)):
            raise ValueError("a joint cannot parent itself")
        self._validate_acyclic(parents)

        norms = np.linalg.norm(rest_quaternions, axis=-1)
        if not np.allclose(norms, 1.0, atol=QUATERNION_NORM_TOLERANCE, rtol=0.0):
            raise ValueError("rest_local_quat_wxyz must contain unit quaternions")

        roles = {str(role): int(index) for role, index in self.roles.items()}
        for role, index in roles.items():
            if not role:
                raise ValueError("semantic role names must be non-empty")
            if index < 0 or index >= joint_count:
                raise ValueError(f"role {role!r} references invalid joint index {index}")
        by_index: dict[int, set[str]] = {}
        for role, index in roles.items():
            by_index.setdefault(index, set()).add(role)
        permitted_alias_groups = (
            {"root", "pelvis"},
            {"left_ankle", "left_foot", "left_toe"},
            {"right_ankle", "right_foot", "right_toe"},
        )
        for aliases in by_index.values():
            aliases_are_permitted = any(aliases.issubset(group) for group in permitted_alias_groups)
            if len(aliases) > 1 and not aliases_are_permitted:
                raise ValueError(
                    f"roles must be one-to-one; duplicate assignment: {sorted(aliases)}"
                )

        markers: dict[str, MarkerSpec] = {}
        for marker_name, raw_marker in self.markers.items():
            marker = (
                raw_marker
                if isinstance(raw_marker, MarkerSpec)
                else MarkerSpec.model_validate(raw_marker)
            )
            self._resolve_joint_reference(marker.joint, names)
            markers[str(marker_name)] = marker

        limits: dict[int, JointLimitSpec] = {}
        for raw_index, raw_limit in self.joint_limits.items():
            index = int(raw_index)
            if index < 0 or index >= joint_count:
                raise ValueError(f"joint limit references invalid joint index {index}")
            limit = (
                raw_limit
                if isinstance(raw_limit, JointLimitSpec)
                else JointLimitSpec.model_validate(raw_limit)
            )
            limits[index] = limit

        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "rest_offsets_m", offsets)
        object.__setattr__(self, "rest_local_quat_wxyz", rest_quaternions)
        object.__setattr__(self, "roles", MappingProxyType(roles))
        object.__setattr__(self, "markers", MappingProxyType(markers))
        object.__setattr__(self, "joint_limits", MappingProxyType(limits))
        object.__setattr__(self, "metadata", frozen_json_mapping(self.metadata, name="metadata"))

    @staticmethod
    def _validate_acyclic(parents: NDArray[np.int32]) -> None:
        state = np.zeros(parents.shape[0], dtype=np.int8)

        def visit(joint: int) -> None:
            if state[joint] == 1:
                raise ValueError("parent hierarchy contains a cycle")
            if state[joint] == 2:
                return
            state[joint] = 1
            parent = int(parents[joint])
            if parent >= 0:
                visit(parent)
            state[joint] = 2

        for joint_index in range(parents.shape[0]):
            visit(joint_index)

    @staticmethod
    def _resolve_joint_reference(reference: str | int, names: tuple[str, ...]) -> int:
        if isinstance(reference, int):
            if reference < 0 or reference >= len(names):
                raise ValueError(f"marker references invalid joint index {reference}")
            return reference
        try:
            return names.index(reference)
        except ValueError as exc:
            raise ValueError(f"marker references unknown joint {reference!r}") from exc

    @property
    def num_joints(self) -> int:
        """Return the number of joints."""
        return len(self.joint_names)

    @property
    def root_index(self) -> int:
        """Return the unique root joint index."""
        return int(np.flatnonzero(self.parents == -1)[0])

    @property
    def topological_order(self) -> tuple[int, ...]:
        """Return parents before children, independent of storage order."""
        children: list[list[int]] = [[] for _ in range(self.num_joints)]
        for child, parent in enumerate(self.parents):
            if parent >= 0:
                children[int(parent)].append(child)
        order: list[int] = []
        stack = [self.root_index]
        while stack:
            joint = stack.pop()
            order.append(joint)
            stack.extend(reversed(children[joint]))
        return tuple(order)

    def resolve_joint(self, reference: str | int) -> int:
        """Resolve a numeric or named joint reference."""
        return self._resolve_joint_reference(reference, self.joint_names)

    @property
    def content_hash(self) -> str:
        """Return a stable SHA-256 identity for skeleton structure and metadata."""
        digest = hashlib.sha256()
        digest.update(json.dumps(self.joint_names, separators=(",", ":")).encode())
        for array in (self.parents, self.rest_offsets_m, self.rest_local_quat_wxyz):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes(order="C"))
        semantic = {
            "roles": dict(self.roles),
            "markers": {
                name: marker.model_dump(mode="json") for name, marker in self.markers.items()
            },
            "joint_limits": {
                str(index): limit.model_dump(mode="json")
                for index, limit in self.joint_limits.items()
            },
            "metadata": dict(self.metadata),
        }
        digest.update(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode())
        return f"sha256:{digest.hexdigest()}"
