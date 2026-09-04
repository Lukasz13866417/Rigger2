"""Canonical variable-rig graph data contracts without a learned graph model.

This module is intentionally limited to representation and batching.  It does not
define message passing, attention, a critic, or a training objective.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from motionlab.core.clip_metadata import ClipMetadataAdapter, adapt_motion_clip
from motionlab.core.coordinate_contract import CoordinateContract
from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.math.finite_difference import finite_difference
from motionlab.math.quaternion import (
    quaternion_interval_angular_velocity,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_rotate_vector,
    quaternion_to_matrix,
)
from motionlab.math.rotation6d import matrix_to_rotation_6d
from motionlab.motion._validation import frozen_array
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton
from motionlab.rigging.semantic_roles import FunctionalPart, JointSide

SKELETON_GRAPH_VERSION = "motionlab.skeleton_graph.v1"
MOTION_GRAPH_VERSION = "motionlab.motion_graph.v1"
VARIABLE_RIG_BATCH_VERSION = "motionlab.variable_rig_batch.v1"
LOCAL_BASIS_CONVENTION = "declared_frame_postmultiplied_by_quaternion_equals_reference_frame"


@dataclass(frozen=True)
class FeatureField:
    """One named contiguous field in a graph feature stream."""

    name: str
    width: int
    units: str
    coordinate_space: Literal[
        "canonical_world",
        "root_relative_canonical_world",
        "reference_local_rig",
        "semantic",
        "time",
    ]

    def __post_init__(self) -> None:
        if not self.name or not self.units or self.width < 1:
            raise ValueError("feature fields require a name, units, and positive width")


@dataclass(frozen=True)
class FeatureContract:
    """Versioned layout for one static or dynamic feature stream."""

    name: str
    version: str
    time_varying: bool
    fields: tuple[FeatureField, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if not self.name or not self.version or not fields:
            raise ValueError("feature contracts require a name, version, and fields")
        names = [field.name for field in fields]
        if len(set(names)) != len(names):
            raise ValueError("feature field names must be unique within a stream")
        object.__setattr__(self, "fields", fields)

    @property
    def width(self) -> int:
        """Return the total final-axis width."""
        return sum(field.width for field in self.fields)

    def field_slice(self, name: str) -> slice:
        """Return the exact final-axis slice for a named field."""
        start = 0
        for field in self.fields:
            stop = start + field.width
            if field.name == name:
                return slice(start, stop)
            start = stop
        raise KeyError(name)


STATIC_UNIVERSAL_CONTRACT = FeatureContract(
    name="static_universal",
    version="motionlab.graph_features.static_universal.v1",
    time_varying=False,
    fields=(
        FeatureField("rest_position_root_relative_m", 3, "meter", "root_relative_canonical_world"),
        FeatureField("rest_bone_vector_world_m", 3, "meter", "canonical_world"),
        FeatureField("bone_length_m", 1, "meter", "semantic"),
        FeatureField("functional_part_one_hot", len(FunctionalPart), "unitless", "semantic"),
        FeatureField("side_one_hot", len(JointSide), "unitless", "semantic"),
        FeatureField("role_confidence", 1, "probability", "semantic"),
        FeatureField("is_deform_joint", 1, "boolean", "semantic"),
        FeatureField("is_helper_joint", 1, "boolean", "semantic"),
        FeatureField("is_controller", 1, "boolean", "semantic"),
        FeatureField("is_end_site", 1, "boolean", "semantic"),
        FeatureField("model_joint_mask", 1, "boolean", "semantic"),
    ),
)

STATIC_RIG_CONTRACT = FeatureContract(
    name="static_rig",
    version="motionlab.graph_features.static_rig.v1",
    time_varying=False,
    fields=(
        FeatureField("rest_offset_parent_reference_m", 3, "meter", "reference_local_rig"),
        FeatureField("rest_local_rotation_reference_6d", 6, "rotation_6d", "reference_local_rig"),
        FeatureField("local_axis_confidence", 1, "probability", "reference_local_rig"),
    ),
)

DYNAMIC_UNIVERSAL_CONTRACT = FeatureContract(
    name="dynamic_universal",
    version="motionlab.graph_features.dynamic_universal.v1",
    time_varying=True,
    fields=(
        FeatureField("position_root_relative_m", 3, "meter", "root_relative_canonical_world"),
        FeatureField("linear_velocity_world_mps", 3, "meter_per_second", "canonical_world"),
        FeatureField("angular_velocity_world_radps", 3, "radian_per_second", "canonical_world"),
    ),
)

DYNAMIC_RIG_CONTRACT = FeatureContract(
    name="dynamic_rig",
    version="motionlab.graph_features.dynamic_rig.v1",
    time_varying=True,
    fields=(
        FeatureField("local_rotation_reference_6d", 6, "rotation_6d", "reference_local_rig"),
        FeatureField(
            "local_angular_velocity_reference_radps", 3, "radian_per_second", "reference_local_rig"
        ),
    ),
)

GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT = FeatureContract(
    name="global_dynamic_universal",
    version="motionlab.graph_features.global_dynamic_universal.v1",
    time_varying=True,
    fields=(
        FeatureField("root_translation_world_m", 3, "meter", "canonical_world"),
        FeatureField("root_velocity_world_mps", 3, "meter_per_second", "canonical_world"),
        FeatureField("timestamp_s", 1, "second", "time"),
    ),
)


@dataclass(frozen=True)
class CoordinateBasisMetadata:
    """World-coordinate provenance and per-joint local-basis declarations.

    MotionClip values are already right-handed, Y-up, Z-forward, and metric.  The
    coordinate contract records how an importer reached that space.  A local basis
    quaternion ``C[j]`` follows ``G_reference = G_declared * C[j]``.  It therefore
    permits local features to be normalized without changing physical world motion.
    """

    coordinates: CoordinateContract
    declared_to_reference_quat_wxyz: NDArray[np.float32]
    confidence: NDArray[np.float32]
    source: str
    convention: str = LOCAL_BASIS_CONVENTION

    def __post_init__(self) -> None:
        basis = frozen_array(
            self.declared_to_reference_quat_wxyz,
            dtype=np.dtype(np.float32),
            name="declared_to_reference_quat_wxyz",
        )
        confidence = frozen_array(
            self.confidence,
            dtype=np.dtype(np.float32),
            name="local basis confidence",
        )
        if basis.ndim != 2 or basis.shape[1] != 4 or confidence.shape != basis.shape[:1]:
            raise ValueError("local basis arrays must have shapes [J,4] and [J]")
        if not np.allclose(np.linalg.norm(basis, axis=-1), 1.0, atol=1.0e-5, rtol=0.0):
            raise ValueError("local basis metadata must contain unit quaternions")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("local basis confidence must be within [0,1]")
        if not self.source.strip() or self.convention != LOCAL_BASIS_CONVENTION:
            raise ValueError("local basis source/convention is invalid")
        object.__setattr__(self, "declared_to_reference_quat_wxyz", basis)
        object.__setattr__(self, "confidence", confidence)


def _frozen_bool(value: NDArray[np.bool_], *, name: str) -> NDArray[np.bool_]:
    return frozen_array(value, dtype=np.dtype(np.bool_), name=name)


def _frozen_int32(value: NDArray[np.int32], *, name: str) -> NDArray[np.int32]:
    return frozen_array(value, dtype=np.dtype(np.int32), name=name)


@dataclass(frozen=True)
class SkeletonGraph:
    """Canonical, topology-aware static data for one arbitrary-joint rig."""

    version: str
    joint_names: tuple[str, ...]
    node_keys: tuple[str, ...]
    semantic_roles: tuple[str, ...]
    role_aliases: tuple[tuple[str, ...], ...]
    parents: NDArray[np.int32]
    edge_index: NDArray[np.int32]
    anatomical_parent: NDArray[np.int32]
    anatomical_edge_index: NDArray[np.int32]
    source_joint_index: NDArray[np.int32]
    source_to_canonical_index: NDArray[np.int32]
    functional_part: NDArray[np.int8]
    side: NDArray[np.int8]
    is_deform_joint: NDArray[np.bool_]
    is_helper_joint: NDArray[np.bool_]
    is_controller: NDArray[np.bool_]
    is_end_site: NDArray[np.bool_]
    model_joint_mask: NDArray[np.bool_]
    static_universal: NDArray[np.float32]
    static_rig: NDArray[np.float32]
    basis: CoordinateBasisMetadata

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        keys = tuple(self.node_keys)
        roles = tuple(self.semantic_roles)
        aliases = tuple(tuple(value) for value in self.role_aliases)
        joint_count = len(names)
        if self.version != SKELETON_GRAPH_VERSION:
            raise ValueError(f"unsupported skeleton graph version {self.version!r}")
        if joint_count < 1 or len(keys) != joint_count or len(set(keys)) != joint_count:
            raise ValueError("skeleton graph node keys must be unique and match nonempty joints")
        if len(roles) != joint_count or len(aliases) != joint_count:
            raise ValueError("skeleton graph semantics must match joint count")
        parents = _frozen_int32(self.parents, name="graph parents")
        edge_index = _frozen_int32(self.edge_index, name="graph edge_index")
        anatomical_parent = _frozen_int32(self.anatomical_parent, name="anatomical_parent")
        anatomical_edges = _frozen_int32(
            self.anatomical_edge_index,
            name="anatomical_edge_index",
        )
        source_joint_index = _frozen_int32(
            self.source_joint_index,
            name="source_joint_index",
        )
        source_to_canonical = _frozen_int32(
            self.source_to_canonical_index,
            name="source_to_canonical_index",
        )
        functional_part = frozen_array(
            self.functional_part,
            dtype=np.dtype(np.int8),
            name="functional_part",
        )
        side = frozen_array(self.side, dtype=np.dtype(np.int8), name="side")
        flags = {
            name: _frozen_bool(getattr(self, name), name=name)
            for name in (
                "is_deform_joint",
                "is_helper_joint",
                "is_controller",
                "is_end_site",
                "model_joint_mask",
            )
        }
        static_universal = frozen_array(
            self.static_universal,
            dtype=np.dtype(np.float32),
            name="static_universal",
        )
        static_rig = frozen_array(
            self.static_rig,
            dtype=np.dtype(np.float32),
            name="static_rig",
        )
        if parents.shape != (joint_count,) or anatomical_parent.shape != (joint_count,):
            raise ValueError("graph parent arrays must have shape [J]")
        if np.any((parents < -1) | (parents >= joint_count)) or np.any(
            parents == np.arange(joint_count, dtype=np.int32)
        ):
            raise ValueError("graph parents must be -1 or valid non-self node indices")
        for start in range(joint_count):
            visited: set[int] = set()
            current = start
            while current >= 0:
                if current in visited:
                    raise ValueError("graph parent hierarchy contains a cycle")
                visited.add(current)
                current = int(parents[current])
        if source_joint_index.shape != (joint_count,) or source_to_canonical.shape != (
            joint_count,
        ):
            raise ValueError("graph permutation arrays must have shape [J]")
        if set(source_joint_index.tolist()) != set(range(joint_count)):
            raise ValueError("source_joint_index must be a permutation")
        if not np.array_equal(source_to_canonical[source_joint_index], np.arange(joint_count)):
            raise ValueError("source/canonical joint permutations must be inverses")
        if edge_index.shape != (2, joint_count - 1):
            raise ValueError("edge_index must contain one directed parent edge per non-root joint")
        if anatomical_edges.ndim != 2 or anatomical_edges.shape[0] != 2:
            raise ValueError("anatomical_edge_index must have shape [2,E]")
        for name, value in (("functional_part", functional_part), ("side", side), *flags.items()):
            if value.shape != (joint_count,):
                raise ValueError(f"{name} must have shape [J]")
        if static_universal.shape != (joint_count, STATIC_UNIVERSAL_CONTRACT.width):
            raise ValueError("static_universal does not match its feature contract")
        if static_rig.shape != (joint_count, STATIC_RIG_CONTRACT.width):
            raise ValueError("static_rig does not match its feature contract")
        if self.basis.confidence.shape != (joint_count,):
            raise ValueError("basis metadata must match graph joint count")
        if np.any((functional_part < 0) | (functional_part >= len(FunctionalPart))):
            raise ValueError("functional_part contains an unknown canonical part index")
        if np.any((side < 0) | (side >= len(JointSide))):
            raise ValueError("side contains an unknown canonical side index")
        roots = np.flatnonzero(parents == -1)
        if roots.size != 1:
            raise ValueError("canonical graph must contain exactly one root")
        expected_edges = np.stack(
            (
                parents[parents >= 0],
                np.flatnonzero(parents >= 0).astype(np.int32),
            )
        )
        if not np.array_equal(edge_index, expected_edges):
            raise ValueError("edge_index must be the canonical parent-to-child hierarchy")
        if np.any((anatomical_parent < -1) | (anatomical_parent >= joint_count)):
            raise ValueError("anatomical_parent contains invalid indices")
        anatomical_children = np.flatnonzero(
            flags["model_joint_mask"] & (anatomical_parent >= 0)
        ).astype(np.int32)
        expected_anatomical_edges = np.stack(
            (anatomical_parent[anatomical_children], anatomical_children)
        )
        if not np.array_equal(anatomical_edges, expected_anatomical_edges):
            raise ValueError("anatomical edges must match the collapsed anatomical parents")
        if np.any(
            (anatomical_parent >= 0) & ~flags["model_joint_mask"][np.maximum(anatomical_parent, 0)]
        ):
            raise ValueError("anatomical parents must reference active model joints")
        if np.any(flags["model_joint_mask"] & (flags["is_controller"] | flags["is_end_site"])):
            raise ValueError("controllers/end sites cannot be model joints")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "node_keys", keys)
        object.__setattr__(self, "semantic_roles", roles)
        object.__setattr__(self, "role_aliases", aliases)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "anatomical_parent", anatomical_parent)
        object.__setattr__(self, "anatomical_edge_index", anatomical_edges)
        object.__setattr__(self, "source_joint_index", source_joint_index)
        object.__setattr__(self, "source_to_canonical_index", source_to_canonical)
        object.__setattr__(self, "functional_part", functional_part)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "static_universal", static_universal)
        object.__setattr__(self, "static_rig", static_rig)
        for name, value in flags.items():
            object.__setattr__(self, name, value)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def active_node_indices(self) -> NDArray[np.int64]:
        result = np.flatnonzero(self.model_joint_mask).astype(np.int64)
        result.setflags(write=False)
        return result

    @property
    def active_node_keys(self) -> tuple[str, ...]:
        return tuple(self.node_keys[index] for index in self.active_node_indices)

    @property
    def anatomical_edge_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self.node_keys[parent], self.node_keys[child])
            for parent, child in self.anatomical_edge_index.T
        )

    @property
    def content_hash(self) -> str:
        """Hash canonical graph content while excluding source storage permutation."""
        digest = hashlib.sha256()
        digest.update(self.version.encode())
        digest.update(
            json.dumps(
                {
                    "joint_names": self.joint_names,
                    "node_keys": self.node_keys,
                    "semantic_roles": self.semantic_roles,
                    "role_aliases": self.role_aliases,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for value in (
            self.parents,
            self.anatomical_parent,
            self.functional_part,
            self.side,
            self.is_deform_joint,
            self.is_helper_joint,
            self.is_controller,
            self.is_end_site,
            self.model_joint_mask,
            self.static_universal,
            self.static_rig,
        ):
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes(order="C"))
        return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class MotionGraph:
    """Canonical dynamic streams for one motion on one SkeletonGraph."""

    version: str
    skeleton: SkeletonGraph
    fps: float
    dynamic_universal: NDArray[np.float32]
    dynamic_rig: NDArray[np.float32]
    global_dynamic_universal: NDArray[np.float32]
    frame_mask: NDArray[np.bool_]
    rotation_valid: NDArray[np.bool_]
    position_valid: NDArray[np.bool_]
    rotation_confidence: NDArray[np.float32]
    position_confidence: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.version != MOTION_GRAPH_VERSION:
            raise ValueError(f"unsupported motion graph version {self.version!r}")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("motion graph fps must be finite and positive")
        dynamic_universal = frozen_array(
            self.dynamic_universal,
            dtype=np.dtype(np.float32),
            name="dynamic_universal",
        )
        dynamic_rig = frozen_array(
            self.dynamic_rig,
            dtype=np.dtype(np.float32),
            name="dynamic_rig",
        )
        global_dynamic = frozen_array(
            self.global_dynamic_universal,
            dtype=np.dtype(np.float32),
            name="global_dynamic_universal",
        )
        frame_mask = _frozen_bool(self.frame_mask, name="frame_mask")
        rotation_valid = _frozen_bool(self.rotation_valid, name="rotation_valid")
        position_valid = _frozen_bool(self.position_valid, name="position_valid")
        rotation_confidence = frozen_array(
            self.rotation_confidence,
            dtype=np.dtype(np.float32),
            name="rotation_confidence",
        )
        position_confidence = frozen_array(
            self.position_confidence,
            dtype=np.dtype(np.float32),
            name="position_confidence",
        )
        if dynamic_universal.ndim != 3:
            raise ValueError("dynamic streams must have shape [T,J,F]")
        time, joints = dynamic_universal.shape[:2]
        if joints != self.skeleton.num_joints or time < 1:
            raise ValueError("dynamic stream dimensions do not match the skeleton")
        if dynamic_universal.shape[2] != DYNAMIC_UNIVERSAL_CONTRACT.width:
            raise ValueError("dynamic_universal does not match its feature contract")
        if dynamic_rig.shape != (time, joints, DYNAMIC_RIG_CONTRACT.width):
            raise ValueError("dynamic_rig does not match its feature contract")
        if global_dynamic.shape != (time, GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT.width):
            raise ValueError("global dynamic stream does not match its feature contract")
        if frame_mask.shape != (time,) or not np.all(frame_mask):
            raise ValueError("an unbatched motion graph must mark every frame valid")
        for name, value in (
            ("rotation_valid", rotation_valid),
            ("position_valid", position_valid),
            ("rotation_confidence", rotation_confidence),
            ("position_confidence", position_confidence),
        ):
            if value.shape != (time, joints):
                raise ValueError(f"{name} must have shape [T,J]")
        for confidence, valid in (
            (rotation_confidence, rotation_valid),
            (position_confidence, position_valid),
        ):
            if np.any((confidence < 0.0) | (confidence > 1.0)) or np.any(confidence[~valid] != 0.0):
                raise ValueError("modality confidence must be in [0,1] and zero when invalid")
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "dynamic_universal", dynamic_universal)
        object.__setattr__(self, "dynamic_rig", dynamic_rig)
        object.__setattr__(self, "global_dynamic_universal", global_dynamic)
        object.__setattr__(self, "frame_mask", frame_mask)
        object.__setattr__(self, "rotation_valid", rotation_valid)
        object.__setattr__(self, "position_valid", position_valid)
        object.__setattr__(self, "rotation_confidence", rotation_confidence)
        object.__setattr__(self, "position_confidence", position_confidence)

    @property
    def num_frames(self) -> int:
        return int(self.dynamic_universal.shape[0])


@dataclass(frozen=True)
class VariableRigBatch:
    """Zero-padded NumPy batch with explicit frame, joint, node, and edge masks."""

    version: str
    static_universal: NDArray[np.float32]
    static_rig: NDArray[np.float32]
    dynamic_universal: NDArray[np.float32]
    dynamic_rig: NDArray[np.float32]
    global_dynamic_universal: NDArray[np.float32]
    joint_mask: NDArray[np.bool_]
    model_joint_mask: NDArray[np.bool_]
    frame_mask: NDArray[np.bool_]
    node_frame_mask: NDArray[np.bool_]
    edge_index: NDArray[np.int32]
    edge_mask: NDArray[np.bool_]
    anatomical_edge_index: NDArray[np.int32]
    anatomical_edge_mask: NDArray[np.bool_]
    rotation_valid: NDArray[np.bool_]
    position_valid: NDArray[np.bool_]
    rotation_confidence: NDArray[np.float32]
    position_confidence: NDArray[np.float32]
    fps: NDArray[np.float32]
    node_keys: tuple[tuple[str, ...], ...]
    skeleton_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version != VARIABLE_RIG_BATCH_VERSION:
            raise ValueError(f"unsupported variable-rig batch version {self.version!r}")
        arrays = {
            "static_universal": (self.static_universal, np.dtype(np.float32)),
            "static_rig": (self.static_rig, np.dtype(np.float32)),
            "dynamic_universal": (self.dynamic_universal, np.dtype(np.float32)),
            "dynamic_rig": (self.dynamic_rig, np.dtype(np.float32)),
            "global_dynamic_universal": (
                self.global_dynamic_universal,
                np.dtype(np.float32),
            ),
            "joint_mask": (self.joint_mask, np.dtype(np.bool_)),
            "model_joint_mask": (self.model_joint_mask, np.dtype(np.bool_)),
            "frame_mask": (self.frame_mask, np.dtype(np.bool_)),
            "node_frame_mask": (self.node_frame_mask, np.dtype(np.bool_)),
            "edge_index": (self.edge_index, np.dtype(np.int32)),
            "edge_mask": (self.edge_mask, np.dtype(np.bool_)),
            "anatomical_edge_index": (self.anatomical_edge_index, np.dtype(np.int32)),
            "anatomical_edge_mask": (self.anatomical_edge_mask, np.dtype(np.bool_)),
            "rotation_valid": (self.rotation_valid, np.dtype(np.bool_)),
            "position_valid": (self.position_valid, np.dtype(np.bool_)),
            "rotation_confidence": (self.rotation_confidence, np.dtype(np.float32)),
            "position_confidence": (self.position_confidence, np.dtype(np.float32)),
            "fps": (self.fps, np.dtype(np.float32)),
        }
        frozen: dict[str, NDArray[np.generic]] = {}
        for name, (value, dtype) in arrays.items():
            frozen[name] = frozen_array(value, dtype=dtype, name=name)
        batch, joints = frozen["joint_mask"].shape
        if batch < 1 or joints < 1 or frozen["frame_mask"].shape[0] != batch:
            raise ValueError("batch masks must describe a nonempty [B,J] batch")
        time = frozen["frame_mask"].shape[1]
        expected_shapes = {
            "static_universal": (batch, joints, STATIC_UNIVERSAL_CONTRACT.width),
            "static_rig": (batch, joints, STATIC_RIG_CONTRACT.width),
            "dynamic_universal": (batch, time, joints, DYNAMIC_UNIVERSAL_CONTRACT.width),
            "dynamic_rig": (batch, time, joints, DYNAMIC_RIG_CONTRACT.width),
            "global_dynamic_universal": (
                batch,
                time,
                GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT.width,
            ),
            "model_joint_mask": (batch, joints),
            "node_frame_mask": (batch, time, joints),
            "rotation_valid": (batch, time, joints),
            "position_valid": (batch, time, joints),
            "rotation_confidence": (batch, time, joints),
            "position_confidence": (batch, time, joints),
            "fps": (batch,),
        }
        for name, shape in expected_shapes.items():
            if frozen[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if frozen["edge_index"].ndim != 3 or frozen["edge_index"].shape[:2] != (batch, 2):
            raise ValueError("edge_index must have shape [B,2,E]")
        if frozen["edge_mask"].shape != (batch, frozen["edge_index"].shape[2]):
            raise ValueError("edge_mask must match edge_index")
        if (
            frozen["anatomical_edge_index"].ndim != 3
            or frozen["anatomical_edge_index"].shape[:2] != (batch, 2)
            or frozen["anatomical_edge_mask"].shape
            != (batch, frozen["anatomical_edge_index"].shape[2])
        ):
            raise ValueError("anatomical edge arrays must have matching [B,2,E] data/masks")
        checked_frame_mask = np.asarray(frozen["frame_mask"], dtype=np.bool_)
        checked_joint_mask = np.asarray(frozen["joint_mask"], dtype=np.bool_)
        checked_model_mask = np.asarray(frozen["model_joint_mask"], dtype=np.bool_)
        expected_node_mask = checked_frame_mask[:, :, None] & checked_joint_mask[:, None, :]
        if not np.array_equal(frozen["node_frame_mask"], expected_node_mask):
            raise ValueError("node_frame_mask must be the frame/joint mask outer product")
        if np.any(checked_model_mask & ~checked_joint_mask):
            raise ValueError("padded joints cannot be model joints")
        checked_node_mask = np.asarray(frozen["node_frame_mask"], dtype=np.bool_)
        if np.any(np.asarray(frozen["static_universal"])[~checked_joint_mask] != 0.0) or np.any(
            np.asarray(frozen["static_rig"])[~checked_joint_mask] != 0.0
        ):
            raise ValueError("padded static features must be zero")
        if np.any(np.asarray(frozen["dynamic_universal"])[~checked_node_mask] != 0.0) or np.any(
            np.asarray(frozen["dynamic_rig"])[~checked_node_mask] != 0.0
        ):
            raise ValueError("padded dynamic features must be zero")
        if np.any(np.asarray(frozen["global_dynamic_universal"])[~checked_frame_mask] != 0.0):
            raise ValueError("padded global dynamic features must be zero")
        for validity_name in ("rotation_valid", "position_valid"):
            validity = np.asarray(frozen[validity_name], dtype=np.bool_)
            if np.any(validity & ~checked_node_mask):
                raise ValueError("padded modality validity must be false")
        for confidence_name in ("rotation_confidence", "position_confidence"):
            confidence = np.asarray(frozen[confidence_name], dtype=np.float32)
            if np.any(confidence[~checked_node_mask] != 0.0):
                raise ValueError("padded modality confidence must be zero")
        checked_edge_mask = np.asarray(frozen["edge_mask"], dtype=np.bool_)
        checked_edge_index = np.asarray(frozen["edge_index"], dtype=np.int32)
        checked_anatomical_edge_mask = np.asarray(frozen["anatomical_edge_mask"], dtype=np.bool_)
        checked_anatomical_edge_index = np.asarray(frozen["anatomical_edge_index"], dtype=np.int32)
        if np.any(checked_edge_index.transpose(0, 2, 1)[~checked_edge_mask] != -1) or np.any(
            checked_anatomical_edge_index.transpose(0, 2, 1)[~checked_anatomical_edge_mask] != -1
        ):
            raise ValueError("padded edge indices must use the -1 sentinel")
        if len(self.node_keys) != batch or len(self.skeleton_hashes) != batch:
            raise ValueError("batch identity metadata must match batch size")
        for name, value in frozen.items():
            object.__setattr__(self, name, value)

    @property
    def batch_size(self) -> int:
        return int(self.joint_mask.shape[0])


_ROLE_ORDER = (
    "pelvis",
    "root",
    "spine",
    "spine_1",
    "spine_2",
    "spine_3",
    "chest",
    "upper_chest",
    "neck",
    "head",
    "left_clavicle",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_clavicle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_thigh",
    "left_knee",
    "left_ankle",
    "left_foot",
    "left_toe",
    "right_hip",
    "right_thigh",
    "right_knee",
    "right_ankle",
    "right_foot",
    "right_toe",
)
_ROLE_RANK = {role: index for index, role in enumerate(_ROLE_ORDER)}


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized or "unnamed"


def _node_keys(roles: Sequence[str], names: Sequence[str]) -> tuple[str, ...]:
    values = [
        f"role:{role}" if role != "UNKNOWN" else f"joint:{_slug(name)}"
        for role, name in zip(roles, names, strict=True)
    ]
    if len(set(values)) != len(values):
        values = [f"{value}:{_slug(name)}" for value, name in zip(values, names, strict=True)]
    if len(set(values)) != len(values):
        raise ValueError("unable to construct unique canonical graph node keys")
    return tuple(values)


def _nearest_semantic_part(
    joint: int,
    parents: NDArray[np.int32],
    children: Sequence[Sequence[int]],
    explicit: NDArray[np.int8],
) -> int:
    queue: list[tuple[int, int, int]] = [(joint, 0, 0)]
    visited = {joint}
    candidates: list[tuple[int, int, int]] = []
    while queue and not candidates:
        current, distance, direction_priority = queue.pop(0)
        if current != joint and explicit[current] >= 0:
            candidates.append((distance, direction_priority, int(explicit[current])))
            continue
        next_distance = distance + 1
        for child in children[current]:
            if child not in visited:
                visited.add(child)
                queue.append((child, next_distance, 0))
        parent = int(parents[current])
        if parent >= 0 and parent not in visited:
            visited.add(parent)
            queue.append((parent, next_distance, 1))
    if not candidates:
        return int(FunctionalPart.OTHER_ACCESSORY)
    return min(candidates)[2]


def canonical_anatomical_parts(
    skeleton: Skeleton,
    semantic_roles: Sequence[str],
) -> NDArray[np.int8]:
    """Assign every joint a canonical part, propagating semantics to helper joints."""
    if len(semantic_roles) != skeleton.num_joints:
        raise ValueError("semantic_roles must match skeleton joint count")
    from motionlab.rigging.semantic_roles import functional_part_for_role

    explicit = np.full(skeleton.num_joints, -1, dtype=np.int8)
    for joint, role in enumerate(semantic_roles):
        if role != "UNKNOWN":
            explicit[joint] = int(functional_part_for_role(role))
    children: list[list[int]] = [[] for _ in range(skeleton.num_joints)]
    for child, parent in enumerate(skeleton.parents):
        if parent >= 0:
            children[int(parent)].append(child)
    result = explicit.copy()
    for raw_joint in np.flatnonzero(result < 0):
        joint_index = int(raw_joint)
        result[joint_index] = _nearest_semantic_part(
            joint_index,
            skeleton.parents,
            children,
            explicit,
        )
    return result


def _canonical_anatomical_sides(
    semantic_roles: Sequence[str],
    parts: NDArray[np.int8],
    declared_sides: NDArray[np.int8],
) -> NDArray[np.int8]:
    result = declared_sides.copy()
    for joint, (role, part) in enumerate(zip(semantic_roles, parts, strict=True)):
        if role != "UNKNOWN":
            continue
        functional_part = FunctionalPart(int(part))
        if functional_part in {FunctionalPart.LEFT_ARM, FunctionalPart.LEFT_LEG}:
            result[joint] = int(JointSide.LEFT)
        elif functional_part in {FunctionalPart.RIGHT_ARM, FunctionalPart.RIGHT_LEG}:
            result[joint] = int(JointSide.RIGHT)
        elif functional_part in {
            FunctionalPart.ROOT_PELVIS,
            FunctionalPart.TRUNK_SPINE,
            FunctionalPart.HEAD_NECK,
        }:
            result[joint] = int(JointSide.CENTER)
    return result


def _canonical_order(
    skeleton: Skeleton,
    roles: Sequence[str],
    parts: NDArray[np.int8],
    sides: NDArray[np.int8],
    keys: Sequence[str],
) -> NDArray[np.int32]:
    def canonical_key(joint: int) -> tuple[int, int, int, str]:
        role = roles[joint]
        return (
            _ROLE_RANK.get(role, len(_ROLE_RANK)),
            int(parts[joint]),
            int(sides[joint]),
            keys[joint],
        )

    # Canonical anatomical order is deliberately not a DFS order.  Otherwise inserting an
    # ignorable helper above a semantic joint would move that joint and every later sibling.
    # Parent relationships remain explicit in ``parents``/``edge_index`` and need not be
    # encoded redundantly through array order.
    return np.asarray(sorted(range(skeleton.num_joints), key=canonical_key), dtype=np.int32)


def _read_local_basis(
    skeleton: Skeleton,
    fallback_confidence: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], str]:
    identity = np.zeros((skeleton.num_joints, 4), dtype=np.float32)
    identity[:, 0] = 1.0
    raw = skeleton.metadata.get("local_axis_basis")
    if raw is None:
        return identity, fallback_confidence.copy(), "implicit_declared_basis"
    if not isinstance(raw, dict):
        raise ValueError("local_axis_basis metadata must be an object")
    if raw.get("convention") != LOCAL_BASIS_CONVENTION:
        raise ValueError(f"local_axis_basis convention must be {LOCAL_BASIS_CONVENTION!r}")
    raw_quaternions = raw.get("declared_to_reference_quat_wxyz")
    if isinstance(raw_quaternions, dict):
        quaternions = np.stack(
            [
                np.asarray(raw_quaternions.get(name, identity[index]))
                for index, name in enumerate(skeleton.joint_names)
            ]
        )
    else:
        quaternions = np.asarray(raw_quaternions, dtype=np.float32)
    if quaternions.shape != (skeleton.num_joints, 4):
        raise ValueError("declared_to_reference_quat_wxyz must have shape [J,4]")
    raw_confidence = raw.get("confidence")
    if raw_confidence is None:
        confidence = fallback_confidence.copy()
    elif isinstance(raw_confidence, dict):
        confidence = np.asarray(
            [
                raw_confidence.get(name, fallback_confidence[index])
                for index, name in enumerate(skeleton.joint_names)
            ],
            dtype=np.float32,
        )
    elif np.asarray(raw_confidence).ndim == 0:
        confidence = np.full(skeleton.num_joints, float(raw_confidence), dtype=np.float32)
    else:
        confidence = np.asarray(raw_confidence, dtype=np.float32)
    if confidence.shape != (skeleton.num_joints,):
        raise ValueError("local_axis_basis confidence must be scalar, name mapping, or [J]")
    return quaternions.astype(np.float32), confidence, str(raw.get("source", "declared_metadata"))


def _reference_local_rotations(
    declared: NDArray[np.float32],
    parents: NDArray[np.int32],
    declared_to_reference: NDArray[np.float32],
) -> NDArray[np.float64]:
    result = np.empty(declared.shape, dtype=np.float64)
    for joint, parent in enumerate(parents):
        child_basis = declared_to_reference[joint]
        if parent < 0:
            result[..., joint, :] = quaternion_multiply(declared[..., joint, :], child_basis)
        else:
            result[..., joint, :] = quaternion_multiply(
                quaternion_multiply(
                    quaternion_inverse(declared_to_reference[int(parent)]),
                    declared[..., joint, :],
                ),
                child_basis,
            )
    return result


def _reference_rest_offsets(
    skeleton: Skeleton,
    declared_to_reference: NDArray[np.float32],
) -> NDArray[np.float64]:
    result = skeleton.rest_offsets_m.astype(np.float64).copy()
    for joint, parent in enumerate(skeleton.parents):
        if parent < 0:
            # A root offset is already folded into MotionClip.root_translation_m by importers.
            # It has no parent-local meaning in a graph and is canonicalized to zero.
            result[joint] = 0.0
            continue
        result[joint] = quaternion_rotate_vector(
            quaternion_inverse(declared_to_reference[int(parent)]),
            result[joint],
        )
    return result


def _one_hot(index: NDArray[np.int8], count: int) -> NDArray[np.float32]:
    result = np.zeros((index.shape[0], count), dtype=np.float32)
    result[np.arange(index.shape[0]), index.astype(np.int64)] = 1.0
    return result


def _adapter_from_source(
    source: Skeleton | MotionClip | ClipMetadataAdapter,
) -> ClipMetadataAdapter:
    if isinstance(source, ClipMetadataAdapter):
        return source
    if isinstance(source, MotionClip):
        return adapt_motion_clip(source)
    if isinstance(source, Skeleton):
        rest_clip = MotionClip(
            skeleton=source,
            local_quat_wxyz=source.rest_local_quat_wxyz[None, :, :],
            root_translation_m=np.zeros((1, 3), dtype=np.float32),
            fps=1.0,
        )
        return adapt_motion_clip(rest_clip)
    raise TypeError("source must be a Skeleton, MotionClip, or ClipMetadataAdapter")


def build_skeleton_graph(
    source: Skeleton | MotionClip | ClipMetadataAdapter,
) -> SkeletonGraph:
    """Build a permutation-canonical graph and explicit anatomical projection."""
    adapter = _adapter_from_source(source)
    skeleton = adapter.clip.skeleton
    joint_metadata = adapter.joints
    roles = joint_metadata.semantic_role
    parts = canonical_anatomical_parts(skeleton, roles)
    sides = _canonical_anatomical_sides(roles, parts, joint_metadata.side)
    keys = _node_keys(roles, skeleton.joint_names)
    order = _canonical_order(skeleton, roles, parts, sides, keys)
    inverse = np.empty(skeleton.num_joints, dtype=np.int32)
    inverse[order] = np.arange(skeleton.num_joints, dtype=np.int32)
    parents = np.asarray(
        [
            -1 if skeleton.parents[source_joint] < 0 else inverse[skeleton.parents[source_joint]]
            for source_joint in order
        ],
        dtype=np.int32,
    )
    edge_index = np.stack(
        (
            parents[parents >= 0],
            np.flatnonzero(parents >= 0).astype(np.int32),
        )
    )
    model_mask_source = joint_metadata.critic_joint_mask
    model_mask = model_mask_source[order]
    anatomical_parent = np.full(skeleton.num_joints, -1, dtype=np.int32)
    for joint in range(skeleton.num_joints):
        parent = int(parents[joint])
        while parent >= 0 and not model_mask[parent]:
            parent = int(parents[parent])
        anatomical_parent[joint] = parent
    anatomical_children = np.flatnonzero(model_mask & (anatomical_parent >= 0)).astype(np.int32)
    anatomical_edge_index = np.stack(
        (anatomical_parent[anatomical_children], anatomical_children)
    ).astype(np.int32)

    basis_source, basis_confidence_source, basis_source_name = _read_local_basis(
        skeleton,
        joint_metadata.axis_confidence,
    )
    reference_rest = _reference_local_rotations(
        skeleton.rest_local_quat_wxyz,
        skeleton.parents,
        basis_source,
    )
    reference_offsets = _reference_rest_offsets(skeleton, basis_source)
    rest_position, _ = forward_kinematics_numpy(
        skeleton,
        skeleton.rest_local_quat_wxyz,
        np.zeros(3, dtype=np.float32),
    )
    root_position = rest_position[skeleton.root_index]
    rest_relative = rest_position - root_position
    bone_vector = np.zeros_like(rest_position)
    for joint, parent in enumerate(skeleton.parents):
        if parent >= 0:
            bone_vector[joint] = rest_position[joint] - rest_position[int(parent)]
    bone_length = np.linalg.norm(bone_vector, axis=-1, keepdims=True)
    static_universal_source = np.concatenate(
        (
            rest_relative,
            bone_vector,
            bone_length,
            _one_hot(parts, len(FunctionalPart)),
            _one_hot(sides, len(JointSide)),
            joint_metadata.role_confidence[:, None],
            joint_metadata.is_deform_joint[:, None],
            joint_metadata.is_helper_joint[:, None],
            joint_metadata.is_controller[:, None],
            joint_metadata.is_end_site[:, None],
            joint_metadata.critic_joint_mask[:, None],
        ),
        axis=-1,
        dtype=np.float64,
    ).astype(np.float32)
    static_rig_source = np.concatenate(
        (
            reference_offsets,
            matrix_to_rotation_6d(quaternion_to_matrix(reference_rest)),
            basis_confidence_source[:, None],
        ),
        axis=-1,
    ).astype(np.float32)
    basis = CoordinateBasisMetadata(
        coordinates=adapter.coordinates,
        declared_to_reference_quat_wxyz=basis_source[order],
        confidence=basis_confidence_source[order],
        source=basis_source_name,
    )
    return SkeletonGraph(
        version=SKELETON_GRAPH_VERSION,
        joint_names=tuple(skeleton.joint_names[index] for index in order),
        node_keys=tuple(keys[index] for index in order),
        semantic_roles=tuple(roles[index] for index in order),
        role_aliases=tuple(joint_metadata.role_aliases[index] for index in order),
        parents=parents,
        edge_index=edge_index,
        anatomical_parent=anatomical_parent,
        anatomical_edge_index=anatomical_edge_index,
        source_joint_index=order,
        source_to_canonical_index=inverse,
        functional_part=parts[order],
        side=sides[order],
        is_deform_joint=joint_metadata.is_deform_joint[order],
        is_helper_joint=joint_metadata.is_helper_joint[order],
        is_controller=joint_metadata.is_controller[order],
        is_end_site=joint_metadata.is_end_site[order],
        model_joint_mask=model_mask,
        static_universal=static_universal_source[order],
        static_rig=static_rig_source[order],
        basis=basis,
    )


def _angular_velocity(
    quaternion: NDArray[np.float64],
    fps: float,
) -> NDArray[np.float64]:
    if quaternion.shape[0] == 1:
        return np.zeros((*quaternion.shape[:2], 3), dtype=np.float64)
    interval = quaternion_interval_angular_velocity(
        quaternion,
        np.arange(quaternion.shape[0], dtype=np.float64) / fps,
        time_axis=0,
    )
    result = np.empty((*quaternion.shape[:2], 3), dtype=np.float64)
    result[:-1] = interval
    result[-1] = result[-2]
    return result


def _world_angular_velocity(
    global_rotation: NDArray[np.float64],
    fps: float,
) -> NDArray[np.float64]:
    local_velocity = _angular_velocity(global_rotation, fps)
    return quaternion_rotate_vector(global_rotation, local_velocity)


def build_motion_graph(
    clip: MotionClip,
    *,
    adapter: ClipMetadataAdapter | None = None,
) -> MotionGraph:
    """Encode physical/universal and basis-normalized local streams separately."""
    if adapter is None:
        adapter = adapt_motion_clip(clip)
    elif adapter.clip.content_hash != clip.content_hash:
        raise ValueError("adapter must wrap the supplied clip exactly")
    skeleton_graph = build_skeleton_graph(adapter)
    source_order = skeleton_graph.source_joint_index
    position, global_rotation = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    root = clip.skeleton.root_index
    position_relative = position - position[:, root : root + 1]
    linear_velocity = finite_difference(position, 1.0 / clip.fps, time_axis=0)
    angular_velocity_world = _world_angular_velocity(global_rotation, clip.fps)
    dynamic_universal_source = np.concatenate(
        (position_relative, linear_velocity, angular_velocity_world),
        axis=-1,
    ).astype(np.float32)

    basis_source_order = np.empty_like(skeleton_graph.basis.declared_to_reference_quat_wxyz)
    basis_source_order[source_order] = skeleton_graph.basis.declared_to_reference_quat_wxyz
    reference_local = _reference_local_rotations(
        clip.local_quat_wxyz,
        clip.skeleton.parents,
        basis_source_order,
    )
    local_velocity = _angular_velocity(reference_local, clip.fps)
    dynamic_rig_source = np.concatenate(
        (
            matrix_to_rotation_6d(quaternion_to_matrix(reference_local)),
            local_velocity,
        ),
        axis=-1,
    ).astype(np.float32)
    root_velocity = finite_difference(clip.root_translation_m, 1.0 / clip.fps, time_axis=0)
    global_dynamic = np.concatenate(
        (
            clip.root_translation_m,
            root_velocity,
            clip.timestamps_s[:, None],
        ),
        axis=-1,
    ).astype(np.float32)
    return MotionGraph(
        version=MOTION_GRAPH_VERSION,
        skeleton=skeleton_graph,
        fps=clip.fps,
        dynamic_universal=dynamic_universal_source[:, source_order],
        dynamic_rig=dynamic_rig_source[:, source_order],
        global_dynamic_universal=global_dynamic,
        frame_mask=np.ones(clip.num_frames, dtype=np.bool_),
        rotation_valid=adapter.modalities.rotation_valid[:, source_order],
        position_valid=adapter.modalities.position_valid[:, source_order],
        rotation_confidence=adapter.modalities.rotation_confidence[:, source_order],
        position_confidence=adapter.modalities.position_confidence[:, source_order],
    )


def batch_motion_graphs(graphs: Sequence[MotionGraph]) -> VariableRigBatch:
    """Pad arbitrary frame/joint counts while keeping every invalid value masked."""
    items = tuple(graphs)
    if not items:
        raise ValueError("at least one motion graph is required")
    batch = len(items)
    max_time = max(graph.num_frames for graph in items)
    max_joints = max(graph.skeleton.num_joints for graph in items)
    max_edges = max(graph.skeleton.edge_index.shape[1] for graph in items)
    max_anatomical_edges = max(graph.skeleton.anatomical_edge_index.shape[1] for graph in items)
    static_universal = np.zeros(
        (batch, max_joints, STATIC_UNIVERSAL_CONTRACT.width),
        dtype=np.float32,
    )
    static_rig = np.zeros((batch, max_joints, STATIC_RIG_CONTRACT.width), dtype=np.float32)
    dynamic_universal = np.zeros(
        (batch, max_time, max_joints, DYNAMIC_UNIVERSAL_CONTRACT.width),
        dtype=np.float32,
    )
    dynamic_rig = np.zeros(
        (batch, max_time, max_joints, DYNAMIC_RIG_CONTRACT.width),
        dtype=np.float32,
    )
    global_dynamic = np.zeros(
        (batch, max_time, GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT.width),
        dtype=np.float32,
    )
    joint_mask = np.zeros((batch, max_joints), dtype=np.bool_)
    model_joint_mask = np.zeros_like(joint_mask)
    frame_mask = np.zeros((batch, max_time), dtype=np.bool_)
    edge_index = np.full((batch, 2, max_edges), -1, dtype=np.int32)
    edge_mask = np.zeros((batch, max_edges), dtype=np.bool_)
    anatomical_edge_index = np.full(
        (batch, 2, max_anatomical_edges),
        -1,
        dtype=np.int32,
    )
    anatomical_edge_mask = np.zeros((batch, max_anatomical_edges), dtype=np.bool_)
    rotation_valid = np.zeros((batch, max_time, max_joints), dtype=np.bool_)
    position_valid = np.zeros_like(rotation_valid)
    rotation_confidence = np.zeros((batch, max_time, max_joints), dtype=np.float32)
    position_confidence = np.zeros_like(rotation_confidence)
    fps = np.empty(batch, dtype=np.float32)
    for batch_index, graph in enumerate(items):
        time = graph.num_frames
        joints = graph.skeleton.num_joints
        edges = graph.skeleton.edge_index.shape[1]
        anatomical_edges = graph.skeleton.anatomical_edge_index.shape[1]
        static_universal[batch_index, :joints] = graph.skeleton.static_universal
        static_rig[batch_index, :joints] = graph.skeleton.static_rig
        dynamic_universal[batch_index, :time, :joints] = graph.dynamic_universal
        dynamic_rig[batch_index, :time, :joints] = graph.dynamic_rig
        global_dynamic[batch_index, :time] = graph.global_dynamic_universal
        joint_mask[batch_index, :joints] = True
        model_joint_mask[batch_index, :joints] = graph.skeleton.model_joint_mask
        frame_mask[batch_index, :time] = True
        edge_index[batch_index, :, :edges] = graph.skeleton.edge_index
        edge_mask[batch_index, :edges] = True
        anatomical_edge_index[batch_index, :, :anatomical_edges] = (
            graph.skeleton.anatomical_edge_index
        )
        anatomical_edge_mask[batch_index, :anatomical_edges] = True
        rotation_valid[batch_index, :time, :joints] = graph.rotation_valid
        position_valid[batch_index, :time, :joints] = graph.position_valid
        rotation_confidence[batch_index, :time, :joints] = graph.rotation_confidence
        position_confidence[batch_index, :time, :joints] = graph.position_confidence
        fps[batch_index] = graph.fps
    node_frame_mask = frame_mask[:, :, None] & joint_mask[:, None, :]
    return VariableRigBatch(
        version=VARIABLE_RIG_BATCH_VERSION,
        static_universal=static_universal,
        static_rig=static_rig,
        dynamic_universal=dynamic_universal,
        dynamic_rig=dynamic_rig,
        global_dynamic_universal=global_dynamic,
        joint_mask=joint_mask,
        model_joint_mask=model_joint_mask,
        frame_mask=frame_mask,
        node_frame_mask=node_frame_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        anatomical_edge_index=anatomical_edge_index,
        anatomical_edge_mask=anatomical_edge_mask,
        rotation_valid=rotation_valid,
        position_valid=position_valid,
        rotation_confidence=rotation_confidence,
        position_confidence=position_confidence,
        fps=fps,
        node_keys=tuple(graph.skeleton.node_keys for graph in items),
        skeleton_hashes=tuple(graph.skeleton.content_hash for graph in items),
    )


__all__ = [
    "DYNAMIC_RIG_CONTRACT",
    "DYNAMIC_UNIVERSAL_CONTRACT",
    "GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT",
    "LOCAL_BASIS_CONVENTION",
    "MOTION_GRAPH_VERSION",
    "SKELETON_GRAPH_VERSION",
    "STATIC_RIG_CONTRACT",
    "STATIC_UNIVERSAL_CONTRACT",
    "VARIABLE_RIG_BATCH_VERSION",
    "CoordinateBasisMetadata",
    "FeatureContract",
    "FeatureField",
    "MotionGraph",
    "SkeletonGraph",
    "VariableRigBatch",
    "batch_motion_graphs",
    "build_motion_graph",
    "build_skeleton_graph",
    "canonical_anatomical_parts",
]
