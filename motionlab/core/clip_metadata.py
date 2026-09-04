"""Lossless forward-metadata adapter for the current fixed-rig clip API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.core.coordinate_contract import CoordinateContract
from motionlab.core.provenance import Provenance, provenance_from_clip
from motionlab.motion._validation import frozen_array
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import ContactResult, normalize_ground_plane
from motionlab.rigging.semantic_roles import functional_part_for_role, side_for_role


def _float_array(value: ArrayLike, *, name: str) -> NDArray[np.float32]:
    return frozen_array(value, dtype=np.dtype(np.float32), name=name)


def _bool_array(value: ArrayLike, *, name: str) -> NDArray[np.bool_]:
    array = np.array(value, dtype=np.bool_, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PoseReferenceMetadata:
    """Distinguish rest, bind, and optional anatomical/reference anchors."""

    joint_count: int
    rest_kind: str
    rest_confidence: float
    bind_available: bool
    bind_confidence: float
    bind_local_quat_wxyz: NDArray[np.float32] | None
    pose_anchor_kind: str | None
    pose_anchor_confidence: float
    pose_anchor_position_m: NDArray[np.float32] | None

    def __post_init__(self) -> None:
        if not self.rest_kind.strip():
            raise ValueError("rest_kind must be nonempty")
        if self.joint_count < 1:
            raise ValueError("joint_count must be positive")
        if self.pose_anchor_kind is not None and not self.pose_anchor_kind.strip():
            raise ValueError("pose_anchor_kind must be nonempty when supplied")
        for name in ("rest_confidence", "bind_confidence", "pose_anchor_confidence"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0,1]")
            object.__setattr__(self, name, value)
        if not self.bind_available and self.bind_confidence != 0.0:
            raise ValueError("unavailable bind data must have zero confidence")
        if self.pose_anchor_kind is None and self.pose_anchor_confidence != 0.0:
            raise ValueError("missing pose anchor must have zero confidence")
        bind: NDArray[np.float32] | None = None
        if self.bind_local_quat_wxyz is not None:
            bind = _float_array(self.bind_local_quat_wxyz, name="bind_local_quat_wxyz")
            if bind.shape != (self.joint_count, 4):
                raise ValueError("bind_local_quat_wxyz must have shape [J,4]")
            if not np.allclose(np.linalg.norm(bind, axis=-1), 1.0, atol=1.0e-5, rtol=0.0):
                raise ValueError("bind_local_quat_wxyz must contain unit quaternions")
        if self.bind_available != (bind is not None):
            raise ValueError("bind availability must match embedded bind rotations")
        anchor: NDArray[np.float32] | None = None
        if self.pose_anchor_position_m is not None:
            anchor = _float_array(self.pose_anchor_position_m, name="pose_anchor_position_m")
            if anchor.shape != (self.joint_count, 3):
                raise ValueError("pose_anchor_position_m must have shape [J,3]")
        if (self.pose_anchor_kind is None) != (anchor is None):
            raise ValueError("pose anchor kind and positions must be supplied together")
        object.__setattr__(self, "bind_local_quat_wxyz", bind)
        object.__setattr__(self, "pose_anchor_position_m", anchor)


@dataclass(frozen=True)
class JointMetadata:
    """Fixed anatomical semantics, confidence, and model/repair masks for every joint."""

    semantic_role: tuple[str, ...]
    role_aliases: tuple[tuple[str, ...], ...]
    functional_part: NDArray[np.int8]
    side: NDArray[np.int8]
    role_confidence: NDArray[np.float32]
    axis_confidence: NDArray[np.float32]
    is_deform_joint: NDArray[np.bool_]
    is_helper_joint: NDArray[np.bool_]
    is_controller: NDArray[np.bool_]
    is_end_site: NDArray[np.bool_]
    critic_joint_mask: NDArray[np.bool_]
    repair_joint_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        roles = tuple(self.semantic_role)
        aliases = tuple(tuple(value) for value in self.role_aliases)
        joint_count = len(roles)
        if len(aliases) != joint_count or any(not role for role in roles):
            raise ValueError("semantic roles and aliases must match a nonempty joint list")
        functional_part = _floatless_int8(self.functional_part, name="functional_part")
        side = _floatless_int8(self.side, name="side")
        role_confidence = _float_array(self.role_confidence, name="role_confidence")
        axis_confidence = _float_array(self.axis_confidence, name="axis_confidence")
        flags = {
            name: _bool_array(getattr(self, name), name=name)
            for name in (
                "is_deform_joint",
                "is_helper_joint",
                "is_controller",
                "is_end_site",
                "critic_joint_mask",
                "repair_joint_mask",
            )
        }
        for name, array_value in (
            ("functional_part", functional_part),
            ("side", side),
            ("role_confidence", role_confidence),
            ("axis_confidence", axis_confidence),
        ):
            if array_value.shape != (joint_count,):
                raise ValueError(f"{name} must have shape [J]")
        for name, flag_value in flags.items():
            if flag_value.shape != (joint_count,):
                raise ValueError(f"{name} must have shape [J]")
        for name, value in (
            ("role_confidence", role_confidence),
            ("axis_confidence", axis_confidence),
        ):
            if np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must be within [0,1]")
        if np.any(flags["is_controller"] & flags["is_deform_joint"]):
            raise ValueError("controller joints cannot also be deform joints")
        if np.any(flags["critic_joint_mask"] & flags["is_controller"]):
            raise ValueError("controller joints cannot be active critic joints")
        object.__setattr__(self, "semantic_role", roles)
        object.__setattr__(self, "role_aliases", aliases)
        object.__setattr__(self, "functional_part", functional_part)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "role_confidence", role_confidence)
        object.__setattr__(self, "axis_confidence", axis_confidence)
        for name, flag_array in flags.items():
            object.__setattr__(self, name, flag_array)


def _floatless_int8(value: ArrayLike, *, name: str) -> NDArray[np.int8]:
    array = np.array(value, dtype=np.int8, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ModalityMetadata:
    """Explicit per-frame validity and confidence; missing is never an ordinary zero."""

    rotation_valid: NDArray[np.bool_]
    position_valid: NDArray[np.bool_]
    rotation_confidence: NDArray[np.float32]
    position_confidence: NDArray[np.float32]
    contact_names: tuple[str, ...] = ()
    contact_valid: NDArray[np.bool_] | None = None
    contact_confidence: NDArray[np.float32] | None = None
    contact_source: str | None = None

    def __post_init__(self) -> None:
        rotation_valid = _bool_array(self.rotation_valid, name="rotation_valid")
        position_valid = _bool_array(self.position_valid, name="position_valid")
        rotation_confidence = _float_array(
            self.rotation_confidence,
            name="rotation_confidence",
        )
        position_confidence = _float_array(
            self.position_confidence,
            name="position_confidence",
        )
        if rotation_valid.ndim != 2 or position_valid.shape != rotation_valid.shape:
            raise ValueError("rotation_valid and position_valid must have matching shape [T,J]")
        if (
            rotation_confidence.shape != rotation_valid.shape
            or position_confidence.shape != rotation_valid.shape
        ):
            raise ValueError("modality confidence arrays must match validity shape [T,J]")
        for name, confidence, valid in (
            ("rotation", rotation_confidence, rotation_valid),
            ("position", position_confidence, position_valid),
        ):
            if np.any((confidence < 0.0) | (confidence > 1.0)):
                raise ValueError(f"{name} confidence must be within [0,1]")
            if np.any(confidence[~valid] != 0.0):
                raise ValueError(f"invalid {name} entries must have zero confidence")

        contact_names = tuple(self.contact_names)
        contact_valid: NDArray[np.bool_] | None = None
        contact_confidence: NDArray[np.float32] | None = None
        if self.contact_valid is not None or self.contact_confidence is not None:
            if self.contact_valid is None or self.contact_confidence is None:
                raise ValueError("contact validity and confidence must be supplied together")
            contact_valid = _bool_array(self.contact_valid, name="contact_valid")
            contact_confidence = _float_array(
                self.contact_confidence,
                name="contact_confidence",
            )
            expected = (rotation_valid.shape[0], len(contact_names))
            if contact_valid.shape != expected or contact_confidence.shape != expected:
                raise ValueError("contact arrays must have shape [T, number of contact names]")
            if np.any((contact_confidence < 0.0) | (contact_confidence > 1.0)):
                raise ValueError("contact confidence must be within [0,1]")
            if np.any(contact_confidence[~contact_valid] != 0.0):
                raise ValueError("invalid contact entries must have zero confidence")
            if self.contact_source is None or not self.contact_source.strip():
                raise ValueError("available contact metadata requires a source")
        elif contact_names or self.contact_source is not None:
            raise ValueError("contact names/source require contact validity and confidence")
        object.__setattr__(self, "rotation_valid", rotation_valid)
        object.__setattr__(self, "position_valid", position_valid)
        object.__setattr__(self, "rotation_confidence", rotation_confidence)
        object.__setattr__(self, "position_confidence", position_confidence)
        object.__setattr__(self, "contact_names", contact_names)
        object.__setattr__(self, "contact_valid", contact_valid)
        object.__setattr__(self, "contact_confidence", contact_confidence)


@dataclass(frozen=True)
class GroundMetadata:
    """Per-clip or per-frame normalized ground planes with validity and confidence."""

    frame_count: int
    plane: NDArray[np.float32] | None
    valid: NDArray[np.bool_]
    confidence: NDArray[np.float32]
    source: str

    def __post_init__(self) -> None:
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        valid = _bool_array(self.valid, name="ground valid")
        confidence = _float_array(self.confidence, name="ground confidence")
        if valid.shape != (self.frame_count,) or confidence.shape != valid.shape:
            raise ValueError("ground validity and confidence must have shape [T]")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("ground confidence must be within [0,1]")
        if np.any(confidence[~valid] != 0.0):
            raise ValueError("invalid ground frames must have zero confidence")
        plane: NDArray[np.float32] | None = None
        if self.plane is not None:
            plane = _float_array(self.plane, name="ground plane")
            if plane.shape not in {(1, 4), (self.frame_count, 4)}:
                raise ValueError("ground plane must have shape [1,4] or [T,4]")
            normal_norm = np.linalg.norm(plane[:, :3], axis=-1)
            if not np.allclose(normal_norm, 1.0, atol=1.0e-6, rtol=0.0):
                raise ValueError("ground plane normals must be unit length")
        elif np.any(valid):
            raise ValueError("valid ground frames require a plane")
        if not self.source.strip():
            raise ValueError("ground source must be nonempty")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "plane", plane)


@dataclass(frozen=True)
class ClipMetadataAdapter:
    """Lossless wrapper that exposes future metadata without changing MotionClip."""

    clip: MotionClip
    coordinates: CoordinateContract
    pose_reference: PoseReferenceMetadata
    joints: JointMetadata
    modalities: ModalityMetadata
    ground: GroundMetadata
    provenance: Provenance
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.modalities.rotation_valid.shape != (self.clip.num_frames, self.clip.num_joints):
            raise ValueError("modality metadata does not match wrapped clip")
        if len(self.joints.semantic_role) != self.clip.num_joints:
            raise ValueError("joint metadata does not match wrapped clip")
        if self.ground.frame_count != self.clip.num_frames:
            raise ValueError("ground metadata does not match wrapped clip")
        if self.provenance.motion_id != self.clip.content_hash:
            raise ValueError("provenance motion ID does not match wrapped clip")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_motion_clip(self) -> MotionClip:
        """Return an exact detached fixed-rig clip, preserving its content hash."""
        return self.clip.copy()


def _coordinates(clip: MotionClip) -> CoordinateContract:
    raw = clip.skeleton.metadata.get("source_coordinate_system")
    matrix = clip.skeleton.metadata.get("source_to_canonical_matrix")
    if isinstance(raw, dict) and matrix is not None:
        return CoordinateContract(
            source_handedness=raw["handedness"],
            source_up_axis=str(raw["up_axis"]),
            source_forward_axis=str(raw["forward_axis"]),
            source_unit_scale_to_m=float(raw["unit_scale_to_m"]),
            source_to_canonical_matrix=matrix,
            source="importer_metadata",
            confidence=1.0,
        )
    return CoordinateContract(
        source_handedness="right",
        source_up_axis="Y",
        source_forward_axis="Z",
        source_unit_scale_to_m=1.0,
        source_to_canonical_matrix=np.eye(3),
        source="motionlab_internal_contract",
        confidence=1.0,
    )


def _pose_reference(clip: MotionClip) -> PoseReferenceMetadata:
    metadata = clip.skeleton.metadata
    if metadata.get("source_format") == "BVH":
        return PoseReferenceMetadata(
            joint_count=clip.num_joints,
            rest_kind="bvh_zero_channel_rest",
            rest_confidence=0.25,
            bind_available=False,
            bind_confidence=0.0,
            bind_local_quat_wxyz=None,
            pose_anchor_kind=None,
            pose_anchor_confidence=0.0,
            pose_anchor_position_m=None,
        )
    raw = metadata.get("pose_reference")
    if isinstance(raw, dict):
        return PoseReferenceMetadata(
            joint_count=clip.num_joints,
            rest_kind=str(raw.get("rest_kind", "declared_rest")),
            rest_confidence=float(raw.get("rest_confidence", 1.0)),
            bind_available=raw.get("bind_local_quat_wxyz") is not None,
            bind_confidence=float(raw.get("bind_confidence", 0.0)),
            bind_local_quat_wxyz=raw.get("bind_local_quat_wxyz"),
            pose_anchor_kind=(
                None if raw.get("pose_anchor_kind") is None else str(raw["pose_anchor_kind"])
            ),
            pose_anchor_confidence=float(raw.get("pose_anchor_confidence", 0.0)),
            pose_anchor_position_m=raw.get("pose_anchor_position_m"),
        )
    return PoseReferenceMetadata(
        joint_count=clip.num_joints,
        rest_kind="declared_rest",
        rest_confidence=1.0,
        bind_available=False,
        bind_confidence=0.0,
        bind_local_quat_wxyz=None,
        pose_anchor_kind=None,
        pose_anchor_confidence=0.0,
        pose_anchor_position_m=None,
    )


def _metadata_confidence(raw: Any, *, joint_name: str, role: str, default: float) -> float:
    if isinstance(raw, dict):
        raw = raw.get(role, raw.get(joint_name, default))
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _joint_metadata(clip: MotionClip) -> JointMetadata:
    skeleton = clip.skeleton
    roles_by_joint: list[list[str]] = [[] for _ in range(skeleton.num_joints)]
    for role, joint in skeleton.roles.items():
        roles_by_joint[joint].append(role)
    priority = ("pelvis", "root", "chest", "head")
    primary: list[str] = []
    aliases: list[tuple[str, ...]] = []
    part: list[int] = []
    side: list[int] = []
    role_confidence: list[float] = []
    axis_confidence: list[float] = []
    raw_role_confidence = skeleton.metadata.get("role_confidence")
    raw_axis_confidence = skeleton.metadata.get("local_axis_confidence")
    bvh_default = 0.25 if skeleton.metadata.get("source_format") == "BVH" else 1.0
    raw_flags = skeleton.metadata.get("joint_flags", {})
    if not isinstance(raw_flags, dict):
        raise ValueError("skeleton joint_flags metadata must be an object")

    flag_values: dict[str, list[bool]] = {
        name: []
        for name in (
            "is_deform_joint",
            "is_helper_joint",
            "is_controller",
            "is_end_site",
            "critic_joint_mask",
            "repair_joint_mask",
        )
    }
    for joint, joint_name in enumerate(skeleton.joint_names):
        joint_roles = tuple(sorted(roles_by_joint[joint]))
        aliases.append(joint_roles)
        primary_role: str | None = next(
            (candidate for candidate in priority if candidate in joint_roles),
            None,
        )
        if primary_role is None:
            primary_role = joint_roles[0] if joint_roles else "UNKNOWN"
        primary.append(primary_role)
        part.append(int(functional_part_for_role(primary_role)))
        side.append(int(side_for_role(primary_role)))
        role_confidence.append(
            _metadata_confidence(
                raw_role_confidence,
                joint_name=joint_name,
                role=primary_role,
                default=1.0 if primary_role != "UNKNOWN" else 0.0,
            )
        )
        axis_confidence.append(
            _metadata_confidence(
                raw_axis_confidence,
                joint_name=joint_name,
                role=primary_role,
                default=bvh_default,
            )
        )
        entry = raw_flags.get(joint_name, {})
        if not isinstance(entry, dict):
            raise ValueError(f"joint flags for {joint_name!r} must be an object")
        helper = bool(entry.get("is_helper_joint", False))
        controller = bool(entry.get("is_controller", False))
        end_site = bool(entry.get("is_end_site", False))
        deform = bool(entry.get("is_deform_joint", not (helper or controller or end_site)))
        critic = bool(entry.get("critic_joint_mask", deform and not controller and not end_site))
        repair = bool(entry.get("repair_joint_mask", deform and not controller and not end_site))
        for name, value in (
            ("is_deform_joint", deform),
            ("is_helper_joint", helper),
            ("is_controller", controller),
            ("is_end_site", end_site),
            ("critic_joint_mask", critic),
            ("repair_joint_mask", repair),
        ):
            flag_values[name].append(value)
    return JointMetadata(
        semantic_role=tuple(primary),
        role_aliases=tuple(aliases),
        functional_part=np.asarray(part, dtype=np.int8),
        side=np.asarray(side, dtype=np.int8),
        role_confidence=np.asarray(role_confidence, dtype=np.float32),
        axis_confidence=np.asarray(axis_confidence, dtype=np.float32),
        **{name: np.asarray(value, dtype=np.bool_) for name, value in flag_values.items()},
    )


def _expanded_mask(
    value: ArrayLike | None,
    *,
    default: bool,
    frame_count: int,
    joint_count: int,
    name: str,
) -> NDArray[np.bool_]:
    if value is None:
        return np.full((frame_count, joint_count), default, dtype=np.bool_)
    array = np.asarray(value, dtype=np.bool_)
    if array.shape == (joint_count,):
        array = np.broadcast_to(array, (frame_count, joint_count))
    if array.shape != (frame_count, joint_count):
        raise ValueError(f"{name} must have shape [J] or [T,J]")
    return np.asarray(array, dtype=np.bool_)


def _expanded_confidence(
    value: ArrayLike | None,
    *,
    default: float,
    valid: NDArray[np.bool_],
    name: str,
) -> NDArray[np.float32]:
    if value is None:
        array = np.full(valid.shape, default, dtype=np.float32)
    else:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (valid.shape[1],):
            array = np.broadcast_to(array, valid.shape)
        if array.shape != valid.shape:
            raise ValueError(f"{name} must have shape [J] or [T,J]")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain finite values within [0,1]")
    return np.where(valid, array, 0.0).astype(np.float32)


def _bvh_rotation_valid(clip: MotionClip) -> NDArray[np.bool_] | None:
    channels = clip.skeleton.metadata.get("bvh_channels")
    if not isinstance(channels, list) or len(channels) != clip.num_joints:
        return None
    per_joint = np.asarray(
        [
            isinstance(joint_channels, list)
            and any(str(channel).endswith("rotation") for channel in joint_channels)
            for joint_channels in channels
        ],
        dtype=np.bool_,
    )
    return np.broadcast_to(per_joint, (clip.num_frames, clip.num_joints))


def _ground_metadata(
    clip: MotionClip,
    contacts: ContactResult | None,
    ground_plane: ArrayLike | None,
    ground_source: str | None,
    ground_confidence: ArrayLike | None,
) -> GroundMetadata:
    if contacts is not None:
        return GroundMetadata(
            frame_count=clip.num_frames,
            plane=np.asarray(contacts.ground_plane, dtype=np.float32)[None, :],
            valid=np.ones(clip.num_frames, dtype=np.bool_),
            confidence=np.full(
                clip.num_frames,
                contacts.ground_confidence,
                dtype=np.float32,
            ),
            source=f"contact_result:{contacts.source}",
        )
    raw_plane = ground_plane if ground_plane is not None else clip.metadata.get("ground_plane")
    if raw_plane is None:
        return GroundMetadata(
            frame_count=clip.num_frames,
            plane=None,
            valid=np.zeros(clip.num_frames, dtype=np.bool_),
            confidence=np.zeros(clip.num_frames, dtype=np.float32),
            source="unavailable",
        )
    raw_plane_array = np.asarray(raw_plane, dtype=np.float64)
    if raw_plane_array.shape == (4,):
        plane = normalize_ground_plane(raw_plane_array).astype(np.float32)[None, :]
    elif raw_plane_array.shape in {(1, 4), (clip.num_frames, 4)}:
        plane = np.stack(
            [normalize_ground_plane(frame_plane) for frame_plane in raw_plane_array],
            axis=0,
        ).astype(np.float32)
    else:
        raise ValueError("ground_plane must have shape [4], [1,4], or [T,4]")
    if ground_confidence is None:
        confidence = np.ones(clip.num_frames, dtype=np.float32)
    else:
        confidence = np.asarray(ground_confidence, dtype=np.float32)
        if confidence.ndim == 0:
            confidence = np.full(clip.num_frames, float(confidence), dtype=np.float32)
        if confidence.shape != (clip.num_frames,):
            raise ValueError("ground_confidence must be scalar or have shape [T]")
    source = ground_source or str(clip.metadata.get("ground_plane_source", "clip_metadata"))
    return GroundMetadata(
        frame_count=clip.num_frames,
        plane=plane,
        valid=np.ones(clip.num_frames, dtype=np.bool_),
        confidence=confidence,
        source=source,
    )


def adapt_motion_clip(
    clip: MotionClip,
    *,
    contacts: ContactResult | None = None,
    rotation_valid: ArrayLike | None = None,
    position_valid: ArrayLike | None = None,
    rotation_confidence: ArrayLike | None = None,
    position_confidence: ArrayLike | None = None,
    ground_plane: ArrayLike | None = None,
    ground_source: str | None = None,
    ground_confidence: ArrayLike | None = None,
) -> ClipMetadataAdapter:
    """Expose forward metadata while retaining an exact fixed-rig round-trip."""
    if not isinstance(clip, MotionClip):
        raise TypeError("clip must be a MotionClip")
    default_rotation_valid = _bvh_rotation_valid(clip)
    rotations_valid = _expanded_mask(
        rotation_valid if rotation_valid is not None else default_rotation_valid,
        default=True,
        frame_count=clip.num_frames,
        joint_count=clip.num_joints,
        name="rotation_valid",
    )
    positions_valid = _expanded_mask(
        position_valid,
        default=True,
        frame_count=clip.num_frames,
        joint_count=clip.num_joints,
        name="position_valid",
    )
    rotation_conf = _expanded_confidence(
        rotation_confidence,
        default=1.0,
        valid=rotations_valid,
        name="rotation_confidence",
    )
    position_conf = _expanded_confidence(
        position_confidence,
        default=1.0,
        valid=positions_valid,
        name="position_confidence",
    )
    modalities = ModalityMetadata(
        rotation_valid=rotations_valid,
        position_valid=positions_valid,
        rotation_confidence=rotation_conf,
        position_confidence=position_conf,
        contact_names=() if contacts is None else contacts.marker_names,
        contact_valid=(None if contacts is None else np.ones_like(contacts.hard, dtype=np.bool_)),
        contact_confidence=(None if contacts is None else contacts.confidence),
        contact_source=None if contacts is None else contacts.source,
    )
    return ClipMetadataAdapter(
        clip=clip,
        coordinates=_coordinates(clip),
        pose_reference=_pose_reference(clip),
        joints=_joint_metadata(clip),
        modalities=modalities,
        ground=_ground_metadata(
            clip,
            contacts,
            ground_plane,
            ground_source,
            ground_confidence,
        ),
        provenance=provenance_from_clip(clip),
        metadata={
            "adapter_version": "motionlab.clip_metadata_adapter.v1",
            "neural_input_excludes_provenance": True,
        },
    )
