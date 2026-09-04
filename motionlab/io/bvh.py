"""A strict BVH hierarchy/motion parser and canonical MotionClip adapter."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from motionlab.core.provenance import SplitName, assign_source_lineage
from motionlab.math.quaternion import matrix_to_quaternion
from motionlab.motion.clip import MotionClip
from motionlab.motion.markers import JointLimitSpec, MarkerSpec
from motionlab.motion.skeleton import Skeleton

_TOKEN_PATTERN = re.compile(r"[{}]|[^\s{}]+")
_POSITION_CHANNELS = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
_ROTATION_CHANNELS = {"Xrotation", "Yrotation", "Zrotation"}


class BvhParseError(ValueError):
    """Raised when a BVH file is incomplete or violates the supported grammar."""


@dataclass(frozen=True)
class BvhEndSite:
    """An end-site offset attached to a parsed parent joint."""

    parent: int
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class BvhData:
    """Parsed BVH hierarchy and raw channel samples before coordinate conversion."""

    joint_names: tuple[str, ...]
    parents: NDArray[np.int32]
    offsets: NDArray[np.float64]
    channels: tuple[tuple[str, ...], ...]
    end_sites: tuple[BvhEndSite, ...]
    frame_time_s: float
    frames: NDArray[np.float64]

    @property
    def num_frames(self) -> int:
        """Return the declared number of motion frames."""
        return int(self.frames.shape[0])

    @property
    def num_channels(self) -> int:
        """Return the flattened channel count per frame."""
        return sum(len(channels) for channels in self.channels)


@dataclass(frozen=True)
class BvhImportOptions:
    """Explicit source coordinate and scale information for canonical conversion."""

    handedness: Literal["right", "left"] = "right"
    up_axis: str = "Y"
    forward_axis: str = "Z"
    unit_scale_to_m: float = 1.0

    def __post_init__(self) -> None:
        _axis_vector(self.up_axis)
        _axis_vector(self.forward_axis)
        if abs(float(np.dot(_axis_vector(self.up_axis), _axis_vector(self.forward_axis)))) > 0.0:
            raise ValueError("up_axis and forward_axis must be perpendicular")
        if not np.isfinite(self.unit_scale_to_m) or self.unit_scale_to_m <= 0.0:
            raise ValueError("unit_scale_to_m must be finite and positive")

    @property
    def source_to_canonical_matrix(self) -> NDArray[np.float64]:
        """Return a 3-by-3 vector transform from source axes to MotionLab axes."""
        up = _axis_vector(self.up_axis)
        forward = _axis_vector(self.forward_axis)
        right = np.cross(up, forward)
        if self.handedness == "left":
            right *= -1.0
        return np.stack((right, up, forward), axis=0)


def _axis_vector(axis: str) -> NDArray[np.float64]:
    normalized = axis.strip().upper()
    sign = -1.0 if normalized.startswith("-") else 1.0
    name = normalized.lstrip("+-")
    if name not in {"X", "Y", "Z"}:
        raise ValueError(f"axis must be signed X, Y, or Z; got {axis!r}")
    vector = np.zeros(3, dtype=np.float64)
    vector[{"X": 0, "Y": 1, "Z": 2}[name]] = sign
    return vector


class _Parser:
    def __init__(self, text: str, source: str) -> None:
        self.tokens = cast(list[str], _TOKEN_PATTERN.findall(text))
        self.position = 0
        self.source = source
        self.names: list[str] = []
        self.parents: list[int] = []
        self.offsets: list[tuple[float, float, float]] = []
        self.channels: list[tuple[str, ...]] = []
        self.end_sites: list[BvhEndSite] = []

    def _error(self, message: str) -> BvhParseError:
        nearby = self.tokens[self.position : self.position + 4]
        return BvhParseError(
            f"{self.source}: token {self.position}: {message}; next tokens={nearby!r}"
        )

    def peek(self) -> str:
        if self.position >= len(self.tokens):
            raise self._error("unexpected end of file")
        return self.tokens[self.position]

    def take(self) -> str:
        token = self.peek()
        self.position += 1
        return token

    def expect(self, expected: str) -> None:
        actual = self.take()
        if actual.upper() != expected.upper():
            raise self._error(f"expected {expected!r}, got {actual!r}")

    def take_float(self, label: str) -> float:
        token = self.take()
        try:
            value = float(token)
        except ValueError as exc:
            raise self._error(f"expected numeric {label}, got {token!r}") from exc
        if not np.isfinite(value):
            raise self._error(f"{label} must be finite")
        return value

    def take_int(self, label: str) -> int:
        value = self.take_float(label)
        if not value.is_integer():
            raise self._error(f"{label} must be an integer")
        return int(value)

    def parse(self) -> BvhData:
        self.expect("HIERARCHY")
        self.expect("ROOT")
        root_name = self.take()
        self.parse_joint(root_name, parent=-1)
        self.expect("MOTION")
        frames_label = self.take().rstrip(":").upper()
        if frames_label != "FRAMES":
            raise self._error(f"expected 'Frames:', got {frames_label!r}")
        num_frames = self.take_int("frame count")
        if num_frames < 1:
            raise self._error("frame count must be positive")
        self.expect("FRAME")
        time_label = self.take().rstrip(":").upper()
        if time_label != "TIME":
            raise self._error(f"expected 'Time:', got {time_label!r}")
        frame_time = self.take_float("frame time")
        if frame_time <= 0.0:
            raise self._error("frame time must be positive")

        channel_count = sum(len(channels) for channels in self.channels)
        expected_values = num_frames * channel_count
        remaining = self.tokens[self.position :]
        if len(remaining) != expected_values:
            raise self._error(f"expected {expected_values} motion values, found {len(remaining)}")
        values = np.empty(expected_values, dtype=np.float64)
        for index in range(expected_values):
            values[index] = self.take_float("motion channel")
        frames = values.reshape((num_frames, channel_count))
        return BvhData(
            joint_names=tuple(self.names),
            parents=np.asarray(self.parents, dtype=np.int32),
            offsets=np.asarray(self.offsets, dtype=np.float64),
            channels=tuple(self.channels),
            end_sites=tuple(self.end_sites),
            frame_time_s=frame_time,
            frames=frames,
        )

    def parse_joint(self, name: str, *, parent: int) -> int:
        if not name or name in self.names:
            raise self._error(f"joint name must be nonempty and unique: {name!r}")
        joint = len(self.names)
        self.names.append(name)
        self.parents.append(parent)
        self.offsets.append((0.0, 0.0, 0.0))
        self.channels.append(())

        self.expect("{")
        saw_offset = False
        saw_channels = False
        while True:
            token = self.peek()
            normalized = token.upper()
            if token == "}":
                self.take()
                break
            if normalized == "OFFSET":
                if saw_offset:
                    raise self._error(f"joint {name!r} has multiple OFFSET declarations")
                self.take()
                self.offsets[joint] = (
                    self.take_float("offset X"),
                    self.take_float("offset Y"),
                    self.take_float("offset Z"),
                )
                saw_offset = True
            elif normalized == "CHANNELS":
                if saw_channels:
                    raise self._error(f"joint {name!r} has multiple CHANNELS declarations")
                self.take()
                count = self.take_int("channel count")
                channels = tuple(self.take() for _ in range(count))
                unknown = (
                    set(channels).difference(_POSITION_CHANNELS).difference(_ROTATION_CHANNELS)
                )
                if unknown:
                    raise self._error(f"joint {name!r} has unsupported channels {sorted(unknown)}")
                if len(set(channels)) != len(channels):
                    raise self._error(f"joint {name!r} has duplicate channels")
                self.channels[joint] = channels
                saw_channels = True
            elif normalized == "JOINT":
                self.take()
                self.parse_joint(self.take(), parent=joint)
            elif normalized == "END":
                self.take()
                self.expect("SITE")
                self.parse_end_site(parent=joint)
            else:
                raise self._error(f"unexpected hierarchy token {token!r} in joint {name!r}")
        if not saw_offset or not saw_channels:
            raise self._error(f"joint {name!r} requires OFFSET and CHANNELS")
        return joint

    def parse_end_site(self, *, parent: int) -> None:
        self.expect("{")
        self.expect("OFFSET")
        offset = (
            self.take_float("end-site offset X"),
            self.take_float("end-site offset Y"),
            self.take_float("end-site offset Z"),
        )
        self.expect("}")
        self.end_sites.append(BvhEndSite(parent=parent, offset=offset))


def parse_bvh(text: str, *, source: str = "<memory>") -> BvhData:
    """Parse BVH text while preserving each joint's declared channel order."""
    return _Parser(text, source).parse()


def load_bvh(path: Path) -> BvhData:
    """Read and parse a UTF-8 BVH file."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise BvhParseError(f"unable to read BVH file {source}") from exc
    return parse_bvh(text, source=str(source))


def bvh_to_motion_clip(
    data: BvhData,
    *,
    options: BvhImportOptions | None = None,
    role_names: dict[str, str] | None = None,
    markers: dict[str, MarkerSpec] | None = None,
    joint_limits: dict[str, JointLimitSpec] | None = None,
) -> MotionClip:
    """Convert parsed BVH channels into canonical fixed-rig data."""
    options = BvhImportOptions() if options is None else options
    transform = options.source_to_canonical_matrix
    offsets_m = (data.offsets @ transform.T) * options.unit_scale_to_m
    frame_count = data.num_frames
    joint_count = len(data.joint_names)
    local_quaternions = np.empty((frame_count, joint_count, 4), dtype=np.float32)
    root_positions_source = np.zeros((frame_count, 3), dtype=np.float64)
    cursor = 0
    root = int(np.flatnonzero(data.parents == -1)[0])

    for joint, channels in enumerate(data.channels):
        values = data.frames[:, cursor : cursor + len(channels)]
        cursor += len(channels)
        position_channels = [channel for channel in channels if channel in _POSITION_CHANNELS]
        if position_channels and joint != root:
            raise ValueError(
                f"non-root position channels are unsupported for joint {data.joint_names[joint]!r}"
            )
        for column, channel in enumerate(channels):
            if channel in _POSITION_CHANNELS:
                root_positions_source[:, _POSITION_CHANNELS[channel]] = values[:, column]

        rotation_columns = [
            column for column, channel in enumerate(channels) if channel in _ROTATION_CHANNELS
        ]
        rotation_channels = [channels[column] for column in rotation_columns]
        if rotation_channels:
            sequence = "".join(channel[0].upper() for channel in rotation_channels)
            angles_deg = values[:, rotation_columns]
            if len(rotation_columns) == 1:
                angles_deg = angles_deg[:, 0]
            source_rotation = Rotation.from_euler(sequence, angles_deg, degrees=True).as_matrix()
            canonical_rotation = transform @ source_rotation @ transform.T
            local_quaternions[:, joint] = matrix_to_quaternion(canonical_rotation).astype(
                np.float32
            )
        else:
            local_quaternions[:, joint] = (1.0, 0.0, 0.0, 0.0)

    root_positions_source += data.offsets[root]
    root_translation_m = (root_positions_source @ transform.T * options.unit_scale_to_m).astype(
        np.float32
    )
    rest_quaternions = np.zeros((joint_count, 4), dtype=np.float32)
    rest_quaternions[:, 0] = 1.0

    name_to_index = {name: index for index, name in enumerate(data.joint_names)}
    roles: dict[str, int] = {}
    if role_names is not None:
        for role, name in role_names.items():
            if name not in name_to_index:
                raise ValueError(f"semantic role {role!r} references unknown BVH joint {name!r}")
            roles[role] = name_to_index[name]

    marker_specs = {} if markers is None else dict(markers)
    limit_specs: dict[int, JointLimitSpec] = {}
    if joint_limits is not None:
        for name, limit in joint_limits.items():
            if name not in name_to_index:
                raise ValueError(f"joint limit references unknown BVH joint {name!r}")
            limit_specs[name_to_index[name]] = limit
    end_site_metadata = [
        {
            "parent": end_site.parent,
            "offset_source": list(end_site.offset),
            "offset_m": list(
                np.asarray(end_site.offset, dtype=np.float64)
                @ transform.T
                * options.unit_scale_to_m
            ),
        }
        for end_site in data.end_sites
    ]
    skeleton = Skeleton(
        joint_names=data.joint_names,
        parents=data.parents,
        rest_offsets_m=offsets_m.astype(np.float32),
        rest_local_quat_wxyz=rest_quaternions,
        roles=roles,
        markers=marker_specs,
        joint_limits=limit_specs,
        metadata={
            "source_format": "BVH",
            "source_coordinate_system": {
                "handedness": options.handedness,
                "up_axis": options.up_axis,
                "forward_axis": options.forward_axis,
                "unit_scale_to_m": options.unit_scale_to_m,
            },
            "source_to_canonical_matrix": transform.tolist(),
            "local_axis_confidence": 0.25,
            "rest_rotation_source": "bvh_identity_zero_channel",
            "bvh_channels": [list(channels) for channels in data.channels],
            "bvh_end_sites": end_site_metadata,
        },
    )
    return MotionClip(
        skeleton=skeleton,
        local_quat_wxyz=local_quaternions,
        root_translation_m=root_translation_m,
        fps=1.0 / data.frame_time_s,
        metadata={"source_format": "BVH"},
    )


def import_bvh(
    path: Path,
    *,
    options: BvhImportOptions | None = None,
    role_names: dict[str, str] | None = None,
    markers: dict[str, MarkerSpec] | None = None,
    joint_limits: dict[str, JointLimitSpec] | None = None,
    source_dataset: str = "unspecified_bvh",
    source_clip_id: str | None = None,
    source_take_id: str | None = None,
    split: SplitName | None = None,
    clean_confidence: float | None = None,
) -> MotionClip:
    """Read a BVH file and convert it to canonical MotionLab data."""
    clip = bvh_to_motion_clip(
        load_bvh(path),
        options=BvhImportOptions() if options is None else options,
        role_names=role_names,
        markers=markers,
        joint_limits=joint_limits,
    )
    source_path = Path(path)
    digest = f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    metadata = dict(clip.metadata)
    metadata["source_path"] = str(source_path)
    metadata["source_file_sha256"] = digest
    imported = clip.with_updates(metadata=metadata)
    return assign_source_lineage(
        imported,
        source_dataset=source_dataset,
        source_clip_id=source_path.stem if source_clip_id is None else source_clip_id,
        source_take_id=digest if source_take_id is None else source_take_id,
        split=split,
        importer_version="motionlab-bvh-0.1.0",
        clean_confidence=clean_confidence,
        metadata={"source_format": "BVH"},
    )
