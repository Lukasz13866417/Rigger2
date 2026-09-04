"""Composable numerical parameter blocks for defect-appropriate motion repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from motionlab.math.quaternion import (
    quaternion_difference,
    quaternion_exp,
    quaternion_log,
    quaternion_multiply,
    quaternion_slerp,
)
from motionlab.motion.clip import MotionClip
from motionlab.optimization.spline import SplineResidualLayout, apply_spline_residual
from motionlab.repair.close_loop import close_loop
from motionlab.repair.foot_lock import lock_foot


class MotionParameterBlock(Protocol):
    """Common optimizer-facing contract for one typed parameter block."""

    @property
    def name(self) -> str: ...

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def lower_bounds(self) -> NDArray[np.float64]: ...

    @property
    def upper_bounds(self) -> NDArray[np.float64]: ...

    @property
    def initial_values(self) -> NDArray[np.float64]: ...

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip: ...


def _validated_values(block: MotionParameterBlock, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    expected = len(block.parameter_names)
    if array.shape != (expected,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{block.name} values must be a finite vector of length {expected}")
    if np.any(array < block.lower_bounds) or np.any(array > block.upper_bounds):
        raise ValueError(f"{block.name} values exceed declared bounds")
    return array


@dataclass(frozen=True)
class CompositeMotionParameterization:
    """Concatenate any chosen parameter blocks behind one apply interface."""

    blocks: tuple[MotionParameterBlock, ...]
    name: str = "composite"

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("composite parameterization requires at least one block")
        names = [name for block in self.blocks for name in block.parameter_names]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique across blocks")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for block in self.blocks for name in block.parameter_names)

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.concatenate([block.lower_bounds for block in self.blocks])

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.concatenate([block.upper_bounds for block in self.blocks])

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.concatenate([block.initial_values for block in self.blocks])

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        array = _validated_values(self, values)
        result = clip
        offset = 0
        for block in self.blocks:
            stop = offset + len(block.parameter_names)
            result = block.apply(result, array[offset:stop])
            offset = stop
        return result


@dataclass(frozen=True)
class GenericSplineBlock:
    """Smooth root/SO(3) residual splines for generic pose and coordination."""

    layout: SplineResidualLayout
    rotation_scale_rad: float = 0.35
    root_translation_scale_m: float = 0.12
    root_yaw_scale_rad: float = 0.35
    coefficient_bound: float = 1.5
    name: str = "generic_spline"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name}:{channel}:control_{control}"
            for channel in self.layout.channel_names
            for control in range(self.layout.control_points)
        )

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.full(len(self.parameter_names), -self.coefficient_bound)

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.full(len(self.parameter_names), self.coefficient_bound)

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.zeros(len(self.parameter_names), dtype=np.float64)

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        array = _validated_values(self, values)
        return apply_spline_residual(
            clip,
            self.layout,
            array,
            rotation_scale_rad=self.rotation_scale_rad,
            root_translation_scale_m=self.root_translation_scale_m,
            root_yaw_scale_rad=self.root_yaw_scale_rad,
        )


@dataclass(frozen=True)
class ContactIKBlock:
    """Contact targets, foot-lock blending, pelvis compensation, and IK mapping."""

    side: str
    start_frame: int
    stop_frame: int
    maximum_blend_frames: int = 12
    name: str = "contact_ik"

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("contact side must be left or right")
        if self.start_frame < 0 or self.stop_frame <= self.start_frame:
            raise ValueError("contact interval must be nonempty")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name}:{self.side}:{name}"
            for name in (
                "target_start_x_m",
                "target_start_y_m",
                "target_start_z_m",
                "target_end_x_m",
                "target_end_y_m",
                "target_end_z_m",
                "lock_strength",
                "blend_in_frames",
                "blend_out_frames",
                "pelvis_compensation_limit_m",
                "root_compensation_x_m",
                "root_compensation_y_m",
                "root_compensation_z_m",
            )
        )

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                -0.08,
                -0.08,
                -0.08,
                -0.08,
                -0.08,
                -0.08,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.05,
                -0.05,
                -0.05,
            ],
            dtype=np.float64,
        )

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                0.08,
                0.08,
                0.08,
                0.08,
                0.08,
                0.08,
                1.0,
                self.maximum_blend_frames,
                self.maximum_blend_frames,
                0.05,
                0.05,
                0.05,
                0.05,
            ],
            dtype=np.float64,
        )

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.zeros(len(self.parameter_names), dtype=np.float64)

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        array = _validated_values(self, values)
        if self.stop_frame > clip.num_frames:
            raise ValueError("contact interval exceeds clip")
        root = clip.root_translation_m.copy()
        phase = np.linspace(0.0, np.pi, self.stop_frame - self.start_frame)
        envelope = np.sin(phase) ** 2
        root[self.start_frame : self.stop_frame] += (envelope[:, None] * array[10:13]).astype(
            np.float32
        )
        compensated = clip.with_updates(root_translation_m=root)
        return lock_foot(
            compensated,
            side=self.side,  # type: ignore[arg-type]
            start_frame=self.start_frame,
            stop_frame=self.stop_frame,
            lock_strength=float(array[6]),
            blend_in_frames=round(array[7]),
            blend_out_frames=round(array[8]),
            pelvis_compensation_limit_m=float(array[9]),
            anchor_offset_m=array[:3],
            anchor_offset_end_m=array[3:6],
            ground_plane=clip.metadata.get("ground_plane", (0.0, 1.0, 0.0, 0.0)),
        ).repaired


@dataclass(frozen=True)
class SpeedCadenceBlock:
    """Temporal warp, displacement, stride, and cadence controls."""

    name: str = "speed_cadence"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name}:{name}"
            for name in (
                "global_temporal_warp_fraction",
                "local_temporal_warp_frames",
                "root_displacement_scale_delta",
                "stride_scale_delta",
                "cadence_scale_delta",
            )
        )

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.asarray([-0.15, -8.0, -0.2, -0.15, -0.15], dtype=np.float64)

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.asarray([0.15, 8.0, 0.2, 0.15, 0.15], dtype=np.float64)

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.zeros(len(self.parameter_names), dtype=np.float64)

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        array = _validated_values(self, values)
        frames = clip.num_frames
        if frames < 2:
            return clip.copy()
        unit = np.linspace(0.0, 1.0, frames)
        temporal_scale = 1.0 + array[0] + array[4]
        source_unit = (unit - 0.5) * temporal_scale + 0.5
        source_unit += array[1] * np.sin(np.pi * unit) / (frames - 1)
        source_frame = np.clip(source_unit, 0.0, 1.0) * (frames - 1)
        left = np.floor(source_frame).astype(np.int64)
        right = np.minimum(left + 1, frames - 1)
        fraction = source_frame - left
        local = quaternion_slerp(
            clip.local_quat_wxyz[left],
            clip.local_quat_wxyz[right],
            fraction[:, None],
        ).astype(np.float32)
        root = np.stack(
            [
                np.interp(source_frame, np.arange(frames), clip.root_translation_m[:, axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        relative = root - root[0]
        relative[:, (0, 2)] *= 1.0 + array[2]
        relative[:, 2] *= 1.0 + array[3]
        root = root[0] + relative
        return clip.with_updates(
            local_quat_wxyz=local,
            root_translation_m=root.astype(np.float32),
            metadata={
                **dict(clip.metadata),
                "optimization_parameterization": {
                    "kind": "speed_cadence",
                    "values": dict(zip(self.parameter_names, array.tolist(), strict=True)),
                },
            },
        )


@dataclass(frozen=True)
class LoopSeamBlock:
    """Loop phase/cut plus independently blended pose and root-velocity closure."""

    seam_window_frames: int = 12
    maximum_cut_frames: int = 12
    name: str = "loop_seam"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            f"{self.name}:phase_cut_frames",
            f"{self.name}:seam_pose_strength",
            f"{self.name}:seam_root_velocity_strength",
        )

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.asarray([-self.maximum_cut_frames, 0.0, 0.0], dtype=np.float64)

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.asarray([self.maximum_cut_frames, 1.0, 1.0], dtype=np.float64)

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.zeros(len(self.parameter_names), dtype=np.float64)

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        array = _validated_values(self, values)
        cut = round(array[0])
        local = np.roll(clip.local_quat_wxyz, -cut, axis=0)
        root = np.roll(clip.root_translation_m, -cut, axis=0).copy()
        root += clip.root_translation_m[0] - root[0]
        cut_clip = clip.with_updates(local_quat_wxyz=local, root_translation_m=root)
        full = close_loop(
            cut_clip,
            seam_window_frames=self.seam_window_frames,
            strength=1.0,
        ).repaired
        tangent = quaternion_log(
            quaternion_difference(cut_clip.local_quat_wxyz, full.local_quat_wxyz)
        )
        repaired_local = quaternion_multiply(
            cut_clip.local_quat_wxyz,
            quaternion_exp(float(array[1]) * tangent),
        )
        repaired_root = cut_clip.root_translation_m + float(array[2]) * (
            full.root_translation_m - cut_clip.root_translation_m
        )
        return cut_clip.with_updates(
            local_quat_wxyz=np.asarray(repaired_local, dtype=np.float32),
            root_translation_m=np.asarray(repaired_root, dtype=np.float32),
            metadata={
                **dict(cut_clip.metadata),
                "optimization_parameterization": {
                    "kind": "loop_seam",
                    "phase_cut_frames": cut,
                    "pose_strength": float(array[1]),
                    "root_velocity_strength": float(array[2]),
                },
            },
        )
