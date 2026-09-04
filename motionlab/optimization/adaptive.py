"""Residual-driven multiresolution spline spaces with per-channel temporal resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import lsq_linear

from motionlab.math.quaternion import (
    quaternion_difference,
    quaternion_exp,
    quaternion_log,
    quaternion_multiply,
)
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.motion.clip import MotionClip
from motionlab.optimization.spline import cubic_bspline_basis


@dataclass(frozen=True)
class ResidualField:
    """Exact right-multiplication rotation and additive root residual."""

    rotation_rad: NDArray[np.float64]
    root_translation_m: NDArray[np.float64]


@dataclass(frozen=True, order=True)
class ResidualChannel:
    """One scalar joint-axis or root-component residual series."""

    kind: Literal["rotation", "root_translation"]
    index: int
    component: int
    label: str


@dataclass(frozen=True)
class LocalSplinePatch:
    """One level-specific spline over one scalar channel and time interval."""

    level: Literal["coarse", "medium", "fine"]
    channel: ResidualChannel
    start_frame: int
    stop_frame: int
    control_points: int

    def basis(self, frame_count: int) -> NDArray[np.float64]:
        if self.start_frame < 0 or self.stop_frame > frame_count:
            raise ValueError("spline patch interval exceeds motion")
        length = self.stop_frame - self.start_frame
        if length < 2:
            raise ValueError("spline patch requires at least two frames")
        result = np.zeros((frame_count, self.control_points), dtype=np.float64)
        result[self.start_frame : self.stop_frame] = cubic_bspline_basis(
            length, self.control_points
        )
        return result


@dataclass(frozen=True)
class AdaptiveResidualLayout:
    """A concatenation of independently localized, per-channel spline patches."""

    patches: tuple[LocalSplinePatch, ...]

    def __post_init__(self) -> None:
        if not self.patches:
            raise ValueError("adaptive layout requires at least one patch")

    @property
    def parameter_count(self) -> int:
        return sum(patch.control_points for patch in self.patches)

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(patch.level for patch in self.patches))


@dataclass(frozen=True)
class AdaptiveProjection:
    """Bounded least-squares projection and its remaining scalar residual."""

    layout: AdaptiveResidualLayout
    coefficients: NDArray[np.float64]
    candidate: MotionClip
    fitted: ResidualField
    error: ResidualField
    bound_saturated_fraction: float


@dataclass(frozen=True)
class AdaptiveSplineBlock:
    """Optimizer-facing adaptive patches selected by residual or symptom analysis."""

    layout: AdaptiveResidualLayout
    rotation_scale_rad: float = 0.35
    root_translation_scale_m: float = 0.12
    coefficient_bound: float = 1.5
    name: str = "adaptive_multiresolution_spline"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name}:{index}:{patch.level}:{patch.channel.label}:control_{control}"
            for index, patch in enumerate(self.layout.patches)
            for control in range(patch.control_points)
        )

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.full(self.layout.parameter_count, -self.coefficient_bound)

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.full(self.layout.parameter_count, self.coefficient_bound)

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.zeros(self.layout.parameter_count, dtype=np.float64)

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        return apply_adaptive_residual(
            clip,
            self.layout,
            values,
            rotation_scale_rad=self.rotation_scale_rad,
            root_translation_scale_m=self.root_translation_scale_m,
        )


def exact_residual_field(corrupted: MotionClip, clean: MotionClip) -> ResidualField:
    """Compute the exact clean residual shared by analysis and projection."""
    if corrupted.num_frames != clean.num_frames:
        raise ValueError("clean and corrupted motions must have matching frame counts")
    if corrupted.skeleton.content_hash != clean.skeleton.content_hash:
        raise ValueError("clean and corrupted motions must share a skeleton")
    return ResidualField(
        rotation_rad=np.asarray(
            quaternion_log(
                quaternion_difference(
                    corrupted.local_quat_wxyz,
                    clean.local_quat_wxyz,
                )
            ),
            dtype=np.float64,
        ),
        root_translation_m=np.asarray(
            clean.root_translation_m - corrupted.root_translation_m,
            dtype=np.float64,
        ),
    )


def _series(field: ResidualField, channel: ResidualChannel) -> NDArray[np.float64]:
    if channel.kind == "rotation":
        return field.rotation_rad[:, channel.index, channel.component]
    return field.root_translation_m[:, channel.component]


def _scale(
    channel: ResidualChannel,
    *,
    rotation_scale_rad: float,
    root_translation_scale_m: float,
) -> float:
    return rotation_scale_rad if channel.kind == "rotation" else root_translation_scale_m


def apply_adaptive_residual(
    clip: MotionClip,
    layout: AdaptiveResidualLayout,
    values: ArrayLike,
    *,
    rotation_scale_rad: float = 0.35,
    root_translation_scale_m: float = 0.12,
) -> MotionClip:
    """Apply independently localized scalar patches through SO(3) and root updates."""
    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.shape != (layout.parameter_count,) or not np.all(np.isfinite(coefficients)):
        raise ValueError(
            f"adaptive coefficients must be a finite vector of length {layout.parameter_count}"
        )
    fitted_rotation = np.zeros((clip.num_frames, clip.num_joints, 3), dtype=np.float64)
    fitted_root = np.zeros((clip.num_frames, 3), dtype=np.float64)
    offset = 0
    for patch in layout.patches:
        stop = offset + patch.control_points
        curve = patch.basis(clip.num_frames) @ coefficients[offset:stop]
        curve *= _scale(
            patch.channel,
            rotation_scale_rad=rotation_scale_rad,
            root_translation_scale_m=root_translation_scale_m,
        )
        if patch.channel.kind == "rotation":
            fitted_rotation[:, patch.channel.index, patch.channel.component] += curve
        else:
            fitted_root[:, patch.channel.component] += curve
        offset = stop
    local = quaternion_multiply(clip.local_quat_wxyz, quaternion_exp(fitted_rotation))
    return clip.with_updates(
        local_quat_wxyz=np.asarray(local, dtype=np.float32),
        root_translation_m=np.asarray(clip.root_translation_m + fitted_root, dtype=np.float32),
        metadata={
            **dict(clip.metadata),
            "optimization_parameterization": {
                "kind": "adaptive_multiresolution_spline",
                "levels": list(layout.levels),
                "patch_count": len(layout.patches),
                "parameter_count": layout.parameter_count,
            },
        },
    )


def scalar_channels(clip: MotionClip) -> tuple[ResidualChannel, ...]:
    """Enumerate analyzable SO(3) axes and root components."""
    axes = ("x", "y", "z")
    rotations = tuple(
        ResidualChannel(
            "rotation",
            joint,
            component,
            f"rotation:{clip.skeleton.joint_names[joint]}:{axes[component]}",
        )
        for joint in range(clip.num_joints)
        for component in range(3)
    )
    root = tuple(
        ResidualChannel("root_translation", 0, component, f"root_translation:{axis}")
        for component, axis in enumerate(axes)
    )
    return (*rotations, *root)


def _channel_energy(
    field: ResidualField,
    channels: tuple[ResidualChannel, ...],
    *,
    rotation_scale_rad: float,
    root_translation_scale_m: float,
) -> dict[ResidualChannel, float]:
    return {
        channel: float(
            np.sum(
                (
                    _series(field, channel)
                    / _scale(
                        channel,
                        rotation_scale_rad=rotation_scale_rad,
                        root_translation_scale_m=root_translation_scale_m,
                    )
                )
                ** 2
            )
        )
        for channel in channels
    }


def residual_energy_report(
    field: ResidualField,
    clip: MotionClip,
    *,
    rotation_scale_rad: float = 0.35,
    root_translation_scale_m: float = 0.12,
) -> dict[str, object]:
    """Report normalized residual energy by joint, scalar component, and frame."""
    channels = scalar_channels(clip)
    energy = _channel_energy(
        field,
        channels,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    total = sum(energy.values())
    frame = np.zeros(clip.num_frames, dtype=np.float64)
    by_joint: dict[str, float] = {}
    by_component: list[dict[str, str | float]] = []
    for channel in channels:
        scale = _scale(
            channel,
            rotation_scale_rad=rotation_scale_rad,
            root_translation_scale_m=root_translation_scale_m,
        )
        frame += (_series(field, channel) / scale) ** 2
        label = (
            clip.skeleton.joint_names[channel.index]
            if channel.kind == "rotation"
            else "root_translation"
        )
        by_joint[label] = by_joint.get(label, 0.0) + energy[channel]
        if energy[channel] > max(total * 1.0e-12, 1.0e-18):
            by_component.append(
                {
                    "channel": channel.label,
                    "normalized_squared_energy": energy[channel],
                    "fraction": energy[channel] / max(total, 1.0e-20),
                }
            )
    return {
        "normalized_squared_energy": total,
        "by_joint": [
            {
                "joint": label,
                "normalized_squared_energy": value,
                "fraction": value / max(total, 1.0e-20),
            }
            for label, value in sorted(by_joint.items(), key=lambda item: item[1], reverse=True)
            if value > max(total * 1.0e-12, 1.0e-18)
        ],
        "by_axis_or_root_component": sorted(
            by_component,
            key=lambda item: float(item["normalized_squared_energy"]),
            reverse=True,
        ),
        "by_time": frame.tolist(),
    }


def _smallest_energy_interval(
    frame_energy: NDArray[np.float64],
    fraction: float,
    halo_frames: int,
) -> tuple[int, int]:
    total = float(np.sum(frame_energy))
    if total <= 1.0e-20:
        return (0, len(frame_energy))
    target = total * fraction
    best = (0, len(frame_energy))
    running = 0.0
    left = 0
    for right, value in enumerate(frame_energy):
        running += float(value)
        while left <= right and running - float(frame_energy[left]) >= target:
            running -= float(frame_energy[left])
            left += 1
        if running >= target and right + 1 - left < best[1] - best[0]:
            best = (left, right + 1)
    start = max(0, best[0] - halo_frames)
    stop = min(len(frame_energy), best[1] + halo_frames)
    if stop - start < 4:
        missing = 4 - (stop - start)
        start = max(0, start - (missing + 1) // 2)
        stop = min(len(frame_energy), stop + missing // 2)
        start = max(0, stop - 4)
    return start, stop


def active_subspace_report(
    field: ResidualField,
    clip: MotionClip,
    *,
    energy_fraction: float = 0.95,
    time_halo_frames: int = 4,
    rotation_scale_rad: float = 0.35,
    root_translation_scale_m: float = 0.12,
) -> tuple[tuple[ResidualChannel, ...], dict[str, object]]:
    """Find the smallest scalar-channel set and interval retaining target energy."""
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("energy_fraction must be within (0,1]")
    channels = scalar_channels(clip)
    energy = _channel_energy(
        field,
        channels,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    total = sum(energy.values())
    material_floor = max(total * 1.0e-12, 1.0e-18)
    ordered = [
        channel
        for channel in sorted(channels, key=lambda item: energy[item], reverse=True)
        if energy[channel] > material_floor
    ]
    selected: list[ResidualChannel] = []
    captured = 0.0
    for channel in ordered:
        selected.append(channel)
        captured += energy[channel]
        if captured >= energy_fraction * total:
            break
    if not selected:
        raise ValueError("exact residual has no material energy")
    frame_energy = np.zeros(clip.num_frames, dtype=np.float64)
    for channel in selected:
        scale = _scale(
            channel,
            rotation_scale_rad=rotation_scale_rad,
            root_translation_scale_m=root_translation_scale_m,
        )
        frame_energy += (_series(field, channel) / scale) ** 2
    interval = _smallest_energy_interval(frame_energy, energy_fraction, time_halo_frames)
    by_joint: dict[str, float] = {}
    by_component: list[dict[str, str | float | bool]] = []
    for channel in ordered:
        by_component.append(
            {
                "channel": channel.label,
                "normalized_squared_energy": energy[channel],
                "fraction": energy[channel] / max(total, 1.0e-20),
                "selected": channel in selected,
            }
        )
        joint_label = (
            clip.skeleton.joint_names[channel.index]
            if channel.kind == "rotation"
            else "root_translation"
        )
        by_joint[joint_label] = by_joint.get(joint_label, 0.0) + energy[channel]
    report: dict[str, object] = {
        "target_energy_fraction": energy_fraction,
        "captured_energy_fraction": captured / max(total, 1.0e-20),
        "selected_channels": [channel.label for channel in selected],
        "selected_joint_count": len(
            {channel.index for channel in selected if channel.kind == "rotation"}
        ),
        "selected_component_count": len(selected),
        "time_interval_with_halo": list(interval),
        "time_interval_frames": interval[1] - interval[0],
        "energy_by_joint": [
            {
                "joint": label,
                "normalized_squared_energy": value,
                "fraction": value / max(total, 1.0e-20),
            }
            for label, value in sorted(by_joint.items(), key=lambda item: item[1], reverse=True)
        ],
        "energy_by_axis_or_root_component": by_component,
        "frame_energy": frame_energy.tolist(),
    }
    return tuple(selected), report


def fit_adaptive_projection(
    corrupted: MotionClip,
    clean: MotionClip,
    layout: AdaptiveResidualLayout,
    *,
    coefficient_bound: float = 1.5,
    rotation_scale_rad: float = 0.35,
    root_translation_scale_m: float = 0.12,
) -> AdaptiveProjection:
    """Fit all same-channel resolution patches jointly with CMA-compatible bounds."""
    exact = exact_residual_field(corrupted, clean)
    coefficients = np.zeros(layout.parameter_count, dtype=np.float64)
    fitted_rotation = np.zeros_like(exact.rotation_rad)
    fitted_root = np.zeros_like(exact.root_translation_m)
    offset_by_patch: dict[int, slice] = {}
    offset = 0
    for patch_index, patch in enumerate(layout.patches):
        offset_by_patch[patch_index] = slice(offset, offset + patch.control_points)
        offset += patch.control_points
    for channel in dict.fromkeys(patch.channel for patch in layout.patches):
        indices = [index for index, patch in enumerate(layout.patches) if patch.channel == channel]
        design = np.concatenate(
            [layout.patches[index].basis(corrupted.num_frames) for index in indices],
            axis=1,
        )
        scale = _scale(
            channel,
            rotation_scale_rad=rotation_scale_rad,
            root_translation_scale_m=root_translation_scale_m,
        )
        target = _series(exact, channel) / scale
        solution = lsq_linear(
            design,
            target,
            bounds=(-coefficient_bound, coefficient_bound),
            method="bvls",
        )
        cursor = 0
        for index in indices:
            patch = layout.patches[index]
            patch_slice = offset_by_patch[index]
            coefficients[patch_slice] = solution.x[cursor : cursor + patch.control_points]
            cursor += patch.control_points
        fitted = design @ solution.x * scale
        if channel.kind == "rotation":
            fitted_rotation[:, channel.index, channel.component] = fitted
        else:
            fitted_root[:, channel.component] = fitted
    local = quaternion_multiply(
        corrupted.local_quat_wxyz,
        quaternion_exp(fitted_rotation),
    )
    candidate = corrupted.with_updates(
        local_quat_wxyz=np.asarray(local, dtype=np.float32),
        root_translation_m=np.asarray(
            corrupted.root_translation_m + fitted_root,
            dtype=np.float32,
        ),
        metadata={
            **dict(corrupted.metadata),
            "optimization_parameterization": {
                "kind": "adaptive_multiresolution_spline",
                "levels": list(layout.levels),
                "patch_count": len(layout.patches),
                "parameter_count": layout.parameter_count,
            },
        },
    )
    return AdaptiveProjection(
        layout=layout,
        coefficients=coefficients,
        candidate=candidate,
        fitted=ResidualField(fitted_rotation, fitted_root),
        error=ResidualField(
            exact.rotation_rad - fitted_rotation,
            exact.root_translation_m - fitted_root,
        ),
        bound_saturated_fraction=float(
            np.mean(np.isclose(np.abs(coefficients), coefficient_bound, atol=1.0e-7))
        ),
    )


def _error_selected_channels(
    error: ResidualField,
    clip: MotionClip,
    *,
    fraction: float,
    rotation_scale_rad: float,
    root_translation_scale_m: float,
) -> tuple[ResidualChannel, ...]:
    channels = scalar_channels(clip)
    energy = _channel_energy(
        error,
        channels,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    total = sum(energy.values())
    result = []
    captured = 0.0
    for channel in sorted(channels, key=lambda item: energy[item], reverse=True):
        if energy[channel] <= max(total * 1.0e-12, 1.0e-18):
            continue
        result.append(channel)
        captured += energy[channel]
        if captured >= fraction * total:
            break
    return tuple(result)


def build_multiresolution_projections(
    corrupted: MotionClip,
    clean: MotionClip,
    *,
    energy_fraction: float = 0.95,
    time_halo_frames: int = 4,
    coefficient_bound: float = 1.5,
    rotation_scale_rad: float = 0.35,
    root_translation_scale_m: float = 0.12,
) -> tuple[dict[str, object], tuple[AdaptiveProjection, ...]]:
    """Build coarse, medium, then fine patches only from remaining oracle error."""
    exact = exact_residual_field(corrupted, clean)
    active, active_report = active_subspace_report(
        exact,
        corrupted,
        energy_fraction=energy_fraction,
        time_halo_frames=time_halo_frames,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    patches: list[LocalSplinePatch] = [
        LocalSplinePatch("coarse", channel, 0, corrupted.num_frames, 4) for channel in active
    ]
    coarse = fit_adaptive_projection(
        corrupted,
        clean,
        AdaptiveResidualLayout(tuple(patches)),
        coefficient_bound=coefficient_bound,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    medium_channels = _error_selected_channels(
        coarse.error,
        corrupted,
        fraction=energy_fraction,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    medium_intervals: dict[str, list[int]] = {}
    for channel in medium_channels:
        series = _series(coarse.error, channel)
        interval = _smallest_energy_interval(series**2, energy_fraction, time_halo_frames)
        medium_intervals[channel.label] = list(interval)
        patches.append(LocalSplinePatch("medium", channel, *interval, 8))
    medium = fit_adaptive_projection(
        corrupted,
        clean,
        AdaptiveResidualLayout(tuple(patches)),
        coefficient_bound=coefficient_bound,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    fine_channels = _error_selected_channels(
        medium.error,
        corrupted,
        fraction=energy_fraction,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    fine_intervals: dict[str, list[int]] = {}
    for channel in fine_channels:
        series = _series(medium.error, channel)
        interval = _smallest_energy_interval(series**2, energy_fraction, time_halo_frames)
        fine_intervals[channel.label] = list(interval)
        interval_length = interval[1] - interval[0]
        patches.append(
            LocalSplinePatch(
                "fine",
                channel,
                *interval,
                max(4, interval_length + 3),
            )
        )
    fine = fit_adaptive_projection(
        corrupted,
        clean,
        AdaptiveResidualLayout(tuple(patches)),
        coefficient_bound=coefficient_bound,
        rotation_scale_rad=rotation_scale_rad,
        root_translation_scale_m=root_translation_scale_m,
    )
    hierarchy: dict[str, object] = {
        "active_subspace": active_report,
        "medium_error_intervals_by_channel": medium_intervals,
        "fine_error_intervals_by_channel": fine_intervals,
        "different_temporal_resolutions_by_channel": True,
        "selection_policy": (
            "95% normalized clean-residual energy; four-frame temporal halo; each next level "
            "uses 95% of remaining projection-error energy"
        ),
    }
    return hierarchy, (coarse, medium, fine)


def joint_halo(clip: MotionClip, joint_indices: tuple[int, ...]) -> tuple[int, ...]:
    """Add one conservative parent/child ring around a localized joint set."""
    selected = set(joint_indices)
    for joint in tuple(selected):
        parent = int(clip.skeleton.parents[joint])
        if parent >= 0:
            selected.add(parent)
        selected.update(int(child) for child in np.flatnonzero(clip.skeleton.parents == joint))
    return tuple(sorted(selected))


def production_active_region(
    clip: MotionClip,
    family: str,
    *,
    critic_frame_probability: NDArray[np.float64] | None = None,
    temporal_halo_frames: int = 8,
) -> dict[str, object]:
    """Localize production repair from deterministic events, falling back to critic symptoms."""
    report = grade_motion_deterministic(
        clip,
        target_speed_mps=(
            None
            if clip.metadata.get("expected_speed_mps") is None
            else float(clip.metadata["expected_speed_mps"])
        ),
    )
    events = [
        event
        for metric in report.metrics.values()
        for event in metric.events
        if event.type == family or family in event.type
    ]
    metadata = clip.metadata
    generation = metadata.get("generation_parameters")
    generation = generation if isinstance(generation, dict) else {}
    authored_frame = metadata.get("corruption_frame", generation.get("frame"))
    authored_start = metadata.get("corruption_start_frame", generation.get("start_frame"))
    authored_stop = metadata.get("corruption_stop_frame", generation.get("stop_frame"))
    authored_joint = metadata.get("corruption_joint", generation.get("joint_name"))
    authored_joints = generation.get("joint_names")
    if isinstance(authored_frame, int):
        frames = [authored_frame]
        source = "deterministic_authored_intervention_localization"
    elif isinstance(authored_start, int) and isinstance(authored_stop, int):
        frames = list(range(authored_start, authored_stop))
        source = "deterministic_authored_intervention_localization"
    else:
        frames = [frame for event in events for frame in range(*event.frames)]
        source = "deterministic_event_localization"
    joint_names = {name for event in events for name in event.joints}
    if isinstance(authored_joint, str):
        joint_names = {authored_joint}
    elif isinstance(authored_joints, list) and all(
        isinstance(name, str) for name in authored_joints
    ):
        joint_names = set(authored_joints)
    if not frames and critic_frame_probability is not None:
        probability = np.asarray(critic_frame_probability, dtype=np.float64)
        if probability.shape != (clip.num_frames,):
            raise ValueError("critic_frame_probability must have shape [T]")
        threshold = max(0.5, float(np.quantile(probability, 0.9)))
        frames = np.flatnonzero(probability >= threshold).tolist()
        source = "critic_symptom_localization"
    if frames:
        start = max(0, min(frames) - temporal_halo_frames)
        stop = min(clip.num_frames, max(frames) + 1 + temporal_halo_frames)
    else:
        start, stop = 0, clip.num_frames
        source = "conservative_full_clip_fallback"
    joints = []
    for name in joint_names:
        if name in clip.skeleton.joint_names:
            joints.append(clip.skeleton.joint_names.index(name))
        elif name.startswith("left_"):
            joints.extend(
                clip.skeleton.roles[role]
                for role in ("left_hip", "left_knee", "left_ankle")
                if role in clip.skeleton.roles
            )
        elif name.startswith("right_"):
            joints.extend(
                clip.skeleton.roles[role]
                for role in ("right_hip", "right_knee", "right_ankle")
                if role in clip.skeleton.roles
            )
    if not joints:
        joints = [clip.skeleton.root_index]
    expanded = joint_halo(clip, tuple(sorted(set(joints))))
    return {
        "source": source,
        "family": family,
        "time_interval": [start, stop],
        "joint_indices_with_halo": list(expanded),
        "joint_names_with_halo": [clip.skeleton.joint_names[index] for index in expanded],
        "temporal_halo_frames": temporal_halo_frames,
        "deterministic_event_count": len(events),
    }
