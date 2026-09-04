"""Frequency-domain and low-dimensional smoothing blocks for local rotational jitter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dct, idct
from scipy.signal import butter, sosfiltfilt

from motionlab.math.quaternion import (
    quaternion_difference,
    quaternion_exp,
    quaternion_log,
    quaternion_multiply,
)
from motionlab.motion.clip import MotionClip
from motionlab.optimization.adaptive import ResidualField, exact_residual_field


@dataclass(frozen=True, order=True)
class SpectralMode:
    """One local DCT frequency on one joint tangent axis."""

    joint_index: int
    component: int
    frequency_index: int


@dataclass(frozen=True)
class LocalSpectralJitterBlock:
    """Sparse local DCT correction composed in joint-local SO(3)."""

    start_frame: int
    stop_frame: int
    modes: tuple[SpectralMode, ...]
    coefficient_scale_rad: float = 0.12
    coefficient_bound: float = 3.0
    name: str = "local_spectral_jitter"

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.stop_frame <= self.start_frame:
            raise ValueError("spectral interval must be nonempty")
        if not self.modes or len(self.modes) != len(set(self.modes)):
            raise ValueError("spectral modes must be nonempty and unique")
        length = self.stop_frame - self.start_frame
        if any(
            mode.joint_index < 0
            or mode.component not in {0, 1, 2}
            or mode.frequency_index < 0
            or mode.frequency_index >= length
            for mode in self.modes
        ):
            raise ValueError("spectral mode is outside the declared interval or tangent axes")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        axes = ("x", "y", "z")
        return tuple(
            (
                f"{self.name}:joint_{mode.joint_index}:{axes[mode.component]}:"
                f"dct_{mode.frequency_index}"
            )
            for mode in self.modes
        )

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.full(len(self.modes), -self.coefficient_bound)

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.full(len(self.modes), self.coefficient_bound)

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.zeros(len(self.modes), dtype=np.float64)

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        coefficients = np.asarray(values, dtype=np.float64)
        if coefficients.shape != (len(self.modes),) or not np.all(np.isfinite(coefficients)):
            raise ValueError("spectral coefficients must match the finite mode vector")
        if np.any(coefficients < self.lower_bounds) or np.any(coefficients > self.upper_bounds):
            raise ValueError("spectral coefficients exceed declared bounds")
        if self.stop_frame > clip.num_frames:
            raise ValueError("spectral interval exceeds clip")
        length = self.stop_frame - self.start_frame
        basis = idct(np.eye(length), type=2, norm="ortho", axis=0)
        tangent = np.zeros((clip.num_frames, clip.num_joints, 3), dtype=np.float64)
        for value, mode in zip(coefficients, self.modes, strict=True):
            if mode.joint_index >= clip.num_joints:
                raise ValueError("spectral mode joint exceeds clip skeleton")
            tangent[
                self.start_frame : self.stop_frame,
                mode.joint_index,
                mode.component,
            ] += basis[:, mode.frequency_index] * value * self.coefficient_scale_rad
        local = quaternion_multiply(clip.local_quat_wxyz, quaternion_exp(tangent))
        return clip.with_updates(
            local_quat_wxyz=np.asarray(local, dtype=np.float32),
            metadata={
                **dict(clip.metadata),
                "optimization_parameterization": {
                    "kind": "local_spectral_jitter",
                    "interval": [self.start_frame, self.stop_frame],
                    "mode_count": len(self.modes),
                    "joint_indices": sorted({mode.joint_index for mode in self.modes}),
                },
            },
        )


@dataclass(frozen=True)
class SpectralOracleProjection:
    """Oracle-fitted spectral block and held-out clean-distance evidence."""

    block: LocalSpectralJitterBlock
    coefficients: NDArray[np.float64]
    candidate: MotionClip
    error: ResidualField
    captured_energy_fraction: float


def fit_oracle_spectral_jitter_projection(
    corrupted: MotionClip,
    clean: MotionClip,
    *,
    energy_fraction: float = 0.95,
    coefficient_scale_rad: float = 0.12,
    coefficient_bound: float = 3.0,
) -> SpectralOracleProjection:
    """Select the smallest local DCT mode set retaining the requested oracle energy."""
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("energy_fraction must be within (0,1]")
    residual = exact_residual_field(corrupted, clean)
    frame_energy = np.sum(residual.rotation_rad**2, axis=(1, 2))
    total = float(np.sum(frame_energy))
    if total <= 1.0e-20:
        raise ValueError("spectral oracle requires a material rotational residual")
    # Persisted samples pass through a 6D-rotation round trip.  Exclude its numerical floor
    # so that an actually local corruption does not acquire full-clip/full-rig support.
    material_floor = max(total * 1.0e-10, 1.0e-12)
    material_frames = np.flatnonzero(frame_energy > material_floor)
    start = int(material_frames[0])
    stop = int(material_frames[-1]) + 1
    entries: list[tuple[float, SpectralMode, float]] = []
    for joint in range(corrupted.num_joints):
        for component in range(3):
            transformed = dct(
                residual.rotation_rad[start:stop, joint, component],
                type=2,
                norm="ortho",
            )
            for frequency, coefficient in enumerate(transformed):
                energy = float(coefficient * coefficient)
                if energy > material_floor:
                    entries.append(
                        (energy, SpectralMode(joint, component, frequency), float(coefficient))
                    )
    entries.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[SpectralMode, float]] = []
    captured = 0.0
    for energy, mode, coefficient in entries:
        selected.append((mode, coefficient))
        captured += energy
        if captured >= energy_fraction * total:
            break
    block = LocalSpectralJitterBlock(
        start,
        stop,
        tuple(mode for mode, _ in selected),
        coefficient_scale_rad=coefficient_scale_rad,
        coefficient_bound=coefficient_bound,
    )
    coefficients = np.asarray(
        [coefficient / coefficient_scale_rad for _, coefficient in selected],
        dtype=np.float64,
    )
    if np.any(np.abs(coefficients) > coefficient_bound):
        raise ValueError("oracle spectral coefficients exceed the declared optimizer bounds")
    candidate = block.apply(corrupted, coefficients)
    return SpectralOracleProjection(
        block=block,
        coefficients=coefficients,
        candidate=candidate,
        error=exact_residual_field(candidate, clean),
        captured_energy_fraction=captured / total,
    )


@dataclass(frozen=True)
class JitterSmoothingBlock:
    """Low-dimensional Butterworth smoothing for production CMA."""

    joint_indices: tuple[int, ...]
    start_frame: int
    stop_frame: int
    maximum_blend_frames: int = 12
    maximum_cutoff_hz: float = 24.0
    name: str = "jitter_smoothing"

    def __post_init__(self) -> None:
        if not self.joint_indices or len(self.joint_indices) != len(set(self.joint_indices)):
            raise ValueError("jitter smoothing joints must be nonempty and unique")
        if self.start_frame < 0 or self.stop_frame - self.start_frame < 8:
            raise ValueError("jitter smoothing interval must contain at least eight frames")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            f"{self.name}:cutoff_frequency_hz",
            f"{self.name}:smoothing_strength",
            f"{self.name}:blend_in_frames",
            f"{self.name}:blend_out_frames",
            *(f"{self.name}:joint_{joint}:strength" for joint in self.joint_indices),
        )

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.asarray(
            [0.5, 0.0, 0.0, 0.0, *([0.0] * len(self.joint_indices))],
            dtype=np.float64,
        )

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                self.maximum_cutoff_hz,
                1.0,
                self.maximum_blend_frames,
                self.maximum_blend_frames,
                *([1.0] * len(self.joint_indices)),
            ],
            dtype=np.float64,
        )

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                min(12.0, self.maximum_cutoff_hz),
                0.0,
                3.0,
                3.0,
                *([1.0] * len(self.joint_indices)),
            ],
            dtype=np.float64,
        )

    def apply(self, clip: MotionClip, values: ArrayLike) -> MotionClip:
        parameters = np.asarray(values, dtype=np.float64)
        if parameters.shape != (len(self.parameter_names),) or not np.all(np.isfinite(parameters)):
            raise ValueError("jitter smoothing values must match the finite parameter vector")
        if np.any(parameters < self.lower_bounds) or np.any(parameters > self.upper_bounds):
            raise ValueError("jitter smoothing values exceed declared bounds")
        if self.stop_frame > clip.num_frames or max(self.joint_indices) >= clip.num_joints:
            raise ValueError("jitter smoothing region exceeds clip")
        cutoff = float(parameters[0])
        nyquist = clip.fps / 2.0
        if cutoff >= nyquist:
            raise ValueError("jitter smoothing cutoff must remain below Nyquist")
        interval = clip.local_quat_wxyz[self.start_frame : self.stop_frame]
        reference = interval[0]
        reference_series = np.broadcast_to(reference, interval.shape)
        tangent = quaternion_log(quaternion_difference(reference_series, interval))
        sos = butter(2, cutoff / nyquist, btype="lowpass", output="sos")
        filtered = np.asarray(tangent, dtype=np.float64).copy()
        for joint in self.joint_indices:
            for component in range(3):
                filtered[:, joint, component] = sosfiltfilt(
                    sos,
                    tangent[:, joint, component],
                    padtype="odd",
                )
        smooth = quaternion_multiply(reference_series, quaternion_exp(filtered))
        correction = quaternion_log(quaternion_difference(interval, smooth))
        length = self.stop_frame - self.start_frame
        envelope = np.ones(length, dtype=np.float64)
        blend_in = min(round(parameters[2]), length)
        blend_out = min(round(parameters[3]), length)
        if blend_in:
            phase = np.linspace(0.0, 1.0, blend_in)
            envelope[:blend_in] *= phase * phase * (3.0 - 2.0 * phase)
        if blend_out:
            phase = np.linspace(1.0, 0.0, blend_out)
            envelope[-blend_out:] *= phase * phase * (3.0 - 2.0 * phase)
        local = clip.local_quat_wxyz.copy()
        for offset, joint in enumerate(self.joint_indices):
            strength = float(parameters[1] * parameters[4 + offset])
            delta = quaternion_exp(correction[:, joint] * envelope[:, None] * strength)
            local[self.start_frame : self.stop_frame, joint] = quaternion_multiply(
                interval[:, joint], delta
            ).astype(np.float32)
        return clip.with_updates(
            local_quat_wxyz=local,
            metadata={
                **dict(clip.metadata),
                "optimization_parameterization": {
                    "kind": "jitter_smoothing",
                    "joint_indices": list(self.joint_indices),
                    "interval": [self.start_frame, self.stop_frame],
                    "cutoff_frequency_hz": cutoff,
                    "smoothing_strength": float(parameters[1]),
                },
            },
        )
