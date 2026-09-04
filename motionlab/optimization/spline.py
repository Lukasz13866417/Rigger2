"""Smooth cubic B-spline residual parameterization for fixed-rig motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline

from motionlab.math.quaternion import quaternion_exp, quaternion_multiply
from motionlab.motion.clip import MotionClip
from motionlab.motion.skeleton import Skeleton

DEFAULT_OPTIMIZED_ROLES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "spine",
    "spine_1",
    "spine_2",
    "chest",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)


def cubic_bspline_basis(
    frame_count: int,
    control_points: int,
    *,
    internal_knots: tuple[float, ...] | None = None,
) -> NDArray[np.float64]:
    """Return a clamped basis whose rows sum to one at every frame."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    if control_points < 4:
        raise ValueError("cubic B-splines require at least four control points")
    degree = 3
    internal_count = control_points - degree - 1
    if internal_knots is None:
        internal = (
            np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
            if internal_count
            else np.empty(0, dtype=np.float64)
        )
    else:
        internal = np.asarray(internal_knots, dtype=np.float64)
        if internal.shape != (internal_count,):
            raise ValueError(f"expected {internal_count} internal knots, got {internal.shape}")
        if (
            not np.all(np.isfinite(internal))
            or np.any(internal <= 0.0)
            or np.any(internal >= 1.0)
            or np.any(np.diff(internal) < 0.0)
        ):
            raise ValueError("internal knots must be finite, sorted, and strictly inside (0,1)")
    knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
    time = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    basis = BSpline.design_matrix(time, knots, degree, extrapolate=False).toarray()
    return np.asarray(basis, dtype=np.float64)


@dataclass(frozen=True)
class SplineResidualLayout:
    """Stable channel layout for joint tangent, root translation, and root yaw splines."""

    control_points: int
    joint_indices: tuple[int, ...]
    joint_roles: tuple[str, ...]
    channel_names: tuple[str, ...]
    internal_knots: tuple[float, ...] | None = None

    @classmethod
    def for_skeleton(
        cls,
        skeleton: Skeleton,
        *,
        control_points: int,
        optimized_roles: tuple[str, ...] = DEFAULT_OPTIMIZED_ROLES,
        internal_knots: tuple[float, ...] | None = None,
    ) -> SplineResidualLayout:
        selected: list[int] = []
        selected_roles: list[str] = []
        for role in optimized_roles:
            index = skeleton.roles.get(role)
            if index is None or index == skeleton.root_index or index in selected:
                continue
            selected.append(index)
            selected_roles.append(role)
        if not selected:
            raise ValueError("skeleton has none of the requested optimization roles")
        rotation_channels = tuple(
            f"rotation:{role}:{axis}" for role in selected_roles for axis in ("x", "y", "z")
        )
        channels = (
            *rotation_channels,
            "root_translation:x",
            "root_translation:y",
            "root_translation:z",
            "root_yaw",
        )
        # Validate the degree/control-count contract immediately.
        cubic_bspline_basis(2, control_points, internal_knots=internal_knots)
        return cls(
            control_points,
            tuple(selected),
            tuple(selected_roles),
            channels,
            internal_knots,
        )

    @classmethod
    def for_joint_indices(
        cls,
        skeleton: Skeleton,
        *,
        control_points: int,
        joint_indices: tuple[int, ...],
        internal_knots: tuple[float, ...] | None = None,
    ) -> SplineResidualLayout:
        """Create an analysis-driven layout from explicit non-root joint indices."""
        selected = tuple(dict.fromkeys(int(index) for index in joint_indices))
        if not selected or any(
            index < 0 or index >= skeleton.num_joints or index == skeleton.root_index
            for index in selected
        ):
            raise ValueError("joint_indices must contain valid, unique non-root joints")
        by_joint = {
            index: sorted(
                role for role, role_index in skeleton.roles.items() if role_index == index
            )
            for index in selected
        }
        labels = tuple(
            roles[0] if roles else skeleton.joint_names[index] for index, roles in by_joint.items()
        )
        rotation_channels = tuple(
            f"rotation:{label}:{axis}" for label in labels for axis in ("x", "y", "z")
        )
        channels = (
            *rotation_channels,
            "root_translation:x",
            "root_translation:y",
            "root_translation:z",
            "root_yaw",
        )
        cubic_bspline_basis(2, control_points, internal_knots=internal_knots)
        return cls(control_points, selected, labels, channels, internal_knots)

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)

    @property
    def parameter_count(self) -> int:
        return self.channel_count * self.control_points

    def coefficient_matrix(self, values: ArrayLike) -> NDArray[np.float64]:
        coefficients = np.asarray(values, dtype=np.float64)
        if coefficients.shape != (self.parameter_count,):
            raise ValueError(
                f"expected {self.parameter_count} spline parameters, got {coefficients.shape}"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("spline parameters contain non-finite values")
        return coefficients.reshape((self.channel_count, self.control_points))

    def basis(self, frame_count: int) -> NDArray[np.float64]:
        """Evaluate this layout's uniform or explicitly localized cubic basis."""
        return cubic_bspline_basis(
            frame_count,
            self.control_points,
            internal_knots=self.internal_knots,
        )


def apply_spline_residual(
    clip: MotionClip,
    layout: SplineResidualLayout,
    values: ArrayLike,
    *,
    rotation_scale_rad: float,
    root_translation_scale_m: float,
    root_yaw_scale_rad: float,
) -> MotionClip:
    """Apply local ``R_new = R_old * Exp(delta)`` and smooth root residuals."""
    coefficients = layout.coefficient_matrix(values)
    basis = layout.basis(clip.num_frames)
    curves = basis @ coefficients.T
    local = np.asarray(clip.local_quat_wxyz, dtype=np.float64).copy()
    rotation_channels = len(layout.joint_indices) * 3
    joint_tangent = curves[:, :rotation_channels].reshape(
        (clip.num_frames, len(layout.joint_indices), 3)
    )
    joint_delta = quaternion_exp(joint_tangent * rotation_scale_rad)
    for offset, joint in enumerate(layout.joint_indices):
        local[:, joint] = quaternion_multiply(local[:, joint], joint_delta[:, offset])

    root = np.asarray(clip.root_translation_m, dtype=np.float64).copy()
    root += curves[:, rotation_channels : rotation_channels + 3] * root_translation_scale_m
    yaw_tangent = np.zeros((clip.num_frames, 3), dtype=np.float64)
    yaw_tangent[:, 1] = curves[:, -1] * root_yaw_scale_rad
    root_joint = clip.skeleton.root_index
    local[:, root_joint] = quaternion_multiply(local[:, root_joint], quaternion_exp(yaw_tangent))
    metadata = dict(clip.metadata)
    metadata["optimization_parameterization"] = {
        "kind": "cubic_bspline_residual",
        "control_points": layout.control_points,
        "joint_roles": list(layout.joint_roles),
        "rotation_composition": "R_new=R_old*Exp(delta_omega)",
        "root_channels": ["translation_x", "translation_y", "translation_z", "yaw"],
    }
    return clip.with_updates(
        local_quat_wxyz=local.astype(np.float32),
        root_translation_m=root.astype(np.float32),
        metadata=metadata,
    )


def refine_coefficients(
    values: ArrayLike,
    coarse: SplineResidualLayout,
    refined: SplineResidualLayout,
) -> NDArray[np.float64]:
    """Least-squares lift a coarse residual curve into a refined control lattice."""
    if coarse.channel_names != refined.channel_names:
        raise ValueError("coarse and refined layouts must have identical channels")
    sample_count = max(64, refined.control_points * 8)
    coarse_curve = coarse.basis(sample_count) @ coarse.coefficient_matrix(values).T
    refined_basis = refined.basis(sample_count)
    solution = np.linalg.lstsq(refined_basis, coarse_curve, rcond=None)[0].T
    return np.asarray(solution.reshape(-1), dtype=np.float64)
