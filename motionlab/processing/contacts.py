"""Confidence-aware heuristic foot contact detection on a general ground plane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from motionlab.kinematics.fk import marker_world_positions
from motionlab.math.finite_difference import finite_difference
from motionlab.motion._validation import frozen_array
from motionlab.motion.clip import MotionClip
from motionlab.processing._signals import clean_contact_mask

FOOT_MARKER_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")


class ContactConfig(BaseModel):
    """Configurable physical thresholds for heuristic marker contact detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    height_enter_m: float = Field(default=0.035, ge=0.0)
    height_exit_m: float = Field(default=0.050, ge=0.0)
    speed_enter_mps: float = Field(default=0.15, ge=0.0)
    speed_exit_mps: float = Field(default=0.22, ge=0.0)
    min_contact_frames: int = Field(default=3, ge=1)
    close_gap_frames: int = Field(default=2, ge=0)
    reference_leg_length_m: float = Field(default=0.9, gt=0.0)
    scale_thresholds_by_leg_length: bool = True
    ground_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def exit_thresholds_are_relaxed(self) -> ContactConfig:
        """Require hysteresis exit thresholds to be no stricter than entry thresholds."""
        if self.height_exit_m < self.height_enter_m:
            raise ValueError("height_exit_m must be >= height_enter_m")
        if self.speed_exit_mps < self.speed_enter_mps:
            raise ValueError("speed_exit_mps must be >= speed_enter_mps")
        return self


@dataclass(frozen=True)
class ContactResult:
    """Hard and soft marker contacts with physical evidence arrays."""

    marker_names: tuple[str, ...]
    hard: NDArray[np.bool_]
    confidence: NDArray[np.float32]
    height_m: NDArray[np.float32]
    tangential_speed_mps: NDArray[np.float32]
    ground_plane: NDArray[np.float32]
    source: str
    ground_confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hard = np.ascontiguousarray(self.hard, dtype=np.bool_)
        confidence = frozen_array(
            self.confidence,
            dtype=np.dtype(np.float32),
            name="contact confidence",
        )
        height = frozen_array(self.height_m, dtype=np.dtype(np.float32), name="contact height")
        speed = frozen_array(
            self.tangential_speed_mps,
            dtype=np.dtype(np.float32),
            name="contact tangential speed",
        )
        plane = frozen_array(
            self.ground_plane,
            dtype=np.dtype(np.float32),
            name="ground plane",
        )
        if hard.ndim != 2 or hard.shape[1] != len(self.marker_names):
            raise ValueError("hard contact array must have shape [T, number of markers]")
        expected = (hard.shape[0], len(self.marker_names))
        if confidence.shape != expected or height.shape != expected or speed.shape != expected:
            raise ValueError("contact evidence arrays must have the same [T,M] shape")
        if plane.shape != (4,):
            raise ValueError("ground_plane must have shape [4]")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("contact confidence values must be within [0,1]")
        if not np.isfinite(self.ground_confidence) or not 0.0 <= self.ground_confidence <= 1.0:
            raise ValueError("ground_confidence must be finite and within [0,1]")
        hard.setflags(write=False)
        object.__setattr__(self, "hard", hard)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "tangential_speed_mps", speed)
        object.__setattr__(self, "ground_plane", plane)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def normalize_ground_plane(ground_plane: ArrayLike) -> NDArray[np.float64]:
    """Normalize plane coefficients ``n dot x + d = 0`` to a unit normal."""
    plane = np.asarray(ground_plane, dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError("ground_plane must be a finite array with shape [4]")
    normal_length = np.linalg.norm(plane[:3])
    if normal_length < 1.0e-8:
        raise ValueError("ground plane normal must be nonzero")
    return plane / normal_length


def _leg_length_m(clip: MotionClip) -> float | None:
    lengths: list[float] = []
    for side in ("left", "right"):
        indices: list[int] = []
        for role in (f"{side}_hip", f"{side}_knee", f"{side}_ankle"):
            index = clip.skeleton.roles.get(role)
            if index is None:
                indices = []
                break
            indices.append(index)
        if indices:
            lengths.append(
                float(np.linalg.norm(clip.skeleton.rest_offsets_m[indices[1]]))
                + float(np.linalg.norm(clip.skeleton.rest_offsets_m[indices[2]]))
            )
    return None if not lengths else float(np.mean(lengths))


def _marker_velocity(position: NDArray[np.float64], fps: float) -> NDArray[np.float64]:
    return finite_difference(position, 1.0 / fps, time_axis=0)


def _sigmoid(value: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def detect_foot_contacts(
    clip: MotionClip,
    *,
    ground_plane: ArrayLike = (0.0, 1.0, 0.0, 0.0),
    config: ContactConfig | None = None,
) -> ContactResult:
    """Detect heel/toe contacts from plane-relative height and tangential speed."""
    settings = ContactConfig() if config is None else config
    plane = normalize_ground_plane(ground_plane)
    missing = [name for name in FOOT_MARKER_NAMES if name not in clip.skeleton.markers]
    if missing:
        raise ValueError(f"contact detection requires marker definitions: {missing}")
    markers = marker_world_positions(clip, FOOT_MARKER_NAMES)
    marker_position = np.stack([markers[name] for name in FOOT_MARKER_NAMES], axis=1)
    signed_height = marker_position @ plane[:3] + plane[3]
    velocity = np.stack(
        [_marker_velocity(marker_position[:, marker], clip.fps) for marker in range(4)],
        axis=1,
    )
    normal_velocity = np.sum(velocity * plane[:3], axis=-1, keepdims=True) * plane[:3]
    tangential_speed = np.linalg.norm(velocity - normal_velocity, axis=-1)

    threshold_scale = 1.0
    leg_length = _leg_length_m(clip)
    if settings.scale_thresholds_by_leg_length and leg_length is not None:
        threshold_scale = leg_length / settings.reference_leg_length_m
    height_enter = settings.height_enter_m * threshold_scale
    height_exit = settings.height_exit_m * threshold_scale
    speed_enter = settings.speed_enter_mps * threshold_scale
    speed_exit = settings.speed_exit_mps * threshold_scale

    hard = np.zeros_like(signed_height, dtype=np.bool_)
    for marker in range(len(FOOT_MARKER_NAMES)):
        active = False
        for frame in range(clip.num_frames):
            height_threshold = height_exit if active else height_enter
            speed_threshold = speed_exit if active else speed_enter
            active = bool(
                signed_height[frame, marker] <= height_threshold
                and tangential_speed[frame, marker] <= speed_threshold
            )
            hard[frame, marker] = active
        hard[:, marker] = clean_contact_mask(
            hard[:, marker],
            min_true_frames=settings.min_contact_frames,
            max_false_gap_frames=settings.close_gap_frames,
        )

    height_width = max(height_exit - height_enter, 0.01 * threshold_scale)
    speed_width = max(speed_exit - speed_enter, 0.04 * threshold_scale)
    height_score = _sigmoid((height_exit - signed_height) / height_width)
    speed_score = _sigmoid((speed_exit - tangential_speed) / speed_width)
    confidence = height_score * speed_score * settings.ground_confidence
    return ContactResult(
        marker_names=FOOT_MARKER_NAMES,
        hard=hard,
        confidence=confidence.astype(np.float32),
        height_m=signed_height.astype(np.float32),
        tangential_speed_mps=tangential_speed.astype(np.float32),
        ground_plane=plane.astype(np.float32),
        source="heuristic_height_speed",
        ground_confidence=settings.ground_confidence,
        metadata={
            "height_enter_m": height_enter,
            "height_exit_m": height_exit,
            "speed_enter_mps": speed_enter,
            "speed_exit_mps": speed_exit,
            "min_contact_frames": settings.min_contact_frames,
            "close_gap_frames": settings.close_gap_frames,
            "threshold_scale": threshold_scale,
            "estimated_leg_length_m": leg_length,
        },
    )
