"""Immutable fixed-rig animation clip representation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motionlab.constants import QUATERNION_NORM_TOLERANCE
from motionlab.motion._validation import frozen_array, frozen_json_mapping
from motionlab.motion.skeleton import Skeleton


@dataclass(frozen=True)
class MotionClip:
    """A sampled local-pose animation on one skeleton.

    ``local_quat_wxyz`` has shape ``[T,J,4]`` and contains absolute local orientations.
    ``root_translation_m`` has shape ``[T,3]`` in world metres.
    """

    skeleton: Skeleton
    local_quat_wxyz: NDArray[np.float32]
    root_translation_m: NDArray[np.float32]
    fps: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.skeleton, Skeleton):
            raise TypeError("skeleton must be a Skeleton")
        rotations = frozen_array(
            self.local_quat_wxyz,
            dtype=np.dtype(np.float32),
            name="local_quat_wxyz",
        )
        translations = frozen_array(
            self.root_translation_m,
            dtype=np.dtype(np.float32),
            name="root_translation_m",
        )
        if rotations.ndim != 3 or rotations.shape[1:] != (self.skeleton.num_joints, 4):
            raise ValueError(
                "local_quat_wxyz must have shape "
                f"[T,{self.skeleton.num_joints},4], got {rotations.shape}"
            )
        if rotations.shape[0] < 1:
            raise ValueError("motion must contain at least one frame")
        if translations.shape != (rotations.shape[0], 3):
            raise ValueError(
                f"root_translation_m must have shape ({rotations.shape[0]}, 3), "
                f"got {translations.shape}"
            )
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        norms = np.linalg.norm(rotations, axis=-1)
        if not np.allclose(norms, 1.0, atol=QUATERNION_NORM_TOLERANCE, rtol=0.0):
            raise ValueError("local_quat_wxyz must contain unit quaternions")

        object.__setattr__(self, "local_quat_wxyz", rotations)
        object.__setattr__(self, "root_translation_m", translations)
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "metadata", frozen_json_mapping(self.metadata, name="metadata"))

    @property
    def num_frames(self) -> int:
        """Return the number of sampled frames."""
        return int(self.local_quat_wxyz.shape[0])

    @property
    def num_joints(self) -> int:
        """Return the skeleton joint count."""
        return self.skeleton.num_joints

    @property
    def duration_s(self) -> float:
        """Return elapsed time from the first through the last sample."""
        return (self.num_frames - 1) / self.fps

    @property
    def timestamps_s(self) -> NDArray[np.float64]:
        """Return monotonically increasing sample timestamps in seconds."""
        return np.arange(self.num_frames, dtype=np.float64) / self.fps

    def copy(self) -> MotionClip:
        """Return a detached immutable copy."""
        return MotionClip(
            skeleton=self.skeleton,
            local_quat_wxyz=self.local_quat_wxyz.copy(),
            root_translation_m=self.root_translation_m.copy(),
            fps=self.fps,
            metadata=dict(self.metadata),
        )

    def slice_frames(self, start: int, stop: int) -> MotionClip:
        """Return a nonempty half-open frame slice ``[start, stop)``."""
        if start < 0 or stop > self.num_frames or start >= stop:
            raise ValueError(
                f"invalid nonempty frame slice [{start}, {stop}) for {self.num_frames} frames"
            )
        metadata = dict(self.metadata)
        metadata["source_frame_slice"] = [start, stop]
        metadata["parent_motion_id"] = self.content_hash
        from motionlab.core.provenance import append_operation_lineage

        metadata = append_operation_lineage(
            self,
            metadata,
            operation="slice_frames",
            kind="slice",
            parameters={"start_frame": start, "stop_frame": stop},
            version="motionlab-0.1.0",
        )
        return MotionClip(
            skeleton=self.skeleton,
            local_quat_wxyz=self.local_quat_wxyz[start:stop],
            root_translation_m=self.root_translation_m[start:stop],
            fps=self.fps,
            metadata=metadata,
        )

    def with_updates(
        self,
        *,
        local_quat_wxyz: NDArray[np.float32] | None = None,
        root_translation_m: NDArray[np.float32] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MotionClip:
        """Return a new validated clip containing selected replacements."""
        return MotionClip(
            skeleton=self.skeleton,
            local_quat_wxyz=(self.local_quat_wxyz if local_quat_wxyz is None else local_quat_wxyz),
            root_translation_m=(
                self.root_translation_m if root_translation_m is None else root_translation_m
            ),
            fps=self.fps,
            metadata=dict(self.metadata) if metadata is None else metadata,
        )

    @property
    def content_hash(self) -> str:
        """Return a stable SHA-256 identity for the skeleton, samples, and metadata."""
        digest = hashlib.sha256()
        digest.update(self.skeleton.content_hash.encode())
        digest.update(np.float64(self.fps).tobytes())
        for array in (self.local_quat_wxyz, self.root_translation_m):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes(order="C"))
        digest.update(
            json.dumps(dict(self.metadata), sort_keys=True, separators=(",", ":")).encode()
        )
        return f"sha256:{digest.hexdigest()}"
