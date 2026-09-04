"""Minimal static 3D skeleton preview for coordinate and hierarchy debugging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from motionlab.kinematics.fk import forward_kinematics_numpy, marker_world_positions
from motionlab.motion.clip import MotionClip


def save_skeleton_preview(clip: MotionClip, output: Path, *, frame: int = 0) -> None:
    """Render one frame, hierarchy edges, root trajectory, and virtual markers to an image."""
    if frame < 0 or frame >= clip.num_frames:
        raise ValueError(f"frame must be in [0, {clip.num_frames}), got {frame}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "preview requires the visualization extra: uv sync --extra visualization"
        ) from exc

    positions, _ = forward_kinematics_numpy(
        clip.skeleton,
        clip.local_quat_wxyz,
        clip.root_translation_m,
    )
    figure = plt.figure(figsize=(7, 7))
    axes = figure.add_subplot(111, projection="3d")
    pose = positions[frame]
    for joint, parent in enumerate(clip.skeleton.parents):
        if parent >= 0:
            segment = pose[[int(parent), joint]]
            axes.plot(segment[:, 0], segment[:, 2], segment[:, 1], color="black", linewidth=2)
    axes.scatter(pose[:, 0], pose[:, 2], pose[:, 1], color="tab:blue", s=18)
    axes.plot(
        clip.root_translation_m[:, 0],
        clip.root_translation_m[:, 2],
        clip.root_translation_m[:, 1],
        color="tab:orange",
        alpha=0.6,
        label="root trajectory",
    )
    if clip.skeleton.markers:
        markers = marker_world_positions(clip)
        marker_pose = np.stack([values[frame] for values in markers.values()])
        axes.scatter(
            marker_pose[:, 0],
            marker_pose[:, 2],
            marker_pose[:, 1],
            color="tab:red",
            marker="x",
            s=45,
            label="markers",
        )

    all_points: NDArray[np.float64] = np.asarray(
        np.concatenate((positions.reshape((-1, 3)), clip.root_translation_m), axis=0),
        dtype=np.float64,
    )
    center = np.mean(all_points, axis=0)
    extent = max(float(np.ptp(all_points, axis=0).max()), 1.0) * 0.55
    axes.set_xlim(center[0] - extent, center[0] + extent)
    axes.set_ylim(center[2] - extent, center[2] + extent)
    axes.set_zlim(max(0.0, center[1] - extent), center[1] + extent)
    axes.set_xlabel("+X right (m)")
    axes.set_ylabel("+Z forward (m)")
    axes.set_zlabel("+Y up (m)")
    axes.set_title(f"MotionLab frame {frame}/{clip.num_frames - 1}")
    axes.legend(loc="upper right")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)


def save_motion_comparison_strip(
    reference: MotionClip,
    candidate: MotionClip,
    output: Path,
    *,
    frame_count: int = 6,
) -> None:
    """Render a compact, shared-scale overlay of two matching motion clips."""
    if reference.skeleton.content_hash != candidate.skeleton.content_hash:
        raise ValueError("comparison clips must use the same skeleton")
    if reference.num_frames != candidate.num_frames:
        raise ValueError("comparison clips must have the same frame count")
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "comparison requires the visualization extra: uv sync --extra visualization"
        ) from exc

    reference_position, _ = forward_kinematics_numpy(
        reference.skeleton,
        reference.local_quat_wxyz,
        reference.root_translation_m,
    )
    candidate_position, _ = forward_kinematics_numpy(
        candidate.skeleton,
        candidate.local_quat_wxyz,
        candidate.root_translation_m,
    )
    frames = np.linspace(0, reference.num_frames - 1, frame_count).round().astype(int)
    points = np.concatenate((reference_position[frames], candidate_position[frames]), axis=0)
    minimum = points.reshape((-1, 3)).min(axis=0)
    maximum = points.reshape((-1, 3)).max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = max(float(np.max(maximum - minimum)), 0.5) * 0.55
    figure = plt.figure(figsize=(3.0 * frame_count, 3.2))
    for column, frame in enumerate(frames, start=1):
        axes = figure.add_subplot(1, frame_count, column, projection="3d")
        for pose, color, label in (
            (reference_position[frame], "tab:blue", "reference"),
            (candidate_position[frame], "tab:red", "candidate"),
        ):
            for joint, parent in enumerate(reference.skeleton.parents):
                if parent >= 0:
                    segment = pose[[int(parent), joint]]
                    axes.plot(
                        segment[:, 0],
                        segment[:, 2],
                        segment[:, 1],
                        color=color,
                        linewidth=1.4,
                        alpha=0.75,
                    )
            axes.scatter(pose[:, 0], pose[:, 2], pose[:, 1], color=color, s=6, label=label)
        axes.set_xlim(center[0] - extent, center[0] + extent)
        axes.set_ylim(center[2] - extent, center[2] + extent)
        axes.set_zlim(center[1] - extent, center[1] + extent)
        axes.set_title(f"frame {frame}")
        axes.set_axis_off()
        if column == 1:
            axes.legend(loc="upper left", fontsize=7)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(figure)
