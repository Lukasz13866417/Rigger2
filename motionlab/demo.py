"""Runnable deterministic vertical slice using only generated fixture data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from motionlab.corruptions.foot_slide import corrupt_foot_slide_root_drift
from motionlab.io.npz import save_motion_npz
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.metrics.artifacts import save_deterministic_report
from motionlab.processing.contacts import ContactConfig
from motionlab.repair.foot_lock import lock_foot
from motionlab.testing.synthetic import make_synthetic_contact_walk


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def run_deterministic_demo(
    workspace: Path,
    *,
    render_previews: bool = True,
) -> dict[str, Any]:
    """Create, corrupt, repair, grade, and serialize a deterministic contact walk."""
    output = Path(workspace)
    output.mkdir(parents=True, exist_ok=True)
    clean = make_synthetic_contact_walk(num_cycles=2, fps=60.0)
    corruption = corrupt_foot_slide_root_drift(
        clean,
        start_frame=5,
        stop_frame=25,
        distance_m=0.04,
        direction_world=(1.0, 0.0, 0.0),
        seed=1234,
    )
    repaired_result = lock_foot(
        corruption.corrupted,
        side="left",
        start_frame=5,
        stop_frame=25,
        anchor_mode="first",
        lock_strength=1.0,
        blend_in_frames=3,
        blend_out_frames=3,
    )
    repaired = repaired_result.repaired

    # The demo knows the interval is intended stance. A relaxed speed threshold keeps the
    # corrupted planted foot observable instead of relabeling a severe slide as swing.
    contact_config = ContactConfig(speed_enter_mps=0.6, speed_exit_mps=0.7)
    clean_report = grade_motion_deterministic(
        clean,
        target_speed_mps=1.0,
        contact_config=contact_config,
    )
    corrupt_report = grade_motion_deterministic(
        corruption.corrupted,
        target_speed_mps=1.0,
        contact_config=contact_config,
    )
    repaired_report = grade_motion_deterministic(
        repaired,
        target_speed_mps=1.0,
        contact_config=contact_config,
    )

    motions = {
        "clean": output / "clean.npz",
        "corrupted": output / "corrupted.npz",
        "repaired": output / "repaired.npz",
    }
    reports = {
        "clean": output / "clean_report.json",
        "corrupted": output / "corrupted_report.json",
        "repaired": output / "repaired_report.json",
    }
    dense_reports = {
        "clean": output / "clean_report.dense.npz",
        "corrupted": output / "corrupted_report.dense.npz",
        "repaired": output / "repaired_report.dense.npz",
    }
    for name, clip in (
        ("clean", clean),
        ("corrupted", corruption.corrupted),
        ("repaired", repaired),
    ):
        save_motion_npz(motions[name], clip)
    for name, report in (
        ("clean", clean_report),
        ("corrupted", corrupt_report),
        ("repaired", repaired_report),
    ):
        save_deterministic_report(reports[name], report, dense_path=dense_reports[name])

    preview_paths: dict[str, str] = {}
    preview_warning: str | None = None
    if render_previews:
        try:
            from motionlab.visualization.skeleton_plot import save_skeleton_preview

            for name, clip in (
                ("clean", clean),
                ("corrupted", corruption.corrupted),
                ("repaired", repaired),
            ):
                preview = output / f"{name}_preview.png"
                save_skeleton_preview(clip, preview, frame=15)
                preview_paths[name] = str(preview)
        except RuntimeError as exc:
            preview_warning = str(exc)

    history = [
        {
            "motion_id": clean.content_hash,
            "parent_motion_id": None,
            "operation": "make_synthetic_contact_walk",
            "parameters": {"num_cycles": 2, "fps": 60.0},
        },
        {
            "motion_id": corruption.corrupted.content_hash,
            "parent_motion_id": clean.content_hash,
            "operation": "corrupt_foot_slide_root_drift",
            "parameters": dict(corruption.parameters),
        },
        {
            "motion_id": repaired.content_hash,
            "parent_motion_id": corruption.corrupted.content_hash,
            "operation": "lock_foot",
            "parameters": dict(repaired.metadata["operator_parameters"]),
            "metric_before": {"left_foot_slip_cm": repaired_result.before_slip_cm},
            "metric_after": {"left_foot_slip_cm": repaired_result.after_slip_cm},
        },
    ]
    _write_json(output / "history.json", history)
    reduction = 1.0 - repaired_result.after_slip_cm / repaired_result.before_slip_cm
    summary: dict[str, Any] = {
        "workspace": str(output),
        "motion_ids": {
            name: clip.content_hash
            for name, clip in (
                ("clean", clean),
                ("corrupted", corruption.corrupted),
                ("repaired", repaired),
            )
        },
        "motions": {name: str(path) for name, path in motions.items()},
        "reports": {name: str(path) for name, path in reports.items()},
        "dense_reports": {name: str(path) for name, path in dense_reports.items()},
        "previews": preview_paths,
        "history": str(output / "history.json"),
        "foot_lock": {
            "before_slip_cm": repaired_result.before_slip_cm,
            "after_slip_cm": repaired_result.after_slip_cm,
            "reduction_fraction": reduction,
            "maximum_ik_residual_m": repaired_result.maximum_ik_residual_m,
            "maximum_ik_target_residual_m": repaired_result.maximum_ik_target_residual_m,
            "actual_post_repair_marker_slip_cm": (
                repaired_result.actual_post_repair_marker_slip_cm
            ),
        },
        "warnings": [] if preview_warning is None else [preview_warning],
    }
    _write_json(output / "summary.json", summary)
    return summary
