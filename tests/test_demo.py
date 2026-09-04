from __future__ import annotations

import json
from pathlib import Path

from motionlab.demo import run_deterministic_demo
from motionlab.io.npz import load_motion_npz


def test_deterministic_demo_exports_reproducible_artifacts(tmp_path: Path) -> None:
    summary = run_deterministic_demo(tmp_path, render_previews=False)
    assert summary["foot_lock"]["reduction_fraction"] >= 0.90
    assert summary["foot_lock"]["maximum_ik_residual_m"] < 1.0e-4
    assert summary["foot_lock"]["maximum_ik_target_residual_m"] < 1.0e-4
    assert (
        summary["foot_lock"]["actual_post_repair_marker_slip_cm"]
        == summary["foot_lock"]["after_slip_cm"]
    )
    for name in ("clean", "corrupted", "repaired"):
        assert Path(summary["motions"][name]).is_file()
        assert Path(summary["reports"][name]).is_file()
        assert Path(summary["dense_reports"][name]).is_file()
        clip = load_motion_npz(Path(summary["motions"][name]))
        assert clip.content_hash == summary["motion_ids"][name]
    history = json.loads(Path(summary["history"]).read_text())
    assert [entry["operation"] for entry in history] == [
        "make_synthetic_contact_walk",
        "corrupt_foot_slide_root_drift",
        "lock_foot",
    ]
    assert (tmp_path / "summary.json").is_file()
