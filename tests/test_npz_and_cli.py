from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from motionlab.cli import app
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.metrics.artifacts import load_dense_metric_artifact
from motionlab.motion.clip import MotionClip
from motionlab.testing.synthetic import (
    make_nonidentity_target_humanoid,
    make_synthetic_contact_walk,
    make_synthetic_walk,
)

runner = CliRunner()


def test_npz_round_trip_preserves_clip_and_skeleton(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "walk.npz"
    original = make_synthetic_walk(num_frames=17, fps=48.0)
    save_motion_npz(path, original)
    loaded = load_motion_npz(path)
    assert loaded.content_hash == original.content_hash
    assert loaded.skeleton.content_hash == original.skeleton.content_hash
    assert loaded.skeleton.joint_names == original.skeleton.joint_names
    assert dict(loaded.skeleton.roles) == dict(original.skeleton.roles)
    assert dict(loaded.metadata) == dict(original.metadata)
    np.testing.assert_array_equal(loaded.local_quat_wxyz, original.local_quat_wxyz)
    np.testing.assert_array_equal(loaded.root_translation_m, original.root_translation_m)


def test_npz_rejects_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, format_version=np.asarray("unknown"))
    with pytest.raises(ValueError, match="missing required fields"):
        load_motion_npz(path)


def test_cli_help_and_config_validation() -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "make-synthetic" in help_result.stdout

    config_result = runner.invoke(app, ["check-config", "configs/default.yaml", "--json"])
    assert config_result.exit_code == 0
    payload = json.loads(config_result.stdout)
    assert payload["motion"]["fps"] == 60.0


def test_cli_synthetic_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.npz"
    create_result = runner.invoke(
        app,
        ["make-synthetic", str(output), "--frames", "19", "--fps", "30"],
    )
    assert create_result.exit_code == 0
    validation = runner.invoke(app, ["validate-motion", str(output), "--json"])
    assert validation.exit_code == 0
    payload = json.loads(validation.stdout)
    assert payload["frames"] == 19
    assert payload["joints"] == 16
    assert payload["fps"] == 30.0


def test_cli_inspects_and_converts_bvh(tmp_path: Path) -> None:
    fixture = "tests/fixtures/minimal_zxy.bvh"
    inspect_result = runner.invoke(app, ["inspect-bvh", fixture, "--json"])
    assert inspect_result.exit_code == 0
    inspected = json.loads(inspect_result.stdout)
    assert inspected["joint_names"] == ["Hips", "Knee"]
    assert inspected["channel_order"][0][-3:] == ["Zrotation", "Xrotation", "Yrotation"]

    output = tmp_path / "converted.npz"
    convert_result = runner.invoke(
        app,
        ["convert-bvh", fixture, "--output", str(output), "--target-fps", "20"],
    )
    assert convert_result.exit_code == 0
    converted = load_motion_npz(output)
    assert converted.num_frames == 3
    assert converted.fps == 20.0
    provenance = converted.metadata["provenance"]
    assert provenance["source_dataset"] == "unspecified_bvh"
    assert provenance["source_take_id"].startswith("sha256:")
    assert provenance["split_lineage_id"].startswith("split:sha256:")
    assert [step["operation"] for step in provenance["operations"]] == [
        "source",
        "resample_motion",
    ]


def test_cli_writes_deterministic_metric_report(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    report = tmp_path / "reports" / "metrics.json"
    save_motion_npz(motion, make_synthetic_walk(num_frames=31, fps=30.0))
    result = runner.invoke(
        app,
        [
            "metrics",
            str(motion),
            "--output",
            str(report),
            "--target-speed-mps",
            "1.2",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(report.read_text())
    assert payload["motion_id"] == load_motion_npz(motion).content_hash
    assert payload["metrics"]["root_speed"]["clip_value"] == pytest.approx(1.2)
    assert "frame_values" not in payload["metrics"]["root_speed"]
    dense_path = report.with_name("metrics.dense.npz")
    assert dense_path.is_file()
    artifact = load_dense_metric_artifact(
        dense_path,
        expected_motion_id=payload["motion_id"],
        expected_sha256=payload["metadata"]["dense_artifact"]["sha256"],
    )
    assert artifact.motion_id == payload["motion_id"]
    dense_key = payload["metrics"]["root_speed"]["metadata"]["dense_fields"]["frame_values"]
    np.testing.assert_allclose(artifact.arrays[dense_key], 1.2, atol=1.0e-5)
    assert payload["metadata"]["dense_artifact"]["path"] == "metrics.dense.npz"
    dense_path.write_bytes(dense_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_dense_metric_artifact(
            dense_path,
            expected_sha256=payload["metadata"]["dense_artifact"]["sha256"],
        )


def test_cli_corrupt_and_repair_round_trip(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean.npz"
    bad_path = tmp_path / "bad.npz"
    repaired_path = tmp_path / "repaired.npz"
    save_motion_npz(clean_path, make_synthetic_contact_walk())
    corrupt_result = runner.invoke(
        app,
        [
            "corrupt",
            str(clean_path),
            "--type",
            "foot_slide",
            "--start-frame",
            "5",
            "--stop-frame",
            "25",
            "--distance-m",
            "0.04",
            "--output",
            str(bad_path),
        ],
    )
    assert corrupt_result.exit_code == 0
    repair_result = runner.invoke(
        app,
        [
            "repair",
            str(bad_path),
            "--operator",
            "lock_foot",
            "--side",
            "left",
            "--start-frame",
            "5",
            "--stop-frame",
            "25",
            "--output",
            str(repaired_path),
        ],
    )
    assert repair_result.exit_code == 0
    result_payload = json.loads(repair_result.stdout)
    assert (
        result_payload["metrics"]["after_slip_cm"]
        <= 0.1 * result_payload["metrics"]["before_slip_cm"]
    )
    assert load_motion_npz(repaired_path).metadata["operation"] == "lock_foot"
    assert result_payload["metrics"]["maximum_ik_target_residual_m"] < 1.0e-4
    assert result_payload["metrics"]["actual_post_repair_marker_slip_cm"] == pytest.approx(
        result_payload["metrics"]["after_slip_cm"]
    )


def test_cli_retargets_to_nonidentity_reference_skeleton(tmp_path: Path) -> None:
    source = make_synthetic_walk(num_frames=17)
    target_skeleton = make_nonidentity_target_humanoid(scale=1.25)
    target_reference = MotionClip(
        target_skeleton,
        np.broadcast_to(
            target_skeleton.rest_local_quat_wxyz,
            (2, target_skeleton.num_joints, 4),
        ),
        np.zeros((2, 3)),
        fps=source.fps,
    )
    source_path = tmp_path / "source.npz"
    target_path = tmp_path / "target_reference.npz"
    output_path = tmp_path / "retargeted.npz"
    save_motion_npz(source_path, source)
    save_motion_npz(target_path, target_reference)
    result = runner.invoke(
        app,
        ["retarget", str(source_path), str(target_path), "--output", str(output_path)],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    retargeted = load_motion_npz(output_path)
    assert retargeted.skeleton.content_hash == target_skeleton.content_hash
    assert retargeted.num_frames == source.num_frames
    assert payload["root_scale_ratio"] == pytest.approx(1.25)
