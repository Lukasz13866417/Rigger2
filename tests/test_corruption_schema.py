from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from motionlab.cli import app
from motionlab.core.provenance import assign_source_lineage
from motionlab.corruptions import (
    CORRUPTION_CATALOG,
    CORRUPTION_CHANNEL_NAMES,
    CORRUPTION_DEFECT_NAMES,
    CORRUPTION_PART_NAMES,
    corrupt_foot_slide_hip_rotation_drift,
    corrupt_foot_slide_root_drift,
    corrupt_ground_penetration_root_offset,
    corrupt_joint_pop_rotation_pulse,
    corrupt_loop_seam_end_rotation,
    load_corruption_artifact,
    save_corruption_artifact,
    verify_ordinal_chain,
)
from motionlab.io.npz import save_motion_npz
from motionlab.rigging.semantic_roles import FunctionalPart
from motionlab.testing.synthetic import make_synthetic_contact_walk

runner = CliRunner()


def _lineaged_clip():
    return assign_source_lineage(
        make_synthetic_contact_walk(),
        source_dataset="synthetic",
        source_clip_id="walk-001",
        source_take_id="take-001",
        split="train",
        clean_confidence=0.95,
    )


def test_corruption_result_separates_intervention_symptom_and_responsibility() -> None:
    clip = make_synthetic_contact_walk()
    result = corrupt_foot_slide_root_drift(
        clip,
        start_frame=5,
        stop_frame=25,
        distance_m=0.04,
        seed=7,
    )
    expected_semantic = (
        clip.num_frames,
        len(CORRUPTION_PART_NAMES),
        len(CORRUPTION_DEFECT_NAMES),
    )
    assert result.intervention_mask.shape == (
        clip.num_frames,
        clip.num_joints,
        len(CORRUPTION_CHANNEL_NAMES),
    )
    assert result.symptom_mask.shape == expected_semantic
    assert result.responsibility_target.shape == expected_semantic
    assert result.responsibility_confidence.shape == expected_semantic
    assert not result.intervention_mask.flags.writeable
    assert not result.symptom_mask.flags.writeable

    root = clip.skeleton.root_index
    assert np.all(np.flatnonzero(np.any(result.intervention_mask, axis=(0, 2))) == [root])
    root_part = int(FunctionalPart.ROOT_PELVIS) - 1
    left_leg = int(FunctionalPart.LEFT_LEG) - 1
    defect = CORRUPTION_DEFECT_NAMES.index("foot_slide")
    assert not np.any(result.symptom_mask[:, root_part, defect])
    assert np.any(result.symptom_mask[:, left_leg, defect])
    assert np.any(result.responsibility_target[:, root_part, defect] > 0.0)
    assert result.postcondition.hard_negative
    assert result.measured_severity == pytest.approx(result.postcondition.delta)
    assert result.severity_parameter == 0.04
    assert result.corrupted is result.corrupted_motion
    assert result.repair_target is clip


def test_contact_truth_is_separate_from_corruption_sensitive_inference() -> None:
    result = corrupt_foot_slide_root_drift(
        make_synthetic_contact_walk(),
        start_frame=5,
        stop_frame=25,
        distance_m=0.08,
    )
    contacts = result.contact_supervision
    assert contacts is not None
    np.testing.assert_array_equal(contacts.truth_hard, contacts.inferred_clean_hard)
    assert not np.array_equal(contacts.truth_hard, contacts.inferred_corrupted_hard)
    assert contacts.truth_source.startswith("clean_inference_proxy:")
    assert contacts.inference_source == "heuristic_height_speed"


def test_catalog_mechanisms_are_measured_and_hold_out_one_foot_slide_variant() -> None:
    clip = make_synthetic_contact_walk()
    results = (
        corrupt_foot_slide_hip_rotation_drift(
            clip,
            side="left",
            start_frame=5,
            stop_frame=25,
            angle_rad=0.12,
        ),
        corrupt_ground_penetration_root_offset(
            clip,
            start_frame=5,
            stop_frame=25,
            depth_m=0.04,
        ),
        corrupt_joint_pop_rotation_pulse(
            clip,
            joint="left_knee",
            frame=20,
            angle_rad=0.25,
        ),
        corrupt_loop_seam_end_rotation(
            clip,
            joint="chest",
            seam_window_frames=12,
            angle_rad=0.3,
        ),
    )
    assert all(result.postcondition.hard_negative for result in results)
    assert all(result.postcondition.after > result.postcondition.before for result in results)
    assert results[0].catalog_partition == "heldout_eval"
    foot_entries = [entry for entry in CORRUPTION_CATALOG if entry.family == "foot_slide"]
    assert {entry.partition for entry in foot_entries} == {"train", "heldout_eval"}
    assert len({entry.mechanism for entry in foot_entries}) >= 2


def test_ordinal_chain_uses_measured_outcomes_not_requested_parameters() -> None:
    clip = make_synthetic_contact_walk()
    mild = corrupt_foot_slide_root_drift(
        clip,
        start_frame=5,
        stop_frame=25,
        distance_m=0.02,
    )
    severe = corrupt_foot_slide_root_drift(
        clip,
        start_frame=5,
        stop_frame=25,
        distance_m=0.06,
    )
    chain = verify_ordinal_chain((mild, severe))
    assert chain.ordered_motion_ids[0] == clip.content_hash
    assert chain.measured_values[0] == mild.postcondition.before
    assert chain.measured_values[1] < chain.measured_values[2]
    with pytest.raises(ValueError, match="measured defect values"):
        verify_ordinal_chain((severe, mild))


def test_corruption_artifact_requires_lineage_and_round_trips_pickle_free(tmp_path: Path) -> None:
    untagged = corrupt_foot_slide_root_drift(
        make_synthetic_contact_walk(),
        start_frame=5,
        stop_frame=25,
        distance_m=0.04,
    )
    with pytest.raises(ValueError, match="complete source and split lineage"):
        save_corruption_artifact(tmp_path / "untagged", untagged)

    result = corrupt_foot_slide_root_drift(
        _lineaged_clip(),
        start_frame=5,
        stop_frame=25,
        distance_m=0.04,
    )
    artifact_dir = tmp_path / "artifact"
    manifest_path = save_corruption_artifact(artifact_dir, result)
    artifact = load_corruption_artifact(artifact_dir)
    assert manifest_path == artifact_dir / "manifest.json"
    assert artifact.clean_motion.content_hash == result.source_motion_id
    assert artifact.corrupted_motion.content_hash == result.corrupted_motion.content_hash
    np.testing.assert_array_equal(artifact.arrays["intervention_mask"], result.intervention_mask)
    assert "contact_truth_hard" in artifact.manifest["privileged_dense_fields"]
    assert "corruption_mechanism" in artifact.manifest["neural_input_exclusions"]
    assert artifact.manifest["source"]["clean_confidence"] == 0.95
    with np.load(artifact_dir / "labels.npz", allow_pickle=False) as archive:
        for key in archive.files:
            assert archive[key].dtype != np.dtype(object)

    labels = artifact_dir / "labels.npz"
    labels.write_bytes(labels.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_corruption_artifact(artifact_dir)


def test_cli_can_emit_lineage_safe_training_artifact_and_catalog(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    corrupted = tmp_path / "corrupted.npz"
    artifact_dir = tmp_path / "example"
    save_motion_npz(source, _lineaged_clip())
    result = runner.invoke(
        app,
        [
            "corrupt",
            str(source),
            "--start-frame",
            "5",
            "--stop-frame",
            "25",
            "--distance-m",
            "0.04",
            "--output",
            str(corrupted),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["hard_negative"] is True
    assert payload["artifact_manifest"] == str(artifact_dir / "manifest.json")
    load_corruption_artifact(artifact_dir)

    catalog_result = runner.invoke(app, ["corruption-catalog"])
    assert catalog_result.exit_code == 0
    catalog_payload = json.loads(catalog_result.stdout)
    assert len(catalog_payload["entries"]) == len(CORRUPTION_CATALOG)
    assert any(entry["partition"] == "heldout_eval" for entry in catalog_payload["entries"])
