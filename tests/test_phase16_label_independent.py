from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motionlab.io.npz import load_motion_npz
from motionlab.perceptual.candidate_pool import (
    CANDIDATE_POOL_VERSION,
    UNLABELED,
    _adversarial_optimizer_metadata,
    _anchor_dataset_source_paths,
    _frozen_render_hash,
    _PoolBuilder,
    _resolve_external_path,
    _validate_adversarial_optimizer_semantics,
    validate_unlabeled_candidate_pool,
)
from motionlab.perceptual.post_label_forensics import (
    _completed_raters,
    build_post_label_forensics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_MANIFEST = PROJECT_ROOT / "artifacts/phase15_mannequin_pilot/pilot_manifest.json"
PROTOCOL_FREEZE = PROJECT_ROOT / "artifacts/phase15_mannequin_pilot/protocol_freeze.json"
CANDIDATE_POOL = PROJECT_ROOT / "artifacts/phase16_label_independent/unlabeled_candidate_pool"


def _pilot_fixture() -> tuple[dict[str, object], dict[str, object], Path]:
    if not PILOT_MANIFEST.is_file() or not PROTOCOL_FREEZE.is_file():
        pytest.skip("frozen Phase 15 protocol artifacts are unavailable")
    manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    stimulus = manifest["stimuli"][0]
    root = Path(manifest["stimulus_directory"])
    return manifest, stimulus, root


def test_candidate_record_is_renderable_and_contains_no_preference(tmp_path: Path) -> None:
    _, stimulus, root = _pilot_fixture()
    motion = load_motion_npz(root / str(stimulus["motion"]))
    reference = load_motion_npz(root / str(stimulus["phase_reference_motion"]))
    freeze_id, render_hash = _frozen_render_hash(PROTOCOL_FREEZE)
    pool = tmp_path / "pool"
    pool.mkdir()
    builder = _PoolBuilder(pool, freeze_id=freeze_id, frozen_render_hash=render_hash)
    builder.add(
        motion,
        reference,
        population=UNLABELED,
        candidate_group="synthetic_fixture",
        mechanism="fixture_only",
        source_lineage={"source_clip_id": stimulus["source_id"]},
        parameter_metadata={"severity": "fixture"},
        deterministic_feasibility={"acceptance": {"all_required_satisfied": True}},
        origin_kind="deterministic_test_fixture",
        upstream_identifier="fixture-1",
    )
    assert len(builder.records) == 1
    (pool / "candidates.jsonl").write_text(
        json.dumps(builder.records[0], sort_keys=True) + "\n", encoding="utf-8"
    )
    (pool / "rejected.json").write_text("[]\n", encoding="utf-8")
    (pool / "summary.json").write_text(
        json.dumps(
            {
                "format_version": CANDIDATE_POOL_VERSION,
                "candidate_count": 1,
                "frozen_render_protocol_hash": render_hash,
                "protocol_freeze_id": freeze_id,
                "protocol_freeze": str(PROTOCOL_FREEZE.resolve()),
                "rejected_count": 0,
                "rejected_origin_counts": {},
                "population_counts": {UNLABELED: 1},
                "candidate_group_counts": {"synthetic_fixture": 1},
                "mechanism_counts": {"fixture_only": 1},
                "origin_counts": {"deterministic_test_fixture": 1},
                "unique_source_count": 1,
                "renderable_candidate_count": 1,
                "deterministically_feasible_candidate_count": 1,
                "adversarial_optimizer_evidence_record_count": 0,
                "mechanically_valid_uncanny_stress_candidate_count": 0,
                "phase16_perceptual_cma_invoked": False,
                "subjective_label_policy": {"upstream_pair_preferences_imported": False},
            }
        ),
        encoding="utf-8",
    )
    records = validate_unlabeled_candidate_pool(pool)
    assert records[0]["subjective_preference"] is None
    assert records[0]["renderability"]["protocol_freeze_id"] == freeze_id
    with pytest.raises(ValueError, match="real source clip lineage"):
        builder.add(
            motion,
            reference,
            population=UNLABELED,
            candidate_group="adversarial_optimizer_output",
            mechanism="preserved_critic_exploit",
            source_lineage={"corpus_record_id": "fixture-corpus-row"},
            parameter_metadata={},
            deterministic_feasibility={"acceptance": {"all_required_satisfied": False}},
            origin_kind="phase9_or_phase10_adversarial_cma_output",
            upstream_identifier="fixture-corpus-row",
            deduplicate=False,
        )


def test_adversarial_optimizer_evidence_is_content_addressed(tmp_path: Path) -> None:
    evidence_root = tmp_path / "upstream"
    evidence_root.mkdir()
    source_run = evidence_root / "source_run.json"
    trajectory = evidence_root / "trajectory.json"
    run_config = evidence_root / "run_config.json"
    optimization = evidence_root / "optimization.json"
    coefficients = evidence_root / "coefficients.npz"
    red_team = {"mode": "red_team", "objective_weights": {"foot_slide": 1.0}}
    source_run.write_text(json.dumps(red_team), encoding="utf-8")
    history = [{"generation": 0}]
    trajectory.write_text(json.dumps(history), encoding="utf-8")
    run_config.write_text(
        json.dumps(
            {
                "config": {
                    "seed": 3301,
                    "objective_weights": {"foot_slide": 1.0},
                    "refined_control_points": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    optimization.write_text(
        json.dumps(
            {
                **red_team,
                "history": history,
                "refined_control_points": 2,
                "artifacts": {"coefficients": str(coefficients)},
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        coefficients,
        coefficients=np.zeros(4, dtype=np.float64),
        channel_names=np.asarray(["left_foot", "right_foot"]),
        control_points=np.asarray(2),
    )
    output = tmp_path / "pool"
    builder = _PoolBuilder(output, freeze_id="freeze", frozen_render_hash="render")
    metadata = _adversarial_optimizer_metadata(
        builder,
        {
            "seed": 3301,
            "source_run": str(source_run),
            "trajectory": str(trajectory),
            "original_selected_family": "foot_slide",
            "artifacts": {
                "run_config.json": str(run_config),
                "optimization.json": str(optimization),
                "final_spline_coefficients.npz": str(coefficients),
            },
        },
        tmp_path / "manifest.json",
    )
    assert metadata["seed"] == 3301
    assert set(metadata["artifacts"]) == {
        "source_run",
        "trajectory",
        "run_config",
        "optimization",
        "spline_coefficients",
    }
    for evidence in metadata["artifacts"].values():
        copied = output / evidence["path"]
        assert copied.is_file()
        assert copied.stat().st_size == evidence["byte_count"]


def test_external_evidence_resolution_is_anchored_to_manifest_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    manifest = repository / "artifacts" / "corpus" / "manifest.json"
    referenced = repository / "artifacts" / "run" / "optimization.json"
    manifest.parent.mkdir(parents=True)
    referenced.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    referenced.write_text("{}", encoding="utf-8")
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    assert (
        _resolve_external_path(
            "artifacts/run/optimization.json",
            manifest.resolve(),
        )
        == referenced.resolve()
    )
    escaped = tmp_path / "outside-evidence.json"
    escaped.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes allowed root"):
        _resolve_external_path(str(escaped.resolve()), manifest.resolve())


def test_source_dataset_paths_are_anchored_without_cwd_dependence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset_directory = repository / "artifacts" / "dataset"
    source = repository / "artifacts" / "sources" / "clip.npz"
    dataset_directory.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture")
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    dataset = SimpleNamespace(records=[{"source_path": "artifacts/sources/clip.npz"}])

    _anchor_dataset_source_paths(dataset, dataset_directory)

    assert dataset.records[0]["source_path"] == str(source.resolve())


def test_canonical_adversarial_evidence_rejects_cross_candidate_swap() -> None:
    if not (CANDIDATE_POOL / "summary.json").is_file():
        pytest.skip("generated Phase 16 candidate pool is unavailable")
    records = validate_unlabeled_candidate_pool(CANDIDATE_POOL)
    adversarial = [row for row in records if row["adversarial_optimizer_output"]]
    assert len(adversarial) == 18
    assert len(list((CANDIDATE_POOL / "optimizer_evidence").iterdir())) == 72
    tampered = copy.deepcopy(adversarial[0])
    swapped_optimizer = copy.deepcopy(adversarial[1]["parameter_metadata"]["optimizer"])
    tampered["parameter_metadata"]["optimizer"] = swapped_optimizer

    with pytest.raises(ValueError, match="bound to a different candidate"):
        _validate_adversarial_optimizer_semantics(
            CANDIDATE_POOL,
            tampered,
            swapped_optimizer,
        )


def test_post_label_forensics_refuses_unrated_and_builds_after_dummy_rating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = _pilot_fixture()
    stimulus = next(
        item
        for item in manifest["stimuli"]
        if item.get("variant_id")
        and sum(
            other.get("variant_id") is not None and other.get("source_id") == item.get("source_id")
            for other in manifest["stimuli"]
        )
        >= 2
    )
    stimulus_id = str(stimulus["stimulus_id"])
    pilot_root = tmp_path / "synthetic_pilot"
    pilot_root.mkdir()
    pilot = pilot_root / "pilot_manifest.json"
    pilot.write_text(json.dumps(manifest), encoding="utf-8")
    observations = pilot_root / "raw_observations.jsonl"
    served_playlist = pilot_root / "served_playlist.jsonl"
    protocol_freeze = pilot_root / "protocol_freeze.json"
    protocol_freeze.write_text(
        json.dumps(
            {
                "pilot_manifest": pilot.name,
                "append_only_evidence_prefixes": {
                    "observations": {"path": observations.name},
                    "served_playlist": {"path": served_playlist.name},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "motionlab.perceptual.observation_auth.validate_active_evaluation_protocol",
        lambda *args, **kwargs: {"status": "valid"},
    )
    monkeypatch.setattr(
        "motionlab.perceptual.post_label_forensics.validate_current_protocol_observations",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "motionlab.perceptual.post_label_forensics._completed_raters",
        lambda *args, **kwargs: {"synthetic-rater"},
    )
    served_playlist.write_text("", encoding="utf-8")
    observations.write_text(
        json.dumps(
            {
                "format_version": "synthetic.test.observation.v1",
                "observation_id": "dummy-observation-1",
                "task_type": "naturalness",
                "rating": 4,
                "stimulus_ids": ["some-other-stimulus"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no valid completed observation"):
        build_post_label_forensics(
            pilot,
            observations,
            (stimulus_id,),
            tmp_path / "must-not-exist",
            protocol_freeze_path=protocol_freeze,
        )
    served = {
        "rater_id": "synthetic-rater",
        "session_id": "synthetic-session",
        "session_index": 0,
        "trial_id": "synthetic-trial",
        "trial_index": 0,
        "seed": manifest["seed"],
        "task_type": "naturalness",
        "presentation_order": [stimulus_id],
        "schedule_reason": "synthetic_fixture",
    }
    served_playlist.write_text(json.dumps(served) + "\n", encoding="utf-8")
    completed = {
        "format_version": "motionlab.subjective_observation.v2",
        "observation_id": "observation:sha256:" + "a" * 64,
        "render_protocol_hash": manifest["render_protocol_hash"],
        "task_type": "naturalness",
        "rating": 4,
        "pair_outcome_display_order": None,
        "pair_outcome_canonical": None,
        "stimulus_ids": [stimulus_id],
        "source_ids": [stimulus["source_id"]],
        "variant_ids": [stimulus["variant_id"]],
        "rater_id": served["rater_id"],
        "session_id": served["session_id"],
        "session_index": served["session_index"],
        "trial_id": served["trial_id"],
        "trial_index": served["trial_index"],
        "playlist_seed": served["seed"],
        "presentation_order": served["presentation_order"],
        "schedule_reason": served["schedule_reason"],
    }
    observations.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the active pilot"):
        build_post_label_forensics(
            pilot,
            observations,
            (stimulus_id,),
            pilot_root / "must-stay-blinded",
            protocol_freeze_path=protocol_freeze,
        )
    output = tmp_path / "forensics"
    report = build_post_label_forensics(
        pilot,
        observations,
        (stimulus_id,),
        output,
        protocol_freeze_path=protocol_freeze,
    )
    bundle = json.loads((output / "forensics.json").read_text(encoding="utf-8"))
    assert report["unlocked_by_observation_count"] == 1
    assert bundle["unlocking_observation_ids"] == [completed["observation_id"]]
    assert bundle["items"][0]["variant"]["contacts"]["available"] is True
    assert bundle["items"][0]["variant"]["joint_motion"]["angular_jerk_rad_s3"]
    html = (output / "viewer.html").read_text(encoding="utf-8")
    assert "Analysis-only view" in html
    assert "skeleton overlay" in html
    assert "root trajectory" in html

    second = next(
        item
        for item in manifest["stimuli"]
        if item["stimulus_id"] != stimulus_id
        and item["source_id"] == stimulus["source_id"]
        and item.get("variant_id")
    )
    second_id = str(second["stimulus_id"])
    pair_served = {
        **served,
        "trial_id": "synthetic-pair-trial",
        "trial_index": 1,
        "task_type": "pair_comparison",
        "presentation_order": [second_id, stimulus_id],
    }
    pair_completed = {
        **completed,
        "observation_id": "observation:sha256:" + "b" * 64,
        "trial_id": pair_served["trial_id"],
        "trial_index": pair_served["trial_index"],
        "task_type": "pair_comparison",
        "rating": None,
        "pair_outcome_display_order": "a_better",
        "pair_outcome_canonical": "second_better",
        "stimulus_ids": [stimulus_id, second_id],
        "presentation_order": pair_served["presentation_order"],
        # Real same-source adaptive trials persist one deduplicated source ID.
        "source_ids": [stimulus["source_id"]],
        "variant_ids": [stimulus["variant_id"], second["variant_id"]],
    }
    observations.write_text(json.dumps(pair_completed) + "\n", encoding="utf-8")
    served_playlist.write_text(
        json.dumps(pair_served) + "\n",
        encoding="utf-8",
    )
    pair_report = build_post_label_forensics(
        pilot,
        observations,
        (stimulus_id, second_id),
        tmp_path / "pair_forensics",
        protocol_freeze_path=protocol_freeze,
    )
    assert pair_report["unlocked_by_observation_count"] == 1
    with pytest.raises(ValueError, match="no valid completed observation"):
        build_post_label_forensics(
            pilot,
            observations,
            (stimulus_id,),
            tmp_path / "single_must_not_unlock_from_pair",
            protocol_freeze_path=protocol_freeze,
        )


def test_post_label_forensics_waits_until_every_observed_rater_is_complete() -> None:
    manifest = {
        "sessions": [
            {
                "trials": [
                    {"trial_id": "base-0"},
                    {"trial_id": "base-1"},
                ]
            }
        ],
        "adaptive_policy": {"enabled": True, "maximum_followups_total": 1},
    }
    partial = [
        {
            "rater_id": "rater-a",
            "trial_id": "base-0",
            "schedule_reason": "base",
        },
        {
            "rater_id": "rater-a",
            "trial_id": "base-1",
            "schedule_reason": "base",
        },
    ]
    with pytest.raises(ValueError, match="pilot must be complete"):
        _completed_raters(manifest, partial)
    complete = [
        *partial,
        {
            "rater_id": "rater-a",
            "trial_id": "adaptive-0",
            "schedule_reason": "adaptive:fixture",
        },
    ]
    assert _completed_raters(manifest, complete) == {"rater-a"}
    with pytest.raises(ValueError, match="every observed rater"):
        _completed_raters(
            manifest,
            [
                *complete,
                {
                    "rater_id": "rater-b",
                    "trial_id": "base-0",
                    "schedule_reason": "base",
                },
            ],
        )
