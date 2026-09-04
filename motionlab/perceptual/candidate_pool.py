"""Label-free perceptual candidates rendered under an immutable protocol contract.

This module is deliberately separate from the active subjective pilot.  It never reads or
writes observations and it does not build a playlist.  Every retained motion is either marked
``UNLABELED`` or ``CALIBRATION_ONLY``; upstream pair directions are intentionally discarded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.forensics import motions_from_sample
from motionlab.dataset.io import file_sha256
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.motion.clip import MotionClip
from motionlab.optimization.acceptance import (
    ProductionAcceptance,
    ProductionThresholds,
    assess_production_acceptance,
    declare_reference_speed_task,
)
from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.perceptual.perturbations import (
    PerturbationSpec,
    apply_perceptual_perturbation,
)
from motionlab.perceptual.protocol_freeze import validate_active_evaluation_protocol
from motionlab.perceptual.subjective import (
    MANNEQUIN_VIEWING_SETTINGS,
    _two_cycle_payload,
    render_protocol_hash,
)

CANDIDATE_POOL_VERSION = "motionlab.unlabeled_perceptual_candidate_pool.v2"
UNLABELED = "UNLABELED"
CALIBRATION_ONLY = "CALIBRATION_ONLY"

# Mid-band additions fill gaps between the Phase 12 variants.  None carries a preference.
ADDITIONAL_PERTURBATIONS = (
    PerturbationSpec("excessive_rigidification", 0.65, "train", "UNORDERED", None, (), None),
    PerturbationSpec("excessive_smoothing", 3.0, "train", "UNORDERED", None, (), None),
    PerturbationSpec("upper_body_phase_mismatch", 7.0, "train", "UNORDERED", None, (), None),
    PerturbationSpec("asymmetric_limb_timing", 4.0, "heldout", "UNORDERED", None, (), None),
    PerturbationSpec(
        "torso_counter_rotation_mismatch", 0.11, "heldout", "UNORDERED", None, (), None
    ),
    PerturbationSpec(
        "phase_aligned_style_inconsistency", 0.50, "train", "UNORDERED", None, (), None
    ),
    PerturbationSpec("weight_transfer_proxy", 0.006, "train", "UNORDERED", None, (), None),
)


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _acceptance_dict(value: ProductionAcceptance) -> dict[str, Any]:
    return {
        "raw_metrics": value.raw_metrics,
        "constraints": value.constraints,
        "hard_invalid_reasons": list(value.hard_invalid_reasons),
        "all_required_satisfied": value.all_required_satisfied,
        "satisficing_penalty": value.satisficing_penalty,
    }


def _persist_motion(output: Path, clip: MotionClip) -> str:
    relative = Path("motions") / f"{clip.content_hash.removeprefix('sha256:')}.npz"
    destination = output / relative
    if not destination.is_file():
        save_motion_npz(destination, clip)
    return relative.as_posix()


def _visual_motion_hash(clip: MotionClip) -> str:
    """Hash only render-relevant samples, excluding provenance and purpose metadata."""
    digest = hashlib.sha256()
    digest.update(clip.skeleton.content_hash.encode())
    digest.update(str(clip.fps).encode())
    for value in (clip.local_quat_wxyz, clip.root_translation_m):
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return f"visual-motion:sha256:{digest.hexdigest()}"


def _reference_root(manifest_path: Path) -> Path:
    resolved = manifest_path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    for parent in resolved.parents:
        if parent.name == "artifacts":
            return parent.parent
    return resolved.parent


def _resolve_external_path(raw: str, manifest_path: Path) -> Path:
    allowed_root = _reference_root(manifest_path)
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(f"referenced motion escapes allowed root: {raw}") from exc
        if resolved.is_file():
            return resolved
        raise ValueError(f"referenced motion does not exist: {raw}")
    bases: list[Path] = []
    current = manifest_path.resolve().parent
    while True:
        bases.append(current)
        if current == allowed_root:
            break
        if current.parent == current:
            raise ValueError("manifest is outside its declared reference root")
        current = current.parent
    for base in bases:
        anchored = (base / candidate).resolve()
        try:
            anchored.relative_to(allowed_root)
        except ValueError:
            continue
        if anchored.is_file():
            return anchored
    raise ValueError(f"referenced motion does not exist: {raw}")


def _persist_external_evidence(
    output: Path,
    raw: str,
    manifest_path: Path,
) -> dict[str, Any]:
    """Copy optimizer evidence into the pool under a content-addressed name."""
    source = _resolve_external_path(raw, manifest_path)
    digest = file_sha256(source)
    suffix = "".join(source.suffixes) or ".bin"
    relative = Path("optimizer_evidence") / f"{digest}{suffix}"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        shutil.copyfile(source, destination)
    if file_sha256(destination) != digest:
        raise ValueError(f"copied optimizer evidence digest changed: {source}")
    return {
        "path": relative.as_posix(),
        "sha256": digest,
        "byte_count": destination.stat().st_size,
        "upstream_reference": raw,
    }


def _adversarial_optimizer_metadata(
    builder: _PoolBuilder,
    record: Mapping[str, Any],
    adversarial_manifest: Path,
) -> dict[str, Any]:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("adversarial corpus record has no optimizer artifacts")
    references = {
        "source_run": record.get("source_run"),
        "trajectory": record.get("trajectory"),
        "run_config": artifacts.get("run_config.json"),
        "optimization": artifacts.get("optimization.json"),
        "spline_coefficients": artifacts.get("final_spline_coefficients.npz"),
    }
    missing = [
        name for name, value in references.items() if not isinstance(value, str) or not value
    ]
    if missing:
        raise ValueError(f"adversarial corpus record lacks optimizer evidence: {missing}")
    seed = record.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("adversarial corpus record has no integer optimizer seed")
    selected_family = str(record.get("original_selected_family") or "")
    if not selected_family:
        raise ValueError("adversarial corpus record has no selected optimizer family")
    persisted = {
        name: _persist_external_evidence(
            builder.output,
            str(raw),
            adversarial_manifest,
        )
        for name, raw in references.items()
    }
    run_config = json.loads(
        (builder.output / persisted["run_config"]["path"]).read_text(encoding="utf-8")
    )
    config = run_config.get("config") if isinstance(run_config, dict) else None
    if (
        not isinstance(config, dict)
        or config.get("seed") != seed
        or set(config.get("objective_weights", {})) != {selected_family}
    ):
        raise ValueError("adversarial run configuration disagrees with corpus metadata")
    optimization_report: dict[str, Any] | None = None
    for name in ("source_run", "optimization"):
        report = json.loads((builder.output / persisted[name]["path"]).read_text(encoding="utf-8"))
        if (
            not isinstance(report, dict)
            or report.get("mode") != "red_team"
            or set(report.get("objective_weights", {})) != {selected_family}
        ):
            raise ValueError(f"adversarial {name} disagrees with corpus metadata")
        if name == "optimization":
            optimization_report = report
    trajectory = json.loads(
        (builder.output / persisted["trajectory"]["path"]).read_text(encoding="utf-8")
    )
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("adversarial optimizer trajectory is empty or malformed")
    if optimization_report is None or optimization_report.get("history") != trajectory:
        raise ValueError("adversarial optimizer trajectory disagrees with optimization history")
    optimization_artifacts = optimization_report.get("artifacts")
    coefficient_reference = (
        optimization_artifacts.get("coefficients")
        if isinstance(optimization_artifacts, Mapping)
        else None
    )
    if not isinstance(coefficient_reference, str) or not coefficient_reference:
        raise ValueError("adversarial optimization report has no coefficient artifact")
    linked_coefficient_path = _resolve_external_path(
        coefficient_reference,
        adversarial_manifest,
    )
    linked_coefficient_digest = file_sha256(linked_coefficient_path)
    if linked_coefficient_digest != persisted["spline_coefficients"]["sha256"]:
        raise ValueError("adversarial spline coefficients disagree with optimization report")
    coefficient_path = builder.output / persisted["spline_coefficients"]["path"]
    with np.load(coefficient_path, allow_pickle=False) as archive:
        if set(archive.files) != {"coefficients", "channel_names", "control_points"}:
            raise ValueError("adversarial spline-coefficient evidence is malformed")
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
        control_points = int(archive["control_points"])
        channel_count = len(archive["channel_names"])
    if (
        coefficients.ndim != 1
        or not np.all(np.isfinite(coefficients))
        or control_points <= 0
        or coefficients.size != control_points * channel_count
        or config.get("refined_control_points") != control_points
        or optimization_report.get("refined_control_points") != control_points
    ):
        raise ValueError("adversarial spline-coefficient dimensions are inconsistent")
    return {
        "seed": seed,
        "selected_family": selected_family,
        "source_run": str(record["source_run"]),
        "artifacts": persisted,
        "evidence_linkage": {
            "trajectory_matches_optimization_history": True,
            "coefficient_upstream_reference": coefficient_reference,
            "coefficient_file_sha256": linked_coefficient_digest,
            "final_control_points": control_points,
        },
    }


def _optimizer_evidence_binding(optimizer: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in optimizer.items() if key != "evidence_bundle_id"}


def _optimization_motion_binding(
    builder: _PoolBuilder,
    optimizer: Mapping[str, Any],
    adversarial_manifest: Path,
    source: MotionClip,
    output: MotionClip,
) -> dict[str, Any]:
    """Bind the row's motions to the original/final paths declared by optimization.json."""
    artifacts = optimizer.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("adversarial optimizer metadata has no evidence artifacts")
    optimization_evidence = artifacts.get("optimization")
    if not isinstance(optimization_evidence, Mapping):
        raise ValueError("adversarial optimizer metadata has no optimization evidence")
    optimization_path = builder.output / str(optimization_evidence.get("path", ""))
    report = json.loads(optimization_path.read_text(encoding="utf-8"))
    report_artifacts = report.get("artifacts") if isinstance(report, dict) else None
    if not isinstance(report_artifacts, Mapping):
        raise ValueError("adversarial optimization report has no motion artifacts")
    result: dict[str, Any] = {}
    for role, expected in (("original", source), ("final", output)):
        raw = report_artifacts.get(role)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"adversarial optimization report has no {role} motion")
        path = _resolve_external_path(raw, adversarial_manifest)
        motion = load_motion_npz(path)
        if motion.content_hash != expected.content_hash:
            raise ValueError(f"adversarial {role} motion disagrees with the optimization report")
        result[role] = {
            "upstream_reference": raw,
            "motion_content_hash": motion.content_hash,
            "file_sha256": file_sha256(path),
        }
    return result


def _pool_local_path(root: Path, raw: Any, *, label: str) -> Path:
    path = (root.resolve() / str(raw or "")).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"candidate {label} escapes the pool") from exc
    if not path.is_file():
        raise ValueError(f"candidate references missing {label}")
    return path


def _validate_adversarial_optimizer_semantics(
    root: Path,
    item: Mapping[str, Any],
    optimizer: Mapping[str, Any],
) -> None:
    lineage = item["source_lineage"]
    origin = item["origin"]
    expected_binding = {
        "corpus_record_id": lineage.get("corpus_record_id"),
        "critic_normalization": lineage.get("critic_normalization"),
        "source_motion_content_hash": item.get("phase_reference_content_hash"),
        "output_motion_content_hash": item.get("motion_content_hash"),
    }
    if (
        any(optimizer.get(key) != value for key, value in expected_binding.items())
        or origin.get("upstream_identifier") != expected_binding["corpus_record_id"]
    ):
        raise ValueError("adversarial optimizer evidence is bound to a different candidate")
    if optimizer.get("evidence_bundle_id") != _canonical_id(
        "optimizer-evidence",
        _optimizer_evidence_binding(optimizer),
    ):
        raise ValueError("adversarial optimizer evidence bundle ID is invalid")
    artifacts = optimizer["artifacts"]

    def evidence_path(name: str) -> Path:
        return _pool_local_path(
            root,
            artifacts[name].get("path"),
            label=f"optimizer evidence {name}",
        )

    selected_family = optimizer.get("selected_family")
    run_config = json.loads(evidence_path("run_config").read_text(encoding="utf-8"))
    config = run_config.get("config") if isinstance(run_config, dict) else None
    if (
        not isinstance(config, dict)
        or config.get("seed") != optimizer.get("seed")
        or set(config.get("objective_weights", {})) != {selected_family}
    ):
        raise ValueError("adversarial run configuration is inconsistent")
    if artifacts["source_run"].get("upstream_reference") != optimizer.get("source_run"):
        raise ValueError("adversarial source-run reference is inconsistent")
    for name in ("source_run", "optimization"):
        report = json.loads(evidence_path(name).read_text(encoding="utf-8"))
        if (
            not isinstance(report, dict)
            or report.get("mode") != "red_team"
            or set(report.get("objective_weights", {})) != {selected_family}
        ):
            raise ValueError(f"adversarial {name} evidence is inconsistent")
    optimization_report = json.loads(evidence_path("optimization").read_text(encoding="utf-8"))
    report_artifacts = (
        optimization_report.get("artifacts") if isinstance(optimization_report, dict) else None
    )
    motion_binding = optimizer.get("optimization_motion_artifacts")
    if not isinstance(report_artifacts, dict) or not isinstance(motion_binding, dict):
        raise ValueError("adversarial optimizer motion binding is missing")
    expected_motion_hashes = {
        "original": item.get("phase_reference_content_hash"),
        "final": item.get("motion_content_hash"),
    }
    for role, expected_hash in expected_motion_hashes.items():
        binding = motion_binding.get(role)
        bound_file_digest = binding.get("file_sha256") if isinstance(binding, dict) else None
        if (
            not isinstance(binding, dict)
            or binding.get("upstream_reference") != report_artifacts.get(role)
            or binding.get("motion_content_hash") != expected_hash
            or not isinstance(bound_file_digest, str)
            or not bound_file_digest.startswith("sha256:")
            or len(bound_file_digest) != 71
            or any(
                character not in "0123456789abcdef"
                for character in bound_file_digest.removeprefix("sha256:")
            )
        ):
            raise ValueError("adversarial optimizer motion binding is inconsistent")
    trajectory = json.loads(evidence_path("trajectory").read_text(encoding="utf-8"))
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("adversarial optimizer trajectory is empty or malformed")
    if optimization_report.get("history") != trajectory:
        raise ValueError("adversarial optimizer trajectory is inconsistent")
    linkage = optimizer.get("evidence_linkage")
    if (
        not isinstance(linkage, dict)
        or linkage.get("trajectory_matches_optimization_history") is not True
        or linkage.get("coefficient_upstream_reference") != report_artifacts.get("coefficients")
        or linkage.get("coefficient_file_sha256") != artifacts["spline_coefficients"].get("sha256")
    ):
        raise ValueError("adversarial optimizer evidence linkage is inconsistent")
    with np.load(evidence_path("spline_coefficients"), allow_pickle=False) as archive:
        if set(archive.files) != {"coefficients", "channel_names", "control_points"}:
            raise ValueError("adversarial spline-coefficient evidence is malformed")
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
        control_points = int(archive["control_points"])
        channel_count = len(archive["channel_names"])
    if (
        coefficients.ndim != 1
        or not np.all(np.isfinite(coefficients))
        or control_points <= 0
        or coefficients.size != control_points * channel_count
        or linkage.get("final_control_points") != control_points
        or config.get("refined_control_points") != control_points
        or optimization_report.get("refined_control_points") != control_points
    ):
        raise ValueError("adversarial spline-coefficient dimensions are inconsistent")


def _fresh_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"candidate-pool output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _frozen_render_hash(protocol_freeze: Path) -> tuple[str, str]:
    freeze_path = Path(protocol_freeze).resolve()
    validation = validate_active_evaluation_protocol(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    contract = freeze.get("contract", {})
    mannequin = contract.get("mannequin_render", {}) if isinstance(contract, dict) else {}
    frozen_hash = (
        str(mannequin.get("render_protocol_hash", "")) if isinstance(mannequin, dict) else ""
    )
    current_hash = render_protocol_hash(MANNEQUIN_VIEWING_SETTINGS)
    if frozen_hash != current_hash:
        raise ValueError(
            "protocol freeze does not identify the current immutable mannequin render contract"
        )
    freeze_id = str(freeze.get("freeze_id") or freeze.get("protocol_id") or "")
    if not freeze_id or validation.get("freeze_id") != freeze_id:
        raise ValueError("protocol freeze has no validated stable freeze identifier")
    return freeze_id, frozen_hash


def _donor_for(
    sources: list[tuple[dict[str, Any], MotionClip]], position: int
) -> tuple[dict[str, Any], MotionClip]:
    record, _ = sources[position]
    alternatives = [
        item
        for offset, item in enumerate(sources)
        if offset != position
        and item[0].get("source_split") == record.get("source_split")
        and item[0].get("source_clip_id") != record.get("source_clip_id")
    ]
    if not alternatives:
        raise ValueError("phase-aligned style perturbation requires a same-split donor")
    return alternatives[position % len(alternatives)]


def _mechanism_group(mechanism: str) -> str:
    if mechanism in {"excessive_rigidification", "excessive_smoothing"}:
        return "oversmoothed_or_rigid"
    if mechanism in {"upper_body_phase_mismatch", "asymmetric_limb_timing"}:
        return "coordination_or_phase"
    if mechanism in {"torso_counter_rotation_mismatch", "weight_transfer_proxy"}:
        return "pelvis_or_torso_relationship"
    if mechanism == "phase_aligned_style_inconsistency":
        return "style_incoherent_phase_aligned"
    if mechanism == "preserved_critic_exploit":
        return "adversarial_optimizer_output"
    if mechanism == "equivalent_origin_shift":
        return "objective_equivalence_control"
    return "mechanically_screened_variant"


class _PoolBuilder:
    def __init__(
        self,
        output: Path,
        *,
        freeze_id: str,
        frozen_render_hash: str,
    ) -> None:
        self.output = output
        self.freeze_id = freeze_id
        self.frozen_render_hash = frozen_render_hash
        self.records: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self._identities: set[str] = set()

    def reject(self, *, origin_kind: str, identifier: str, reason: str) -> None:
        self.rejected.append(
            {"origin_kind": origin_kind, "upstream_identifier": identifier, "reason": reason}
        )

    def contains_visual_identity(
        self,
        motion: MotionClip,
        phase_reference: MotionClip,
        *,
        population: str,
        mechanism: str,
    ) -> bool:
        payload = {
            "population": population,
            "visual_motion": _visual_motion_hash(motion),
            "phase_reference_visual_motion": _visual_motion_hash(phase_reference),
            "mechanism": mechanism,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) in self._identities

    def add(
        self,
        motion: MotionClip,
        phase_reference: MotionClip,
        *,
        population: str,
        candidate_group: str,
        mechanism: str,
        source_lineage: dict[str, Any],
        parameter_metadata: dict[str, Any],
        deterministic_feasibility: dict[str, Any],
        origin_kind: str,
        upstream_identifier: str,
        deduplicate: bool = True,
    ) -> None:
        visual_hash = _visual_motion_hash(motion)
        reference_visual_hash = _visual_motion_hash(phase_reference)
        identity_payload = {
            "population": population,
            "visual_motion": visual_hash,
            "phase_reference_visual_motion": reference_visual_hash,
            "mechanism": mechanism,
        }
        visual_identity = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        if not deduplicate:
            identity_payload["origin_instance"] = upstream_identifier
        if deduplicate and visual_identity in self._identities:
            return
        self._identities.add(visual_identity)
        motion_relative = _persist_motion(self.output, motion)
        phase_relative = _persist_motion(self.output, phase_reference)
        try:
            payload = _two_cycle_payload(
                self.output / motion_relative,
                phase_reference_path=self.output / phase_relative,
                viewing_settings=MANNEQUIN_VIEWING_SETTINGS,
            )
        except ValueError as exc:
            self.reject(
                origin_kind=origin_kind,
                identifier=upstream_identifier,
                reason=f"not renderable under frozen protocol: {exc}",
            )
            return
        candidate_id = _canonical_id("candidate", identity_payload)
        acceptance = deterministic_feasibility.get("acceptance", {})
        is_feasible = bool(
            isinstance(acceptance, dict) and acceptance.get("all_required_satisfied", False)
        )
        if candidate_group == "adversarial_optimizer_output" and not source_lineage.get(
            "source_clip_id"
        ):
            raise ValueError("adversarial candidates require their real source clip lineage")
        source_id = str(
            source_lineage.get("source_clip_id")
            or source_lineage.get("corpus_record_id")
            or "unknown"
        )
        self.records.append(
            {
                "candidate_id": candidate_id,
                "stimulus_id": candidate_id,
                "stimulus_population": population,
                "subjective_label_status": population,
                "subjective_preference": None,
                "critic_training_eligible": False,
                "quality_evaluation_eligible": False,
                "future_human_label_collection_eligible": population == UNLABELED,
                "active_query_eligible": population == UNLABELED,
                "source_id": source_id,
                "style": source_id,
                "family": mechanism,
                "deterministic_feasible": is_feasible,
                "renderable": True,
                "adversarial_optimizer_output": (candidate_group == "adversarial_optimizer_output"),
                "candidate_group": candidate_group,
                "mechanism": mechanism,
                "source_lineage": source_lineage,
                "parameter_metadata": parameter_metadata,
                "deterministic_feasibility": deterministic_feasibility,
                "origin": {
                    "kind": origin_kind,
                    "upstream_identifier": upstream_identifier,
                },
                "motion": motion_relative,
                "phase_reference_motion": phase_relative,
                "motion_content_hash": motion.content_hash,
                "visual_motion_hash": visual_hash,
                "motion_file_sha256": file_sha256(self.output / motion_relative),
                "phase_reference_content_hash": phase_reference.content_hash,
                "phase_reference_visual_motion_hash": reference_visual_hash,
                "renderability": {
                    "eligible": True,
                    "protocol_freeze_id": self.freeze_id,
                    "render_protocol_hash": payload["render_protocol_hash"],
                    "render_hash": payload["render_hash"],
                    "cycle_window": payload["cycle_window"],
                },
            }
        )


def _clean_sources(
    dataset: FixedRigSampleDataset,
) -> list[tuple[dict[str, Any], MotionClip]]:
    sources = []
    for index, record in enumerate(dataset.records):
        if record.get("sample_role") != "clean":
            continue
        clean, _ = motions_from_sample(dataset, index)
        clean, _ = declare_reference_speed_task(clean, clean)
        sources.append((record, clean))
    return sources


def _anchor_dataset_source_paths(
    dataset: FixedRigSampleDataset,
    dataset_directory: Path,
) -> None:
    """Resolve repo-relative source clips without depending on the caller's CWD."""
    manifest_path = Path(dataset_directory).resolve() / "manifest.jsonl"
    for record in dataset.records:
        raw = record.get("source_path")
        if not isinstance(raw, str) or not raw:
            raise ValueError("candidate source-dataset record has no source_path")
        record["source_path"] = str(_resolve_external_path(raw, manifest_path))


def _add_clean_windows(
    builder: _PoolBuilder,
    sources: list[tuple[dict[str, Any], MotionClip]],
    limits: ProductionThresholds,
) -> None:
    for record, clean in sources:
        acceptance = assess_production_acceptance(clean, clean, thresholds=limits)
        builder.add(
            clean,
            clean,
            population=UNLABELED,
            candidate_group="clean_100style_window",
            mechanism="clean_reference",
            source_lineage={
                "source_clip_id": record.get("source_clip_id"),
                "source_sample_id": record.get("sample_id"),
                "source_split": record.get("source_split"),
                "source_window_start": record.get("source_window_start"),
            },
            parameter_metadata={"severity": "clean", "parameters": {}},
            deterministic_feasibility={
                "assessment_source": "recomputed_production_acceptance",
                "acceptance": _acceptance_dict(acceptance),
            },
            origin_kind="phase8_clean_100style_window",
            upstream_identifier=str(record.get("sample_id")),
        )


def _add_subtle_corruptions(
    builder: _PoolBuilder,
    dataset: FixedRigSampleDataset,
    limits: ProductionThresholds,
) -> None:
    for index, record in enumerate(dataset.records):
        if (
            record.get("sample_role") != "hard_corruption"
            or record.get("severity_label") != "subtle"
        ):
            continue
        clean, candidate = motions_from_sample(dataset, index)
        clean, candidate = declare_reference_speed_task(clean, candidate)
        acceptance = assess_production_acceptance(clean, candidate, thresholds=limits)
        if not acceptance.all_required_satisfied:
            builder.reject(
                origin_kind="phase8_subtle_corruption",
                identifier=str(record.get("sample_id")),
                reason="failed deterministic production acceptance",
            )
            continue
        builder.add(
            candidate,
            clean,
            population=UNLABELED,
            candidate_group="hard_feasible_subtle",
            mechanism=str(record.get("catalog_mechanism_key") or "unknown"),
            source_lineage={
                "source_clip_id": record.get("source_clip_id"),
                "source_sample_id": record.get("sample_id"),
                "source_split": record.get("source_split"),
                "source_window_start": record.get("source_window_start"),
            },
            parameter_metadata={
                "severity": "subtle",
                "measured_severity": record.get("measured_severity"),
                "severity_bin": record.get("severity_bin"),
            },
            deterministic_feasibility={
                "assessment_source": "recomputed_production_acceptance",
                "acceptance": _acceptance_dict(acceptance),
            },
            origin_kind="phase8_subtle_corruption",
            upstream_identifier=str(record.get("sample_id")),
        )


def _add_existing_perceptual_variants(
    builder: _PoolBuilder,
    pair_directory: Path,
) -> None:
    for pair in load_perceptual_pairs(pair_directory):
        motion = load_motion_npz(pair_directory / str(pair["motion_b"]))
        phase_reference = load_motion_npz(pair_directory / str(pair["motion_a"]))
        mechanism = str(pair["perturbation_mechanism"])
        if mechanism == "preserved_critic_exploit":
            if not builder.contains_visual_identity(
                motion,
                phase_reference,
                population=UNLABELED,
                mechanism=mechanism,
            ):
                raise ValueError(
                    "Phase 12 critic exploit is absent from the authoritative optimizer corpus"
                )
            continue
        builder.add(
            motion,
            phase_reference,
            population=UNLABELED,
            candidate_group=_mechanism_group(mechanism),
            mechanism=mechanism,
            source_lineage={
                "source_clip_id": pair.get("source_clip_id"),
                "source_sample_id": pair.get("source_sample_id"),
                "source_split": pair.get("source_split"),
                "source_window_start": pair.get("source_window_start"),
            },
            parameter_metadata={
                "severity": "upstream_parameterized",
                "strength": pair.get("perturbation_strength"),
                "mechanism_partition": pair.get("mechanism_partition"),
            },
            deterministic_feasibility={
                "assessment_source": "phase12_persisted_acceptance",
                "acceptance": pair["motion_b_acceptance"],
            },
            origin_kind="phase12_hard_feasible_candidate",
            upstream_identifier=str(pair["pair_id"]),
        )


def _add_new_perceptual_variants(
    builder: _PoolBuilder,
    sources: list[tuple[dict[str, Any], MotionClip]],
    limits: ProductionThresholds,
) -> None:
    for position, (record, clean) in enumerate(sources):
        donor_record, donor = _donor_for(sources, position)
        for spec in ADDITIONAL_PERTURBATIONS:
            raw = apply_perceptual_perturbation(clean, spec, donor=donor)
            clean_task, candidate = declare_reference_speed_task(clean, raw)
            acceptance = assess_production_acceptance(clean_task, candidate, thresholds=limits)
            upstream_id = f"{record.get('sample_id')}:{spec.mechanism}:{spec.strength}"
            if not acceptance.all_required_satisfied:
                builder.reject(
                    origin_kind="phase16_new_constraint_screened_variant",
                    identifier=upstream_id,
                    reason="failed deterministic production acceptance",
                )
                continue
            lineage = {
                "source_clip_id": record.get("source_clip_id"),
                "source_sample_id": record.get("sample_id"),
                "source_split": record.get("source_split"),
                "source_window_start": record.get("source_window_start"),
            }
            if spec.mechanism == "phase_aligned_style_inconsistency":
                lineage["phase_aligned_donor"] = {
                    "source_clip_id": donor_record.get("source_clip_id"),
                    "source_sample_id": donor_record.get("sample_id"),
                    "source_split": donor_record.get("source_split"),
                }
            builder.add(
                candidate,
                clean_task,
                population=UNLABELED,
                candidate_group=_mechanism_group(spec.mechanism),
                mechanism=spec.mechanism,
                source_lineage=lineage,
                parameter_metadata={
                    "severity": "mid_band_unlabeled",
                    "strength": spec.strength,
                    "mechanism_partition": spec.mechanism_partition,
                },
                deterministic_feasibility={
                    "assessment_source": "recomputed_production_acceptance",
                    "acceptance": _acceptance_dict(acceptance),
                },
                origin_kind="phase16_new_constraint_screened_variant",
                upstream_identifier=upstream_id,
            )


def _add_adversarial_outputs(
    builder: _PoolBuilder,
    adversarial_manifest: Path,
    limits: ProductionThresholds,
) -> None:
    manifest = json.loads(Path(adversarial_manifest).read_text(encoding="utf-8"))
    for record in manifest.get("records", []):
        pair = record.get("adversarial_quality_pair", {})
        record_id = str(record.get("corpus_record_id") or record.get("id") or "unknown")
        source_path = _resolve_external_path(str(pair.get("preferred", "")), adversarial_manifest)
        output_path = _resolve_external_path(str(pair.get("rejected", "")), adversarial_manifest)
        source = load_motion_npz(source_path)
        output = load_motion_npz(output_path)
        source_metadata = dict(source.metadata)
        raw_provenance = source_metadata.get("provenance", {})
        provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
        source_clip_id = source_metadata.get("source_clip_id") or provenance.get("source_clip_id")
        if not source_clip_id:
            raise ValueError(f"adversarial source has no source_clip_id lineage: {record_id}")
        corpus_lineage = record.get("lineage", {})
        discovery_label = (
            str(corpus_lineage.get("discovery_label", ""))
            if isinstance(corpus_lineage, Mapping)
            else ""
        )
        if "layer_norm" in discovery_label:
            critic_normalization = "layer_norm"
        elif "group_norm" in discovery_label:
            critic_normalization = "group_norm"
        else:
            raise ValueError(
                f"adversarial corpus record has no declared critic normalization: {record_id}"
            )
        optimizer_metadata = _adversarial_optimizer_metadata(
            builder,
            record,
            adversarial_manifest,
        )
        optimizer_metadata["optimization_motion_artifacts"] = _optimization_motion_binding(
            builder,
            optimizer_metadata,
            adversarial_manifest,
            source,
            output,
        )
        optimizer_metadata.update(
            {
                "corpus_record_id": record_id,
                "critic_normalization": critic_normalization,
                "source_motion_content_hash": source.content_hash,
                "output_motion_content_hash": output.content_hash,
            }
        )
        optimizer_metadata["evidence_bundle_id"] = _canonical_id(
            "optimizer-evidence",
            _optimizer_evidence_binding(optimizer_metadata),
        )
        source_task, output_task = declare_reference_speed_task(source, output)
        acceptance = assess_production_acceptance(source_task, output_task, thresholds=limits)
        builder.add(
            output,
            source,
            population=UNLABELED,
            candidate_group="adversarial_optimizer_output",
            mechanism="preserved_critic_exploit",
            source_lineage={
                "source_clip_id": str(source_clip_id),
                "source_sample_id": source_metadata.get("sample_id"),
                "source_split": source_metadata.get("source_split") or provenance.get("split"),
                "source_window_start": source_metadata.get("source_window_start"),
                "source_take_id": provenance.get("source_take_id"),
                "source_dataset": provenance.get("source_dataset"),
                "corpus_record_id": record_id,
                "optimizer_source_motion": source.content_hash,
                "critic_normalization": critic_normalization,
                "optimizer_discovery_label": discovery_label,
            },
            parameter_metadata={
                "severity": "optimizer_discovered",
                "optimizer_run_parameters_available": True,
                "optimizer": optimizer_metadata,
            },
            deterministic_feasibility={
                "assessment_source": "recomputed_production_acceptance",
                "acceptance": _acceptance_dict(acceptance),
            },
            origin_kind="phase9_or_phase10_adversarial_cma_output",
            upstream_identifier=record_id,
            deduplicate=False,
        )


def _add_calibration_ladders(
    builder: _PoolBuilder,
    calibration_manifest: Path,
    limits: ProductionThresholds,
) -> None:
    manifest = json.loads(Path(calibration_manifest).read_text(encoding="utf-8"))
    by_id = {str(item["stimulus_id"]): item for item in manifest.get("stimuli", [])}
    ladder_lookup = {
        stimulus_id: str(ladder.get("ladder_id"))
        for ladder in manifest.get("ladders", [])
        for stimulus_id in ladder.get("stimulus_ids", [])
    }
    for stimulus_id, stimulus in by_id.items():
        motion_path = calibration_manifest.parent / str(stimulus["motion"])
        phase_path = calibration_manifest.parent / str(stimulus["phase_reference_motion"])
        motion = load_motion_npz(motion_path)
        phase_reference = load_motion_npz(phase_path)
        reference_task, candidate_task = declare_reference_speed_task(phase_reference, motion)
        acceptance = assess_production_acceptance(reference_task, candidate_task, thresholds=limits)
        origin = stimulus.get("origin", {})
        builder.add(
            motion,
            phase_reference,
            population=CALIBRATION_ONLY,
            candidate_group="calibration_severity_ladder",
            mechanism=str(stimulus.get("calibration_family", "unknown")),
            source_lineage={
                "source_clip_id": stimulus.get("source_id"),
                "source_sample_id": origin.get("sample_id"),
                "source_split": stimulus.get("source_split"),
                "source_window_start": stimulus.get("source_window_start"),
            },
            parameter_metadata={
                "severity": stimulus.get("calibration_severity"),
                "preset_parameter": origin.get("preset_parameter"),
                "preset_parameter_semantics": origin.get("preset_parameter_semantics"),
                "ladder_id": ladder_lookup.get(stimulus_id),
            },
            deterministic_feasibility={
                "assessment_source": "recomputed_for_metadata_only",
                "acceptance": _acceptance_dict(acceptance),
                "production_eligibility_required": False,
            },
            origin_kind="phase15_calibration_only_ladder",
            upstream_identifier=stimulus_id,
            deduplicate=False,
        )


def validate_unlabeled_candidate_pool(directory: Path) -> list[dict[str, Any]]:
    """Validate purpose separation, files, feasibility declarations, and render contracts."""
    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rejected = json.loads((root / "rejected.json").read_text(encoding="utf-8"))
    if not isinstance(rejected, list) or not all(isinstance(row, dict) for row in rejected):
        raise ValueError("candidate rejection ledger must be a JSON list of objects")
    records = [
        json.loads(line)
        for line in (root / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if summary.get("format_version") != CANDIDATE_POOL_VERSION or not records:
        raise ValueError("unsupported or empty perceptual candidate pool")
    ids = [str(item.get("candidate_id")) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    frozen_hash = str(summary.get("frozen_render_protocol_hash", ""))
    freeze_id = str(summary.get("protocol_freeze_id", ""))
    protocol_freeze = Path(str(summary.get("protocol_freeze") or ""))
    if not protocol_freeze.is_absolute():
        protocol_freeze = root / protocol_freeze
    validated_freeze_id, validated_render_hash = _frozen_render_hash(protocol_freeze)
    if freeze_id != validated_freeze_id or frozen_hash != validated_render_hash:
        raise ValueError("candidate summary disagrees with the validated protocol freeze")
    for item in records:
        if item.get("stimulus_id") != item.get("candidate_id"):
            raise ValueError("candidate stimulus and candidate IDs must match")
        origin = item.get("origin")
        if not isinstance(origin, dict):
            raise ValueError("candidate lacks origin metadata")
        identity_payload = {
            "population": item.get("stimulus_population"),
            "visual_motion": item.get("visual_motion_hash"),
            "phase_reference_visual_motion": item.get("phase_reference_visual_motion_hash"),
            "mechanism": item.get("mechanism"),
        }
        if item.get("candidate_group") in {
            "adversarial_optimizer_output",
            "calibration_severity_ladder",
        }:
            identity_payload["origin_instance"] = origin.get("upstream_identifier")
        if item.get("candidate_id") != _canonical_id("candidate", identity_payload):
            raise ValueError("candidate ID does not bind its motion and origin identity")
        population = item.get("stimulus_population")
        if population not in {UNLABELED, CALIBRATION_ONLY}:
            raise ValueError("candidate has an invalid label population")
        if item.get("subjective_label_status") != population:
            raise ValueError("candidate label status disagrees with its population")
        if item.get("subjective_preference") is not None:
            raise ValueError("unlabeled candidate pool contains a subjective preference")
        if item.get("critic_training_eligible") is not False:
            raise ValueError("an unlabeled candidate cannot be critic-training eligible")
        if item.get("quality_evaluation_eligible") is not False:
            raise ValueError("an unlabeled candidate cannot be quality-evaluation eligible")
        should_collect = population == UNLABELED
        if (
            item.get("future_human_label_collection_eligible") is not should_collect
            or item.get("active_query_eligible") is not should_collect
        ):
            raise ValueError("candidate eligibility disagrees with its label population")
        if not isinstance(item.get("source_lineage"), dict):
            raise ValueError("candidate lacks source lineage")
        is_adversarial = item.get("candidate_group") == "adversarial_optimizer_output"
        if item.get("adversarial_optimizer_output") is not is_adversarial:
            raise ValueError("candidate adversarial flag disagrees with its group")
        if item.get("adversarial_optimizer_output"):
            source_clip_id = item["source_lineage"].get("source_clip_id")
            if not source_clip_id or item.get("source_id") != source_clip_id:
                raise ValueError("adversarial candidate lacks real source-clip lineage")
            if item["source_lineage"].get("critic_normalization") not in {
                "group_norm",
                "layer_norm",
            }:
                raise ValueError("adversarial candidate lacks critic-normalization lineage")
            parameter_metadata = item.get("parameter_metadata")
            optimizer = (
                parameter_metadata.get("optimizer")
                if isinstance(parameter_metadata, dict)
                else None
            )
            if (
                not isinstance(optimizer, dict)
                or not isinstance(optimizer.get("seed"), int)
                or isinstance(optimizer.get("seed"), bool)
                or not isinstance(optimizer.get("selected_family"), str)
                or not optimizer.get("selected_family")
                or not isinstance(optimizer.get("source_run"), str)
                or not optimizer.get("source_run")
            ):
                raise ValueError("adversarial candidate lacks optimizer run metadata")
            optimizer_artifacts = optimizer.get("artifacts")
            expected_evidence = {
                "source_run",
                "trajectory",
                "run_config",
                "optimization",
                "spline_coefficients",
            }
            if (
                not isinstance(optimizer_artifacts, dict)
                or set(optimizer_artifacts) != expected_evidence
            ):
                raise ValueError("adversarial candidate has incomplete optimizer evidence")
            for evidence in optimizer_artifacts.values():
                if not isinstance(evidence, dict):
                    raise ValueError("adversarial optimizer evidence is malformed")
                evidence_path = (root / str(evidence.get("path", ""))).resolve()
                try:
                    evidence_path.relative_to(root.resolve())
                except ValueError as exc:
                    raise ValueError("adversarial optimizer evidence escapes the pool") from exc
                if (
                    not evidence_path.is_file()
                    or evidence_path.stat().st_size != evidence.get("byte_count")
                    or file_sha256(evidence_path) != evidence.get("sha256")
                ):
                    raise ValueError("adversarial optimizer evidence changed")
            _validate_adversarial_optimizer_semantics(root, item, optimizer)
        feasibility = item.get("deterministic_feasibility")
        if not isinstance(feasibility, dict) or not isinstance(feasibility.get("acceptance"), dict):
            raise ValueError("candidate lacks deterministic-feasibility evidence")
        if item.get("deterministic_feasible") is not bool(
            feasibility["acceptance"].get("all_required_satisfied", False)
        ):
            raise ValueError("candidate feasibility flag disagrees with its acceptance evidence")
        if item.get("renderable") is not True:
            raise ValueError("candidate renderable flag is invalid")
        renderability = item.get("renderability", {})
        if (
            not renderability.get("eligible")
            or renderability.get("render_protocol_hash") != frozen_hash
        ):
            raise ValueError("candidate is not renderable under the frozen protocol")
        if renderability.get("protocol_freeze_id") != freeze_id:
            raise ValueError("candidate references the wrong protocol freeze")
        motion_path = _pool_local_path(root, item.get("motion"), label="motion")
        phase_reference_path = _pool_local_path(
            root,
            item.get("phase_reference_motion"),
            label="phase_reference_motion",
        )
        if file_sha256(motion_path) != item.get("motion_file_sha256"):
            raise ValueError("candidate motion file digest changed")
        motion = load_motion_npz(motion_path)
        phase_reference = load_motion_npz(phase_reference_path)
        if (
            motion.content_hash != item.get("motion_content_hash")
            or _visual_motion_hash(motion) != item.get("visual_motion_hash")
            or phase_reference.content_hash != item.get("phase_reference_content_hash")
            or _visual_motion_hash(phase_reference)
            != item.get("phase_reference_visual_motion_hash")
        ):
            raise ValueError("candidate motion or phase-reference content changed")
        payload = _two_cycle_payload(
            motion_path,
            phase_reference_path=phase_reference_path,
            viewing_settings=MANNEQUIN_VIEWING_SETTINGS,
        )
        if any(
            payload[key] != renderability.get(key)
            for key in ("render_protocol_hash", "render_hash", "cycle_window")
        ):
            raise ValueError("candidate frozen render payload changed")
    if len(records) != int(summary.get("candidate_count", -1)):
        raise ValueError("candidate summary count does not match manifest")
    populations = Counter(str(item["stimulus_population"]) for item in records)
    groups = Counter(str(item["candidate_group"]) for item in records)
    mechanisms = Counter(str(item["mechanism"]) for item in records)
    origins = Counter(str(item["origin"]["kind"]) for item in records)
    sources = {
        str(item["source_lineage"].get("source_clip_id"))
        for item in records
        if item["source_lineage"].get("source_clip_id")
    }
    expected_summary = {
        "population_counts": dict(sorted(populations.items())),
        "candidate_group_counts": dict(sorted(groups.items())),
        "mechanism_counts": dict(sorted(mechanisms.items())),
        "origin_counts": dict(sorted(origins.items())),
        "unique_source_count": len(sources),
        "renderable_candidate_count": len(records),
        "deterministically_feasible_candidate_count": sum(
            bool(item["deterministic_feasible"]) for item in records
        ),
        "adversarial_optimizer_evidence_record_count": sum(
            bool(item["adversarial_optimizer_output"]) for item in records
        ),
        "mechanically_valid_uncanny_stress_candidate_count": sum(
            bool(item["deterministic_feasible"])
            and item["candidate_group"]
            in {
                "coordination_or_phase",
                "pelvis_or_torso_relationship",
                "style_incoherent_phase_aligned",
            }
            for item in records
        ),
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"candidate summary {field} does not match manifest")
    rejected_origins = Counter(str(row.get("origin_kind")) for row in rejected)
    if summary.get("rejected_count") != len(rejected) or summary.get(
        "rejected_origin_counts"
    ) != dict(sorted(rejected_origins.items())):
        raise ValueError("candidate rejection summary does not match its ledger")
    if summary.get("phase16_perceptual_cma_invoked") is not False:
        raise ValueError("candidate summary must attest that Phase 16 did not invoke CMA")
    label_policy = summary.get("subjective_label_policy")
    if (
        not isinstance(label_policy, dict)
        or label_policy.get("upstream_pair_preferences_imported") is not False
    ):
        raise ValueError("candidate summary does not preserve unlabeled semantics")
    return records


def generate_unlabeled_candidate_pool(
    source_dataset_directory: Path,
    pair_dataset_directory: Path,
    adversarial_manifest: Path,
    calibration_manifest: Path,
    protocol_freeze: Path,
    output_directory: Path,
    *,
    thresholds: ProductionThresholds | None = None,
) -> dict[str, Any]:
    """Build a separate, render-validated pool without importing subjective directions."""
    limits = ProductionThresholds() if thresholds is None else thresholds
    output = Path(output_directory)
    _fresh_output(output)
    freeze_id, frozen_hash = _frozen_render_hash(protocol_freeze)
    builder = _PoolBuilder(output, freeze_id=freeze_id, frozen_render_hash=frozen_hash)
    dataset = FixedRigSampleDataset(source_dataset_directory, regime="all")
    _anchor_dataset_source_paths(dataset, Path(source_dataset_directory))
    sources = _clean_sources(dataset)

    _add_clean_windows(builder, sources, limits)
    _add_subtle_corruptions(builder, dataset, limits)
    # Preserve the optimizer corpus identifiers for visually duplicate Phase 12 candidates by
    # inserting the complete Phase 9/10 corpus first; subsequent pair-dataset duplicates collapse.
    _add_adversarial_outputs(builder, Path(adversarial_manifest), limits)
    _add_existing_perceptual_variants(builder, Path(pair_dataset_directory))
    _add_new_perceptual_variants(builder, sources, limits)
    _add_calibration_ladders(builder, Path(calibration_manifest), limits)

    builder.records.sort(key=lambda item: str(item["candidate_id"]))
    (output / "candidates.jsonl").write_text(
        "".join(
            json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in builder.records
        ),
        encoding="utf-8",
    )
    _json_write(output / "rejected.json", builder.rejected)
    populations = Counter(str(item["stimulus_population"]) for item in builder.records)
    groups = Counter(str(item["candidate_group"]) for item in builder.records)
    mechanisms = Counter(str(item["mechanism"]) for item in builder.records)
    origins = Counter(str(item["origin"]["kind"]) for item in builder.records)
    rejected_origins = Counter(str(item["origin_kind"]) for item in builder.rejected)
    sources_seen = {
        str(item["source_lineage"].get("source_clip_id"))
        for item in builder.records
        if item["source_lineage"].get("source_clip_id")
    }
    report = {
        "format_version": CANDIDATE_POOL_VERSION,
        "protocol_freeze": str(Path(protocol_freeze).resolve()),
        "protocol_freeze_id": freeze_id,
        "frozen_render_protocol_hash": frozen_hash,
        "candidate_count": len(builder.records),
        "rejected_count": len(builder.rejected),
        "population_counts": dict(sorted(populations.items())),
        "candidate_group_counts": dict(sorted(groups.items())),
        "mechanism_counts": dict(sorted(mechanisms.items())),
        "origin_counts": dict(sorted(origins.items())),
        "rejected_origin_counts": dict(sorted(rejected_origins.items())),
        "unique_source_count": len(sources_seen),
        "input_clean_100style_window_count": len(sources),
        "renderable_candidate_count": len(builder.records),
        "deterministically_feasible_candidate_count": sum(
            bool(item["deterministic_feasible"]) for item in builder.records
        ),
        "adversarial_optimizer_evidence_record_count": sum(
            bool(item["adversarial_optimizer_output"]) for item in builder.records
        ),
        "phase16_perceptual_cma_invoked": False,
        "mechanically_valid_uncanny_stress_candidate_count": sum(
            bool(item["deterministic_feasible"])
            and item["candidate_group"]
            in {
                "coordination_or_phase",
                "pelvis_or_torso_relationship",
                "style_incoherent_phase_aligned",
            }
            for item in builder.records
        ),
        "production_thresholds": limits.model_dump(mode="json"),
        "inputs": {
            "source_dataset": str(Path(source_dataset_directory).resolve()),
            "pair_dataset": str(Path(pair_dataset_directory).resolve()),
            "adversarial_manifest": str(Path(adversarial_manifest).resolve()),
            "calibration_manifest": str(Path(calibration_manifest).resolve()),
        },
        "subjective_label_policy": {
            "UNLABELED": "no subjective quality direction is asserted",
            "CALIBRATION_ONLY": "ordered generated severity is not quality ground truth",
            "upstream_pair_preferences_imported": False,
        },
        "additional_perturbations": [asdict(spec) for spec in ADDITIONAL_PERTURBATIONS],
        "manifest": str((output / "candidates.jsonl").resolve()),
    }
    _json_write(output / "summary.json", report)
    validate_unlabeled_candidate_pool(output)
    return report
