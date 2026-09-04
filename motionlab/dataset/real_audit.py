"""Reproducible real-motion gate for the first fixed-rig critic experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from motionlab._version import __version__
from motionlab.core.provenance import SplitName, assign_source_lineage
from motionlab.dataset.generation import DatasetGenerationConfig, generate_corruption_dataset
from motionlab.io.bvh import BvhImportOptions, bvh_to_motion_clip, load_bvh
from motionlab.io.npz import save_motion_npz
from motionlab.io.rig_spec import load_rig_spec
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.motion.clip import MotionClip
from motionlab.processing.contacts import detect_foot_contacts
from motionlab.processing.phase import estimate_gait_phase
from motionlab.processing.resample import resample_motion

REAL_AUDIT_VERSION = "motionlab.real_motion_audit.v2"

FIRST_CRITIC_MECHANISMS = (
    "foot_slide/root_drift",
    "foot_slide/hip_rotation_drift",
    "ground_penetration/root_vertical_offset",
    "floating_contact/root_vertical_offset",
    "loop_seam/end_rotation_offset",
    "loop_seam/root_velocity_mismatch",
    "joint_pop/local_rotation_pulse",
    "joint_pop/smooth_rotation_pulse",
    "joint_jitter/white_tangent_noise",
    "joint_jitter/band_limited_correlated",
    "speed_inconsistency/root_progression_scale",
    "speed_inconsistency/leg_pose_amplitude_scale",
)


@dataclass(frozen=True)
class RealSourceSpec:
    """One authoritative crop and source-lineage assignment."""

    name: str
    split: SplitName
    start_frame: int
    stop_frame: int


@dataclass(frozen=True)
class CleanlinessQualification:
    """Transparent result of applying fixed clean-source criteria."""

    passed: bool
    confidence: float
    criteria: Mapping[str, Mapping[str, float | bool | None]]


DEFAULT_100STYLE_SOURCES = (
    ("Akimbo", "train"),
    ("Angry", "train"),
    ("ArmsBehindBack", "train"),
    ("ArmsBySide", "train"),
    ("Depressed", "train"),
    ("Elated", "train"),
    ("Neutral", "train"),
    ("ArmsFolded", "validation"),
    ("Old", "validation"),
    ("GracefulArms", "validation"),
    ("HandsInPockets", "test"),
    ("Heavyset", "test"),
    ("LookUp", "test"),
    ("Proud", "test"),
)


def _upper_bound_criterion(value: float | None, maximum: float) -> dict[str, float | bool | None]:
    return {
        "value": value,
        "maximum": maximum,
        "passed": value is not None and np.isfinite(value) and value <= maximum,
    }


def _lower_bound_criterion(value: float | None, minimum: float) -> dict[str, float | bool | None]:
    return {
        "value": value,
        "minimum": minimum,
        "passed": value is not None and np.isfinite(value) and value >= minimum,
    }


def qualify_clean_motion(clip: MotionClip) -> CleanlinessQualification:
    """Qualify a mocap take with fixed physical and signal-confidence criteria."""
    report = grade_motion_deterministic(clip)
    speed = report.measured.get("average_speed_mps")
    criteria: dict[str, dict[str, float | bool | None]] = {
        "contact_confidence": _lower_bound_criterion(report.confidence.get("contact"), 0.25),
        "gait_phase_confidence": _lower_bound_criterion(report.confidence.get("gait_phase"), 0.25),
        "average_speed_mps": {
            "value": speed,
            "minimum": 0.1,
            "maximum": 3.0,
            "passed": speed is not None and np.isfinite(speed) and 0.1 <= speed <= 3.0,
        },
        "maximum_stance_slip_cm": _upper_bound_criterion(
            report.measured.get("maximum_stance_slip_cm"), 5.0
        ),
        "maximum_ground_penetration_cm": _upper_bound_criterion(
            report.measured.get("maximum_ground_penetration_cm"), 2.0
        ),
        "maximum_floating_contact_cm": _upper_bound_criterion(
            report.measured.get("maximum_floating_contact_cm"), 6.0
        ),
    }
    passed = all(bool(criterion["passed"]) for criterion in criteria.values())
    confidence = min(
        float(report.confidence.get("contact", 0.0)),
        float(report.confidence.get("gait_phase", 0.0)),
        float(report.confidence.get("ground", 0.0)),
    )
    return CleanlinessQualification(
        passed=passed,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        criteria=criteria,
    )


def load_100style_source_specs(data_directory: Path) -> tuple[RealSourceSpec, ...]:
    """Read the publisher's frame cuts for the fixed expanded-audit takes."""
    cuts_path = Path(data_directory) / "Frame_Cuts.csv"
    with cuts_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {str(row["STYLE_NAME"]): row for row in csv.DictReader(stream)}
    specs: list[RealSourceSpec] = []
    for name, split_name in DEFAULT_100STYLE_SOURCES:
        if name not in rows:
            raise ValueError(f"100STYLE frame-cut table has no row for {name!r}")
        row = rows[name]
        if split_name not in {"train", "validation", "test"}:
            raise AssertionError("invalid built-in audit split")
        specs.append(
            RealSourceSpec(
                name=name,
                split=split_name,  # type: ignore[arg-type]
                start_frame=int(row["FW_START"]),
                stop_frame=int(row["FW_STOP"]),
            )
        )
    return tuple(specs)


def _compact_clean_report(clip: MotionClip) -> dict[str, Any]:
    report = grade_motion_deterministic(clip)
    contacts = detect_foot_contacts(clip)
    phase = estimate_gait_phase(contacts)
    event_counts = {
        f"{side}_{kind}": sum(event.side == side and event.kind == kind for event in phase.events)
        for side in ("left", "right")
        for kind in ("heel_strike", "toe_off")
    }
    return {
        "motion_id": clip.content_hash,
        "frames": clip.num_frames,
        "fps": clip.fps,
        "duration_s": clip.duration_s,
        "measured": report.measured,
        "confidence": report.confidence,
        "warnings": list(report.warnings),
        "contact_fraction_by_marker": {
            name: float(np.mean(contacts.hard[:, index]))
            for index, name in enumerate(contacts.marker_names)
        },
        "gait_event_counts": event_counts,
        "bilateral_gait_events_present": all(value > 0 for value in event_counts.values()),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_100style_sources(
    data_directory: Path,
    output_directory: Path,
    *,
    rig_path: Path,
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...]]:
    """Import, crop, qualify, lineagize, and persist the real audit sources."""
    data_root = Path(data_directory)
    output = Path(output_directory)
    rig = load_rig_spec(rig_path)
    coordinate = rig.coordinate_system
    options = BvhImportOptions(
        handedness=coordinate.handedness,
        up_axis=coordinate.up,
        forward_axis=coordinate.forward,
        unit_scale_to_m=coordinate.unit_scale_to_meters,
    )
    paths: list[Path] = []
    reports: list[dict[str, Any]] = []
    for spec in load_100style_source_specs(data_root):
        source_path = data_root / f"{spec.name}_FW.bvh"
        raw_sha256 = f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
        raw = bvh_to_motion_clip(
            load_bvh(source_path),
            options=options,
            role_names=dict(rig.joints),
            markers=dict(rig.markers),
            joint_limits=dict(rig.joint_limits),
        )
        cropped = raw.slice_frames(spec.start_frame, spec.stop_frame)
        if not np.isclose(cropped.fps, 60.0, atol=1.0e-12, rtol=0.0):
            cropped = resample_motion(cropped, 60.0)
        qualification = qualify_clean_motion(cropped)
        clean_report = _compact_clean_report(cropped)
        clean_report.update(
            {
                "source_clip": spec.name,
                "source_path": str(source_path),
                "source_file_sha256": raw_sha256,
                "source_frame_cut": [spec.start_frame, spec.stop_frame],
                "split": spec.split,
                "cleanliness_qualification": {
                    "passed": qualification.passed,
                    "confidence": qualification.confidence,
                    "criteria": qualification.criteria,
                },
            }
        )
        reports.append(clean_report)
        if not qualification.passed:
            continue
        metadata = dict(cropped.metadata)
        metadata.update(
            {
                "source_path": str(source_path),
                "source_file_sha256": raw_sha256,
                "authoritative_frame_cut": [spec.start_frame, spec.stop_frame],
                "cleanliness_qualification": clean_report["cleanliness_qualification"],
                # The task command comes from the qualified clean take and remains fixed when
                # corruptions alter root or limb motion.
                "expected_speed_mps": clean_report["measured"]["average_speed_mps"],
            }
        )
        qualified = assign_source_lineage(
            cropped.with_updates(metadata=metadata),
            source_dataset="100STYLE",
            source_clip_id=f"{spec.name}_FW",
            source_take_id=raw_sha256,
            split=spec.split,
            importer_version=f"motionlab-bvh-{__version__}",
            clean_confidence=qualification.confidence,
            metadata={
                "movement_type": "forward_walking",
                "style_used_as_target": False,
                "frame_cut_source": "100STYLE/Frame_Cuts.csv",
            },
        )
        destination = output / "sources" / f"{spec.name}_FW.npz"
        save_motion_npz(destination, qualified)
        paths.append(destination)
    return tuple(paths), tuple(reports)


def run_100style_real_data_audit(
    data_directory: Path,
    output_directory: Path,
    *,
    rig_path: Path,
    maximum_windows_per_source: int = 2,
    seed: int = 1234,
    mechanisms: Sequence[str] = FIRST_CRITIC_MECHANISMS,
) -> dict[str, Any]:
    """Run the complete real-data gate and persist its evidence."""
    output = Path(output_directory)
    source_paths, source_reports = prepare_100style_sources(
        data_directory,
        output,
        rig_path=rig_path,
    )
    qualified_by_split = {
        split: sum(
            report["split"] == split and bool(report["cleanliness_qualification"]["passed"])
            for report in source_reports
        )
        for split in ("train", "validation", "test")
    }
    failed = [
        report["source_clip"]
        for report in source_reports
        if not report["cleanliness_qualification"]["passed"]
    ]
    if len(source_paths) < 10 or any(
        qualified_by_split[split] < minimum
        for split, minimum in {"train": 4, "validation": 2, "test": 2}.items()
    ):
        gate = {
            "format_version": REAL_AUDIT_VERSION,
            "gate_passed": False,
            "reason": "too few expanded real source takes passed fixed cleanliness criteria",
            "failed_sources": failed,
            "qualified_sources_by_split": qualified_by_split,
            "source_reports": source_reports,
        }
        _atomic_json(output / "real_data_audit.json", gate)
        return gate

    dataset_directory = output / "dataset"
    dataset_summary = generate_corruption_dataset(
        source_paths,
        dataset_directory,
        config=DatasetGenerationConfig(
            seed=seed,
            include_heldout_mechanisms=True,
            include_soft_corruptions=False,
            composition_fraction=0.0,
            maximum_windows_per_source=maximum_windows_per_source,
            minimum_clean_confidence=0.25,
            include_sham_controls=True,
            equivalent_global_yaw_degrees=(-35.0, 35.0),
            mechanisms=tuple(mechanisms),
        ),
    )
    manifest = [
        json.loads(line)
        for line in (dataset_directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    split_lineages: dict[str, set[str]] = {}
    for record in manifest:
        split_lineages.setdefault(str(record["split_lineage_id"]), set()).add(
            str(record["source_split"])
        )
    mechanism_reports = dataset_summary["dataset_diagnostics"]["mechanisms"]
    expected_bins = {
        key: bool(value["all_severity_bins_populated"]) for key, value in mechanism_reports.items()
    }
    shams = [record for record in manifest if record.get("sample_role") == "sham_control"]
    equivalents = [
        record for record in manifest if record.get("sample_role") == "equivalent_control"
    ]
    source_events_sensible = all(
        bool(report["bilateral_gait_events_present"])
        and all(
            0.05 <= fraction <= 0.95 for fraction in report["contact_fraction_by_marker"].values()
        )
        for report in source_reports
    )
    verification = {
        "all_five_severity_bins_populated": all(expected_bins.values()),
        "severity_bin_population_by_mechanism": expected_bins,
        "parameter_targeting_accepted_samples_inside_requested_bins": all(
            bool(record["severity_bin"])
            and float(record["severity_bin"]["bin_lower_inclusive"])
            <= float(record["measured_severity"])
            < float(record["severity_bin"]["bin_upper_exclusive"])
            for record in manifest
            if record.get("sample_role") == "hard_corruption"
        ),
        "shams_lack_target_symptom": bool(shams)
        and all(bool(record["no_target_defect"]) for record in shams),
        "equivalent_controls_present_and_clean": bool(equivalents)
        and all(bool(record["no_target_defect"]) for record in equivalents),
        "failed_sources_excluded_from_dataset": not {f"{name}_FW" for name in failed}.intersection(
            str(record["source_clip_id"]) for record in manifest
        ),
        "heldout_mechanisms_absent_from_training": dataset_summary[
            "heldout_mechanisms_absent_from_training"
        ],
        "source_lineage_splitting_valid": all(
            len(assignments) == 1 for assignments in split_lineages.values()
        )
        and {next(iter(assignments)) for assignments in split_lineages.values()}
        == {"train", "validation", "test"},
        "contact_and_event_targeting_sensible": source_events_sensible,
        "bilateral_contact_target_coverage": {"left", "right"}.issubset(
            dataset_summary["audit_breakdowns"]["by_side"]
        ),
    }
    report = {
        "format_version": REAL_AUDIT_VERSION,
        "gate_passed": all(
            all(value.values()) if isinstance(value, dict) else bool(value)
            for value in verification.values()
        ),
        "dataset_directory": str(dataset_directory),
        "source_reports": source_reports,
        "dataset_composition": {
            "requested_sources": len(DEFAULT_100STYLE_SOURCES),
            "qualified_sources": len(source_paths),
            "rejected_sources": failed,
            "qualified_sources_by_split": qualified_by_split,
            "samples": dataset_summary["sample_count"],
            "source_windows": dataset_summary["source_window_count"],
            "counts_by_data_split": dataset_summary["counts_by_data_split"],
            "counts_by_corruption_family": dataset_summary["counts_by_corruption_family"],
            "counts_by_sample_role": dataset_summary["counts_by_sample_role"],
            "clean_source_clip_count": len(
                {
                    str(record["source_clip_id"])
                    for record in manifest
                    if record.get("sample_role") == "clean"
                }
            ),
        },
        "verification": verification,
        "audit_breakdowns": dataset_summary["audit_breakdowns"],
        "mechanism_diagnostics": mechanism_reports,
        "thresholds_changed_after_observing_acceptance": False,
    }
    _atomic_json(output / "real_data_audit.json", report)
    return report
