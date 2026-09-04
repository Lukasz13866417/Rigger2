"""Oracle representability audit for the local spectral jitter parameterization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.forensics import motion_distances, motions_from_sample
from motionlab.io.npz import save_motion_npz
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.optimization.jitter_blocks import (
    JitterSmoothingBlock,
    SpectralOracleProjection,
    fit_oracle_spectral_jitter_projection,
)

JITTER_REPRESENTABILITY_VERSION = "motionlab.jitter_spectral_representability.v1"


def _localized_smoothness(
    clip: Any,
    joint_indices: tuple[int, ...],
    start: int,
    stop: int,
) -> float:
    values = np.asarray(smoothness_metric(clip).joint_frame_values, dtype=np.float64)
    return float(np.percentile(values[start:stop, joint_indices], 95.0))


def _projection_report(
    projection: SpectralOracleProjection,
    clean: Any,
) -> dict[str, Any]:
    residual = projection.error.rotation_rad
    return {
        "spectral_parameter_count": len(projection.block.parameter_names),
        "interval": [projection.block.start_frame, projection.block.stop_frame],
        "joint_indices": sorted({mode.joint_index for mode in projection.block.modes}),
        "frequency_indices": sorted({mode.frequency_index for mode in projection.block.modes}),
        "captured_oracle_energy_fraction": projection.captured_energy_fraction,
        "residual_rotation_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(residual))))),
        "residual_rotation_max_deg": float(np.degrees(np.max(np.abs(residual)))),
        "distance_from_clean": motion_distances(clean, projection.candidate),
    }


def _best_smoothing_projection(
    corrupted: Any,
    clean: Any,
    joints: tuple[int, ...],
    start: int,
    stop: int,
) -> tuple[Any, dict[str, Any]]:
    block = JitterSmoothingBlock(joints, start, stop)
    best: tuple[float, Any, dict[str, float]] | None = None
    attempts = 0
    for cutoff in (3.0, 5.0, 8.0, 12.0):
        for strength in (0.5, 0.75, 1.0):
            for blend in (2.0, 4.0, 6.0):
                attempts += 1
                values = block.initial_values
                values[0] = cutoff
                values[1] = strength
                values[2] = blend
                values[3] = blend
                candidate = block.apply(corrupted, values)
                distance = motion_distances(clean, candidate)
                score = float(distance["local_rotation_geodesic_rms_deg"])
                settings = {
                    "cutoff_frequency_hz": cutoff,
                    "smoothing_strength": strength,
                    "blend_in_frames": blend,
                    "blend_out_frames": blend,
                }
                if best is None or score < best[0]:
                    best = (score, candidate, settings)
    assert best is not None
    return best[1], {
        "parameter_count": len(block.parameter_names),
        "parameter_names": list(block.parameter_names),
        "attempt_count": attempts,
        "oracle_selected_settings": best[2],
        "distance_from_clean": motion_distances(clean, best[1]),
        "selection_uses_clean_reference": True,
        "production_use": "parameters are intended for deterministic-objective CMA",
    }


def _select_jitter_cases(
    dataset: FixedRigSampleDataset,
    sources: int,
) -> list[int]:
    by_source: dict[str, tuple[float, int]] = {}
    for index, record in enumerate(dataset.records):
        if (
            record.get("sample_role") != "hard_corruption"
            or record.get("catalog_partition") != "train"
            or record.get("corruption_family") != "joint_jitter"
            or record.get("severity_label") != "moderate"
        ):
            continue
        source = str(record["source_clip_id"])
        severity = float(record.get("measured_severity") or 0.0)
        incumbent = by_source.get(source)
        if incumbent is None or severity > incumbent[0]:
            by_source[source] = (severity, index)
    if len(by_source) < sources:
        raise ValueError("insufficient distinct jitter sources")
    return [
        item[1][1]
        for item in sorted(by_source.items(), key=lambda item: (-item[1][0], item[0]))[:sources]
    ]


def run_jitter_spectral_representability(
    dataset_directory: Path,
    output_directory: Path,
    *,
    sources: int = 2,
) -> dict[str, Any]:
    """Verify high-bandwidth jitter with sparse spectral and smoothing blocks."""
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for index in _select_jitter_cases(dataset, sources):
        record = dataset.records[index]
        clean, corrupted = motions_from_sample(dataset, index)
        projection_95 = fit_oracle_spectral_jitter_projection(
            corrupted,
            clean,
            energy_fraction=0.95,
        )
        projection_full = fit_oracle_spectral_jitter_projection(
            corrupted,
            clean,
            energy_fraction=1.0,
        )
        joints = tuple(sorted({mode.joint_index for mode in projection_full.block.modes}))
        start = projection_full.block.start_frame
        stop = projection_full.block.stop_frame
        smoothed, smoothing_report = _best_smoothing_projection(
            corrupted,
            clean,
            joints,
            start,
            stop,
        )
        case_id = f"joint_jitter__{record['source_clip_id']}__{str(record['sample_id'])[14:26]}"
        case_output = output / case_id
        save_motion_npz(case_output / "clean_evaluation_only.npz", clean)
        save_motion_npz(case_output / "corrupted.npz", corrupted)
        save_motion_npz(case_output / "spectral_95.npz", projection_95.candidate)
        save_motion_npz(case_output / "spectral_full.npz", projection_full.candidate)
        save_motion_npz(case_output / "smoothing_oracle_grid_best.npz", smoothed)
        case = {
            "case_id": case_id,
            "source_clip_id": record["source_clip_id"],
            "sample_id": record["sample_id"],
            "representation": "local DCT tangent residual composed in SO(3)",
            "generic_spline_resolution_increased": False,
            "affected_joint_indices": list(joints),
            "affected_joint_names": [clean.skeleton.joint_names[joint] for joint in joints],
            "interval": [start, stop],
            "localized_smoothness_before_rad_s3_p95": _localized_smoothness(
                corrupted, joints, start, stop
            ),
            "spectral_95": _projection_report(projection_95, clean),
            "spectral_full": _projection_report(projection_full, clean),
            "smoothing_block": {
                **smoothing_report,
                "localized_smoothness_after_rad_s3_p95": _localized_smoothness(
                    smoothed, joints, start, stop
                ),
            },
        }
        cases.append(case)
        (case_output / "representability.json").write_text(
            json.dumps(case, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    report = {
        "format_version": JITTER_REPRESENTABILITY_VERSION,
        "case_count": len(cases),
        "conclusion": (
            "high-bandwidth jitter uses a local spectral block; generic spline density "
            "was not increased"
        ),
        "cases": cases,
    }
    (output / "jitter_spectral_representability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
