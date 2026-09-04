"""Purpose-separated calibration stimuli and the revised mannequin pilot.

The calibration bank deliberately contains conspicuous, interpretable defects.  It is
kept separate from the hard-feasible residual-perception dataset so those examples can
teach the rating scale without becoming critic targets.  The revised pilot then combines
hidden broad-range anchors with hard-feasible stimuli in one copied, immutable asset tree.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.corruptions.physical import corrupt_loop_seam_end_rotation
from motionlab.corruptions.temporal import corrupt_pelvis_vertical_curve
from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.forensics import motions_from_sample
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.motion.clip import MotionClip
from motionlab.perceptual.dataset import load_perceptual_pairs
from motionlab.perceptual.perturbations import PerturbationSpec, apply_perceptual_perturbation
from motionlab.perceptual.subjective import (
    MANNEQUIN_VIEWING_SETTINGS,
    NATURALNESS_SCALE,
    PILOT_MANIFEST_VERSION,
    STYLE_SCALE,
    SUBJECTIVE_PROTOCOL_VERSION,
    _canonical_hash,
    _constrained_order,
    _public_stimulus_id,
    _public_trial,
    _side_class,
    _two_cycle_payload,
    render_protocol_hash,
)

CALIBRATION_BANK_VERSION = "motionlab.subjective_calibration_bank.v1"
CALIBRATION_ONLY = "CALIBRATION_ONLY"
HARD_FEASIBLE_PERCEPTUAL = "HARD_FEASIBLE_PERCEPTUAL"
EVALUATION_ANCHOR = "EVALUATION_ANCHOR"
STIMULUS_POPULATIONS = frozenset({CALIBRATION_ONLY, HARD_FEASIBLE_PERCEPTUAL, EVALUATION_ANCHOR})
MEASUREMENT_COHORTS = frozenset({"calibration_easy", "hard_feasible"})
CALIBRATION_SEVERITIES = ("clean", "mild", "medium", "strong", "severe")
_PROTECTED_HUMAN_EVIDENCE_NAMES = (
    "raw_observations.jsonl",
    "served_playlist.jsonl",
    "PILOT_PAUSED.json",
)

_SOURCE_SEVERITY = {
    "mild": "near_threshold",
    "medium": "moderate",
    "strong": "clear",
    "severe": "severe",
}
_MEASURED_LADDER_MECHANISMS = (
    ("foot_slide", "foot_slide/hip_rotation_drift"),
    ("floating_contact", "floating_contact/root_vertical_offset"),
    ("joint_jitter", "joint_jitter/white_tangent_noise"),
    ("joint_pop", "joint_pop/local_rotation_pulse"),
    ("stride_pose_amplitude", "speed_inconsistency/leg_pose_amplitude_scale"),
    ("ground_penetration", "ground_penetration/root_vertical_offset"),
)
_PRESET_LADDERS: dict[str, tuple[float | None, ...]] = {
    "upper_body_phase_mismatch": (None, 2.0, 5.0, 9.0, 15.0),
    "excessive_rigidification": (None, 0.20, 0.45, 0.70, 1.0),
    # A lower low-pass cutoff removes more authored high-frequency character.
    "excessive_smoothing": (None, 10.0, 6.0, 3.0, 1.2),
    "torso_counter_rotation_mismatch": (None, 0.025, 0.055, 0.10, 0.16),
    "asymmetric_limb_timing": (None, 2.0, 4.0, 7.0, 11.0),
    "reduced_pelvis_bounce": (None, 0.80, 0.55, 0.25, 0.0),
    "loop_seam": (None, 0.01, 0.06, 0.20, 0.65),
}
_DEFAULT_TUTORIAL_FAMILIES = (
    "foot_slide",
    "upper_body_phase_mismatch",
    "torso_counter_rotation_mismatch",
)
_ANCHOR_RECIPES = (
    ("clean_reference", "clean", None),
    ("clean_reference", "clean", None),
    ("floating_contact", "mild", "floating_contact/root_vertical_offset"),
    ("foot_slide", "medium", "foot_slide/hip_rotation_drift"),
    ("joint_pop", "strong", "joint_pop/local_rotation_pulse"),
    ("loop_seam", "strong", "loop_seam/end_rotation_offset"),
    ("joint_jitter", "severe", "joint_jitter/white_tangent_noise"),
    ("ground_penetration", "severe", "ground_penetration/root_vertical_offset"),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        raise ValueError(f"required JSONL file is missing: {path}")
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@contextmanager
def _fresh_output_directory(destination: Path) -> Iterator[Path]:
    """Build beside a nonexistent destination, then publish it with one rename."""
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _guard_fresh_pilot_output(output: Path) -> None:
    protected = [output / name for name in _PROTECTED_HUMAN_EVIDENCE_NAMES]
    present = [path for path in protected if path.exists()]
    if present:
        raise FileExistsError(
            "refusing to replace pilot human evidence or pause sentinel: "
            + ", ".join(str(path) for path in present)
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _purpose_metadata(
    clip: MotionClip,
    *,
    population: str,
    family: str | None,
    severity: str | None,
) -> MotionClip:
    return clip.with_updates(
        metadata={
            **dict(clip.metadata),
            "stimulus_population": population,
            "measurement_cohort": (
                "hard_feasible" if population == HARD_FEASIBLE_PERCEPTUAL else "calibration_easy"
            ),
            "calibration_family": family,
            "calibration_severity": severity,
            "critic_training_eligible": population == HARD_FEASIBLE_PERCEPTUAL,
        }
    )


def _visual_motion_hash(clip: MotionClip) -> str:
    """Hash rendered motion samples without purpose/provenance metadata."""
    digest = hashlib.sha256()
    digest.update(clip.skeleton.content_hash.encode())
    digest.update(str(clip.fps).encode())
    for array in (clip.local_quat_wxyz, clip.root_translation_m):
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return "visual-motion:sha256:" + digest.hexdigest()


def _preset_parameter_semantics(family: str) -> str:
    return {
        "upper_body_phase_mismatch": "frame_shift",
        "excessive_rigidification": "blend_strength",
        "excessive_smoothing": "low_pass_cutoff_hz",
        "torso_counter_rotation_mismatch": "counter_rotation_amplitude_rad",
        "asymmetric_limb_timing": "frame_shift",
        "reduced_pelvis_bounce": "vertical_amplitude_scale",
        "loop_seam": "end_rotation_offset_rad",
    }[family]


def _apply_preset(clip: MotionClip, family: str, strength: float) -> MotionClip:
    if family == "reduced_pelvis_bounce":
        return corrupt_pelvis_vertical_curve(
            clip,
            amplitude_scale=strength,
        ).corrupted_motion
    if family == "loop_seam":
        return corrupt_loop_seam_end_rotation(
            clip,
            joint="chest",
            seam_window_frames=max(12, clip.num_frames // 2),
            angle_rad=strength,
            axis_local=(0.0, 1.0, 0.0),
            strict_postcondition=False,
        ).corrupted_motion
    spec = PerturbationSpec(
        mechanism=family,
        strength=strength,
        mechanism_partition="train",
        supervision_category="UNORDERED",
        preference=None,
        reason_tags=("calibration",),
        label_basis=None,
    )
    return apply_perceptual_perturbation(clip, spec)


def _rendered_geometry_hash(render: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("positions", "global_quat_wxyz"):
        value = np.asarray(render[name], dtype=np.float32)
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return "rendered-geometry:sha256:" + digest.hexdigest()


def _rendered_difference(
    render: dict[str, Any],
    clean_render: dict[str, Any],
) -> dict[str, float | bool]:
    positions = np.asarray(render["positions"], dtype=np.float64)
    clean_positions = np.asarray(clean_render["positions"], dtype=np.float64)
    rotations = np.asarray(render["global_quat_wxyz"], dtype=np.float64)
    clean_rotations = np.asarray(clean_render["global_quat_wxyz"], dtype=np.float64)
    if positions.shape != clean_positions.shape or rotations.shape != clean_rotations.shape:
        raise ValueError("calibration rung and clean reference render shapes differ")
    if np.array_equal(positions, clean_positions) and np.array_equal(rotations, clean_rotations):
        return {
            "tracked_position_max_abs_m": 0.0,
            "global_rotation_max_geodesic_deg": 0.0,
            "representable": False,
        }
    position_difference = float(np.max(np.abs(positions - clean_positions)))
    rotation_norm = np.linalg.norm(rotations, axis=-1, keepdims=True)
    clean_rotation_norm = np.linalg.norm(clean_rotations, axis=-1, keepdims=True)
    if np.any(rotation_norm <= 0.0) or np.any(clean_rotation_norm <= 0.0):
        raise ValueError("calibration render contains a zero quaternion")
    rotations = rotations / rotation_norm
    clean_rotations = clean_rotations / clean_rotation_norm
    dots = np.abs(np.sum(rotations * clean_rotations, axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    rotation_difference = float(np.max(np.rad2deg(2.0 * np.arccos(dots))))
    return {
        "tracked_position_max_abs_m": position_difference,
        "global_rotation_max_geodesic_deg": rotation_difference,
        "representable": position_difference > 1.0e-5 or rotation_difference > 1.0e-3,
    }


def _save_ladder_motion(
    root: Path,
    clip: MotionClip,
    *,
    family: str,
    severity: str,
) -> str:
    relative = Path("motions") / _slug(family) / f"{severity}-{clip.content_hash[7:23]}.npz"
    save_motion_npz(root / relative, clip)
    return relative.as_posix()


def _render_metadata(root: Path, motion: str, phase_reference: str) -> dict[str, Any]:
    return _two_cycle_payload(
        root / motion,
        phase_reference_path=root / phase_reference,
        viewing_settings=MANNEQUIN_VIEWING_SETTINGS,
    )


def _dataset_indices(
    dataset: FixedRigSampleDataset,
) -> tuple[dict[str, int], dict[str, int], dict[str, list[int]]]:
    sample_index: dict[str, int] = {}
    clean_by_group: dict[str, int] = {}
    chains: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        sample_index[str(record["sample_id"])] = index
        group = str(record.get("counterfactual_group_id") or "")
        if record.get("sample_role") == "clean" and group:
            clean_by_group[group] = index
        chain = str(record.get("ordinal_chain_id") or "")
        if record.get("sample_role") == "hard_corruption" and chain:
            chains[chain].append(index)
    return sample_index, clean_by_group, chains


def _complete_measured_chains(
    dataset: FixedRigSampleDataset,
    chains: dict[str, list[int]],
    mechanism: str,
) -> list[dict[str, int]]:
    required = set(_SOURCE_SEVERITY.values())
    result: list[dict[str, int]] = []
    for indices in chains.values():
        rows = [dataset.records[index] for index in indices]
        if not rows or rows[0].get("catalog_mechanism_key") != mechanism:
            continue
        by_severity = {
            str(row.get("severity_label")): index for row, index in zip(rows, indices, strict=True)
        }
        if required.issubset(by_severity):
            result.append({name: by_severity[name] for name in required})
    return result


def _source_key(record: dict[str, Any]) -> tuple[str, int]:
    return (str(record.get("source_clip_id")), int(record.get("source_window_start", 0)))


def _make_calibration_stimulus(
    root: Path,
    clip: MotionClip,
    phase_reference_clip: MotionClip,
    *,
    family: str,
    severity: str,
    origin: dict[str, Any],
) -> dict[str, Any]:
    prepared = _purpose_metadata(
        clip,
        population=CALIBRATION_ONLY,
        family=family,
        severity=severity,
    )
    prepared_reference = _purpose_metadata(
        phase_reference_clip,
        population=CALIBRATION_ONLY,
        family=family,
        severity="clean",
    )
    reference_motion = _save_ladder_motion(
        root,
        prepared_reference,
        family=family,
        severity="clean",
    )
    motion = (
        reference_motion
        if severity == "clean"
        else _save_ladder_motion(root, prepared, family=family, severity=severity)
    )
    render = _render_metadata(root, motion, reference_motion)
    clean_render = _render_metadata(root, reference_motion, reference_motion)
    persisted = load_motion_npz(root / motion)
    stimulus_id = _public_stimulus_id(motion, MANNEQUIN_VIEWING_SETTINGS)
    return {
        "stimulus_id": stimulus_id,
        "motion": motion,
        "phase_reference_motion": reference_motion,
        "motion_content_hash": persisted.content_hash,
        "visual_content_hash": _visual_motion_hash(persisted),
        "render_hash": render["render_hash"],
        "rendered_geometry_hash": _rendered_geometry_hash(render),
        "rendered_difference_from_clean": _rendered_difference(render, clean_render),
        "render_protocol_hash": render_protocol_hash(MANNEQUIN_VIEWING_SETTINGS),
        "cycle_window": render["cycle_window"],
        "source_id": str(origin["source_clip_id"]),
        "source_window_start": int(origin["source_window_start"]),
        "source_split": str(origin["source_split"]),
        "stimulus_population": CALIBRATION_ONLY,
        "measurement_cohort": "calibration_easy",
        "calibration_family": family,
        "calibration_severity": severity,
        "critic_training_eligible": False,
        "quality_evaluation_eligible": False,
        "origin": origin,
    }


def _find_renderable_clean_indices(
    dataset: FixedRigSampleDataset,
    root: Path,
    *,
    excluded_source_keys: set[tuple[str, int]],
) -> list[int]:
    result: list[int] = []
    scratch = root / ".eligibility"
    for index, record in enumerate(dataset.records):
        if (
            record.get("sample_role") != "clean"
            or record.get("source_split") != "train"
            or _source_key(record) in excluded_source_keys
        ):
            continue
        clean, _ = motions_from_sample(dataset, index)
        path = scratch / f"{index}.npz"
        save_motion_npz(path, clean)
        try:
            relative = path.relative_to(root).as_posix()
            _render_metadata(root, relative, relative)
        except ValueError:
            continue
        result.append(index)
    shutil.rmtree(scratch, ignore_errors=True)
    return result


def prepare_subjective_calibration_bank(
    corruption_dataset_directory: Path,
    output_directory: Path,
    *,
    seed: int = 9501,
) -> dict[str, Any]:
    """Create measured and purpose-built five-rung calibration ladders.

    The destination must not exist.  Every motion is copied into the bank, every rung is
    validated with the fixed mannequin two-cycle renderer, and every stimulus is explicitly
    ineligible for critic training and quality evaluation.
    """
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    source_root = Path(corruption_dataset_directory)
    dataset = FixedRigSampleDataset(source_root, regime="all", cache_samples=False)
    _, clean_by_group, chains = _dataset_indices(dataset)
    rng = random.Random(seed)

    with _fresh_output_directory(output) as staging:
        stimuli: list[dict[str, Any]] = []
        ladders: list[dict[str, Any]] = []
        used_source_keys: set[tuple[str, int]] = set()

        for family, mechanism in _MEASURED_LADDER_MECHANISMS:
            candidates = _complete_measured_chains(dataset, chains, mechanism)
            rng.shuffle(candidates)
            selected_rows: dict[str, int] | None = None
            clean: MotionClip | None = None
            for candidate in candidates:
                representative = dataset.records[candidate["near_threshold"]]
                if representative.get("source_split") != "train":
                    continue
                if _source_key(representative) in used_source_keys:
                    continue
                group = str(representative.get("counterfactual_group_id") or "")
                clean_index = clean_by_group.get(group)
                if clean_index is None:
                    continue
                proposed, _ = motions_from_sample(dataset, clean_index)
                scratch = staging / ".candidate.npz"
                save_motion_npz(scratch, proposed)
                try:
                    _render_metadata(staging, scratch.name, scratch.name)
                except ValueError:
                    scratch.unlink(missing_ok=True)
                    continue
                scratch.unlink(missing_ok=True)
                selected_rows = candidate
                clean = proposed
                break
            if selected_rows is None or clean is None:
                raise ValueError(f"no render-eligible complete measured chain for {mechanism}")

            rung_ids: list[str] = []
            measured_values: list[float] = []
            representative = dataset.records[selected_rows["near_threshold"]]
            used_source_keys.add(_source_key(representative))
            for severity in CALIBRATION_SEVERITIES:
                if severity == "clean":
                    clip = clean
                    row = representative
                    measured = 0.0
                    source_label = "clean"
                    clean_record = dataset.records[
                        clean_by_group[str(row["counterfactual_group_id"])]
                    ]
                    sample_id = str(clean_record["sample_id"])
                else:
                    source_label = _SOURCE_SEVERITY[severity]
                    row_index = selected_rows[source_label]
                    row = dataset.records[row_index]
                    _, clip = motions_from_sample(dataset, row_index)
                    measured = float(row["measured_severity"])
                    measured_values.append(measured)
                    sample_id = str(row["sample_id"])
                origin = {
                    "kind": "phase8_measured_ordinal_chain",
                    "corruption_dataset": str(source_root.resolve()),
                    "sample_id": sample_id,
                    "ordinal_chain_id": representative["ordinal_chain_id"],
                    "source_clip_id": row["source_clip_id"],
                    "source_window_start": row["source_window_start"],
                    "source_split": row["source_split"],
                    "catalog_mechanism_key": mechanism,
                    "source_severity_label": source_label,
                    "measured_severity": measured,
                    "severity_metric": (
                        None
                        if severity == "clean"
                        else row.get("severity_bin", {}).get("metric_name")
                    ),
                }
                stimulus = _make_calibration_stimulus(
                    staging,
                    clip,
                    clean,
                    family=family,
                    severity=severity,
                    origin=origin,
                )
                stimuli.append(stimulus)
                rung_ids.append(stimulus["stimulus_id"])
            if measured_values != sorted(measured_values) or len(set(measured_values)) != 4:
                raise ValueError(
                    f"measured calibration ladder is not strictly ordered: {mechanism}"
                )
            ladders.append(
                {
                    "calibration_family": family,
                    "construction": "phase8_measured_ordinal_chain",
                    "catalog_mechanism_key": mechanism,
                    "ordered_severities": list(CALIBRATION_SEVERITIES),
                    "stimulus_ids": rung_ids,
                }
            )

        clean_indices = _find_renderable_clean_indices(
            dataset,
            staging,
            excluded_source_keys=used_source_keys,
        )
        rng.shuffle(clean_indices)
        if len(clean_indices) < len(_PRESET_LADDERS):
            raise ValueError(
                "not enough independent render-eligible clean windows for preset ladders"
            )
        for (family, strengths), clean_index in zip(
            _PRESET_LADDERS.items(),
            clean_indices[: len(_PRESET_LADDERS)],
            strict=True,
        ):
            record = dataset.records[clean_index]
            clean, _ = motions_from_sample(dataset, clean_index)
            used_source_keys.add(_source_key(record))
            rung_ids = []
            for severity, strength in zip(CALIBRATION_SEVERITIES, strengths, strict=True):
                clip = clean if strength is None else _apply_preset(clean, family, strength)
                origin = {
                    "kind": "calibration_only_preset",
                    "corruption_dataset": str(source_root.resolve()),
                    "sample_id": record["sample_id"],
                    "source_clip_id": record["source_clip_id"],
                    "source_window_start": record["source_window_start"],
                    "source_split": record["source_split"],
                    "preset_parameter": strength,
                    "preset_parameter_semantics": _preset_parameter_semantics(family),
                }
                stimulus = _make_calibration_stimulus(
                    staging,
                    clip,
                    clean,
                    family=family,
                    severity=severity,
                    origin=origin,
                )
                stimuli.append(stimulus)
                rung_ids.append(stimulus["stimulus_id"])
            ladders.append(
                {
                    "calibration_family": family,
                    "construction": "calibration_only_preset",
                    "ordered_severities": list(CALIBRATION_SEVERITIES),
                    "preset_parameters": list(strengths),
                    "stimulus_ids": rung_ids,
                }
            )

        manifest = {
            "format_version": CALIBRATION_BANK_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "seed": seed,
            "source_corruption_dataset": str(source_root.resolve()),
            "stimulus_directory": str(output.resolve()),
            "render_protocol_hash": render_protocol_hash(MANNEQUIN_VIEWING_SETTINGS),
            "viewing_settings": MANNEQUIN_VIEWING_SETTINGS,
            "purpose_policy": {
                "stimulus_population": CALIBRATION_ONLY,
                "measurement_cohort": "calibration_easy",
                "critic_training_eligible": False,
                "quality_evaluation_eligible": False,
                "severity_and_family_visible_only_during_calibration": True,
            },
            "ordered_severities": list(CALIBRATION_SEVERITIES),
            "ladder_count": len(ladders),
            "stimulus_count": len(stimuli),
            "ladders": ladders,
            "stimuli": sorted(stimuli, key=lambda item: item["stimulus_id"]),
        }
        validate_calibration_bank_manifest(manifest, staging)
        _write_json(staging / "calibration_bank.json", manifest)
        summary = {
            "format_version": CALIBRATION_BANK_VERSION,
            "status": "ready_for_calibration_tutorial",
            "manifest": str((output / "calibration_bank.json").resolve()),
            "ladder_count": len(ladders),
            "stimulus_count": len(stimuli),
            "measured_ladder_count": len(_MEASURED_LADDER_MECHANISMS),
            "preset_ladder_count": len(_PRESET_LADDERS),
            "critic_training_eligible_count": 0,
        }
        _write_json(staging / "calibration_bank_preparation.json", summary)
    return summary


def validate_calibration_bank_manifest(manifest: dict[str, Any], root: Path) -> None:
    """Enforce complete ladders and calibration-only purpose separation."""
    if manifest.get("format_version") != CALIBRATION_BANK_VERSION:
        raise ValueError("unsupported calibration-bank format")
    if manifest.get("viewing_settings") != MANNEQUIN_VIEWING_SETTINGS:
        raise ValueError("calibration bank must use the mannequin viewing protocol")
    stimuli = {str(item["stimulus_id"]): item for item in manifest.get("stimuli", [])}
    if len(stimuli) != int(manifest.get("stimulus_count", -1)):
        raise ValueError("calibration-bank stimulus IDs must be unique and counted exactly")
    families: set[str] = set()
    for ladder in manifest.get("ladders", []):
        family = str(ladder.get("calibration_family"))
        if family in families:
            raise ValueError(f"duplicate calibration ladder for {family}")
        families.add(family)
        if ladder.get("ordered_severities") != list(CALIBRATION_SEVERITIES):
            raise ValueError(f"calibration ladder has incomplete severity order: {family}")
        rung_ids = list(ladder.get("stimulus_ids", []))
        if len(rung_ids) != len(CALIBRATION_SEVERITIES) or len(set(rung_ids)) != len(rung_ids):
            raise ValueError(f"calibration ladder must contain five unique rungs: {family}")
        rung_render_hashes: set[str] = set()
        rung_geometry_hashes: set[str] = set()
        rung_visual_hashes: set[str] = set()
        for severity, stimulus_id in zip(CALIBRATION_SEVERITIES, rung_ids, strict=True):
            stimulus = stimuli.get(str(stimulus_id))
            if stimulus is None:
                raise ValueError(f"calibration ladder references unknown stimulus {stimulus_id}")
            if (
                stimulus.get("stimulus_population") != CALIBRATION_ONLY
                or stimulus.get("measurement_cohort") != "calibration_easy"
                or stimulus.get("critic_training_eligible") is not False
                or stimulus.get("quality_evaluation_eligible") is not False
                or stimulus.get("calibration_family") != family
                or stimulus.get("calibration_severity") != severity
            ):
                raise ValueError(f"calibration purpose metadata is inconsistent: {stimulus_id}")
            if not (Path(root) / str(stimulus["motion"])).is_file():
                raise ValueError(f"calibration motion is missing: {stimulus['motion']}")
            if not (Path(root) / str(stimulus["phase_reference_motion"])).is_file():
                raise ValueError(
                    f"calibration phase reference is missing: {stimulus['phase_reference_motion']}"
                )
            persisted = load_motion_npz(Path(root) / str(stimulus["motion"]))
            if stimulus.get("motion_content_hash") != persisted.content_hash:
                raise ValueError(f"calibration motion content hash is stale: {stimulus_id}")
            if stimulus.get("visual_content_hash") != _visual_motion_hash(persisted):
                raise ValueError(f"calibration visual content hash is stale: {stimulus_id}")
            difference = stimulus.get("rendered_difference_from_clean", {})
            if severity != "clean" and difference.get("representable") is not True:
                raise ValueError(
                    f"calibration rung is invisible after tracked rendering: {family}/{severity}"
                )
            if severity == "clean" and (
                difference.get("representable") is not False
                or float(difference.get("tracked_position_max_abs_m", -1.0)) != 0.0
                or float(difference.get("global_rotation_max_geodesic_deg", -1.0)) > 1.0e-8
            ):
                raise ValueError(f"clean calibration rung must have zero render delta: {family}")
            rung_render_hashes.add(str(stimulus.get("render_hash")))
            rung_geometry_hashes.add(str(stimulus.get("rendered_geometry_hash")))
            rung_visual_hashes.add(str(stimulus.get("visual_content_hash")))
        if (
            len(rung_render_hashes) != len(CALIBRATION_SEVERITIES)
            or len(rung_geometry_hashes) != len(CALIBRATION_SEVERITIES)
            or len(rung_visual_hashes) != len(CALIBRATION_SEVERITIES)
        ):
            raise ValueError(f"calibration ladder contains duplicate rendered content: {family}")
    if len(families) != int(manifest.get("ladder_count", -1)):
        raise ValueError("calibration-bank ladder count is inconsistent")


def _copy_motion(source: Path, root: Path, *, namespace: str) -> str:
    if not source.is_file():
        raise ValueError(f"stimulus motion is missing: {source}")
    relative = Path("motions") / namespace / source.name
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"copied motion basename collision: {source.name}")
    else:
        shutil.copy2(source, destination)
    return relative.as_posix()


def _balanced_pair_selection(
    candidates: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    count: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for audit, pair in candidates:
        buckets[str(pair["perturbation_mechanism"])].append((audit, pair))
    for values in buckets.values():
        values.sort(
            key=lambda value: (
                int(value[0].get("audit_rank", 10**9)),
                str(value[1]["pair_id"]),
            )
        )
        # Rank remains primary, while ties remain reproducibly randomized.
        tied: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for value in values:
            tied[int(value[0].get("audit_rank", 10**9))].append(value)
        values[:] = []
        for rank in sorted(tied):
            rng.shuffle(tied[rank])
            values.extend(tied[rank])
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    families = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        families.sort(
            key=lambda family: (
                sum(str(pair["perturbation_mechanism"]) == family for _, pair in selected),
                family,
            )
        )
        progressed = False
        for family in families:
            if buckets[family] and len(selected) < count:
                selected.append(buckets[family].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible pairs exist for {count} selections")
    return selected


def _hfp_stimulus(
    root: Path,
    pair_root: Path,
    pair: dict[str, Any],
    audit: dict[str, Any],
    *,
    motion_key: str,
    variant_kind: str,
) -> dict[str, Any]:
    original_motion = str(pair[motion_key])
    original_reference = str(pair["motion_a"])
    motion = _copy_motion(pair_root / original_motion, root, namespace="hard_feasible")
    reference = _copy_motion(
        pair_root / original_reference,
        root,
        namespace="hard_feasible",
    )
    render = _render_metadata(root, motion, reference)
    persisted = load_motion_npz(root / motion)
    stimulus_id = _public_stimulus_id(motion, MANNEQUIN_VIEWING_SETTINGS)
    difficulty = str(audit.get("difficulty", "medium"))
    quality = (
        "excellent_candidate"
        if variant_kind == "clean_reference"
        else {
            "clear": "poor_candidate",
            "medium": "mid_candidate",
            "subtle": "near_reference_candidate",
        }.get(difficulty, "mid_candidate")
    )
    return {
        "stimulus_id": stimulus_id,
        "motion": motion,
        "phase_reference_motion": reference,
        "motion_content_hash": persisted.content_hash,
        "visual_content_hash": _visual_motion_hash(persisted),
        "render_hash": render["render_hash"],
        "render_protocol_hash": render_protocol_hash(MANNEQUIN_VIEWING_SETTINGS),
        "cycle_window": render["cycle_window"],
        "source_id": str(pair["source_clip_id"]),
        "source_sample_id": pair.get("source_sample_id"),
        "source_window_start": persisted.metadata.get("source_window_start"),
        "source_split": str(pair.get("source_split", "")),
        "variant_id": f"{pair['pair_id']}:{'a' if motion_key == 'motion_a' else 'b'}",
        "variant_kind": variant_kind,
        "family": (
            "clean_reference"
            if variant_kind == "clean_reference"
            else str(pair["perturbation_mechanism"])
        ),
        "style": str(pair["source_clip_id"]),
        "side_class": (
            "neutral"
            if variant_kind == "clean_reference"
            else _side_class(str(pair["perturbation_mechanism"]))
        ),
        "display_mirror_x": False,
        "expected_quality_stratum": quality,
        "hidden_repeat_group_id": _canonical_hash("repeat", {"stimulus_id": stimulus_id}),
        "selection_reasons": list(audit.get("selection_reasons", [])),
        "optimization_output": bool(pair.get("adversarial_origin", False)),
        "pair_id": str(pair["pair_id"]),
        "stimulus_population": HARD_FEASIBLE_PERCEPTUAL,
        "measurement_cohort": "hard_feasible",
        "calibration_family": None,
        "calibration_severity": None,
        "critic_training_eligible": True,
        "quality_evaluation_eligible": True,
    }


def _pair_is_renderable(pair_root: Path, pair: dict[str, Any]) -> bool:
    source = pair_root / str(pair["motion_a"])
    try:
        _two_cycle_payload(
            source,
            phase_reference_path=source,
            viewing_settings=MANNEQUIN_VIEWING_SETTINGS,
        )
    except ValueError:
        return False
    return True


def _select_hfp_pairs(
    pair_root: Path,
    pairs: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    *,
    stimulus_count: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_pair = {str(pair["pair_id"]): pair for pair in pairs}
    eligible = [
        (audit, by_pair[str(audit["pair_id"])])
        for audit in audit_rows
        if str(audit.get("pair_id")) in by_pair
        and _pair_is_renderable(pair_root, by_pair[str(audit["pair_id"])])
    ]
    if not eligible:
        raise ValueError("audit queue contains no mannequin-renderable hard-feasible pairs")
    by_reference: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in eligible:
        by_reference[str(item[1]["motion_a"])].append(item)
    ranked_groups = sorted(
        by_reference.items(),
        key=lambda item: (
            min(int(value[0].get("audit_rank", 10**9)) for value in item[1]),
            item[0],
        ),
    )
    # At most eight neutral references keeps most of the 32-stimulus cohort focused on the
    # difficult residual variants while retaining broad source/style coverage.
    group_count = min(8, len(ranked_groups), stimulus_count // 2)
    chosen_groups = ranked_groups[:group_count]
    candidate_count = stimulus_count - group_count
    pool = [value for _, values in chosen_groups for value in values]
    if candidate_count < 12:
        raise ValueError("hard-feasible cohort leaves fewer than 12 comparison candidates")
    return _balanced_pair_selection(pool, candidate_count, rng)


def _anchor_stimuli(
    dataset: FixedRigSampleDataset,
    root: Path,
    *,
    excluded_source_keys: set[tuple[str, int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    _, clean_by_group, _ = _dataset_indices(dataset)
    result: list[dict[str, Any]] = []
    used = set(excluded_source_keys)
    for family, severity, mechanism in _ANCHOR_RECIPES:
        candidates: list[int] = []
        source_label = _SOURCE_SEVERITY.get(severity)
        for index, record in enumerate(dataset.records):
            if _source_key(record) in used:
                continue
            if mechanism is None:
                if record.get("sample_role") == "clean":
                    candidates.append(index)
            elif (
                record.get("sample_role") == "hard_corruption"
                and record.get("catalog_mechanism_key") == mechanism
                and record.get("severity_label") == source_label
            ):
                candidates.append(index)
        rng.shuffle(candidates)
        selected: tuple[int, MotionClip, str, str, dict[str, Any]] | None = None
        for index in candidates:
            record = dataset.records[index]
            if mechanism is None:
                clean, _ = motions_from_sample(dataset, index)
                candidate = clean
            else:
                group = str(record.get("counterfactual_group_id") or "")
                clean_index = clean_by_group.get(group)
                if clean_index is None:
                    continue
                clean, candidate = motions_from_sample(dataset, index)
            prepared_clean = _purpose_metadata(
                clean,
                population=EVALUATION_ANCHOR,
                family=family,
                severity="clean",
            )
            prepared_candidate = _purpose_metadata(
                candidate,
                population=EVALUATION_ANCHOR,
                family=family,
                severity=severity,
            )
            namespace = f"anchors/{len(result):02d}-{_slug(family)}"
            reference = _copy_or_save_clip(root, prepared_clean, namespace, "reference")
            motion = (
                reference
                if severity == "clean"
                else _copy_or_save_clip(root, prepared_candidate, namespace, severity)
            )
            try:
                render = _render_metadata(root, motion, reference)
            except ValueError:
                # A failed candidate is confined to the fresh staging tree and is never published.
                continue
            selected = (index, prepared_candidate, motion, reference, render)
            break
        if selected is None:
            raise ValueError(f"no render-eligible independent anchor for {family}/{severity}")
        index, prepared_candidate, motion, reference, render = selected
        record = dataset.records[index]
        used.add(_source_key(record))
        stimulus_id = _public_stimulus_id(motion, MANNEQUIN_VIEWING_SETTINGS)
        persisted = load_motion_npz(root / motion)
        result.append(
            {
                "stimulus_id": stimulus_id,
                "motion": motion,
                "phase_reference_motion": reference,
                "motion_content_hash": persisted.content_hash,
                "visual_content_hash": _visual_motion_hash(persisted),
                "render_hash": render["render_hash"],
                "render_protocol_hash": render_protocol_hash(MANNEQUIN_VIEWING_SETTINGS),
                "cycle_window": render["cycle_window"],
                "source_id": str(record["source_clip_id"]),
                "source_sample_id": str(record["sample_id"]),
                "source_window_start": int(record["source_window_start"]),
                "source_split": str(record["source_split"]),
                "variant_id": f"anchor:{record['sample_id']}:{severity}",
                "variant_kind": "evaluation_anchor",
                "family": family,
                "style": str(record["source_clip_id"]),
                "side_class": "bilateral",
                "display_mirror_x": False,
                "expected_quality_stratum": f"anchor_{severity}",
                "hidden_repeat_group_id": _canonical_hash("repeat", {"stimulus_id": stimulus_id}),
                "selection_reasons": ["hidden_session_anchor", "broad_scale_coverage"],
                "optimization_output": False,
                "pair_id": None,
                "stimulus_population": EVALUATION_ANCHOR,
                "measurement_cohort": "calibration_easy",
                "calibration_family": family,
                "calibration_severity": severity,
                "critic_training_eligible": False,
                "quality_evaluation_eligible": True,
            }
        )
    return result


def _copy_or_save_clip(root: Path, clip: MotionClip, namespace: str, label: str) -> str:
    relative = Path("motions") / namespace / f"{_slug(label)}-{clip.content_hash[7:23]}.npz"
    if not (root / relative).exists():
        save_motion_npz(root / relative, clip)
    return relative.as_posix()


def _copy_tutorial_ladders(
    bank: dict[str, Any],
    bank_root: Path,
    root: Path,
    *,
    families: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bank_stimuli = {str(item["stimulus_id"]): item for item in bank["stimuli"]}
    by_family = {str(ladder["calibration_family"]): ladder for ladder in bank["ladders"]}
    tutorial_stimuli: list[dict[str, Any]] = []
    tutorial_trials: list[dict[str, Any]] = []
    for tutorial_index, family in enumerate(families):
        ladder = by_family.get(family)
        if ladder is None:
            raise ValueError(f"calibration bank has no requested tutorial family {family}")
        copied_ids: list[str] = []
        for old_id in ladder["stimulus_ids"]:
            item = bank_stimuli[str(old_id)]
            motion = _copy_motion(
                bank_root / str(item["motion"]),
                root,
                namespace=f"tutorial/{_slug(family)}",
            )
            reference = _copy_motion(
                bank_root / str(item["phase_reference_motion"]),
                root,
                namespace=f"tutorial/{_slug(family)}",
            )
            stimulus_id = _public_stimulus_id(motion, MANNEQUIN_VIEWING_SETTINGS)
            render = _render_metadata(root, motion, reference)
            copied = {
                **item,
                "stimulus_id": stimulus_id,
                "motion": motion,
                "phase_reference_motion": reference,
                "render_hash": render["render_hash"],
                "cycle_window": render["cycle_window"],
            }
            tutorial_stimuli.append(copied)
            copied_ids.append(stimulus_id)
        render_hashes = [
            _presented_render_hash(
                next(item for item in tutorial_stimuli if item["stimulus_id"] == stimulus_id),
                False,
            )
            for stimulus_id in copied_ids
        ]
        tutorial_trials.append(
            {
                "tutorial_id": _canonical_hash(
                    "tutorial", {"family": family, "stimuli": copied_ids}
                ),
                "tutorial_index": tutorial_index,
                "task_type": "calibration",
                "stimulus_ids": copied_ids,
                "presentation_order": copied_ids,
                "render_hashes": render_hashes,
                "stimulus_populations": [CALIBRATION_ONLY] * len(copied_ids),
                "measurement_cohorts": ["calibration_easy"] * len(copied_ids),
                "critic_training_eligible": False,
                "tutorial_disclosure": {
                    "calibration_family": family,
                    "ordered_severities": list(CALIBRATION_SEVERITIES),
                    "instruction": (
                        "View the same defect from clean through severe. Replay and inspect "
                        "freely; these examples only teach the rating scale."
                    ),
                },
            }
        )
    return tutorial_stimuli, tutorial_trials


def _presented_render_hash(stimulus: dict[str, Any], mirror_x: bool) -> str:
    return _canonical_hash(
        "presented-render",
        {"base_render_hash": stimulus["render_hash"], "mirror_x": mirror_x},
    )


def _single_trial(
    stimulus: dict[str, Any],
    *,
    session_index: int,
    occurrence: int,
    seed: int,
) -> dict[str, Any]:
    population = str(stimulus["stimulus_population"])
    return {
        "trial_id": _canonical_hash(
            "trial",
            {
                "stimulus": stimulus["stimulus_id"],
                "session": session_index,
                "occurrence": occurrence,
                "seed": seed,
            },
        ),
        "task_type": "naturalness",
        "stimulus_ids": [stimulus["stimulus_id"]],
        "source_ids": [stimulus["source_id"]],
        "families": [stimulus["family"]],
        "styles": [stimulus["style"]],
        "side_classes": [stimulus["side_class"]],
        "expected_quality_strata": [stimulus["expected_quality_stratum"]],
        "hidden_repeat_group_ids": [stimulus["hidden_repeat_group_id"]],
        "hidden_anchor": population == EVALUATION_ANCHOR,
        "comparison_mode": None,
        "presentation_order": [stimulus["stimulus_id"]],
        "view_mirror_x": bool(stimulus["display_mirror_x"]),
        "render_hashes": [_presented_render_hash(stimulus, bool(stimulus["display_mirror_x"]))],
        "pair_id": stimulus.get("pair_id"),
        "schedule_reason": (
            "hidden_exact_anchor_repeat"
            if population == EVALUATION_ANCHOR and occurrence > 0
            else (
                "hidden_exact_hard_feasible_repeat"
                if population == HARD_FEASIBLE_PERCEPTUAL and occurrence > 0
                else (
                    "hidden_broad_range_anchor"
                    if population == EVALUATION_ANCHOR
                    else "hard_feasible_single_stimulus"
                )
            )
        ),
        "repeat_occurrence_index": occurrence,
        "session_index": session_index,
        "seed": seed,
        "stimulus_populations": [population],
        "measurement_cohorts": [str(stimulus["measurement_cohort"])],
        "critic_training_eligible": bool(stimulus["critic_training_eligible"]),
    }


def _select_hfp_repeat_stimuli(
    stimuli: Sequence[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    candidates = sorted(
        stimuli,
        key=lambda item: (
            item["variant_kind"] != "candidate",
            not bool(item["selection_reasons"]),
            not bool(item["optimization_output"]),
            item["family"],
            item["stimulus_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    used_families: Counter[str] = Counter()
    while len(selected) < count:
        available = [item for item in candidates if item not in selected]
        if not available:
            raise ValueError("not enough hard-feasible stimuli for hidden repeats")
        chosen = min(
            available,
            key=lambda item: (
                used_families[str(item["family"])],
                item["variant_kind"] != "candidate",
                not bool(item["selection_reasons"]),
                item["stimulus_id"],
            ),
        )
        selected.append(chosen)
        used_families[str(chosen["family"])] += 1
    return selected


def _cross_session_repeat_gaps(
    sessions: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, int]:
    positions: dict[str, list[int]] = defaultdict(list)
    global_position = 0
    for trials in sessions:
        for trial in trials:
            for group in trial["hidden_repeat_group_ids"]:
                positions[str(group)].append(global_position)
            global_position += 1
    return {group: values[1] - values[0] for group, values in positions.items() if len(values) == 2}


def _comparison_pool(
    selected_pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    stimulus_by_original_motion: dict[str, dict[str, Any]],
    *,
    count: int,
    seed: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    chosen = _balanced_pair_selection(selected_pairs, count, rng)
    clean_first = [True] * (count // 2) + [False] * (count - count // 2)
    rng.shuffle(clean_first)
    trials: list[dict[str, Any]] = []
    ordered = zip(chosen, clean_first, strict=True)
    for index, ((audit, pair), show_clean_first) in enumerate(ordered):
        del audit
        clean = stimulus_by_original_motion[str(pair["motion_a"])]
        candidate = stimulus_by_original_motion[str(pair["motion_b"])]
        canonical = [clean["stimulus_id"], candidate["stimulus_id"]]
        order = canonical if show_clean_first else list(reversed(canonical))
        session_index = index % 2
        trials.append(
            {
                "trial_id": _canonical_hash(
                    "trial",
                    {
                        "adaptive_pair": pair["pair_id"],
                        "session": session_index,
                        "seed": seed,
                    },
                ),
                "trial_index": -1,
                "task_type": "pair_comparison",
                "stimulus_ids": canonical,
                "source_ids": [str(pair["source_clip_id"])],
                "families": [str(pair["perturbation_mechanism"])],
                "styles": [str(pair["source_clip_id"])],
                "side_classes": [_side_class(str(pair["perturbation_mechanism"]))],
                "expected_quality_strata": [
                    clean["expected_quality_stratum"],
                    candidate["expected_quality_stratum"],
                ],
                "hidden_repeat_group_ids": [
                    _canonical_hash("pair-repeat", {"pair": pair["pair_id"]})
                ],
                "hidden_anchor": False,
                "comparison_mode": "same_viewport_toggle",
                "presentation_order": order,
                "view_mirror_x": False,
                "render_hashes": [
                    _presented_render_hash(
                        clean if stimulus_id == clean["stimulus_id"] else candidate,
                        False,
                    )
                    for stimulus_id in canonical
                ],
                "pair_id": str(pair["pair_id"]),
                "equivalence_control": pair.get("preference") == "approximately_equal",
                "schedule_reason": "adaptive:precomputed_information_stratified_toggle_pool",
                "repeat_occurrence_index": 0,
                "session_index": session_index,
                "eligible_session_indices": [session_index],
                "seed": seed,
                "stimulus_populations": [
                    HARD_FEASIBLE_PERCEPTUAL,
                    HARD_FEASIBLE_PERCEPTUAL,
                ],
                "measurement_cohorts": ["hard_feasible", "hard_feasible"],
                "critic_training_eligible": True,
            }
        )
    return trials


def _bank_path(value: Path) -> tuple[Path, Path]:
    supplied = Path(value)
    path = supplied / "calibration_bank.json" if supplied.is_dir() else supplied
    if not path.is_file():
        raise ValueError(f"calibration-bank manifest is missing: {path}")
    return path, path.parent


def prepare_mannequin_subjective_pilot(
    pair_dataset_directory: Path,
    audit_queue_path: Path,
    corruption_dataset_directory: Path,
    calibration_bank: Path,
    output_directory: Path,
    *,
    seed: int = 9502,
    scored_stimulus_count: int = 40,
    hard_feasible_stimulus_count: int = 32,
    anchor_count: int = 8,
    comparison_count: int = 12,
    minimum_repeat_gap: int = 12,
    tutorial_families: Sequence[str] = _DEFAULT_TUTORIAL_FAMILIES,
) -> dict[str, Any]:
    """Prepare the purpose-separated 40-stimulus revised mannequin pilot."""
    if scored_stimulus_count != 40:
        raise ValueError("the revised pilot is intentionally fixed to 40 scored stimuli")
    if hard_feasible_stimulus_count != 32 or anchor_count != 8:
        raise ValueError("the revised pilot requires 32 hard-feasible stimuli and 8 anchors")
    if not 10 <= comparison_count <= 15:
        raise ValueError("the revised pilot requires 10 to 15 adaptive comparisons")
    if hard_feasible_stimulus_count + anchor_count != scored_stimulus_count:
        raise ValueError("scored population counts are inconsistent")
    if not tutorial_families:
        raise ValueError("the revised pilot requires a visible calibration tutorial")
    output = Path(output_directory)
    _guard_fresh_pilot_output(output)

    pair_root = Path(pair_dataset_directory)
    corruption_root = Path(corruption_dataset_directory)
    bank_manifest_path, bank_root = _bank_path(calibration_bank)
    bank = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    validate_calibration_bank_manifest(bank, bank_root)
    pairs = load_perceptual_pairs(pair_root)
    audit_rows = _read_jsonl(audit_queue_path)
    dataset = FixedRigSampleDataset(corruption_root, regime="all", cache_samples=False)
    rng = random.Random(seed)

    with _fresh_output_directory(output) as staging:
        internal_bank_root = staging / "calibration_bank"
        shutil.copytree(bank_root, internal_bank_root)
        internal_bank = json.loads(json.dumps(bank))
        internal_bank["stimulus_directory"] = str((output / "calibration_bank").resolve())
        _write_json(internal_bank_root / "calibration_bank.json", internal_bank)
        internal_preparation_path = internal_bank_root / "calibration_bank_preparation.json"
        if internal_preparation_path.is_file():
            internal_preparation = json.loads(internal_preparation_path.read_text(encoding="utf-8"))
            internal_preparation["manifest"] = str(
                (output / "calibration_bank" / "calibration_bank.json").resolve()
            )
            _write_json(internal_preparation_path, internal_preparation)
        validate_calibration_bank_manifest(internal_bank, internal_bank_root)

        selected_pairs = _select_hfp_pairs(
            pair_root,
            pairs,
            audit_rows,
            stimulus_count=hard_feasible_stimulus_count,
            rng=rng,
        )
        first_by_reference: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for audit, pair in selected_pairs:
            first_by_reference.setdefault(str(pair["motion_a"]), (audit, pair))
        if len(first_by_reference) + len(selected_pairs) != hard_feasible_stimulus_count:
            raise ValueError(
                "hard-feasible pair selection did not produce exactly 32 unique motions"
            )

        scored: list[dict[str, Any]] = []
        by_original_motion: dict[str, dict[str, Any]] = {}
        for original, (audit, pair) in sorted(first_by_reference.items()):
            item = _hfp_stimulus(
                staging,
                pair_root,
                pair,
                audit,
                motion_key="motion_a",
                variant_kind="clean_reference",
            )
            scored.append(item)
            by_original_motion[original] = item
        for audit, pair in selected_pairs:
            item = _hfp_stimulus(
                staging,
                pair_root,
                pair,
                audit,
                motion_key="motion_b",
                variant_kind="candidate",
            )
            scored.append(item)
            by_original_motion[str(pair["motion_b"])] = item

        bank_source_keys = {
            (
                str(item["origin"].get("source_clip_id")),
                int(item["origin"].get("source_window_start", 0)),
            )
            for item in bank["stimuli"]
            if item["calibration_family"] in set(tutorial_families)
        }
        hfp_source_sample_ids = {str(pair.get("source_sample_id")) for _, pair in selected_pairs}
        excluded_source_keys = set(bank_source_keys)
        excluded_source_keys.update(
            _source_key(record)
            for record in dataset.records
            if str(record.get("sample_id")) in hfp_source_sample_ids
        )
        anchors = _anchor_stimuli(
            dataset,
            staging,
            excluded_source_keys=excluded_source_keys,
            rng=rng,
        )
        if len(anchors) != anchor_count:
            raise ValueError("anchor construction did not produce exactly eight stimuli")
        scored.extend(anchors)

        tutorial_stimuli, tutorial_trials = _copy_tutorial_ladders(
            internal_bank,
            internal_bank_root,
            staging,
            families=tutorial_families,
        )
        all_stimuli = scored + tutorial_stimuli
        stimulus_by_id = {str(item["stimulus_id"]): item for item in all_stimuli}
        if len(stimulus_by_id) != len(all_stimuli):
            raise ValueError("pilot stimulus IDs collide across purpose populations")

        hfp = [item for item in scored if item["stimulus_population"] == HARD_FEASIBLE_PERCEPTUAL]
        rng.shuffle(hfp)
        repeated_hfp = _select_hfp_repeat_stimuli(hfp, count=4)
        remaining_hfp = [item for item in hfp if item not in repeated_hfp]
        # All repeated items originate in session zero, so occurrence indices remain
        # chronological even if sessions are collected on different days.
        session_zero_hfp = repeated_hfp + remaining_hfp[:16]
        session_one_hfp = remaining_hfp[16:]
        if len(session_zero_hfp) != 20 or len(session_one_hfp) != 12:
            raise ValueError("hard-feasible session allocation is inconsistent")
        repeated_anchor_indices = {0, 3, 4, 7}
        repeated_anchors = [
            anchor for index, anchor in enumerate(anchors) if index in repeated_anchor_indices
        ]
        single_anchors = [
            anchor for index, anchor in enumerate(anchors) if index not in repeated_anchor_indices
        ]
        sessions: list[list[dict[str, Any]]] = [[], []]
        for item in session_zero_hfp:
            sessions[0].append(_single_trial(item, session_index=0, occurrence=0, seed=seed))
        for item in session_one_hfp:
            sessions[1].append(_single_trial(item, session_index=1, occurrence=0, seed=seed))
        for anchor in repeated_anchors:
            sessions[0].append(_single_trial(anchor, session_index=0, occurrence=0, seed=seed))
            sessions[1].append(_single_trial(anchor, session_index=1, occurrence=1, seed=seed))
        for anchor in single_anchors:
            sessions[1].append(_single_trial(anchor, session_index=1, occurrence=0, seed=seed))
        for item in repeated_hfp:
            sessions[1].append(_single_trial(item, session_index=1, occurrence=1, seed=seed))
        sessions[0] = _constrained_order(
            sessions[0],
            rng=rng,
            minimum_repeat_gap=minimum_repeat_gap,
        )
        previously_seen = {
            str(value) for trial in sessions[0] for value in trial["hidden_repeat_group_ids"]
        }
        unordered_session_one = sessions[1]
        for _ in range(500):
            proposed = _constrained_order(
                unordered_session_one,
                rng=rng,
                minimum_repeat_gap=minimum_repeat_gap,
                previously_seen_repeat_groups=previously_seen,
            )
            gaps = _cross_session_repeat_gaps((sessions[0], proposed))
            if gaps and min(gaps.values()) >= minimum_repeat_gap:
                sessions[1] = proposed
                break
        else:
            raise ValueError("cannot meet cross-session hidden-repeat spacing after 500 schedules")
        for trials in sessions:
            for trial_index, trial in enumerate(trials):
                trial["trial_index"] = trial_index

        adaptive_pool = _comparison_pool(
            selected_pairs,
            by_original_motion,
            count=comparison_count,
            seed=seed,
            rng=rng,
        )
        for session_index in range(2):
            next_index = len(sessions[session_index])
            for trial in adaptive_pool:
                if trial["session_index"] == session_index:
                    trial["trial_index"] = next_index
                    next_index += 1

        anchor_ids = {str(item["stimulus_id"]) for item in anchors}
        repeated_anchor_ids = {str(item["stimulus_id"]) for item in repeated_anchors}
        repeated_hfp_ids = {str(item["stimulus_id"]) for item in repeated_hfp}
        scored_ids = {str(item["stimulus_id"]) for item in scored}
        repeat_gaps = _cross_session_repeat_gaps(sessions)
        manifest = {
            "format_version": PILOT_MANIFEST_VERSION,
            "protocol_version": SUBJECTIVE_PROTOCOL_VERSION,
            "seed": seed,
            "created_utc": datetime.now(UTC).isoformat(),
            "pair_dataset_directory": str(pair_root.resolve()),
            "stimulus_directory": str(output.resolve()),
            "audit_queue": str(Path(audit_queue_path).resolve()),
            "corruption_dataset_directory": str(corruption_root.resolve()),
            "calibration_bank": str(
                (output / "calibration_bank" / "calibration_bank.json").resolve()
            ),
            "render_protocol_hash": render_protocol_hash(MANNEQUIN_VIEWING_SETTINGS),
            "viewing_settings": MANNEQUIN_VIEWING_SETTINGS,
            "scale_anchors": {
                "naturalness": list(NATURALNESS_SCALE),
                "style_adherence": list(STYLE_SCALE),
            },
            "unique_stimulus_count": scored_stimulus_count,
            "total_asset_stimulus_count": len(all_stimuli),
            "scored_stimulus_ids": sorted(scored_ids),
            "stimulus_population_counts": dict(
                sorted(Counter(item["stimulus_population"] for item in all_stimuli).items())
            ),
            "session_count": 2,
            "minimum_intervening_trials_for_repeat": minimum_repeat_gap,
            "repeat_validation": {
                "hidden_repeat_count": len(repeat_gaps),
                "minimum_global_trial_index_distance": min(repeat_gaps.values()),
                "all_repeats_cross_session": True,
            },
            "render_validation": {
                "all_stimuli_use_exactly_two_detected_gait_cycles": True,
                "all_stimuli_use_fixed_frame_rate_hz": MANNEQUIN_VIEWING_SETTINGS["frame_rate_hz"],
                "validated_stimulus_count": len(all_stimuli),
                "renderer_protocol": MANNEQUIN_VIEWING_SETTINGS["protocol"],
            },
            "presented_side_counts": dict(
                sorted(
                    Counter(
                        side
                        for trials in sessions
                        for trial in trials
                        for side in trial["side_classes"]
                    ).items()
                )
            ),
            "stimuli": sorted(all_stimuli, key=lambda item: item["stimulus_id"]),
            "calibration_tutorial": tutorial_trials,
            "hidden_anchor_stimulus_ids": sorted(anchor_ids),
            "hidden_repeated_anchor_stimulus_ids": sorted(repeated_anchor_ids),
            "hidden_repeated_hard_feasible_stimulus_ids": sorted(repeated_hfp_ids),
            "sessions": [
                {"session_index": index, "trial_count": len(trials), "trials": trials}
                for index, trials in enumerate(sessions)
            ],
            "adaptive_comparison_pool": adaptive_pool,
            "adaptive_policy": {
                "enabled": True,
                "selection_mode": "response_adaptive_toggle_pool",
                "maximum_followups_per_session": (comparison_count + 1) // 2,
                "maximum_followups_total": comparison_count,
                "followup_pool_counts_by_session": [
                    sum(trial["session_index"] == session for trial in adaptive_pool)
                    for session in range(2)
                ],
                "followup_task_type": "pair_comparison",
                "required_comparison_mode": "same_viewport_toggle",
                "eligible_stimulus_populations": [HARD_FEASIBLE_PERCEPTUAL],
                "priorities": [
                    "close_single_stimulus_ratings",
                    "within_rater_repeat_disagreement",
                    "model_human_error",
                    "underrepresented_source_family",
                ],
                "comparison_restriction": "same source and same editing goal",
                "realized_order_log": "served_playlist.jsonl",
            },
            "pilot_status": "ready_for_human_collection",
            "human_observation_count_created": 0,
            "critic_retraining_permitted": False,
        }
        validate_mannequin_pilot_manifest(manifest, staging)
        _write_json(staging / "pilot_manifest.json", manifest)
        summary = {
            "format_version": PILOT_MANIFEST_VERSION,
            "status": "ready_for_human_collection",
            "manifest": str((output / "pilot_manifest.json").resolve()),
            "scored_stimulus_count": scored_stimulus_count,
            "hard_feasible_stimulus_count": hard_feasible_stimulus_count,
            "hidden_anchor_count": anchor_count,
            "hidden_anchor_repeat_trial_count": len(repeated_anchors),
            "hard_feasible_repeat_trial_count": len(repeated_hfp),
            "hidden_repeat_trial_count": len(repeated_anchors) + len(repeated_hfp),
            "calibration_tutorial_ladder_count": len(tutorial_trials),
            "calibration_tutorial_stimulus_count": len(tutorial_stimuli),
            "adaptive_toggle_comparison_count": len(adaptive_pool),
            "session_base_trial_counts": [len(trials) for trials in sessions],
            "human_observations": 0,
            "critic_retraining_started": False,
        }
        _write_json(staging / "pilot_preparation.json", summary)
    return summary


def validate_mannequin_pilot_manifest(manifest: dict[str, Any], root: Path) -> None:
    """Validate population separation, blinding, counts, and copied-asset isolation."""
    if manifest.get("format_version") != PILOT_MANIFEST_VERSION:
        raise ValueError("unsupported revised-pilot manifest format")
    if manifest.get("protocol_version") != SUBJECTIVE_PROTOCOL_VERSION:
        raise ValueError("revised pilot must use the current subjective protocol")
    if manifest.get("viewing_settings") != MANNEQUIN_VIEWING_SETTINGS:
        raise ValueError("revised pilot must use MANNEQUIN_VIEWING_SETTINGS")
    if Path(manifest.get("stimulus_directory", "")).resolve() != Path(root).resolve() and not (
        Path(root).name.startswith(".")
        and Path(manifest.get("stimulus_directory", "")).resolve()
        == (Path(root).parent / Path(root).name.split(".building-")[0].lstrip(".")).resolve()
    ):
        raise ValueError("revised pilot stimulus directory does not identify its asset tree")

    stimuli = {str(item["stimulus_id"]): item for item in manifest.get("stimuli", [])}
    if len(stimuli) != len(manifest.get("stimuli", [])):
        raise ValueError("revised pilot has duplicate stimulus IDs")
    for stimulus in stimuli.values():
        population = stimulus.get("stimulus_population")
        cohort = stimulus.get("measurement_cohort")
        if population not in STIMULUS_POPULATIONS or cohort not in MEASUREMENT_COHORTS:
            raise ValueError("revised pilot has unknown population or measurement cohort")
        should_train = population == HARD_FEASIBLE_PERCEPTUAL
        if stimulus.get("critic_training_eligible") is not should_train:
            raise ValueError("critic-training eligibility does not match stimulus population")
        motion_path = Path(root) / str(stimulus["motion"])
        if not motion_path.is_file():
            raise ValueError(f"revised-pilot motion is missing: {stimulus['motion']}")
        if not (Path(root) / str(stimulus["phase_reference_motion"])).is_file():
            raise ValueError(
                f"revised-pilot phase reference is missing: {stimulus['phase_reference_motion']}"
            )
        persisted = load_motion_npz(motion_path)
        if stimulus.get("motion_content_hash") != persisted.content_hash:
            raise ValueError(
                f"revised-pilot motion content hash is stale: {stimulus['stimulus_id']}"
            )
        if stimulus.get("visual_content_hash") != _visual_motion_hash(persisted):
            raise ValueError(f"revised-pilot visual hash is stale: {stimulus['stimulus_id']}")

    scored_ids = set(str(value) for value in manifest.get("scored_stimulus_ids", []))
    if len(scored_ids) != 40:
        raise ValueError("revised pilot must have exactly 40 scored stimuli")
    scored = [stimuli[value] for value in scored_ids]
    scored_counts = Counter(item["stimulus_population"] for item in scored)
    if scored_counts != Counter({HARD_FEASIBLE_PERCEPTUAL: 32, EVALUATION_ANCHOR: 8}):
        raise ValueError("scored pilot must contain 32 hard-feasible stimuli and 8 anchors")
    if any(item["stimulus_population"] == CALIBRATION_ONLY for item in scored):
        raise ValueError("calibration-only stimuli cannot enter scored trials")

    tutorials = list(manifest.get("calibration_tutorial", []))
    if not tutorials:
        raise ValueError("revised pilot has no visible calibration tutorial")
    tutorial_ids: set[str] = set()
    for tutorial in tutorials:
        if tutorial.get("task_type") != "calibration":
            raise ValueError("tutorial entry must use the calibration task type")
        disclosure = tutorial.get("tutorial_disclosure")
        if not isinstance(disclosure, dict) or disclosure.get("ordered_severities") != list(
            CALIBRATION_SEVERITIES
        ):
            raise ValueError("tutorial must disclose the complete five-rung severity order")
        ids = [str(value) for value in tutorial.get("stimulus_ids", [])]
        if len(ids) != 5:
            raise ValueError("each tutorial entry must contain exactly five ladder rungs")
        if any(stimuli[value]["stimulus_population"] != CALIBRATION_ONLY for value in ids):
            raise ValueError("tutorial can contain only calibration-only stimuli")
        tutorial_ids.update(ids)
        public = _public_trial(tutorial, manifest)
        if public.get("tutorial_disclosure") != disclosure:
            raise ValueError("tutorial disclosure is not available to the public tutorial view")
    if tutorial_ids & scored_ids:
        raise ValueError("tutorial and scored stimuli must be disjoint")
    tutorial_visual_hashes = {
        str(stimuli[stimulus_id]["visual_content_hash"]) for stimulus_id in tutorial_ids
    }
    scored_visual_hashes = {
        str(stimuli[stimulus_id]["visual_content_hash"]) for stimulus_id in scored_ids
    }
    if tutorial_visual_hashes & scored_visual_hashes:
        raise ValueError("tutorial and scored stimuli overlap by visual content hash")

    def source_window_key(stimulus: dict[str, Any]) -> tuple[str, int] | None:
        start = stimulus.get("source_window_start")
        if start is None:
            return None
        return (str(stimulus.get("source_id")), int(start))

    tutorial_source_windows = {
        key
        for stimulus_id in tutorial_ids
        if (key := source_window_key(stimuli[stimulus_id])) is not None
    }
    scored_source_windows = {
        key
        for stimulus_id in scored_ids
        if (key := source_window_key(stimuli[stimulus_id])) is not None
    }
    if tutorial_source_windows & scored_source_windows:
        raise ValueError("tutorial and scored stimuli overlap by source window")

    anchors = set(str(value) for value in manifest.get("hidden_anchor_stimulus_ids", []))
    if len(anchors) != 8 or anchors != {
        stimulus_id
        for stimulus_id in scored_ids
        if stimuli[stimulus_id]["stimulus_population"] == EVALUATION_ANCHOR
    }:
        raise ValueError("hidden anchor IDs do not match the evaluation-anchor population")
    base_trials = [trial for session in manifest.get("sessions", []) for trial in session["trials"]]
    naturalness_counts = Counter(
        str(trial["stimulus_ids"][0])
        for trial in base_trials
        if trial.get("task_type") == "naturalness"
    )
    repeated_anchors = set(
        str(value) for value in manifest.get("hidden_repeated_anchor_stimulus_ids", [])
    )
    repeated_hfp = set(
        str(value) for value in manifest.get("hidden_repeated_hard_feasible_stimulus_ids", [])
    )
    if len(repeated_anchors) != 4 or not repeated_anchors.issubset(anchors):
        raise ValueError("exactly four evaluation anchors must be hidden repeats")
    if len(repeated_hfp) != 4 or any(
        stimuli[value]["stimulus_population"] != HARD_FEASIBLE_PERCEPTUAL for value in repeated_hfp
    ):
        raise ValueError("exactly four hard-feasible stimuli must be hidden repeats")
    if any(
        naturalness_counts[value] != (2 if value in repeated_anchors else 1) for value in anchors
    ):
        raise ValueError("anchor repeat allocation is inconsistent")
    if any(
        naturalness_counts[value] != (2 if value in repeated_hfp else 1)
        for value in scored_ids
        if value not in anchors
    ):
        raise ValueError("hard-feasible repeat allocation is inconsistent")
    if [session.get("trial_count") for session in manifest.get("sessions", [])] != [24, 24]:
        raise ValueError("revised pilot must contain 24 base presentations per session")
    repeat_gaps = _cross_session_repeat_gaps(
        [session["trials"] for session in manifest["sessions"]]
    )
    minimum_repeat_gap = int(manifest.get("minimum_intervening_trials_for_repeat", -1))
    if len(repeat_gaps) != 8 or min(repeat_gaps.values()) < minimum_repeat_gap:
        raise ValueError("global hidden-repeat spacing is below the declared minimum")
    groups_by_session = [
        Counter(
            str(group) for trial in session["trials"] for group in trial["hidden_repeat_group_ids"]
        )
        for session in manifest["sessions"]
    ]
    if any(
        groups_by_session[0][group] != 1 or groups_by_session[1][group] != 1
        for group in repeat_gaps
    ):
        raise ValueError("hidden repeats must occur once in each session")
    repeat_validation = manifest.get("repeat_validation", {})
    if repeat_validation != {
        "hidden_repeat_count": len(repeat_gaps),
        "minimum_global_trial_index_distance": min(repeat_gaps.values()),
        "all_repeats_cross_session": True,
    }:
        raise ValueError("hidden-repeat validation metadata is stale")

    for trial in base_trials:
        ids = [str(value) for value in trial["stimulus_ids"]]
        if not set(ids).issubset(scored_ids):
            raise ValueError("base scored trial references a tutorial-only stimulus")
        expected_populations = [stimuli[value]["stimulus_population"] for value in ids]
        expected_cohorts = [stimuli[value]["measurement_cohort"] for value in ids]
        if trial.get("stimulus_populations") != expected_populations:
            raise ValueError("trial stimulus populations do not match stimulus metadata")
        if trial.get("measurement_cohorts") != expected_cohorts:
            raise ValueError("trial measurement cohorts do not match stimulus metadata")
        if "tutorial_disclosure" in trial:
            raise ValueError("scored trials must remain blinded")
        public = _public_trial(trial, manifest)
        if any(
            key in public
            for key in (
                "source_ids",
                "families",
                "hidden_anchor",
                "stimulus_populations",
                "measurement_cohorts",
                "critic_training_eligible",
            )
        ):
            raise ValueError("public scored trial leaks private purpose or source metadata")

    comparisons = list(manifest.get("adaptive_comparison_pool", []))
    if not 10 <= len(comparisons) <= 15:
        raise ValueError("revised pilot requires 10 to 15 adaptive comparisons")
    if any(
        trial.get("task_type") != "pair_comparison"
        or trial.get("comparison_mode") != "same_viewport_toggle"
        or trial.get("stimulus_populations") != [HARD_FEASIBLE_PERCEPTUAL, HARD_FEASIBLE_PERCEPTUAL]
        or trial.get("measurement_cohorts") != ["hard_feasible", "hard_feasible"]
        or trial.get("critic_training_eligible") is not True
        or [int(value) for value in trial.get("eligible_session_indices", [])]
        != [int(trial.get("session_index", -1))]
        or any(
            stimuli[str(value)]["stimulus_population"] != HARD_FEASIBLE_PERCEPTUAL
            for value in trial.get("stimulus_ids", [])
        )
        or set(str(value) for value in trial["stimulus_ids"]) & anchors
        for trial in comparisons
    ):
        raise ValueError("adaptive pool must contain only hard-feasible same-viewport toggles")
    policy = manifest.get("adaptive_policy", {})
    expected_policy = {
        "enabled": True,
        "selection_mode": "response_adaptive_toggle_pool",
        "followup_task_type": "pair_comparison",
        "required_comparison_mode": "same_viewport_toggle",
        "eligible_stimulus_populations": [HARD_FEASIBLE_PERCEPTUAL],
    }
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("adaptive policy does not match the revised toggle-only contract")
    if policy.get("maximum_followups_total") != len(comparisons):
        raise ValueError("adaptive total cap does not match the comparison pool")
    pool_counts = [
        sum(int(trial["session_index"]) == session for trial in comparisons) for session in range(2)
    ]
    if policy.get("followup_pool_counts_by_session") != pool_counts:
        raise ValueError("adaptive follow-up pool counts are stale")
    cap = int(policy.get("maximum_followups_per_session", -1))
    if cap != max(pool_counts) or sum(min(cap, count) for count in pool_counts) != len(comparisons):
        raise ValueError("adaptive per-session caps cannot reach the declared comparison pool")
    scored_pair_trials = [
        trial
        for trial in (*base_trials, *comparisons)
        if trial.get("task_type") == "pair_comparison"
    ]
    if any(trial.get("comparison_mode") != "same_viewport_toggle" for trial in scored_pair_trials):
        raise ValueError("every scored pair must use the same-viewport toggle")
    if manifest.get("human_observation_count_created") != 0:
        raise ValueError("pilot preparation cannot create human observations")
    if manifest.get("critic_retraining_permitted") is not False:
        raise ValueError("critic retraining must remain paused for the revised pilot")
