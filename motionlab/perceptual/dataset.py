"""Hard-feasible pair generation for fixed-rig residual perceptual quality."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from motionlab.critic.data import FixedRigSampleDataset
from motionlab.critic.forensics import motions_from_sample
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.motion.clip import MotionClip
from motionlab.optimization.acceptance import (
    ProductionAcceptance,
    ProductionThresholds,
    assess_production_acceptance,
    declare_reference_speed_task,
)
from motionlab.perceptual.perturbations import (
    PERTURBATION_SPECS,
    apply_perceptual_perturbation,
)

PERCEPTUAL_DATASET_VERSION = "motionlab.hard_feasible_perceptual_pairs.v1"

_LEGACY_OBJECTIVE_EQUIVALENCE_SIGNATURE = {
    "perturbation_mechanism": "equivalent_origin_shift",
    "preference": "approximately_equal",
    "supervision_category": "CERTAIN",
    "label_basis": "exact world-origin equivalence",
}


def objective_equivalence_control(record: dict[str, Any]) -> tuple[bool, str]:
    """Return an explicit objective-control decision and its provenance.

    Newly generated records persist the boolean directly.  The sole legacy migration is limited
    to the exact world-origin construction that originally defined the control; a human tie or an
    arbitrary ``approximately_equal`` target is never promoted to an objective control.
    """
    if "objective_equivalence_control" in record:
        value = record["objective_equivalence_control"]
        if not isinstance(value, bool):
            raise ValueError("objective_equivalence_control must be boolean")
        provenance = record.get(
            "objective_equivalence_control_provenance",
            "explicit_pair_dataset_field",
        )
        if not isinstance(provenance, str) or not provenance:
            raise ValueError("objective-equivalence provenance must be a nonempty string")
        return value, provenance
    legacy_match = all(
        record.get(key) == value for key, value in _LEGACY_OBJECTIVE_EQUIVALENCE_SIGNATURE.items()
    )
    return legacy_match, (
        "legacy_exact_world_origin_construction"
        if legacy_match
        else "legacy_record_without_objective_equivalence_declaration"
    )


def _acceptance_dict(value: ProductionAcceptance) -> dict[str, Any]:
    return {
        "raw_metrics": value.raw_metrics,
        "constraints": value.constraints,
        "hard_invalid_reasons": list(value.hard_invalid_reasons),
        "all_required_satisfied": value.all_required_satisfied,
        "satisficing_penalty": value.satisficing_penalty,
    }


def _pair_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "perceptual-pair:sha256:" + hashlib.sha256(encoded).hexdigest()


def _save_motion_once(root: Path, clip: MotionClip) -> str:
    relative = Path("motions") / f"{clip.content_hash.removeprefix('sha256:')}.npz"
    path = root / relative
    if not path.exists():
        save_motion_npz(path, clip)
    return relative.as_posix()


def _clean_records(dataset: FixedRigSampleDataset) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, record)
        for index, record in enumerate(dataset.records)
        if record.get("sample_role") == "clean"
    ]


def _donor_for(
    motions: list[tuple[int, dict[str, Any], MotionClip]],
    position: int,
) -> MotionClip:
    _, record, _ = motions[position]
    same_split = [
        motion
        for offset, (_, other, motion) in enumerate(motions)
        if offset != position
        and other.get("source_split") == record.get("source_split")
        and other.get("source_clip_id") != record.get("source_clip_id")
    ]
    if not same_split:
        raise ValueError("style perturbation requires a donor in the same source split")
    return same_split[position % len(same_split)]


def _base_pair_record(
    *,
    source_path: str,
    candidate_path: str,
    source_record: dict[str, Any],
    mechanism: str,
    mechanism_partition: str,
    category: str,
    preference: str | None,
    reason_tags: tuple[str, ...],
    label_basis: str | None,
    source_acceptance: ProductionAcceptance,
    candidate_acceptance: ProductionAcceptance,
    adversarial_origin: bool,
    strength: float | None,
) -> dict[str, Any]:
    identity = {
        "source_motion": source_path,
        "candidate_motion": candidate_path,
        "mechanism": mechanism,
        "strength": strength,
        "source_sample_id": source_record.get("sample_id"),
        "adversarial_origin": adversarial_origin,
    }
    return {
        "pair_id": _pair_id(identity),
        "motion_a": source_path,
        "motion_b": candidate_path,
        "source_clip_id": source_record.get("source_clip_id"),
        "source_sample_id": source_record.get("sample_id"),
        "source_split": source_record.get("source_split", "adversarial"),
        "perturbation_mechanism": mechanism,
        "perturbation_strength": strength,
        "mechanism_partition": mechanism_partition,
        "supervision_category": category,
        "preference": preference,
        "reason_tags": list(reason_tags),
        "label_basis": label_basis,
        "objective_equivalence_control": (
            mechanism == "equivalent_origin_shift"
            and preference == "approximately_equal"
            and category == "CERTAIN"
            and label_basis == "exact world-origin equivalence"
        ),
        "objective_equivalence_control_provenance": "explicit_pair_dataset_field",
        "adversarial_origin": adversarial_origin,
        "both_hard_feasible": True,
        "motion_a_acceptance": _acceptance_dict(source_acceptance),
        "motion_b_acceptance": _acceptance_dict(candidate_acceptance),
    }


def _load_adversarial_pairs(
    corpus_manifest: Path,
    output: Path,
    thresholds: ProductionThresholds,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(Path(corpus_manifest).read_text(encoding="utf-8"))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in manifest["records"]:
        pair = record["adversarial_quality_pair"]
        source, candidate = declare_reference_speed_task(
            load_motion_npz(Path(pair["preferred"])),
            load_motion_npz(Path(pair["rejected"])),
        )
        source_acceptance = assess_production_acceptance(source, source, thresholds=thresholds)
        candidate_acceptance = assess_production_acceptance(
            source, candidate, thresholds=thresholds
        )
        corpus_id = str(record.get("corpus_record_id", record.get("id", "unknown")))
        if not source_acceptance.all_required_satisfied or not (
            candidate_acceptance.all_required_satisfied
        ):
            rejected.append(
                {
                    "corpus_record_id": corpus_id,
                    "adversarial_origin": True,
                    "benchmark_inclusion": "red_team_only",
                    "source_reasons": list(source_acceptance.hard_invalid_reasons),
                    "candidate_reasons": list(candidate_acceptance.hard_invalid_reasons),
                    "source_band_failures": [
                        name
                        for name, result in source_acceptance.constraints.items()
                        if result["required"] and not result["satisfied"]
                    ],
                    "candidate_band_failures": [
                        name
                        for name, result in candidate_acceptance.constraints.items()
                        if result["required"] and not result["satisfied"]
                    ],
                }
            )
            continue
        source_path = _save_motion_once(output, source)
        candidate_path = _save_motion_once(output, candidate)
        accepted.append(
            _base_pair_record(
                source_path=source_path,
                candidate_path=candidate_path,
                source_record={
                    "sample_id": corpus_id,
                    "source_clip_id": corpus_id,
                    "source_split": "adversarial",
                },
                mechanism="preserved_critic_exploit",
                mechanism_partition="adversarial",
                category="CERTAIN",
                preference="a_better",
                reason_tags=("naturalness", "coordination"),
                label_basis="existing manual overlay inspection: source > exploit",
                source_acceptance=source_acceptance,
                candidate_acceptance=candidate_acceptance,
                adversarial_origin=True,
                strength=None,
            )
        )
    return accepted, rejected


def generate_hard_feasible_perceptual_dataset(
    source_dataset_directory: Path,
    adversarial_corpus_manifest: Path,
    output_directory: Path,
    *,
    thresholds: ProductionThresholds | None = None,
) -> dict[str, Any]:
    """Generate pairs first, then retain only candidates passing every production gate."""
    limits = ProductionThresholds() if thresholds is None else thresholds
    dataset = FixedRigSampleDataset(source_dataset_directory, regime="all")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reconstructed = [
        (index, record, motions_from_sample(dataset, index)[0])
        for index, record in _clean_records(dataset)
    ]
    pairs: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    excluded_sources: list[dict[str, Any]] = []
    for position, (_, record, raw_source) in enumerate(reconstructed):
        source, _ = declare_reference_speed_task(raw_source, raw_source)
        source_acceptance = assess_production_acceptance(source, source, thresholds=limits)
        if not source_acceptance.all_required_satisfied:
            excluded_sources.append(
                {
                    "source_sample_id": record["sample_id"],
                    "source_clip_id": record["source_clip_id"],
                    "acceptance": _acceptance_dict(source_acceptance),
                }
            )
            continue
        donor = _donor_for(reconstructed, position)
        source_path = _save_motion_once(output, source)
        for spec in PERTURBATION_SPECS:
            raw_candidate = apply_perceptual_perturbation(source, spec, donor=donor)
            _, candidate = declare_reference_speed_task(source, raw_candidate)
            acceptance = assess_production_acceptance(source, candidate, thresholds=limits)
            if not acceptance.all_required_satisfied:
                rejected_candidates.append(
                    {
                        "source_sample_id": record["sample_id"],
                        "source_clip_id": record["source_clip_id"],
                        "mechanism": spec.mechanism,
                        "strength": spec.strength,
                        "acceptance": _acceptance_dict(acceptance),
                    }
                )
                continue
            candidate_path = _save_motion_once(output, candidate)
            pairs.append(
                _base_pair_record(
                    source_path=source_path,
                    candidate_path=candidate_path,
                    source_record=record,
                    mechanism=spec.mechanism,
                    mechanism_partition=spec.mechanism_partition,
                    category=spec.supervision_category,
                    preference=spec.preference,
                    reason_tags=spec.reason_tags,
                    label_basis=spec.label_basis,
                    source_acceptance=source_acceptance,
                    candidate_acceptance=acceptance,
                    adversarial_origin=False,
                    strength=spec.strength,
                )
            )
    adversarial, rejected_adversarial = _load_adversarial_pairs(
        adversarial_corpus_manifest,
        output,
        limits,
    )
    pairs.extend(adversarial)
    pairs.sort(key=lambda item: item["pair_id"])
    manifest_path = output / "pairs.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(pair, sort_keys=True, allow_nan=False) + "\n" for pair in pairs),
        encoding="utf-8",
    )
    categories = Counter(str(pair["supervision_category"]) for pair in pairs)
    mechanisms = Counter(str(pair["perturbation_mechanism"]) for pair in pairs)
    split_counts = Counter(str(pair["source_split"]) for pair in pairs)
    report = {
        "format_version": PERCEPTUAL_DATASET_VERSION,
        "source_dataset": str(source_dataset_directory),
        "adversarial_corpus": str(adversarial_corpus_manifest),
        "production_thresholds": limits.model_dump(mode="json"),
        "hard_feasible_pair_count": len(pairs),
        "human_labeled_pair_count": categories.get("HUMAN_LABELED", 0),
        "certain_pair_count": categories.get("CERTAIN", 0),
        "unordered_pair_count": categories.get("UNORDERED", 0),
        "equivalent_control_count": sum(objective_equivalence_control(pair)[0] for pair in pairs),
        "hard_feasible_adversarial_pair_count": len(adversarial),
        "red_team_only_adversarial_pair_count": len(rejected_adversarial),
        "category_counts": dict(sorted(categories.items())),
        "mechanism_counts": dict(sorted(mechanisms.items())),
        "source_split_counts": dict(sorted(split_counts.items())),
        "rejected_generated_candidate_count": len(rejected_candidates),
        "excluded_source_count": len(excluded_sources),
        "rejected_generated_candidates": rejected_candidates,
        "excluded_sources": excluded_sources,
        "adversarial_red_team_only": rejected_adversarial,
        "manifest": str(manifest_path),
        "label_policy": {
            "CERTAIN": "only exact equivalence, strong declared rules, or prior manual review",
            "HUMAN_LABELED": "reserved for labels collected by the local pair UI",
            "UNORDERED": "no quality direction is asserted from perturbation parameters",
        },
        "perturbation_specs": [asdict(spec) for spec in PERTURBATION_SPECS],
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def load_perceptual_pairs(directory: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the hard-feasible pair manifest."""
    root = Path(directory)
    records = [
        json.loads(line)
        for line in (root / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("perceptual pair dataset is empty")
    for record in records:
        value, provenance = objective_equivalence_control(record)
        record["objective_equivalence_control"] = value
        record["objective_equivalence_control_provenance"] = provenance
    ids = [record.get("pair_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("perceptual pair IDs must be unique")
    for record in records:
        if not record.get("both_hard_feasible"):
            raise ValueError("perceptual manifest contains a hard-infeasible pair")
        for field in ("motion_a", "motion_b"):
            path = root / str(record.get(field, ""))
            if not path.is_file():
                raise ValueError(f"perceptual pair motion is missing: {path}")
    return records
