"""MotionLab command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console

from motionlab._version import __version__
from motionlab.config import load_config
from motionlab.corruptions.artifacts import save_corruption_artifact
from motionlab.corruptions.catalog import CORRUPTION_CATALOG, CORRUPTION_CATALOG_VERSION
from motionlab.corruptions.foot_slide import corrupt_foot_slide_root_drift
from motionlab.critic.evaluation import evaluate_fixed_rig_tcn
from motionlab.critic.forensics import motions_from_sample
from motionlab.critic.model import FixedRigTCNConfig
from motionlab.critic.training import (
    CriticTrainingConfig,
    run_tiny_overfit_gate,
    train_fixed_rig_tcn,
)
from motionlab.dataset.generation import (
    DatasetGenerationConfig,
    generate_corruption_dataset,
    validate_corruption_dataset,
)
from motionlab.dataset.real_audit import run_100style_real_data_audit
from motionlab.demo import run_deterministic_demo
from motionlab.io.bvh import BvhImportOptions, import_bvh, load_bvh
from motionlab.io.npz import load_motion_npz, save_motion_npz
from motionlab.io.rig_spec import load_rig_spec
from motionlab.kinematics.retarget import retarget_rest_relative
from motionlab.metrics.aggregate import grade_motion_deterministic
from motionlab.metrics.artifacts import save_deterministic_report
from motionlab.noticeability.cli import register_commands as register_noticeability_commands
from motionlab.optimization.cmaes import CMAOptimizationConfig, optimize_motion_with_cma
from motionlab.optimization.diagnostics import (
    deterministic_production_rerun_gate,
    reclassify_production_constraints,
    run_multiresolution_oracle_representability,
)
from motionlab.optimization.experiment import run_first_cma_experiment
from motionlab.optimization.jitter_benchmark import run_jitter_spectral_representability
from motionlab.optimization.optimizer_validation import run_guaranteed_representable_cma_suite
from motionlab.optimization.production_benchmark import (
    run_autonomous_deterministic_repair_benchmark,
    run_production_cma_benchmark,
)
from motionlab.optimization.repair_stress import run_deterministic_repair_stress_test
from motionlab.perceptual.active_query import ActiveQueryConfig, prepare_active_query_schedule
from motionlab.perceptual.calibration_bank import (
    prepare_mannequin_subjective_pilot,
    prepare_subjective_calibration_bank,
)
from motionlab.perceptual.camera_continuation import prepare_camera_continuation
from motionlab.perceptual.candidate_pool import generate_unlabeled_candidate_pool
from motionlab.perceptual.comparator_readiness import (
    prepare_comparator_readiness_report,
    prepare_post_label_comparator_evaluation,
)
from motionlab.perceptual.dataset import generate_hard_feasible_perceptual_dataset
from motionlab.perceptual.experiment import train_and_evaluate_perceptual_residual
from motionlab.perceptual.human_audit import (
    analyze_perceptual_human_audit,
    prepare_perceptual_human_audit,
)
from motionlab.perceptual.labeling import serve_pair_labeler
from motionlab.perceptual.motioncritic_baseline import prepare_motioncritic_baseline_report
from motionlab.perceptual.optimization import run_gated_perceptual_cma
from motionlab.perceptual.post_label_forensics import build_post_label_forensics
from motionlab.perceptual.protocol_freeze import (
    freeze_active_evaluation_protocol,
    validate_active_evaluation_protocol,
)
from motionlab.perceptual.relative_experiment import run_relative_comparator_experiment
from motionlab.perceptual.relative_optimization import run_gated_relative_perceptual_cma
from motionlab.perceptual.reporting import compile_perceptual_milestone_report
from motionlab.perceptual.subjective import prepare_subjective_pilot
from motionlab.perceptual.subjective_analysis import analyze_subjective_pilot
from motionlab.processing.canonicalize import canonicalize_origin_and_facing
from motionlab.processing.resample import resample_motion
from motionlab.repair.close_loop import close_loop
from motionlab.repair.foot_lock import lock_foot
from motionlab.testing.synthetic import make_synthetic_walk

app = typer.Typer(
    name="motionlab",
    help="Import, validate, grade, and repair humanoid walking animation.",
    no_args_is_help=True,
)
console = Console()
register_noticeability_commands(app)


@app.command()
def version() -> None:
    """Print the installed MotionLab version."""
    console.print(__version__)


@app.command("check-config")
def check_config(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Validate an application YAML configuration."""
    config = load_config(path)
    if json_output:
        console.print(json.dumps(config.model_dump(mode="json"), sort_keys=True))
    else:
        console.print(f"Valid configuration: {path}")


@app.command("make-synthetic")
def make_synthetic(
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    frames: Annotated[int, typer.Option(min=2)] = 121,
    fps: Annotated[float, typer.Option(min=1.0)] = 60.0,
) -> None:
    """Create a deterministic synthetic humanoid walk fixture."""
    clip = make_synthetic_walk(num_frames=frames, fps=fps)
    save_motion_npz(output, clip)
    console.print(f"Wrote {clip.num_frames}-frame fixture: {output}")


@app.command("inspect-bvh")
def inspect_bvh(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Parse a BVH and report its hierarchy, channels, and timing."""
    data = load_bvh(path)
    result = {
        "path": str(path),
        "frames": data.num_frames,
        "frame_time_s": data.frame_time_s,
        "fps": 1.0 / data.frame_time_s,
        "joints": len(data.joint_names),
        "channels": data.num_channels,
        "joint_names": list(data.joint_names),
        "channel_order": [list(channels) for channels in data.channels],
        "end_sites": len(data.end_sites),
    }
    if json_output:
        console.print(json.dumps(result, sort_keys=True))
    else:
        console.print(
            f"BVH: {data.num_frames} frames, {len(data.joint_names)} joints, "
            f"{1.0 / data.frame_time_s:g} fps, {data.num_channels} channels"
        )


@app.command("convert-bvh")
def convert_bvh(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    rig_path: Annotated[
        Path | None,
        typer.Option("--rig", exists=True, dir_okay=False, readable=True),
    ] = None,
    source_handedness: Annotated[str, typer.Option()] = "right",
    source_up: Annotated[str, typer.Option()] = "Y",
    source_forward: Annotated[str, typer.Option()] = "Z",
    unit_scale_to_m: Annotated[float, typer.Option(min=1.0e-12)] = 1.0,
    target_fps: Annotated[float, typer.Option(min=1.0)] = 60.0,
    canonicalize_facing: Annotated[bool, typer.Option()] = False,
    source_dataset: Annotated[str, typer.Option()] = "unspecified_bvh",
    source_clip_id: Annotated[str | None, typer.Option()] = None,
    source_take_id: Annotated[str | None, typer.Option()] = None,
    split: Annotated[str | None, typer.Option()] = None,
    clean_confidence: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
) -> None:
    """Convert BVH motion into canonical, versioned MotionLab NPZ data."""
    if source_handedness not in {"right", "left"}:
        raise typer.BadParameter("must be 'right' or 'left'", param_hint="source_handedness")
    if split not in {None, "train", "validation", "test"}:
        raise typer.BadParameter(
            "must be 'train', 'validation', or 'test'",
            param_hint="split",
        )
    rig = None if rig_path is None else load_rig_spec(rig_path)
    clip = import_bvh(
        path,
        options=BvhImportOptions(
            handedness=source_handedness,  # type: ignore[arg-type]
            up_axis=source_up,
            forward_axis=source_forward,
            unit_scale_to_m=unit_scale_to_m,
        ),
        role_names=None if rig is None else dict(rig.joints),
        markers=None if rig is None else dict(rig.markers),
        joint_limits=None if rig is None else dict(rig.joint_limits),
        source_dataset=source_dataset,
        source_clip_id=source_clip_id,
        source_take_id=source_take_id,
        split=split,  # type: ignore[arg-type]
        clean_confidence=clean_confidence,
    )
    clip = resample_motion(clip, target_fps)
    if canonicalize_facing:
        clip = canonicalize_origin_and_facing(clip)
    save_motion_npz(output, clip)
    console.print(f"Converted {path} -> {output} ({clip.num_frames} frames at {clip.fps:g} fps)")


@app.command("validate-motion")
def validate_motion(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Load and validate a MotionLab NPZ motion."""
    clip = load_motion_npz(path)
    result = {
        "motion_id": clip.content_hash,
        "frames": clip.num_frames,
        "joints": clip.num_joints,
        "fps": clip.fps,
        "duration_s": clip.duration_s,
    }
    if json_output:
        console.print(json.dumps(result, sort_keys=True))
    else:
        console.print(
            f"Valid motion {clip.content_hash[:19]}: "
            f"{clip.num_frames} frames, {clip.num_joints} joints, {clip.fps:g} fps"
        )


@app.command("preview")
def preview(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    frame: Annotated[int, typer.Option(min=0)] = 0,
) -> None:
    """Save a static 3D skeleton preview for a MotionLab NPZ frame."""
    from motionlab.visualization.skeleton_plot import save_skeleton_preview

    clip = load_motion_npz(path)
    save_skeleton_preview(clip, output, frame=frame)
    console.print(f"Wrote preview: {output}")


@app.command("metrics")
def metrics(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    target_speed_mps: Annotated[float | None, typer.Option(min=0.0)] = None,
    ground_height_m: Annotated[float, typer.Option()] = 0.0,
    dense_output: Annotated[
        Path | None,
        typer.Option("--dense-output", dir_okay=False),
    ] = None,
) -> None:
    """Generate compact JSON diagnostics plus a linked dense NPZ artifact."""
    clip = load_motion_npz(path)
    report = grade_motion_deterministic(
        clip,
        ground_plane=(0.0, 1.0, 0.0, -ground_height_m),
        target_speed_mps=target_speed_mps,
    )
    summary_path, artifact_path = save_deterministic_report(
        output,
        report,
        dense_path=dense_output,
    )
    console.print(f"Wrote deterministic report: {summary_path} (dense: {artifact_path})")


@app.command("retarget")
def retarget(
    source_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_reference_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    scale_root_translation: Annotated[
        bool,
        typer.Option("--scale-root-translation/--no-scale-root-translation"),
    ] = True,
) -> None:
    """Retarget a clip onto the skeleton stored in a target-reference MotionLab NPZ."""
    source = load_motion_npz(source_path)
    target_reference = load_motion_npz(target_reference_path)
    result = retarget_rest_relative(
        source,
        target_reference.skeleton,
        scale_root_translation=scale_root_translation,
    )
    save_motion_npz(output, result)
    console.print(
        json.dumps(
            {
                "motion_id": result.content_hash,
                "output": str(output),
                "source_skeleton_id": source.skeleton.content_hash,
                "target_skeleton_id": target_reference.skeleton.content_hash,
                "shared_roles": result.metadata["shared_roles"],
                "root_scale_ratio": result.metadata["root_scale_ratio"],
            },
            sort_keys=True,
        )
    )


@app.command("run-demo")
def run_demo(
    workspace: Annotated[Path, typer.Option("--workspace", file_okay=False)],
    previews: Annotated[bool, typer.Option("--previews/--no-previews")] = True,
) -> None:
    """Run the generated clean/corrupt/repair deterministic vertical slice."""
    summary = run_deterministic_demo(workspace, render_previews=previews)
    console.print(json.dumps(summary, sort_keys=True))


@app.command("corrupt")
def corrupt(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    defect_type: Annotated[str, typer.Option("--type")] = "foot_slide",
    start_frame: Annotated[int, typer.Option(min=0)] = 5,
    stop_frame: Annotated[int, typer.Option(min=1)] = 25,
    distance_m: Annotated[float, typer.Option(min=0.0)] = 0.04,
    seed: Annotated[int, typer.Option()] = 0,
    artifact_dir: Annotated[
        Path | None,
        typer.Option("--artifact-dir", file_okay=False),
    ] = None,
) -> None:
    """Apply a measured corruption and optionally save dense training supervision."""
    if defect_type != "foot_slide":
        raise typer.BadParameter(
            "the current vertical slice supports only 'foot_slide'",
            param_hint="type",
        )
    clip = load_motion_npz(path)
    result = corrupt_foot_slide_root_drift(
        clip,
        start_frame=start_frame,
        stop_frame=stop_frame,
        distance_m=distance_m,
        seed=seed,
    )
    save_motion_npz(output, result.corrupted)
    artifact_manifest = (
        None if artifact_dir is None else save_corruption_artifact(artifact_dir, result)
    )
    payload = {
        "motion_id": result.corrupted.content_hash,
        "defect_type": result.defect_type,
        "mechanism": result.corruption_mechanism,
        "output": str(output),
        "parameters": dict(result.parameters),
        "measured_severity": result.measured_severity,
        "preference_confidence": result.preference_confidence,
        "hard_negative": result.postcondition.hard_negative,
    }
    if artifact_manifest is not None:
        payload["artifact_manifest"] = str(artifact_manifest)
    typer.echo(json.dumps(payload, sort_keys=True))


@app.command("corruption-catalog")
def corruption_catalog() -> None:
    """Print the versioned mechanism catalog and train/held-out assignments."""
    typer.echo(
        json.dumps(
            {
                "catalog_version": CORRUPTION_CATALOG_VERSION,
                "entries": [
                    {
                        "family": entry.family,
                        "mechanism": entry.mechanism,
                        "callable": entry.callable_name,
                        "measured_metric": entry.measured_metric,
                        "partition": entry.partition,
                        "hard_negative": entry.hard_negative,
                    }
                    for entry in CORRUPTION_CATALOG
                ],
            },
            sort_keys=True,
        )
    )


@app.command("generate-dataset")
def generate_dataset(
    sources: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seed: Annotated[int, typer.Option()] = 1234,
    window_frames: Annotated[int, typer.Option(min=16)] = 160,
    stride_frames: Annotated[int, typer.Option(min=1)] = 32,
    include_heldout: Annotated[
        bool,
        typer.Option("--heldout-mechanisms/--no-heldout-mechanisms"),
    ] = True,
    include_soft: Annotated[
        bool,
        typer.Option("--soft-corruptions/--no-soft-corruptions"),
    ] = True,
    composition_fraction: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    maximum_windows_per_source: Annotated[int | None, typer.Option(min=1)] = None,
    minimum_clean_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    severity_search_evaluations: Annotated[int, typer.Option(min=4, max=64)] = 14,
    include_sham_controls: Annotated[
        bool,
        typer.Option("--sham-controls/--no-sham-controls"),
    ] = True,
    cross_mechanism_relative_tolerance: Annotated[
        float,
        typer.Option(min=1.0e-6, max=1.0),
    ] = 0.2,
    mechanisms: Annotated[list[str] | None, typer.Option("--mechanism")] = None,
) -> None:
    """Generate or resume a split-safe fixed-window corruption dataset."""
    summary = generate_corruption_dataset(
        sources,
        output,
        config=DatasetGenerationConfig(
            seed=seed,
            window_frames=window_frames,
            stride_frames=stride_frames,
            include_heldout_mechanisms=include_heldout,
            include_soft_corruptions=include_soft,
            composition_fraction=composition_fraction,
            maximum_windows_per_source=maximum_windows_per_source,
            minimum_clean_confidence=minimum_clean_confidence,
            severity_search_evaluations=severity_search_evaluations,
            include_sham_controls=include_sham_controls,
            cross_mechanism_relative_tolerance=cross_mechanism_relative_tolerance,
            mechanisms=None if mechanisms is None else tuple(mechanisms),
        ),
    )
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "sample_count": summary["sample_count"],
                "preference_pair_count": summary["preference_pair_count"],
                "consistency_target_count": summary["consistency_target_count"],
                "sham_control_count": summary["sham_control_count"],
                "counts_by_data_split": summary["counts_by_data_split"],
                "resumed_existing_sample_count": summary["resumed_existing_sample_count"],
            },
            sort_keys=True,
        )
    )


@app.command("validate-dataset")
def validate_dataset(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Verify sample checksums, split lineage, holdout isolation, and normalization scope."""
    typer.echo(json.dumps(validate_corruption_dataset(directory), sort_keys=True))


@app.command("audit-100style")
def audit_100style(
    data_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    rig_path: Annotated[
        Path,
        typer.Option("--rig", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/rigs/100style.yaml"),
    maximum_windows_per_source: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option()] = 1234,
) -> None:
    """Run the fixed-threshold real-data gate on the expanded 100STYLE subset."""
    report = run_100style_real_data_audit(
        data_directory,
        output,
        rig_path=rig_path,
        maximum_windows_per_source=maximum_windows_per_source,
        seed=seed,
    )
    typer.echo(
        json.dumps(
            {
                "output": str(output / "real_data_audit.json"),
                "gate_passed": report["gate_passed"],
                "dataset_composition": report.get("dataset_composition"),
            },
            sort_keys=True,
        )
    )


@app.command("tiny-overfit-critic")
def tiny_overfit_critic(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    epochs: Annotated[int, typer.Option(min=1)] = 250,
    maximum_samples: Annotated[int, typer.Option(min=4)] = 18,
    seed: Annotated[int, typer.Option()] = 2027,
) -> None:
    """Require the fixed-rig TCN to memorize a tiny defect/localization/ranking set."""
    typer.echo(
        json.dumps(
            run_tiny_overfit_gate(
                dataset_directory,
                output,
                seed=seed,
                epochs=epochs,
                maximum_samples=maximum_samples,
            ),
            sort_keys=True,
        )
    )


@app.command("train-fixed-rig-critic")
def train_fixed_rig_critic(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    epochs: Annotated[int, typer.Option(min=1)] = 30,
    batch_size: Annotated[int, typer.Option(min=1)] = 16,
    learning_rate: Annotated[float, typer.Option(min=1.0e-8)] = 1.0e-3,
    hidden_channels: Annotated[int, typer.Option(min=8, max=512)] = 128,
    normalization: Annotated[str, typer.Option()] = "group_norm",
    seed: Annotated[int, typer.Option()] = 2027,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", exists=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Train the intentionally simple noncausal fixed-rig flat TCN baseline."""
    from motionlab.critic.data import FixedRigSampleDataset

    if normalization not in {"group_norm", "layer_norm"}:
        raise typer.BadParameter("must be group_norm or layer_norm", param_hint="normalization")
    dataset = FixedRigSampleDataset(dataset_directory, regime="train", cache_samples=False)
    result = train_fixed_rig_tcn(
        dataset_directory,
        output,
        training_config=CriticTrainingConfig(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        ),
        model_config=FixedRigTCNConfig(
            num_joints=dataset.num_joints,
            hidden_channels=hidden_channels,
            normalization=cast(Literal["group_norm", "layer_norm"], normalization),
        ),
        resume_checkpoint=resume,
    )
    typer.echo(
        json.dumps(
            {
                "last_checkpoint": str(result.checkpoint_path),
                "best_checkpoint": str(result.best_checkpoint_path),
                "stopped_epoch": result.stopped_epoch,
            },
            sort_keys=True,
        )
    )


@app.command("evaluate-fixed-rig-critic")
def evaluate_fixed_rig_critic(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seed: Annotated[int, typer.Option()] = 2027,
) -> None:
    """Evaluate known/held-out mechanisms, shams, ranking, shortcuts, and baselines."""
    report = evaluate_fixed_rig_tcn(
        dataset_directory,
        checkpoint,
        output,
        seed=seed,
    )
    typer.echo(
        json.dumps(
            {
                "output": str(output / "evaluation.json"),
                "sham": report["sham"],
                "mechanism_id_linear_probe": report["mechanism_id_linear_probe"],
            },
            sort_keys=True,
        )
    )


@app.command("optimize-critic-cma")
def optimize_critic_cma(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    sample_id: Annotated[str, typer.Option("--sample-id")],
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    family: Annotated[list[str], typer.Option("--family")],
    mode: Annotated[str, typer.Option()] = "red_team",
    seed: Annotated[int, typer.Option()] = 3301,
    coarse_iterations: Annotated[int, typer.Option(min=1)] = 6,
    refined_iterations: Annotated[int, typer.Option(min=1)] = 8,
    population_size: Annotated[int | None, typer.Option(min=4)] = None,
) -> None:
    """Optimize selected family severity heads over a smooth 4-to-8 spline residual."""
    from motionlab.critic.data import FixedRigSampleDataset

    if mode not in {"red_team", "production"}:
        raise typer.BadParameter("must be red_team or production", param_hint="mode")
    dataset = FixedRigSampleDataset(dataset_directory, regime="all")
    matching = [
        index for index, record in enumerate(dataset.records) if record["sample_id"] == sample_id
    ]
    if len(matching) != 1:
        raise typer.BadParameter("sample ID was not found exactly once", param_hint="sample-id")
    _, original = motions_from_sample(dataset, matching[0])
    weights = {name: 1.0 for name in family}
    result = optimize_motion_with_cma(
        original,
        checkpoint,
        dataset_directory,
        output,
        config=CMAOptimizationConfig(
            seed=seed,
            mode=mode,  # type: ignore[arg-type]
            objective_weights=weights,
            coarse_max_iterations=coarse_iterations,
            refined_max_iterations=refined_iterations,
            population_size=population_size,
        ),
    )
    typer.echo(
        json.dumps(
            {
                "report": str(result.report_path),
                "final": str(result.final_motion_path),
                "objective_before": result.objective_before,
                "objective_after": result.objective_after,
                "category": result.category,
                "category_label": result.category_label,
                "adversarial_training_eligible": result.adversarial_training_eligible,
            },
            sort_keys=True,
        )
    )


@app.command("run-first-cma-experiment")
def first_cma_experiment(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", exists=True, dir_okay=False, readable=True)
    ],
    evaluation_report: Annotated[
        Path, typer.Option("--evaluation", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seeds: Annotated[list[int] | None, typer.Option("--seed")] = None,
    coarse_iterations: Annotated[int, typer.Option(min=1)] = 4,
    refined_iterations: Annotated[int, typer.Option(min=1)] = 5,
    population_size: Annotated[int | None, typer.Option(min=4)] = 8,
) -> None:
    """Run foot-slide, jitter, pop, and eligible floating experiments in both modes."""
    report = run_first_cma_experiment(
        dataset_directory,
        checkpoint,
        evaluation_report,
        output,
        seeds=(3301, 3302, 3303) if not seeds else tuple(seeds),
        coarse_max_iterations=coarse_iterations,
        refined_max_iterations=refined_iterations,
        population_size=population_size,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "experiment.json"),
                "families": report["families"],
                "run_count": len(report["runs"]),
                "category_counts": report["category_counts"],
            },
            sort_keys=True,
        )
    )


@app.command("validate-cma-inverse")
def validate_cma_inverse(
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    dimensions: Annotated[list[int] | None, typer.Option("--dimension")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seed")] = None,
    evaluations_per_dimension: Annotated[int, typer.Option(min=1)] = 300,
) -> None:
    """Validate CMA on exact, feasible spline inverse problems without a critic."""
    report = run_guaranteed_representable_cma_suite(
        output,
        dimensions=(10, 25, 50, 100) if not dimensions else tuple(dimensions),
        seeds=(4701, 4702, 4703) if not seeds else tuple(seeds),
        evaluations_per_dimension=evaluations_per_dimension,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "suite.json"),
                "by_dimension": report["by_dimension"],
                "optimizer_decision": report["optimizer_decision"],
            },
            sort_keys=True,
        )
    )


@app.command("run-adaptive-oracle-diagnostics")
def adaptive_oracle_diagnostics(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    experiment_report: Annotated[
        Path, typer.Option("--experiment", exists=True, dir_okay=False, readable=True)
    ],
    previous_representability: Annotated[
        Path, typer.Option("--previous", exists=True, dir_okay=False, readable=True)
    ],
    failure_matrix: Annotated[
        Path, typer.Option("--failure-matrix", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Fit residual-driven oracle spaces, reclassify constraints, and apply the CMA gate."""
    oracle_directory = output / "multiresolution_oracle"
    oracle = run_multiresolution_oracle_representability(
        dataset_directory,
        experiment_report,
        previous_representability,
        oracle_directory,
    )
    classification = reclassify_production_constraints(
        dataset_directory,
        experiment_report,
        failure_matrix,
        oracle_directory / "multiresolution_representability.json",
        output / "production_constraint_reclassification.json",
    )
    gate = deterministic_production_rerun_gate(
        oracle_directory / "multiresolution_representability.json",
        output / "production_cma_gate.json",
    )
    typer.echo(
        json.dumps(
            {
                "before_representable_count": oracle["before_representable_count"],
                "after_representable_count": oracle["after_representable_count"],
                "production_valid_count": oracle["after_production_valid_count"],
                "production_cma_status": gate["status"],
                "remaining_constraint_causes": classification["cause_counts"],
            },
            sort_keys=True,
        )
    )


@app.command("run-production-cma-benchmark")
def production_cma_benchmark(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seeds: Annotated[list[int] | None, typer.Option("--seed")] = None,
    sources_per_family: Annotated[int, typer.Option(min=2)] = 2,
    maximum_evaluations: Annotated[int, typer.Option(min=1)] = 1_200,
) -> None:
    """Run constrained deterministic CMA on slide, floating, and pop cases."""
    report = run_production_cma_benchmark(
        dataset_directory,
        output,
        seeds=(6101, 6102, 6103) if not seeds else tuple(seeds),
        sources_per_family=sources_per_family,
        maximum_evaluations=maximum_evaluations,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "production_benchmark.json"),
                "executed_run_count": report["executed_run_count"],
                "eligibility": report["eligibility_audit"]["totals"],
                "by_family": report["by_family"],
            },
            sort_keys=True,
        )
    )


@app.command("run-jitter-spectral-audit")
def jitter_spectral_audit(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    sources: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    """Verify local spectral jitter representability and the compact smoothing block."""
    report = run_jitter_spectral_representability(
        dataset_directory,
        output,
        sources=sources,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "jitter_spectral_representability.json"),
                "case_count": report["case_count"],
                "conclusion": report["conclusion"],
            },
            sort_keys=True,
        )
    )


@app.command("run-autonomous-deterministic-repair")
def autonomous_deterministic_repair(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seed: Annotated[int, typer.Option()] = 7101,
    sources_per_family: Annotated[int, typer.Option(min=2)] = 2,
    maximum_evaluations: Annotated[int, typer.Option(min=1)] = 1_200,
) -> None:
    """Run task-directed block selection and constrained CMA without a clean target."""
    report = run_autonomous_deterministic_repair_benchmark(
        dataset_directory,
        output,
        seed=seed,
        sources_per_family=sources_per_family,
        maximum_evaluations=maximum_evaluations,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "autonomous_benchmark.json"),
                "case_count": report["case_count"],
                "selection_accuracy": report["overall_selection_accuracy"],
                "repair_success_rate": report["overall_repair_success_rate"],
                "by_family": report["by_family"],
            },
            sort_keys=True,
        )
    )


@app.command("run-deterministic-repair-stress")
def deterministic_repair_stress(
    dataset_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    prior_production_report: Annotated[
        Path | None,
        typer.Option("--prior-production-report", exists=True, dir_okay=False),
    ] = None,
    seed: Annotated[int, typer.Option()] = 9201,
    sources_per_family: Annotated[int, typer.Option(min=1)] = 2,
    maximum_evaluations: Annotated[int, typer.Option(min=1)] = 240,
    minimum_joint_limit_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
) -> None:
    """Stress the specialized-first deterministic portfolio on additional real styles."""
    report = run_deterministic_repair_stress_test(
        dataset_directory,
        output,
        prior_production_report=prior_production_report,
        seed=seed,
        sources_per_family=sources_per_family,
        maximum_evaluations=maximum_evaluations,
        minimum_joint_limit_confidence=minimum_joint_limit_confidence,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "deterministic_repair_stress.json"),
                "executed_case_count": report["executed_case_count"],
                "production_eligible_case_count": report["production_eligible_case_count"],
                "production_success_rate": report["production_success_rate"],
                "by_family": report["by_family"],
            },
            sort_keys=True,
        )
    )


@app.command("generate-perceptual-pairs")
def generate_perceptual_pairs(
    source_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    adversarial_manifest: Annotated[
        Path, typer.Option("--adversarial-manifest", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Generate constraint-screened fixed-rig perceptual-quality pairs."""
    report = generate_hard_feasible_perceptual_dataset(
        source_dataset,
        adversarial_manifest,
        output,
    )
    typer.echo(
        json.dumps(
            {
                "summary": str(output / "summary.json"),
                "hard_feasible_pairs": report["hard_feasible_pair_count"],
                "human_labeled_pairs": report["human_labeled_pair_count"],
                "adversarial_pairs": report["hard_feasible_adversarial_pair_count"],
            },
            sort_keys=True,
        )
    )


@app.command("train-perceptual-residual")
def train_perceptual_residual(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    source_dataset: Annotated[Path, typer.Option("--source-dataset", exists=True, file_okay=False)],
    backbone: Annotated[Path, typer.Option("--backbone", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    human_labels: Annotated[
        Path | None, typer.Option("--human-labels", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Train and gate the frozen-GroupNorm perceptual residual head."""
    result = train_and_evaluate_perceptual_residual(
        pair_dataset,
        source_dataset,
        backbone,
        output,
        human_labels_path=human_labels,
    )
    typer.echo(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "evaluation": str(result.evaluation_path),
                "perceptual_cma_gate_passed": result.optimization_gate_passed,
            },
            sort_keys=True,
        )
    )


@app.command("label-perceptual-pairs")
def label_perceptual_pairs(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    pilot: Annotated[Path, typer.Option("--pilot", exists=True, dir_okay=False)],
    observations: Annotated[Path, typer.Option("--observations", dir_okay=False)],
    served_playlist: Annotated[
        Path | None, typer.Option("--served-playlist", dir_okay=False)
    ] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
) -> None:
    """Serve the blinded single-stimulus and same-viewport evaluation interface."""
    console.print(f"Calibrated subjective evaluator: http://{host}:{port}")
    serve_pair_labeler(
        pair_dataset,
        observations,
        pilot_manifest=pilot,
        served_playlist_path=served_playlist,
        host=host,
        port=port,
    )


@app.command("prepare-camera-continuation")
def prepare_free_camera_pilot(
    pilot: Annotated[Path, typer.Option("--pilot", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Version a frozen pilot for freer camera inspection, retaining submitted progress."""
    report = prepare_camera_continuation(pilot, output)
    freeze = freeze_active_evaluation_protocol(Path(report["pilot_manifest"]))
    console.print_json(
        json.dumps(
            {
                **report,
                "status": "frozen_ready_for_collection",
                "freeze_id": freeze["freeze_id"],
                "protocol_id": freeze["protocol_id"],
            },
            sort_keys=True,
        )
    )


@app.command("prepare-subjective-pilot")
def prepare_hybrid_subjective_pilot(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    audit_queue: Annotated[Path, typer.Option("--audit-queue", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seed: Annotated[int, typer.Option()] = 9401,
    animations: Annotated[int, typer.Option(min=30, max=50)] = 42,
) -> None:
    """Prepare the frozen legacy skeleton pilot (not the revised mannequin study)."""
    report = prepare_subjective_pilot(
        pair_dataset,
        audit_queue,
        output,
        seed=seed,
        unique_stimulus_count=animations,
    )
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("prepare-subjective-calibration-bank")
def prepare_calibration_bank(
    corruption_dataset: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seed: Annotated[int, typer.Option()] = 9501,
) -> None:
    """Build immutable, five-rung mannequin calibration ladders."""
    report = prepare_subjective_calibration_bank(
        corruption_dataset,
        output,
        seed=seed,
    )
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("prepare-mannequin-subjective-pilot")
def prepare_revised_mannequin_subjective_pilot(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    audit_queue: Annotated[
        Path, typer.Option("--audit-queue", exists=True, dir_okay=False, readable=True)
    ],
    corruption_dataset: Annotated[
        Path,
        typer.Option(
            "--corruption-dataset",
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ],
    calibration_bank: Annotated[
        Path, typer.Option("--calibration-bank", exists=True, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    seed: Annotated[int, typer.Option()] = 9502,
    comparisons: Annotated[int, typer.Option(min=10, max=15)] = 12,
) -> None:
    """Build the purpose-separated 40-stimulus mannequin pilot."""
    report = prepare_mannequin_subjective_pilot(
        pair_dataset,
        audit_queue,
        corruption_dataset,
        calibration_bank,
        output,
        seed=seed,
        comparison_count=comparisons,
    )
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("freeze-active-evaluation-protocol")
def freeze_subjective_protocol(
    pilot: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    observations: Annotated[Path | None, typer.Option("--observations", dir_okay=False)] = None,
    served_playlist: Annotated[
        Path | None, typer.Option("--served-playlist", dir_okay=False)
    ] = None,
) -> None:
    """Freeze the exact active protocol, assets, and append-only evidence prefixes."""
    report = freeze_active_evaluation_protocol(
        pilot,
        freeze_path=output,
        observations_path=observations,
        served_playlist_path=served_playlist,
    )
    typer.echo(
        json.dumps(
            {
                "freeze_id": report["freeze_id"],
                "protocol_id": report["protocol_id"],
                "immutable_snapshot_id": report["immutable_snapshot_id"],
            },
            sort_keys=True,
        )
    )


@app.command("validate-active-evaluation-protocol")
def validate_subjective_protocol(
    freeze: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    pilot: Annotated[
        Path | None, typer.Option("--pilot", exists=True, dir_okay=False, readable=True)
    ] = None,
    observations: Annotated[Path | None, typer.Option("--observations", dir_okay=False)] = None,
    served_playlist: Annotated[
        Path | None, typer.Option("--served-playlist", dir_okay=False)
    ] = None,
) -> None:
    """Fail closed if a frozen pilot asset or collected log prefix drifted."""
    report = validate_active_evaluation_protocol(
        freeze,
        pilot_manifest_path=pilot,
        observations_path=observations,
        served_playlist_path=served_playlist,
    )
    typer.echo(
        json.dumps(
            {
                "status": report["status"],
                "freeze_id": report["freeze_id"],
                "protocol_id": report["protocol_id"],
            },
            sort_keys=True,
        )
    )


@app.command("generate-unlabeled-perceptual-candidates")
def generate_unlabeled_perceptual_candidates(
    source_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    pair_dataset: Annotated[
        Path, typer.Option("--pair-dataset", exists=True, file_okay=False, readable=True)
    ],
    adversarial_manifest: Annotated[
        Path, typer.Option("--adversarial-manifest", exists=True, dir_okay=False, readable=True)
    ],
    calibration_manifest: Annotated[
        Path, typer.Option("--calibration-manifest", exists=True, dir_okay=False, readable=True)
    ],
    protocol_freeze: Annotated[
        Path, typer.Option("--protocol-freeze", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Build a separate render-validated pool with no subjective preference labels."""
    report = generate_unlabeled_candidate_pool(
        source_dataset,
        pair_dataset,
        adversarial_manifest,
        calibration_manifest,
        protocol_freeze,
        output,
    )
    typer.echo(
        json.dumps(
            {
                "manifest": report["manifest"],
                "candidate_count": report["candidate_count"],
                "population_counts": report["population_counts"],
                "rejected_count": report["rejected_count"],
            },
            sort_keys=True,
        )
    )


@app.command("schedule-active-human-queries")
def schedule_active_queries(
    candidate_pool: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    observations: Annotated[Path, typer.Option("--observations", dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    session_id: Annotated[str, typer.Option("--session-id")],
    latent_quality: Annotated[
        Path | None, typer.Option("--latent-quality", exists=True, dir_okay=False)
    ] = None,
    critic_predictions: Annotated[
        Path | None, typer.Option("--critic-predictions", exists=True, dir_okay=False)
    ] = None,
    maximum: Annotated[int, typer.Option("--maximum", min=2)] = 20,
    maximum_singles: Annotated[int, typer.Option("--maximum-singles", min=1)] = 12,
    maximum_comparisons: Annotated[int, typer.Option("--maximum-comparisons", min=1)] = 8,
) -> None:
    """Recommend a bounded balanced mix of RATE_SINGLE and COMPARE queries."""
    report = prepare_active_query_schedule(
        candidate_pool,
        observations,
        output,
        session_id=session_id,
        latent_quality_path=latent_quality,
        critic_predictions_path=critic_predictions,
        config=ActiveQueryConfig(
            max_recommendations=maximum,
            max_rate_single=maximum_singles,
            max_compare=maximum_comparisons,
        ),
    )
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "recommendation_counts": report["recommendation_counts"],
                "candidate_audit": report["candidate_audit"],
            },
            sort_keys=True,
        )
    )


@app.command("build-post-label-forensics")
def post_label_forensics(
    pilot: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    protocol_freeze: Annotated[
        Path,
        typer.Option("--protocol-freeze", exists=True, dir_okay=False, readable=True),
    ],
    observations: Annotated[
        Path, typer.Option("--observations", exists=True, dir_okay=False, readable=True)
    ],
    stimulus_ids: Annotated[list[str], typer.Option("--stimulus-id")],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    critic_output: Annotated[
        list[str] | None,
        typer.Option("--critic-output", help="Optional STIMULUS_ID=temporal_outputs.npz"),
    ] = None,
) -> None:
    """Unlock diagnostics only after the exact judgment and blinded pilot are complete."""
    critic_paths: dict[str, Path] = {}
    for raw in critic_output or []:
        stimulus_id, separator, path_value = raw.partition("=")
        if not separator or not stimulus_id or not path_value:
            raise typer.BadParameter(
                "expected STIMULUS_ID=temporal_outputs.npz", param_hint="critic-output"
            )
        path = Path(path_value)
        if not path.is_file():
            raise typer.BadParameter(f"critic output does not exist: {path}")
        critic_paths[stimulus_id] = path
    report = build_post_label_forensics(
        pilot,
        observations,
        tuple(stimulus_ids),
        output,
        protocol_freeze_path=protocol_freeze,
        critic_outputs=critic_paths,
    )
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("analyze-subjective-pilot")
def analyze_hybrid_subjective_pilot(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    pilot: Annotated[Path, typer.Option("--pilot", exists=True, dir_okay=False)],
    observations: Annotated[Path, typer.Option("--observations", dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    protocol_freeze: Annotated[
        Path | None,
        typer.Option("--protocol-freeze", exists=True, dir_okay=False, readable=True),
    ] = None,
    served_playlist: Annotated[
        Path | None,
        typer.Option("--served-playlist", exists=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Fit the latent model and report pilot reliability without critic retraining."""
    report = analyze_subjective_pilot(
        pair_dataset,
        pilot,
        observations,
        output,
        protocol_freeze_path=protocol_freeze,
        served_playlist_path=served_playlist,
    )
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("evaluate-comparator-readiness")
def comparator_readiness(
    pairs: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    predictions: Annotated[
        Path, typer.Option("--predictions", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    evidence_origin: Annotated[
        Literal["synthetic_fixture", "human_pilot"], typer.Option("--evidence-origin")
    ] = "synthetic_fixture",
    swap_tolerance: Annotated[float, typer.Option("--swap-tolerance", min=0.0)] = 1.0e-7,
) -> None:
    """Evaluate all comparator ablations from predictions without fitting a model."""
    report = prepare_comparator_readiness_report(
        pairs,
        predictions,
        output,
        evidence_origin=evidence_origin,
        swap_tolerance=swap_tolerance,
    )
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "training_invoked": report["training_invoked"],
            },
            sort_keys=True,
        )
    )


@app.command("evaluate-post-label-comparator-ablations")
def post_label_comparator_ablations(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    pilot: Annotated[Path, typer.Option("--pilot", exists=True, dir_okay=False)],
    observations: Annotated[
        Path, typer.Option("--observations", exists=True, dir_okay=False, readable=True)
    ],
    protocol_freeze: Annotated[
        Path, typer.Option("--protocol-freeze", exists=True, dir_okay=False, readable=True)
    ],
    pilot_report: Annotated[
        Path, typer.Option("--pilot-report", exists=True, dir_okay=False, readable=True)
    ],
    absolute_evaluation: Annotated[
        Path, typer.Option("--absolute-evaluation", exists=True, dir_okay=False, readable=True)
    ],
    relative_experiment: Annotated[
        Path, typer.Option("--relative-experiment", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
    served_playlist: Annotated[
        Path | None,
        typer.Option("--served-playlist", exists=True, dir_okay=False, readable=True),
    ] = None,
    swap_tolerance: Annotated[float, typer.Option("--swap-tolerance", min=0.0)] = 1.0e-7,
) -> None:
    """Evaluate all three ablations on authenticated direct-human pairs after pilot completion."""
    report = prepare_post_label_comparator_evaluation(
        pair_dataset,
        pilot,
        observations,
        protocol_freeze,
        pilot_report,
        absolute_evaluation,
        relative_experiment,
        output,
        served_playlist_path=served_playlist,
        swap_tolerance=swap_tolerance,
    )
    typer.echo(
        json.dumps(
            {
                "output": str(Path(output) / "comparator_evaluation.json"),
                "status": report["status"],
                "training_invoked": report["training_invoked"],
            },
            sort_keys=True,
        )
    )


@app.command("evaluate-motioncritic-baseline")
def motioncritic_baseline(
    rig_contract: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    human_pairs: Annotated[
        Path | None, typer.Option("--human-pairs", exists=True, dir_okay=False, readable=True)
    ] = None,
    predictions: Annotated[
        Path | None, typer.Option("--predictions", exists=True, dir_okay=False, readable=True)
    ] = None,
) -> None:
    """Assess MotionCritic compatibility or score frozen predictions against human pairs."""
    report = prepare_motioncritic_baseline_report(
        rig_contract,
        output,
        human_pairs_path=human_pairs,
        predictions_path=predictions,
    )
    typer.echo(json.dumps({"output": str(output), "status": report["status"]}, sort_keys=True))


@app.command("prepare-perceptual-human-audit")
def prepare_human_audit(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    absolute_evaluation: Annotated[
        Path, typer.Option("--absolute-evaluation", exists=True, dir_okay=False)
    ],
    absolute_cma_report: Annotated[
        Path, typer.Option("--absolute-cma-report", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Prepare the required held-out-source human-audit queue."""
    report = prepare_perceptual_human_audit(
        pair_dataset,
        absolute_evaluation,
        absolute_cma_report,
        output,
    )
    typer.echo(json.dumps(report, sort_keys=True))


@app.command("analyze-perceptual-human-audit")
def analyze_human_audit(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    audit_queue: Annotated[Path, typer.Option("--audit-queue", exists=True, dir_okay=False)],
    labels: Annotated[Path, typer.Option("--labels", dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Compute human/synthetic agreement and audited supervision policy."""
    report = analyze_perceptual_human_audit(pair_dataset, audit_queue, labels, output)
    typer.echo(
        json.dumps(
            {
                "report": str(output),
                "status": report["status"],
                "human_labels": report["human_labeled_required_pair_count"],
                "remaining": report["remaining_pair_count"],
            },
            sort_keys=True,
        )
    )


@app.command("run-relative-comparator-experiment")
def relative_comparator_experiment(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    source_dataset: Annotated[Path, typer.Option("--source-dataset", exists=True, file_okay=False)],
    backbone: Annotated[Path, typer.Option("--backbone", exists=True, dir_okay=False)],
    human_labels: Annotated[Path, typer.Option("--human-labels", dir_okay=False)],
    human_audit: Annotated[Path, typer.Option("--human-audit", exists=True, dir_okay=False)],
    absolute_evaluation: Annotated[
        Path, typer.Option("--absolute-evaluation", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Run matched frozen/fine-tuned relative comparators after human audit."""
    result = run_relative_comparator_experiment(
        pair_dataset,
        source_dataset,
        backbone,
        human_labels,
        human_audit,
        absolute_evaluation,
        output,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(result.report_path),
                "optimization_gate_passed": result.optimization_gate_passed,
                "selected_checkpoint": (
                    None if result.selected_checkpoint is None else str(result.selected_checkpoint)
                ),
            },
            sort_keys=True,
        )
    )


@app.command("run-relative-perceptual-cma")
def relative_perceptual_cma(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    source_dataset: Annotated[Path, typer.Option("--source-dataset", exists=True, file_okay=False)],
    checkpoint: Annotated[Path, typer.Option()],
    experiment: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Run anchor-relative CMA only after its comparator gate passes."""
    report = run_gated_relative_perceptual_cma(
        pair_dataset,
        source_dataset,
        checkpoint,
        experiment,
        output,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "relative_perceptual_cma.json"),
                "status": report["status"],
                "cma_es_invoked": report.get("cma_es_invoked", True),
            },
            sort_keys=True,
        )
    )


@app.command("run-perceptual-cma")
def perceptual_cma(
    pair_dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    source_dataset: Annotated[Path, typer.Option("--source-dataset", exists=True, file_okay=False)],
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    evaluation: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Run trust-region perceptual CMA only if the preference gate permits it."""
    report = run_gated_perceptual_cma(
        pair_dataset,
        source_dataset,
        checkpoint,
        evaluation,
        output,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output / "perceptual_cma.json"),
                "status": report["status"],
                "cma_es_invoked": report.get("cma_es_invoked", True),
            },
            sort_keys=True,
        )
    )


@app.command("summarize-perceptual-milestone")
def summarize_perceptual_milestone(
    dataset_summary: Annotated[
        Path, typer.Option("--dataset-summary", exists=True, dir_okay=False)
    ],
    evaluation: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    cma_report: Annotated[Path, typer.Option("--cma-report", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Compile the fixed-rig perceptual milestone decision report."""
    report = compile_perceptual_milestone_report(
        dataset_summary,
        evaluation,
        cma_report,
        output,
    )
    typer.echo(
        json.dumps(
            {
                "report": str(output),
                "critic_ready": report["decision"]["perceptual_critic_ready_for_optimization"],
                "next_priority": report["decision"]["next_priority"],
            },
            sort_keys=True,
        )
    )


@app.command("repair")
def repair(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    operator: Annotated[str, typer.Option()] = "lock_foot",
    side: Annotated[str, typer.Option()] = "left",
    start_frame: Annotated[int, typer.Option(min=0)] = 5,
    stop_frame: Annotated[int, typer.Option(min=1)] = 25,
    seam_window_frames: Annotated[int, typer.Option(min=2)] = 12,
) -> None:
    """Apply a deterministic repair operator and save a new immutable motion version."""
    clip = load_motion_npz(path)
    if operator == "lock_foot":
        if side not in {"left", "right"}:
            raise typer.BadParameter("must be 'left' or 'right'", param_hint="side")
        result = lock_foot(
            clip,
            side=side,  # type: ignore[arg-type]
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
        repaired = result.repaired
        summary = {
            "before_slip_cm": result.before_slip_cm,
            "after_slip_cm": result.after_slip_cm,
            "maximum_ik_residual_m": result.maximum_ik_residual_m,
            "maximum_ik_target_residual_m": result.maximum_ik_target_residual_m,
            "actual_post_repair_marker_slip_cm": result.actual_post_repair_marker_slip_cm,
        }
    elif operator == "close_loop":
        loop_result = close_loop(clip, seam_window_frames=seam_window_frames)
        repaired = loop_result.repaired
        summary = {
            "before_seam": loop_result.before_seam,
            "after_seam": loop_result.after_seam,
            "reduction_fraction": loop_result.reduction_fraction,
        }
    else:
        raise typer.BadParameter(
            "supported operators are 'lock_foot' and 'close_loop'",
            param_hint="operator",
        )
    save_motion_npz(output, repaired)
    console.print(
        json.dumps(
            {
                "motion_id": repaired.content_hash,
                "operator": operator,
                "output": str(output),
                "metrics": summary,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    """Invoke the Typer application."""
    app()
