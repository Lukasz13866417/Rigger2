"""Fixed-rig residual perceptual-quality experiments."""

from motionlab.perceptual.dataset import (
    generate_hard_feasible_perceptual_dataset,
    load_perceptual_pairs,
)
from motionlab.perceptual.experiment import (
    PerceptualTrainingConfig,
    load_perceptual_model,
    train_and_evaluate_perceptual_residual,
)
from motionlab.perceptual.human_audit import (
    analyze_perceptual_human_audit,
    prepare_perceptual_human_audit,
)
from motionlab.perceptual.labeling import (
    apply_human_labels,
    build_labeling_queue,
    serve_pair_labeler,
)
from motionlab.perceptual.model import FixedRigPerceptualResidual
from motionlab.perceptual.optimization import run_gated_perceptual_cma
from motionlab.perceptual.relative_experiment import run_relative_comparator_experiment
from motionlab.perceptual.relative_model import FixedRigRelativeComparator
from motionlab.perceptual.relative_optimization import run_gated_relative_perceptual_cma
from motionlab.perceptual.reporting import compile_perceptual_milestone_report

__all__ = [
    "FixedRigPerceptualResidual",
    "FixedRigRelativeComparator",
    "PerceptualTrainingConfig",
    "analyze_perceptual_human_audit",
    "apply_human_labels",
    "build_labeling_queue",
    "compile_perceptual_milestone_report",
    "generate_hard_feasible_perceptual_dataset",
    "load_perceptual_model",
    "load_perceptual_pairs",
    "prepare_perceptual_human_audit",
    "run_gated_perceptual_cma",
    "run_gated_relative_perceptual_cma",
    "run_relative_comparator_experiment",
    "serve_pair_labeler",
    "train_and_evaluate_perceptual_residual",
]
