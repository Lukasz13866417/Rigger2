"""Low-dimensional learned-critic optimization experiments."""

from motionlab.optimization.acceptance import (
    ProductionAcceptance,
    ProductionThresholds,
    assess_production_acceptance,
    declare_reference_speed_task,
    rank_repair_candidates,
)
from motionlab.optimization.adaptive import (
    AdaptiveResidualLayout,
    AdaptiveSplineBlock,
    LocalSplinePatch,
    active_subspace_report,
    apply_adaptive_residual,
    build_multiresolution_projections,
    production_active_region,
)
from motionlab.optimization.benchmark_semantics import (
    ProductionBenchmarkPair,
    prepare_synthetic_production_benchmark,
)
from motionlab.optimization.block_cma import BlockCMAResult, optimize_parameter_block_with_cma
from motionlab.optimization.cmaes import (
    CMAOptimizationConfig,
    OptimizationResult,
    optimize_motion_with_cma,
)
from motionlab.optimization.jitter_benchmark import run_jitter_spectral_representability
from motionlab.optimization.jitter_blocks import (
    JitterSmoothingBlock,
    LocalSpectralJitterBlock,
    SpectralMode,
    fit_oracle_spectral_jitter_projection,
)
from motionlab.optimization.optimizer_validation import run_guaranteed_representable_cma_suite
from motionlab.optimization.parameter_blocks import (
    CompositeMotionParameterization,
    ContactIKBlock,
    GenericSplineBlock,
    LoopSeamBlock,
    MotionParameterBlock,
    SpeedCadenceBlock,
)
from motionlab.optimization.portfolio import (
    REPAIR_METHOD_PORTFOLIO,
    declared_repair_methods,
    select_repair_from_portfolio,
)
from motionlab.optimization.production_benchmark import (
    audit_production_benchmark_eligibility,
    run_autonomous_deterministic_repair_benchmark,
    run_production_cma_benchmark,
)
from motionlab.optimization.repair_stress import (
    DETERMINISTIC_REPAIR_STRESS_VERSION,
    STRESS_FAMILIES,
    StressCaseSelection,
    StressObjective,
    deterministic_stress_objective,
    run_deterministic_repair_stress_test,
    run_deterministic_stress_case,
    select_additional_repair_stress_cases,
    select_specialized_first,
)
from motionlab.optimization.spline import SplineResidualLayout, apply_spline_residual

__all__ = [
    "DETERMINISTIC_REPAIR_STRESS_VERSION",
    "REPAIR_METHOD_PORTFOLIO",
    "STRESS_FAMILIES",
    "AdaptiveResidualLayout",
    "AdaptiveSplineBlock",
    "BlockCMAResult",
    "CMAOptimizationConfig",
    "CompositeMotionParameterization",
    "ContactIKBlock",
    "GenericSplineBlock",
    "JitterSmoothingBlock",
    "LocalSpectralJitterBlock",
    "LocalSplinePatch",
    "LoopSeamBlock",
    "MotionParameterBlock",
    "OptimizationResult",
    "ProductionAcceptance",
    "ProductionBenchmarkPair",
    "ProductionThresholds",
    "SpectralMode",
    "SpeedCadenceBlock",
    "SplineResidualLayout",
    "StressCaseSelection",
    "StressObjective",
    "active_subspace_report",
    "apply_adaptive_residual",
    "apply_spline_residual",
    "assess_production_acceptance",
    "audit_production_benchmark_eligibility",
    "build_multiresolution_projections",
    "declare_reference_speed_task",
    "declared_repair_methods",
    "deterministic_stress_objective",
    "fit_oracle_spectral_jitter_projection",
    "optimize_motion_with_cma",
    "optimize_parameter_block_with_cma",
    "prepare_synthetic_production_benchmark",
    "production_active_region",
    "rank_repair_candidates",
    "run_autonomous_deterministic_repair_benchmark",
    "run_deterministic_repair_stress_test",
    "run_deterministic_stress_case",
    "run_guaranteed_representable_cma_suite",
    "run_jitter_spectral_representability",
    "run_production_cma_benchmark",
    "select_additional_repair_stress_cases",
    "select_repair_from_portfolio",
    "select_specialized_first",
]
