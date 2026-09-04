"""Fixed-rig corruption dataset generation and pickle-free sample persistence."""

from motionlab.dataset.audit import (
    CollateralDefectError,
    CollateralFinding,
    CollateralRule,
    SingleDefectPolicy,
    single_defect_policy,
    validate_single_defect,
)
from motionlab.dataset.features import (
    DATASET_SAMPLE_VERSION,
    NEURAL_INPUT_FIELDS,
    TrainingSample,
    clean_training_sample,
    corruption_training_sample,
    equivalent_training_sample,
    sham_training_sample,
)
from motionlab.dataset.generation import (
    AUTOMATIC_MECHANISMS,
    DATASET_GENERATION_VERSION,
    SEVERITY_LABELS,
    DatasetGenerationConfig,
    generate_corruption_dataset,
    validate_corruption_dataset,
)
from motionlab.dataset.io import (
    NORMALIZATION_VERSION,
    NormalizationStatistics,
    fit_training_normalization,
    load_normalization_statistics,
    load_training_sample,
    save_normalization_statistics,
    save_training_sample,
)
from motionlab.dataset.real_audit import (
    FIRST_CRITIC_MECHANISMS,
    CleanlinessQualification,
    prepare_100style_sources,
    qualify_clean_motion,
    run_100style_real_data_audit,
)
from motionlab.dataset.targeting import (
    ObservedSeverityBin,
    ObservedSeverityScale,
    SeverityTargetingError,
    TargetedCorruption,
    target_observed_severity,
)

__all__ = [
    "AUTOMATIC_MECHANISMS",
    "DATASET_GENERATION_VERSION",
    "DATASET_SAMPLE_VERSION",
    "FIRST_CRITIC_MECHANISMS",
    "NEURAL_INPUT_FIELDS",
    "NORMALIZATION_VERSION",
    "SEVERITY_LABELS",
    "CleanlinessQualification",
    "CollateralDefectError",
    "CollateralFinding",
    "CollateralRule",
    "DatasetGenerationConfig",
    "NormalizationStatistics",
    "ObservedSeverityBin",
    "ObservedSeverityScale",
    "SeverityTargetingError",
    "SingleDefectPolicy",
    "TargetedCorruption",
    "TrainingSample",
    "clean_training_sample",
    "corruption_training_sample",
    "equivalent_training_sample",
    "fit_training_normalization",
    "generate_corruption_dataset",
    "load_normalization_statistics",
    "load_training_sample",
    "prepare_100style_sources",
    "qualify_clean_motion",
    "run_100style_real_data_audit",
    "save_normalization_statistics",
    "save_training_sample",
    "sham_training_sample",
    "single_defect_policy",
    "target_observed_severity",
    "validate_corruption_dataset",
    "validate_single_defect",
]
