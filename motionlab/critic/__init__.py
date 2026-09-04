"""Learned critic baselines, intentionally limited to one fixed rig."""

from motionlab.critic.model import (
    FIRST_CRITIC_DEFECTS,
    CriticOutput,
    FixedRigFlatTCN,
    FixedRigTCNConfig,
)
from motionlab.critic.training import (
    CriticTrainingConfig,
    run_tiny_overfit_gate,
    train_fixed_rig_tcn,
)

__all__ = [
    "FIRST_CRITIC_DEFECTS",
    "CriticOutput",
    "CriticTrainingConfig",
    "FixedRigFlatTCN",
    "FixedRigTCNConfig",
    "run_tiny_overfit_gate",
    "train_fixed_rig_tcn",
]
