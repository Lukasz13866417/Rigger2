"""Residual perceptual head on the frozen fixed-rig GroupNorm TCN backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from motionlab.critic.model import FixedRigFlatTCN

SUPPORTED_PERCEPTUAL_DIMENSIONS = (
    "naturalness",
    "coordination",
    "rigidity_smoothing",
)


@dataclass(frozen=True)
class PerceptualOutput:
    """Primary residual quality score and only label-supported auxiliary scores."""

    score: torch.Tensor
    representation: torch.Tensor
    auxiliary_scores: dict[str, torch.Tensor]


class FixedRigPerceptualResidual(nn.Module):
    """Keep the existing TCN and learn quality only within the hard-feasible region."""

    def __init__(
        self,
        backbone: FixedRigFlatTCN,
        *,
        representation_channels: int = 64,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        hidden = backbone.config.hidden_channels
        self.perceptual_representation = nn.Sequential(
            nn.Linear(hidden, representation_channels),
            nn.GELU(),
        )
        self.preference_head = nn.Linear(representation_channels, 1)
        self.auxiliary_heads = nn.ModuleDict(
            {
                name: nn.Linear(representation_channels, 1)
                for name in SUPPORTED_PERCEPTUAL_DIMENSIONS
            }
        )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> FixedRigPerceptualResidual:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, frame_features: torch.Tensor) -> PerceptualOutput:
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_output = self.backbone(frame_features)
        else:
            backbone_output = self.backbone(frame_features)
        representation = self.perceptual_representation(backbone_output.representation)
        return PerceptualOutput(
            score=self.preference_head(representation).squeeze(-1),
            representation=representation,
            auxiliary_scores={
                name: head(representation).squeeze(-1)
                for name, head in self.auxiliary_heads.items()
            },
        )

    def score_from_backbone_representation(
        self,
        representation: torch.Tensor,
    ) -> PerceptualOutput:
        """Evaluate heads on a cached frozen-backbone representation."""
        perceptual = self.perceptual_representation(representation)
        return PerceptualOutput(
            score=self.preference_head(perceptual).squeeze(-1),
            representation=perceptual,
            auxiliary_scores={
                name: head(perceptual).squeeze(-1) for name, head in self.auxiliary_heads.items()
            },
        )
