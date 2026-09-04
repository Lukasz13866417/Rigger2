"""Symmetric fixed-rig relative motion comparator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from motionlab.critic.model import FixedRigFlatTCN


class RelativeComparatorConfig(BaseModel):
    """Small comparison-head configuration with unchanged input normalization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_channels: int = Field(default=64, ge=8, le=256)
    kernel_size: int = Field(default=3, ge=1, le=7)
    fine_tune_backbone: bool = False


@dataclass(frozen=True)
class RelativePreferenceOutput:
    """Three-way preferences ordered as A-better, B-better, approximately-equal."""

    logits: torch.Tensor
    probabilities: torch.Tensor


class _TemporalEvidence(nn.Module):
    def __init__(self, inputs: int, hidden: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(inputs, hidden, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        evidence: torch.Tensor = self.network(features.transpose(1, 2))
        return evidence.mean(dim=-1).squeeze(1)


class FixedRigRelativeComparator(nn.Module):
    """Compare aligned temporal features without assuming a global scalar utility."""

    def __init__(
        self,
        backbone: FixedRigFlatTCN,
        config: RelativeComparatorConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = RelativeComparatorConfig() if config is None else config
        hidden = backbone.config.hidden_channels
        self.directional_head = _TemporalEvidence(
            hidden * 5,
            self.config.comparison_channels,
            self.config.kernel_size,
        )
        self.equality_head = _TemporalEvidence(
            hidden * 3,
            self.config.comparison_channels,
            self.config.kernel_size,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.config.fine_tune_backbone:
            for module in (self.backbone.input_projection, self.backbone.blocks):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        else:
            self.backbone.eval()

    def train(self, mode: bool = True) -> FixedRigRelativeComparator:
        super().train(mode)
        if not self.config.fine_tune_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _ordered_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        difference = b - a
        return torch.cat((a, b, difference, difference.abs(), a * b), dim=-1)

    @staticmethod
    def _symmetric_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat(((a + b) * 0.5, (b - a).abs(), a * b), dim=-1)

    def compare_temporal(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
    ) -> RelativePreferenceOutput:
        """Apply the exactly swap-equivariant comparison head to encoded sequences."""
        if encoded_a.shape != encoded_b.shape or encoded_a.ndim != 3:
            raise ValueError("paired temporal encodings must have equal [B,T,H] shapes")
        evidence_ab = self.directional_head(self._ordered_features(encoded_a, encoded_b))
        evidence_ba = self.directional_head(self._ordered_features(encoded_b, encoded_a))
        b_better = (evidence_ab - evidence_ba) * 0.5
        equality = self.equality_head(self._symmetric_features(encoded_a, encoded_b))
        logits = torch.stack((-b_better, b_better, equality), dim=-1)
        return RelativePreferenceOutput(logits, torch.softmax(logits, dim=-1))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.fine_tune_backbone:
            return self.backbone.encode_temporal(features)
        with torch.no_grad():
            return self.backbone.encode_temporal(features)

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> RelativePreferenceOutput:
        return self.compare_temporal(self.encode(features_a), self.encode(features_b))
