"""Small fixed-rig flattened temporal-convolution critic baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn

from motionlab.corruptions.base import CORRUPTION_DEFECT_NAMES, CORRUPTION_PART_NAMES

FIRST_CRITIC_DEFECTS = (
    "foot_slide",
    "ground_penetration",
    "floating_contact",
    "loop_seam",
    "joint_pop",
    "joint_jitter",
    "speed_inconsistency",
)


class FixedRigTCNConfig(BaseModel):
    """Frozen architecture declaration for the intentionally simple baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_joints: int = Field(ge=1)
    joint_features: int = Field(default=15, ge=1)
    global_features: int = Field(default=21, ge=1)
    hidden_channels: int = Field(default=128, ge=8, le=512)
    kernel_size: int = Field(default=3)
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization: Literal["group_norm", "layer_norm"] = "group_norm"
    defect_names: tuple[str, ...] = FIRST_CRITIC_DEFECTS

    @field_validator("kernel_size")
    @classmethod
    def kernel_is_odd(cls, value: int) -> int:
        if value not in {3, 5}:
            raise ValueError("kernel_size must be 3 or 5")
        return value

    @field_validator("dilations")
    @classmethod
    def dilations_are_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not 4 <= len(value) <= 6 or any(item < 1 for item in value):
            raise ValueError("dilations must contain four to six positive values")
        return value

    @field_validator("defect_names")
    @classmethod
    def defects_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("defect_names must be nonempty and unique")
        unknown = set(value).difference(CORRUPTION_DEFECT_NAMES)
        if unknown:
            raise ValueError(f"unknown defect names: {sorted(unknown)}")
        return value

    @property
    def input_channels(self) -> int:
        return self.num_joints * self.joint_features + self.global_features


class PerFrameLayerNorm(nn.LayerNorm):
    """Layer-normalize channels independently at every frame of a [B,C,T] tensor."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = super().forward(value.transpose(1, 2))
        return normalized.transpose(1, 2)


class ResidualTemporalBlock(nn.Module):
    """A same-length noncausal residual Conv1D block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        normalization: Literal["group_norm", "layer_norm"] = "group_norm",
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        normalization_layer: type[nn.GroupNorm] | type[PerFrameLayerNorm]
        normalization_arguments: tuple[int, ...]
        if normalization == "group_norm":
            normalization_layer = nn.GroupNorm
            normalization_arguments = (1, channels)
        else:
            normalization_layer = PerFrameLayerNorm
            normalization_arguments = (channels,)
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            normalization_layer(*normalization_arguments),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            normalization_layer(*normalization_arguments),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        transformed: torch.Tensor = self.network(value)
        return value + transformed


@dataclass(frozen=True)
class CriticOutput:
    """Future-compatible baseline outputs without per-joint causal-blame claims."""

    frame_logits: torch.Tensor
    frame_severity: torch.Tensor
    part_logits: torch.Tensor
    clip_logits: torch.Tensor
    clip_severity: torch.Tensor
    family_ranking_score: torch.Tensor
    representation: torch.Tensor


class FixedRigFlatTCN(nn.Module):
    """Flatten fixed joints per frame, then apply a compact dilated TCN."""

    def __init__(self, config: FixedRigTCNConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_channels
        defects = len(config.defect_names)
        parts = len(CORRUPTION_PART_NAMES)
        self.input_projection = nn.Conv1d(config.input_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(
                hidden,
                config.kernel_size,
                dilation,
                config.dropout,
                config.normalization,
            )
            for dilation in config.dilations
        )
        self.frame_logits_head = nn.Conv1d(hidden, defects, kernel_size=1)
        self.frame_severity_head = nn.Conv1d(hidden, defects, kernel_size=1)
        self.part_logits_head = nn.Conv1d(hidden, parts * defects, kernel_size=1)
        self.clip_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.clip_logits_head = nn.Linear(hidden, defects)
        self.clip_severity_head = nn.Linear(hidden, defects)
        # Synthetic ordering is only valid within each defect family.  Keeping one output per
        # family prevents this experiment from quietly learning an unsupported global notion of
        # motion quality.
        self.family_ranking_head = nn.Linear(hidden, defects)

    def encode_temporal(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Return shared per-frame features as ``[batch, time, hidden]``."""
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape [B,T,C]")
        if frame_features.shape[-1] != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels, "
                f"got {frame_features.shape[-1]}"
            )
        hidden: torch.Tensor = self.input_projection(frame_features.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.transpose(1, 2)

    def forward(self, frame_features: torch.Tensor) -> CriticOutput:
        temporal = self.encode_temporal(frame_features)
        hidden = temporal.transpose(1, 2)
        frame_logits = self.frame_logits_head(hidden).transpose(1, 2)
        frame_severity = torch.nn.functional.softplus(
            self.frame_severity_head(hidden).transpose(1, 2)
        )
        batch, _, frames = hidden.shape
        part_logits = self.part_logits_head(hidden).view(
            batch,
            len(CORRUPTION_PART_NAMES),
            len(self.config.defect_names),
            frames,
        )
        part_logits = part_logits.permute(0, 3, 1, 2)
        pooled = torch.cat((hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=-1)
        representation = self.clip_projection(pooled)
        return CriticOutput(
            frame_logits=frame_logits,
            frame_severity=frame_severity,
            part_logits=part_logits,
            clip_logits=self.clip_logits_head(representation),
            clip_severity=torch.nn.functional.softplus(self.clip_severity_head(representation)),
            family_ranking_score=torch.nn.functional.softplus(
                self.family_ranking_head(representation)
            ),
            representation=representation,
        )
