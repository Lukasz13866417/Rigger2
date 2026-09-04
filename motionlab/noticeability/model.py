"""Minimal clip-level noticeability head and explicit production satisficing contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from motionlab.critic.model import FixedRigFlatTCN


class FixedRigNoticeability(nn.Module):
    """Keep the GroupNorm TCN; use the same declared style/goal context as the rater."""

    def __init__(self, backbone: FixedRigFlatTCN, contexts: tuple[str, ...]) -> None:
        super().__init__()
        if (
            backbone.config.normalization != "group_norm"
            or not contexts
            or len(set(contexts)) != len(contexts)
        ):
            raise ValueError("GroupNorm baseline and unique explicit style/goal contexts required")
        self.backbone = backbone
        self.contexts = contexts
        self.noticeability_clip_head = nn.Sequential(
            nn.Linear(backbone.config.hidden_channels + len(contexts), 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(
        self, features: torch.Tensor, context_indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        representation = self.backbone(features).representation
        context = torch.nn.functional.one_hot(context_indices, len(self.contexts)).to(
            representation.dtype
        )
        logits = self.noticeability_clip_head(torch.cat([representation, context], dim=-1))
        probabilities = logits.softmax(dim=-1)
        return {
            "noticeability_clip_logits": logits,
            "noticeability_probabilities": probabilities,
            "p_notice_spontaneous": probabilities[:, 2],
        }


def clip_noticeability_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 3 or labels.shape != logits.shape[:1]:
        raise ValueError("human noticeability is one clip/event target, never replicated per frame")
    return torch.nn.functional.cross_entropy(logits, labels)


def choose_operating_point(
    clean_probabilities: list[float],
    human_yes_probabilities: list[float],
    *,
    split_role: str,
    maximum_false_alarm: float = 0.05,
    minimum_recall: float = 0.8,
) -> dict[str, Any]:
    if split_role != "held_out_validation":
        raise ValueError("choose tau on held-out human validation, not training/test predictions")
    values = np.asarray([*clean_probabilities, *human_yes_probabilities], float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("invalid noticeability probabilities")
    if not 0 < maximum_false_alarm < 1 or not 0 < minimum_recall <= 1:
        raise ValueError("invalid operating-point policy")
    if len(clean_probabilities) < 40 or len(human_yes_probabilities) < 20:
        return {"tau": None, "status": "insufficient_held_out_human_support"}
    clean, yes = np.asarray(clean_probabilities), np.asarray(human_yes_probabilities)
    tradeoffs = [
        {
            "tau": float(t),
            "false_alarm": float((clean > t).mean()),
            "recall": float((yes > t).mean()),
        }
        for t in sorted(set([*values.tolist(), 0.0, 1.0]))
    ]
    permitted = [
        r
        for r in tradeoffs
        if r["false_alarm"] <= maximum_false_alarm and r["recall"] >= minimum_recall
    ]
    if not permitted:
        return {"tau": None, "status": "no_supported_operating_point", "tradeoffs": tradeoffs}
    chosen = max(permitted, key=lambda r: (r["recall"], -r["false_alarm"], -r["tau"]))
    return {
        **chosen,
        "status": "candidate_requires_review",
        "split_role": split_role,
        "tradeoffs": tradeoffs,
        "clean_count": len(clean),
        "human_yes_count": len(yes),
        "maximum_false_alarm": maximum_false_alarm,
    }


def production_noticeability(
    p_notice: float, *, tau: float, deterministic_feasible: bool
) -> dict[str, Any]:
    if not np.isfinite([p_notice, tau]).all() or not 0 <= p_notice <= 1 or not 0 <= tau <= 1:
        raise ValueError("explicit calibrated probabilities and reviewed tau required")
    return {
        "feasible": deterministic_feasible and p_notice <= tau,
        "perceptual_penalty": max(0.0, p_notice - tau),
        "mode": "production_satisficing",
    }


def red_team_objective(p_notice: float) -> float:
    """Separate aggressive probe, not the production objective. Persist human reversals."""
    if not np.isfinite(p_notice) or not 0 <= p_notice <= 1:
        raise ValueError("invalid probability")
    return p_notice
