"""Stable JSON-friendly schemas for deterministic diagnostic results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticEvent(BaseModel):
    """A localized diagnostic interval using half-open frame bounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    source: str = "deterministic"
    frames: tuple[int, int]
    joints: tuple[str, ...] = ()
    severity: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    suggested_operators: tuple[str, ...] = ()


class MetricResult(BaseModel):
    """One metric's clip, dense, event, unit, confidence, and provenance data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    clip_value: float | None
    frame_values: list[float] | None = None
    joint_frame_values: list[list[float]] | None = None
    events: tuple[DiagnosticEvent, ...] = ()
    units: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeterministicReport(BaseModel):
    """Compact deterministic grading report suitable for JSON serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion_id: str
    model_version: None = None
    rig: str
    fps: float
    num_frames: int
    duration_s: float
    measured: dict[str, float | None]
    metrics: dict[str, MetricResult]
    warnings: tuple[str, ...] = ()
    confidence: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
