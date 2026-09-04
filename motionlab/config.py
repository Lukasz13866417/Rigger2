"""Validated application configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CoordinateSystemConfig(BaseModel):
    """Coordinate metadata supplied by an adapter or application config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    handedness: Literal["right", "left"] = "right"
    up: Literal["X", "Y", "Z"] = "Y"
    forward: str = "Z"
    unit_scale_to_meters: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def axes_are_distinct(self) -> CoordinateSystemConfig:
        """Reject coordinate descriptions with parallel up and forward axes."""
        normalized_forward = self.forward.lstrip("+-").upper()
        if normalized_forward not in {"X", "Y", "Z"}:
            raise ValueError("forward must be one of X, Y, Z, +X, +Y, +Z, -X, -Y, -Z")
        if normalized_forward == self.up:
            raise ValueError("up and forward axes must be distinct")
        return self


class MotionConfig(BaseModel):
    """Core motion processing defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fps: float = Field(default=60.0, gt=0.0)
    coordinate_system: CoordinateSystemConfig = CoordinateSystemConfig()


class LoggingConfig(BaseModel):
    """Structured logging settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_output: bool = Field(default=False, alias="json")


class PathsConfig(BaseModel):
    """Optional application paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace: Path | None = None


class AppConfig(BaseModel):
    """Top-level MotionLab configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 1234
    motion: MotionConfig = MotionConfig()
    logging: LoggingConfig = LoggingConfig()
    paths: PathsConfig = PathsConfig()


def load_config(path: Path) -> AppConfig:
    """Load and validate a YAML application configuration file."""
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return AppConfig.model_validate(raw)
