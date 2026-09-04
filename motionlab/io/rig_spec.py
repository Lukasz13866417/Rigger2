"""Validated YAML schema for semantic target-rig configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from motionlab.config import CoordinateSystemConfig
from motionlab.motion.markers import JointLimitSpec, MarkerSpec


class RetargetSpec(BaseModel):
    """Optional semantic aliases and source-to-target local basis corrections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_role_map: Mapping[str, str] = {}
    basis_correction_quat_wxyz: Mapping[str, tuple[float, float, float, float]] = {}


class RigSpec(BaseModel):
    """Human-authored rig semantics kept separate from concrete skeleton arrays."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    coordinate_system: CoordinateSystemConfig
    joints: Mapping[str, str]
    markers: Mapping[str, MarkerSpec]
    retarget: RetargetSpec = RetargetSpec()
    joint_limits: Mapping[str, JointLimitSpec] = {}

    @field_validator("name")
    @classmethod
    def name_is_nonempty(cls, value: str) -> str:
        """Require a stable, nonempty rig name."""
        if not value.strip():
            raise ValueError("rig name must be nonempty")
        return value

    @field_validator("joints")
    @classmethod
    def joint_mapping_is_nonempty(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        """Require semantic and concrete joint names to be nonempty."""
        if not value:
            raise ValueError("rig must define at least one semantic joint mapping")
        if any(not role.strip() or not name.strip() for role, name in value.items()):
            raise ValueError("rig joint roles and names must be nonempty")
        return value


def load_rig_spec(path: Path) -> RigSpec:
    """Load and validate a target-rig YAML specification."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"rig specification root must be a mapping: {path}")
    return RigSpec.model_validate(raw)
