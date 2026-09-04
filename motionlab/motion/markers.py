"""Marker and joint-limit schemas shared by rigs and serialized skeletons."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MarkerSpec(BaseModel):
    """A virtual marker attached to a joint at a local metric offset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    joint: str | int
    local_offset_m: tuple[float, float, float]

    @field_validator("local_offset_m")
    @classmethod
    def offset_is_finite(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        """Reject non-finite marker coordinates."""
        if not all(math.isfinite(component) for component in value):
            raise ValueError("marker offset must be finite")
        return value


class JointLimitSpec(BaseModel):
    """Optional swing/twist or Euler angular bounds supplied by a target rig."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    representation: Literal["swing_twist", "euler_xyz"]
    swing_x_deg: tuple[float, float] | None = None
    swing_z_deg: tuple[float, float] | None = None
    twist_deg: tuple[float, float] | None = None
    x_deg: tuple[float, float] | None = None
    y_deg: tuple[float, float] | None = None
    z_deg: tuple[float, float] | None = None
    valid: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str | None = None

    @model_validator(mode="after")
    def bounds_are_ordered_and_finite(self) -> JointLimitSpec:
        """Validate every supplied inclusive angular interval."""
        for field_name in (
            "swing_x_deg",
            "swing_z_deg",
            "twist_deg",
            "x_deg",
            "y_deg",
            "z_deg",
        ):
            bounds = getattr(self, field_name)
            if bounds is not None:
                if not all(math.isfinite(value) for value in bounds):
                    raise ValueError(f"{field_name} bounds must be finite")
                if bounds[0] > bounds[1]:
                    raise ValueError(f"{field_name} lower bound exceeds upper bound")
        if self.source is not None and not self.source.strip():
            raise ValueError("joint-limit source must be nonempty when supplied")
        return self
