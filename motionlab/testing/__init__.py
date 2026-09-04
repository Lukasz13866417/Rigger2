"""Deterministic fixtures used by tests and smoke commands."""

from motionlab.testing.synthetic import (
    make_synthetic_contact_walk,
    make_synthetic_humanoid,
    make_synthetic_walk,
)

__all__ = ["make_synthetic_contact_walk", "make_synthetic_humanoid", "make_synthetic_walk"]
