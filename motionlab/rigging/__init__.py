"""Rig semantics and validation helpers.

Graph data contracts live in :mod:`motionlab.rigging.graphs` and intentionally are
not imported eagerly here: clip metadata depends on semantic roles during startup.
"""

from motionlab.rigging.semantic_roles import FunctionalPart, JointSide

__all__ = ["FunctionalPart", "JointSide"]
