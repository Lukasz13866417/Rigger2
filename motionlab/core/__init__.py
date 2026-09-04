"""Forward-compatible contracts layered over the fixed-rig representation."""

from motionlab.core.clip_metadata import ClipMetadataAdapter, adapt_motion_clip
from motionlab.core.coordinate_contract import CoordinateContract
from motionlab.core.provenance import Provenance, assign_source_lineage

__all__ = [
    "ClipMetadataAdapter",
    "CoordinateContract",
    "Provenance",
    "adapt_motion_clip",
    "assign_source_lineage",
]
