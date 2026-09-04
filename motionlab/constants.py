"""Named numerical tolerances and canonical coordinate constants."""

from typing import Final

FLOAT_EPS: Final[float] = 1.0e-8
QUATERNION_NORM_TOLERANCE: Final[float] = 1.0e-5
ROTATION_MATRIX_TOLERANCE: Final[float] = 2.0e-5
FK_POSITION_TOLERANCE_M: Final[float] = 1.0e-5
TIME_EPS_S: Final[float] = 1.0e-9

CANONICAL_HANDEDNESS: Final[str] = "right"
CANONICAL_UP_AXIS: Final[str] = "Y"
CANONICAL_FORWARD_AXIS: Final[str] = "Z"
NPZ_FORMAT_VERSION: Final[str] = "motionlab.motion_clip.v1"
