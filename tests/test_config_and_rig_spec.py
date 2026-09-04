from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from motionlab.config import AppConfig, CoordinateSystemConfig, load_config
from motionlab.io.rig_spec import load_rig_spec


def test_default_configuration_and_example_rig_load() -> None:
    config = load_config(Path("configs/default.yaml"))
    rig = load_rig_spec(Path("configs/rigs/example_target.yaml"))
    assert config.motion.fps == 60.0
    assert rig.joints["left_foot"] == "LeftFoot"
    assert rig.coordinate_system.unit_scale_to_meters == 1.0
    assert rig.joint_limits["LeftLeg"].representation == "swing_twist"
    assert rig.joint_limits["LeftLeg"].confidence == pytest.approx(0.8)


def test_coordinate_axes_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        CoordinateSystemConfig(up="Y", forward="-Y")


def test_unknown_configuration_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AppConfig.model_validate({"seed": 1, "surprise": True})
