"""Deterministic fixed anatomical vocabulary for humanoid metadata."""

from __future__ import annotations

from enum import IntEnum


class JointSide(IntEnum):
    UNKNOWN = 0
    CENTER = 1
    LEFT = 2
    RIGHT = 3


class FunctionalPart(IntEnum):
    UNKNOWN = 0
    ROOT_PELVIS = 1
    TRUNK_SPINE = 2
    HEAD_NECK = 3
    LEFT_ARM = 4
    RIGHT_ARM = 5
    LEFT_LEG = 6
    RIGHT_LEG = 7
    OTHER_ACCESSORY = 8


FUNCTIONAL_PART_NAMES = tuple(part.name.lower() for part in FunctionalPart)

_ROOT_ROLES = {"root", "pelvis"}
_TRUNK_ROLES = {"spine", "spine_1", "spine_2", "spine_3", "chest", "upper_chest"}
_HEAD_ROLES = {"neck", "head"}


def side_for_role(role: str) -> JointSide:
    if role.startswith("left_"):
        return JointSide.LEFT
    if role.startswith("right_"):
        return JointSide.RIGHT
    if role in _ROOT_ROLES | _TRUNK_ROLES | _HEAD_ROLES:
        return JointSide.CENTER
    return JointSide.UNKNOWN


def functional_part_for_role(role: str) -> FunctionalPart:
    if role in _ROOT_ROLES:
        return FunctionalPart.ROOT_PELVIS
    if role in _TRUNK_ROLES:
        return FunctionalPart.TRUNK_SPINE
    if role in _HEAD_ROLES:
        return FunctionalPart.HEAD_NECK
    if role.startswith("left_"):
        if _is_arm_role(role):
            return FunctionalPart.LEFT_ARM
        if _is_leg_role(role):
            return FunctionalPart.LEFT_LEG
        return FunctionalPart.OTHER_ACCESSORY
    if role.startswith("right_"):
        if _is_arm_role(role):
            return FunctionalPart.RIGHT_ARM
        if _is_leg_role(role):
            return FunctionalPart.RIGHT_LEG
        return FunctionalPart.OTHER_ACCESSORY
    if role == "UNKNOWN":
        return FunctionalPart.OTHER_ACCESSORY
    return FunctionalPart.OTHER_ACCESSORY


def _is_arm_role(role: str) -> bool:
    return any(token in role for token in ("clavicle", "shoulder", "arm", "elbow", "wrist", "hand"))


def _is_leg_role(role: str) -> bool:
    return any(token in role for token in ("hip", "thigh", "knee", "leg", "ankle", "foot", "toe"))
