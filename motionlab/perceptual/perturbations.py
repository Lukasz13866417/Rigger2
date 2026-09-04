"""Constraint-screened fixed-rig perturbations for residual perceptual quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from motionlab.math.quaternion import (
    quaternion_difference,
    quaternion_exp,
    quaternion_log,
    quaternion_multiply,
    quaternion_slerp,
)
from motionlab.motion.clip import MotionClip
from motionlab.optimization.jitter_blocks import JitterSmoothingBlock

Preference = Literal["a_better", "b_better", "approximately_equal"]
SupervisionCategory = Literal["CERTAIN", "HUMAN_LABELED", "UNORDERED"]


@dataclass(frozen=True)
class PerturbationSpec:
    """One candidate-construction rule and its deliberately limited supervision claim."""

    mechanism: str
    strength: float
    mechanism_partition: Literal["train", "heldout"]
    supervision_category: SupervisionCategory
    preference: Preference | None
    reason_tags: tuple[str, ...]
    label_basis: str | None


PERTURBATION_SPECS = (
    PerturbationSpec(
        "excessive_rigidification",
        0.55,
        "train",
        "CERTAIN",
        "a_better",
        ("naturalness", "rigidity/smoothing"),
        "strong rule: most authored upper-body variation is removed",
    ),
    PerturbationSpec(
        "excessive_rigidification",
        0.75,
        "train",
        "CERTAIN",
        "a_better",
        ("naturalness", "rigidity/smoothing"),
        "strong rule: most authored upper-body variation is removed",
    ),
    PerturbationSpec(
        "excessive_smoothing",
        2.0,
        "train",
        "CERTAIN",
        "a_better",
        ("naturalness", "rigidity/smoothing"),
        "strong rule: low-pass filtering deliberately removes authored impacts",
    ),
    PerturbationSpec(
        "excessive_smoothing",
        4.0,
        "train",
        "CERTAIN",
        "a_better",
        ("naturalness", "rigidity/smoothing"),
        "strong rule: low-pass filtering deliberately removes authored impacts",
    ),
    PerturbationSpec(
        "upper_body_phase_mismatch",
        5.0,
        "train",
        "UNORDERED",
        None,
        ("coordination",),
        None,
    ),
    PerturbationSpec(
        "upper_body_phase_mismatch",
        9.0,
        "train",
        "UNORDERED",
        None,
        ("coordination",),
        None,
    ),
    PerturbationSpec(
        "phase_aligned_style_inconsistency",
        0.35,
        "train",
        "UNORDERED",
        None,
        ("style",),
        None,
    ),
    PerturbationSpec(
        "weight_transfer_proxy",
        0.004,
        "train",
        "UNORDERED",
        None,
        ("weight transfer",),
        None,
    ),
    PerturbationSpec(
        "equivalent_origin_shift",
        0.12,
        "train",
        "CERTAIN",
        "approximately_equal",
        (),
        "exact world-origin equivalence",
    ),
    PerturbationSpec(
        "torso_counter_rotation_mismatch",
        0.08,
        "heldout",
        "CERTAIN",
        "a_better",
        ("naturalness", "coordination"),
        "strong rule: an added torso counter-rotation conflicts with the authored cycle",
    ),
    PerturbationSpec(
        "torso_counter_rotation_mismatch",
        0.14,
        "heldout",
        "CERTAIN",
        "a_better",
        ("naturalness", "coordination"),
        "strong rule: an added torso counter-rotation conflicts with the authored cycle",
    ),
    PerturbationSpec(
        "asymmetric_limb_timing",
        7.0,
        "heldout",
        "UNORDERED",
        None,
        ("coordination",),
        None,
    ),
)


def _upper_body_joints(clip: MotionClip) -> tuple[int, ...]:
    roles = clip.skeleton.roles
    requested = (
        "spine_1",
        "spine_2",
        "chest",
        "left_shoulder",
        "left_elbow",
        "right_shoulder",
        "right_elbow",
    )
    return tuple(dict.fromkeys(roles[role] for role in requested if role in roles))


def _arm_joints(clip: MotionClip, side: str | None = None) -> tuple[int, ...]:
    roles = clip.skeleton.roles
    sides = ("left", "right") if side is None else (side,)
    return tuple(
        roles[name]
        for current in sides
        for name in (f"{current}_shoulder", f"{current}_elbow", f"{current}_wrist")
        if name in roles
    )


def _edit_envelope(frames: int, margin: int = 12) -> np.ndarray:
    envelope = np.ones(frames, dtype=np.float64)
    blend = min(margin, max(1, frames // 4))
    phase = np.linspace(0.0, 1.0, blend)
    ramp = phase * phase * (3.0 - 2.0 * phase)
    envelope[:blend] = ramp
    envelope[-blend:] = ramp[::-1]
    return envelope


def _blend_joint_targets(
    clip: MotionClip,
    joints: tuple[int, ...],
    target: np.ndarray,
    amount: np.ndarray,
    *,
    mechanism: str,
) -> MotionClip:
    local = clip.local_quat_wxyz.copy()
    local[:, joints] = quaternion_slerp(
        local[:, joints],
        target[:, joints],
        amount[:, None],
    ).astype(np.float32)
    return clip.with_updates(
        local_quat_wxyz=local,
        metadata={
            **dict(clip.metadata),
            "perceptual_perturbation": mechanism,
        },
    )


def _phase_shift(
    clip: MotionClip,
    joints: tuple[int, ...],
    frames: int,
    *,
    mechanism: str,
) -> MotionClip:
    indices = np.clip(np.arange(clip.num_frames) + frames, 0, clip.num_frames - 1)
    target = clip.local_quat_wxyz[indices]
    return _blend_joint_targets(
        clip,
        joints,
        target,
        _edit_envelope(clip.num_frames),
        mechanism=mechanism,
    )


def _rigidify(clip: MotionClip, strength: float) -> MotionClip:
    joints = _upper_body_joints(clip)
    target = clip.local_quat_wxyz.copy()
    for joint in joints:
        reference = clip.local_quat_wxyz[clip.num_frames // 2, joint]
        reference_series = np.broadcast_to(reference, (clip.num_frames, 4))
        tangent = quaternion_log(
            quaternion_difference(reference_series, clip.local_quat_wxyz[:, joint])
        )
        mean = np.mean(tangent, axis=0)
        fixed = quaternion_multiply(reference, quaternion_exp(mean))
        target[:, joint] = fixed
    return _blend_joint_targets(
        clip,
        joints,
        target,
        _edit_envelope(clip.num_frames) * strength,
        mechanism="excessive_rigidification",
    )


def _smooth(clip: MotionClip, cutoff_hz: float) -> MotionClip:
    joints = _upper_body_joints(clip)
    block = JitterSmoothingBlock(joints, 0, clip.num_frames)
    values = block.initial_values
    values[0] = cutoff_hz
    values[1] = 1.0
    values[2:4] = 12.0
    smoothed = block.apply(clip, values)
    return smoothed.with_updates(
        metadata={
            **dict(smoothed.metadata),
            "perceptual_perturbation": "excessive_smoothing",
        }
    )


def _counter_rotation(clip: MotionClip, amplitude_rad: float) -> MotionClip:
    joints = tuple(
        clip.skeleton.roles[role] for role in ("spine_2", "chest") if role in clip.skeleton.roles
    )
    phase = np.linspace(0.0, 4.0 * np.pi, clip.num_frames)
    envelope = _edit_envelope(clip.num_frames)
    tangent = np.zeros((clip.num_frames, clip.num_joints, 3), dtype=np.float64)
    for offset, joint in enumerate(joints):
        tangent[:, joint, 1] = amplitude_rad * np.sin(phase + offset * np.pi) * envelope
    local = quaternion_multiply(clip.local_quat_wxyz, quaternion_exp(tangent))
    return clip.with_updates(
        local_quat_wxyz=np.asarray(local, dtype=np.float32),
        metadata={
            **dict(clip.metadata),
            "perceptual_perturbation": "torso_counter_rotation_mismatch",
        },
    )


def _style_mix(clip: MotionClip, donor: MotionClip, strength: float) -> MotionClip:
    if clip.skeleton.content_hash != donor.skeleton.content_hash:
        raise ValueError("style donor must share the fixed rig")
    joints = _upper_body_joints(clip)
    return _blend_joint_targets(
        clip,
        joints,
        donor.local_quat_wxyz,
        _edit_envelope(clip.num_frames) * strength,
        mechanism="phase_aligned_style_inconsistency",
    )


def apply_perceptual_perturbation(
    clip: MotionClip,
    spec: PerturbationSpec,
    *,
    donor: MotionClip | None = None,
) -> MotionClip:
    """Create one candidate without asserting quality beyond the supplied spec."""
    if spec.mechanism == "excessive_rigidification":
        return _rigidify(clip, spec.strength)
    if spec.mechanism == "excessive_smoothing":
        return _smooth(clip, spec.strength)
    if spec.mechanism == "upper_body_phase_mismatch":
        return _phase_shift(
            clip,
            _arm_joints(clip),
            round(spec.strength),
            mechanism=spec.mechanism,
        )
    if spec.mechanism == "asymmetric_limb_timing":
        return _phase_shift(
            clip,
            _arm_joints(clip, "left"),
            round(spec.strength),
            mechanism=spec.mechanism,
        )
    if spec.mechanism == "torso_counter_rotation_mismatch":
        return _counter_rotation(clip, spec.strength)
    if spec.mechanism == "phase_aligned_style_inconsistency":
        if donor is None:
            raise ValueError("style inconsistency requires a phase-aligned donor")
        return _style_mix(clip, donor, spec.strength)
    if spec.mechanism == "weight_transfer_proxy":
        root = clip.root_translation_m.copy()
        phase = np.linspace(0.0, 4.0 * np.pi, clip.num_frames)
        root[:, 0] += (spec.strength * np.sin(phase) * _edit_envelope(clip.num_frames)).astype(
            np.float32
        )
        return clip.with_updates(
            root_translation_m=root,
            metadata={**dict(clip.metadata), "perceptual_perturbation": spec.mechanism},
        )
    if spec.mechanism == "equivalent_origin_shift":
        root = clip.root_translation_m.copy()
        root[:, 0] += spec.strength
        root[:, 2] -= spec.strength * 0.5
        return clip.with_updates(
            root_translation_m=root,
            metadata={**dict(clip.metadata), "perceptual_perturbation": spec.mechanism},
        )
    raise ValueError(f"unsupported perceptual perturbation {spec.mechanism!r}")
