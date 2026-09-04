"""Gait event extraction and confidence-aware continuous phase estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from motionlab.motion._validation import frozen_array
from motionlab.processing.contacts import ContactResult


@dataclass(frozen=True)
class GaitEvent:
    """A discrete gait event at one sampled frame."""

    kind: Literal["heel_strike", "toe_off"]
    side: Literal["left", "right"]
    frame: int
    confidence: float


@dataclass(frozen=True)
class PhaseResult:
    """Continuous phase where left strike is 0 and right strike is pi modulo 2pi."""

    phase_rad: NDArray[np.float32]
    phase_sincos: NDArray[np.float32]
    valid: NDArray[np.bool_]
    events: tuple[GaitEvent, ...]
    confidence: float
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        phase = frozen_array(self.phase_rad, dtype=np.dtype(np.float32), name="phase_rad")
        sincos = frozen_array(
            self.phase_sincos,
            dtype=np.dtype(np.float32),
            name="phase_sincos",
        )
        valid = np.ascontiguousarray(self.valid, dtype=np.bool_)
        if phase.ndim != 1:
            raise ValueError("phase_rad must be one-dimensional")
        if sincos.shape != (phase.shape[0], 2) or valid.shape != phase.shape:
            raise ValueError("phase arrays must have shapes [T], [T,2], and [T]")
        valid.setflags(write=False)
        object.__setattr__(self, "phase_rad", phase)
        object.__setattr__(self, "phase_sincos", sincos)
        object.__setattr__(self, "valid", valid)


def _rising_edges(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    padded = np.pad(mask.astype(np.int8), (1, 0))
    return np.flatnonzero(np.diff(padded) == 1)


def _falling_edges(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    padded = np.pad(mask.astype(np.int8), (0, 1))
    return np.flatnonzero(np.diff(padded) == -1) + 1


def extract_gait_events(contacts: ContactResult) -> tuple[GaitEvent, ...]:
    """Extract heel strikes and whole-foot toe-offs from cleaned marker contacts."""
    marker_index = {name: index for index, name in enumerate(contacts.marker_names)}
    required = {"left_heel", "left_toe", "right_heel", "right_toe"}
    if not required.issubset(marker_index):
        raise ValueError(f"gait event extraction requires markers {sorted(required)}")

    events: list[GaitEvent] = []
    for side in ("left", "right"):
        heel_index = marker_index[f"{side}_heel"]
        toe_index = marker_index[f"{side}_toe"]
        heel_contact = contacts.hard[:, heel_index]
        foot_contact = heel_contact | contacts.hard[:, toe_index]
        for frame in _rising_edges(heel_contact):
            events.append(
                GaitEvent(
                    kind="heel_strike",
                    side=side,
                    frame=int(frame),
                    confidence=float(contacts.confidence[frame, heel_index]),
                )
            )
        for frame in _falling_edges(foot_contact):
            evidence_frame = min(int(frame), contacts.hard.shape[0] - 1)
            events.append(
                GaitEvent(
                    kind="toe_off",
                    side=side,
                    frame=int(frame),
                    confidence=float(contacts.confidence[evidence_frame, toe_index]),
                )
            )
    return tuple(sorted(events, key=lambda event: (event.frame, event.kind, event.side)))


def estimate_gait_phase(contacts: ContactResult) -> PhaseResult:
    """Interpolate monotonic phase from alternating left and right heel strikes."""
    events = extract_gait_events(contacts)
    raw_strikes = [event for event in events if event.kind == "heel_strike"]
    frame_count = contacts.hard.shape[0]
    warnings: list[str] = []
    strikes: list[GaitEvent] = []
    for event in raw_strikes:
        same_frame = [candidate for candidate in raw_strikes if candidate.frame == event.frame]
        if len({candidate.side for candidate in same_frame}) > 1:
            if (
                not warnings
                or warnings[-1] != f"ambiguous bilateral heel strike at frame {event.frame}"
            ):
                warnings.append(f"ambiguous bilateral heel strike at frame {event.frame}")
            continue
        strikes.append(event)
    if len(strikes) < 2:
        warnings.append("fewer than two heel strikes; continuous gait phase is unavailable")
        unavailable_phase = np.zeros(frame_count, dtype=np.float32)
        return PhaseResult(
            phase_rad=unavailable_phase,
            phase_sincos=np.stack((np.sin(unavailable_phase), np.cos(unavailable_phase)), axis=-1),
            valid=np.zeros(frame_count, dtype=np.bool_),
            events=events,
            confidence=0.0,
            warnings=tuple(warnings),
        )

    anchor_frame = np.asarray([strike.frame for strike in strikes], dtype=np.float64)
    anchor_phase = np.empty(len(strikes), dtype=np.float64)
    anchor_phase[0] = 0.0 if strikes[0].side == "left" else np.pi
    alternation_errors = 0
    for index in range(1, len(strikes)):
        if strikes[index].side == strikes[index - 1].side:
            anchor_phase[index] = anchor_phase[index - 1] + 2.0 * np.pi
            alternation_errors += 1
        else:
            anchor_phase[index] = anchor_phase[index - 1] + np.pi
    if alternation_errors:
        warnings.append(f"{alternation_errors} repeated-side heel-strike transitions")

    frame = np.arange(frame_count, dtype=np.float64)
    phase: NDArray[np.float64] = np.asarray(
        np.interp(frame, anchor_frame, anchor_phase), dtype=np.float64
    )
    first_slope = (anchor_phase[1] - anchor_phase[0]) / (anchor_frame[1] - anchor_frame[0])
    last_slope = (anchor_phase[-1] - anchor_phase[-2]) / (anchor_frame[-1] - anchor_frame[-2])
    before = frame < anchor_frame[0]
    after = frame > anchor_frame[-1]
    phase[before] = anchor_phase[0] + (frame[before] - anchor_frame[0]) * first_slope
    phase[after] = anchor_phase[-1] + (frame[after] - anchor_frame[-1]) * last_slope

    event_confidence = float(np.mean([strike.confidence for strike in strikes]))
    structure_confidence = min(1.0, (len(strikes) - 1) / 3.0)
    if alternation_errors:
        structure_confidence *= max(0.0, 1.0 - alternation_errors / (len(strikes) - 1))
    confidence = event_confidence * structure_confidence * contacts.ground_confidence
    phase_float = phase.astype(np.float32)
    return PhaseResult(
        phase_rad=phase_float,
        phase_sincos=np.stack((np.sin(phase_float), np.cos(phase_float)), axis=-1).astype(
            np.float32
        ),
        valid=np.ones(frame_count, dtype=np.bool_),
        events=events,
        confidence=confidence,
        warnings=tuple(warnings),
    )
