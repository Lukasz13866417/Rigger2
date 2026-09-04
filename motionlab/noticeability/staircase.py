"""Deterministic interleaved staircase decisions, derived only from direct responses."""

from __future__ import annotations

import random
from typing import Any


def staircase_step(level: int, response: str, levels: int) -> int:
    if response not in {"NO", "MAYBE", "YES"} or not 0 <= level < levels:
        raise ValueError("invalid staircase state")
    return min(levels - 1, max(0, level + {"NO": 1, "MAYBE": 0, "YES": -1}[response]))


def select_interleaved(
    tracks: dict[str, list[str]],
    history: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[str, int, str]:
    """Balance independent family/source tracks and never immediately repeat a track."""
    if len(tracks) < 2:
        raise ValueError("interleaved staircases require at least two tracks")
    counts = {key: sum(r["track"] == key for r in history) for key in tracks}
    eligible = [k for k in tracks if not history or k != history[-1]["track"]]
    random.Random(seed + len(history)).shuffle(eligible)
    track = min(eligible, key=lambda key: counts[key])
    level = len(tracks[track]) // 2
    for row in history:
        if row["track"] == track:
            level = staircase_step(level, row["spontaneous_notice"], len(tracks[track]))
    return track, level, tracks[track][level]
