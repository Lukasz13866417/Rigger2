"""Write the deterministic synthetic walk used by local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from motionlab.io.npz import save_motion_npz
from motionlab.testing.synthetic import make_synthetic_walk


def main() -> None:
    """Parse arguments and write a synthetic MotionLab NPZ."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=60.0)
    arguments = parser.parse_args()
    save_motion_npz(
        arguments.output,
        make_synthetic_walk(num_frames=arguments.frames, fps=arguments.fps),
    )


if __name__ == "__main__":
    main()
