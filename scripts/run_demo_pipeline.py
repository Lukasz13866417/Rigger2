"""Run MotionLab's deterministic clean/corrupt/repair vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motionlab.demo import run_deterministic_demo


def main() -> None:
    """Parse demo arguments and print the generated artifact summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--no-previews", action="store_true")
    arguments = parser.parse_args()
    summary = run_deterministic_demo(
        arguments.workspace,
        render_previews=not arguments.no_previews,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
