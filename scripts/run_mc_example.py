#!/usr/bin/env python3
"""Example Monte Carlo structure-generation run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILDER = ROOT / "build_mc_structure.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mc_example_output"),
        help="directory for generated structures (default: ./mc_example_output)",
    )
    parser.add_argument("--nrep", type=int, default=21)
    parser.add_argument("--mc-sweeps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--nearest-interaction",
        type=float,
        default=-0.5,
        help="reduced nearest-neighbor Ti-Ti pair energy",
    )
    parser.add_argument(
        "--second-interaction",
        type=float,
        default=0.5,
        help="reduced second-neighbor Ti-Ti pair energy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--nrep",
            str(args.nrep),
            "--ti-ti-interaction",
            str(args.nearest_interaction),
            "--ti-ti-second-interaction",
            str(args.second_interaction),
            "--mc-beta",
            "1.0",
            "--mc-sweeps",
            str(args.mc_sweeps),
            "--o-displacement",
            "0.100",
            "--unlike-second-repulsion",
            "0.075",
            "--ti-ti-nn-repulsion",
            "0.025",
            "--seed",
            str(args.seed),
            "--stem",
            "rocks_mc_example",
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
