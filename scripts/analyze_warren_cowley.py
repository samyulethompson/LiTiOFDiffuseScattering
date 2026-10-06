#!/usr/bin/env python3
"""Calculate shell-resolved Warren-Cowley parameters for a LiFeO2 sweep."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


RESULT_RE = re.compile(
    r"results_lifeo2_(unnorm|norm)_total_([^_]+)_ratio_([^_]+)_temp_[^/]+$"
)
SHELL_SQUARED_DISTANCES = (2, 4, 6, 8, 10, 12)


def decode_number(token: str) -> float:
    return float(token.replace("p", "."))


def encode_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def shell_vectors(squared_distance: int) -> np.ndarray:
    limit = math.isqrt(squared_distance)
    vectors = [
        (i, j, k)
        for i in range(-limit, limit + 1)
        for j in range(-limit, limit + 1)
        for k in range(-limit, limit + 1)
        if i * i + j * j + k * k == squared_distance
    ]
    return np.asarray(vectors, dtype=int)


def read_cation_configuration(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    fractional_coordinates = []
    spins = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0] not in {"Li", "Fe"}:
                continue
            fractional_coordinates.append(tuple(map(float, fields[3:6])))
            spins.append(1 if fields[0] == "Li" else -1)

    n_cations = len(fractional_coordinates)
    nrep = round((n_cations / 4.0) ** (1.0 / 3.0))
    if n_cations != 4 * nrep**3:
        raise ValueError(f"Expected 4*nrep^3 cations in {path}, found {n_cations}")

    period = 2 * nrep
    coordinates = np.rint(
        np.asarray(fractional_coordinates, dtype=float) * period
    ).astype(int) % period
    if len(np.unique(coordinates, axis=0)) != n_cations:
        raise ValueError(f"Duplicate cation lattice coordinates found in {path}")
    return coordinates, np.asarray(spins, dtype=np.int8), period


def calculate_alphas(path: Path) -> list[float]:
    coordinates, spins, period = read_cation_configuration(path)
    lattice = np.zeros((period, period, period), dtype=np.int8)
    lattice[tuple(coordinates.T)] = spins

    alphas = []
    for squared_distance in SHELL_SQUARED_DISTANCES:
        correlations = []
        for vector in shell_vectors(squared_distance):
            neighbor_coordinates = (coordinates + vector) % period
            correlations.append(
                np.mean(spins * lattice[tuple(neighbor_coordinates.T)])
            )
        alphas.append(float(np.mean(correlations)))
    return alphas


def result_parameters(path: Path) -> tuple[str, float, float]:
    result_dir = path.parents[1].name
    match = RESULT_RE.match(result_dir)
    if not match:
        raise ValueError(f"Could not parse sweep parameters from {result_dir}")
    return match.group(1), decode_number(match.group(2)), decode_number(match.group(3))


def find_inputs(root: Path, total: float, snapshot: int) -> list[Path]:
    pattern = (
        "*/results_lifeo2_*_total_*_ratio_*_temp_*/scatty_structures/"
        f"rocks_lifeo2_pair_mc_atoms_{snapshot:02d}.cif"
    )
    inputs = [
        path
        for path in root.glob(pattern)
        if math.isclose(result_parameters(path)[1], total)
    ]
    inputs.sort(key=lambda path: result_parameters(path)[2])
    if len(inputs) != 11:
        raise RuntimeError(
            f"Expected 11 structures for total strength {total:g} eV, found {len(inputs)}"
        )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root directory containing the sweep runs")
    parser.add_argument(
        "--total",
        type=float,
        default=0.2,
        help="total J1+J2 interaction strength in eV (default: 1)",
    )
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.root / (
        f"warren_cowley_{encode_number(args.total)}eV.csv"
    )
    rows = []
    for path in find_inputs(args.root, args.total, args.snapshot):
        alphas = calculate_alphas(path)
        row = {
            "series": result_parameters(path)[0],
            "mixing_ratio": result_parameters(path)[2],
            **{f"alpha_{index}": alpha for index, alpha in enumerate(alphas, 1)},
            "1+3alpha_1": 1 + 3 * alphas[0],
            "1+4alpha_1+alpha_2": 1 + 4 * alphas[0] + alphas[1],
            "1+3alpha_2+3alpha_4+alpha_6": (
                1 + 3 * alphas[1] + 3 * alphas[3] + alphas[5]
            ),
            "source": str(path.relative_to(args.root)),
        }
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
