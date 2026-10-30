#!/usr/bin/env python3
"""Calculate the first six Warren-Cowley parameters for equiatomic Li/Fe structures.

At c_Li = c_Fe = 1/2, alpha_s = <sigma_i sigma_j>_s, with sigma=+1
for Li and -1 for Fe. Shell vectors are expressed in units of a/2.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.cif import parse_cif

SHELL_SQUARED_DISTANCES = (2, 4, 6, 8, 10, 12)


def read_structure(path: Path, frame: int = 0):
    """Read an explicit P1 CIF atom list or a frame from extended XYZ."""
    if path.suffix.lower() != ".cif":
        return read(path, index=frame)
    if frame != 0:
        raise ValueError("Use frame 0 for a single-structure CIF.")
    block = next(parse_cif(str(path)))
    operations = block.get(
        "_space_group_symop_operation_xyz",
        block.get("_symmetry_equiv_pos_as_xyz", ["x,y,z"]),
    )
    if isinstance(operations, str):
        operations = [operations]
    number = block.get(
        "_space_group_it_number", block.get("_symmetry_int_tables_number", 1)
    )
    if int(number) != 1 or any(op.replace(" ", "") != "x,y,z" for op in operations):
        raise ValueError(
            "Provide an explicit P1 supercell CIF without symmetry expansion."
        )
    if not np.allclose(block.get("_atom_site_occupancy", 1), 1):
        raise ValueError(
            "Cation occupations must be explicit, with unit site occupancy."
        )
    positions = np.column_stack([block[f"_atom_site_fract_{axis}"] for axis in "xyz"])
    return Atoms(
        symbols=block.get_symbols(),
        scaled_positions=positions,
        cell=block.get_cell(),
        pbc=True,
    )


def read_cation_configuration(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read ideal Li/Fe sites on the FCC lattice, in units of a/2."""
    atoms = read_structure(path)
    symbols = np.asarray(atoms.get_chemical_symbols())
    if not set(symbols).issubset({"Li", "Fe", "O"}) or not atoms.pbc.all():
        raise ValueError(f"Expected a periodic Li/Fe/O structure: {path}")
    mask = np.isin(symbols, ["Li", "Fe"])
    n_cations = int(mask.sum())
    nrep = round((n_cations / 4) ** (1 / 3))
    if nrep < 2 or n_cations != 4 * nrep**3:
        raise ValueError(f"Expected 4*nrep^3 cation sites with nrep >= 2: {path}")
    if not np.allclose(atoms.cell.array, np.eye(3) * atoms.cell[0, 0]):
        raise ValueError(f"Expected an axis-aligned cubic cell: {path}")
    period = 2 * nrep
    fractional = atoms.get_scaled_positions()[mask] * period
    if not np.allclose(fractional, np.rint(fractional), atol=1e-5, rtol=0):
        raise ValueError(f"Cations must lie on ideal rocksalt lattice sites: {path}")
    coordinates = np.rint(fractional).astype(int) % period
    if len(np.unique(coordinates, axis=0)) != n_cations or np.any(
        coordinates.sum(axis=1) % 2 != 1
    ):
        raise ValueError(
            f"Cation sites are incomplete or do not match the FCC basis: {path}"
        )
    spins = np.where(symbols[mask] == "Li", 1, -1).astype(np.int8)
    return coordinates, spins, period


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


def calculate_alphas(path: Path) -> list[float]:
    coordinates, spins, period = read_cation_configuration(path)
    if np.count_nonzero(spins == 1) * 2 != len(spins):
        raise ValueError("This spin-correlation expression requires equiatomic Li/Fe.")
    lattice = np.zeros((period, period, period), dtype=np.int8)
    lattice[tuple(coordinates.T)] = spins

    alphas = []
    for squared_distance in SHELL_SQUARED_DISTANCES:
        correlations = []
        for vector in shell_vectors(squared_distance):
            neighbor_coordinates = (coordinates + vector) % period
            correlations.append(np.mean(spins * lattice[tuple(neighbor_coordinates.T)]))
        alphas.append(float(np.mean(correlations)))
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structures",
        nargs="+",
        type=Path,
        help="Ideal Li/Fe CIF or extended-XYZ files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/warren_cowley.csv")
    )
    args = parser.parse_args()
    rows = []
    for path in args.structures:
        alphas = calculate_alphas(path)
        rows.append(
            {
                "source_file": str(path),
                **{f"alpha_{i}": alpha for i, alpha in enumerate(alphas, 1)},
                "1+3alpha_1": 1 + 3 * alphas[0],
                "1+4alpha_1+alpha_2": 1 + 4 * alphas[0] + alphas[1],
                "1+3alpha_2+3alpha_4+alpha_6": 1
                + 3 * alphas[1]
                + 3 * alphas[3]
                + alphas[5],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
