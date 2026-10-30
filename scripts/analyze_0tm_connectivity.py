#!/usr/bin/env python3
"""Calculate tetrahedral populations and local 0-TM connectivity in Li/Ti rocksalt.

A tetrahedral site is 0-TM when all four neighbouring cations are Li.
Connectivity is the number of 0-TM sites among its six edge-sharing neighbours.
This local degree does not determine network percolation or tortuosity.
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.cif import parse_cif

CATION_BASIS = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16)


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


def cation_and_tetrahedron_tables(nrep: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the cation lookup and four cations around each tetrahedron."""
    period = 2 * nrep
    lookup = np.full((period, period, period), -1, dtype=np.int32)
    site = 0
    for ix in range(nrep):
        for iy in range(nrep):
            for iz in range(nrep):
                positions = CATION_BASIS + 2 * np.array([ix, iy, iz])
                lookup[positions[:, 0], positions[:, 1], positions[:, 2]] = np.arange(
                    site, site + 4, dtype=np.int32
                )
                site += 4

    grid = np.indices((period, period, period), dtype=np.int16).reshape(3, -1).T
    corner_offsets = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.int16,
    )
    corners = (grid[:, None, :] + corner_offsets[None, :, :]) % period
    corner_sites = lookup[corners[..., 0], corners[..., 1], corners[..., 2]]
    tetra_to_cation = np.sort(corner_sites, axis=1)[:, 4:]
    if np.any(tetra_to_cation < 0) or np.any(np.sum(corner_sites >= 0, axis=1) != 4):
        raise RuntimeError("Failed to construct four cation neighbors per tetrahedron")
    return lookup, tetra_to_cation


def zero_tm_and_degree(
    is_ti: np.ndarray, tetra_to_cation: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 0-TM mask and number of neighboring 0-TM sites."""
    zero_tm = ~np.any(is_ti[tetra_to_cation], axis=1)
    cube = zero_tm.reshape(period, period, period)
    degree_cube = np.zeros_like(cube, dtype=np.int8)
    for axis in range(3):
        degree_cube += np.roll(cube, 1, axis=axis)
        degree_cube += np.roll(cube, -1, axis=axis)
    return zero_tm, degree_cube.ravel()[zero_tm]


def tm_populations(
    is_ti: np.ndarray, tetra_to_cation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tm_per_tetra = np.count_nonzero(is_ti[tetra_to_cation], axis=1)
    counts = np.bincount(tm_per_tetra, minlength=5)
    return counts, counts / len(tm_per_tetra)


def read_occupations(path: Path, frame: int) -> tuple[np.ndarray, int]:
    atoms = read_structure(path, frame)
    symbols = np.asarray(atoms.get_chemical_symbols())
    if not set(symbols).issubset({"Li", "Ti", "O"}) or not atoms.pbc.all():
        raise ValueError("Expected a periodic Li/Ti/O structure.")
    mask = np.isin(symbols, ["Li", "Ti"])
    n_cations = int(mask.sum())
    nrep = round((n_cations / 4) ** (1 / 3))
    if nrep < 1 or n_cations != 4 * nrep**3:
        raise ValueError("Expected 4*nrep^3 cation sites.")
    if not np.allclose(atoms.cell.array, np.eye(3) * atoms.cell[0, 0]):
        raise ValueError("Expected an axis-aligned cubic cell.")
    period = 2 * nrep
    positions = atoms.get_scaled_positions()[mask] * period
    # Assign small displacements to their parent ideal lattice sites.
    if np.any(np.abs(positions - np.rint(positions)) > 0.25):
        raise ValueError("Cations cannot be assigned unambiguously to ideal sites.")
    coordinates = np.rint(positions).astype(int) % period
    lookup, _ = cation_and_tetrahedron_tables(nrep)
    sites = lookup[tuple(coordinates.T)]
    if np.any(sites < 0) or len(np.unique(sites)) != n_cations:
        raise ValueError("Cation sites are incomplete or do not match the FCC basis.")
    is_ti = np.empty(n_cations, dtype=bool)
    is_ti[sites] = symbols[mask] == "Ti"
    return is_ti, nrep


def degree_distribution(zero_tm: np.ndarray, degree: np.ndarray) -> np.ndarray:
    if not zero_tm.any():
        return np.full(7, np.nan)
    return np.bincount(degree, minlength=7) / zero_tm.sum()


def write_csv(path: Path, columns: list[str], rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structure",
        type=Path,
        help="Li/Ti CIF or extended-XYZ file; gzip is supported.",
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Frame index in a multi-frame trajectory."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/0tm"))
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.random_samples < 1:
        parser.error("--random-samples must be at least 1.")
    is_ti, nrep = read_occupations(args.structure, args.frame)
    n_cations = len(is_ti)
    n_ti = int(is_ti.sum())
    n_li = n_cations - n_ti
    _, tetra_to_cation = cation_and_tetrahedron_tables(nrep)
    _, model_populations = tm_populations(is_ti, tetra_to_cation)
    model_zero, model_degree = zero_tm_and_degree(is_ti, tetra_to_cation, 2 * nrep)
    model_distribution = degree_distribution(model_zero, model_degree)
    random_populations = [
        comb(n_ti, k) * comb(n_li, 4 - k) / comb(n_cations, 4) for k in range(5)
    ]
    rng = np.random.default_rng(args.seed)
    random_distributions = []
    random_degrees = []
    for _ in range(args.random_samples):
        zero_tm, degree = zero_tm_and_degree(
            rng.permutation(is_ti), tetra_to_cation, 2 * nrep
        )
        random_distributions.append(degree_distribution(zero_tm, degree))
        random_degrees.append(degree.mean() if degree.size else float("nan"))
    random_mean = np.mean(random_distributions, axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "tm_populations.csv",
        ["tm_neighbors", "model_fraction", "random_expected_fraction"],
        ([k, model_populations[k], random_populations[k]] for k in range(5)),
    )
    write_csv(
        args.output_dir / "connectivity.csv",
        [
            "zero_tm_neighbors",
            "model_fraction_of_zero_tm_sites",
            "random_mean_fraction",
        ],
        ([k, model_distribution[k], random_mean[k]] for k in range(7)),
    )
    write_csv(
        args.output_dir / "summary.csv",
        ["parameter", "value"],
        [
            ["source_file", str(args.structure)],
            ["frame", args.frame],
            ["n_cations", n_cations],
            ["n_ti", n_ti],
            ["n_li", n_li],
            ["n_tetrahedral_sites", len(tetra_to_cation)],
            ["n_zero_tm_sites", int(model_zero.sum())],
            [
                "model_mean_zero_tm_degree",
                model_degree.mean() if model_degree.size else float("nan"),
            ],
            ["random_mean_zero_tm_degree", np.mean(random_degrees)],
            ["random_samples", args.random_samples],
            ["random_seed", args.seed],
        ],
    )
    print(f"Wrote population, connectivity, and summary CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
