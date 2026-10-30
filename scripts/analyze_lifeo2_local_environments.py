#!/usr/bin/env python3
"""Count tetrahedral, O-centred octahedral, and cubic Li/Fe environments.

The random reference is the exact hypergeometric distribution at fixed composition:
P(k) = C(N_Fe,k) C(N_Li,m-k) / C(N_cations,m), for an m-site motif.
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


def read_cif_cations(path: Path) -> tuple[np.ndarray, int]:
    coordinates, spins, period = read_cation_configuration(path)
    grid = np.full((period,) * 3, -1, dtype=np.int8)
    grid[tuple(coordinates.T)] = (spins == -1).astype(np.int8)
    return grid, period // 2


def motif_indices(nrep: int) -> dict[str, np.ndarray]:
    period = 2 * nrep
    shape = (period,) * 3
    points = np.indices(shape).reshape(3, -1).T
    parity = points.sum(axis=1) % 2

    def flat(coordinates: np.ndarray) -> np.ndarray:
        return np.ravel_multi_index(coordinates.T % period, shape)

    oxygen = points[parity == 0]
    axial = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )
    octahedra = np.stack([flat(oxygen + delta) for delta in axial], axis=1)

    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    tetrahedra = []
    for center_floor in points:
        vertices = (center_floor + corners) % period
        vertices = vertices[vertices.sum(axis=1) % 2 == 1]
        tetrahedra.append(flat(vertices))

    cations = points[parity == 1]
    cubes = np.stack([flat(cations + 2 * delta) for delta in corners], axis=1)
    return {
        "tetrahedron": np.asarray(tetrahedra, dtype=np.int64),
        "octahedron": octahedra,
        "cube": cubes,
    }


def random_fixed_composition(n_sites: int, n_fe: int, motif_size: int) -> np.ndarray:
    denominator = math.comb(n_sites, motif_size)
    return np.array(
        [
            math.comb(n_fe, count)
            * math.comb(n_sites - n_fe, motif_size - count)
            / denominator
            for count in range(motif_size + 1)
        ]
    )


def analyze_file(
    path: Path, cached_indices: dict[int, dict[str, np.ndarray]]
) -> list[dict]:
    grid, nrep = read_cif_cations(path)
    if nrep not in cached_indices:
        cached_indices[nrep] = motif_indices(nrep)
    indices = cached_indices[nrep]
    flat_grid = grid.ravel()
    n_cations = int(np.count_nonzero(flat_grid >= 0))
    n_fe = int(np.count_nonzero(flat_grid == 1))
    rows = []
    for motif, vertex_indices in indices.items():
        values = flat_grid[vertex_indices]
        if np.any(values < 0):
            raise RuntimeError(f"Invalid non-cation vertex in {motif}")
        fe_counts = values.sum(axis=1)
        observed_counts = np.bincount(fe_counts, minlength=values.shape[1] + 1)
        random_fraction = random_fixed_composition(n_cations, n_fe, values.shape[1])
        for count, observed_count in enumerate(observed_counts):
            fraction = observed_count / len(fe_counts)
            rows.append(
                {
                    "motif": motif,
                    "motif_size": values.shape[1],
                    "n_fe": count,
                    "count": int(observed_count),
                    "n_motifs": len(fe_counts),
                    "observed_fraction": fraction,
                    "random_fraction": random_fraction[count],
                    "observed_over_random": (
                        fraction / random_fraction[count]
                        if random_fraction[count]
                        else float("nan")
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structures",
        nargs="+",
        type=Path,
        help="Ideal Li/Fe CIF or extended-XYZ files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/local_environment_counts.csv")
    )
    args = parser.parse_args()
    rows = []
    cache = {}
    for path in args.structures:
        for row in analyze_file(path, cache):
            rows.append({"source_file": str(path), **row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
