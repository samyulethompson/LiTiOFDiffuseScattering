#!/usr/bin/env python3
"""Count all-Li (0-TM) tetrahedral sites in a Li/Ti rocksalt structure."""

from __future__ import annotations

import argparse
import csv
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


def count_zero_tm_sites(is_ti: np.ndarray, tetra_to_cation: np.ndarray) -> int:
    """A tetrahedron is 0-TM when none of its four cation sites contains Ti."""
    return int(np.count_nonzero(~np.any(is_ti[tetra_to_cation], axis=1)))


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
    parser.add_argument("--output", type=Path, default=Path("results/0tm_counts.csv"))
    args = parser.parse_args()
    is_ti, nrep = read_occupations(args.structure, args.frame)
    _, tetra_to_cation = cation_and_tetrahedron_tables(nrep)
    n_tetrahedra = len(tetra_to_cation)
    count = count_zero_tm_sites(is_ti, tetra_to_cation)
    row = {
        "source_file": str(args.structure),
        "frame": args.frame,
        "n_cations": len(is_ti),
        "n_li": int(np.count_nonzero(~is_ti)),
        "n_ti": int(np.count_nonzero(is_ti)),
        "n_tetrahedral_sites": n_tetrahedra,
        "n_zero_tm_sites": count,
        "zero_tm_fraction": count / n_tetrahedra,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(
        f"0-TM tetrahedral sites: {count}/{n_tetrahedra} ({row['zero_tm_fraction']:.6f})"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
