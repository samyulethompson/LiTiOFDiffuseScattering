#!/usr/bin/env python3
"""Fixed-composition Li/Ti Monte Carlo and the pair-based displacement model.

H/(k_B T) = K1 * N_TiTi(first shell) + K2 * N_TiTi(second shell).
The defaults reproduce the supplied example cell; all energy coefficients are reduced.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from ase import Atoms

ANION_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=float,
)

CATION_BASIS = np.array(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ],
    dtype=float,
)


def anion_site_id(ix: int, iy: int, iz: int, basis_index: int, nrep: int) -> int:
    return (((ix * nrep + iy) * nrep + iz) * len(ANION_BASIS)) + basis_index


def cation_site_id(ix: int, iy: int, iz: int, basis_index: int, nrep: int) -> int:
    return (((ix * nrep + iy) * nrep + iz) * len(CATION_BASIS)) + basis_index


def doubled_position(
    ix: int, iy: int, iz: int, basis: np.ndarray
) -> tuple[int, int, int]:
    basis_doubled = np.rint(2 * basis).astype(int)
    return (
        2 * ix + int(basis_doubled[0]),
        2 * iy + int(basis_doubled[1]),
        2 * iz + int(basis_doubled[2]),
    )


def nearest_anion_cation_vectors() -> list[tuple[int, int, int]]:
    vectors = []
    for axis in range(3):
        for sign in (-1, 1):
            vector = [0, 0, 0]
            vector[axis] = sign
            vectors.append(tuple(vector))
    return vectors


def nearest_cation_cation_vectors() -> list[tuple[int, int, int]]:
    vectors = []
    for zero_axis in range(3):
        axes = [axis for axis in range(3) if axis != zero_axis]
        for sign_a in (-1, 1):
            for sign_b in (-1, 1):
                vector = [0, 0, 0]
                vector[axes[0]] = sign_a
                vector[axes[1]] = sign_b
                vectors.append(tuple(vector))
    return vectors


def second_nearest_cation_cation_vectors() -> list[tuple[int, int, int]]:
    vectors = []
    for axis in range(3):
        for sign in (-1, 1):
            vector = [0, 0, 0]
            vector[axis] = 2 * sign
            vectors.append(tuple(vector))
    return vectors


def build_site_positions(nrep: int, basis: np.ndarray, site_id_func) -> np.ndarray:
    positions = np.zeros((nrep**3 * len(basis), 3), dtype=np.int32)
    for ix in range(nrep):
        for iy in range(nrep):
            for iz in range(nrep):
                for basis_index, site_basis in enumerate(basis):
                    site = site_id_func(ix, iy, iz, basis_index, nrep)
                    positions[site] = doubled_position(ix, iy, iz, site_basis)
    return positions


def build_anion_cation_neighbors(
    nrep: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    period = 2 * nrep
    anion_positions = build_site_positions(nrep, ANION_BASIS, anion_site_id)
    cation_positions = build_site_positions(nrep, CATION_BASIS, cation_site_id)
    cation_lookup = {
        tuple(position): site for site, position in enumerate(cation_positions)
    }
    neighbors = [[] for _ in range(len(anion_positions))]
    unit_vectors = [[] for _ in range(len(anion_positions))]

    for anion, doubled in enumerate(anion_positions):
        for vector in nearest_anion_cation_vectors():
            target = tuple((doubled[axis] + vector[axis]) % period for axis in range(3))
            cation = cation_lookup[target]
            delta_cells = np.array(vector, dtype=float) / 2.0
            unit = delta_cells / np.linalg.norm(delta_cells)
            neighbors[anion].append(cation)
            unit_vectors[anion].append(unit)

    neighbor_arrays = [
        np.array(site_neighbors, dtype=np.int32) for site_neighbors in neighbors
    ]
    vector_arrays = [
        np.array(site_vectors, dtype=float) for site_vectors in unit_vectors
    ]
    if any(len(site_neighbors) != 6 for site_neighbors in neighbor_arrays):
        raise RuntimeError(
            "Expected exactly 6 nearest cation neighbors for every anion site."
        )
    return neighbor_arrays, vector_arrays


def build_cation_cation_neighbors(
    nrep: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    period = 2 * nrep
    cation_positions = build_site_positions(nrep, CATION_BASIS, cation_site_id)
    cation_lookup = {
        tuple(position): site for site, position in enumerate(cation_positions)
    }
    neighbors = [[] for _ in range(len(cation_positions))]
    away_unit_vectors = [[] for _ in range(len(cation_positions))]

    for cation, doubled in enumerate(cation_positions):
        for vector in nearest_cation_cation_vectors():
            target = tuple((doubled[axis] + vector[axis]) % period for axis in range(3))
            neighbor = cation_lookup[target]
            delta_cells = np.array(vector, dtype=float) / 2.0
            away = -delta_cells / np.linalg.norm(delta_cells)
            neighbors[cation].append(neighbor)
            away_unit_vectors[cation].append(away)

    neighbor_arrays = [
        np.array(site_neighbors, dtype=np.int32) for site_neighbors in neighbors
    ]
    vector_arrays = [
        np.array(site_vectors, dtype=float) for site_vectors in away_unit_vectors
    ]
    if any(len(site_neighbors) != 12 for site_neighbors in neighbor_arrays):
        raise RuntimeError(
            "Expected exactly 12 nearest cation neighbors for every cation site."
        )
    return neighbor_arrays, vector_arrays


def build_second_cation_cation_neighbors(
    nrep: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    period = 2 * nrep
    cation_positions = build_site_positions(nrep, CATION_BASIS, cation_site_id)
    cation_lookup = {
        tuple(position): site for site, position in enumerate(cation_positions)
    }
    neighbors = [[] for _ in range(len(cation_positions))]
    toward_unit_vectors = [[] for _ in range(len(cation_positions))]

    for cation, doubled in enumerate(cation_positions):
        for vector in second_nearest_cation_cation_vectors():
            target = tuple((doubled[axis] + vector[axis]) % period for axis in range(3))
            neighbor = cation_lookup[target]
            delta_cells = np.array(vector, dtype=float) / 2.0
            toward = delta_cells / np.linalg.norm(delta_cells)
            neighbors[cation].append(neighbor)
            toward_unit_vectors[cation].append(toward)

    neighbor_arrays = [
        np.array(site_neighbors, dtype=np.int32) for site_neighbors in neighbors
    ]
    vector_arrays = [
        np.array(site_vectors, dtype=float) for site_vectors in toward_unit_vectors
    ]
    if any(len(site_neighbors) != 6 for site_neighbors in neighbor_arrays):
        raise RuntimeError(
            "Expected exactly 6 second-nearest cation neighbors for every cation site."
        )
    return neighbor_arrays, vector_arrays


def randomized_occupations(
    n_sites: int, true_fraction: float, rng: np.random.Generator, label: str
) -> np.ndarray:
    n_true = int(round(n_sites * true_fraction))
    if not math.isclose(n_true / n_sites, true_fraction):
        raise ValueError(
            f"Cannot represent {label} fraction {true_fraction} exactly with {n_sites} sites."
        )
    occupied = np.zeros(n_sites, dtype=bool)
    occupied[:n_true] = True
    rng.shuffle(occupied)
    return occupied


def count_ti_ti_pairs(is_ti: np.ndarray, ti_neighbor_counts: np.ndarray) -> int:
    return int(np.sum(ti_neighbor_counts[is_ti]) // 2)


def run_ti_li_occupational_mc(
    is_ti: np.ndarray,
    nn_neighbors: list[np.ndarray],
    second_neighbors: list[np.ndarray],
    ti_ti_nn_interaction: float,
    ti_ti_second_interaction: float,
    beta: float,
    sweeps: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int, int, int, int]:
    """Metropolis Ti/Li swaps at fixed composition.

    The interaction values are energies per Ti-Ti pair. Negative values favor
    Ti-Ti pairs and positive values disfavor them; both signs are supported.
    """

    ti_sites = np.flatnonzero(is_ti).copy()
    li_sites = np.flatnonzero(~is_ti).copy()
    ti_nn_counts = np.array(
        [
            int(np.count_nonzero(is_ti[site_neighbors]))
            for site_neighbors in nn_neighbors
        ],
        dtype=np.int32,
    )
    ti_second_counts = np.array(
        [
            int(np.count_nonzero(is_ti[site_neighbors]))
            for site_neighbors in second_neighbors
        ],
        dtype=np.int32,
    )
    initial_nn_pairs = count_ti_ti_pairs(is_ti, ti_nn_counts)
    initial_second_pairs = count_ti_ti_pairs(is_ti, ti_second_counts)
    steps = int(round(sweeps * len(is_ti)))
    accepted = 0

    for _ in range(steps):
        ti_slot = int(rng.integers(len(ti_sites)))
        li_slot = int(rng.integers(len(li_sites)))
        old_ti_site = int(ti_sites[ti_slot])
        old_li_site = int(li_sites[li_slot])
        are_nn = bool(np.any(nn_neighbors[old_ti_site] == old_li_site))
        are_second = bool(np.any(second_neighbors[old_ti_site] == old_li_site))
        delta_nn_pairs = (ti_nn_counts[old_li_site] - int(are_nn)) - ti_nn_counts[
            old_ti_site
        ]
        delta_second_pairs = (
            ti_second_counts[old_li_site] - int(are_second)
        ) - ti_second_counts[old_ti_site]
        delta_energy = ti_ti_nn_interaction * float(
            delta_nn_pairs
        ) + ti_ti_second_interaction * float(delta_second_pairs)
        if delta_energy <= 0.0 or rng.random() < math.exp(-beta * delta_energy):
            is_ti[old_ti_site] = False
            is_ti[old_li_site] = True
            ti_sites[ti_slot] = old_li_site
            li_sites[li_slot] = old_ti_site
            for neighbor in nn_neighbors[old_ti_site]:
                ti_nn_counts[neighbor] -= 1
            for neighbor in nn_neighbors[old_li_site]:
                ti_nn_counts[neighbor] += 1
            for neighbor in second_neighbors[old_ti_site]:
                ti_second_counts[neighbor] -= 1
            for neighbor in second_neighbors[old_li_site]:
                ti_second_counts[neighbor] += 1
            accepted += 1

    final_nn_pairs = count_ti_ti_pairs(is_ti, ti_nn_counts)
    final_second_pairs = count_ti_ti_pairs(is_ti, ti_second_counts)
    return (
        is_ti,
        accepted / max(steps, 1),
        initial_nn_pairs,
        final_nn_pairs,
        initial_second_pairs,
        final_second_pairs,
    )


def o_towards_ti_displacements(
    nrep: int,
    is_ti: np.ndarray,
    toward_ti_ang: float,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors, vectors_to_cations = build_anion_cation_neighbors(nrep)
    displacements = np.zeros((len(neighbors), 3), dtype=float)
    ti_neighbor_counts = np.zeros(len(neighbors), dtype=np.int32)

    for anion, site_neighbors in enumerate(neighbors):
        ti_mask = is_ti[site_neighbors]
        ti_neighbor_counts[anion] = int(np.count_nonzero(ti_mask))
        if toward_ti_ang and np.any(ti_mask):
            displacements[anion] += toward_ti_ang * np.sum(
                vectors_to_cations[anion][ti_mask],
                axis=0,
            )
    return displacements, ti_neighbor_counts


def second_neighbor_unlike_repulsion_displacements(
    is_ti: np.ndarray,
    second_neighbors: list[np.ndarray],
    second_toward_unit_vectors: list[np.ndarray],
    contribution_ang: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move Li and Ti away from unlike second-neighbor cations along <100>."""
    displacements = np.zeros((len(is_ti), 3), dtype=float)
    ti_second_neighbor_counts = np.zeros(len(is_ti), dtype=np.int32)
    li_second_neighbor_counts = np.zeros(len(is_ti), dtype=np.int32)
    for site, site_neighbors in enumerate(second_neighbors):
        same_species_mask = is_ti[site_neighbors] == is_ti[site]
        if is_ti[site]:
            ti_second_neighbor_counts[site] = int(np.count_nonzero(same_species_mask))
        else:
            li_second_neighbor_counts[site] = int(np.count_nonzero(same_species_mask))
        unlike_mask = ~same_species_mask
        if contribution_ang and np.any(unlike_mask):
            displacements[site] -= contribution_ang * np.sum(
                second_toward_unit_vectors[site][unlike_mask],
                axis=0,
            )
    return displacements, ti_second_neighbor_counts, li_second_neighbor_counts


def nearest_neighbor_ti_repulsion_displacements(
    is_ti: np.ndarray,
    nearest_neighbors: list[np.ndarray],
    nearest_away_unit_vectors: list[np.ndarray],
    contribution_ang: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Ti displacements away from each nearest-neighbor Ti."""
    displacements = np.zeros((len(is_ti), 3), dtype=float)
    ti_neighbor_counts = np.zeros(len(is_ti), dtype=np.int32)
    for site, site_neighbors in enumerate(nearest_neighbors):
        if not is_ti[site]:
            continue
        ti_mask = is_ti[site_neighbors]
        ti_neighbor_counts[site] = int(np.count_nonzero(ti_mask))
        if contribution_ang and np.any(ti_mask):
            displacements[site] += contribution_ang * np.sum(
                nearest_away_unit_vectors[site][ti_mask],
                axis=0,
            )
    return displacements, ti_neighbor_counts


def make_rocksalt_atoms(
    nrep: int,
    lattice_constant: float,
    is_ti: np.ndarray,
    anion_displacements_ang: np.ndarray | None = None,
    cation_displacements_ang: np.ndarray | None = None,
) -> Atoms:
    symbols = []
    scaled_positions = []
    anion_displacements_scaled = None
    cation_displacements_scaled = None
    if anion_displacements_ang is not None:
        anion_displacements_scaled = anion_displacements_ang / (lattice_constant * nrep)
    if cation_displacements_ang is not None:
        cation_displacements_scaled = cation_displacements_ang / (
            lattice_constant * nrep
        )

    anion_index = 0
    for ix in range(nrep):
        for iy in range(nrep):
            for iz in range(nrep):
                cell_offset = np.array([ix, iy, iz], dtype=float)
                for basis in ANION_BASIS:
                    symbols.append("O")
                    scaled = (cell_offset + basis) / nrep
                    if anion_displacements_scaled is not None:
                        scaled = scaled + anion_displacements_scaled[anion_index]
                    scaled_positions.append((scaled % 1.0).tolist())
                    anion_index += 1

    cation_index = 0
    for ix in range(nrep):
        for iy in range(nrep):
            for iz in range(nrep):
                cell_offset = np.array([ix, iy, iz], dtype=float)
                for basis in CATION_BASIS:
                    symbols.append("Ti" if is_ti[cation_index] else "Li")
                    scaled = (cell_offset + basis) / nrep
                    if cation_displacements_scaled is not None:
                        scaled = scaled + cation_displacements_scaled[cation_index]
                    scaled_positions.append((scaled % 1.0).tolist())
                    cation_index += 1

    cell = np.eye(3) * lattice_constant * nrep
    return Atoms(
        symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mc_example"))
    parser.add_argument(
        "--nrep", type=int, default=21, help="Conventional cells per axis."
    )
    parser.add_argument("--lattice-constant", type=float, default=4.1, help="Angstrom.")
    parser.add_argument("--ti-fraction", type=float, default=2 / 7)
    parser.add_argument("--j1-over-kbt", type=float, default=-0.5)
    parser.add_argument("--j2-over-kbt", type=float, default=0.5)
    parser.add_argument(
        "--mc-sweeps", type=float, default=20, help="Attempted exchanges per cation."
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--o-displacement",
        type=float,
        default=0.100,
        help="Angstrom per nearest Ti-O pair.",
    )
    parser.add_argument(
        "--ti-ti-nn-repulsion",
        type=float,
        default=0.025,
        help="Angstrom per nearest Ti-Ti pair, per Ti.",
    )
    parser.add_argument(
        "--unlike-second-repulsion",
        type=float,
        default=0.075,
        help="Angstrom per second-neighbour Li-Ti pair, per cation.",
    )
    args = parser.parse_args()
    values = [
        args.lattice_constant,
        args.ti_fraction,
        args.j1_over_kbt,
        args.j2_over_kbt,
        args.mc_sweeps,
        args.o_displacement,
        args.ti_ti_nn_repulsion,
        args.unlike_second_repulsion,
    ]
    if not all(math.isfinite(value) for value in values):
        parser.error("Numerical parameters must be finite.")
    if args.nrep < 3 or args.lattice_constant <= 0 or not 0 < args.ti_fraction < 1:
        parser.error(
            "Require nrep >= 3, positive lattice constant, and 0 < ti-fraction < 1."
        )
    if (
        min(
            args.mc_sweeps,
            args.o_displacement,
            args.ti_ti_nn_repulsion,
            args.unlike_second_repulsion,
        )
        < 0
    ):
        parser.error("Sweep counts and displacement amplitudes must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_cations = 4 * args.nrep**3
    is_ti = randomized_occupations(n_cations, args.ti_fraction, rng, "Ti")
    nearest, nearest_away = build_cation_cation_neighbors(args.nrep)
    second, second_toward = build_second_cation_cation_neighbors(args.nrep)
    is_ti, acceptance, initial_nn, final_nn, initial_second, final_second = (
        run_ti_li_occupational_mc(
            is_ti,
            nearest,
            second,
            args.j1_over_kbt,
            args.j2_over_kbt,
            1.0,
            args.mc_sweeps,
            rng,
        )
    )
    # Displacements are assigned after MC and do not enter its acceptance energy.
    oxygen_displacements, _ = o_towards_ti_displacements(
        args.nrep, is_ti, args.o_displacement
    )
    cation_displacements, _, _ = second_neighbor_unlike_repulsion_displacements(
        is_ti, second, second_toward, args.unlike_second_repulsion
    )
    nearest_displacements, _ = nearest_neighbor_ti_repulsion_displacements(
        is_ti, nearest, nearest_away, args.ti_ti_nn_repulsion
    )
    cation_displacements += nearest_displacements
    ideal = make_rocksalt_atoms(args.nrep, args.lattice_constant, is_ti)
    displaced = make_rocksalt_atoms(
        args.nrep,
        args.lattice_constant,
        is_ti,
        oxygen_displacements,
        cation_displacements,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ideal.write(args.output_dir / "mc_occupations.extxyz")
    displaced.write(args.output_dir / "displaced.extxyz")
    summary = {
        "nrep": args.nrep,
        "lattice_constant_ang": args.lattice_constant,
        "n_cations": n_cations,
        "n_ti": int(is_ti.sum()),
        "n_li": int((~is_ti).sum()),
        "j1_over_kbt": args.j1_over_kbt,
        "j2_over_kbt": args.j2_over_kbt,
        "mc_sweeps": args.mc_sweeps,
        "attempted_exchanges": round(args.mc_sweeps * n_cations),
        "seed": args.seed,
        "acceptance_fraction": acceptance,
        "initial_ti_ti_nn_pairs": initial_nn,
        "final_ti_ti_nn_pairs": final_nn,
        "initial_ti_ti_second_pairs": initial_second,
        "final_ti_ti_second_pairs": final_second,
        "o_displacement_ang_per_pair": args.o_displacement,
        "ti_ti_nn_repulsion_ang_per_pair": args.ti_ti_nn_repulsion,
        "unlike_second_repulsion_ang_per_pair": args.unlike_second_repulsion,
    }
    with (args.output_dir / "mc_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(summary.items())
    print(
        f"Wrote occupational and displaced structures and mc_summary.csv to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
