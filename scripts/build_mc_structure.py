#!/usr/bin/env python3
"""Build a Li/Ti/O rocksalt model with Monte Carlo cation occupations."""

import argparse
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


def write_scatty_structure(
    atoms: Atoms, path: Path, index: int, cations: list[str]
) -> None:
    """Write an ASE rocksalt structure in Scatty's atom-list format."""
    nrep = round((len(atoms) / 8) ** (1 / 3))
    if 8 * nrep**3 != len(atoms):
        raise ValueError(f"Cannot infer a cubic rocksalt supercell from {len(atoms)} atoms")
    supercell = np.array([nrep, nrep, nrep], dtype=int)
    unique_elements = sorted(set(atoms.get_chemical_symbols()))
    for element in cations:
        if element not in unique_elements:
            unique_elements.append(element)

    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    symbols = atoms.get_chemical_symbols()
    atom_counts = {
        element: symbols.count(element) / (len(symbols) / 2)
        for element in unique_elements
    }
    vectors = np.array(
        sorted(
            ([x, y, z] for x in (0.0, 0.5) for y in (0.0, 0.5) for z in (0.0, 0.5)),
            key=lambda vector: (list(vector).count(0.5) not in (0, 2), vector),
        )
    )

    lines = [
        f"TITLE {unique_elements}",
        (
            f"CELL {a / nrep:.6f} {b / nrep:.6f} {c / nrep:.6f} "
            f"{alpha:.6f} {beta:.6f} {gamma:.6f}"
        ),
        f"BOX {nrep} {nrep} {nrep}",
    ]
    lines.extend(f"SITE {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vectors)
    for site_index in range(8):
        if site_index < 4:
            elements = [element for element in unique_elements if element in {"O", "F"}]
        else:
            elements = [element for element in unique_elements if element not in {"O", "F"}]
        occupations = " ".join(
            f"{element} {atom_counts.get(element, 0.0):.6f}" for element in elements
        )
        lines.append(f"OCC {occupations}")

    fractional_coordinates = atoms.get_scaled_positions()
    fractional_coordinates[np.isclose(fractional_coordinates, 0.0, atol=1e-12)] = 0.0
    fractional_coordinates[np.isclose(fractional_coordinates, 1.0, atol=1e-12)] = 0.0
    for fractional, symbol in zip(fractional_coordinates, symbols):
        position = fractional * supercell
        image = np.floor(position + 1e-10).astype(int) % supercell
        unit_cell_coordinate = np.round((position % 1.0) * 2) / 2 % 1.0
        matches = np.flatnonzero(
            np.all(np.isclose(vectors, unit_cell_coordinate, atol=1e-8), axis=1)
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one crystallographic-site match for {unit_cell_coordinate}"
            )
        crystallographic_site = int(matches[0]) + 1
        ix, iy, iz = image
        lines.append(
            f"ATOM {crystallographic_site} {ix} {iy} {iz} 0 0 0 {symbol}"
        )

    output = path.with_name(f"{path.name}_atoms_{index:02d}.txt")
    output.write_text("\n".join(lines) + "\n")


def anion_site_id(ix: int, iy: int, iz: int, basis_index: int, nrep: int) -> int:
    return (((ix * nrep + iy) * nrep + iz) * len(ANION_BASIS)) + basis_index


def cation_site_id(ix: int, iy: int, iz: int, basis_index: int, nrep: int) -> int:
    return (((ix * nrep + iy) * nrep + iz) * len(CATION_BASIS)) + basis_index


def doubled_position(ix: int, iy: int, iz: int, basis: np.ndarray) -> tuple[int, int, int]:
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


def build_anion_cation_neighbors(nrep: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    period = 2 * nrep
    anion_positions = build_site_positions(nrep, ANION_BASIS, anion_site_id)
    cation_positions = build_site_positions(nrep, CATION_BASIS, cation_site_id)
    cation_lookup = {tuple(position): site for site, position in enumerate(cation_positions)}
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

    neighbor_arrays = [np.array(site_neighbors, dtype=np.int32) for site_neighbors in neighbors]
    vector_arrays = [np.array(site_vectors, dtype=float) for site_vectors in unit_vectors]
    if any(len(site_neighbors) != 6 for site_neighbors in neighbor_arrays):
        raise RuntimeError("Expected exactly 6 nearest cation neighbors for every anion site.")
    return neighbor_arrays, vector_arrays


def build_cation_cation_neighbors(nrep: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    period = 2 * nrep
    cation_positions = build_site_positions(nrep, CATION_BASIS, cation_site_id)
    cation_lookup = {tuple(position): site for site, position in enumerate(cation_positions)}
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

    neighbor_arrays = [np.array(site_neighbors, dtype=np.int32) for site_neighbors in neighbors]
    vector_arrays = [np.array(site_vectors, dtype=float) for site_vectors in away_unit_vectors]
    if any(len(site_neighbors) != 12 for site_neighbors in neighbor_arrays):
        raise RuntimeError("Expected exactly 12 nearest cation neighbors for every cation site.")
    return neighbor_arrays, vector_arrays


def build_second_cation_cation_neighbors(nrep: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    period = 2 * nrep
    cation_positions = build_site_positions(nrep, CATION_BASIS, cation_site_id)
    cation_lookup = {tuple(position): site for site, position in enumerate(cation_positions)}
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

    neighbor_arrays = [np.array(site_neighbors, dtype=np.int32) for site_neighbors in neighbors]
    vector_arrays = [np.array(site_vectors, dtype=float) for site_vectors in toward_unit_vectors]
    if any(len(site_neighbors) != 6 for site_neighbors in neighbor_arrays):
        raise RuntimeError("Expected exactly 6 second-nearest cation neighbors for every cation site.")
    return neighbor_arrays, vector_arrays


def randomized_occupations(n_sites: int, true_fraction: float, rng: np.random.Generator, label: str) -> np.ndarray:
    n_true = int(round(n_sites * true_fraction))
    if not math.isclose(n_true / n_sites, true_fraction):
        raise ValueError(f"Cannot represent {label} fraction {true_fraction} exactly with {n_sites} sites.")
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
        [int(np.count_nonzero(is_ti[site_neighbors])) for site_neighbors in nn_neighbors],
        dtype=np.int32,
    )
    ti_second_counts = np.array(
        [int(np.count_nonzero(is_ti[site_neighbors])) for site_neighbors in second_neighbors],
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
        delta_nn_pairs = (ti_nn_counts[old_li_site] - int(are_nn)) - ti_nn_counts[old_ti_site]
        delta_second_pairs = (
            ti_second_counts[old_li_site] - int(are_second)
        ) - ti_second_counts[old_ti_site]
        delta_energy = (
            ti_ti_nn_interaction * float(delta_nn_pairs)
            + ti_ti_second_interaction * float(delta_second_pairs)
        )
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


def displacement_stats(prefix: str, mask: np.ndarray, displacements_ang: np.ndarray) -> list[str]:
    magnitudes = np.linalg.norm(displacements_ang[mask], axis=1)
    if magnitudes.size == 0:
        return [
            f"min_{prefix}_displacement_ang,nan",
            f"mean_{prefix}_displacement_ang,nan",
            f"max_{prefix}_displacement_ang,nan",
        ]
    return [
        f"min_{prefix}_displacement_ang,{float(np.min(magnitudes)):.8f}",
        f"mean_{prefix}_displacement_ang,{float(np.mean(magnitudes)):.8f}",
        f"max_{prefix}_displacement_ang,{float(np.max(magnitudes)):.8f}",
    ]


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
        cation_displacements_scaled = cation_displacements_ang / (lattice_constant * nrep)

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
    return Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)


def rewrite_scatty_displacements(
    path: Path,
    anion_displacements_ang: np.ndarray,
    cation_displacements_ang: np.ndarray,
    lattice_constant: float,
) -> None:
    lines = path.read_text().splitlines()
    anion_index = 0
    cation_index = 0
    rewritten = []
    for line in lines:
        if not line.startswith("ATOM "):
            rewritten.append(line)
            continue
        parts = line.split()
        symbol = parts[-1]
        if symbol == "O":
            disp = anion_displacements_ang[anion_index] / lattice_constant
            parts[5:8] = [f"{disp[0]:.8f}", f"{disp[1]:.8f}", f"{disp[2]:.8f}"]
            anion_index += 1
        elif symbol in {"Li", "Ti"}:
            disp = cation_displacements_ang[cation_index] / lattice_constant
            parts[5:8] = [f"{disp[0]:.8f}", f"{disp[1]:.8f}", f"{disp[2]:.8f}"]
            cation_index += 1
        rewritten.append(" ".join(parts))
    if anion_index != len(anion_displacements_ang):
        raise RuntimeError(f"Rewrote {anion_index} anions, expected {len(anion_displacements_ang)}.")
    if cation_index != len(cation_displacements_ang):
        raise RuntimeError(f"Rewrote {cation_index} cations, expected {len(cation_displacements_ang)}.")
    path.write_text("\n".join(rewritten) + "\n")


def write_displacive_scatty_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "NAME hk0",
                "CENTRE 0 0 0",
                "X_AXIS 4 0 0 50",
                "Y_AXIS 0 4 0 50",
                "Z_AXIS 0 0 4 50",
                "WINDOW 2",
                "PPM_OUTPUT",
                "SUM parallel",
                "RADIATION E",
                "SYMMETRY m3m",
                "EXPANSION_ORDER 10",
                "",
            ]
        )
    )


def write_summary(
    path: Path,
    args: argparse.Namespace,
    is_ti: np.ndarray,
    anion_displacements_ang: np.ndarray,
    cation_displacements_ang: np.ndarray,
    o_ti_neighbor_counts: np.ndarray,
    ti_nn_neighbor_counts: np.ndarray,
    ti_second_neighbor_counts: np.ndarray,
    li_second_neighbor_counts: np.ndarray,
    mc_acceptance: float,
    initial_ti_ti_pairs: int,
    final_ti_ti_pairs: int,
    initial_second_ti_ti_pairs: int,
    final_second_ti_ti_pairs: int,
) -> None:
    o_magnitudes = np.linalg.norm(anion_displacements_ang, axis=1)
    cation_magnitudes = np.linalg.norm(cation_displacements_ang, axis=1)
    lines = [
        f"nrep,{args.nrep}",
        f"lattice_constant_ang,{args.lattice_constant}",
        f"seed,{args.seed}",
        f"li_to_ti_ratio,{np.count_nonzero(~is_ti) / np.count_nonzero(is_ti):.12g}",
        f"ti_fraction,{args.ti_fraction}",
        f"n_cations,{len(is_ti)}",
        f"n_ti,{int(np.count_nonzero(is_ti))}",
        f"n_li,{int(np.count_nonzero(~is_ti))}",
        f"n_anions,{len(anion_displacements_ang)}",
        f"n_o,{len(anion_displacements_ang)}",
        f"ti_ti_nn_interaction,{args.ti_ti_interaction}",
        f"ti_ti_second_interaction,{args.ti_ti_second_interaction}",
        f"mc_beta,{args.mc_beta}",
        f"mc_sweeps,{args.mc_sweeps}",
        f"mc_acceptance,{mc_acceptance:.8f}",
        f"initial_ti_ti_nn_pairs,{initial_ti_ti_pairs}",
        f"final_ti_ti_nn_pairs,{final_ti_ti_pairs}",
        f"initial_ti_ti_second_pairs,{initial_second_ti_ti_pairs}",
        f"final_ti_ti_second_pairs,{final_second_ti_ti_pairs}",
        f"o_towards_each_ti_neighbor_contribution_ang,{args.o_displacement}",
        f"ti_away_from_each_nearest_ti_neighbor_contribution_ang,{args.ti_ti_nn_repulsion}",
        f"unlike_cations_away_at_second_neighbor_contribution_ang,{args.unlike_second_repulsion}",
        f"min_o_displacement_ang,{float(np.min(o_magnitudes)):.8f}",
        f"mean_o_displacement_ang,{float(np.mean(o_magnitudes)):.8f}",
        f"max_o_displacement_ang,{float(np.max(o_magnitudes)):.8f}",
        f"max_li_displacement_ang,{float(np.max(cation_magnitudes[~is_ti])):.8f}",
    ]
    lines.extend(displacement_stats("li", ~is_ti, cation_displacements_ang))
    lines.extend(displacement_stats("ti", is_ti, cation_displacements_ang))
    for count in range(7):
        lines.append(f"n_o_with_{count}_ti_neighbors,{int(np.count_nonzero(o_ti_neighbor_counts == count))}")
    for count in range(13):
        lines.append(
            f"n_ti_with_{count}_nearest_ti_neighbors,{int(np.count_nonzero(is_ti & (ti_nn_neighbor_counts == count)))}"
        )
    for count in range(7):
        lines.append(
            f"n_ti_with_{count}_second_ti_neighbors,{int(np.count_nonzero(is_ti & (ti_second_neighbor_counts == count)))}"
        )
    for count in range(7):
        lines.append(
            f"n_li_with_{count}_second_li_neighbors,{int(np.count_nonzero((~is_ti) & (li_second_neighbor_counts == count)))}"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def build_structure(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    scatty_dir = output_dir / "scatty_structures"
    output_dir.mkdir(parents=True, exist_ok=True)
    scatty_dir.mkdir(parents=True, exist_ok=True)

    n_cations = args.nrep**3 * len(CATION_BASIS)
    cation_cation_neighbors, nearest_away_cations = build_cation_cation_neighbors(args.nrep)
    second_cation_neighbors, second_toward_cations = build_second_cation_cation_neighbors(args.nrep)
    is_ti = randomized_occupations(n_cations, args.ti_fraction, rng, "Ti")
    (
        is_ti,
        mc_acceptance,
        initial_pairs,
        final_pairs,
        initial_second_pairs,
        final_second_pairs,
    ) = run_ti_li_occupational_mc(
        is_ti,
        cation_cation_neighbors,
        second_cation_neighbors,
        args.ti_ti_interaction,
        args.ti_ti_second_interaction,
        args.mc_beta,
        args.mc_sweeps,
        rng,
    )
    anion_displacements_ang, o_ti_neighbor_counts = o_towards_ti_displacements(
        args.nrep,
        is_ti,
        args.o_displacement,
    )
    cation_displacements_ang, ti_second_neighbor_counts, li_second_neighbor_counts = (
        second_neighbor_unlike_repulsion_displacements(
            is_ti,
            second_cation_neighbors,
            second_toward_cations,
            args.unlike_second_repulsion,
        )
    )
    nn_repulsion_displacements_ang, ti_nn_neighbor_counts = (
        nearest_neighbor_ti_repulsion_displacements(
            is_ti,
            cation_cation_neighbors,
            nearest_away_cations,
            args.ti_ti_nn_repulsion,
        )
    )
    cation_displacements_ang += nn_repulsion_displacements_ang

    ideal_atoms = make_rocksalt_atoms(args.nrep, args.lattice_constant, is_ti)
    displaced_atoms = make_rocksalt_atoms(
        args.nrep,
        args.lattice_constant,
        is_ti,
        anion_displacements_ang,
        cation_displacements_ang,
    )

    stem = scatty_dir / args.stem
    write_displacive_scatty_config(scatty_dir / "scatty_config.txt")
    scatty_paths = []
    for scatty_index in (0, 1):
        write_scatty_structure(ideal_atoms, stem, scatty_index, cations=["Li", "Ti"])
        scatty_path = scatty_dir / f"{args.stem}_atoms_{scatty_index:02d}.txt"
        rewrite_scatty_displacements(
            scatty_path,
            anion_displacements_ang,
            cation_displacements_ang,
            args.lattice_constant,
        )
        scatty_paths.append(scatty_path)

    cif_path = scatty_dir / f"{args.stem}_atoms_00.cif"
    extxyz_path = scatty_dir / f"{args.stem}_atoms_00.extxyz"
    mc_cif_path = output_dir / f"{args.stem}_mc_occupations.cif"
    mc_extxyz_path = output_dir / f"{args.stem}_mc_occupations.extxyz"
    ideal_atoms.write(mc_cif_path)
    ideal_atoms.write(mc_extxyz_path)
    displaced_atoms.write(cif_path)
    displaced_atoms.write(extxyz_path)
    summary_path = output_dir / "ti_clustered_o_ti_displaced_summary.csv"
    write_summary(
        summary_path,
        args,
        is_ti,
        anion_displacements_ang,
        cation_displacements_ang,
        o_ti_neighbor_counts,
        ti_nn_neighbor_counts,
        ti_second_neighbor_counts,
        li_second_neighbor_counts,
        mc_acceptance,
        initial_pairs,
        final_pairs,
        initial_second_pairs,
        final_second_pairs,
    )

    print(
        f"Built {n_cations} cation sites with {int(np.count_nonzero(~is_ti))} Li "
        f"and {int(np.count_nonzero(is_ti))} Ti."
    )
    print(f"Built {len(anion_displacements_ang)} O anions.")
    print(
        f"MC changed Ti-Ti NN pairs from {initial_pairs} to {final_pairs} "
        f"with acceptance {mc_acceptance:.4f}."
    )
    print(
        f"MC changed Ti-Ti second-neighbor pairs from {initial_second_pairs} "
        f"to {final_second_pairs}."
    )
    print(f"Moved O by summed {args.o_displacement} A contributions toward each Ti nearest neighbor.")
    print(
        f"Moved Ti by summed {args.ti_ti_nn_repulsion} A contributions "
        "away from each Ti nearest neighbor along <110>."
    )
    print(
        f"Moved unlike cations by summed {args.unlike_second_repulsion} A contributions "
        "away from each other at second-neighbor distance along <100>."
    )
    print(f"Wrote pre-displacement MC CIF to {mc_cif_path}")
    print(f"Wrote pre-displacement MC extxyz to {mc_extxyz_path}")
    print(f"Wrote Scatty files to {scatty_paths[0]} and {scatty_paths[1]}")
    print(f"Wrote displaced CIF to {cif_path}")
    print(f"Wrote displaced extxyz to {extxyz_path}")
    print(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Run 71 model: occupational Monte Carlo followed by O-to-Ti, "
            "nearest Ti-Ti, and second-neighbor unlike-cation displacements."
        )
    )
    parser.add_argument("--nrep", type=int, default=21, help="Rocksalt conventional cells per axis.")
    parser.add_argument("--lattice-constant", type=float, default=4.1, help="Cubic cell length in Angstrom.")
    parser.add_argument("--ti-fraction", type=float, default=2.0 / 7.0, help="Fraction of cation sites occupied by Ti.")
    parser.add_argument(
        "--ti-ti-interaction",
        type=float,
        default=-0.5,
        help="Negative energy per nearest-neighbor Ti-Ti pair.",
    )
    parser.add_argument(
        "--ti-ti-second-interaction",
        type=float,
        default=0.5,
        help="Energy per second-neighbor Ti-Ti pair; positive values are repulsive.",
    )
    parser.add_argument("--mc-beta", type=float, default=1.0, help="Inverse temperature for dimensionless MC.")
    parser.add_argument("--mc-sweeps", type=float, default=20.0, help="Attempted Ti/Li swaps per cation site.")
    parser.add_argument("--o-displacement", type=float, default=0.1, help="O displacement toward Ti in Angstrom.")
    parser.add_argument(
        "--ti-ti-nn-repulsion",
        type=float,
        default=0.025,
        help="Ti displacement contribution away from each nearest-neighbor Ti in Angstrom.",
    )
    parser.add_argument(
        "--unlike-second-repulsion",
        type=float,
        default=0.075,
        help="Displacement contribution away from each unlike second-neighbor cation in Angstrom.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument("--stem", default="rocks_ti_clustered_o_ti_displaced", help="Shared output stem.")
    parser.add_argument(
        "--output-dir",
        default="mc_output",
        help="Directory for output files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_structure(parse_args())
