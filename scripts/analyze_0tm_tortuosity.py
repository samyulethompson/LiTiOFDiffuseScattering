#!/usr/bin/env python3
"""Relative microscopic tortuosity of the periodic 0-TM tetrahedral network.

For each sampled site, tau is the shortest six-neighbour path to a nonzero
periodic image of the same site, divided by the open-lattice path 2*nrep.
Searches use a 3 x 3 x 3 image halo, as in the Run 81 calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, config
from scipy.stats import t as student_t

from analyze_0tm_connectivity import cation_and_tetrahedron_tables
from analyze_0tm_percolation import iter_configurations, tetra_dsu, write_rows


def largest_wrapping_component(active_cube: np.ndarray) -> np.ndarray:
    """Select the largest component containing a nonzero winding cycle."""
    period = active_cube.shape[0]
    active = active_cube.ravel()
    dsu = tetra_dsu(active, period)
    ids = np.flatnonzero(active)
    roots = np.array([dsu.find(int(site))[0] for site in ids], dtype=np.int32)
    unique, sizes = np.unique(roots, return_counts=True)
    wrapping = np.array([np.any(dsu.wrap_axes[root]) for root in unique], dtype=bool)
    keep = np.zeros(active.size, dtype=bool)
    if wrapping.any():
        candidates = unique[wrapping]
        largest = candidates[np.argmax(sizes[wrapping])]
        keep[ids[roots == largest]] = True
    return keep.reshape(active_cube.shape)


def component_neighbors(component: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return six-neighbour indices and image shifts for a periodic component."""
    period = component.shape[0]
    coords = np.argwhere(component).astype(np.int16)
    lookup = np.full(component.shape, -1, dtype=np.int32)
    lookup[tuple(coords.T)] = np.arange(len(coords), dtype=np.int32)
    neighbors = np.full((len(coords), 6), -1, dtype=np.int32)
    image_shifts = np.zeros((len(coords), 6), dtype=np.int8)

    slot = 0
    for axis in range(3):
        for sign in (-1, 1):
            raw = coords[:, axis].astype(np.int32) + sign
            shifted_coords = coords.copy()
            shifted_coords[:, axis] = np.mod(raw, period)
            neighbors[:, slot] = lookup[tuple(shifted_coords.T)]
            # Encode a shift of -1, 0, or +1 along the relevant axis. The
            # other two components are necessarily zero for a grid edge.
            crossed = (raw < 0).astype(np.int8) * -1 + (raw >= period).astype(np.int8)
            image_shifts[:, slot] = crossed
            slot += 1
    return neighbors, image_shifts


@njit(parallel=True, cache=True)
def periodic_return_lengths(
    neighbors: np.ndarray,
    image_shifts: np.ndarray,
    sources: np.ndarray,
    n_workers: int,
) -> np.ndarray:
    """Shortest hops from each source to any nonzero periodic image.

    Image layers use 0..26 with the central image at 13. Each worker reuses a
    generation-stamped visited array and queue to avoid clearing a large halo
    array for every source.
    """
    n_sites = neighbors.shape[0]
    n_states = 27 * n_sites
    result = np.full(len(sources), -1, dtype=np.int32)

    for worker in prange(n_workers):
        seen = np.zeros(n_states, dtype=np.int32)
        distance = np.empty(n_states, dtype=np.int32)
        queue = np.empty(n_states, dtype=np.int32)
        generation = np.int32(0)

        for source_position in range(worker, len(sources), n_workers):
            generation += 1
            if generation == np.iinfo(np.int32).max:
                seen[:] = 0
                generation = 1
            source = int(sources[source_position])
            start = 13 * n_sites + source
            head = 0
            tail = 1
            queue[0] = start
            seen[start] = generation
            distance[start] = 0

            found = -1
            while head < tail and found < 0:
                state = int(queue[head])
                head += 1
                layer = state // n_sites
                site = state - layer * n_sites
                x_image = layer // 9
                y_image = (layer // 3) % 3
                z_image = layer % 3
                next_distance = int(distance[state]) + 1

                for slot in range(6):
                    neighbor = int(neighbors[site, slot])
                    if neighbor < 0:
                        continue
                    axis = slot // 2
                    shift = int(image_shifts[site, slot])
                    nx = x_image
                    ny = y_image
                    nz = z_image
                    if axis == 0:
                        nx += shift
                    elif axis == 1:
                        ny += shift
                    else:
                        nz += shift
                    if nx < 0 or nx > 2 or ny < 0 or ny > 2 or nz < 0 or nz > 2:
                        continue
                    next_layer = nx * 9 + ny * 3 + nz
                    next_state = next_layer * n_sites + neighbor
                    if seen[next_state] == generation:
                        continue
                    if neighbor == source and next_layer != 13:
                        found = next_distance
                        break
                    seen[next_state] = generation
                    distance[next_state] = next_distance
                    queue[tail] = next_state
                    tail += 1
            result[source_position] = found
    return result


def summarize(values: np.ndarray) -> dict[str, float | int]:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sample_std": sd,
        "standard_error_of_mean": se,
        "mean_ci95_lower": mean - critical * se,
        "mean_ci95_upper": mean + critical * se,
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "percentile_2p5": float(np.percentile(values, 2.5)),
        "percentile_97p5": float(np.percentile(values, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Li/Ti structure or Run 81 configurations_bitpacked.npz.",
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--max-configs",
        type=int,
        help="Maximum configurations per ensemble kind; default: all.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sites-per-config",
        type=int,
        default=4096,
        help="Sampled sites in the largest wrapping cluster (default: 4096).",
    )
    selection.add_argument("--all-sites", action="store_true")
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tortuosity"))
    args = parser.parse_args()
    if args.max_configs is not None and args.max_configs < 1:
        parser.error("--max-configs must be positive.")
    if args.sites_per_config < 1:
        parser.error("--sites-per-config must be positive.")
    if not 1 <= args.workers <= config.NUMBA_NUM_THREADS:
        parser.error(f"--workers must be between 1 and {config.NUMBA_NUM_THREADS}.")
    set_num_threads(args.workers)
    rng = np.random.default_rng(args.seed)
    cached_tables, rows = {}, []
    for index, kind, is_ti, nrep in iter_configurations(
        args.input, args.max_configs, args.frame
    ):
        period = 2 * nrep
        if nrep not in cached_tables:
            cached_tables[nrep] = cation_and_tetrahedron_tables(nrep)[1]
        zero_tm = ~np.any(is_ti[cached_tables[nrep]], axis=1)
        component = largest_wrapping_component(zero_tm.reshape((period,) * 3))
        neighbors, image_shifts = component_neighbors(component)
        sources = np.arange(len(neighbors), dtype=np.int32)
        sampled = not args.all_sites and args.sites_per_config < len(sources)
        if sampled:
            sources = np.sort(
                rng.choice(sources, size=args.sites_per_config, replace=False)
            ).astype(np.int32)
        if len(sources):
            hops = periodic_return_lengths(
                neighbors, image_shifts, sources, min(args.workers, len(sources))
            )
            if np.any(hops < 0):
                raise RuntimeError(
                    f"No return found within the 3x3x3 halo for {np.count_nonzero(hops < 0)} sites in {kind} configuration {index}."
                )
            relative = hops.astype(float) / period
        else:
            hops = np.empty(0, dtype=np.int32)
            relative = np.empty(0)
        row = {
            "index": index,
            "structure_kind": kind,
            "periodic_cluster_sites": len(neighbors),
            "sites_evaluated": len(sources),
            "sampled_sites": sampled,
            "mean_return_hops": float(hops.mean()) if len(hops) else float("nan"),
            "mean_relative_tortuosity": (
                float(relative.mean()) if len(relative) else float("nan")
            ),
            "site_tortuosity_sample_std": (
                float(relative.std(ddof=1)) if len(relative) > 1 else float("nan")
            ),
            "min_relative_tortuosity": (
                float(relative.min()) if len(relative) else float("nan")
            ),
            "median_relative_tortuosity": (
                float(np.median(relative)) if len(relative) else float("nan")
            ),
            "max_relative_tortuosity": (
                float(relative.max()) if len(relative) else float("nan")
            ),
            "source_file": str(args.input),
            "nrep": nrep,
            "ideal_path_hops": period,
            "sampling_seed": args.seed,
        }
        rows.append(row)
        print(
            f"{index:03d} {kind}: cluster={len(neighbors)}, evaluated={len(sources)}, tau={row['mean_relative_tortuosity']:.6f}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "tortuosity_by_configuration.csv", rows)
    summary_rows = []
    for kind in dict.fromkeys(r["structure_kind"] for r in rows):
        values = np.array(
            [r["mean_relative_tortuosity"] for r in rows if r["structure_kind"] == kind]
        )
        valid = values[np.isfinite(values)]
        stats = (
            summarize(valid)
            if len(valid)
            else {
                key: 0 if key == "n" else float("nan")
                for key in (
                    "n",
                    "mean",
                    "sample_std",
                    "standard_error_of_mean",
                    "mean_ci95_lower",
                    "mean_ci95_upper",
                    "min",
                    "median",
                    "max",
                    "percentile_2p5",
                    "percentile_97p5",
                )
            }
        )
        summary_rows.append(
            {"structure_kind": kind, "n_configurations": len(values), **stats}
        )
    write_rows(args.output_dir / "tortuosity_summary.csv", summary_rows)
    print(f"Wrote per-configuration and ensemble-summary CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
