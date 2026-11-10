#!/usr/bin/env python3
"""Periodic percolation of 0-TM tetrahedral and Li diffusion networks.

Wrapping is detected from nonzero lattice translations around graph cycles.
The accessible fraction is the fraction of active nodes in any wrapping cluster.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from count_0tm_sites import cation_and_tetrahedron_tables, read_occupations


class PeriodicDSU:
    """Union-find with integer coordinates and directional wrap detection."""

    def __init__(self, n: int, period: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)
        # coordinate(node) - coordinate(parent)
        self.delta = np.zeros((n, 3), dtype=np.int32)
        self.wrap_axes = np.zeros((n, 3), dtype=bool)
        self.period = period

    def find(self, node: int) -> tuple[int, np.ndarray]:
        parent = int(self.parent[node])
        if parent == node:
            return node, np.zeros(3, dtype=np.int32)
        root, parent_delta = self.find(parent)
        self.delta[node] += parent_delta
        self.parent[node] = root
        return root, self.delta[node].copy()

    def union(self, a: int, b: int, displacement_a_to_b: np.ndarray) -> None:
        """Impose coordinate(b) = coordinate(a) + displacement_a_to_b."""
        root_a, delta_a = self.find(a)
        root_b, delta_b = self.find(b)
        displacement = np.asarray(displacement_a_to_b, dtype=np.int32)
        if root_a == root_b:
            cycle = delta_a + displacement - delta_b
            if np.any(cycle):
                if np.any(cycle % self.period):
                    raise RuntimeError(f"Non-periodic inconsistent cycle: {cycle}")
                self.wrap_axes[root_a] |= cycle != 0
            return

        combined_wrap = self.wrap_axes[root_a] | self.wrap_axes[root_b]
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
            self.delta[root_a] = delta_b - delta_a - displacement
            self.wrap_axes[root_b] = combined_wrap
        else:
            self.parent[root_b] = root_a
            self.delta[root_b] = delta_a + displacement - delta_b
            self.wrap_axes[root_a] = combined_wrap
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_a] += 1


def summarize_dsu(
    dsu: PeriodicDSU, active: np.ndarray
) -> tuple[dict[str, object], list[dict[str, object]]]:
    active_ids = np.flatnonzero(active)
    roots = np.array([dsu.find(int(node))[0] for node in active_ids], dtype=np.int32)
    unique_roots, sizes = np.unique(roots, return_counts=True)
    order = np.argsort(sizes)[::-1]
    unique_roots = unique_roots[order]
    sizes = sizes[order]
    cluster_rows = []
    wrapping_sizes = []
    total_wrap_axes = np.zeros(3, dtype=bool)
    for rank, (root, size) in enumerate(zip(unique_roots, sizes), start=1):
        axes = dsu.wrap_axes[int(root)].copy()
        if np.any(axes):
            wrapping_sizes.append(int(size))
            total_wrap_axes |= axes
        cluster_rows.append(
            {
                "rank": rank,
                "size": int(size),
                "fraction_of_active_sites": float(size / len(active_ids)),
                "wrap_x": bool(axes[0]),
                "wrap_y": bool(axes[1]),
                "wrap_z": bool(axes[2]),
            }
        )
    largest = int(sizes[0]) if len(sizes) else 0
    accessible = int(sum(wrapping_sizes))
    summary = {
        "active_sites": int(len(active_ids)),
        "n_clusters": int(len(sizes)),
        "largest_cluster": largest,
        "largest_cluster_fraction": (
            float(largest / len(active_ids)) if len(active_ids) else 0.0
        ),
        "wrap_x": bool(total_wrap_axes[0]),
        "wrap_y": bool(total_wrap_axes[1]),
        "wrap_z": bool(total_wrap_axes[2]),
        "wraps_all_directions": bool(np.all(total_wrap_axes)),
        "n_wrapping_clusters": int(len(wrapping_sizes)),
        "wrapping_cluster_sizes": ";".join(map(str, wrapping_sizes)),
        "accessible_sites": accessible,
        "accessible_fraction": (
            float(accessible / len(active_ids)) if len(active_ids) else 0.0
        ),
    }
    return summary, cluster_rows


def li_graph_stats(
    is_ti: np.ndarray,
    tetra_to_cation: np.ndarray,
    zero_tm: np.ndarray,
    cation_lookup: np.ndarray,
    period: int,
):
    dsu = PeriodicDSU(len(is_ti), period)
    cation_coordinates = np.empty((len(is_ti), 3), dtype=np.int32)
    grid_coordinates = np.argwhere(cation_lookup >= 0)
    cation_coordinates[cation_lookup[tuple(grid_coordinates.T)]] = grid_coordinates
    active_tetrahedra = tetra_to_cation[zero_tm]
    for col_a, col_b in combinations(range(4), 2):
        a_values = active_tetrahedra[:, col_a]
        b_values = active_tetrahedra[:, col_b]
        raw_displacements = cation_coordinates[b_values] - cation_coordinates[a_values]
        displacements = raw_displacements - period * np.rint(
            raw_displacements / period
        ).astype(np.int32)
        for a, b, displacement in zip(a_values, b_values, displacements):
            dsu.union(int(a), int(b), displacement)
    return summarize_dsu(dsu, ~is_ti)


def tetra_dsu(zero_tm: np.ndarray, period: int) -> PeriodicDSU:
    """Join the six edge-sharing neighbours of each active tetrahedral site."""
    dsu = PeriodicDSU(len(zero_tm), period)
    site_ids = np.arange(len(zero_tm), dtype=np.int32).reshape((period,) * 3)
    active = zero_tm.reshape((period,) * 3)
    for axis in range(3):
        neighbors = np.roll(site_ids, -1, axis=axis)
        connected = active & np.roll(active, -1, axis=axis)
        displacement = np.zeros(3, dtype=np.int32)
        displacement[axis] = 1
        for a, b in zip(site_ids[connected], neighbors[connected]):
            dsu.union(int(a), int(b), displacement)
    return dsu


def tetra_graph_stats(zero_tm: np.ndarray, period: int):
    return summarize_dsu(tetra_dsu(zero_tm, period), zero_tm)


def iter_configurations(path: Path, max_configs: int | None = None, frame: int = 0):
    """Yield (index, kind, Ti occupations, nrep) in Run 81 sampling order."""
    if path.suffix.lower() != ".npz":
        if max_configs not in (None, 1):
            raise ValueError(
                "--max-configs applies to a packed ensemble; use --frame for a structure file."
            )
        is_ti, nrep = read_occupations(path, frame)
        yield frame, "structure", is_ti, nrep
        return
    if frame != 0:
        raise ValueError("--frame applies to structure files, not packed ensembles.")
    with np.load(path, allow_pickle=False) as data:
        n_cations = int(data["n_cations"])
        nrep = round((n_cations / 4) ** (1 / 3))
        if nrep < 1 or n_cations != 4 * nrep**3:
            raise ValueError("Expected 4*nrep^3 cations in the ensemble.")
        mc, random = data["mc"], data["random"]
        if (
            mc.ndim != 2
            or mc.shape != random.shape
            or mc.shape[1] != (n_cations + 7) // 8
        ):
            raise ValueError(
                "Packed mc and random arrays must have matching configuration counts and cation lengths."
            )
        if mc.dtype != np.uint8 or random.dtype != np.uint8 or len(mc) == 0:
            raise ValueError("Packed occupations must be nonempty uint8 arrays.")
        count = len(mc) if max_configs is None else min(max_configs, len(mc))
        for index in range(count):
            for kind, packed in (("run71_mc", mc[index]), ("random", random[index])):
                is_ti = np.unpackbits(packed, bitorder="little")[:n_cations].astype(
                    bool
                )
                yield index, kind, is_ti, nrep


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--output-dir", type=Path, default=Path("results/percolation"))
    args = parser.parse_args()
    if args.max_configs is not None and args.max_configs < 1:
        parser.error("--max-configs must be positive.")
    summaries, clusters = [], []
    cached_tables = {}
    for index, kind, is_ti, nrep in iter_configurations(
        args.input, args.max_configs, args.frame
    ):
        if nrep not in cached_tables:
            cached_tables[nrep] = cation_and_tetrahedron_tables(nrep)
        lookup, tetra = cached_tables[nrep]
        period = 2 * nrep
        zero_tm = ~np.any(is_ti[tetra], axis=1)
        networks = {
            "tetra_0tm": tetra_graph_stats(zero_tm, period),
            "li_0tm": li_graph_stats(is_ti, tetra, zero_tm, lookup, period),
        }
        for network, (summary, cluster_rows) in networks.items():
            identity = {
                "source_file": str(args.input),
                "index": index,
                "structure_kind": kind,
                "network": network,
            }
            summaries.append({**identity, **summary})
            clusters.extend({**identity, **row} for row in cluster_rows)
        print(
            f"{index:03d} {kind}: tetrahedral accessible fraction={networks['tetra_0tm'][0]['accessible_fraction']:.6f}; Li accessible fraction={networks['li_0tm'][0]['accessible_fraction']:.6f}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "percolation.csv", summaries)
    cluster_path = args.output_dir / "cluster_sizes.csv"
    if clusters:
        write_rows(cluster_path, clusters)
    else:
        with cluster_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "source_file",
                    "index",
                    "structure_kind",
                    "network",
                    "rank",
                    "size",
                    "fraction_of_active_sites",
                    "wrap_x",
                    "wrap_y",
                    "wrap_z",
                ]
            )
    numeric_fields = [
        key
        for key in summaries[0]
        if key
        not in {
            "source_file",
            "index",
            "structure_kind",
            "network",
            "wrapping_cluster_sizes",
        }
    ]
    ensemble_rows = []
    for kind in dict.fromkeys(row["structure_kind"] for row in summaries):
        for network in ("tetra_0tm", "li_0tm"):
            selected = [
                r
                for r in summaries
                if r["structure_kind"] == kind and r["network"] == network
            ]
            for field in numeric_fields:
                values = np.array([r[field] for r in selected], dtype=float)
                ensemble_rows.append(
                    {
                        "structure_kind": kind,
                        "network": network,
                        "metric": field,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "sample_std": (
                            float(values.std(ddof=1))
                            if len(values) > 1
                            else float("nan")
                        ),
                    }
                )
    write_rows(args.output_dir / "ensemble_summary.csv", ensemble_rows)
    print(
        f"Wrote percolation, cluster-size, and ensemble-summary CSVs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
