#!/usr/bin/env python3
"""Count Li/Fe tetrahedral, O-octahedral, and cubic environments in a sweep."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_RE = re.compile(
    r"results_lifeo2_(unnorm|norm)_total_([^_]+)_ratio_([^_]+)_temp_([^/]+)$"
)


def decode_number(token: str) -> float:
    return float(token.replace("p", "."))


def encode_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_cif_cations(path: Path) -> tuple[np.ndarray, int]:
    records = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] in {"Li", "Fe"}:
                records.append((fields[0], *map(float, fields[3:6])))

    n_cations = len(records)
    nrep = round((n_cations / 4.0) ** (1.0 / 3.0))
    if n_cations != 4 * nrep**3:
        raise ValueError(f"Expected 4*nrep^3 cations in {path}, found {n_cations}")

    period = 2 * nrep
    grid = np.full((period, period, period), -1, dtype=np.int8)
    for symbol, fx, fy, fz in records:
        x, y, z = np.rint(period * np.array([fx, fy, fz])).astype(int) % period
        if grid[x, y, z] >= 0:
            raise ValueError(f"Duplicate cation lattice coordinate in {path}")
        grid[x, y, z] = int(symbol == "Fe")
    return grid, nrep


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

    corners = np.array(
        [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    )
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
            math.comb(n_fe, count) * math.comb(n_sites - n_fe, motif_size - count)
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
                    "observed_over_random": fraction / random_fraction[count],
                }
            )
    return rows


def result_parameters(path: Path) -> tuple[str, float, float, str]:
    result_dir = path.parents[1].name
    match = RESULT_RE.match(result_dir)
    if not match:
        raise ValueError(f"Could not parse sweep parameters from {result_dir}")
    return (
        match.group(1),
        decode_number(match.group(2)),
        decode_number(match.group(3)),
        match.group(4),
    )


def find_inputs(root: Path, totals: list[float], snapshot: int) -> list[Path]:
    pattern = (
        "*/results_lifeo2_*_total_*_ratio_*_temp_*/scatty_structures/"
        f"rocks_lifeo2_pair_mc_atoms_{snapshot:02d}.cif"
    )
    inputs = [
        path
        for path in root.glob(pattern)
        if any(math.isclose(result_parameters(path)[1], total) for total in totals)
    ]
    inputs.sort(key=lambda path: result_parameters(path)[1:3])
    for total in totals:
        count = sum(math.isclose(result_parameters(path)[1], total) for path in inputs)
        if count != 11:
            raise RuntimeError(
                f"Expected 11 structures for total strength {total:g} eV, found {count}"
            )
    return inputs


def plot_heatmaps(rows: list[dict], output_dir: Path) -> list[Path]:
    outputs = []
    for total in sorted({float(row["total_strength"]) for row in rows}):
        subset = [row for row in rows if math.isclose(float(row["total_strength"]), total)]
        series = str(subset[0]["series"])
        ratios = sorted({float(row["mixing_ratio"]) for row in subset})
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
        for column, motif in enumerate(("tetrahedron", "octahedron", "cube")):
            motif_rows = [row for row in subset if row["motif"] == motif]
            size = int(motif_rows[0]["motif_size"])
            observed = np.zeros((size + 1, len(ratios)))
            enrichment = np.zeros_like(observed)
            for row in motif_rows:
                i = int(row["n_fe"])
                j = ratios.index(float(row["mixing_ratio"]))
                observed[i, j] = float(row["observed_fraction"])
                enrichment[i, j] = np.log2(
                    max(float(row["observed_over_random"]), 2.0**-4)
                )
            image = axes[0, column].imshow(
                observed, origin="lower", aspect="auto", cmap="viridis",
                extent=(-0.05, 1.05, -0.5, size + 0.5), vmin=0,
            )
            fig.colorbar(image, ax=axes[0, column], label="observed fraction")
            limit = min(4.0, max(0.25, float(np.nanmax(np.abs(enrichment)))))
            image = axes[1, column].imshow(
                enrichment, origin="lower", aspect="auto", cmap="coolwarm",
                extent=(-0.05, 1.05, -0.5, size + 0.5), vmin=-limit, vmax=limit,
            )
            fig.colorbar(image, ax=axes[1, column], label=r"$\log_2$(observed / random)")
            axes[0, column].set_title(f"{motif.capitalize()} ({size} cations)")
            for ax in axes[:, column]:
                ax.set_xticks(ratios)
                ax.set_yticks(range(size + 1))
                ax.set_xlabel("Mixing parameter, x")
                ax.set_ylabel("Fe atoms in environment")
        strength_label = (
            f"coordination-normalised A = {total:g} eV"
            if series == "norm"
            else f"unnormalised J₁ + J₂ = {total:g} eV"
        )
        fig.suptitle(
            f"Local Li/Fe environments; {strength_label}\n"
            "Random reference: exact fixed-composition mixing"
        )
        path = output_dir / f"local_environments_total_{encode_number(total)}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs.append(path)
    return outputs


def plot_distribution(
    rows: list[dict],
    output_dir: Path,
    motif: str,
    motif_size: int,
    ratios: tuple[float, ...],
    total: float,
) -> Path:
    selected = []
    for ratio in ratios:
        ratio_rows = sorted(
            (
                row for row in rows
                if row["motif"] == motif
                and math.isclose(float(row["total_strength"]), total)
                and math.isclose(float(row["mixing_ratio"]), ratio)
            ),
            key=lambda row: int(row["n_fe"]),
        )
        if len(ratio_rows) != motif_size + 1:
            raise ValueError(f"Incomplete {motif} populations at x={ratio:g}")
        selected.append(ratio_rows)

    x_values = np.arange(motif_size + 1)
    observed = [
        np.array([float(row["observed_fraction"]) for row in group])
        for group in selected
    ]
    random = np.array([float(row["random_fraction"]) for row in selected[0]])
    series = str(selected[0][0]["series"])
    mean = float(np.sum(x_values * random))
    sigma = float(np.sqrt(np.sum((x_values - mean) ** 2 * random)))
    curve_x = np.linspace(-0.5, motif_size + 0.5, 500)
    normal_curve = np.exp(-0.5 * ((curve_x - mean) / sigma) ** 2)
    normal_curve /= sigma * np.sqrt(2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(8 if motif_size == 4 else 11, 5.5))
    width = 0.26
    offsets = (np.arange(len(ratios)) - (len(ratios) - 1) / 2) * width
    colors = ("#3678b5", "#55a868", "#e1812c")
    for values, ratio, offset, color in zip(observed, ratios, offsets, colors):
        ax.bar(x_values + offset, values, width, label=f"Observed, x = {ratio:g}", color=color)
    ax.plot(curve_x, normal_curve, color="black", linewidth=2.2, label="Normal approximation")
    label = {"tetrahedron": "Tetrahedral", "octahedron": "Octahedral", "cube": "Cubic"}[motif]
    ax.set_xticks(x_values, [f"Li$_{{{motif_size-k}}}$Fe$_{{{k}}}$" for k in x_values])
    ax.set_xlabel(f"{label} cation environment")
    ax.set_ylabel(f"Fraction of {label.lower()} environments")
    strength_label = (
        f"coordination-normalised A = {total:g} eV"
        if series == "norm"
        else f"unnormalised J₁ + J₂ = {total:g} eV"
    )
    ax.set_title(f"{strength_label} {label.lower()} distributions")
    ax.set_ylim(0, 1.1 * max(*(np.max(values) for values in observed), np.max(normal_curve)))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    ratio_token = "_".join(encode_number(ratio) for ratio in ratios)
    path = output_dir / (
        f"{label.lower()}_distribution_total_{encode_number(total)}_"
        f"mixing_{ratio_token}_observed_vs_gaussian.png"
    )
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root directory containing the sweep runs")
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--totals", nargs="+", type=float, default=[0.2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / "local_environment_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cache: dict[int, dict[str, np.ndarray]] = {}
    for path in find_inputs(args.root, args.totals, args.snapshot):
        series, total, ratio, temperature = result_parameters(path)
        for row in analyze_file(path, cache):
            row.update(
                {
                    "result_dir": path.parents[1].name,
                    "series": series,
                    "snapshot": args.snapshot,
                    "total_strength": total,
                    "mixing_ratio": ratio,
                    "temperature": temperature,
                    "source_file": str(path.relative_to(args.root)),
                }
            )
            rows.append(row)

    fields = [
        "result_dir", "series", "snapshot", "total_strength", "mixing_ratio", "temperature",
        "motif", "motif_size", "n_fe", "count", "n_motifs", "observed_fraction",
        "random_fraction", "observed_over_random", "source_file",
    ]
    csv_path = output_dir / "local_environment_counts.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    plots = plot_heatmaps(rows, output_dir)
    for total in args.totals:
        plots.extend(
            [
                plot_distribution(rows, output_dir, "tetrahedron", 4, (0.0, 1.0), total),
                plot_distribution(rows, output_dir, "cube", 8, (0.0, 1.0), total),
                plot_distribution(rows, output_dir, "octahedron", 6, (0.0, 0.4, 1.0), total),
            ]
        )
    print(f"Wrote {csv_path}")
    for path in plots:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
