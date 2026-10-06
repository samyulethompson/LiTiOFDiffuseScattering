#!/usr/bin/env python3
"""Plot 0-TM populations and connectivity for a rocksalt structure."""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.ticker import PercentFormatter


MODEL_COLOR = "#245C5A"
RANDOM_COLOR = "#9EC7C3"
CATION_BASIS = np.array(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16
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
    if np.any(tetra_to_cation < 0) or np.any(
        np.sum(corner_sites >= 0, axis=1) != 4
    ):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structure",
        type=Path,
        help="input structure readable by ASE",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("0tm_connectivity"),
        help="output path without an extension (default: ./0tm_connectivity)",
    )
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.random_samples < 1:
        parser.error("--random-samples must be at least 1")
    return args


def main() -> None:
    args = parse_args()

    atoms = read(args.structure)
    symbols = np.asarray(atoms.get_chemical_symbols())
    cation_symbols = symbols[np.isin(symbols, ["Li", "Ti"])]
    is_ti = cation_symbols == "Ti"
    n_cations = len(is_ti)
    n_ti = int(is_ti.sum())
    n_li = n_cations - n_ti
    nrep = round((n_cations / 4) ** (1 / 3))
    if 4 * nrep**3 != n_cations:
        raise RuntimeError(
            f"Cannot infer a cubic rocksalt supercell from {n_cations} cations"
        )
    period = 2 * nrep
    _, tetra_to_cation = cation_and_tetrahedron_tables(nrep)

    _, model_tm_fraction = tm_populations(is_ti, tetra_to_cation)
    model_zero, model_degree = zero_tm_and_degree(is_ti, tetra_to_cation, period)
    model_degree_fraction = np.bincount(model_degree, minlength=7) / model_zero.sum()

    random_tm_fraction = np.array(
        [
            comb(n_ti, k) * comb(n_li, 4 - k) / comb(n_cations, 4)
            for k in range(5)
        ]
    )

    rng = np.random.default_rng(args.seed)
    random_degree_fractions = np.empty((args.random_samples, 7), dtype=float)
    random_mean_degree = np.empty(args.random_samples, dtype=float)
    for sample in range(args.random_samples):
        shuffled = rng.permutation(is_ti)
        zero_tm, degree = zero_tm_and_degree(shuffled, tetra_to_cation, period)
        random_degree_fractions[sample] = (
            np.bincount(degree, minlength=7) / zero_tm.sum()
        )
        random_mean_degree[sample] = degree.mean()
    random_degree_mean = random_degree_fractions.mean(axis=0)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.504, 6.15), gridspec_kw={"hspace": 0.42}
    )

    x_tm = np.arange(5)
    width = 0.36
    ax_a.bar(
        x_tm - width / 2,
        random_tm_fraction,
        width,
        color=RANDOM_COLOR,
        label="Random",
        linewidth=0,
    )
    ax_a.bar(
        x_tm + width / 2,
        model_tm_fraction,
        width,
        color=MODEL_COLOR,
        label="Model",
        linewidth=0,
    )
    ax_a.set_xticks(x_tm, [f"{k}-TM" for k in range(5)])
    ax_a.set_xlabel("Tetrahedral environment")
    ax_a.set_ylabel("Proportion of tetrahedral sites")
    ax_a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_a.set_ylim(0, 0.43)
    ax_a.set_yticks(np.arange(0, 0.41, 0.1))
    ax_a.legend(frameon=False, loc="upper right", handlelength=1.2)

    x_degree = np.arange(7)
    ax_b.plot(
        x_degree,
        random_degree_mean,
        color=RANDOM_COLOR,
        marker="s",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.1,
        label="Random mean",
        zorder=3,
    )
    ax_b.plot(
        x_degree,
        model_degree_fraction,
        color=MODEL_COLOR,
        marker="o",
        markersize=4.0,
        markeredgewidth=0,
        linewidth=1.3,
        label="Model",
        zorder=4,
    )
    ax_b.set_xticks(x_degree)
    ax_b.set_xlabel("Number of neighboring 0-TM sites")
    ax_b.set_ylabel("Proportion of 0-TM sites")
    ax_b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_b.set_ylim(0, 0.35)
    ax_b.set_yticks(np.arange(0, 0.31, 0.1))
    ax_b.legend(frameon=False, loc="upper left", handlelength=1.7)

    for axis, panel in ((ax_a, "a"), (ax_b, "b")):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out")
        axis.text(
            -0.18,
            1.04,
            panel,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Fig. | Tetrahedral environments and 0-TM connectivity in the model. "
        "a, Proportions of tetrahedral sites coordinated by 0-4 transition-metal "
        "cations in the model and the exact fixed-composition random expectation. "
        "b, Conditional distribution of the number of nearest 0-TM tetrahedral "
        "neighbors around each 0-TM site. Neighbors are the six edge-sharing "
        "tetrahedral sites at a/2. The random curve shows the mean from "
        f"{args.random_samples:,} fixed-composition random occupational configurations."
    )
    args.output_stem.with_name(args.output_stem.name + "_caption.txt").write_text(
        caption + "\n"
    )

    population_csv = args.output_stem.with_name(
        args.output_stem.name + "_tm_populations.csv"
    )
    with population_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tm_neighbors", "model_fraction", "random_expected_fraction"])
        for k in range(5):
            writer.writerow([k, model_tm_fraction[k], random_tm_fraction[k]])

    connectivity_csv = args.output_stem.with_name(
        args.output_stem.name + "_connectivity.csv"
    )
    with connectivity_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "zero_tm_neighbors",
                "model_fraction_of_zero_tm_sites",
                "random_mean_fraction",
            ]
        )
        for k in range(7):
            writer.writerow([k, model_degree_fraction[k], random_degree_mean[k]])

    print(f"Model TM fractions: {model_tm_fraction}")
    print(f"Random TM fractions: {random_tm_fraction}")
    print(f"Model mean 0-TM degree: {model_degree.mean():.6f}")
    print(f"Random mean 0-TM degree: {random_mean_degree.mean():.6f}")
    print(
        f"Wrote {args.output_stem}.pdf, .svg, .png, _caption.txt, "
        "_tm_populations.csv and _connectivity.csv"
    )


if __name__ == "__main__":
    main()
