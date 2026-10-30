#!/usr/bin/env python3
"""Recreate the Run-71 Nat Commun 0-TM figure using Run-81 ensembles."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT.parents[1] / "3d-deltaPDF_informed/run81/ensemble_summary.csv"
OUTPUT_STEM = ROOT / "run81_0tm_natcommun_ensemble"

MODEL_COLOR = "#245C5A"
RANDOM_COLOR = "#9EC7C3"


def load_summary() -> dict[tuple[str, str], dict[str, str]]:
    with SUMMARY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["structure_kind"], row["metric"]): row for row in rows}


def values(
    summary: dict[tuple[str, str], dict[str, str]],
    kind: str,
    metrics: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray([float(summary[kind, metric]["mean"]) for metric in metrics])
    sd = np.asarray(
        [float(summary[kind, metric]["sample_std"]) for metric in metrics]
    )
    return mean, sd


def write_data(
    path: Path,
    coordinate_name: str,
    coordinates: np.ndarray,
    model_mean: np.ndarray,
    model_sd: np.ndarray,
    random_mean: np.ndarray,
    random_sd: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                coordinate_name,
                "model_mean",
                "model_sample_sd",
                "random_mean",
                "random_sample_sd",
            ]
        )
        for row in zip(coordinates, model_mean, model_sd, random_mean, random_sd):
            writer.writerow(row)


def main() -> None:
    summary = load_summary()
    tm_metrics = [f"tm{k}_fraction" for k in range(5)]
    degree_metrics = [f"zero_tm_degree_{k}_fraction" for k in range(7)]
    model_tm, model_tm_sd = values(summary, "run71_mc", tm_metrics)
    random_tm, random_tm_sd = values(summary, "random", tm_metrics)
    model_degree, model_degree_sd = values(summary, "run71_mc", degree_metrics)
    random_degree, random_degree_sd = values(summary, "random", degree_metrics)

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

    # Match the original 89-mm-wide, two-panel Run-71 figure.
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.504, 6.15), gridspec_kw={"hspace": 0.42}
    )

    x_tm = np.arange(5)
    width = 0.36
    error_style = {
        "elinewidth": 0.8,
        "capsize": 2.0,
        "capthick": 0.8,
        "ecolor": "#333333",
    }
    ax_a.bar(
        x_tm - width / 2,
        random_tm,
        width,
        yerr=random_tm_sd,
        error_kw=error_style,
        color=RANDOM_COLOR,
        label="Random",
        linewidth=0,
    )
    ax_a.bar(
        x_tm + width / 2,
        model_tm,
        width,
        yerr=model_tm_sd,
        error_kw=error_style,
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
    ax_b.errorbar(
        x_degree,
        random_degree,
        yerr=random_degree_sd,
        color=RANDOM_COLOR,
        marker="s",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.1,
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.8,
        label="Random mean",
        zorder=3,
    )
    ax_b.errorbar(
        x_degree,
        model_degree,
        yerr=model_degree_sd,
        color=MODEL_COLOR,
        marker="o",
        markersize=4.0,
        markeredgewidth=0,
        linewidth=1.3,
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.8,
        label="Model mean",
        zorder=4,
    )
    ax_b.set_xticks(x_degree)
    ax_b.set_xlabel("Number of neighboring 0-TM sites")
    ax_b.set_ylabel("Proportion of 0-TM sites")
    ax_b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_b.set_ylim(0, 0.35)
    ax_b.set_yticks(np.arange(0, 0.31, 0.1))
    ax_b.legend(frameon=False, loc="upper left", handlelength=1.7)

    for ax, panel in ((ax_a, "a"), (ax_b, "b")):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(
            -0.18,
            1.04,
            panel,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_data(
        OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_tm_populations.csv"),
        "tm_neighbors",
        x_tm,
        model_tm,
        model_tm_sd,
        random_tm,
        random_tm_sd,
    )
    write_data(
        OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_connectivity.csv"),
        "zero_tm_neighbors",
        x_degree,
        model_degree,
        model_degree_sd,
        random_degree,
        random_degree_sd,
    )
    caption = (
        "Fig. | Ensemble-averaged tetrahedral environments and 0-TM connectivity. "
        "a, Mean proportions of tetrahedral sites coordinated by 0-4 transition-"
        "metal cations in 100 independent Run-71 occupational models and 100 "
        "fixed-composition random configurations. b, Mean conditional distribution "
        "of the number of nearest 0-TM tetrahedral neighbors around each 0-TM site. "
        "Neighbors are the six edge-sharing tetrahedral sites at a/2. Error bars in "
        "both panels are sample standard deviations across the 100 configurations."
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_caption.txt").write_text(
        caption + "\n"
    )
    print(f"Wrote {OUTPUT_STEM}.png, .pdf, .svg and source-data CSV files")


if __name__ == "__main__":
    main()
