#!/usr/bin/env python3
"""Figure 4: ensemble tetrahedral-site populations.

Run with: python plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT_STEM = OUTPUT / "run81_0tm_natcommun_ensemble_panel_a_no_label"

SUMMARY = DATA / "ensemble_summary.csv"

MODEL_COLOR = "#245C5A"

RANDOM_COLOR = "#9EC7C3"


def load_summary() -> dict[tuple[str, str], dict[str, str]]:
    with SUMMARY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["structure_kind"], row["metric"]): row for row in rows}


def values(
    summary: dict[tuple[str, str], dict[str, str]], kind: str, metrics: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray([float(summary[kind, metric]["mean"]) for metric in metrics])
    sd = np.asarray([float(summary[kind, metric]["sample_std"]) for metric in metrics])
    return (mean, sd)


def main() -> None:
    summary = load_summary()
    tm_metrics = [f"tm{k}_fraction" for k in range(5)]
    model_tm, model_tm_sd = values(summary, "run71_mc", tm_metrics)
    random_tm, random_tm_sd = values(summary, "random", tm_metrics)
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
    fig, ax = plt.subplots(figsize=(3.504, 2.9))
    x_tm = np.arange(5)
    width = 0.36
    error_style = {
        "elinewidth": 0.8,
        "capsize": 2.0,
        "capthick": 0.8,
        "ecolor": "#333333",
    }
    ax.bar(
        x_tm - width / 2,
        random_tm,
        width,
        yerr=random_tm_sd,
        error_kw=error_style,
        color=RANDOM_COLOR,
        label="Random",
        linewidth=0,
    )
    ax.bar(
        x_tm + width / 2,
        model_tm,
        width,
        yerr=model_tm_sd,
        error_kw=error_style,
        color=MODEL_COLOR,
        label="Model",
        linewidth=0,
    )
    ax.set_xticks(x_tm, [f"{k}-TM" for k in range(5)])
    ax.set_xlabel("Tetrahedral environment")
    ax.set_ylabel("Proportion of tetrahedral sites")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.43)
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_STEM}.png, .pdf, and .svg")


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    main()
