#!/usr/bin/env python3
"""Create a publication-ready 0.2 eV J1/J2 mixing-sweep figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SWEEP_ROOT = (
    HERE.parents[1]
    / "pair_interaction_nn_nnn_lifeo2_strength_map"
    / "low_strength_0p1_0p2_0p3_sweep"
)
OUTPUT_STEM = HERE / "mixing_sweep_0p2eV_100_plane"
LOCAL_ENVIRONMENT_CSV = SWEEP_ROOT / "local_environment_analysis" / "local_environment_counts.csv"

TOTAL_STRENGTH_EV = 0.2
TEMPERATURE_K = 1000
MIXING_VALUES = np.linspace(0.0, 1.0, 11)
ENVIRONMENT_MIXING_VALUES = (0.0, 0.5, 1.0)
# Sequential light-to-dark blue palette, ordered with the mixing fraction.
ENVIRONMENT_COLORS = ("#9ECAE1", "#4292C6", "#084594")
HIGHLIGHT_BORDER_WIDTH = 1.8
NORMALIZATION_PERCENTILE = 99.5


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 6.5,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def ratio_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def find_surface(mixing: float) -> Path:
    token = ratio_token(mixing)
    pattern = (
        f"*/results_lifeo2_unnorm_total_0p2_ratio_{token}_temp_1000K/"
        "scatty_structures/*_100vox_sc_list.txt"
    )
    matches = sorted(SWEEP_ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one 0.2 eV surface for lambda={mixing:g}, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_surface(path: Path) -> np.ndarray:
    data = np.loadtxt(path, usecols=(0, 1, 2, 3))
    tolerance = 1.0e-8
    keep = (
        (np.abs(data[:, 0]) < tolerance)
        & (data[:, 1] >= -tolerance)
        & (data[:, 1] <= 2.0 + tolerance)
        & (data[:, 2] >= -tolerance)
        & (data[:, 2] <= 2.0 + tolerance)
    )
    data = data[keep]
    k_values = np.unique(np.round(data[:, 1], 10))
    l_values = np.unique(np.round(data[:, 2], 10))
    image = np.full((len(l_values), len(k_values)), np.nan, dtype=float)
    for _, k_value, l_value, intensity in data:
        row = np.searchsorted(l_values, round(l_value, 10))
        column = np.searchsorted(k_values, round(k_value, 10))
        image[row, column] = intensity
    if image.shape != (101, 101):
        raise ValueError(f"Expected a 101x101 [100] surface, got {image.shape} from {path}")
    return image


def normalize_panel(image: np.ndarray) -> tuple[np.ndarray, float]:
    finite = image[np.isfinite(image)]
    limit = float(np.percentile(finite, NORMALIZATION_PERCENTILE))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Panel normalization limit is not positive")
    return np.clip(image / limit, 0.0, 1.0), limit


def load_environment_distribution(
    motif: str, motif_size: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load selected mixing fractions and the fixed-composition random fractions."""
    with LOCAL_ENVIRONMENT_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    observed = []
    random = None
    for mixing in ENVIRONMENT_MIXING_VALUES:
        selected = sorted(
            (
                row
                for row in rows
                if row["motif"] == motif
                and np.isclose(float(row["total_strength"]), TOTAL_STRENGTH_EV)
                and np.isclose(float(row["mixing_ratio"]), mixing)
            ),
            key=lambda row: int(row["n_fe"]),
        )
        if len(selected) != motif_size + 1:
            raise ValueError(
                f"Expected {motif_size + 1} {motif} populations at lambda={mixing:g}, "
                f"found {len(selected)}"
            )
        observed.append(np.array([float(row["observed_fraction"]) for row in selected]))
        if random is None:
            random = np.array([float(row["random_fraction"]) for row in selected])
    if random is None:
        raise ValueError(f"No random reference found for {motif}")
    return observed, random


def plot_environment_distribution(
    ax: plt.Axes,
    motif: str,
    motif_size: int,
    title: str,
) -> None:
    observed, random = load_environment_distribution(motif, motif_size)
    positions = np.arange(motif_size + 1, dtype=float)
    width = 0.24
    offsets = width * (np.arange(len(ENVIRONMENT_MIXING_VALUES)) - 1)
    for values, mixing, color, offset in zip(
        observed, ENVIRONMENT_MIXING_VALUES, ENVIRONMENT_COLORS, offsets
    ):
        ax.bar(
            positions + offset,
            values,
            width,
            color=color,
            label=rf"$\lambda={mixing:g}$",
            zorder=2,
        )

    random_mean = float(np.sum(positions * random))
    random_sigma = float(np.sqrt(np.sum((positions - random_mean) ** 2 * random)))
    curve_x = np.linspace(-0.5, motif_size + 0.5, 500)
    normal_curve = np.exp(-0.5 * ((curve_x - random_mean) / random_sigma) ** 2)
    normal_curve /= random_sigma * np.sqrt(2.0 * np.pi)
    ax.plot(
        curve_x,
        normal_curve,
        color="black",
        linewidth=1.1,
        label="Random reference",
        zorder=3,
    )

    ax.set_xlim(-0.5, motif_size + 0.5)
    ax.set_ylim(0.0, 0.92)
    ax.set_xticks(positions)
    ax.set_yticks(np.arange(0.0, 1.0, 0.2))
    ax.set_xlabel(r"M atoms per environment, $n_{\mathrm{M}}$")
    ax.set_title(title, pad=4.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    """Place labels with the same physical offset from each panel's left edge."""
    position = ax.get_position()
    fig.text(
        position.x0 - 0.030,
        position.y1 + 0.012,
        label,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    set_publication_style()
    paths = [find_surface(value) for value in MIXING_VALUES]
    raw_images = [load_surface(path) for path in paths]
    normalized_and_limits = [normalize_panel(image) for image in raw_images]
    images = [item[0] for item in normalized_and_limits]
    limits = [item[1] for item in normalized_and_limits]

    cmap = mpl.colormaps["gray"].copy()
    cmap.set_bad("black")
    fig = plt.figure(figsize=(7.20, 3.55))
    outer_grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.62, 0.20, 1.55],
        hspace=0.15,
        left=0.085,
        right=0.995,
        bottom=0.14,
        top=0.95,
    )
    scattering_grid = outer_grid[0].subgridspec(1, 11, wspace=0.025)
    axes = [fig.add_subplot(scattering_grid[0, index]) for index in range(11)]

    for index, (ax, image, mixing) in enumerate(zip(axes, images, MIXING_VALUES)):
        ax.imshow(
            image,
            origin="lower",
            extent=[0.0, 2.0, 0.0, 2.0],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
            rasterized=True,
        )
        ax.set_facecolor("black")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, 2.0)
        ax.set_title(rf"$\lambda={mixing:g}$", pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        matching_environment_index = next(
            (
                environment_index
                for environment_index, environment_mixing in enumerate(
                    ENVIRONMENT_MIXING_VALUES
                )
                if np.isclose(mixing, environment_mixing)
            ),
            None,
        )
        for spine in ax.spines.values():
            if matching_environment_index is None:
                spine.set_visible(False)
            else:
                spine.set_visible(True)
                spine.set_color(ENVIRONMENT_COLORS[matching_environment_index])
                spine.set_linewidth(HIGHLIGHT_BORDER_WIDTH)
                spine.set_zorder(5)

        if np.isclose(mixing, 0.0):
            ax.text(
                0.5,
                -0.10,
                r"$J_1$ dominated",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.5,
                color=ENVIRONMENT_COLORS[0],
                clip_on=False,
            )
        elif np.isclose(mixing, 1.0):
            ax.text(
                0.5,
                -0.10,
                r"$J_2$ dominated",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.5,
                color=ENVIRONMENT_COLORS[-1],
                clip_on=False,
            )

        if index < len(axes) - 1:
            ax.axvline(2.0, color="white", alpha=0.45, linewidth=0.25, clip_on=False)

    mixing_label_ax = fig.add_subplot(outer_grid[1])
    mixing_label_ax.set_axis_off()
    mixing_label_ax.text(
        0.5,
        0.68,
        r"Mixing fraction, $\lambda=J_2/(J_1+J_2)$",
        ha="center",
        va="center",
        fontsize=7.0,
    )

    environment_grid = outer_grid[2].subgridspec(1, 3, wspace=0.32)
    tetrahedral_ax = fig.add_subplot(environment_grid[0, 0])
    octahedral_ax = fig.add_subplot(environment_grid[0, 1], sharey=tetrahedral_ax)
    cubic_ax = fig.add_subplot(environment_grid[0, 2], sharey=tetrahedral_ax)
    plot_environment_distribution(
        tetrahedral_ax, "tetrahedron", 4, "Tetrahedral environments"
    )
    plot_environment_distribution(
        octahedral_ax, "octahedron", 6, "Octahedral environments"
    )
    plot_environment_distribution(cubic_ax, "cube", 8, "Cubic environments")
    tetrahedral_ax.set_ylabel("Fraction of environments")
    octahedral_ax.tick_params(labelleft=False)
    cubic_ax.tick_params(labelleft=False)
    tetrahedral_ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.06, 1.0),
        fontsize=6.0,
        handlelength=2.2,
        borderaxespad=0.2,
    )
    add_panel_label(fig, axes[0], "a")
    add_panel_label(fig, tetrahedral_ax, "b")
    add_panel_label(fig, octahedral_ax, "c")
    add_panel_label(fig, cubic_ax, "d")

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata = [
        f"sweep_root={SWEEP_ROOT}",
        f"total_strength_eV={TOTAL_STRENGTH_EV}",
        f"temperature_K={TEMPERATURE_K}",
        "mixing_definition=lambda=J2/(J1+J2)",
        "mixing_values=" + ",".join(f"{value:g}" for value in MIXING_VALUES),
        "plane=[100], h=0, 0<=k,l<=2 r.l.u.",
        f"normalization=independent_per_panel_{NORMALIZATION_PERCENTILE}th_percentile",
        "display_scale=linear_grayscale_0_to_1",
        f"local_environment_source={LOCAL_ENVIRONMENT_CSV}",
        "local_environment_motifs=tetrahedron,octahedron,cube",
        "local_environment_mixing_values="
        + ",".join(f"{value:g}" for value in ENVIRONMENT_MIXING_VALUES),
        "local_environment_reference=normal_approximation_to_fixed_composition_random_distribution",
    ]
    for mixing, path, limit in zip(MIXING_VALUES, paths, limits):
        metadata.append(f"lambda_{ratio_token(mixing)}_source={path}")
        metadata.append(
            f"lambda_{ratio_token(mixing)}_raw_normalization_limit={limit:.12g}"
        )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        "\n".join(metadata) + "\n"
    )

    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
