#!/usr/bin/env python3
"""Build the single-column LiTiOF introductory structure/refinement figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter


HERE = Path(__file__).resolve().parent
STRUCTURE_PATH = HERE / "drx_cell.png"
REFINEMENT_PATH = HERE / "Li3TiO3F_1_50LF_obs_calc_diff.csv"
MAIN_TICKS_PATH = HERE / "Li3TiO3F_1_50LF_ticks_Fm-3m.txt"
SECONDARY_TICKS_PATH = HERE / "Li3TiO3F_1_50LF_ticks_C2-c.txt"
OUTPUT_STEM = HERE / "LiTiOF_structure_rietveld"

# Nature Communications figures use an 89 mm single-column width.
FIGURE_WIDTH_IN = 89.0 / 25.4
FIGURE_HEIGHT_IN = 3.62


def set_publication_style() -> None:
    """Use the compact typography and strokes shared by the LiTiOF figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.7,
            "ytick.minor.size": 1.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def load_refinement() -> tuple[np.ndarray, ...]:
    data = np.loadtxt(REFINEMENT_PATH, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected four refinement columns, got {data.shape}")
    two_theta, observed, calculated, difference = data.T
    if not np.allclose(observed - calculated, difference, atol=2.0e-3):
        raise ValueError("Column four is not observed minus calculated")
    if np.any(np.diff(two_theta) <= 0.0):
        raise ValueError("2theta values must increase monotonically")
    return two_theta, observed, calculated, difference


def load_bragg_positions(path: Path, x_min: float, x_max: float) -> np.ndarray:
    """Load 2theta (column six) from a phase-resolved reflection list."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"Expected at least six reflection columns in {path}")
    positions = np.asarray(data[:, 5], dtype=float)
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"Non-finite 2theta value in {path}")
    return positions[(positions >= x_min) & (positions <= x_max)]


def add_panel_label(ax: plt.Axes, label: str, *, image_panel: bool = False) -> None:
    ax.text(
        -0.115,
        0.985 if image_panel else 1.035,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
        zorder=10,
    )


def main() -> None:
    set_publication_style()
    structure = mpimg.imread(STRUCTURE_PATH)
    two_theta, observed, calculated, difference = load_refinement()
    main_bragg_positions = load_bragg_positions(
        MAIN_TICKS_PATH, float(two_theta[0]), float(two_theta[-1])
    )
    secondary_bragg_positions = load_bragg_positions(
        SECONDARY_TICKS_PATH, float(two_theta[0]), float(two_theta[-1])
    )

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.10),
        hspace=0.12,
        left=0.13,
        right=0.96,
        bottom=0.12,
        top=0.985,
    )

    # Panel a: use the supplied schematic without resampling or altering its colours.
    image_ax = fig.add_subplot(grid[0])
    image_ax.imshow(structure, interpolation="none", aspect="equal")
    image_ax.set_axis_off()
    add_panel_label(image_ax, "a", image_panel=True)

    # Panel b: conventional Rietveld profile with the residual offset below the fit.
    refinement_ax = fig.add_subplot(grid[1])
    fit_span = float(max(np.ptp(observed), np.ptp(calculated)))
    residual_offset = float(min(np.min(observed), np.min(calculated)) - 0.105 * fit_span)
    shifted_difference = difference + residual_offset

    refinement_ax.scatter(
        two_theta,
        observed,
        s=3.0,
        c="black",
        edgecolors="none",
        rasterized=True,
        label="Observed",
        zorder=2,
    )
    refinement_ax.plot(
        two_theta,
        calculated,
        color="#d73027",
        linewidth=0.75,
        label="Calculated",
        zorder=3,
    )
    refinement_ax.plot(
        two_theta,
        shifted_difference,
        color="#4575b4",
        linewidth=0.55,
        label="Difference",
        zorder=2,
    )
    refinement_ax.axhline(residual_offset, color="0.62", linewidth=0.45, zorder=1)

    # Conventional Rietveld reflection marks, positioned between the baseline
    # and the vertically offset difference curve.
    bragg_y_top = 0.18 * residual_offset
    bragg_y_bottom = 0.40 * residual_offset
    refinement_ax.vlines(
        main_bragg_positions,
        bragg_y_bottom,
        bragg_y_top,
        color="black",
        linewidth=0.55,
        zorder=3,
    )
    secondary_y_top = 0.47 * residual_offset
    secondary_y_bottom = 0.69 * residual_offset
    refinement_ax.vlines(
        secondary_bragg_positions,
        secondary_y_bottom,
        secondary_y_top,
        color="0.50",
        linewidth=0.55,
        zorder=3,
    )
    # Draw attention to the weak low-angle feature.
    peak_window = (two_theta >= 18.2) & (two_theta <= 18.8)
    weak_peak_y = float(np.max(observed[peak_window]))
    refinement_ax.annotate(
        "",
        xy=(18.49, weak_peak_y + 380.0),
        xytext=(18.49, weak_peak_y + 2280.0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "0.50",
            "linewidth": 0.8,
            "mutation_scale": 7.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=5,
    )

    y_min = float(np.min(shifted_difference) - 0.03 * fit_span)
    y_max = float(max(np.max(observed), np.max(calculated)) + 0.045 * fit_span)
    refinement_ax.set_xlim(float(two_theta[0]), float(two_theta[-1]))
    refinement_ax.set_ylim(y_min, y_max)
    refinement_ax.set_xlabel(r"$2\theta$ (°)")
    refinement_ax.set_ylabel("Intensity (arb. units)")
    refinement_ax.set_xticks(np.arange(10.0, 100.1, 10.0))
    refinement_ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    refinement_ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    refinement_ax.spines["top"].set_visible(False)
    refinement_ax.spines["right"].set_visible(False)
    refinement_ax.tick_params(top=False, right=False)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((0, 0))
    refinement_ax.yaxis.set_major_formatter(y_formatter)
    refinement_ax.yaxis.get_offset_text().set_fontsize(6.0)
    refinement_ax.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0.25,
        labelspacing=0.28,
        markerscale=2.0,
    )
    add_panel_label(refinement_ax, "b")

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"))
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600)
    plt.close(fig)

    metadata = "\n".join(
        [
            f"structure_source={STRUCTURE_PATH}",
            f"refinement_source={REFINEMENT_PATH}",
            "refinement_columns=2theta,observed,calculated,observed_minus_calculated",
            f"two_theta_range_degrees={two_theta[0]:.8g},{two_theta[-1]:.8g}",
            f"difference_vertical_offset={residual_offset:.12g}",
            f"main_phase_ticks_source={MAIN_TICKS_PATH}",
            f"main_phase_ticks_plotted={main_bragg_positions.size}",
            f"secondary_phase_ticks_source={SECONDARY_TICKS_PATH}",
            f"secondary_phase_ticks_plotted={secondary_bragg_positions.size}",
            "weak_peak_arrow_degrees=18.49",
            "figure_width_mm=89",
            f"figure_height_in={FIGURE_HEIGHT_IN}",
            "panel_order=a_structure_schematic,b_rietveld_refinement",
        ]
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        metadata + "\n", encoding="utf-8"
    )

    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
