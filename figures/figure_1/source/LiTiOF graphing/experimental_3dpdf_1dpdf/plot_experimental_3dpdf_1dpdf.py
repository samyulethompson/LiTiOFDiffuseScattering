#!/usr/bin/env python3
"""Create a publication-ready experimental 3D-/1D-deltaPDF five-panel figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import Akima1DInterpolator


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / "LTiOF_dataset"
XY0_PATH = DATA_ROOT / "3ddpdf_xy_slices_30uc_scale" / "3ddpdf_xy0.npy"
XY0P5_PATH = DATA_ROOT / "3ddpdf_xy_slices_30uc_scale" / "3ddpdf_xy0p5.npy"
PROFILE_ROOT = DATA_ROOT / "experimental_1d_deltapdf"
REFINEMENT_PATH = HERE.parent / "LiTiOF_1DPDF_refinement.txt"
OUTPUT_STEM = HERE / "experimental_xy0_100_110_refinement"

XY_LIMIT = 3.0
PROFILE_MIN = 0.0
PROFILE_MAX = 3.0
LATTICE_CONSTANT_ANG = 4.1
PIXELS_PER_UNIT_CELL = 24.0
IMAGE_PERCENTILE = 99.5
REFINEMENT_R_MIN = 1.00
REFINEMENT_R_STEP = 0.02
REFINEMENT_R_MAX = 10.00

POSITIVE_CMAP = LinearSegmentedColormap.from_list(
    "white_to_red", ["#ffffff", "#d73027"]
)
NEGATIVE_CMAP = LinearSegmentedColormap.from_list(
    "blue_to_white", ["#4575b4", "#ffffff"]
)


def set_publication_style() -> None:
    """Use a compact, clean style suitable for a Nature-family manuscript."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
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
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def load_xy_slice(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.load(path)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected a square 2D xy0 array, got {image.shape}")
    half_width = (image.shape[0] - 1) / (2.0 * PIXELS_PER_UNIT_CELL)
    coordinate = np.linspace(-half_width, half_width, image.shape[0])
    keep = (coordinate >= -XY_LIMIT) & (coordinate <= XY_LIMIT)
    return coordinate[keep], image[np.ix_(keep, keep)]


def plot_xy_slice(
    ax: plt.Axes,
    coordinate: np.ndarray,
    values: np.ndarray,
    title: str,
) -> tuple[plt.AxesImage, float]:
    """Plot one experimental xy slice with its own robust symmetric scale."""
    finite = values[np.isfinite(values)]
    image_limit = float(np.percentile(np.abs(finite), IMAGE_PERCENTILE))
    image = ax.imshow(
        values,
        origin="lower",
        extent=[coordinate[0], coordinate[-1], coordinate[0], coordinate[-1]],
        cmap="RdBu_r",
        vmin=-image_limit,
        vmax=image_limit,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set(
        xlim=(-XY_LIMIT, XY_LIMIT),
        ylim=(-XY_LIMIT, XY_LIMIT),
        xticks=np.arange(-3, 4, 1),
        yticks=np.arange(-3, 4, 1),
        xlabel=r"$x$ (unit cells)",
        ylabel=r"$y$ (unit cells)",
        title=title,
    )
    ax.set_xticks(np.arange(-3, 3.1, 0.5), minor=True)
    ax.set_yticks(np.arange(-3, 3.1, 0.5), minor=True)
    ax.set_box_aspect(1)
    return image, image_limit


def add_direction_vectors(ax: plt.Axes) -> None:
    """Mark the [100] and [110] directions used for panels b and c."""
    arrow = {
        "arrowstyle": "-|>",
        "color": "black",
        "lw": 0.75,
        "mutation_scale": 7,
    }
    label = {
        "fontsize": 6.2,
        "color": "black",
        "bbox": {"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.5},
    }
    ax.annotate("", xy=(1.65, 0.0), xytext=(0.0, 0.0), arrowprops=arrow, zorder=5)
    ax.annotate("", xy=(1.17, 1.17), xytext=(0.0, 0.0), arrowprops=arrow, zorder=5)
    ax.text(1.72, -0.12, r"$[100]$", ha="left", va="top", zorder=6, **label)
    ax.text(1.23, 1.23, r"$[110]$", ha="left", va="bottom", zorder=6, **label)


def load_profile(direction: str) -> tuple[np.ndarray, np.ndarray]:
    path = PROFILE_ROOT / f"experimental_1d_deltapdf_{direction}.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    distance_lattice = np.asarray(data["signed_distance_lattice_units"], dtype=float)
    distance = np.asarray(data["signed_distance_angstrom"], dtype=float)
    intensity = np.asarray(data["deltaPDF_intensity_raw"], dtype=float)
    keep = (distance_lattice >= PROFILE_MIN) & (distance_lattice <= PROFILE_MAX)
    return distance[keep], intensity[keep]


def load_refinement() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load observed, calculated, and difference curves on the implicit r grid."""
    data = np.loadtxt(REFINEMENT_PATH)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            f"Expected three refinement columns (observed, calculated, difference), "
            f"got {data.shape}"
        )
    if not np.allclose(data[:, 0] - data[:, 1], data[:, 2], atol=2.0e-6):
        raise ValueError("Refinement column 3 is not observed minus calculated")

    # The exported file omits r. Its 2,951 rows correspond to 1.00--60.00 A
    # sampled every 0.02 A; retain the requested low-r through 15 A region.
    distance = REFINEMENT_R_MIN + REFINEMENT_R_STEP * np.arange(data.shape[0])
    keep = distance <= REFINEMENT_R_MAX + np.finfo(float).eps
    return distance[keep], data[keep, 0], data[keep, 1], data[keep, 2]


def add_signed_gradient(
    ax: plt.Axes,
    distance: np.ndarray,
    intensity: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Fill a profile with white-to-red/blue gradients clipped by the curve."""
    nx, ny = 1200, 700
    x_dense = np.linspace(x_min, x_max, nx)
    y_dense = np.linspace(y_min, y_max, ny)
    interpolator = Akima1DInterpolator(distance, intensity)
    curve = np.asarray(interpolator(x_dense), dtype=float)

    rgba = np.zeros((ny, nx, 4), dtype=float)
    positive_norm = Normalize(vmin=0.0, vmax=max(y_max, np.finfo(float).eps))
    negative_norm = Normalize(vmin=y_min, vmax=0.0)
    positive_rows = y_dense >= 0.0
    negative_rows = ~positive_rows
    rgba[positive_rows, :, :3] = POSITIVE_CMAP(positive_norm(y_dense[positive_rows]))[:, None, :3]
    rgba[negative_rows, :, :3] = NEGATIVE_CMAP(negative_norm(y_dense[negative_rows]))[:, None, :3]

    yy = y_dense[:, None]
    cc = curve[None, :]
    inside = ((cc >= 0.0) & (yy >= 0.0) & (yy <= cc)) | (
        (cc < 0.0) & (yy <= 0.0) & (yy >= cc)
    )
    rgba[:, :, 3] = inside.astype(float)

    ax.imshow(
        rgba,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="bilinear",
        aspect="auto",
        zorder=1,
    )
    ax.plot(x_dense, curve, color="black", linewidth=0.65, zorder=3)
    ax.scatter(
        distance,
        intensity,
        s=1.2,
        c="black",
        edgecolors="none",
        zorder=4,
        clip_on=True,
    )
    ax.axhline(0.0, color="0.30", linewidth=0.55, zorder=2)


def format_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.tick_params(top=False, right=False)
    ax.set_box_aspect(1)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.18) -> None:
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def main() -> None:
    set_publication_style()
    coordinate, xy0 = load_xy_slice(XY0_PATH)
    coordinate_0p5, xy0p5 = load_xy_slice(XY0P5_PATH)
    r100, pdf100 = load_profile("100")
    r110, pdf110 = load_profile("110")
    fit_r, observed, calculated, difference = load_refinement()

    combined = np.concatenate([pdf100[np.isfinite(pdf100)], pdf110[np.isfinite(pdf110)]])
    normalization_factor = float(np.max(np.abs(combined)))
    if not np.isfinite(normalization_factor) or normalization_factor <= 0.0:
        raise ValueError("Cannot normalize empty or zero-valued 1D profiles")
    pdf100 = pdf100 / normalization_factor
    pdf110 = pdf110 / normalization_factor
    combined = combined / normalization_factor
    data_min = min(float(np.min(combined)), 0.0)
    data_max = max(float(np.max(combined)), 0.0)
    padding = 0.055 * (data_max - data_min)
    y_min, y_max = data_min - padding, data_max + padding

    fig = plt.figure(figsize=(7.20, 5.25), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.015, hspace=0.01)
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    xy0p5_ax = fig.add_subplot(grid[1, 0])
    fit_ax = fig.add_subplot(grid[1, 1:])

    image, image_limit = plot_xy_slice(
        axes[0],
        coordinate,
        xy0,
        r"Experimental 3D-$\Delta$PDF, $xy0$",
    )
    add_direction_vectors(axes[0])
    _, image_limit_0p5 = plot_xy_slice(
        xy0p5_ax,
        coordinate_0p5,
        xy0p5,
        r"Experimental 3D-$\Delta$PDF, $xy0.5$",
    )
    colorbar = fig.colorbar(
        image,
        ax=axes[0],
        orientation="vertical",
        location="right",
        fraction=0.050,
        pad=0.035,
        shrink=0.70,
        aspect=16,
        ticks=[-image_limit, 0.0, image_limit],
    )
    colorbar.ax.set_yticklabels([r"$-1$", "0", r"$+1$"])
    colorbar.ax.tick_params(length=2, width=0.55, labelsize=5.5, pad=1.5)
    colorbar.outline.set_linewidth(0.55)

    for ax, distance, intensity, direction in (
        (axes[1], r100, pdf100, "100"),
        (axes[2], r110, pdf110, "110"),
    ):
        profile_min_ang = PROFILE_MIN * LATTICE_CONSTANT_ANG
        profile_max_ang = PROFILE_MAX * LATTICE_CONSTANT_ANG
        lattice_spacing_ang = LATTICE_CONSTANT_ANG * (
            1.0 / np.sqrt(2.0) if direction == "110" else 0.5
        )
        lattice_points_ang = np.arange(
            0.0,
            profile_max_ang + np.finfo(float).eps,
            lattice_spacing_ang,
        )
        for lattice_point in lattice_points_ang:
            ax.axvline(
                lattice_point,
                color="0.78",
                linewidth=0.55,
                zorder=2,
            )
        add_signed_gradient(
            ax,
            distance,
            intensity,
            profile_min_ang,
            profile_max_ang,
            y_min,
            y_max,
        )
        format_axes(ax)
        ax.set_xlim(profile_min_ang, profile_max_ang)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(0.0, profile_max_ang + 0.1, 4.0))
        ax.set_xticks(np.arange(0.0, profile_max_ang + 0.1, 1.0), minor=True)
        ax.set_xlabel(r"$r$ ($\AA$)")
        ax.set_title(rf"Experimental 3D-$\Delta$PDF cut, $[{direction}]$")

    for ax in axes[1:]:
        ax.set_ylabel(r"Normalized $\Delta$PDF intensity")

    for ax, label in zip(axes, ("a", "b", "c")):
        add_panel_label(ax, label)
    add_panel_label(xy0p5_ax, "d")

    fit_span = float(max(np.ptp(observed), np.ptp(calculated)))
    residual_offset = float(min(np.min(observed), np.min(calculated)) - 0.20 * fit_span)
    shifted_difference = difference + residual_offset
    fit_ax.plot(
        fit_r,
        observed,
        color="black",
        linewidth=0.65,
        label="Observed",
        zorder=2,
    )
    fit_ax.plot(
        fit_r,
        calculated,
        color="#d73027",
        linewidth=0.85,
        label="Calculated",
        zorder=3,
    )
    fit_ax.plot(
        fit_r,
        shifted_difference,
        color="#4575b4",
        linewidth=0.65,
        label="Difference (offset)",
        zorder=2,
    )
    fit_ax.axhline(residual_offset, color="0.55", linewidth=0.45, zorder=1)
    fit_ax.set(
        xlim=(REFINEMENT_R_MIN, REFINEMENT_R_MAX),
        xlabel=r"$r$ ($\AA$)",
        ylabel=r"$G(r)$ (arb. units)",
    )
    fit_ax.set_title("Experimental 1D-PDF refinement", pad=18.0)
    fit_ax.set_xticks(np.arange(REFINEMENT_R_MIN, REFINEMENT_R_MAX + 0.1, 2.0))
    fit_ax.set_xticks(np.arange(1.0, REFINEMENT_R_MAX + 0.1, 0.5), minor=True)
    fit_ax.set_box_aspect(0.43)
    fit_ax.spines["top"].set_visible(False)
    fit_ax.spines["right"].set_visible(False)
    fit_ax.tick_params(top=False, right=False)
    fit_ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.7,
        handletextpad=0.8,
        columnspacing=2.4,
        borderaxespad=0.0,
    )
    add_panel_label(fit_ax, "e", x=-0.08)

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata = "\n".join(
        [
            f"xy0_source={XY0_PATH}",
            f"xy0p5_source={XY0P5_PATH}",
            f"profile_100_source={PROFILE_ROOT / 'experimental_1d_deltapdf_100.csv'}",
            f"profile_110_source={PROFILE_ROOT / 'experimental_1d_deltapdf_110.csv'}",
            f"refinement_source={REFINEMENT_PATH}",
            f"xy_display_range_unit_cells=-{XY_LIMIT},+{XY_LIMIT}",
            f"profile_display_range_unit_cells={PROFILE_MIN},{PROFILE_MAX}",
            f"profile_display_range_angstrom={PROFILE_MIN * LATTICE_CONSTANT_ANG},{PROFILE_MAX * LATTICE_CONSTANT_ANG}",
            f"profile_lattice_constant_angstrom={LATTICE_CONSTANT_ANG}",
            "profile_lattice_lines_100_angstrom=half_integer_multiples_of_lattice_constant",
            "profile_lattice_lines_110_angstrom=integer_multiples_of_lattice_constant_over_sqrt(2)",
            f"xy_symmetric_color_limit_raw={image_limit:.12g}",
            f"xy0p5_symmetric_color_limit_raw={image_limit_0p5:.12g}",
            f"xy_color_limit_percentile_abs={IMAGE_PERCENTILE}",
            "xy_slices_normalized_independently",
            "xy0_direction_vectors=100,110",
            f"profile_shared_normalization_factor_raw={normalization_factor:.12g}",
            "profile_normalization=max_abs_across_100_and_110_in_displayed_range",
            "profile_y_limits=shared_between_100_and_110",
            "profile_points=raw_samples_shown_as_black_dots",
            "profile_line=Akima_interpolating_spline",
            "positive_fill=white_at_zero_to_red",
            "negative_fill=white_at_zero_to_blue",
            "refinement_columns=observed,calculated,observed_minus_calculated",
            f"refinement_implicit_r_min_angstrom={REFINEMENT_R_MIN}",
            f"refinement_implicit_r_step_angstrom={REFINEMENT_R_STEP}",
            f"refinement_display_r_max_angstrom={REFINEMENT_R_MAX}",
            f"refinement_difference_vertical_offset={residual_offset:.12g}",
        ]
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(metadata + "\n")

    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
