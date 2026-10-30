#!/usr/bin/env python3
"""Figure 1: experimental overview and integrated 3D-deltaPDF.

Run with: python plot.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import h5py
from scipy.ndimage import rotate
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT_STEM = (
    OUTPUT / "LiTiOF_intro_experimental_overview_3x3_integrated_model_3dpdf_transposed"
)

PXRD_REFINEMENT_PATH = DATA / "Li3TiO3F_1_50LF_obs_calc_diff.csv"

PXRD_MAIN_TICKS_PATH = DATA / "Li3TiO3F_1_50LF_ticks_Fm-3m.txt"

PXRD_SECONDARY_TICKS_PATH = DATA / "Li3TiO3F_1_50LF_ticks_C2-c.txt"


def pxrd_load_refinement() -> tuple[np.ndarray, ...]:
    data = np.loadtxt(PXRD_REFINEMENT_PATH, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected four refinement columns, got {data.shape}")
    two_theta, observed, calculated, difference = data.T
    if not np.allclose(observed - calculated, difference, atol=0.002):
        raise ValueError("Column four is not observed minus calculated")
    if np.any(np.diff(two_theta) <= 0.0):
        raise ValueError("2theta values must increase monotonically")
    return (two_theta, observed, calculated, difference)


def pxrd_load_bragg_positions(path: Path, x_min: float, x_max: float) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"Expected at least six reflection columns in {path}")
    positions = np.asarray(data[:, 5], dtype=float)
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"Non-finite 2theta value in {path}")
    return positions[(positions >= x_min) & (positions <= x_max)]


SCATTERING_SOURCE = DATA / "scattering_planes_positive_2d_arrays.h5"

SCATTERING_DIFFRACTION_STRIP = DATA / "image.png"

SCATTERING_DISPLAY_LIMIT_RLU = 6.0

SCATTERING_VOXELS_PER_RLU = 30.0

SCATTERING_NORMALIZATION_PERCENTILE = 99.5

SCATTERING_AXIS_LABELS = {
    "100": ("$[010]$ (r.l.u.)", "$[001]$ (r.l.u.)"),
    "110": ("$[1\\bar{1}0]$ (r.l.u.)", "$[001]$ (r.l.u.)"),
    "111": ("$[1\\bar{1}0]$ (r.l.u.)", "$[11\\bar{2}]$ (r.l.u.)"),
    "112": ("$[1\\bar{1}0]$ (r.l.u.)", "$[11\\bar{1}]$ (r.l.u.)"),
}

SCATTERING_DIFFRACTION_PANEL_INDEX = {"100": 0, "111": 2}

SCATTERING_DIFFRACTION_CENTER_XY = {"100": (239, 237), "111": (237, 233)}

SCATTERING_DIFFRACTION_ROTATION_DEG = {"100": 5.5 + 45.0, "111": 17.0 + 30.0}

SCATTERING_DIFFRACTION_HALF_WIDTH_PIXELS = {
    "100": round(220 / np.sqrt(2.0)),
    "111": 154,
}


def scattering_crop_and_normalize(
    image: np.ndarray, coordinate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    keep = (coordinate >= -SCATTERING_DISPLAY_LIMIT_RLU) & (
        coordinate <= SCATTERING_DISPLAY_LIMIT_RLU
    )
    cropped = np.asarray(image[np.ix_(keep, keep)], dtype=float)
    finite = cropped[np.isfinite(cropped)]
    if finite.size == 0:
        raise ValueError("Diffuse-scattering crop contains no finite pixels")
    limit = float(np.percentile(finite, SCATTERING_NORMALIZATION_PERCENTILE))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Diffuse-scattering normalization limit is not positive")
    normalized = np.clip(cropped / limit, 0.0, 1.0)
    return (coordinate[keep], normalized, limit)


def scattering_load_diffraction_panels() -> dict[str, np.ndarray]:
    strip = np.asarray(plt.imread(SCATTERING_DIFFRACTION_STRIP))
    rgb = strip[..., :3]
    column_brightness = np.mean(rgb, axis=(0, 2))
    non_gutter = column_brightness > 0.02
    starts = np.flatnonzero(non_gutter & np.r_[True, ~non_gutter[:-1]])
    stops = np.flatnonzero(non_gutter & np.r_[~non_gutter[1:], True]) + 1
    runs = [(int(start), int(stop)) for (start, stop) in zip(starts, stops)]
    if len(runs) != 3:
        raise ValueError(
            f"Expected three images separated by black gutters in {SCATTERING_DIFFRACTION_STRIP}; found {len(runs)}"
        )
    panels: dict[str, np.ndarray] = {}
    for zone, panel_index in SCATTERING_DIFFRACTION_PANEL_INDEX.items():
        start, stop = runs[panel_index]
        panel = rgb[:, start:stop]
        height, width = panel.shape[:2]
        center_x, center_y = SCATTERING_DIFFRACTION_CENTER_XY[zone]
        radius = min(center_x, width - 1 - center_x, center_y, height - 1 - center_y)
        panel = panel[
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
        panel = rotate(
            panel,
            angle=SCATTERING_DIFFRACTION_ROTATION_DEG[zone],
            axes=(1, 0),
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        half_width = SCATTERING_DIFFRACTION_HALF_WIDTH_PIXELS[zone]
        rotated_center = panel.shape[0] // 2
        panels[zone] = np.clip(
            panel[
                rotated_center - half_width : rotated_center + half_width + 1,
                rotated_center - half_width : rotated_center + half_width + 1,
            ],
            0.0,
            1.0,
        )
    return panels


ZONE110_DIFFRACTION_STRIP = DATA / "image.png"

ZONE110_CENTER_XY = (239, 234)

ZONE110_ROTATION_DEG_CCW = 123.5

ZONE110_HALF_WIDTH_PIXELS = 216


def zone110_load_middle_panel() -> np.ndarray:
    strip = np.asarray(plt.imread(ZONE110_DIFFRACTION_STRIP))
    rgb = strip[..., :3]
    column_brightness = np.mean(rgb, axis=(0, 2))
    non_gutter = column_brightness > 0.02
    starts = np.flatnonzero(non_gutter & np.r_[True, ~non_gutter[:-1]])
    stops = np.flatnonzero(non_gutter & np.r_[~non_gutter[1:], True]) + 1
    runs = [(int(start), int(stop)) for (start, stop) in zip(starts, stops)]
    if len(runs) != 3:
        raise ValueError(
            f"Expected three images separated by black gutters in {ZONE110_DIFFRACTION_STRIP}; found {len(runs)}"
        )
    start, stop = runs[1]
    panel = rgb[:, start:stop]
    height, width = panel.shape[:2]
    center_x, center_y = ZONE110_CENTER_XY
    radius = min(center_x, width - 1 - center_x, center_y, height - 1 - center_y)
    centered = panel[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ]
    aligned = rotate(
        centered,
        angle=ZONE110_ROTATION_DEG_CCW,
        axes=(1, 0),
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    midpoint = aligned.shape[0] // 2
    return np.clip(
        aligned[
            midpoint
            - ZONE110_HALF_WIDTH_PIXELS : midpoint
            + ZONE110_HALF_WIDTH_PIXELS
            + 1,
            midpoint
            - ZONE110_HALF_WIDTH_PIXELS : midpoint
            + ZONE110_HALF_WIDTH_PIXELS
            + 1,
        ],
        0.0,
        1.0,
    )


DELTAPDF_INTEGRATED_XY0_PATH = DATA / "3ddpdf_boxint_hk0.npy"


def deltapdf_percentile_abs_limit(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot scale an array without finite values")
    limit = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Cannot scale an empty or zero-valued array")
    return limit


def deltapdf_plot_split_xy0(
    ax: plt.Axes,
    coordinate: np.ndarray,
    integrated: np.ndarray,
    unintegrated: np.ndarray,
    percentile: float,
) -> tuple[plt.AxesImage, float, float]:
    integrated_limit = deltapdf_percentile_abs_limit(integrated, percentile)
    unintegrated_limit = deltapdf_percentile_abs_limit(unintegrated, percentile)
    composite = np.empty_like(unintegrated, dtype=float)
    left = coordinate < 0.0
    composite[:, left] = integrated[:, left] / integrated_limit
    composite[:, ~left] = unintegrated[:, ~left] / unintegrated_limit
    composite = np.clip(composite, -1.0, 1.0)
    image = ax.imshow(
        composite,
        origin="lower",
        extent=[coordinate[0], coordinate[-1], coordinate[0], coordinate[-1]],
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set(
        xlim=(-3.0, 3.0),
        ylim=(-3.0, 3.0),
        xticks=np.arange(-3, 4, 1),
        yticks=np.arange(-3, 4, 1),
        xlabel="$x$ (unit cells)",
        ylabel="$y$ (unit cells)",
        title="Experimental 3D-$\\Delta$PDF, $xy0$",
    )
    ax.set_xticks(np.arange(-3, 3.1, 0.5), minor=True)
    ax.set_yticks(np.arange(-3, 3.1, 0.5), minor=True)
    ax.set_box_aspect(1)
    ax.axvline(0.0, color="0.25", linewidth=0.65, zorder=4)
    label_style = {
        "fontsize": 6.2,
        "bbox": {"fc": "white", "ec": "none", "alpha": 0.78, "pad": 0.6},
        "zorder": 6,
    }
    ax.text(
        0.03,
        0.97,
        "Integrated",
        transform=ax.transAxes,
        ha="left",
        va="top",
        **label_style,
    )
    ax.text(
        0.97,
        0.97,
        "Unintegrated",
        transform=ax.transAxes,
        ha="right",
        va="top",
        **label_style,
    )
    return (image, integrated_limit, unintegrated_limit)


EXPERIMENTAL_XY0_PATH = DATA / "3ddpdf_xy0.npy"

EXPERIMENTAL_XY_LIMIT = 3.0

EXPERIMENTAL_PIXELS_PER_UNIT_CELL = 24.0

EXPERIMENTAL_IMAGE_PERCENTILE = 99.5


def experimental_set_publication_style() -> None:
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


def experimental_load_xy_slice(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.load(path)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected a square 2D xy0 array, got {image.shape}")
    half_width = (image.shape[0] - 1) / (2.0 * EXPERIMENTAL_PIXELS_PER_UNIT_CELL)
    coordinate = np.linspace(-half_width, half_width, image.shape[0])
    keep = (coordinate >= -EXPERIMENTAL_XY_LIMIT) & (
        coordinate <= EXPERIMENTAL_XY_LIMIT
    )
    return (coordinate[keep], image[np.ix_(keep, keep)])


MODEL_CACHE = DATA / "run80_02_full_calculated_planes_100_110_111.npz"

STRUCTURE_PATH = DATA / "drx_cell2.png"

MODEL_INTEGRATED_3DPDF_PATH = DATA / "3dpdf_boxint7_lattice.npz"

MODEL_DIRECT_AXIS_PATH = DATA / "direct_axis_lattice_units.txt"

MM_PER_INCH = 25.4

FIGURE_WIDTH_MM = 180.0

FIGURE_HEIGHT_MM = 163.0

FIGURE_SIZE_IN = (FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)


def add_panel_label(
    ax: plt.Axes, label: str, *, x: float = -0.16, y: float = 1.075, va: str = "top"
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va=va,
        clip_on=False,
        zorder=20,
    )


def plot_intro_structure(ax: plt.Axes) -> None:
    structure = mpimg.imread(STRUCTURE_PATH)
    rgb = structure[..., :3]
    nonwhite = np.any(rgb < 0.985, axis=2)
    rows, columns = np.where(nonwhite)
    if rows.size and columns.size:
        pad = 3
        row_min = max(int(rows.min()) - pad, 0)
        row_max = min(int(rows.max()) + pad + 1, structure.shape[0])
        column_min = max(int(columns.min()) - pad, 0)
        column_max = min(int(columns.max()) + pad + 1, structure.shape[1])
        structure = structure[row_min:row_max, column_min:column_max]
    ax.imshow(structure, interpolation="none", aspect="equal")
    height, width = structure.shape[:2]
    margin_fraction = 0.045
    ax.set_xlim(-margin_fraction * width, (1.0 + margin_fraction) * width)
    ax.set_ylim((1.0 + margin_fraction) * height, -margin_fraction * height)
    ax.set_axis_off()


def plot_intro_rietveld(ax: plt.Axes) -> None:
    two_theta, observed, calculated, difference = pxrd_load_refinement()
    main_bragg = pxrd_load_bragg_positions(
        PXRD_MAIN_TICKS_PATH, float(two_theta[0]), float(two_theta[-1])
    )
    secondary_bragg = pxrd_load_bragg_positions(
        PXRD_SECONDARY_TICKS_PATH, float(two_theta[0]), float(two_theta[-1])
    )
    fit_span = float(max(np.ptp(observed), np.ptp(calculated)))
    residual_offset = float(
        min(np.min(observed), np.min(calculated)) - 0.105 * fit_span
    )
    shifted_difference = difference + residual_offset
    ax.scatter(
        two_theta,
        observed,
        s=1.7,
        c="black",
        edgecolors="none",
        rasterized=True,
        label="Observed",
        zorder=2,
    )
    ax.plot(
        two_theta,
        calculated,
        color="#d73027",
        linewidth=0.65,
        label="Calculated",
        zorder=3,
    )
    ax.plot(
        two_theta,
        shifted_difference,
        color="#4575b4",
        linewidth=0.48,
        label="Difference",
        zorder=2,
    )
    ax.axhline(residual_offset, color="0.62", linewidth=0.4, zorder=1)
    ax.vlines(
        main_bragg,
        0.4 * residual_offset,
        0.18 * residual_offset,
        color="black",
        linewidth=0.42,
        zorder=3,
    )
    ax.vlines(
        secondary_bragg,
        0.69 * residual_offset,
        0.47 * residual_offset,
        color="0.50",
        linewidth=0.42,
        zorder=3,
    )
    ax.set(
        xlim=(float(two_theta[0]), float(two_theta[-1])),
        ylim=(
            float(np.min(shifted_difference) - 0.03 * fit_span),
            float(max(np.max(observed), np.max(calculated)) + 0.045 * fit_span),
        ),
        xlabel="$2\\theta$ (°)",
        ylabel="Intensity (arb. units)",
    )
    ax.set_xticks(np.arange(10.0, 100.1, 20.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines[["top", "right"]].set_visible(True)
    ax.tick_params(top=False, right=False)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(5.2)
    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.5,
        handletextpad=0.3,
        borderaxespad=0.15,
        labelspacing=0.15,
        markerscale=2.0,
        fontsize=5.0,
    )


def plot_model_plane(
    ax: plt.Axes, axis: np.ndarray, image: np.ndarray, zone: str, label: str | None
) -> None:
    cmap = mpl.colormaps["gray"].copy()
    cmap.set_bad("black")
    ax.imshow(
        image,
        origin="lower",
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    xlabel, ylabel = SCATTERING_AXIS_LABELS[zone]
    ax.set(
        xlim=(-6.0, 6.0),
        ylim=(-6.0, 6.0),
        xticks=np.arange(-6, 7, 3),
        yticks=np.arange(-6, 7, 3),
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.set_title(f"$[{zone}]$", pad=0.5)
    ax.set_xticks(np.arange(-6, 6.1, 1), minor=True)
    ax.set_yticks(np.arange(-6, 6.1, 1), minor=True)
    ax.xaxis.labelpad = 0.0
    ax.yaxis.labelpad = 0.0
    ax.tick_params(axis="both", which="major", pad=1.0)
    ax.set_box_aspect(1)
    if label is not None:
        add_panel_label(ax, label, y=1.025, va="bottom")


def add_dummy_bragg_peaks(image: np.ndarray, axis: np.ndarray, zone: str) -> np.ndarray:
    plane_geometry = {
        "100": (
            np.array([1, 0, 0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        "110": (
            np.array([1, 1, 0]),
            np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
            np.array([0.0, 0.0, 1.0]),
        ),
        "111": (
            np.array([1, 1, 1]),
            np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
            np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0),
        ),
    }
    if zone not in plane_geometry:
        raise ValueError(f"Unsupported Bragg-peak zone: {zone}")
    normal, axis_u, axis_v = plane_geometry[zone]
    peaks = []
    for h in range(-8, 9):
        for k in range(-8, 9):
            for l in range(-8, 9):
                q_integer = np.array([h, k, l])
                if int(q_integer @ normal) != 0:
                    continue
                if not h % 2 == k % 2 == l % 2:
                    continue
                q = q_integer.astype(float)
                peaks.append((float(q @ axis_u), float(q @ axis_v)))
    x_grid, y_grid = np.meshgrid(axis, axis)
    result = np.asarray(image, dtype=float).copy()
    for x_peak, y_peak in peaks:
        if not (axis[0] <= x_peak <= axis[-1] and axis[0] <= y_peak <= axis[-1]):
            continue
        radius_squared = (x_grid - x_peak) ** 2 + (y_grid - y_peak) ** 2
        spot = 0.35 * np.exp(-radius_squared / (2.0 * 0.105**2))
        spot += 0.75 * np.exp(-radius_squared / (2.0 * 0.04**2))
        result = np.maximum(result, np.clip(spot, 0.0, 1.0))
    return result


def plot_zone_axis(
    ax: plt.Axes, image: np.ndarray, zone: str, label: str | None
) -> None:
    if zone == "110":
        height, width = image.shape[:2]
        keep_fraction = 0.7
        half_height = int(round(0.5 * keep_fraction * height))
        half_width = int(round(0.5 * keep_fraction * width))
        center_y, center_x = (height // 2, width // 2)
        image = image[
            center_y - half_height : center_y + half_height,
            center_x - half_width : center_x + half_width,
        ]
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_title(f"$[{zone}]$", pad=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label is not None:
        add_panel_label(ax, label, x=-0.1, y=1.025, va="bottom")


def plot_reconstruction(
    ax: plt.Axes,
    handle: h5py.File,
    coordinate: np.ndarray,
    zone: str,
    label: str | None,
) -> float:
    axis, image, limit = scattering_crop_and_normalize(
        handle[f"scattering_filled/{zone}"][:], coordinate
    )
    cmap = mpl.colormaps["gray"].copy()
    cmap.set_bad("black")
    ax.imshow(
        image,
        origin="lower",
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    xlabel, ylabel = SCATTERING_AXIS_LABELS[zone]
    ax.set(
        xlim=(-SCATTERING_DISPLAY_LIMIT_RLU, SCATTERING_DISPLAY_LIMIT_RLU),
        ylim=(-SCATTERING_DISPLAY_LIMIT_RLU, SCATTERING_DISPLAY_LIMIT_RLU),
        xticks=np.arange(-6, 7, 3),
        yticks=np.arange(-6, 7, 3),
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.set_title(f"$[{zone}]$", pad=0.5)
    ax.set_xticks(np.arange(-6, 6.1, 1), minor=True)
    ax.set_yticks(np.arange(-6, 6.1, 1), minor=True)
    ax.xaxis.labelpad = 0.0
    ax.yaxis.labelpad = 0.0
    ax.tick_params(axis="both", which="major", pad=1.0)
    ax.set_box_aspect(1)
    if label is not None:
        add_panel_label(ax, label, y=1.025, va="bottom")
    return limit


def load_deltapdf_data():
    coordinate, unintegrated = experimental_load_xy_slice(EXPERIMENTAL_XY0_PATH)
    integrated_coordinate, integrated = experimental_load_xy_slice(
        DELTAPDF_INTEGRATED_XY0_PATH
    )
    if not np.allclose(coordinate, integrated_coordinate):
        raise ValueError("Integrated and unintegrated xy0 grids do not match")
    return (coordinate, integrated, unintegrated)


def plot_deltapdf_xy0(
    ax: plt.Axes,
    fig: plt.Figure,
    coordinate: np.ndarray,
    integrated: np.ndarray,
    unintegrated: np.ndarray,
    label: str,
) -> tuple[float, float, plt.Axes]:
    image, integrated_limit, unintegrated_limit = deltapdf_plot_split_xy0(
        ax, coordinate, integrated, unintegrated, EXPERIMENTAL_IMAGE_PERCENTILE
    )
    add_deltapdf_interaction_vectors(ax)
    colorbar_ax = ax.inset_axes([1.045, 0.18, 0.045, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=[-1.0, 0.0, 1.0])
    colorbar.ax.set_yticklabels(["$-1$", "0", "$+1$"])
    colorbar.ax.tick_params(length=2.4, width=0.6, labelsize=6.0, pad=1.5)
    colorbar.outline.set_linewidth(0.6)
    ax.set(
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        xticks=np.arange(-2.0, 2.1, 1.0),
        yticks=np.arange(-2.0, 2.1, 1.0),
    )
    add_panel_label(ax, label, x=-0.045, y=1.025, va="bottom")
    return (integrated_limit, unintegrated_limit, colorbar_ax)


def load_model_integrated_deltapdf() -> tuple[np.ndarray, np.ndarray, float, float]:
    with np.load(MODEL_INTEGRATED_3DPDF_PATH) as data:
        lattice_axis = np.asarray(data["lattice_axis"], dtype=float)
        integrated_volume = np.asarray(data["integrated_sum"], dtype=float)
    expected_shape = (lattice_axis.size,) * 3
    if integrated_volume.shape != expected_shape:
        raise ValueError(
            f"Integrated volume shape {integrated_volume.shape} does not match lattice axis shape {expected_shape}"
        )
    zero_index = int(np.argmin(np.abs(lattice_axis)))
    if not np.isclose(lattice_axis[zero_index], 0.0):
        raise ValueError("Calculated integrated 3D-deltaPDF has no z=0 slice")
    xy0 = integrated_volume[zero_index]
    scale_values = xy0.copy()
    scale_values[zero_index, zero_index] = np.nan
    finite = np.abs(scale_values[np.isfinite(scale_values)])
    color_limit = float(np.max(finite)) if finite.size else np.nan
    if not np.isfinite(color_limit) or color_limit <= 0.0:
        raise ValueError("Calculated integrated 3D-deltaPDF has no finite signal")
    direct_axis = np.loadtxt(MODEL_DIRECT_AXIS_PATH)
    if direct_axis.ndim != 1 or direct_axis.size < 2:
        raise ValueError("Calculated 3D-deltaPDF direct-space axis is invalid")
    voxel_spacing = float(np.mean(np.diff(direct_axis)))
    footprint_width = 7.0 * voxel_spacing
    return (lattice_axis, xy0, color_limit, footprint_width)


def plot_model_integrated_deltapdf(
    ax: plt.Axes,
    fig: plt.Figure,
    lattice_axis: np.ndarray,
    xy0: np.ndarray,
    color_limit: float,
    footprint_width: float,
    label: str,
) -> plt.Axes:
    rectangles = []
    values = []
    half_width = 0.5 * footprint_width
    for iy, y_value in enumerate(lattice_axis):
        for ix, x_value in enumerate(lattice_axis):
            rectangles.append(
                Rectangle(
                    (x_value - half_width, y_value - half_width),
                    footprint_width,
                    footprint_width,
                )
            )
            values.append(xy0[iy, ix] / color_limit)
    collection = PatchCollection(
        rectangles,
        cmap="RdBu_r",
        norm=mpl.colors.Normalize(vmin=-1.0, vmax=1.0, clip=True),
        edgecolor="none",
        linewidth=0.0,
        rasterized=True,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    ax.set(
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        xticks=np.arange(-2.0, 2.1, 1.0),
        yticks=np.arange(-2.0, 2.1, 1.0),
        xlabel="$x$ (unit cells)",
        ylabel="$y$ (unit cells)",
        title="Calculated integrated octahedral-type\n" + "3D-$\\Delta$PDF, $xy0$",
    )
    ax.set_xticks(np.arange(-2.0, 2.1, 0.5), minor=True)
    ax.set_yticks(np.arange(-2.0, 2.1, 0.5), minor=True)
    ax.set_box_aspect(1)
    colorbar_ax = ax.inset_axes([1.045, 0.18, 0.045, 0.64])
    colorbar = fig.colorbar(collection, cax=colorbar_ax, ticks=[-1.0, 0.0, 1.0])
    colorbar.ax.set_yticklabels(["$-1$", "0", "$+1$"])
    colorbar.ax.tick_params(length=2.4, width=0.6, labelsize=6.0, pad=1.5)
    colorbar.outline.set_linewidth(0.6)
    add_panel_label(ax, label, x=-0.045, y=1.025, va="bottom")
    return colorbar_ax


def add_deltapdf_interaction_vectors(ax: plt.Axes) -> None:
    label = {
        "fontsize": 6.2,
        "color": "black",
        "bbox": {"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.5},
    }
    interaction_arrow = {
        "arrowstyle": "-|>",
        "color": "black",
        "lw": 0.9,
        "mutation_scale": 7,
    }
    ax.annotate(
        "", xy=(-0.5, 0.5), xytext=(0.0, 0.0), arrowprops=interaction_arrow, zorder=7
    )
    ax.annotate(
        "", xy=(-1.0, 0.0), xytext=(0.0, 0.0), arrowprops=interaction_arrow, zorder=7
    )
    ax.text(-0.64, 0.54, "$J_1$", ha="right", va="bottom", zorder=8, **label)
    ax.text(-0.57, -0.12, "$J_2$", ha="center", va="top", zorder=8, **label)


def compress_scattering_rows(
    fig: plt.Figure, matrix_axes: np.ndarray, shift: float = 0.03
) -> None:
    fig.canvas.draw()
    fig.set_layout_engine(None)
    row_shifts = (-shift, 0.0, shift)
    for row, delta_y in enumerate(row_shifts):
        for ax in matrix_axes[row, :]:
            position = ax.get_position()
            ax.set_position(
                [position.x0, position.y0 + delta_y, position.width, position.height]
            )


def size_and_center_pxrd(
    pxrd_ax: plt.Axes,
    matrix_axes: np.ndarray,
    width_scale: float = 0.86,
    height_scale: float = 0.9,
) -> None:
    position = pxrd_ax.get_position()
    left = matrix_axes[0, 0].get_position().x0
    right = matrix_axes[0, -1].get_position().x1
    center_x = 0.5 * sum(matrix_axes[0, 1].get_position().intervalx)
    width = width_scale * (right - left)
    height = height_scale * position.height
    pxrd_ax.set_position([center_x - 0.5 * width, position.y1 - height, width, height])


def enlarge_structure_panel(
    ax: plt.Axes, target_ax: plt.Axes, scale: float = 1.4
) -> None:
    position = ax.get_position()
    target = target_ax.get_position()
    width = scale * position.width
    height = scale * position.height
    center_x = 0.5 * (target.x0 + target.x1)
    ax.set_position([center_x - 0.5 * width, position.y1 - height, width, height])


def main() -> None:
    experimental_set_publication_style()
    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
        }
    )
    fig = plt.figure(figsize=FIGURE_SIZE_IN, constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.045, hspace=0.055)
    outer = fig.add_gridspec(
        3, 3, height_ratios=(0.72, 1.0, 1.0), width_ratios=(0.85, 0.85, 1.3)
    )
    top = outer[0, :].subgridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.06)
    intro_rietveld_ax = fig.add_subplot(top[0, 0])
    intro_structure_ax = fig.add_subplot(top[0, 1])
    plot_intro_rietveld(intro_rietveld_ax)
    plot_intro_structure(intro_structure_ax)
    add_panel_label(intro_rietveld_ax, "a", x=-0.07, y=1.02)
    add_panel_label(intro_structure_ax, "b", x=-0.07, y=1.02)
    matrix_grid = outer[1:, 0:2].subgridspec(3, 3, wspace=0.035, hspace=-0.02)
    matrix_axes = np.array(
        [
            [fig.add_subplot(matrix_grid[row, column]) for column in range(3)]
            for row in range(3)
        ],
        dtype=object,
    )
    zones = ("100", "110", "111")
    labels = (("c", "d", "e"), (None, None, None), (None, None, None))
    model_images: dict[str, np.ndarray] = {}
    model_limits: dict[str, float] = {}
    with np.load(MODEL_CACHE) as model:
        model_axis = np.asarray(model["axis"], dtype=float)
        for zone in zones:
            model_images[zone] = np.asarray(model[f"image_{zone}"], dtype=float)
            model_limits[zone] = float(model[f"normalization_limit_{zone}"])
    for row, zone in enumerate(zones):
        image = add_dummy_bragg_peaks(model_images[zone], model_axis, zone)
        plot_model_plane(matrix_axes[row, 0], model_axis, image, zone, labels[row][0])
    zone_axis_images = scattering_load_diffraction_panels()
    zone_axis_images["110"] = zone110_load_middle_panel()
    for row, zone in enumerate(zones):
        plot_zone_axis(
            matrix_axes[row, 1], zone_axis_images[zone], zone, labels[row][1]
        )
    reconstruction_limits: dict[str, float] = {}
    with h5py.File(SCATTERING_SOURCE, "r") as handle:
        coordinate = (
            np.asarray(handle["u_axis_voxel_units"][:], dtype=float)
            / SCATTERING_VOXELS_PER_RLU
        )
        for row, zone in enumerate(zones):
            reconstruction_limits[zone] = plot_reconstruction(
                matrix_axes[row, 2], handle, coordinate, zone, labels[row][2]
            )
    deltapdf_coordinate, integrated, unintegrated = load_deltapdf_data()
    (
        model_deltapdf_axis,
        model_integrated_xy0,
        model_integrated_limit,
        model_footprint_width,
    ) = load_model_integrated_deltapdf()
    pdf_slot = outer[1, 2].subgridspec(
        3, 3, width_ratios=(0.14, 0.84, 0.02), height_ratios=(0.04, 0.92, 0.04)
    )
    deltapdf_ax = fig.add_subplot(pdf_slot[1, 1])
    integrated_limit, unintegrated_limit, colorbar_ax = plot_deltapdf_xy0(
        deltapdf_ax, fig, deltapdf_coordinate, integrated, unintegrated, "f"
    )
    model_pdf_slot = outer[2, 2].subgridspec(
        3, 3, width_ratios=(0.14, 0.84, 0.02), height_ratios=(0.04, 0.92, 0.04)
    )
    model_deltapdf_ax = fig.add_subplot(model_pdf_slot[1, 1])
    model_colorbar_ax = plot_model_integrated_deltapdf(
        model_deltapdf_ax,
        fig,
        model_deltapdf_axis,
        model_integrated_xy0,
        model_integrated_limit,
        model_footprint_width,
        "g",
    )
    transpose_scattering_matrix(fig, matrix_axes, shift=0.012)
    size_and_lower_pxrd(intro_rietveld_ax, matrix_axes)
    enlarge_structure_panel(intro_structure_ax, deltapdf_ax)
    add_row_headings(fig, matrix_axes)
    add_reconstructed_row(fig, matrix_axes, deltapdf_ax)
    add_calculated_row(fig, matrix_axes, model_deltapdf_ax)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600)
    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"))
    plt.close(fig)
    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


RECONSTRUCTED_ROW_SHIFT = 0.02

PXRD_DOWNWARD_SHIFT = 0.01

GROUP_EDGE_PADDING_PT = 0.5

BLUE_ARROW_DOWNWARD_SHIFT = 0.02


def transpose_scattering_matrix(fig, matrix_axes, shift=0.012):
    compress_scattering_rows(fig, matrix_axes, shift=shift)
    positions = [[ax.get_position().frozen() for ax in row] for row in matrix_axes]
    transposed = matrix_axes[:, [1, 2, 0]].T.copy()
    matrix_axes[:] = transposed
    for row in range(3):
        for column in range(3):
            ax = matrix_axes[row, column]
            previous = positions[row][column]
            width = 0.134
            height = width * FIGURE_WIDTH_MM / FIGURE_HEIGHT_MM
            ax.set_position(
                [
                    (0.052, 0.235, 0.418)[column],
                    previous.y0
                    + 0.5 * (previous.height - height)
                    + (RECONSTRUCTED_ROW_SHIFT if row == 1 else 0.0),
                    width,
                    height,
                ]
            )
            for text in list(ax.texts):
                if text.get_text() in ("c", "d", "e"):
                    text.remove()
            if column == 0:
                add_panel_label(
                    ax,
                    ("c", "d", "e")[row],
                    x=-0.1 if row == 0 else -0.16,
                    y=1.025,
                    va="bottom",
                )


def size_and_lower_pxrd(pxrd_ax, matrix_axes):
    size_and_center_pxrd(pxrd_ax, matrix_axes)
    position = pxrd_ax.get_position()
    pxrd_ax.set_position(
        [
            position.x0,
            position.y0 - PXRD_DOWNWARD_SHIFT,
            position.width,
            position.height,
        ]
    )


def add_row_headings(fig, matrix_axes):
    fig.set_dpi(600)
    fig.canvas.draw()
    headings = (
        "Li$_{2.5}$TiO$_3$F$_{0.5}$ zone axes",
        "Li$_{2.5}$TiO$_3$F$_{0.5}$ reconstructed scattering",
        "Calculated octahedral-type diffuse scattering",
    )
    for row, heading in enumerate(headings):
        boxes = [ax.get_position() for ax in matrix_axes[row]]
        fig.text(
            0.5 * (boxes[0].x0 + boxes[-1].x1),
            max((box.y1 for box in boxes)) + 0.03,
            heading,
            fontsize=6.2,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=20,
        )


def add_row_group_and_arrow(
    fig, matrix_axes, target_ax, *, row, box_color, arrow_color
):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for ax in matrix_axes[row]
    ]
    left = min((box.x0 for box in boxes)) - 0.003
    right = max((box.x1 for box in boxes)) + 0.009
    bottom = min((box.y0 for box in boxes)) - 0.008
    top = max((box.y1 for box in boxes)) + 0.008
    edge_padding = GROUP_EDGE_PADDING_PT / (72.0 * fig.get_figheight())
    if row == 1:
        label_boxes = [
            ax.xaxis.label.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            for ax in matrix_axes[row]
        ]
        bottom = min((box.y0 for box in label_boxes)) - edge_padding
    elif row == 2:
        panel_label = next(
            (text for text in matrix_axes[row, 0].texts if text.get_text() == "e")
        )
        label_box = panel_label.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        top = label_box.y1 + edge_padding
    fig.add_artist(
        FancyBboxPatch(
            (left, bottom),
            right - left,
            top - bottom,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            transform=fig.transFigure,
            facecolor=box_color,
            edgecolor="none",
            linewidth=0,
            clip_on=False,
            zorder=-1,
        )
    )
    target = target_ax.get_position()
    row_box = matrix_axes[row, -1].get_position()
    start = (right + 0.008, row_box.y0 + 0.5 * row_box.height)
    end = (target.x0 - 0.024, max(start[1], target.y0 + 0.08 * target.height))
    if row == 2:
        start = (start[0], start[1] - BLUE_ARROW_DOWNWARD_SHIFT)
        end = (end[0], end[1] - BLUE_ARROW_DOWNWARD_SHIFT)
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.9,
            color=arrow_color,
            clip_on=False,
            zorder=30,
        )
    )
    fig.text(
        0.5 * (start[0] + end[0]),
        max(start[1], end[1]) + 0.012,
        "Fourier\ntransform",
        fontsize=6.2,
        color=arrow_color,
        ha="center",
        va="bottom",
        linespacing=1.0,
        zorder=31,
    )


def add_reconstructed_row(fig, matrix_axes, target_ax):
    add_row_group_and_arrow(
        fig, matrix_axes, target_ax, row=1, box_color="#f2dede", arrow_color="#a95757"
    )


def add_calculated_row(fig, matrix_axes, target_ax):
    add_row_group_and_arrow(
        fig, matrix_axes, target_ax, row=2, box_color="#dbeaf1", arrow_color="#52788b"
    )


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    main()
