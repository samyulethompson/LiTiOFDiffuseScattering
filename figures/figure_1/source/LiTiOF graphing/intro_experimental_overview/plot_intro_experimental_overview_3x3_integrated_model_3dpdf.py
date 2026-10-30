#!/usr/bin/env python3
"""Build the LiTiOF overview with calculated integrated 3D-deltaPDF panel g."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter


HERE = Path(__file__).resolve().parent
GRAPH_ROOT = HERE.parent
INTRO_SCRIPT = GRAPH_ROOT / "Intro_figure" / "plot_intro_figure.py"
SCATTERING_SCRIPT = (
    GRAPH_ROOT
    / "experimental_diffuse_scattering"
    / "plot_experimental_diffuse_scattering.py"
)
DELTAPDF_SCRIPT = (
    GRAPH_ROOT
    / "experimental_3dpdf_1dpdf_abc"
    / "plot_experimental_3dpdf_1dpdf_abc.py"
)
ZONE110_SCRIPT = (
    GRAPH_ROOT
    / "experimental_diffuse_scattering"
    / "plot_experimental_110_zone_axis.py"
)
MODEL_CACHE = HERE / "run80_02_full_calculated_planes_100_110_111.npz"
STRUCTURE_PATH = HERE / "drx_cell2.png"
MODEL_3DPDF_ROOT = (
    HERE.parents[1]
    / "3d-deltaPDF_informed"
    / "run80"
    / "02_without_displacements"
)
MODEL_INTEGRATED_3DPDF_PATH = MODEL_3DPDF_ROOT / "3dpdf_boxint7_lattice.npz"
MODEL_DIRECT_AXIS_PATH = MODEL_3DPDF_ROOT / "direct_axis_lattice_units.txt"
OUTPUT_STEM = (
    HERE / "LiTiOF_intro_experimental_overview_3x3_integrated_model_3dpdf"
)

LINECUT_MIN_ANGSTROM = 1.0
LINECUT_Y_LIMIT = 0.4

# Nature Communications uses an 180 mm double-column width.  Keeping the
# artwork at its final physical size prevents otherwise legible 7 pt labels
# from being reduced to ~4.5 pt during typesetting.  The 163 mm height leaves
# a little room beneath the journal's 170 mm main-figure limit.
MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 163.0
FIGURE_SIZE_IN = (
    FIGURE_WIDTH_MM / MM_PER_INCH,
    FIGURE_HEIGHT_MM / MM_PER_INCH,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.16,
    y: float = 1.075,
    va: str = "top",
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


def plot_intro_structure(ax: plt.Axes, intro) -> None:
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


def plot_intro_rietveld(ax: plt.Axes, intro) -> None:
    two_theta, observed, calculated, difference = intro.load_refinement()
    main_bragg = intro.load_bragg_positions(
        intro.MAIN_TICKS_PATH, float(two_theta[0]), float(two_theta[-1])
    )
    secondary_bragg = intro.load_bragg_positions(
        intro.SECONDARY_TICKS_PATH, float(two_theta[0]), float(two_theta[-1])
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
    ax.axhline(residual_offset, color="0.62", linewidth=0.40, zorder=1)
    ax.vlines(
        main_bragg,
        0.40 * residual_offset,
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
        xlabel=r"$2\theta$ (°)",
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
    ax: plt.Axes,
    axis: np.ndarray,
    image: np.ndarray,
    zone: str,
    scattering,
    label: str | None,
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
    xlabel, ylabel = scattering.AXIS_LABELS[zone]
    ax.set(
        xlim=(-6.0, 6.0),
        ylim=(-6.0, 6.0),
        xticks=np.arange(-6, 7, 3),
        yticks=np.arange(-6, 7, 3),
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.set_title(rf"$[{zone}]$", pad=0.5)
    ax.set_xticks(np.arange(-6, 6.1, 1), minor=True)
    ax.set_yticks(np.arange(-6, 6.1, 1), minor=True)
    ax.xaxis.labelpad = 0.0
    ax.yaxis.labelpad = 0.0
    ax.tick_params(axis="both", which="major", pad=1.0)
    ax.set_box_aspect(1)
    if label is not None:
        add_panel_label(ax, label, y=1.025, va="bottom")


def add_dummy_bragg_peaks(
    image: np.ndarray, axis: np.ndarray, zone: str
) -> np.ndarray:
    """Add compact synthetic F-lattice Bragg spots to a calculated plane."""
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
                # Conventional F-lattice reflections are all even or all odd.
                if not (h % 2 == k % 2 == l % 2):
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
        spot += 0.75 * np.exp(-radius_squared / (2.0 * 0.040**2))
        result = np.maximum(result, np.clip(spot, 0.0, 1.0))
    return result


def plot_zone_axis(
    ax: plt.Axes,
    image: np.ndarray,
    zone: str,
    label: str | None,
) -> None:
    if zone == "110":
        # A modest central crop removes most of the rotated black corners and
        # brings the [110] diffraction features to the scale of its neighbours.
        height, width = image.shape[:2]
        keep_fraction = 0.70
        half_height = int(round(0.5 * keep_fraction * height))
        half_width = int(round(0.5 * keep_fraction * width))
        center_y, center_x = height // 2, width // 2
        image = image[
            center_y - half_height : center_y + half_height,
            center_x - half_width : center_x + half_width,
        ]
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_title(rf"$[{zone}]$", pad=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label is not None:
        add_panel_label(ax, label, x=-0.10, y=1.025, va="bottom")


def plot_reconstruction(
    ax: plt.Axes,
    scattering,
    handle: h5py.File,
    coordinate: np.ndarray,
    zone: str,
    label: str | None,
) -> float:
    axis, image, limit = scattering.crop_and_normalize(
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
    xlabel, ylabel = scattering.AXIS_LABELS[zone]
    ax.set(
        xlim=(-scattering.DISPLAY_LIMIT_RLU, scattering.DISPLAY_LIMIT_RLU),
        ylim=(-scattering.DISPLAY_LIMIT_RLU, scattering.DISPLAY_LIMIT_RLU),
        xticks=np.arange(-6, 7, 3),
        yticks=np.arange(-6, 7, 3),
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.set_title(rf"$[{zone}]$", pad=0.5)
    ax.set_xticks(np.arange(-6, 6.1, 1), minor=True)
    ax.set_yticks(np.arange(-6, 6.1, 1), minor=True)
    ax.xaxis.labelpad = 0.0
    ax.yaxis.labelpad = 0.0
    ax.tick_params(axis="both", which="major", pad=1.0)
    ax.set_box_aspect(1)
    if label is not None:
        add_panel_label(ax, label, y=1.025, va="bottom")
    return limit


def load_deltapdf_data(deltapdf):
    plotting = deltapdf.load_base_plotting()
    coordinate, unintegrated = plotting.load_xy_slice(plotting.XY0_PATH)
    integrated_coordinate, integrated = plotting.load_xy_slice(
        deltapdf.INTEGRATED_XY0_PATH
    )
    if not np.allclose(coordinate, integrated_coordinate):
        raise ValueError("Integrated and unintegrated xy0 grids do not match")

    profiles = {}
    full_profiles = []
    for direction in ("100", "110"):
        distance, intensity = plotting.load_profile(direction)
        full_profiles.append(intensity[np.isfinite(intensity)])
        keep = distance >= LINECUT_MIN_ANGSTROM - np.finfo(float).eps
        distance = distance[keep]
        intensity = intensity[keep]
        profiles[direction] = (distance, intensity)
    # Use the same shared full-profile normalization as the standalone
    # experimental 3D-/1D-deltaPDF figure.  Filtering to r >= 1 Angstrom
    # before finding the maximum incorrectly enlarged both displayed cuts.
    normalization = float(np.max(np.abs(np.concatenate(full_profiles))))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Cannot normalize empty or zero-valued line cuts")
    normalized_profiles = {
        direction: (distance, intensity / normalization)
        for direction, (distance, intensity) in profiles.items()
    }
    return (
        plotting,
        coordinate,
        integrated,
        unintegrated,
        normalized_profiles,
        normalization,
    )


def plot_deltapdf_xy0(
    ax: plt.Axes,
    fig: plt.Figure,
    plotting,
    deltapdf,
    coordinate: np.ndarray,
    integrated: np.ndarray,
    unintegrated: np.ndarray,
    label: str,
) -> tuple[float, float, plt.Axes]:
    image, integrated_limit, unintegrated_limit = deltapdf.plot_split_xy0(
        ax,
        coordinate,
        integrated,
        unintegrated,
        plotting.IMAGE_PERCENTILE,
    )
    add_deltapdf_interaction_vectors(ax)
    # A slightly wider, taller bar is much easier to read at the final 180 mm
    # print width, while remaining visually subordinate to the map.
    colorbar_ax = ax.inset_axes([1.045, 0.18, 0.045, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=[-1.0, 0.0, 1.0])
    colorbar.ax.set_yticklabels([r"$-1$", "0", r"$+1$"])
    colorbar.ax.tick_params(length=2.4, width=0.6, labelsize=6.0, pad=1.5)
    colorbar.outline.set_linewidth(0.6)
    ax.set(
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        xticks=np.arange(-2.0, 2.1, 1.0),
        yticks=np.arange(-2.0, 2.1, 1.0),
    )
    add_panel_label(ax, label, x=-0.045, y=1.025, va="bottom")
    return integrated_limit, unintegrated_limit, colorbar_ax


def load_model_integrated_deltapdf() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load the calculated box-integrated xy0 lattice slice and its scale."""
    with np.load(MODEL_INTEGRATED_3DPDF_PATH) as data:
        lattice_axis = np.asarray(data["lattice_axis"], dtype=float)
        integrated_volume = np.asarray(data["integrated_sum"], dtype=float)
    expected_shape = (lattice_axis.size,) * 3
    if integrated_volume.shape != expected_shape:
        raise ValueError(
            f"Integrated volume shape {integrated_volume.shape} does not match "
            f"lattice axis shape {expected_shape}"
        )

    zero_index = int(np.argmin(np.abs(lattice_axis)))
    if not np.isclose(lattice_axis[zero_index], 0.0):
        raise ValueError("Calculated integrated 3D-deltaPDF has no z=0 slice")
    xy0 = integrated_volume[zero_index]

    # The self-correlation at the origin is much larger than the pair peaks.
    # Match the source preview by excluding it from the colour normalization.
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
    return lattice_axis, xy0, color_limit, footprint_width


def plot_model_integrated_deltapdf(
    ax: plt.Axes,
    fig: plt.Figure,
    lattice_axis: np.ndarray,
    xy0: np.ndarray,
    color_limit: float,
    footprint_width: float,
    label: str,
) -> plt.Axes:
    """Plot the calculated 7x7x7 box-integrated xy0 lattice peaks."""
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
        xlabel=r"$x$ (unit cells)",
        ylabel=r"$y$ (unit cells)",
        title="Calculated integrated octahedral-type\n"
        + r"3D-$\Delta$PDF, $xy0$",
    )
    ax.set_xticks(np.arange(-2.0, 2.1, 0.5), minor=True)
    ax.set_yticks(np.arange(-2.0, 2.1, 0.5), minor=True)
    ax.set_box_aspect(1)

    colorbar_ax = ax.inset_axes([1.045, 0.18, 0.045, 0.64])
    colorbar = fig.colorbar(collection, cax=colorbar_ax, ticks=[-1.0, 0.0, 1.0])
    colorbar.ax.set_yticklabels([r"$-1$", "0", r"$+1$"])
    colorbar.ax.tick_params(length=2.4, width=0.6, labelsize=6.0, pad=1.5)
    colorbar.outline.set_linewidth(0.6)
    add_panel_label(ax, label, x=-0.045, y=1.025, va="bottom")
    return colorbar_ax


def add_deltapdf_interaction_vectors(ax: plt.Axes) -> None:
    """Add only the J1/J2 interaction vectors inside panel f."""
    label = {
        "fontsize": 6.2,
        "color": "black",
        "bbox": {"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.5},
    }

    # Match the interaction vectors in panel b: J1 terminates at the nearest
    # diagonal red square, while J2 terminates at the horizontal blue square.
    interaction_arrow = {
        "arrowstyle": "-|>",
        "color": "black",
        "lw": 0.9,
        "mutation_scale": 7,
    }
    ax.annotate(
        "",
        xy=(-0.50, 0.50),
        xytext=(0.0, 0.0),
        arrowprops=interaction_arrow,
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(-1.00, 0.00),
        xytext=(0.0, 0.0),
        arrowprops=interaction_arrow,
        zorder=7,
    )
    ax.text(-0.64, 0.54, r"$J_1$", ha="right", va="bottom", zorder=8, **label)
    ax.text(-0.57, -0.12, r"$J_2$", ha="center", va="top", zorder=8, **label)


def add_scattering_column_titles(
    fig: plt.Figure,
    matrix_axes: np.ndarray,
) -> None:
    """Place one shared heading above each diffuse-scattering column."""
    fig.canvas.draw()
    headings = (
        "Calculated octahedral-type\ndiffuse scattering",
        "Li$_{2.5}$TiO$_3$F$_{0.5}$ zone\naxes",
        "Li$_{2.5}$TiO$_3$F$_{0.5}$ reconstructed\nscattering",
    )
    heading_y = max(ax.get_position().y1 for ax in matrix_axes[0, :]) + 0.034
    for column, heading in enumerate(headings):
        box = matrix_axes[0, column].get_position()
        fig.text(
            0.5 * (box.x0 + box.x1),
            heading_y,
            heading,
            fontsize=6.2,
            fontweight="bold",
            ha="center",
            va="bottom",
            linespacing=0.92,
            zorder=20,
        )


def compress_scattering_rows(
    fig: plt.Figure,
    matrix_axes: np.ndarray,
    shift: float = 0.030,
) -> None:
    """Remove excess row whitespace after constrained layout has stabilized."""
    fig.canvas.draw()
    # Freeze the completed macro layout; only the matrix rows move below.
    fig.set_layout_engine(None)
    row_shifts = (-shift, 0.0, shift)
    for row, delta_y in enumerate(row_shifts):
        for ax in matrix_axes[row, :]:
            position = ax.get_position()
            ax.set_position(
                [
                    position.x0,
                    position.y0 + delta_y,
                    position.width,
                    position.height,
                ]
            )


def size_and_center_pxrd(
    pxrd_ax: plt.Axes,
    matrix_axes: np.ndarray,
    width_scale: float = 0.86,
    height_scale: float = 0.90,
) -> None:
    """Shrink panel a and center it over panel d while preserving its top."""
    position = pxrd_ax.get_position()
    left = matrix_axes[0, 0].get_position().x0
    right = matrix_axes[0, -1].get_position().x1
    center_x = 0.5 * sum(matrix_axes[0, 1].get_position().intervalx)
    width = width_scale * (right - left)
    height = height_scale * position.height
    pxrd_ax.set_position(
        [center_x - 0.5 * width, position.y1 - height, width, height]
    )


def enlarge_structure_panel(
    ax: plt.Axes, target_ax: plt.Axes, scale: float = 1.40
) -> None:
    """Enlarge panel b and center it horizontally over panel f."""
    position = ax.get_position()
    target = target_ax.get_position()
    width = scale * position.width
    height = scale * position.height
    center_x = 0.5 * (target.x0 + target.x1)
    ax.set_position(
        [
            center_x - 0.5 * width,
            position.y1 - height,
            width,
            height,
        ]
    )


def match_linecut_width_to_deltapdf(
    linecut_axes: tuple[plt.Axes, plt.Axes],
    deltapdf_ax: plt.Axes,
) -> None:
    """Align both line-cut axes to the 3D-deltaPDF data-axis width."""
    target = deltapdf_ax.get_position()
    for ax in linecut_axes:
        position = ax.get_position()
        ax.set_position([target.x0, position.y0, target.width, position.height])


def expand_linecut_heights(
    linecut_axes: tuple[plt.Axes, plt.Axes], interpanel_gap: float = 0.055
) -> None:
    """Use excess inter-panel whitespace to make both line cuts taller."""
    top_ax, bottom_ax = linecut_axes
    top = top_ax.get_position()
    bottom = bottom_ax.get_position()
    available = max(0.0, top.y0 - bottom.y1 - interpanel_gap)
    growth = 0.5 * available
    top_ax.set_position([top.x0, top.y0 - growth, top.width, top.height + growth])
    bottom_ax.set_position(
        [bottom.x0, bottom.y0, bottom.width, bottom.height + growth]
    )


def add_shared_linecut_ylabel(
    fig: plt.Figure,
    linecut_axes: tuple[plt.Axes, plt.Axes],
) -> None:
    """Center one y label across the complete two-axis line-cut group."""
    top_ax, bottom_ax = linecut_axes
    top = top_ax.get_position()
    bottom = bottom_ax.get_position()
    fig.text(
        top.x0 - 0.045,
        0.5 * (top.y1 + bottom.y0),
        "Normalized intensity",
        rotation=90,
        ha="center",
        va="center",
        fontsize=mpl.rcParams["axes.labelsize"],
    )


def add_reconstruction_group_and_arrow(
    fig: plt.Figure,
    matrix_axes: np.ndarray,
    target_ax: plt.Axes,
) -> None:
    """Shade the reconstructed column and point it at the 3D-deltaPDF."""
    fig.canvas.draw()
    reconstructed_axes = matrix_axes[:, -1]
    renderer = fig.canvas.get_renderer()
    content_boxes = []
    for ax in reconstructed_axes:
        artists = [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]
        boxes = [ax.get_window_extent(renderer)]
        boxes.extend(
            artist.get_window_extent(renderer)
            for artist in artists
            if artist.get_visible() and artist.get_text()
        )
        converted = [box.transformed(fig.transFigure.inverted()) for box in boxes]
        content_boxes.extend(converted)
    left = min(box.x0 for box in content_boxes)
    right = max(box.x1 for box in content_boxes)
    bottom = min(box.y0 for box in content_boxes)
    top = max(box.y1 for box in content_boxes)
    pad_x, pad_y = 0.003, 0.008
    box_shift_x = 0.006
    box_color = "#f2dede"
    arrow_color = "#a95757"
    box_left = left - pad_x + box_shift_x
    box_width = right - left + 2.0 * pad_x
    box_right = box_left + box_width
    group_box = FancyBboxPatch(
        (box_left, bottom - pad_y),
        box_width,
        top - bottom + 2.0 * pad_y,
        boxstyle="round,pad=0.0,rounding_size=0.010",
        transform=fig.transFigure,
        facecolor=box_color,
        edgecolor="none",
        linewidth=0.0,
        clip_on=False,
        zorder=-1,
    )
    fig.add_artist(group_box)

    target_box = target_ax.get_position()
    arrow_y = target_box.y0 + 0.52 * target_box.height
    arrow_y = float(np.clip(arrow_y, bottom + 0.02, top - 0.02))
    end = (target_box.x0 - 0.067, arrow_y)
    start = (end[0] - 0.045, arrow_y)
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        mutation_scale=13.0,
        linewidth=1.9,
        color=arrow_color,
        clip_on=False,
        zorder=30,
    )
    fig.add_artist(arrow)
    label_x = 0.50 * (start[0] + end[0])
    fig.text(
        label_x,
        arrow_y + 0.010,
        "Fourier",
        fontsize=6.5,
        color=arrow_color,
        ha="center",
        va="bottom",
        zorder=31,
    )
    fig.text(
        label_x,
        arrow_y - 0.010,
        "transform",
        fontsize=6.5,
        color=arrow_color,
        ha="center",
        va="top",
        zorder=31,
    )


def add_calculated_group_and_arrow(
    fig: plt.Figure,
    matrix_axes: np.ndarray,
    target_ax: plt.Axes,
) -> None:
    """Shade calculated panel c and route its Fourier-transform arrow to g."""
    fig.canvas.draw()
    calculated_axes = matrix_axes[:, 0]
    renderer = fig.canvas.get_renderer()
    content_boxes = []
    for ax in calculated_axes:
        artists = [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]
        boxes = [ax.get_window_extent(renderer)]
        boxes.extend(
            artist.get_window_extent(renderer)
            for artist in artists
            if artist.get_visible() and artist.get_text()
        )
        content_boxes.extend(
            box.transformed(fig.transFigure.inverted()) for box in boxes
        )

    left = min(box.x0 for box in content_boxes)
    right = max(box.x1 for box in content_boxes)
    bottom = min(box.y0 for box in content_boxes)
    top = max(box.y1 for box in content_boxes)
    pad_x, pad_y = 0.003, 0.008
    box_shift_x = 0.006
    box_color = "#dbeaf1"
    arrow_color = "#52788b"
    box_left = left - pad_x + box_shift_x
    box_width = right - left + 2.0 * pad_x
    box_right = box_left + box_width
    group_box = FancyBboxPatch(
        (box_left, bottom - pad_y),
        box_width,
        top - bottom + 2.0 * pad_y,
        boxstyle="round,pad=0.0,rounding_size=0.010",
        transform=fig.transFigure,
        facecolor=box_color,
        edgecolor="none",
        linewidth=0.0,
        clip_on=False,
        zorder=-1,
    )
    fig.add_artist(group_box)

    # Match the short e-to-f connector: colour links the arrow beside g back
    # to the calculated column without drawing across the intervening panels.
    middle_bottom = matrix_axes[1, 0].get_position().y0
    bottom_top = matrix_axes[2, 0].get_position().y1
    arrow_y = 0.5 * (middle_bottom + bottom_top)
    target_box = target_ax.get_position()
    end = (target_box.x0 - 0.067, arrow_y)
    start = (end[0] - 0.045, arrow_y)
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        mutation_scale=13.0,
        linewidth=1.9,
        color=arrow_color,
        clip_on=False,
        zorder=30,
    )
    fig.add_artist(arrow)
    label_x = 0.5 * (start[0] + end[0])
    fig.text(
        label_x,
        arrow_y + 0.010,
        "Fourier",
        fontsize=6.5,
        color=arrow_color,
        ha="center",
        va="bottom",
        zorder=31,
    )
    fig.text(
        label_x,
        arrow_y - 0.010,
        "transform",
        fontsize=6.5,
        color=arrow_color,
        ha="center",
        va="top",
        zorder=31,
    )


def add_offset_signed_gradient(
    ax: plt.Axes,
    plotting,
    distance: np.ndarray,
    intensity: np.ndarray,
    x_min: float,
    x_max: float,
    local_min: float,
    local_max: float,
    offset: float,
) -> None:
    nx, ny = 1200, 420
    x_dense = np.linspace(x_min, x_max, nx)
    y_relative = np.linspace(local_min, local_max, ny)
    curve = np.asarray(plotting.Akima1DInterpolator(distance, intensity)(x_dense))
    curve = np.clip(curve, local_min, local_max)

    rgba = np.zeros((ny, nx, 4), dtype=float)
    positive_rows = y_relative >= 0.0
    negative_rows = ~positive_rows
    positive_scale = max(local_max, np.finfo(float).eps)
    negative_scale = max(abs(local_min), np.finfo(float).eps)
    rgba[positive_rows, :, :3] = plotting.POSITIVE_CMAP(
        np.clip(y_relative[positive_rows] / positive_scale, 0.0, 1.0)
    )[:, None, :3]
    rgba[negative_rows, :, :3] = plotting.NEGATIVE_CMAP(
        np.clip((y_relative[negative_rows] - local_min) / negative_scale, 0.0, 1.0)
    )[:, None, :3]
    yy = y_relative[:, None]
    cc = curve[None, :]
    inside = ((cc >= 0.0) & (yy >= 0.0) & (yy <= cc)) | (
        (cc < 0.0) & (yy <= 0.0) & (yy >= cc)
    )
    rgba[:, :, 3] = inside.astype(float)
    ax.imshow(
        rgba,
        origin="lower",
        extent=[x_min, x_max, offset + local_min, offset + local_max],
        interpolation="bilinear",
        aspect="auto",
        zorder=1,
    )
    ax.plot(x_dense, offset + curve, color="black", linewidth=0.65, zorder=4)
    ax.scatter(
        distance,
        offset + intensity,
        s=2.0,
        facecolors="black",
        edgecolors="none",
        rasterized=True,
        clip_on=True,
        zorder=5,
    )
    ax.axhline(offset, color="0.55", linewidth=0.45, zorder=3)


def plot_split_linecuts(
    axes: tuple[plt.Axes, plt.Axes],
    plotting,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    label: str,
) -> None:
    profile_max = 12.0

    for ax, direction in zip(axes, ("100", "110")):
        distance, intensity = profiles[direction]
        local_min, local_max = -LINECUT_Y_LIMIT, LINECUT_Y_LIMIT
        lattice_spacing = plotting.LATTICE_CONSTANT_ANG * (
            1.0 / np.sqrt(2.0) if direction == "110" else 0.5
        )
        first_index = int(np.ceil(LINECUT_MIN_ANGSTROM / lattice_spacing))
        lattice_points = np.arange(
            first_index * lattice_spacing,
            profile_max + np.finfo(float).eps,
            lattice_spacing,
        )
        for lattice_point in lattice_points:
            ax.axvline(
                lattice_point,
                color="0.80",
                linewidth=0.48,
                zorder=0,
            )
        add_offset_signed_gradient(
            ax,
            plotting,
            distance,
            intensity,
            LINECUT_MIN_ANGSTROM,
            profile_max,
            local_min,
            local_max,
            0.0,
        )
        ax.set(
            xlim=(LINECUT_MIN_ANGSTROM, profile_max),
            ylim=(local_min, local_max),
            xticks=[1.0, 4.0, 8.0, 12.0],
        )
        ax.set_xticks(np.arange(1.0, profile_max + 0.1, 1.0), minor=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(top=False, right=False)

        # Label the crystallographic repeat count on a direction-specific
        # top axis.  The F-centred lattice repeats every a/2 along [100] and
        # every a/sqrt(2) along [110], so the two axes intentionally differ.
        to_unit_cells = lambda r, spacing=lattice_spacing: r / spacing
        to_angstrom = lambda n, spacing=lattice_spacing: n * spacing
        unit_cell_ax = ax.secondary_xaxis(
            "top", functions=(to_unit_cells, to_angstrom)
        )
        unit_cell_ax.set_in_layout(False)
        first_cell = int(np.ceil(LINECUT_MIN_ANGSTROM / lattice_spacing))
        last_cell = int(np.floor(profile_max / lattice_spacing))
        unit_cell_ax.set_xticks(np.arange(first_cell, last_cell + 1))
        unit_cell_ax.set_xlabel("Unit cells", labelpad=1.0)
        unit_cell_ax.tick_params(
            axis="x", which="major", length=2.5, width=0.55, pad=1.0
        )
        ax.text(
            0.97,
            0.92,
            rf"$[{direction}]$",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="top",
            bbox={"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.6},
            zorder=6,
        )

    top_ax, bottom_ax = axes
    top_ax.set_title(r"Experimental 1D-$\Delta$PDF", pad=22.0)
    top_ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    bottom_ax.set_xlabel(r"$r$ ($\AA$)")
    add_panel_label(top_ax, label, x=-0.045, y=1.025, va="bottom")


def main() -> None:
    if not MODEL_CACHE.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_CACHE}; run extend_run80_plane_cache_with_110.py first"
        )

    intro = load_module("litiof_intro", INTRO_SCRIPT)
    scattering = load_module("litiof_scattering", SCATTERING_SCRIPT)
    deltapdf = load_module("litiof_deltapdf", DELTAPDF_SCRIPT)
    zone110 = load_module("litiof_zone110", ZONE110_SCRIPT)
    plotting = deltapdf.load_base_plotting()
    plotting.set_publication_style()
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
    fig.get_layout_engine().set(
        w_pad=0.030,
        h_pad=0.030,
        wspace=0.045,
        hspace=0.055,
    )
    outer = fig.add_gridspec(
        3,
        3,
        height_ratios=(0.72, 1.0, 1.0),
        width_ratios=(0.85, 0.85, 1.30),
    )

    # Keep the compact, independently laid-out top row so the PDF column
    # retains its larger original width. PXRD is aligned to c--e after layout.
    top = outer[0, :].subgridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.06)
    intro_rietveld_ax = fig.add_subplot(top[0, 0])
    intro_structure_ax = fig.add_subplot(top[0, 1])
    plot_intro_rietveld(intro_rietveld_ax, intro)
    plot_intro_structure(intro_structure_ax, intro)
    add_panel_label(intro_rietveld_ax, "a", x=-0.07, y=1.02)
    add_panel_label(intro_structure_ax, "b", x=-0.07, y=1.02)

    # Macro-grid positions 4, 5, 7, and 8 contain the nested 3x3 scattering
    # matrix. Rows are zone axes; columns are calculated, measured, and
    # reconstructed diffuse scattering.
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
        plot_model_plane(
            matrix_axes[row, 0],
            model_axis,
            image,
            zone,
            scattering,
            labels[row][0],
        )

    zone_axis_images = scattering.load_diffraction_panels()
    zone_axis_images["110"] = zone110.load_middle_panel()
    for row, zone in enumerate(zones):
        plot_zone_axis(
            matrix_axes[row, 1],
            zone_axis_images[zone],
            zone,
            labels[row][1],
        )

    reconstruction_limits: dict[str, float] = {}
    with h5py.File(scattering.SOURCE, "r") as handle:
        coordinate = (
            np.asarray(handle["u_axis_voxel_units"][:], dtype=float)
            / scattering.VOXELS_PER_RLU
        )
        for row, zone in enumerate(zones):
            reconstruction_limits[zone] = plot_reconstruction(
                matrix_axes[row, 2],
                scattering,
                handle,
                coordinate,
                zone,
                labels[row][2],
            )

    (
        plotting,
        deltapdf_coordinate,
        integrated,
        unintegrated,
        _profiles,
        _profile_normalization,
    ) = load_deltapdf_data(deltapdf)
    (
        model_deltapdf_axis,
        model_integrated_xy0,
        model_integrated_limit,
        model_footprint_width,
    ) = load_model_integrated_deltapdf()

    # Return the PDF panels to macro-grid positions 6 and 9. The 3D-deltaPDF
    # is inset within position 6 to leave a clear transform/brace corridor.
    pdf_slot = outer[1, 2].subgridspec(
        3,
        3,
        width_ratios=(0.14, 0.84, 0.02),
        height_ratios=(0.04, 0.92, 0.04),
    )
    deltapdf_ax = fig.add_subplot(pdf_slot[1, 1])
    integrated_limit, unintegrated_limit, colorbar_ax = plot_deltapdf_xy0(
        deltapdf_ax,
        fig,
        plotting,
        deltapdf,
        deltapdf_coordinate,
        integrated,
        unintegrated,
        "f",
    )

    model_pdf_slot = outer[2, 2].subgridspec(
        3,
        3,
        width_ratios=(0.14, 0.84, 0.02),
        height_ratios=(0.04, 0.92, 0.04),
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

    # Square image panels are width-limited, so GridSpec hspace alone has
    # little visible effect. Move the outer rows explicitly toward the middle
    # row to create a genuinely tighter and symmetric vertical rhythm.
    compress_scattering_rows(fig, matrix_axes, shift=0.012)
    size_and_center_pxrd(intro_rietveld_ax, matrix_axes)
    enlarge_structure_panel(intro_structure_ax, deltapdf_ax)
    add_scattering_column_titles(fig, matrix_axes)
    add_reconstruction_group_and_arrow(fig, matrix_axes, deltapdf_ax)
    add_calculated_group_and_arrow(fig, matrix_axes, model_deltapdf_ax)

    # Do not use bbox_inches="tight": it changes the requested physical page
    # size.  Constrained layout already keeps all artists inside this exact
    # 180 x 163 mm canvas.
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600)
    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"))
    plt.close(fig)

    metadata = [
        "layout=macro_3x3_with_nested_3x3_scattering_matrix",
        "top_row=compact_a_pxrd_refinement_centered_over_d_and_enlarged_b_drx_cell",
        "panel_a_width=0.86x_c_through_e_span",
        "panel_a_height=0.90x_original_top_aligned",
        "macro_positions_4_5_7_8=nested_3x3_scattering_matrix",
        "macro_position_6=l_experimental_3d_deltapdf",
        "macro_position_9=g_calculated_integrated_3d_deltapdf_aligned_below_experimental_3d_deltapdf",
        "scattering_matrix_columns=calculated_Run80,experimental_zone_axis,experimental_reconstruction",
        "scattering_matrix_shared_column_titles=Calculated_octahedral-type_diffuse_scattering,Li2.5TiO3F0.5_zone_axes,Li2.5TiO3F0.5_reconstructed_scattering",
        "scattering_matrix_mini_titles=zone_indices_only",
        "scattering_matrix_column_title_gap=increased_to_clear_group_backgrounds",
        "scattering_matrix_column_labels=c_calculated,d_zone_axis,e_reconstructed",
        "scattering_matrix_row_1=calculated_100,zone_axis_100,reconstructed_100",
        "scattering_matrix_row_2=calculated_110,zone_axis_110,reconstructed_110",
        "scattering_matrix_row_3=calculated_111,zone_axis_111,reconstructed_111",
        "scattering_matrix_spacing=moderate_vertical_gap_slightly_wider_column_gap",
        "panels_c_f_i=full_calculated_Run80_02_planes_not_halfplanes",
        "panels_d_g_j=experimental_zone_axis_patterns_100,110,111",
        "panels_e_h_k=experimental_reconstructed_planes_100,110,111",
        "panel_f=experimental_3d_deltapdf_xy0_integrated_left_unintegrated_right",
        "panel_f_display_bounds_unit_cells=-2,+2_on_both_axes",
        "panel_f_crystallographic_direction_vectors=none",
        "panel_f_interaction_vectors=J1_origin_to_diagonal_red_square,J2_origin_to_horizontal_blue_square",
        "reconstructed_scattering_group=three_panels_on_shared_rounded_pale_red_background_without_border",
        "reconstructed_to_experimental_3d_deltapdf_annotation=short_thick_horizontal_muted_red_arrow_with_Fourier_above_and_transform_below",
        "calculated_scattering_group=three_panels_on_shared_rounded_pale_blue_background_without_border",
        "calculated_to_model_3d_deltapdf_annotation=short_thick_horizontal_dark_blue_arrow_beside_g_with_Fourier_above_and_transform_below",
        "panel_g=run80_02_calculated_7x7x7_box_integrated_3d_deltapdf_xy0",
        "panel_g_display_bounds_unit_cells=-2,+2_on_both_axes",
        "panel_g_color_normalization=max_abs_nonorigin_integrated_peak",
        "panel_g_colorbar=normalized_symmetric_minus1_to_plus1",
        "panel_a_axes=boxed",
        "panel_f_colorbar_position=outside_right_of_data_axes",
        "panel_f_colorbar=tall_readable_vertical_bar",
        "panel_f_and_g_data_axes=matched_square_layout",
        "panel_b_image_scale=enlarged_1.40x_without_distortion",
        "panel_b_horizontal_center=aligned_with_panel_f",
        f"figure_width_mm={FIGURE_WIDTH_MM:.1f}",
        f"figure_height_mm={FIGURE_HEIGHT_MM:.1f}",
        "figure_target=Nature_Communications_double_column",
        f"intro_structure_source={STRUCTURE_PATH}",
        f"intro_refinement_source={intro.REFINEMENT_PATH}",
        f"calculated_plane_cache={MODEL_CACHE}",
        "calculated_origin_punch_applied=false",
        "calculated_dummy_bragg_peaks=allowed_F_lattice_reflections",
        f"diffuse_scattering_source={scattering.SOURCE}",
        f"zone_axis_110_source={zone110.DIFFRACTION_STRIP}",
        "zone_axis_110_composite_crop_fraction=0.70",
        f"deltapdf_integrated_xy0_source={deltapdf.INTEGRATED_XY0_PATH}",
        f"deltapdf_unintegrated_xy0_source={plotting.XY0_PATH}",
        f"model_integrated_3d_deltapdf_source={MODEL_INTEGRATED_3DPDF_PATH}",
        f"model_direct_axis_source={MODEL_DIRECT_AXIS_PATH}",
        "model_integrated_3d_deltapdf_slice=xy0",
        "model_integrated_3d_deltapdf_box=7x7x7_voxels",
        f"model_integrated_3d_deltapdf_footprint_width={model_footprint_width:.12g}",
        f"model_integrated_3d_deltapdf_nonorigin_color_limit_raw={model_integrated_limit:.12g}",
        f"integrated_xy0_symmetric_color_limit_raw={integrated_limit:.12g}",
        f"unintegrated_xy0_symmetric_color_limit_raw={unintegrated_limit:.12g}",
    ]
    for zone in zones:
        metadata.append(
            f"calculated_{zone}_raw_normalization_limit={model_limits[zone]:.12g}"
        )
        metadata.append(
            f"reconstructed_{zone}_raw_normalization_limit="
            f"{reconstruction_limits[zone]:.12g}"
        )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )

    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
