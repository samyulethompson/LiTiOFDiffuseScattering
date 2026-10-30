#!/usr/bin/env python3
"""Create the three-panel experimental 3D-/1D-deltaPDF figure (a--c)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = (
    HERE.parent
    / "experimental_3dpdf_1dpdf"
    / "plot_experimental_3dpdf_1dpdf.py"
)
DATA_ROOT = HERE.parents[1] / "LTiOF_dataset"
INTEGRATED_XY0_PATH = (
    DATA_ROOT / "3ddpdf_hk_slices" / "3ddpdf_boxint_hk0.npy"
)
OUTPUT_STEM = HERE / "experimental_xy0_integrated_100_110"


def load_base_plotting():
    """Load the established plotting helpers without modifying the original."""
    spec = importlib.util.spec_from_file_location("experimental_3dpdf_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting workflow from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile_abs_limit(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot scale an array without finite values")
    limit = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Cannot scale an empty or zero-valued array")
    return limit


def plot_split_xy0(
    ax: plt.Axes,
    coordinate: np.ndarray,
    integrated: np.ndarray,
    unintegrated: np.ndarray,
    percentile: float,
) -> tuple[plt.AxesImage, float, float]:
    """Plot integrated (left) and unintegrated (right) xy0 half-planes."""
    integrated_limit = percentile_abs_limit(integrated, percentile)
    unintegrated_limit = percentile_abs_limit(unintegrated, percentile)

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
        xlabel=r"$x$ (unit cells)",
        ylabel=r"$y$ (unit cells)",
        title=r"Experimental 3D-$\Delta$PDF, $xy0$",
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
    return image, integrated_limit, unintegrated_limit


def main() -> None:
    plotting = load_base_plotting()
    plotting.set_publication_style()

    coordinate, unintegrated_xy0 = plotting.load_xy_slice(plotting.XY0_PATH)
    integrated_coordinate, integrated_xy0 = plotting.load_xy_slice(
        INTEGRATED_XY0_PATH
    )
    if not np.allclose(coordinate, integrated_coordinate):
        raise ValueError("Integrated and unintegrated xy0 grids do not match")

    r100, pdf100 = plotting.load_profile("100")
    r110, pdf110 = plotting.load_profile("110")
    combined = np.concatenate(
        [pdf100[np.isfinite(pdf100)], pdf110[np.isfinite(pdf110)]]
    )
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

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.20, 2.55),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.015, wspace=0.01)

    image, integrated_limit, unintegrated_limit = plot_split_xy0(
        axes[0],
        coordinate,
        integrated_xy0,
        unintegrated_xy0,
        plotting.IMAGE_PERCENTILE,
    )
    plotting.add_direction_vectors(axes[0])
    colorbar = fig.colorbar(
        image,
        ax=axes[0],
        orientation="vertical",
        location="right",
        fraction=0.050,
        pad=0.035,
        shrink=0.70,
        aspect=16,
        ticks=[-1.0, 0.0, 1.0],
    )
    colorbar.ax.set_yticklabels([r"$-1$", "0", r"$+1$"])
    colorbar.ax.tick_params(length=2, width=0.55, labelsize=5.5, pad=1.5)
    colorbar.outline.set_linewidth(0.55)

    for ax, distance, intensity, direction in (
        (axes[1], r100, pdf100, "100"),
        (axes[2], r110, pdf110, "110"),
    ):
        profile_min_ang = plotting.PROFILE_MIN * plotting.LATTICE_CONSTANT_ANG
        profile_max_ang = plotting.PROFILE_MAX * plotting.LATTICE_CONSTANT_ANG
        lattice_spacing_ang = plotting.LATTICE_CONSTANT_ANG * (
            1.0 / np.sqrt(2.0) if direction == "110" else 0.5
        )
        lattice_points_ang = np.arange(
            0.0,
            profile_max_ang + np.finfo(float).eps,
            lattice_spacing_ang,
        )
        for lattice_point in lattice_points_ang:
            ax.axvline(lattice_point, color="0.78", linewidth=0.55, zorder=2)
        plotting.add_signed_gradient(
            ax,
            distance,
            intensity,
            profile_min_ang,
            profile_max_ang,
            y_min,
            y_max,
        )
        plotting.format_axes(ax)
        ax.set_xlim(profile_min_ang, profile_max_ang)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(0.0, profile_max_ang + 0.1, 4.0))
        ax.set_xticks(np.arange(0.0, profile_max_ang + 0.1, 1.0), minor=True)
        ax.set_xlabel(r"$r$ ($\AA$)")
        ax.set_ylabel(r"Normalized $\Delta$PDF intensity")
        ax.set_title(rf"Experimental 3D-$\Delta$PDF cut, $[{direction}]$")

    for ax, label in zip(axes, ("a", "b", "c")):
        plotting.add_panel_label(ax, label)

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata = "\n".join(
        [
            f"integrated_xy0_source={INTEGRATED_XY0_PATH}",
            f"unintegrated_xy0_source={plotting.XY0_PATH}",
            "panel_a_left_half=integrated_3d_deltapdf",
            "panel_a_right_half=unintegrated_3d_deltapdf",
            f"xy_display_range_unit_cells=-{plotting.XY_LIMIT},+{plotting.XY_LIMIT}",
            f"xy_color_limit_percentile_abs={plotting.IMAGE_PERCENTILE}",
            f"integrated_xy_symmetric_color_limit_raw={integrated_limit:.12g}",
            f"unintegrated_xy_symmetric_color_limit_raw={unintegrated_limit:.12g}",
            "xy_halfplanes_normalized_independently",
            "xy0_direction_vectors=100,110",
            f"profile_100_source={plotting.PROFILE_ROOT / 'experimental_1d_deltapdf_100.csv'}",
            f"profile_110_source={plotting.PROFILE_ROOT / 'experimental_1d_deltapdf_110.csv'}",
            f"profile_display_range_unit_cells={plotting.PROFILE_MIN},{plotting.PROFILE_MAX}",
            f"profile_display_range_angstrom={plotting.PROFILE_MIN * plotting.LATTICE_CONSTANT_ANG},{plotting.PROFILE_MAX * plotting.LATTICE_CONSTANT_ANG}",
            f"profile_lattice_constant_angstrom={plotting.LATTICE_CONSTANT_ANG}",
            "profile_lattice_lines_100_angstrom=half_integer_multiples_of_lattice_constant",
            "profile_lattice_lines_110_angstrom=integer_multiples_of_lattice_constant_over_sqrt(2)",
            f"profile_shared_normalization_factor_raw={normalization_factor:.12g}",
            "profile_normalization=max_abs_across_100_and_110_in_displayed_range",
            "profile_y_limits=shared_between_100_and_110",
            "profile_points=raw_samples_shown_as_black_dots",
            "profile_line=Akima_interpolating_spline",
            "positive_fill=white_at_zero_to_red",
            "negative_fill=white_at_zero_to_blue",
        ]
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        metadata + "\n"
    )

    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
