#!/usr/bin/env python3
"""Create a publication-ready experimental diffraction/scattering figure."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / "LTiOF_dataset"
SOURCE = (
    DATA_ROOT
    / "scattering_planes_positive_greyscale"
    / "scattering_planes_positive_2d_arrays.h5"
)
DIFFRACTION_STRIP = HERE / "image.png"
OUTPUT_STEM = HERE / "experimental_diffuse_scattering_100_110_111_112"

ZONES = ("100", "110", "111", "112")
DISPLAY_LIMIT_RLU = 6.0
VOXELS_PER_RLU = 30.0
NORMALIZATION_PERCENTILE = 99.5

AXIS_LABELS = {
    "100": (r"$[010]$ (r.l.u.)", r"$[001]$ (r.l.u.)"),
    "110": (r"$[1\bar{1}0]$ (r.l.u.)", r"$[001]$ (r.l.u.)"),
    "111": (r"$[1\bar{1}0]$ (r.l.u.)", r"$[11\bar{2}]$ (r.l.u.)"),
    "112": (r"$[1\bar{1}0]$ (r.l.u.)", r"$[11\bar{1}]$ (r.l.u.)"),
}

# The supplied strip contains [100], [110], and [111], separated by black
# gutters. Only [100] and [111] are used in the manuscript figure. The centres
# and rotations below were measured from the direct beams and Bragg-spot
# directions, respectively. Positive rotations are counter-clockwise on the
# displayed image. Keeping the registration here makes it reproducible while
# leaving image.png untouched.
DIFFRACTION_PANEL_INDEX = {"100": 0, "111": 2}
DIFFRACTION_CENTER_XY = {"100": (239, 237), "111": (237, 233)}
# First register the source photographs to b/e (5.5 and 17 degrees), then apply
# the requested symmetry-related rotations. A 45-degree rotation of the square
# [100] net selects the sqrt(2)-larger diagonal spacing; a 30-degree rotation of
# the triangular [111] net selects the sqrt(3)-larger spacing. The crop radii
# therefore shrink by those same factors to keep the transformed spots aligned.
DIFFRACTION_ROTATION_DEG = {"100": 5.5 + 45.0, "111": 17.0 + 30.0}
DIFFRACTION_HALF_WIDTH_PIXELS = {
    "100": round(220 / np.sqrt(2.0)),
    # The upper three-spot row is 127--130 px above the direct beam. A 154 px
    # half-width maps its mean position to +5 when the panel edge is +6.
    "111": 154,
}


def set_publication_style() -> None:
    """Use the same compact Nature-family style as the 3D-/1D-deltaPDF figure."""
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def crop_and_normalize(
    image: np.ndarray, coordinate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    keep = (coordinate >= -DISPLAY_LIMIT_RLU) & (coordinate <= DISPLAY_LIMIT_RLU)
    cropped = np.asarray(image[np.ix_(keep, keep)], dtype=float)
    finite = cropped[np.isfinite(cropped)]
    if finite.size == 0:
        raise ValueError("Diffuse-scattering crop contains no finite pixels")
    limit = float(np.percentile(finite, NORMALIZATION_PERCENTILE))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Diffuse-scattering normalization limit is not positive")
    normalized = np.clip(cropped / limit, 0.0, 1.0)
    return coordinate[keep], normalized, limit


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def load_diffraction_panels() -> dict[str, np.ndarray]:
    """Split the strip, centre each direct beam, and align its Bragg lattice."""
    strip = np.asarray(plt.imread(DIFFRACTION_STRIP))
    rgb = strip[..., :3]
    column_brightness = np.mean(rgb, axis=(0, 2))
    non_gutter = column_brightness > 0.02
    starts = np.flatnonzero(non_gutter & np.r_[True, ~non_gutter[:-1]])
    stops = np.flatnonzero(non_gutter & np.r_[~non_gutter[1:], True]) + 1
    runs = [(int(start), int(stop)) for start, stop in zip(starts, stops)]
    if len(runs) != 3:
        raise ValueError(
            f"Expected three images separated by black gutters in {DIFFRACTION_STRIP}; "
            f"found {len(runs)}"
        )

    panels: dict[str, np.ndarray] = {}
    for zone, panel_index in DIFFRACTION_PANEL_INDEX.items():
        start, stop = runs[panel_index]
        panel = rgb[:, start:stop]
        height, width = panel.shape[:2]
        center_x, center_y = DIFFRACTION_CENTER_XY[zone]
        radius = min(center_x, width - 1 - center_x, center_y, height - 1 - center_y)
        panel = panel[
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
        panel = rotate(
            panel,
            angle=DIFFRACTION_ROTATION_DEG[zone],
            axes=(1, 0),
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        half_width = DIFFRACTION_HALF_WIDTH_PIXELS[zone]
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


def main() -> None:
    set_publication_style()
    cmap = mpl.colormaps["gray"].copy()
    cmap.set_bad("black")
    limits: dict[str, float] = {}

    diffraction_panels = load_diffraction_panels()
    # 7.2 inches is the Nature-family double-column width and matches the
    # experimental 3D-/1D-deltaPDF figure in the neighbouring folder.
    fig, axes = plt.subplots(2, 3, figsize=(7.20, 4.72), constrained_layout=True)

    for ax, zone, label in zip(axes[:, 0], ("100", "111"), "ad"):
        ax.imshow(diffraction_panels[zone], cmap="gray", interpolation="nearest")
        ax.set_title(rf"Zone-axis image, $[{zone}]$", pad=3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label)

    scattering_axes = (axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2])
    with h5py.File(SOURCE, "r") as handle:
        coordinate = np.asarray(handle["u_axis_voxel_units"][:], dtype=float) / VOXELS_PER_RLU
        for ax, zone, label in zip(scattering_axes, ZONES, "bcef"):
            dataset = handle[f"scattering_filled/{zone}"]
            axis, image, limit = crop_and_normalize(dataset[:], coordinate)
            limits[zone] = limit
            artist = ax.imshow(
                image,
                origin="lower",
                extent=[axis[0], axis[-1], axis[0], axis[-1]],
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                rasterized=True,
            )
            xlabel, ylabel = AXIS_LABELS[zone]
            ax.set(
                xlim=(-DISPLAY_LIMIT_RLU, DISPLAY_LIMIT_RLU),
                ylim=(-DISPLAY_LIMIT_RLU, DISPLAY_LIMIT_RLU),
                xticks=np.arange(-6, 7, 2),
                yticks=np.arange(-6, 7, 2),
                xlabel=xlabel,
                ylabel=ylabel,
            )
            ax.set_title(
                rf"Diffuse-scattering reconstruction, $[{zone}]$", pad=3.5
            )
            ax.set_xticks(np.arange(-6, 6.1, 1), minor=True)
            ax.set_yticks(np.arange(-6, 6.1, 1), minor=True)
            ax.set_box_aspect(1)
            add_panel_label(ax, label)

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata_lines = [
        f"source={SOURCE}",
        f"diffraction_strip_source={DIFFRACTION_STRIP}",
        "diffraction_strip_order=100,110,111",
        "diffraction_panels_used=100,111",
        "diffraction_crop=automatic_black_gutter_split_then_direct_beam_centered_square_crop",
        "diffraction_center_xy_pixels=100:239,237;111:237,233",
        "diffraction_rotation_degrees_ccw=100:50.5,111:47.0",
        "diffraction_final_half_width_pixels=100:156,111:154",
        "diffraction_111_vertical_calibration=upper_three_spot_row_at_approximately_plus5_for_plus6_edge",
        "diffraction_symmetry_zoom=100:sqrt(2),111:sqrt(3)",
        "datasets=scattering_filled/100,110,111,112",
        "source_processing=Bragg-punched volume filled before plane extraction; negative values clipped to zero",
        f"display_range_rlu=-{DISPLAY_LIMIT_RLU},+{DISPLAY_LIMIT_RLU}",
        f"normalization=independent_per_panel_{NORMALIZATION_PERCENTILE}th_percentile",
        "display_scale=linear_grayscale_0_to_1",
    ]
    metadata_lines.extend(
        f"zone_{zone}_raw_normalization_limit={limits[zone]:.12g}" for zone in ZONES
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        "\n".join(metadata_lines) + "\n"
    )

    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
