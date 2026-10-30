#!/usr/bin/env python3
"""Export the aligned middle ([110]) zone-axis photograph as its own figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate


HERE = Path(__file__).resolve().parent
DIFFRACTION_STRIP = HERE / "image.png"
OUTPUT_STEM = HERE / "experimental_zone_axis_110"

# Measured from the direct beam and the two orthogonal Bragg-spot rows in the
# middle photograph.  The initial +33.5 degree alignment makes those rows
# horizontal and vertical; the requested additional 90-degree rotation gives a
# total rotation of +123.5 degrees.  The 216 px half-width places the [1-10] and [001] reflections
# on the same -6...+6 reciprocal-space framing used by panel c of the paper
# figure (within the small perspective distortion of the photograph).
CENTER_XY = (239, 234)
ROTATION_DEG_CCW = 123.5
HALF_WIDTH_PIXELS = 216


def set_publication_style() -> None:
    """Match the typography used by the six-panel paper figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def load_middle_panel() -> np.ndarray:
    """Split the strip at its gutters, then centre, rotate, and crop [110]."""
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

    start, stop = runs[1]
    panel = rgb[:, start:stop]
    height, width = panel.shape[:2]
    center_x, center_y = CENTER_XY
    radius = min(center_x, width - 1 - center_x, center_y, height - 1 - center_y)
    centered = panel[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ]
    aligned = rotate(
        centered,
        angle=ROTATION_DEG_CCW,
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
            midpoint - HALF_WIDTH_PIXELS : midpoint + HALF_WIDTH_PIXELS + 1,
            midpoint - HALF_WIDTH_PIXELS : midpoint + HALF_WIDTH_PIXELS + 1,
        ],
        0.0,
        1.0,
    )


def main() -> None:
    set_publication_style()
    panel = load_middle_panel()

    # A compact, standalone single-column figure with the same title treatment
    # as the photographic panels in the paper composite.
    fig, ax = plt.subplots(figsize=(3.35, 3.45), constrained_layout=True)
    ax.imshow(panel, interpolation="nearest")
    ax.set_title(r"Zone-axis image, $[110]$", pad=3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata = "\n".join(
        [
            f"source={DIFFRACTION_STRIP}",
            "diffraction_strip_panel=middle_110",
            f"direct_beam_center_xy_pixels={CENTER_XY[0]},{CENTER_XY[1]}",
            f"rotation_degrees_ccw={ROTATION_DEG_CCW}",
            f"final_half_width_pixels={HALF_WIDTH_PIXELS}",
            "alignment=Bragg_spot_rows_horizontal_and_vertical",
            "framing=approximately_minus6_to_plus6_rlu_matching_paper_panel_c",
        ]
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        metadata + "\n"
    )

    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
