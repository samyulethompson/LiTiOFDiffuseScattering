#!/usr/bin/env python3
"""Render the original overview with scattering types as rows and zones as columns.

Reuse the original plotting/data processing functions without changing the source
figure. Run with the building conda environment.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import plot_intro_experimental_overview_3x3_integrated_model_3dpdf as original


OUTPUT_STEM = Path(__file__).resolve().parent / (
    "LiTiOF_intro_experimental_overview_3x3_integrated_model_3dpdf_transposed"
)
original_compress = original.compress_scattering_rows
original_size_pxrd = original.size_and_center_pxrd
RECONSTRUCTED_ROW_SHIFT = 0.020
PXRD_DOWNWARD_SHIFT = 0.010
GROUP_EDGE_PADDING_PT = 0.5
BLUE_ARROW_DOWNWARD_SHIFT = 0.020


def transpose_scattering_matrix(fig, matrix_axes, shift=0.012):
    original_compress(fig, matrix_axes, shift=shift)
    positions = [[ax.get_position().frozen() for ax in row] for row in matrix_axes]
    # Old columns: model, zone axis, reconstruction. New rows: zone axis,
    # reconstruction, model. Move axes, never rotate or transpose their data.
    transposed = matrix_axes[:, [1, 2, 0]].T.copy()
    matrix_axes[:] = transposed
    for row in range(3):
        for column in range(3):
            ax = matrix_axes[row, column]
            previous = positions[row][column]
            width = 0.134
            height = width * original.FIGURE_WIDTH_MM / original.FIGURE_HEIGHT_MM
            ax.set_position([
                (0.052, 0.235, 0.418)[column],
                previous.y0 + 0.5 * (previous.height - height)
                + (RECONSTRUCTED_ROW_SHIFT if row == 1 else 0.0),
                width, height,
            ])
            for text in list(ax.texts):
                if text.get_text() in ("c", "d", "e"):
                    text.remove()
            if column == 0:
                original.add_panel_label(
                    ax, ("c", "d", "e")[row],
                    x=-0.10 if row == 0 else -0.16,
                    y=1.025, va="bottom",
                )


def size_and_lower_pxrd(pxrd_ax, matrix_axes):
    original_size_pxrd(pxrd_ax, matrix_axes)
    position = pxrd_ax.get_position()
    pxrd_ax.set_position([
        position.x0, position.y0 - PXRD_DOWNWARD_SHIFT,
        position.width, position.height,
    ])


def add_row_headings(fig, matrix_axes):
    # Measure text at the PNG export resolution so math-text bounds match
    # the final image instead of a lower-resolution canvas preview.
    fig.set_dpi(600)
    fig.canvas.draw()
    headings = (
        r"Li$_{2.5}$TiO$_3$F$_{0.5}$ zone axes",
        r"Li$_{2.5}$TiO$_3$F$_{0.5}$ reconstructed scattering",
        "Calculated octahedral-type diffuse scattering",
    )
    for row, heading in enumerate(headings):
        boxes = [ax.get_position() for ax in matrix_axes[row]]
        fig.text(
            0.5 * (boxes[0].x0 + boxes[-1].x1),
            max(box.y1 for box in boxes) + 0.030,
            heading, fontsize=6.2, fontweight="bold",
            ha="center", va="bottom", zorder=20,
        )


def add_row_group_and_arrow(fig, matrix_axes, target_ax, *, row, box_color, arrow_color):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for ax in matrix_axes[row]
    ]
    left = min(box.x0 for box in boxes) - 0.003
    right = max(box.x1 for box in boxes) + 0.009
    bottom = min(box.y0 for box in boxes) - 0.008
    top = max(box.y1 for box in boxes) + 0.008
    edge_padding = GROUP_EDGE_PADDING_PT / (72.0 * fig.get_figheight())
    if row == 1:
        # End the red shading immediately below the x-axis labels.
        label_boxes = [
            ax.xaxis.label.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            for ax in matrix_axes[row]
        ]
        bottom = min(box.y0 for box in label_boxes) - edge_padding
    elif row == 2:
        # Begin the blue shading immediately above the panel e label, leaving
        # the shared row heading in the white space between the groups.
        panel_label = next(text for text in matrix_axes[row, 0].texts if text.get_text() == "e")
        label_box = panel_label.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        top = label_box.y1 + edge_padding
    fig.add_artist(FancyBboxPatch(
        (left, bottom), right - left, top - bottom,
        boxstyle="round,pad=0.0,rounding_size=0.010",
        transform=fig.transFigure, facecolor=box_color, edgecolor="none",
        linewidth=0, clip_on=False, zorder=-1,
    ))
    # Route the transform connectors from their corresponding scattering rows
    # into the original PDF panels through the empty corridor.
    target = target_ax.get_position()
    row_box = matrix_axes[row, -1].get_position()
    start = (right + 0.008, row_box.y0 + 0.5 * row_box.height)
    end = (target.x0 - 0.024, max(start[1], target.y0 + 0.08 * target.height))
    if row == 2:
        # Clear panel g's vertical y-axis label with the blue connector.
        start = (start[0], start[1] - BLUE_ARROW_DOWNWARD_SHIFT)
        end = (end[0], end[1] - BLUE_ARROW_DOWNWARD_SHIFT)
    fig.add_artist(FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=13, linewidth=1.9, color=arrow_color,
        clip_on=False, zorder=30,
    ))
    fig.text(
        0.5 * (start[0] + end[0]), max(start[1], end[1]) + 0.012,
        "Fourier\ntransform", fontsize=6.2, color=arrow_color,
        ha="center", va="bottom", linespacing=1.0, zorder=31,
    )


def add_reconstructed_row(fig, matrix_axes, target_ax):
    add_row_group_and_arrow(
        fig, matrix_axes, target_ax, row=1,
        box_color="#f2dede", arrow_color="#a95757",
    )


def add_calculated_row(fig, matrix_axes, target_ax):
    add_row_group_and_arrow(
        fig, matrix_axes, target_ax, row=2,
        box_color="#dbeaf1", arrow_color="#52788b",
    )


def main():
    original.OUTPUT_STEM = OUTPUT_STEM
    original.compress_scattering_rows = transpose_scattering_matrix
    original.size_and_center_pxrd = size_and_lower_pxrd
    original.add_scattering_column_titles = add_row_headings
    original.add_reconstruction_group_and_arrow = add_reconstructed_row
    original.add_calculated_group_and_arrow = add_calculated_row
    original.main()

    metadata_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt")
    # Retain data provenance and normalization, replace obsolete layout notes.
    old_metadata = metadata_path.read_text().splitlines()
    provenance = [line for line in old_metadata if (
        "source=" in line or "normalization" in line or "limit" in line
        or line.startswith(("figure_", "calculated_plane_cache=", "model_", "zone_axis_110_", "calculated_dummy_", "calculated_origin_"))
    )]
    layout = [
        "layout=transposed_nested_3x3_scattering_matrix",
        "scattering_matrix_columns=100,110,111",
        "scattering_matrix_row_1=zone_axis_100,zone_axis_110,zone_axis_111",
        "scattering_matrix_row_2=reconstructed_100,reconstructed_110,reconstructed_111",
        "scattering_matrix_row_3=calculated_100,calculated_110,calculated_111",
        "scattering_matrix_row_labels=c_zone_axis,d_reconstructed,e_calculated",
        "reconstructed_scattering_group=shared_rounded_pale_red_row",
        "calculated_scattering_group=shared_rounded_pale_blue_row",
        f"reconstructed_row_upward_shift_fraction={RECONSTRUCTED_ROW_SHIFT}",
        f"pxrd_downward_shift_fraction={PXRD_DOWNWARD_SHIFT}",
        f"group_boundary_padding_pt={GROUP_EDGE_PADDING_PT}",
        f"blue_arrow_downward_shift_fraction={BLUE_ARROW_DOWNWARD_SHIFT}",
        "red_group_bottom=bottom_of_xaxis_labels_plus_padding",
        "blue_group_top=top_of_panel_e_label_plus_padding",
        "panel_a=original_pxrd_refinement",
        "panel_b=original_structure",
        "panel_f=original_experimental_3d_deltapdf_xy0",
        "panel_g=original_calculated_integrated_3d_deltapdf_xy0",
        "data_processing=unchanged_original_plotting_helpers",
    ]
    metadata_path.write_text("\n".join(layout + provenance) + "\n")


if __name__ == "__main__":
    main()
