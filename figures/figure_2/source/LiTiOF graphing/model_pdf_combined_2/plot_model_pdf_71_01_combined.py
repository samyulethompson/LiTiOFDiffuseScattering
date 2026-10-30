#!/usr/bin/env python3
"""Combine selected final-model 71_01 planes with the model and PDF panels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
GRAPH_ROOT = HERE.parent
PDF_SCRIPT = HERE / "plot_model_pdf_combined.py"
FINAL_MODEL_SCRIPT = GRAPH_ROOT / "final_model_71_01" / "plot_final_model_71_01.py"
SCHEMATIC = HERE / "model_schematic2.png"
OUTPUT_STEM = HERE / "model_pdf_71_01_combined"

FIGURE_WIDTH_IN = 7.20
FIGURE_HEIGHT_IN = 9.00
LEFT_X = 0.127
LEFT_WIDTH = 0.185
LEFT_HEIGHT = LEFT_WIDTH * FIGURE_WIDTH_IN / FIGURE_HEIGHT_IN
LEFT_BOTTOMS = (0.482, 0.266, 0.050)
RIGHT_X = 0.410
RIGHT_WIDTH = 0.560
# The source PNG contains a 23 px white margin at left and 71 px at bottom.
# Use the measured non-white bounding box so width balancing is based on the
# visible schematic rather than the padded raster canvas.
SCHEMATIC_CROP = (0, 2529, 23, 4251)  # top, bottom, left, right
SCHEMATIC_ASPECT = (SCHEMATIC_CROP[3] - SCHEMATIC_CROP[2]) / (
    SCHEMATIC_CROP[1] - SCHEMATIC_CROP[0]
)
TOP_PDF_WIDTH = (LEFT_WIDTH + RIGHT_WIDTH) / (1.0 + SCHEMATIC_ASPECT)
TOP_SCHEMATIC_WIDTH = SCHEMATIC_ASPECT * TOP_PDF_WIDTH
TOP_HEIGHT = TOP_PDF_WIDTH * FIGURE_WIDTH_IN / FIGURE_HEIGHT_IN
ABCD_SHIFT_X = 0.045
TOP_SCHEMATIC_X = LEFT_X + 0.5 * (LEFT_WIDTH - TOP_PDF_WIDTH) + ABCD_SHIFT_X
# Keep each panel's physical size and aspect, with a little extra clearance
# between the schematic legend and panel b's y-axis label.
TOP_PDF_RIGHT_SHIFT = 0.015
TOP_PDF_X = (
    RIGHT_X + 0.5 * (RIGHT_WIDTH + TOP_SCHEMATIC_WIDTH)
    - TOP_PDF_WIDTH + TOP_PDF_RIGHT_SHIFT
)
TOP_BOTTOM = 0.710
LOWER_LEFT_X = TOP_SCHEMATIC_X


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_final_model():
    wrapper = load_module("final_model_71_01_wrapper", FINAL_MODEL_SCRIPT)
    plotting = wrapper.load_template()
    plotting.MODEL_NAME = "71_01"
    plotting.MODEL_DESCRIPTION = (
        "run 71 case 01; J1/kBT=-0.5, J2/kBT=+0.5; "
        "run 61_02 displacements"
    )
    plotting.MODEL_ROOT = wrapper.MODEL_ROOT
    plotting.MODEL_SCATTERING = (
        wrapper.MODEL_ROOT
        / "scatty_structures"
        / "rocks_run71_01_run61_02_displacements_kbt1_q10_exp20_bragg_removed_hk0_sc_list.npz"
    )
    return plotting


def add_final_model_panels(fig: plt.Figure, plotting):
    axes = [fig.add_axes([TOP_PDF_X, TOP_BOTTOM, TOP_PDF_WIDTH, TOP_HEIGHT])]
    for bottom in LEFT_BOTTOMS:
        axes.append(
            fig.add_axes([LOWER_LEFT_X, bottom, LEFT_WIDTH, LEFT_HEIGHT])
        )
    metadata = []

    pdf_cmap = mpl.colormaps["RdBu_r"].copy()
    pdf_cmap.set_bad("white")
    composite, exp_limit, model_limit = plotting.make_pdf_composite("3ddpdf/xy0", 0.0)
    axes[0].imshow(
        composite,
        origin="lower",
        extent=[-plotting.PDF_LIMIT, plotting.PDF_LIMIT] * 2,
        cmap=pdf_cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    axes[0].set(
        xlim=(-plotting.PDF_LIMIT, plotting.PDF_LIMIT),
        ylim=(-plotting.PDF_LIMIT, plotting.PDF_LIMIT),
        xticks=np.arange(-2, 3, 1),
        yticks=np.arange(-2, 3, 1),
        xlabel=r"$x$ (unit cells)",
        ylabel=r"$y$ (unit cells)",
        title=r"3D-$\Delta$PDF, $z=0$",
    )
    axes[0].set_xticks(np.arange(-2, 2.1, 0.5), minor=True)
    axes[0].set_yticks(np.arange(-2, 2.1, 0.5), minor=True)
    plotting.label_halfplanes(axes[0])
    metadata.extend(
        [
            f"3ddpdf_z0_experiment_abs_limit={exp_limit:.12g}",
            f"3ddpdf_z0_model_abs_limit={model_limit:.12g}",
        ]
    )

    q, intensity = plotting.load_model_scattering()
    scattering_cmap = mpl.colormaps["gray_r"].copy()
    scattering_cmap.set_bad("white")
    with h5py.File(plotting.EXPERIMENT_SCATTERING, "r") as handle:
        for ax, zone in zip(axes[1:], ("100", "110", "111")):
            composite, exp_limit, model_limit = plotting.make_scattering_composite(
                q, intensity, zone, handle
            )
            ax.imshow(
                composite,
                origin="lower",
                extent=[-plotting.SCATTERING_LIMIT, plotting.SCATTERING_LIMIT] * 2,
                cmap=scattering_cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                rasterized=True,
            )
            spec = plotting.ZONE_AXES[zone]
            ax.set(
                xlim=(-plotting.SCATTERING_LIMIT, plotting.SCATTERING_LIMIT),
                ylim=(-plotting.SCATTERING_LIMIT, plotting.SCATTERING_LIMIT),
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
                title=rf"Diffuse scattering, $[{zone}]$",
            )
            plotting.set_crystallographic_ticks(ax, spec)
            plotting.label_halfplanes(ax)
            metadata.extend(
                [
                    f"scattering_{zone}_experiment_limit={exp_limit:.12g}",
                    f"scattering_{zone}_model_limit={model_limit:.12g}",
                ]
            )

    # Panel letters are added together in figure coordinates after all panels
    # exist, so common rows and columns use exactly the same anchors.
    for ax in axes:
        ax.set_box_aspect(1)
    return axes, metadata


def add_panel_letter(fig: plt.Figure, label: str, *, x: float, y: float):
    fig.text(
        x,
        y,
        label,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def main() -> None:
    pdf = load_module("model_pdf_combined_base", PDF_SCRIPT)
    final_model = configure_final_model()
    pdf.style_pdf_axes()
    final_model.set_publication_style()

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False
    )
    _, metadata = add_final_model_panels(fig, final_model)

    schematic_ax = fig.add_axes(
        [TOP_SCHEMATIC_X, TOP_BOTTOM, TOP_SCHEMATIC_WIDTH, TOP_HEIGHT]
    )
    schematic_image = plt.imread(SCHEMATIC)
    top, bottom, left, right = SCHEMATIC_CROP
    schematic_ax.imshow(
        schematic_image[top:bottom, left:right], interpolation="none"
    )
    schematic_ax.set_aspect("equal", adjustable="box")
    schematic_ax.axis("off")

    records = pdf.read_records()
    fit_ax, residual_ax = pdf.add_pdf_panels(
        fig,
        records,
        fit_rect=[RIGHT_X, 0.260, RIGHT_WIDTH, 0.370],
        residual_rect=[RIGHT_X, 0.050, RIGHT_WIDTH, 0.160],
    )
    left_label_x = LOWER_LEFT_X - 0.18 * LEFT_WIDTH
    right_label_x = RIGHT_X - 0.030
    top_label_y = TOP_BOTTOM + 1.08 * TOP_HEIGHT
    scattering_label_y = tuple(bottom + 1.08 * LEFT_HEIGHT for bottom in LEFT_BOTTOMS)
    label_positions = (
        ("a", left_label_x, top_label_y),
        ("b", TOP_PDF_X - 0.10 * TOP_PDF_WIDTH, top_label_y),
        ("c", left_label_x, scattering_label_y[0]),
        ("d", left_label_x, scattering_label_y[1]),
        ("e", left_label_x, scattering_label_y[2]),
        ("f", right_label_x, scattering_label_y[0]),
        ("g", right_label_x, scattering_label_y[2]),
    )
    for label, x, y in label_positions:
        add_panel_letter(fig, label, x=x, y=y)

    save_options = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.01,
    }
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, **save_options)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), **save_options)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), **save_options)
    plt.close(fig)

    metadata = [
        "layout=schematic_a_top_left;3ddpdf_b_top_right;scattering_c_d_e_lower_left;pdf_fits_f_and_residuals_g_lower_right",
        "panel_letter_order=top_row_a_b;diffuse_scattering_c_d_e;pdf_fits_f;residuals_g",
        "panel_letter_column_alignment=a_c_d_e;f_g",
        "panel_letter_row_alignment=a_b;c_f;e_g",
        "top_row_width_balance=width_a_plus_width_b_equals_width_c_plus_width_f",
        "top_row_height_balance=panels_a_and_b_have_equal_physical_height",
        "left_alignment=schematic_a_and_scattering_c_d_e_share_left_edge",
        "schematic_blank_border_crop_px=top:0,bottom:71,left:23,right:0",
        "schematic_a_and_scattering_c_d_e_horizontal_shift_canvas_fraction=+0.045",
        f"panel_b_rightward_shift_canvas_fraction={TOP_PDF_RIGHT_SHIFT}",
        "top_row_vertical_shift_canvas_fraction=-0.035",
        "composite_panel_mapping=a:model_schematic2,b:final_model_71_01_a,c:final_model_71_01_b,d:final_model_71_01_c,e:final_model_71_01_e,f:pdf_fits,g:pdf_residuals",
        f"schematic_source={SCHEMATIC}",
        f"model_root={final_model.MODEL_ROOT}",
        f"model_scattering_source={final_model.MODEL_SCATTERING}",
        f"experimental_3ddpdf_source={final_model.EXPERIMENT_3DPDF}",
        f"experimental_scattering_source={final_model.EXPERIMENT_SCATTERING}",
        *(f"pdf_fit_source={HERE / record['fit']}" for record in records),
        *metadata,
    ]
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )
    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
