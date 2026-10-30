#!/usr/bin/env python3
"""Combine the displacement model with the PDF fits and residuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SCHEMATIC_PNG = HERE / "model_schematic.png"
PDF_DIR = HERE

MODELS = (
    {
        "label": "Average",
        "residual_label": "Average",
        "fit": "LiTiOF_avg_PDF_model.txt",
        "color": "#7A4EAB",
        "linestyle": "-",
        "rwp": 19.73,
    },
    {
        "label": "Occupancy",
        "residual_label": "Occ.",
        "fit": "LiTiOF_occ_only_PDF_model.txt",
        "color": "#D47A1F",
        "linestyle": "--",
        "rwp": 18.62,
    },
    {
        "label": "Occ. + disp.",
        "residual_label": "Occ. + disp.",
        "fit": "LiTiOF_occ_disp_PDF_model.txt",
        "color": "#168C82",
        "linestyle": "-",
        "rwp": 15.16,
    },
)


def style_pdf_axes() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_records():
    records = []
    for model in MODELS:
        record = dict(model)
        values = np.loadtxt(PDF_DIR / model["fit"], delimiter=",")
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(f"Expected four columns in {model['fit']}; got {values.shape}")
        record["data"] = values
        records.append(record)
    return records


def add_pdf_panels(fig, records, fit_rect=None, residual_rect=None):
    # The PDF panels extend into the former external-legend margin.
    if fit_rect is None:
        fit_rect = [0.420, 0.385, 0.545, 0.565]
    if residual_rect is None:
        residual_rect = [0.420, 0.095, 0.545, 0.205]
    fit_ax = fig.add_axes(fit_rect)
    residual_ax = fig.add_axes(residual_rect)

    all_y = np.concatenate([record["data"][:, 1:3].ravel() for record in records])
    data_span = float(np.nanpercentile(all_y, 99.5) - np.nanpercentile(all_y, 0.5))
    offset_step = 1.15 * data_span
    offsets = offset_step * np.arange(len(records) - 1, -1, -1)
    y_min = float(np.nanpercentile(all_y, 0.5) - 0.10 * data_span)
    y_max = float(np.nanpercentile(all_y, 99.5) + offsets[0] + 0.12 * data_span)

    for index, (record, offset) in enumerate(zip(records, offsets)):
        r, observed, calculated, _ = record["data"].T
        fit_ax.axhline(offset, color="#BEBEBE", lw=0.45, zorder=0)
        fit_ax.plot(
            r,
            observed + offset,
            color="#2B2B2B",
            lw=1.05,
            label="Observed" if index == 0 else "_nolegend_",
        )
        fit_ax.plot(
            r,
            calculated + offset,
            color=record["color"],
            lw=1.35,
            label="_nolegend_",
        )
        fit_ax.text(
            0.985,
            offset + 0.48 * data_span,
            record["label"] + rf"   $R_{{wp}}$ = {record['rwp']:.2f}%",
            transform=fit_ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=record["color"],
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
        )

    peak_label_y = offsets[1] + 0.60 * offset_step
    for peak_number, peak_r in enumerate((2.0, 3.0, 3.6, 4.1), start=1):
        fit_ax.axvline(
            peak_r,
            color="#8A8A8A",
            lw=0.70,
            ls=(0, (3, 3)),
            zorder=0.4,
        )
        fit_ax.text(
            peak_r,
            peak_label_y,
            f"P{peak_number}",
            ha="center",
            va="center",
            color="#666666",
            fontsize=6.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
        )

    fit_ax.set_xlim(1.5, 10.0)
    fit_ax.set_ylim(y_min, y_max)
    fit_ax.set_ylabel(
        r"Pair distribution function, $G(r)$ ($\mathrm{\AA}^{-2}$)",
        labelpad=2,
    )
    fit_ax.spines[["top", "right"]].set_visible(False)
    fit_ax.tick_params(direction="out", labelbottom=False)
    fit_ax.legend(
        loc="upper left",
        handlelength=1.8,
        bbox_to_anchor=(0.012, 0.988),
        borderaxespad=0,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
    )

    max_abs_residual = max(
        np.max(np.abs(record["data"][:, 3])) for record in records
    )
    for record in records:
        r, _, _, residual = record["data"].T
        residual_ax.plot(
            r,
            residual,
            color=record["color"],
            lw=1.25,
            ls=record["linestyle"],
            label=record["residual_label"],
        )
    residual_ax.axhline(0, color="#707070", lw=0.55, zorder=0)
    residual_ax.set_xlim(1.5, 10.0)
    residual_ax.set_ylim(-1.08 * max_abs_residual, 1.08 * max_abs_residual)
    residual_ax.set_xlabel(r"Pair distance, $r$ ($\mathrm{\AA}$)")
    residual_ax.set_ylabel(r"$\Delta G(r)$ ($\mathrm{\AA}^{-2}$)", labelpad=2)
    residual_ax.spines[["top", "right"]].set_visible(False)
    residual_ax.tick_params(direction="out")
    residual_ax.legend(
        loc="upper center",
        ncol=3,
        handlelength=1.6,
        columnspacing=0.9,
        labelspacing=0.20,
        bbox_to_anchor=(0.57, 0.995),
        borderaxespad=0,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
    )

    return fit_ax, residual_ax


def main() -> None:
    style_pdf_axes()
    fig = plt.figure(figsize=(7.20, 5.30), constrained_layout=False)

    # Treat the complete standalone schematic as one panel.  This preserves
    # its original aspect ratio and scales atoms, labels, arrows and spacing
    # together, avoiding the fixed-font-size overlap caused by resizing its
    # individual Matplotlib artists inside the composite.
    model_ax = fig.add_axes([0.025, 0.10, 0.305, 0.80])
    model_ax.imshow(plt.imread(SCHEMATIC_PNG), interpolation="none")
    model_ax.set_aspect("equal", adjustable="box")
    model_ax.axis("off")

    records = read_records()
    fit_ax, residual_ax = add_pdf_panels(fig, records)

    label_style = {
        "ha": "left",
        "va": "top",
        "fontweight": "bold",
        "fontsize": 9,
    }
    fig.text(0.008, fit_ax.get_position().y1, "a", **label_style)
    fig.text(fit_ax.get_position().x0 - 0.070, fit_ax.get_position().y1, "b", **label_style)
    fig.text(
        residual_ax.get_position().x0 - 0.070,
        residual_ax.get_position().y1,
        "c",
        **label_style,
    )

    output_stem = HERE / "model_pdf_combined"
    save_options = {
        "bbox_inches": "tight",
        "pad_inches": 0.01,
        "facecolor": "white",
    }
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, **save_options)
    fig.savefig(output_stem.with_suffix(".pdf"), **save_options)
    fig.savefig(output_stem.with_suffix(".svg"), **save_options)
    plt.close(fig)

    for extension in ("png", "pdf", "svg"):
        print(f"Saved: {output_stem.with_suffix('.' + extension)}")


if __name__ == "__main__":
    main()
