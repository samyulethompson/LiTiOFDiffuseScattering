#!/usr/bin/env python3
"""Figure 2: model, diffuse scattering, and PDF comparisons.

Run with: python plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT_STEM = OUTPUT / "model_pdf_71_01_combined"

MODEL_ROOT = DATA

MODEL_SCATTERING = (
    DATA
    / "rocks_run71_01_run61_02_displacements_kbt1_q10_exp20_bragg_removed_hk0_sc_list.npz"
)

EXPERIMENT_3DPDF = DATA / "3ddpdf_xy_slices_30uc_scale_2d_arrays.h5"

EXPERIMENT_SCATTERING = DATA / "scattering_planes_positive_2d_arrays.h5"

PDF_LIMIT = 2.0

SCATTERING_LIMIT = 8.0

PDF_PERCENTILE = 99.0

SCATTERING_PERCENTILE = 99.5

EXPERIMENT_VOXELS_PER_RLU = 30.0

ORIGIN_PUNCH_RADIUS_RLU = 1.2

ZONE_AXES = {
    "100": {
        "normal": np.array([1.0, 0.0, 0.0]),
        "u": np.array([0.0, 1.0, 0.0]),
        "v": np.array([0.0, 0.0, 1.0]),
        "xlabel": "$k$ (r.l.u.)",
        "ylabel": "$[001]$ (r.l.u.)",
    },
    "110": {
        "normal": np.array([1.0, 1.0, 0.0]),
        "u": np.array([1.0, -1.0, 0.0]),
        "v": np.array([0.0, 0.0, 1.0]),
        "xlabel": "$[1\\bar{1}0]$ (r.l.u.)",
        "ylabel": "$[001]$ (r.l.u.)",
    },
    "111": {
        "normal": np.array([1.0, 1.0, 1.0]),
        "u": np.array([1.0, -1.0, 0.0]),
        "v": np.array([1.0, 1.0, -2.0]),
        "xlabel": "$[1\\bar{1}0]$ (r.l.u.)",
        "ylabel": "$[11\\bar{2}]$ (r.l.u.)",
    },
    "112": {
        "normal": np.array([1.0, 1.0, 2.0]),
        "u": np.array([1.0, -1.0, 0.0]),
        "v": np.array([1.0, 1.0, -1.0]),
        "xlabel": "$[1\\bar{1}0]$ (r.l.u.)",
        "ylabel": "$[11\\bar{1}]$ (r.l.u.)",
    },
}


def set_publication_style() -> None:
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


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def percentile_limit(values: np.ndarray, percentile: float, absolute: bool) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot normalize an empty half-plane")
    if absolute:
        finite = np.abs(finite)
    limit = float(np.percentile(finite, percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Half-plane normalization limit is not positive")
    return limit


def combine_halves(
    experiment: np.ndarray,
    model: np.ndarray,
    x_grid: np.ndarray,
    percentile: float,
    signed: bool,
) -> tuple[np.ndarray, float, float]:
    exp_mask = x_grid < 0.0
    model_mask = ~exp_mask
    exp_limit = percentile_limit(experiment[exp_mask], percentile, signed)
    model_limit = percentile_limit(model[model_mask], percentile, signed)
    if signed:
        exp_scaled = np.clip(experiment / exp_limit, -1.0, 1.0)
        model_scaled = np.clip(model / model_limit, -1.0, 1.0)
    else:
        exp_scaled = np.clip(experiment / exp_limit, 0.0, 1.0)
        model_scaled = np.clip(model / model_limit, 0.0, 1.0)
    composite = np.full_like(experiment, np.nan, dtype=float)
    composite[exp_mask] = exp_scaled[exp_mask]
    composite[model_mask] = model_scaled[model_mask]
    return (composite, exp_limit, model_limit)


def resample_regular(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (y_axis, x_axis), image, method="nearest", bounds_error=False, fill_value=np.nan
    )
    points = np.column_stack((y_grid.ravel(), x_grid.ravel()))
    return interpolator(points).reshape(x_grid.shape)


def load_model_pdf_slice(target_z: float) -> tuple[np.ndarray, np.ndarray]:
    volume = np.load(MODEL_ROOT / "3dpdf_volume.npz")["volume"]
    axis = np.loadtxt(MODEL_ROOT / "direct_axis_lattice_units.txt")
    upper = int(np.searchsorted(axis, target_z))
    upper = int(np.clip(upper, 0, len(axis) - 1))
    lower = max(0, upper - 1)
    if lower == upper or np.isclose(axis[lower], axis[upper]):
        return (np.asarray(volume[upper]), axis)
    weight = (target_z - axis[lower]) / (axis[upper] - axis[lower])
    return ((1.0 - weight) * volume[lower] + weight * volume[upper], axis)


def make_pdf_composite(
    dataset_name: str, target_z: float
) -> tuple[np.ndarray, float, float]:
    with h5py.File(EXPERIMENT_3DPDF, "r") as handle:
        experiment = np.asarray(handle[dataset_name][:], dtype=float)
        pixels_per_cell = float(handle.attrs["pixels_per_unit_cell"])
    exp_x = np.linspace(
        -(experiment.shape[1] - 1) / (2.0 * pixels_per_cell),
        +(experiment.shape[1] - 1) / (2.0 * pixels_per_cell),
        experiment.shape[1],
    )
    exp_y = np.linspace(
        -(experiment.shape[0] - 1) / (2.0 * pixels_per_cell),
        +(experiment.shape[0] - 1) / (2.0 * pixels_per_cell),
        experiment.shape[0],
    )
    model, model_axis = load_model_pdf_slice(target_z)
    target_axis = np.linspace(-PDF_LIMIT, PDF_LIMIT, 321)
    x_grid, y_grid = np.meshgrid(target_axis, target_axis)
    exp_grid = resample_regular(experiment, exp_x, exp_y, x_grid, y_grid)
    model_grid = resample_regular(model, model_axis, model_axis, x_grid, y_grid)
    return combine_halves(exp_grid, model_grid, x_grid, PDF_PERCENTILE, signed=True)


def estimate_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.round(values, 10))
    differences = np.diff(np.sort(unique))
    differences = differences[differences > 1e-10]
    return float(np.min(differences))


def load_model_scattering() -> tuple[np.ndarray, np.ndarray]:
    with np.load(MODEL_SCATTERING) as archive:
        data = np.column_stack([archive[key] for key in ("h", "k", "l", "intensity")])
    q = data[:, :3]
    intensity = data[:, 3]
    intensity[np.sum(q * q, axis=1) <= ORIGIN_PUNCH_RADIUS_RLU**2 + 1e-12] = 0.0
    return (q, intensity)


def experimental_scattering_coordinates(
    handle: h5py.File, dataset: h5py.Dataset, zone: str
) -> tuple[np.ndarray, np.ndarray]:
    spec = ZONE_AXES[zone]
    target_u = normalize(spec["u"])
    target_v = normalize(spec["v"])
    source_u = np.asarray(dataset.attrs["u_axis_unit_vector_xyz"], dtype=float)
    source_v = np.asarray(dataset.attrs["v_axis_unit_vector_xyz"], dtype=float)
    u = np.asarray(handle["u_axis_voxel_units"][:]) / EXPERIMENT_VOXELS_PER_RLU
    v = np.asarray(handle["v_axis_voxel_units"][:]) / EXPERIMENT_VOXELS_PER_RLU
    u_grid, v_grid = np.meshgrid(u, v)
    x = u_grid * float(source_u @ target_u) + v_grid * float(source_v @ target_u)
    y = u_grid * float(source_u @ target_v) + v_grid * float(source_v @ target_v)
    return (x, y)


def make_scattering_composite(
    q: np.ndarray, intensity: np.ndarray, zone: str, handle: h5py.File
) -> tuple[np.ndarray, float, float]:
    spec = ZONE_AXES[zone]
    normal = normalize(spec["normal"])
    axis_u = normalize(spec["u"])
    axis_v = normalize(spec["v"])
    plane_distance = q @ normal
    in_plane = np.abs(plane_distance) <= 0.51 * estimate_spacing(plane_distance)
    model_q = q[in_plane]
    model_x = model_q @ axis_u
    model_y = model_q @ axis_v
    model_values = intensity[in_plane]
    exp_dataset = handle[f"scattering_filled/{zone}"]
    exp_values = np.asarray(exp_dataset[:], dtype=float)
    exp_x, exp_y = experimental_scattering_coordinates(handle, exp_dataset, zone)
    margin = 0.25
    exp_keep = (
        (np.abs(exp_x) <= SCATTERING_LIMIT + margin)
        & (np.abs(exp_y) <= SCATTERING_LIMIT + margin)
        & np.isfinite(exp_values)
    )
    model_keep = (
        (np.abs(model_x) <= SCATTERING_LIMIT + margin)
        & (np.abs(model_y) <= SCATTERING_LIMIT + margin)
        & np.isfinite(model_values)
    )
    target_size = int(2.0 * SCATTERING_LIMIT * EXPERIMENT_VOXELS_PER_RLU) + 1
    target_axis = np.linspace(-SCATTERING_LIMIT, SCATTERING_LIMIT, target_size)
    x_grid, y_grid = np.meshgrid(target_axis, target_axis)
    exp_grid = griddata(
        np.column_stack((exp_x[exp_keep], exp_y[exp_keep])),
        exp_values[exp_keep],
        (x_grid, y_grid),
        method="linear",
    )
    model_grid = griddata(
        np.column_stack((model_x[model_keep], model_y[model_keep])),
        model_values[model_keep],
        (x_grid, y_grid),
        method="linear",
    )
    return combine_halves(
        exp_grid, model_grid, x_grid, SCATTERING_PERCENTILE, signed=False
    )


def label_halfplanes(ax: plt.Axes) -> None:
    text_style = {
        "transform": ax.transAxes,
        "fontsize": 5.8,
        "va": "top",
        "color": "black",
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        "zorder": 5,
    }
    ax.text(0.025, 0.975, "Experiment", ha="left", **text_style)
    ax.text(0.975, 0.975, "Model", ha="right", **text_style)
    ax.axvline(0.0, color="black", linewidth=0.65, zorder=4)


def set_crystallographic_ticks(ax: plt.Axes, spec: dict[str, object]) -> None:
    for axis_name, vector in (("x", spec["u"]), ("y", spec["v"])):
        vector_length = float(np.linalg.norm(np.asarray(vector, dtype=float)))
        largest_coefficient = int(np.floor(SCATTERING_LIMIT / vector_length + 1e-09))
        coefficients = np.arange(-largest_coefficient, largest_coefficient + 1)
        major_step = 2 if largest_coefficient >= 5 else 1
        major_coefficients = coefficients[coefficients % major_step == 0]
        minor_coefficients = coefficients[coefficients % major_step != 0]
        major_positions = major_coefficients * vector_length
        minor_positions = minor_coefficients * vector_length
        labels = [str(int(value)) for value in major_coefficients]
        if axis_name == "x":
            ax.set_xticks(major_positions, labels=labels)
            ax.set_xticks(minor_positions, minor=True)
        else:
            ax.set_yticks(major_positions, labels=labels)
            ax.set_yticks(minor_positions, minor=True)


PDF_DIR = DATA

MODELS = (
    {
        "label": "Average",
        "residual_label": "Average",
        "fit": "LiTiOF_avg_PDF_model.txt",
        "color": "#7A4EAB",
        "linestyle": "-",
        "rwp": 18.87,
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
            raise ValueError(
                f"Expected four columns in {model['fit']}; got {values.shape}"
            )
        record["data"] = values
        records.append(record)
    return records


def add_pdf_panels(fig, records, fit_rect=None, residual_rect=None):
    if fit_rect is None:
        fit_rect = [0.42, 0.385, 0.545, 0.565]
    if residual_rect is None:
        residual_rect = [0.42, 0.095, 0.545, 0.205]
    fit_ax = fig.add_axes(fit_rect)
    residual_ax = fig.add_axes(residual_rect)
    all_y = np.concatenate([record["data"][:, 1:3].ravel() for record in records])
    data_span = float(np.nanpercentile(all_y, 99.5) - np.nanpercentile(all_y, 0.5))
    offset_step = 1.15 * data_span
    offsets = offset_step * np.arange(len(records) - 1, -1, -1)
    y_min = float(np.nanpercentile(all_y, 0.5) - 0.1 * data_span)
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
            r, calculated + offset, color=record["color"], lw=1.35, label="_nolegend_"
        )
        fit_ax.text(
            0.985,
            offset + 0.48 * data_span,
            record["label"] + f"   $R_{{wp}}$ = {record['rwp']:.2f}%",
            transform=fit_ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=record["color"],
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
        )
    peak_label_y = offsets[1] + 0.6 * offset_step
    for peak_number, peak_r in enumerate((2.0, 3.0, 3.6, 4.1), start=1):
        fit_ax.axvline(peak_r, color="#8A8A8A", lw=0.7, ls=(0, (3, 3)), zorder=0.4)
        fit_ax.text(
            peak_r,
            peak_label_y,
            f"P{peak_number}",
            ha="center",
            va="center",
            color="#666666",
            fontsize=6.3,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 0.35,
            },
        )
    fit_ax.set_xlim(1.5, 10.0)
    fit_ax.set_ylim(y_min, y_max)
    fit_ax.set_ylabel(
        "Pair distribution function, $G(r)$ ($\\mathrm{\\AA}^{-2}$)", labelpad=2
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
    max_abs_residual = max((np.max(np.abs(record["data"][:, 3])) for record in records))
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
    residual_ax.set_xlabel("Pair distance, $r$ ($\\mathrm{\\AA}$)")
    residual_ax.set_ylabel("$\\Delta G(r)$ ($\\mathrm{\\AA}^{-2}$)", labelpad=2)
    residual_ax.spines[["top", "right"]].set_visible(False)
    residual_ax.tick_params(direction="out")
    residual_ax.legend(
        loc="upper center",
        ncol=3,
        handlelength=1.6,
        columnspacing=0.9,
        labelspacing=0.2,
        bbox_to_anchor=(0.57, 0.995),
        borderaxespad=0,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
    )
    return (fit_ax, residual_ax)


SCHEMATIC = DATA / "model_schematic2.png"

FIGURE_WIDTH_IN = 7.2

FIGURE_HEIGHT_IN = 9.0

LEFT_X = 0.127

LEFT_WIDTH = 0.185

LEFT_HEIGHT = LEFT_WIDTH * FIGURE_WIDTH_IN / FIGURE_HEIGHT_IN

LEFT_BOTTOMS = (0.482, 0.266, 0.05)

RIGHT_X = 0.41

RIGHT_WIDTH = 0.56

SCHEMATIC_CROP = (0, 2529, 23, 4251)

SCHEMATIC_ASPECT = (SCHEMATIC_CROP[3] - SCHEMATIC_CROP[2]) / (
    SCHEMATIC_CROP[1] - SCHEMATIC_CROP[0]
)

TOP_PDF_WIDTH = (LEFT_WIDTH + RIGHT_WIDTH) / (1.0 + SCHEMATIC_ASPECT)

TOP_SCHEMATIC_WIDTH = SCHEMATIC_ASPECT * TOP_PDF_WIDTH

TOP_HEIGHT = TOP_PDF_WIDTH * FIGURE_WIDTH_IN / FIGURE_HEIGHT_IN

ABCD_SHIFT_X = 0.045

TOP_SCHEMATIC_X = LEFT_X + 0.5 * (LEFT_WIDTH - TOP_PDF_WIDTH) + ABCD_SHIFT_X

TOP_PDF_RIGHT_SHIFT = 0.015

TOP_PDF_X = (
    RIGHT_X
    + 0.5 * (RIGHT_WIDTH + TOP_SCHEMATIC_WIDTH)
    - TOP_PDF_WIDTH
    + TOP_PDF_RIGHT_SHIFT
)

TOP_BOTTOM = 0.71

LOWER_LEFT_X = TOP_SCHEMATIC_X


def add_final_model_panels(fig: plt.Figure):
    axes = [fig.add_axes([TOP_PDF_X, TOP_BOTTOM, TOP_PDF_WIDTH, TOP_HEIGHT])]
    for bottom in LEFT_BOTTOMS:
        axes.append(fig.add_axes([LOWER_LEFT_X, bottom, LEFT_WIDTH, LEFT_HEIGHT]))
    pdf_cmap = mpl.colormaps["RdBu_r"].copy()
    pdf_cmap.set_bad("white")
    composite, exp_limit, model_limit = make_pdf_composite("3ddpdf/xy0", 0.0)
    axes[0].imshow(
        composite,
        origin="lower",
        extent=[-PDF_LIMIT, PDF_LIMIT] * 2,
        cmap=pdf_cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    axes[0].set(
        xlim=(-PDF_LIMIT, PDF_LIMIT),
        ylim=(-PDF_LIMIT, PDF_LIMIT),
        xticks=np.arange(-2, 3, 1),
        yticks=np.arange(-2, 3, 1),
        xlabel="$x$ (unit cells)",
        ylabel="$y$ (unit cells)",
        title="3D-$\\Delta$PDF, $z=0$",
    )
    axes[0].set_xticks(np.arange(-2, 2.1, 0.5), minor=True)
    axes[0].set_yticks(np.arange(-2, 2.1, 0.5), minor=True)
    label_halfplanes(axes[0])
    q, intensity = load_model_scattering()
    scattering_cmap = mpl.colormaps["gray_r"].copy()
    scattering_cmap.set_bad("white")
    with h5py.File(EXPERIMENT_SCATTERING, "r") as handle:
        for ax, zone in zip(axes[1:], ("100", "110", "111")):
            composite, exp_limit, model_limit = make_scattering_composite(
                q, intensity, zone, handle
            )
            ax.imshow(
                composite,
                origin="lower",
                extent=[-SCATTERING_LIMIT, SCATTERING_LIMIT] * 2,
                cmap=scattering_cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                rasterized=True,
            )
            spec = ZONE_AXES[zone]
            ax.set(
                xlim=(-SCATTERING_LIMIT, SCATTERING_LIMIT),
                ylim=(-SCATTERING_LIMIT, SCATTERING_LIMIT),
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
                title=f"Diffuse scattering, $[{zone}]$",
            )
            set_crystallographic_ticks(ax, spec)
            label_halfplanes(ax)
    for ax in axes:
        ax.set_box_aspect(1)
    return axes


def add_panel_letter(fig: plt.Figure, label: str, *, x: float, y: float):
    fig.text(x, y, label, ha="left", va="top", fontsize=9, fontweight="bold")


def main() -> None:
    style_pdf_axes()
    set_publication_style()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False
    )
    add_final_model_panels(fig)
    schematic_ax = fig.add_axes(
        [TOP_SCHEMATIC_X, TOP_BOTTOM, TOP_SCHEMATIC_WIDTH, TOP_HEIGHT]
    )
    schematic_image = plt.imread(SCHEMATIC)
    top, bottom, left, right = SCHEMATIC_CROP
    schematic_ax.imshow(schematic_image[top:bottom, left:right], interpolation="none")
    schematic_ax.set_aspect("equal", adjustable="box")
    schematic_ax.axis("off")
    records = read_records()
    fit_ax, residual_ax = add_pdf_panels(
        fig,
        records,
        fit_rect=[RIGHT_X, 0.26, RIGHT_WIDTH, 0.37],
        residual_rect=[RIGHT_X, 0.05, RIGHT_WIDTH, 0.16],
    )
    left_label_x = LOWER_LEFT_X - 0.18 * LEFT_WIDTH
    right_label_x = RIGHT_X - 0.03
    top_label_y = TOP_BOTTOM + 1.08 * TOP_HEIGHT
    scattering_label_y = tuple((bottom + 1.08 * LEFT_HEIGHT for bottom in LEFT_BOTTOMS))
    label_positions = (
        ("a", left_label_x, top_label_y),
        ("b", TOP_PDF_X - 0.1 * TOP_PDF_WIDTH, top_label_y),
        ("c", left_label_x, scattering_label_y[0]),
        ("d", left_label_x, scattering_label_y[1]),
        ("e", left_label_x, scattering_label_y[2]),
        ("f", right_label_x, scattering_label_y[0]),
        ("g", right_label_x, scattering_label_y[2]),
    )
    for label, x, y in label_positions:
        add_panel_letter(fig, label, x=x, y=y)
    save_options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.01}
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, **save_options)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), **save_options)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), **save_options)
    plt.close(fig)
    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    main()
