#!/usr/bin/env python3
"""Plot six experimental/model half-plane comparisons for final model 61_02."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata


HERE = Path(__file__).resolve().parent
PAIR_ROOT = HERE.parents[1]
DATA_ROOT = PAIR_ROOT / "LTiOF_dataset"
MODEL_ROOT = (
    PAIR_ROOT
    / "3d-deltaPDF_informed"
    / "61_run60_double_100_cation_modes"
    / "02_tili"
)
MODEL_SCATTERING = (
    MODEL_ROOT
    / "scatty_structures"
    / "rocks_run61_02_tili_q10_exp20_bragg_removed_hk0_sc_list.txt"
)
EXPERIMENT_3DPDF = (
    DATA_ROOT
    / "3ddpdf_xy_slices_30uc_scale"
    / "3ddpdf_xy_slices_30uc_scale_2d_arrays.h5"
)
EXPERIMENT_SCATTERING = (
    DATA_ROOT
    / "scattering_planes_positive_greyscale"
    / "scattering_planes_positive_2d_arrays.h5"
)
OUTPUT_STEM = HERE / "final_model_61_02_halfplane_comparison"
MODEL_NAME = "61_02"
MODEL_DESCRIPTION = "run 61 case 02_tili"

PDF_LIMIT = 2.0
SCATTERING_LIMIT = 8.0
PDF_PERCENTILE = 99.0
SCATTERING_PERCENTILE = 99.5
EXPERIMENT_VOXELS_PER_RLU = 30.0
ORIGIN_PUNCH_RADIUS_RLU = 1.2

PDF_SLICES = (
    ("xy0", "3ddpdf/xy0", 0.0, r"3D-$\Delta$PDF, $z=0$"),
    ("xy0p5", "3ddpdf/xy0p5", 0.5, r"3D-$\Delta$PDF, $z=\frac{1}{2}$"),
)
SCATTERING_ZONES = ("100", "110", "111", "112")
ZONE_AXES = {
    "100": {
        "normal": np.array([1.0, 0.0, 0.0]),
        "u": np.array([0.0, 1.0, 0.0]),
        "v": np.array([0.0, 0.0, 1.0]),
        "xlabel": r"$k$ (r.l.u.)",
        "ylabel": r"$[001]$ (r.l.u.)",
    },
    "110": {
        "normal": np.array([1.0, 1.0, 0.0]),
        "u": np.array([1.0, -1.0, 0.0]),
        "v": np.array([0.0, 0.0, 1.0]),
        "xlabel": r"$[1\bar{1}0]$ (r.l.u.)",
        "ylabel": r"$[001]$ (r.l.u.)",
    },
    "111": {
        "normal": np.array([1.0, 1.0, 1.0]),
        "u": np.array([1.0, -1.0, 0.0]),
        "v": np.array([1.0, 1.0, -2.0]),
        "xlabel": r"$[1\bar{1}0]$ (r.l.u.)",
        "ylabel": r"$[11\bar{2}]$ (r.l.u.)",
    },
    "112": {
        "normal": np.array([1.0, 1.0, 2.0]),
        "u": np.array([1.0, -1.0, 0.0]),
        "v": np.array([1.0, 1.0, -1.0]),
        "xlabel": r"$[1\bar{1}0]$ (r.l.u.)",
        "ylabel": r"$[11\bar{1}]$ (r.l.u.)",
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
    return composite, exp_limit, model_limit


def resample_regular(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (y_axis, x_axis),
        image,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
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
        return np.asarray(volume[upper]), axis
    weight = (target_z - axis[lower]) / (axis[upper] - axis[lower])
    return (1.0 - weight) * volume[lower] + weight * volume[upper], axis


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
    differences = differences[differences > 1.0e-10]
    return float(np.min(differences))


def load_model_scattering() -> tuple[np.ndarray, np.ndarray]:
    with np.load(MODEL_SCATTERING) as archive:
        data = np.column_stack([archive[key] for key in ("h", "k", "l", "intensity")])
    q = data[:, :3]
    intensity = data[:, 3]
    intensity[np.sum(q * q, axis=1) <= ORIGIN_PUNCH_RADIUS_RLU**2 + 1.0e-12] = 0.0
    return q, intensity


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
    return x, y


def make_scattering_composite(
    q: np.ndarray,
    intensity: np.ndarray,
    zone: str,
    handle: h5py.File,
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
        exp_grid,
        model_grid,
        x_grid,
        SCATTERING_PERCENTILE,
        signed=False,
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


def set_crystallographic_ticks(ax: plt.Axes, spec: dict[str, object]) -> None:
    """Label physical distances by coefficients of the indicated hkl vectors."""
    for axis_name, vector in (("x", spec["u"]), ("y", spec["v"])):
        vector_length = float(np.linalg.norm(np.asarray(vector, dtype=float)))
        largest_coefficient = int(np.floor(SCATTERING_LIMIT / vector_length + 1.0e-9))
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


def main() -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.20, 4.85), constrained_layout=True)
    metadata = [
        f"model={MODEL_NAME} ({MODEL_DESCRIPTION})",
        f"model_root={MODEL_ROOT}",
        "halfplanes=experiment_left,model_right",
        "normalization=independent_per_half",
    ]

    pdf_cmap = mpl.colormaps["RdBu_r"].copy()
    pdf_cmap.set_bad("white")
    for ax, (_name, dataset, target_z, title) in zip(axes[:, 0], PDF_SLICES):
        composite, exp_limit, model_limit = make_pdf_composite(dataset, target_z)
        ax.imshow(
            composite,
            origin="lower",
            extent=[-PDF_LIMIT, PDF_LIMIT, -PDF_LIMIT, PDF_LIMIT],
            cmap=pdf_cmap,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set(
            xlim=(-PDF_LIMIT, PDF_LIMIT),
            ylim=(-PDF_LIMIT, PDF_LIMIT),
            xticks=np.arange(-2, 3, 1),
            yticks=np.arange(-2, 3, 1),
            xlabel=r"$x$ (unit cells)",
            ylabel=r"$y$ (unit cells)",
            title=title,
        )
        ax.set_xticks(np.arange(-2, 2.1, 0.5), minor=True)
        ax.set_yticks(np.arange(-2, 2.1, 0.5), minor=True)
        label_halfplanes(ax)
        metadata.extend(
            [
                f"3ddpdf_z{target_z:g}_experiment_abs_limit={exp_limit:.12g}",
                f"3ddpdf_z{target_z:g}_model_abs_limit={model_limit:.12g}",
            ]
        )

    q, intensity = load_model_scattering()
    scattering_cmap = mpl.colormaps["gray_r"].copy()
    scattering_cmap.set_bad("white")
    scattering_axes = (axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2])
    with h5py.File(EXPERIMENT_SCATTERING, "r") as handle:
        for ax, zone in zip(scattering_axes, SCATTERING_ZONES):
            composite, exp_limit, model_limit = make_scattering_composite(
                q, intensity, zone, handle
            )
            ax.imshow(
                composite,
                origin="lower",
                extent=[
                    -SCATTERING_LIMIT,
                    SCATTERING_LIMIT,
                    -SCATTERING_LIMIT,
                    SCATTERING_LIMIT,
                ],
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
                title=rf"Diffuse scattering, $[{zone}]$",
            )
            set_crystallographic_ticks(ax, spec)
            label_halfplanes(ax)
            metadata.extend(
                [
                    f"scattering_{zone}_experiment_limit={exp_limit:.12g}",
                    f"scattering_{zone}_model_limit={model_limit:.12g}",
                ]
            )

    for ax, label in zip(axes.ravel(), "abcdef"):
        ax.set_box_aspect(1)
        add_panel_label(ax, label)

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata.extend(
        [
            f"experimental_3ddpdf_source={EXPERIMENT_3DPDF}",
            f"experimental_scattering_source={EXPERIMENT_SCATTERING}",
            f"model_scattering_source={MODEL_SCATTERING}",
            f"3ddpdf_display_range_unit_cells=-{PDF_LIMIT},+{PDF_LIMIT}",
            f"3ddpdf_normalization_percentile_abs={PDF_PERCENTILE}",
            f"scattering_display_range_rlu=-{SCATTERING_LIMIT},+{SCATTERING_LIMIT}",
            "scattering_display_range_convention=physical_reciprocal_distance_along_unit_vectors",
            "scattering_tick_convention=integer_coefficients_of_labeled_hkl_axis_vectors",
            "scattering_tick_physical_spacing=100:sqrt1,sqrt1;110:sqrt2,sqrt1;111:sqrt2,sqrt6;112:sqrt2,sqrt3",
            f"scattering_normalization_percentile={SCATTERING_PERCENTILE}",
            "3ddpdf_resampling=nearest_neighbor",
            "scattering_resampling=linear",
            "display_scales=3ddpdf_signed_linear_RdBu_r;scattering_linear_gray_r",
        ]
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_metadata.txt").write_text(
        "\n".join(metadata) + "\n"
    )
    for extension in ("png", "pdf", "svg"):
        print(OUTPUT_STEM.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
