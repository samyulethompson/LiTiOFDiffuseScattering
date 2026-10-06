#!/usr/bin/env python3
"""Plot the eleven h=0 diffuse-scattering surfaces in a LiFeO2 J1/J2 sweep."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


RESULT_RE = re.compile(
    r"results_lifeo2_(unnorm|norm)_total_([^_]+)_ratio_([^_]+)_temp_([^/]+)$"
)


def decode_number(token: str) -> float:
    return float(token.replace("p", "."))


def encode_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_surface(path: Path) -> np.ndarray:
    data = np.loadtxt(path, usecols=(0, 1, 2, 3))
    mask = (
        np.isclose(data[:, 0], 0.0, atol=1e-8)
        & (data[:, 1] >= -1e-8)
        & (data[:, 1] <= 2 + 1e-8)
        & (data[:, 2] >= -1e-8)
        & (data[:, 2] <= 2 + 1e-8)
    )
    data = data[mask]
    ks = np.unique(np.round(data[:, 1], 10))
    ls = np.unique(np.round(data[:, 2], 10))
    image = np.full((len(ls), len(ks)), np.nan)
    for _, k, ell, intensity in data:
        image[
            np.searchsorted(ls, round(ell, 10)),
            np.searchsorted(ks, round(k, 10)),
        ] = intensity
    if image.shape != (101, 101):
        raise ValueError(f"Expected 101x101 vertices in {path}, got {image.shape}")
    return image


def result_parameters(path: Path) -> tuple[str, float, float, str]:
    result_dir = path.parents[1].name
    match = RESULT_RE.match(result_dir)
    if not match:
        raise ValueError(f"Could not parse sweep parameters from {result_dir}")
    return (
        match.group(1),
        decode_number(match.group(2)),
        decode_number(match.group(3)),
        match.group(4),
    )


def find_surfaces(root: Path, total: float) -> list[tuple[float, Path]]:
    pattern = (
        "*/results_lifeo2_*_total_*_ratio_*_temp_*/scatty_structures/"
        "*_100vox_sc_list.txt"
    )
    surfaces = [
        (result_parameters(path)[2], path)
        for path in root.glob(pattern)
        if math.isclose(result_parameters(path)[1], total)
    ]
    surfaces.sort()
    if len(surfaces) != 11:
        raise RuntimeError(
            f"Expected 11 surfaces for total strength {total:g} eV, found {len(surfaces)}"
        )
    expected = np.linspace(0.0, 1.0, 11)
    if not np.allclose([ratio for ratio, _ in surfaces], expected):
        raise RuntimeError("Expected mixing ratios from 0 to 1 in increments of 0.1")
    return surfaces


def format_panel(ax, ratio: float, total: float, series_label: str) -> None:
    if series_label == "normalised":
        first_interaction = total * (1 - ratio) / 12
        second_interaction = total * ratio / 6
    else:
        first_interaction = total * (1 - ratio)
        second_interaction = total * ratio
    ax.set_title(
        f"x={ratio:g}\nJ₁={first_interaction:.3f}, J₂={second_interaction:.3f}",
        fontsize=8,
    )
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("k", fontsize=8)


def plot_linear(
    images: list[np.ndarray],
    ratios: list[float],
    total: float,
    temperature: str,
    series_label: str,
    output: Path,
    common_scale: bool,
) -> None:
    fig, axes = plt.subplots(1, 11, figsize=(18, 2.35), sharex=True, sharey=True)
    values = np.concatenate([image.ravel() for image in images])
    common_limits = float(np.nanmin(values)), float(np.nanmax(values))
    plotted = None
    for ax, image, ratio in zip(axes, images, ratios):
        limits = common_limits if common_scale else (
            float(np.nanmin(image)),
            float(np.nanmax(image)),
        )
        plotted = ax.imshow(
            image,
            origin="lower",
            extent=(0, 2, 0, 2),
            cmap="Greys",
            interpolation="nearest",
            aspect="equal",
            vmin=limits[0],
            vmax=limits[1],
        )
        format_panel(ax, ratio, total, series_label)
    axes[0].set_ylabel("l", fontsize=8)
    scale = "common intensity scale" if common_scale else "individual intensity scales"
    strength_label = (
        f"coordination-normalised A={total:g} eV"
        if series_label == "normalised"
        else f"J₁+J₂={total:g} eV"
    )
    fig.suptitle(
        f"LiFeO₂, {temperature}: {series_label} J₁/J₂ mixing, {strength_label}\n"
        f"[100] surface (h=0, 0≤k,l≤2), {scale}",
        fontsize=11,
    )
    if common_scale:
        fig.colorbar(plotted, ax=axes, label="Scatty intensity", pad=0.008, fraction=0.02)
    fig.subplots_adjust(
        left=0.035,
        right=0.965 if common_scale else 0.995,
        bottom=0.16,
        top=0.72,
        wspace=0.08,
    )
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_white_on_black(
    images: list[np.ndarray], ratios: list[float], output: Path
) -> None:
    fig, axes = plt.subplots(1, 11, figsize=(16.5, 2.25), sharex=True, sharey=True)
    for ax, image in zip(axes, images):
        ax.imshow(
            image,
            origin="lower",
            extent=(0, 2, 0, 2),
            cmap="gray",
            interpolation="nearest",
            aspect="equal",
            vmin=float(np.nanmin(image)),
            vmax=float(np.nanmax(image)),
        )
        ax.set_axis_off()
        ax.set_facecolor("black")
    global_axis = fig.add_axes((0, 0.17, 1, 0.08), frameon=False)
    global_axis.set_xlim(-0.05, 1.05)
    global_axis.set_ylim(0, 1)
    global_axis.set_xticks(ratios)
    global_axis.set_xticklabels([f"{ratio:g}" for ratio in ratios])
    global_axis.set_yticks([])
    global_axis.tick_params(axis="x", length=0, pad=1, labelsize=8)
    fig.text(0.5, 0.025, "Mixing parameter, x", ha="center", va="bottom")
    fig.subplots_adjust(left=0, right=1, bottom=1 / 3, top=1, wspace=0, hspace=0)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def plot_log(
    images: list[np.ndarray],
    ratios: list[float],
    total: float,
    temperature: str,
    series_label: str,
    output: Path,
    common_scale: bool,
) -> None:
    all_positive = np.concatenate(
        [image[np.isfinite(image) & (image > 0)] for image in images]
    )
    if all_positive.size == 0:
        raise ValueError("No positive intensities available for logarithmic plotting")
    common_limits = float(np.min(all_positive)), float(np.max(all_positive))
    fig, axes = plt.subplots(
        1, 11, figsize=(18 if common_scale else 19.5, 2.35 if common_scale else 2.7),
        sharex=True, sharey=True,
    )
    plotted = None
    for ax, image, ratio in zip(axes, images, ratios):
        positive = image[np.isfinite(image) & (image > 0)]
        limits = common_limits if common_scale else (
            float(np.min(positive)),
            float(np.max(positive)),
        )
        plotted = ax.imshow(
            np.ma.masked_less_equal(image, 0),
            origin="lower",
            extent=(0, 2, 0, 2),
            cmap="Greys",
            interpolation="nearest",
            aspect="equal",
            norm=LogNorm(*limits),
        )
        format_panel(ax, ratio, total, series_label)
        if not common_scale:
            colorbar = fig.colorbar(
                plotted, ax=ax, orientation="horizontal", pad=0.17, fraction=0.07, aspect=12
            )
            colorbar.ax.tick_params(labelsize=5)
    axes[0].set_ylabel("l", fontsize=8)
    scale = "common" if common_scale else "individual"
    strength_label = (
        f"coordination-normalised A={total:g} eV"
        if series_label == "normalised"
        else f"J₁+J₂={total:g} eV"
    )
    fig.suptitle(
        f"LiFeO₂, {temperature}: {series_label} J₁/J₂ mixing, {strength_label}\n"
        f"[100] surface (h=0, 0≤k,l≤2), {scale} logarithmic intensity scale",
        fontsize=11,
    )
    if common_scale:
        fig.colorbar(
            plotted, ax=axes, label="Scatty intensity (log scale)",
            pad=0.008, fraction=0.02,
        )
    fig.subplots_adjust(
        left=0.035 if common_scale else 0.03,
        right=0.965 if common_scale else 0.995,
        bottom=0.16 if common_scale else 0.23,
        top=0.72 if common_scale else 0.74,
        wspace=0.08 if common_scale else 0.12,
    )
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_total(root: Path, total: float, output_root: Path) -> None:
    surfaces = find_surfaces(root, total)
    ratios = [ratio for ratio, _ in surfaces]
    images = [load_surface(path) for _, path in surfaces]
    series, _, _, temperature = result_parameters(surfaces[0][1])
    series_label = "unnormalised" if series == "unnorm" else "normalised"
    token = encode_number(total)
    output_dir = output_root / f"{series_label}_{token}eV_mixing_plot"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{series_label}_{token}eV_mixing"
    plot_linear(images, ratios, total, temperature, series_label, output_dir / f"{prefix}_common_scale.png", True)
    plot_linear(images, ratios, total, temperature, series_label, output_dir / f"{prefix}_individual_scale.png", False)
    plot_white_on_black(images, ratios, output_dir / f"{prefix}_individual_scale_white_on_black.png")
    plot_log(images, ratios, total, temperature, series_label, output_dir / f"{prefix}_log_common_scale.png", True)
    plot_log(images, ratios, total, temperature, series_label, output_dir / f"{prefix}_log_individual_scale.png", False)
    print(f"Wrote plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root directory containing the sweep runs")
    parser.add_argument("--totals", nargs="+", type=float, default=[0.2])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root or args.root
    for total in args.totals:
        plot_total(args.root, total, output_root)


if __name__ == "__main__":
    main()
