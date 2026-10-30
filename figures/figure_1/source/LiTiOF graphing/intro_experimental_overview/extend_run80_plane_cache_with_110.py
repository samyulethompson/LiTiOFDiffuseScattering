#!/usr/bin/env python3
"""Add the Run 80 full 110 plane to the cached 100/111 calculated slices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import griddata


HERE = Path(__file__).resolve().parent
SCATTY_ROOT = Path(
    "/home/samyt/pair_interaction/3d-deltaPDF_informed/run80/"
    "02_without_displacements/scatty_600_060_006_500points"
)
STEM = "rocks_run80_02_without_displacements_j1p1p5_j2p1_kbt1_q10_exp20_bragg_removed"
VTK = SCATTY_ROOT / f"{STEM}_hk0_sc.vtk"
SOURCE_CACHE = HERE / "run80_02_full_calculated_planes_100_111.npz"
CACHE = HERE / "run80_02_full_calculated_planes_100_110_111.npz"
METADATA = HERE / "run80_02_full_calculated_planes_100_110_111_metadata.txt"

AXIS_LIMIT = 6.0
SOURCE_POINTS = 1001
NORMALIZATION_PERCENTILE = 99.5
CHUNK_BYTES = 64 * 1024 * 1024


def stream_110_plane() -> tuple[np.ndarray, ...]:
    pieces: list[list[np.ndarray]] = [[], [], [], []]
    with VTK.open("rb") as handle:
        dimensions = None
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError("VTK scalar data header was not found")
            if line.startswith(b"DIMENSIONS"):
                dimensions = tuple(map(int, line.split()[1:4]))
            if line.startswith(b"LOOKUP_TABLE"):
                break
        if dimensions != (SOURCE_POINTS, SOURCE_POINTS, SOURCE_POINTS):
            raise ValueError(
                f"Expected a {SOURCE_POINTS}^3 VTK grid, got {dimensions}"
            )

        nx, ny, nz = dimensions
        center = nx // 2
        nxy = nx * ny
        offset = 0
        carry = b""
        while True:
            block = handle.read(CHUNK_BYTES)
            if not block:
                block = carry
                carry = b""
            elif carry:
                block = carry + block
            if not block:
                break
            if handle.peek(1):
                split = block.rfind(b" ")
                if split < 0:
                    carry = block
                    continue
                carry = block[split + 1 :]
                block = block[: split + 1]
            values = np.fromstring(block.decode("ascii"), sep=" ", dtype=np.float64)
            indices = np.arange(offset, offset + values.size, dtype=np.int64)
            ix = indices % nx
            iy = (indices // nx) % ny
            iz = indices // nxy
            mask = ix + iy == 2 * center
            if np.any(mask):
                pieces[0].append(ix[mask].astype(np.int16))
                pieces[1].append(iy[mask].astype(np.int16))
                pieces[2].append(iz[mask].astype(np.int16))
                pieces[3].append(values[mask].astype(np.float32))
            offset += values.size
            if offset % 100_000_000 < values.size:
                print(f"Read {offset:,} / {nx * ny * nz:,} voxels", flush=True)

    expected = nx * ny * nz
    if offset != expected:
        raise ValueError(f"Read {offset} scalar values; expected {expected}")
    return tuple(np.concatenate(component) for component in pieces)


def interpolate_110(
    plane: tuple[np.ndarray, ...], display_axis: np.ndarray
) -> tuple[np.ndarray, float]:
    ix, iy, iz, intensity = plane
    spacing = 2.0 * AXIS_LIMIT / (SOURCE_POINTS - 1)
    q = np.column_stack(
        (
            -AXIS_LIMIT + spacing * ix,
            -AXIS_LIMIT + spacing * iy,
            -AXIS_LIMIT + spacing * iz,
        )
    )
    finite = intensity[np.isfinite(intensity)]
    limit = float(np.percentile(finite, NORMALIZATION_PERCENTILE))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Calculated 110-plane normalization limit is not positive")
    scaled = np.clip(intensity / limit, 0.0, 1.0)
    axis_u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    axis_v = np.array([0.0, 0.0, 1.0])
    x = q @ axis_u
    y = q @ axis_v
    x_grid, y_grid = np.meshgrid(display_axis, display_axis)
    image = griddata(
        np.column_stack((x, y)),
        scaled,
        (x_grid, y_grid),
        method="nearest",
    )
    return np.asarray(image, dtype=np.float32), limit


def main() -> None:
    if CACHE.exists():
        with np.load(CACHE) as cached:
            if all(key in cached for key in ("image_100", "image_110", "image_111")):
                print(CACHE)
                return
    if not SOURCE_CACHE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_CACHE}; run extract_run80_full_planes.py first"
        )
    with np.load(SOURCE_CACHE) as source:
        axis = np.asarray(source["axis"], dtype=float)
        image_100 = np.asarray(source["image_100"], dtype=np.float32)
        image_111 = np.asarray(source["image_111"], dtype=np.float32)
        limit_100 = float(source["normalization_limit_100"])
        limit_111 = float(source["normalization_limit_111"])

    image_110, limit_110 = interpolate_110(stream_110_plane(), axis)
    np.savez_compressed(
        CACHE,
        axis=axis,
        image_100=image_100,
        image_110=image_110,
        image_111=image_111,
        normalization_limit_100=np.array(limit_100),
        normalization_limit_110=np.array(limit_110),
        normalization_limit_111=np.array(limit_111),
    )
    METADATA.write_text(
        "\n".join(
            (
                f"source_vtk={VTK}",
                f"source_100_111_cache={SOURCE_CACHE}",
                f"source_grid={SOURCE_POINTS},{SOURCE_POINTS},{SOURCE_POINTS}",
                "source_range_rlu=-6,+6",
                "planes=100,110,111",
                "planes_are_full_calculated_planes_not_halfplane_composites",
                "origin_punch_applied=false",
                f"normalization=independent_{NORMALIZATION_PERCENTILE}th_percentile",
                f"normalization_limit_100_raw={limit_100:.12g}",
                f"normalization_limit_110_raw={limit_110:.12g}",
                f"normalization_limit_111_raw={limit_111:.12g}",
                f"display_grid={axis.size},{axis.size}",
                "display_interpolation=nearest",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(CACHE)


if __name__ == "__main__":
    main()
