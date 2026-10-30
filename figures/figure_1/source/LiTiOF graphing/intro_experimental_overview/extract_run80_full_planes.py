#!/usr/bin/env python3
"""Stream the Run 80 1001^3 VTK and cache full 100/111 calculated planes."""

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
CACHE = HERE / "run80_02_full_calculated_planes_100_111.npz"
METADATA = HERE / "run80_02_full_calculated_planes_100_111_metadata.txt"

AXIS_LIMIT = 6.0
SOURCE_POINTS = 1001
DISPLAY_POINTS = 361
PUNCH_RADIUS = 0.0
NORMALIZATION_PERCENTILE = 99.5
CHUNK_BYTES = 64 * 1024 * 1024

ZONE_AXES = {
    "100": (
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ),
    "111": (
        np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
        np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0),
    ),
}


def stream_planes() -> dict[str, tuple[np.ndarray, ...]]:
    pieces = {zone: [[], [], [], []] for zone in ("100", "111")}
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
            conditions = {
                "100": ix == center,
                "111": ix + iy + iz == 3 * center,
            }
            for zone, mask in conditions.items():
                if np.any(mask):
                    pieces[zone][0].append(ix[mask].astype(np.int16))
                    pieces[zone][1].append(iy[mask].astype(np.int16))
                    pieces[zone][2].append(iz[mask].astype(np.int16))
                    pieces[zone][3].append(values[mask].astype(np.float32))
            offset += values.size
            if offset % 100_000_000 < values.size:
                print(f"Read {offset:,} / {nx * ny * nz:,} voxels", flush=True)

    expected = nx * ny * nz
    if offset != expected:
        raise ValueError(f"Read {offset} scalar values; expected {expected}")
    return {
        zone: tuple(np.concatenate(component) for component in components)
        for zone, components in pieces.items()
    }


def normalize(values: np.ndarray) -> tuple[np.ndarray, float]:
    finite = values[np.isfinite(values)]
    limit = float(np.percentile(finite, NORMALIZATION_PERCENTILE))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Calculated-plane normalization limit is not positive")
    return np.clip(values / limit, 0.0, 1.0), limit


def interpolate_plane(
    indices_and_intensity: tuple[np.ndarray, ...], zone: str
) -> tuple[np.ndarray, np.ndarray, float]:
    ix, iy, iz, intensity = indices_and_intensity
    spacing = 2.0 * AXIS_LIMIT / (SOURCE_POINTS - 1)
    q = np.column_stack(
        (
            -AXIS_LIMIT + spacing * ix,
            -AXIS_LIMIT + spacing * iy,
            -AXIS_LIMIT + spacing * iz,
        )
    )
    intensity = intensity.copy()
    if PUNCH_RADIUS > 0.0:
        intensity[np.sum(q * q, axis=1) <= PUNCH_RADIUS**2 + 1.0e-12] = 0.0
    scaled, limit = normalize(intensity)

    axis_u, axis_v = ZONE_AXES[zone]
    x = q @ axis_u
    y = q @ axis_v
    axis = np.linspace(-AXIS_LIMIT, AXIS_LIMIT, DISPLAY_POINTS)
    x_grid, y_grid = np.meshgrid(axis, axis)
    image = griddata(
        np.column_stack((x, y)),
        scaled,
        (x_grid, y_grid),
        method="nearest",
    )
    return axis, np.asarray(image, dtype=np.float32), limit


def main() -> None:
    planes = stream_planes()
    axis_100, image_100, limit_100 = interpolate_plane(planes["100"], "100")
    axis_111, image_111, limit_111 = interpolate_plane(planes["111"], "111")
    if not np.allclose(axis_100, axis_111):
        raise ValueError("The calculated-plane display axes do not match")
    np.savez_compressed(
        CACHE,
        axis=axis_100,
        image_100=image_100,
        image_111=image_111,
        normalization_limit_100=np.array(limit_100),
        normalization_limit_111=np.array(limit_111),
    )
    METADATA.write_text(
        "\n".join(
            (
                f"source_vtk={VTK}",
                f"source_grid={SOURCE_POINTS},{SOURCE_POINTS},{SOURCE_POINTS}",
                "source_range_rlu=-6,+6",
                "planes=100,111",
                "planes_are_full_calculated_planes_not_halfplane_composites",
                "origin_punch_radius_rlu=0",
                "origin_punch_applied=false",
                f"normalization=independent_{NORMALIZATION_PERCENTILE}th_percentile",
                f"normalization_limit_100_raw={limit_100:.12g}",
                f"normalization_limit_111_raw={limit_111:.12g}",
                f"display_grid={DISPLAY_POINTS},{DISPLAY_POINTS}",
                "display_interpolation=nearest",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(CACHE)


if __name__ == "__main__":
    main()
