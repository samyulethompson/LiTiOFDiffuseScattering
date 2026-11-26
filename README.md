# LiTiOFDiffuseScattering

Source code and data supporting the associated paper on occupational correlations,
atomic displacements, diffuse scattering, and Li diffusion networks in a
Li/Ti rocksalt model. The repository includes Monte Carlo and structural-analysis
scripts, a supplied example supercell, and the data and plotting scripts
for the four main article figures.

## 1. System requirements

### Software and tested versions

The documented environment is **Python 3.10.19 on Rocky Linux 8.9 (x86-64)**.
The following direct dependencies are pinned in
[scripts/requirements.txt](scripts/requirements.txt) and
[figures/requirements.txt](figures/requirements.txt):

| Package | Version | Used for |
| --- | --- | --- |
| NumPy | 2.2.6 | Numerical arrays and occupations |
| ASE | 3.28.0 | Reading and writing atomic structures |
| SciPy | 1.15.3 | Interpolation and statistical analysis |
| Numba | 0.63.1 | Compiled tortuosity calculations |
| Matplotlib | 3.10.8 | Article figures |
| h5py | 3.16.0 | HDF5 experimental plotting inputs |
| Pillow | 12.1.0 | Figure image inputs |

The tested Numba environment uses llvmlite 0.46.0. Package dependencies are
resolved by pip; no separately compiled repository executable is required.
The scripts use Python 3.10 or later, but other Python versions and operating
systems have not been validated here. The four figure scripts were previously
validated with Python 3.10.19 and the pinned plotting packages; see the
[figure documentation](figures/README.md).

### Hardware

An ordinary CPU computer is sufficient.

## 2. Installation

Run these commands in a terminal with Git and Python 3.10 available:

```bash
git clone https://github.com/SamYT-dur/LiTiOFDiffuseScattering.git
cd LiTiOFDiffuseScattering
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/requirements.txt -r figures/requirements.txt
```

Run all commands below from the repository root with this environment activated.
For calculation scripts alone, installing only `scripts/requirements.txt` is
sufficient. For figure generation alone, use `figures/requirements.txt`.

Allow approximately **5–15 minutes** for installation on a normal desktop.

## 3. Demo: analyze the supplied cell

### Supplied simulated dataset

[examples/demo/dummy_cell.extxyz.gz](examples/demo/dummy_cell.extxyz.gz) is one
complete displaced supercell of the model used for Figure 2.
It is a lossless gzip copy of the original extended-XYZ file, approximately
377 KiB compressed. It can be read directly by ASE without manual decompression.

| Property | Value |
| --- | --- |
| Supercell | 21 × 21 × 21 conventional rocksalt cells |
| Conventional lattice parameter | 4.1 Å |
| Periodic cubic box edge | 86.1 Å |
| Total atoms | 74,088 |
| Li / Ti / O atoms | 26,460 / 10,584 / 37,044 |
| Ti fraction on cation sites | 2/7 |
| MC seed / attempted exchanges per cation | 23 / 20 |
| Reduced Ti–Ti pair interactions | J1/(kBT) = −0.5; J2/(kBT) = +0.5 |

The post-MC displacement recipe moves O toward each nearest Ti by 0.100 Å per
pair, separates nearest Ti–Ti pairs by 0.025 Å per Ti, and separates
second-neighbor Li–Ti pairs by 0.075 Å per cation. Contributions are summed;
displacements do not enter the occupational MC acceptance energy. The file stores
explicit Li, Ti, and O species, Cartesian positions in Å, the cell, and periodic
boundary conditions.

### Count all-Li tetrahedral sites

Count tetrahedral sites whose four neighboring cations are all Li (0-TM sites):

```bash
python scripts/count_0tm_sites.py examples/demo/dummy_cell.extxyz.gz \
  --output results/demo/0tm_counts.csv
```

Expected terminal output:

```text
0-TM tetrahedral sites: 23717/74088 (0.320119)
Wrote results/demo/0tm_counts.csv
```

The demo writes one CSV row with the atom counts, tetrahedral-site count, and
0-TM fraction. Allow a few seconds on a normal desktop (estimated); the counting
demo took **2.11 seconds** on one Intel Xeon Gold 6248R CPU core with the
software versions above.

### Analyze periodic connectivity

```bash
python scripts/analyze_0tm_percolation.py examples/demo/dummy_cell.extxyz.gz \
  --output-dir results/percolation_example
```

Measured runtime: **2.70 seconds** using one CPU core on the reference machine above.

### Estimate microscopic tortuosity

```bash
python scripts/analyze_0tm_tortuosity.py examples/demo/dummy_cell.extxyz.gz \
  --sites-per-config 4096 --seed 81 --workers 4 \
  --output-dir results/tortuosity_example
```

Measured runtime: **13.54 seconds** using four CPU workers on the same machine,
including startup and any required Numba compilation. Desktop timings will vary.

Each command writes CSV results to the requested location. Use `--help` for options.

To use your own data, replace `examples/demo/dummy_cell.extxyz.gz` with your own
periodic cubic Li/Ti/O rocksalt supercell in extended-XYZ or explicit P1 CIF format.

## License

The source code is distributed under the [MIT License](LICENSE).
