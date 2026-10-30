# Article figures

Each figure folder contains one standalone `plot.py` and a `data/` directory with its plotting inputs and supporting crystal structures.

| Figure | Plotting script | Input data |
| --- | --- | --- |
| 1 — Experimental overview | [plot.py](figure_1/plot.py) | [data/](figure_1/data/) |
| 2 — Model, scattering and PDF comparisons | [plot.py](figure_2/plot.py) | [data/](figure_2/data/) |
| 3 — Interaction mixing sweep | [plot.py](figure_3/plot.py) | [data/](figure_3/data/) |
| 4 — 0-TM ensemble populations | [plot.py](figure_4/plot.py) | [data/](figure_4/data/) |

## Reproduce

Use Python 3.10 or later and install the plotting dependencies:

```bash
python -m pip install -r figures/requirements.txt
python figures/figure_1/plot.py
python figures/figure_2/plot.py
python figures/figure_3/plot.py
python figures/figure_4/plot.py
```

Each script resolves data relative to its own location and writes PNG (600 dpi), PDF, and SVG files into that figure's `output/` directory. The reference PNGs below are the article figures.

## Figure 1

![Figure 1](figure_1.png)

PXRD refinement and structure schematic; [100], [110], and [111] zone-axis, reconstructed, and calculated scattering; experimental and calculated integrated 3D-ΔPDFs.

The data folder includes the refinement CSV and phase tick lists, structure and diffraction images, experimental scattering HDF5, integrated and unintegrated experimental xy0 arrays, calculated scattering planes, and the Run-80 integrated ΔPDF lattice volume and direct-space axis. Calculated-plane scaling is retained from the plane data; the integrated model ΔPDF color scale excludes the origin.

## Figure 2

![Figure 2](figure_2.png)

Model schematic, experimental/model 3D-ΔPDF and diffuse-scattering half-plane comparisons, and three PDF fits with residuals. The model is Run 71 case 01 with Run 61_02 displacements.

The data folder includes the schematic, three PDF fit tables, experimental ΔPDF and scattering HDF5 files, the model ΔPDF volume and direct axis, and model scattering values. Model arrays are stored losslessly as compressed NumPy archives: `volume` for the ΔPDF, and `h`, `k`, `l`, `intensity` for scattering. All input values and precision are retained.

## Figure 3

![Figure 3](figure_3.png)

Eleven [100] scattering surfaces at λ = 0, 0.1, …, 1 for J1 + J2 = 0.2 eV and 1000 K, with local-environment distributions at λ = 0, 0.5, 1.

The data folder contains `scattering_lambda_*.txt`, `local_environment_counts.csv`, and all eleven `structure_lambda_*.cif` crystal structures, matched by λ. The CIFs are the exact final snapshot 99 files identified by the local-environment analysis. Each scattering panel selects h = 0 and 0 ≤ k,l ≤ 2 and is independently normalized at the 99.5th percentile. The reference curves use a normal approximation to the fixed-composition random fractions in the CSV. The input workflow calls the transition-metal species Fe; the figure labels it M.

## Figure 4

![Figure 4](figure_4.png)

Tetrahedral-site populations for 100 Run-71 model configurations and 100 random controls from the Run-81 ensemble. The input is `ensemble_summary.csv`. Bars show means of `tm0_fraction` through `tm4_fraction`; error bars show sample standard deviations across configurations.

The data folder also retains all 200 crystal structures: `model_configurations.extxyz.gz` contains the 100 model configurations and `random_configurations.extxyz.gz` contains the 100 random controls. Each frame includes its configuration index, seed, atomic species, positions, and periodic cell (21 × 21 × 21 conventional cells; lattice constant 4.1 Å). These compressed extended-XYZ files can be opened directly with ASE. The model trajectory is a lossless compressed copy of the source file; random-control structures use the exact saved occupations on the same ideal rocksalt lattice.

`configurations_bitpacked.npz` preserves the original occupations and seeds for both ensembles. Arrays `mc` and `random` contain one bit per cation (`1=Ti`, `0=Li`), packed with `bitorder="little"`; trim each unpacked row to `n_cations`. The extended-XYZ files include the oxygen sites as well as the cations.

## Validation

All four standalone scripts were checked against the reference PNGs on 30 October 2026 using Python 3.10.19 and the package versions in `requirements.txt`. All regenerated PNGs have identical pixels to their article references. PNG, PDF, and SVG exports complete successfully. Exact PNG matching can depend on installed fonts and plotting-library versions.
