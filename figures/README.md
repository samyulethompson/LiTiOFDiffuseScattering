# Article figures

Each numbered directory is a self-contained plotting bundle with reference artwork, scripts and helper modules, all data required to plot, provenance checksums, and a reproduction command.

| Article figure | Contents |
| --- | --- |
| [Figure 1](figure_1/) | Experimental overview and calculated integrated ΔPDF |
| [Figure 2](figure_2/) | Model schematic, experimental/model scattering and PDF comparisons |
| [Figure 3](figure_3/) | 0.2 eV interaction mixing sweep and local environments |
| [Figure 4](figure_4/) | 0-TM ensemble population comparison |

## Install and regenerate

Use Python 3.10 or later. The recorded validation environment uses Python 3.10.19.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r figures/requirements.txt
python figures/figure_1/plot.py
python figures/figure_2/plot.py
python figures/figure_3/plot.py
python figures/figure_4/plot.py
```

Each runner writes PNG (600 dpi), PDF, and SVG to its own `output/` directory, leaving the exact article copies in `reference/`. Numerical processing and layout follow the supplied plotting workflows. Figure 2 uses lossless compressed inputs and adapted loaders to keep the full required arrays within GitHub file limits. No simulation, fitting, MACE inference, or relaxation runs are needed.

The bundles deliberately repeat shared experimental files so each figure can be used independently. Helper modules retain functions for related figures; only the entry point `plot.py` and its dependencies are supported by these bundles. Their unused functions may refer to upstream files that are not needed for the requested figure. Historical metadata records the source workstation paths. Per-file SHA-256 provenance and all required relative paths are included in each bundle.

After regenerating, run `python figures/verify.py` to check all bundled hashes and compare PNG pixels with the article references. Exact image matching requires the recorded software and available fonts.

See [VALIDATION.md](VALIDATION.md) for the regeneration checks and reference comparisons.
