# LiTiOFDiffuseScattering

Code and data supporting the associated paper.

The four main article figures are documented in [figures/](figures/README.md). Each figure folder contains one standalone plotting script and its input data.

- [Figure 1: experimental overview](figures/figure_1/)
- [Figure 2: model, scattering and PDF comparisons](figures/figure_2/)
- [Figure 3: interaction mixing sweep (neutron scattering)](figures/figure_3/)
- [Figure 4: 0-TM ensemble populations](figures/figure_4/)

## Calculation examples

The [scripts/](scripts/README.md) directory contains calculation-only examples for occupational Monte Carlo and displacements, Warren–Cowley parameters, local-environment counts, 0-TM populations and connectivity, periodic percolation, and relative microscopic tortuosity. They accept structure files directly and export numerical CSV tables or atomic structures.

```bash
python -m pip install -r scripts/requirements.txt
python scripts/run_mc_example.py --output-dir results/run71_example
python scripts/analyze_warren_cowley.py \
  figures/figure_3/data/structure_lambda_*.cif --output results/warren_cowley.csv
python scripts/analyze_lifeo2_local_environments.py \
  figures/figure_3/data/structure_lambda_*.cif --output results/local_environment_counts.csv
python scripts/analyze_0tm_connectivity.py \
  figures/figure_4/data/model_configurations.extxyz.gz \
  --random-samples 100 --output-dir results/0tm_example
python scripts/analyze_0tm_percolation.py \
  figures/figure_4/data/configurations_bitpacked.npz \
  --max-configs 1 --output-dir results/percolation_example
python scripts/analyze_0tm_tortuosity.py \
  figures/figure_4/data/configurations_bitpacked.npz \
  --max-configs 1 --sites-per-config 4096 --workers 4 \
  --output-dir results/tortuosity_example
```

Run 71 uses reduced pair coefficients J1/(kBT) = −0.5 and J2/(kBT) = +0.5. The separate Figure 3 sweep was generated at 1000 K with J1 + J2 = 0.20 eV, equivalent to a reduced total interaction of 2.3209036. See the [calculation guide](scripts/README.md) for definitions, input assumptions, and output columns.
