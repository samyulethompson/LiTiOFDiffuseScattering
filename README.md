# LiTiOFDiffuseScattering

Code and data supporting the associated paper.

The four main article figures are documented in [figures/](figures/README.md). Each figure folder contains one standalone plotting script and its input data.

- [Figure 1: experimental overview](figures/figure_1/)
- [Figure 2: model, scattering and PDF comparisons](figures/figure_2/)
- [Figure 3: interaction mixing sweep](figures/figure_3/)
- [Figure 4: 0-TM ensemble populations](figures/figure_4/)

## Scripts

- `scripts/build_mc_structure.py` reproduces the Run 71 fixed-composition
  Li/Ti occupational Monte Carlo and displacement model.
- `scripts/run_mc_example.py` provides an example parameter set using reduced
  nearest- and second-neighbor Ti–Ti interactions.
- `scripts/plot_0tm_connectivity.py` compares the resulting 0-TM population
  and connectivity with fixed-composition random disorder.
- `scripts/plot_lifeo2_scattering_sweep.py` plots the eleven diffuse-scattering
  surfaces across a LiFeO2 nearest/next-nearest-neighbor mixing sweep.
- `scripts/analyze_warren_cowley.py` calculates the first six cation-shell
  Warren-Cowley parameters and the combinations used in the paper.
- `scripts/analyze_lifeo2_local_environments.py` counts tetrahedral,
  O-centered octahedral, and cubic Li/Fe environments against an exact
  fixed-composition random reference.

The paper's `kBT = 5` analysis uses the unnormalized `J1 + J2 = 0.2 eV`
sweep. Its `kBT = 1` analysis uses the coordination-normalized `A = 1 eV`
sweep. The analysis scripts default to the 0.2 eV paper case and accept
`--total 1` or `--totals 1`
for the second case. They operate on existing sweep output, so the equivalent
Monte Carlo driver is not duplicated here.

For example:

```bash
python scripts/run_mc_example.py --output-dir mc_example_output
python scripts/plot_0tm_connectivity.py \
  mc_example_output/rocks_mc_example_mc_occupations.extxyz \
  --output-stem mc_example_output/0tm_connectivity

python scripts/plot_lifeo2_scattering_sweep.py /path/to/sweep
python scripts/analyze_warren_cowley.py /path/to/sweep
python scripts/analyze_lifeo2_local_environments.py /path/to/sweep

# Coordination-normalized A = 1 eV (paper kBT = 1 case)
python scripts/plot_lifeo2_scattering_sweep.py /path/to/strength_map --totals 1
python scripts/analyze_warren_cowley.py /path/to/strength_map --total 1
python scripts/analyze_lifeo2_local_environments.py /path/to/strength_map --totals 1
```
