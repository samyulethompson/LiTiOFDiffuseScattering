# Calculation examples

Run commands from the repository root using the dependencies in [requirements.txt](requirements.txt).

## [run_mc_example.py](run_mc_example.py)

Generates Li/Ti occupations using Metropolis Monte Carlo and applies the pair-based displacement model.

```bash
python scripts/run_mc_example.py --output-dir results/mc_example
```

## [analyze_warren_cowley.py](analyze_warren_cowley.py)

Calculates the first six Warren–Cowley parameters and their combinations for equiatomic Li/Fe structures.

```bash
python scripts/analyze_warren_cowley.py \
  figures/figure_3/data/structure_lambda_*.cif --output results/warren_cowley.csv
```

## [analyze_lifeo2_local_environments.py](analyze_lifeo2_local_environments.py)

Counts tetrahedral, octahedral, and cubic Li/Fe environments and calculates exact fixed-composition random references.

```bash
python scripts/analyze_lifeo2_local_environments.py \
  figures/figure_3/data/structure_lambda_*.cif --output results/local_environment_counts.csv
```

## [count_0tm_sites.py](count_0tm_sites.py)

Counts tetrahedral sites whose four neighbouring cations are all Li and exports their count and fraction.

```bash
python scripts/count_0tm_sites.py \
  figures/figure_4/data/model_configurations.extxyz.gz \
  --frame 0 --output results/0tm_counts.csv
```

## [analyze_0tm_percolation.py](analyze_0tm_percolation.py)

Calculates periodic wrapping and accessible fractions for the 0-TM tetrahedral and Li diffusion networks.

```bash
python scripts/analyze_0tm_percolation.py \
  figures/figure_4/data/configurations_bitpacked.npz \
  --max-configs 1 --output-dir results/percolation_example
```

## [analyze_0tm_tortuosity.py](analyze_0tm_tortuosity.py)

Calculates relative microscopic tortuosity from shortest periodic-return paths on the wrapping 0-TM tetrahedral network.

```bash
python scripts/analyze_0tm_tortuosity.py \
  figures/figure_4/data/configurations_bitpacked.npz \
  --max-configs 1 --sites-per-config 4096 --seed 81 --workers 4 \
  --output-dir results/tortuosity_example
```
