# Calculation examples

These scripts illustrate the numerical methods used in the paper. They write CSV tables or atomic structures. Figure rendering is documented separately in [figures/](../figures/README.md).

| Script | Calculation | Output |
| --- | --- | --- |
| [run_mc_example.py](run_mc_example.py) | Fixed-composition Li/Ti Metropolis MC and pair-based displacements | Ideal and displaced extended-XYZ structures; parameter and pair-count CSV |
| [analyze_warren_cowley.py](analyze_warren_cowley.py) | First six cation-shell Warren–Cowley parameters and the combinations used in the paper | One CSV row per input structure |
| [analyze_lifeo2_local_environments.py](analyze_lifeo2_local_environments.py) | Li/Fe tetrahedral, O-centred octahedral, and cubic populations | Counts, fractions, and fixed-composition random expectations in CSV |
| [analyze_0tm_connectivity.py](analyze_0tm_connectivity.py) | Li/Ti tetrahedral populations and local 0-TM connectivity | Population, connectivity, and summary CSVs |
| [analyze_0tm_percolation.py](analyze_0tm_percolation.py) | Periodic wrapping and accessible fractions of tetrahedral and Li networks | Per-configuration, cluster-size, and ensemble-summary CSVs |
| [analyze_0tm_tortuosity.py](analyze_0tm_tortuosity.py) | Shortest periodic-return paths on the wrapping 0-TM network | Per-configuration tortuosity and ensemble-summary CSVs |

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -r scripts/requirements.txt
```

## Occupational MC and displacements

```bash
python scripts/run_mc_example.py --output-dir results/run71_example
```

The defaults are the Run 71 occupational and displacement parameters: a periodic 21 × 21 × 21 conventional-cell supercell, lattice parameter 4.10 Å, Ti fraction 2/7, 20 attempted Li/Ti exchanges per cation, and seed 23. A randomly selected Li site and Ti site are exchanged with Metropolis acceptance. Each distinct Ti–Ti pair is counted once, with 12 first neighbours and six second neighbours. The reduced Hamiltonian is

```math
\frac{H}{k_\mathrm{B}T}=K_1N_{\mathrm{TiTi}}^{(1)}+K_2N_{\mathrm{TiTi}}^{(2)},
\qquad K_1=-0.5,\quad K_2=+0.5.
```

The options `--j1-over-kbt` and `--j2-over-kbt` are dimensionless pair coefficients. The code uses inverse temperature one with these reduced energies. It does not assign a temperature in kelvin.

After MC, the displacement calculation sums three contributions: O moves 0.100 Å towards each nearest Ti; Ti moves 0.025 Å away from each nearest Ti; and both cations move 0.075 Å away from each unlike second-neighbour cation. These amplitudes are per pair and per displaced atom. Displacements do not enter the MC acceptance energy.

Outputs are `mc_occupations.extxyz`, `displaced.extxyz`, and `mc_summary.csv`. A shorter demonstration uses a smaller cell whose cation count still supports exactly the 2/7 Ti fraction:

```bash
python scripts/run_mc_example.py --nrep 7 --mc-sweeps 1 \
  --output-dir results/small_mc_example
```

The Figure 3 LiFeO₂ sweep is a separate model. Its archived parameters are J1 + J2 = 0.20 eV and T = 1000 K, giving kBT = 0.0861733 eV. Thus its reduced total interaction is 2.3209036. Those values should not be confused with the reduced-unit Run 71 parameters above.

## Warren–Cowley parameters

```bash
python scripts/analyze_warren_cowley.py \
  figures/figure_3/data/structure_lambda_*.cif \
  --output results/warren_cowley.csv
```

The calculation accepts the ideal equiatomic Li/Fe structures used for Figure 3. With spins +1 for Li and −1 for Fe, the shell correlation equals the Warren–Cowley parameter at this composition:

```math
\alpha_s=\langle\sigma_i\sigma_j\rangle_s.
```

The first six FCC shells have squared distances 2, 4, 6, 8, 10, and 12 in units of (a/2)². Periodic indexing is applied to every neighbour vector. The CSV includes α1–α6 and the combinations `1+3alpha_1`, `1+4alpha_1+alpha_2`, and `1+3alpha_2+3alpha_4+alpha_6`. The script rejects non-equiatomic compositions because this uncorrected spin-correlation expression does not apply to them.

## Local environments

```bash
python scripts/analyze_lifeo2_local_environments.py \
  figures/figure_3/data/structure_lambda_*.cif \
  --output results/local_environment_counts.csv
```

The motifs contain four cations around each tetrahedral site, six cations around each oxygen site, and eight cations at cube corners separated by one conventional-cell edge. Periodic boundaries are included. Counts are grouped by the number of Fe atoms in each motif.

For a motif of size m, the exact random reference preserves the total numbers of Li and Fe:

```math
P(k)=\frac{\binom{N_\mathrm{Fe}}{k}\binom{N_\mathrm{Li}}{m-k}}
{\binom{N_\mathrm{cations}}{m}}.
```

The script exports this discrete distribution directly. The first two analysis examples accept one or more structure paths. Explicit P1 CIF atom lists and extended-XYZ inputs are read with ASE. The CIF reader uses the listed sites directly, avoiding symmetry expansion of large supercells. For a multi-frame trajectory they use its first frame. Both require an axis-aligned cubic rocksalt cell with ideal cation positions.

## 0-TM populations and local connectivity

```bash
python scripts/analyze_0tm_connectivity.py \
  figures/figure_4/data/model_configurations.extxyz.gz \
  --frame 0 --random-samples 100 --seed 0 \
  --output-dir results/0tm_example
```

This example reads one Li/Ti configuration. `--frame` selects its index in a multi-frame extended-XYZ file; gzip input is supported. Cations are assigned to their parent rocksalt sites, so the result is independent of atom order. Ideal structures and small displacements that remain unambiguous under the nearest-site assignment are accepted.

A tetrahedral site is 0-TM when all four surrounding cations are Li. The local connectivity degree counts its six edge-sharing neighbouring tetrahedral sites at a/2 that are also 0-TM. The random reference for tetrahedral populations is the exact fixed-composition hypergeometric distribution. The connectivity reference is estimated by independently shuffling the same cation occupations for each random sample.

The three outputs are `tm_populations.csv`, `connectivity.csv`, and `summary.csv`. Connectivity fractions are conditional on a tetrahedral site being 0-TM; they are undefined when a configuration contains no such sites and are written as NaN. Local connectivity degree alone does not determine network percolation or tortuosity.

The example above analyses one model and generates its own seeded random controls. The published Figure 4 statistics instead use 100 independent MC trajectories and 100 random controls from Run 81, retained in [figure_4/data/](../figures/figure_4/data/). Their MC seeds are 710000–710099 and their random-control seeds are 810000–810099.

## Periodic percolation

```bash
python scripts/analyze_0tm_percolation.py \
  figures/figure_4/data/configurations_bitpacked.npz \
  --max-configs 1 --output-dir results/percolation_example
```

The example evaluates the first MC configuration and first random control. Omit `--max-configs 1` to evaluate all 100 configurations of each kind. A CIF or extended-XYZ structure can be supplied instead, with `--frame` selecting a trajectory frame. The script reuses the lattice-site reader and tetrahedral geometry in `analyze_0tm_connectivity.py`.

Two graphs are evaluated:

- `tetra_0tm`: active nodes are 0-TM tetrahedral sites, joined to their six edge-sharing neighbours at a/2.
- `li_0tm`: active nodes are all Li cation sites, including isolated sites. The four Li vertices of every 0-TM tetrahedron are connected pairwise, giving Li paths through 0-TM channels.

A union–find calculation tracks integer translations along graph edges. A cluster wraps when a closed graph path accumulates a nonzero supercell translation; crossing a boundary on an open path alone does not establish percolation. `wrap_x`, `wrap_y`, and `wrap_z` report whether any cluster wraps in each direction. `wraps_all_directions` reports whether all three directions are represented across the wrapping clusters.

Outputs are `percolation.csv` (one row per configuration and network), `cluster_sizes.csv`, and `ensemble_summary.csv`. The accessible fraction is the number of active nodes in any wrapping cluster divided by the total active-node count. Its denominator is the number of 0-TM sites for `tetra_0tm` and the total number of Li sites for `li_0tm`. Cluster fractions use the same network-specific denominator. Ensemble summaries report means and sample standard deviations across configurations. A sample standard deviation is undefined for a single configuration and is written as NaN.

## Relative microscopic tortuosity

```bash
python scripts/analyze_0tm_tortuosity.py \
  figures/figure_4/data/configurations_bitpacked.npz \
  --max-configs 1 --sites-per-config 4096 --seed 81 --workers 4 \
  --output-dir results/tortuosity_example
```

This is the relative microscopic tortuosity used for the Run 81 analysis. For a 0-TM tetrahedral site i, the calculation searches for the shortest six-neighbour path to any nonzero periodic image of the same site:

```math
\tau_i = \frac{\min_{\mathbf{n}\ne\mathbf{0}}\ell(i,i+\mathbf{n}L)}{2n_\mathrm{rep}}.
```

Path lengths are measured in graph hops. Searches are restricted to a 3 × 3 × 3 periodic-image halo. The denominator is the shortest periodic-return path on the fully open six-neighbour tetrahedral lattice: 42 hops for the 21 × 21 × 21 Run 81 cells. This is a geometrical path-length measure; it is not a diffusion coefficient or a squared transport-tortuosity factor.

The script selects the largest wrapping tetrahedral cluster. In all 200 saved Run 81 configurations, this is also the largest connected tetrahedral cluster, matching the original calculation. By default, 4,096 sites are sampled uniformly without replacement from that cluster using seed 81; smaller clusters are evaluated in full. `--all-sites` evaluates every site in the selected cluster. `--workers` controls CPU threads and defaults to one.

Outputs are `tortuosity_by_configuration.csv` and `tortuosity_summary.csv`. Each configuration's mean is calculated across its evaluated sites. Ensemble means, sample standard deviations, standard errors, and 95% Student-t confidence intervals are then calculated across those configuration means. A configuration without a wrapping cluster has undefined tortuosity (NaN); summary columns distinguish the total configuration count from the count with defined values. If a wrapping cluster has a sampled site with no return found inside the finite halo, the calculation reports an error.

To reproduce the full published ensemble calculation, omit `--max-configs 1` while retaining `--sites-per-config 4096 --seed 81`. The packed archive is processed in the original order: model 0, random 0, model 1, random 1, and so on. Keeping this order preserves the original random site samples. Analysing an individual structure or frame creates its own site sample instead. A changed worker count preserves the selected sites and path lengths. For the one-configuration examples, ensemble standard deviations, standard errors, and confidence intervals are undefined and are written as NaN; use the full ensemble to estimate variation between configurations.
