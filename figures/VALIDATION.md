# Validation

Validated on 30 October 2026 with Python 3.10.19 in the author's `clusterexp` environment. Package versions are recorded in `requirements.txt`.

All four `plot.py` entry points completed using the bundled scripts and inputs and exported PNG, PDF, and SVG. The reference PNGs are exact copies of the four article files identified by the author.

| Figure | PNG width × height | Comparison with article reference | Provenance hashes |
| --- | --- | --- | --- |
| 1 | 4251 × 3850 | Identical RGBA pixels | 27 / 27 match |
| 2 | 3900 × 5091 | Identical RGBA pixels | 17 / 17 match |
| 3 | 4260 × 2101 | Identical RGBA pixels | 17 / 17 match |
| 4 | 2032 × 1670 | Identical RGBA pixels | 8 / 8 match |

Figure 2 was also regenerated with the final column-compressed scattering loader from `/tmp`, with a Python audit hook rejecting any file reads under the original `/home/samyt/pair_interaction` tree. That run completed, demonstrating independence from the upstream files and current directory. The lossless archive round trips were checked with `numpy.array_equal(..., equal_nan=True)` before plotting: the complete model volume and all four loaded scattering columns retained their original values and precision.

All provenance-listed files are included in Git. Generated `output/` directories are ignored. Existing unrelated local data are excluded. Python/documentation/JSON whitespace checks pass; original scientific text data and SVG artwork are preserved byte for byte, including their original formatting.

To repeat the checks after running the four entry points:

```bash
python figures/verify.py
```

This checks SHA-256 hashes and exact PNG pixel equality. PDF and SVG export completion was checked, but their bytes are not compared because renderer timestamps and generated identifiers may vary. Exact PNG equality can depend on installed fonts and plotting-library versions. Reference artwork is preserved for use independently of regeneration.
