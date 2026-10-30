#!/usr/bin/env python3
"""Regenerate this article figure using only the bundled sources and data."""
from pathlib import Path
import importlib.util
import sys
import matplotlib
matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'source/LiTiOF graphing/mixing_sweep_0p2eV/plot_mixing_sweep_0p2eV.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("article_figure", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
output = ROOT / "output"
output.mkdir(exist_ok=True)
module.OUTPUT_STEM = output / 'mixing_sweep_0p2eV_100_plane'
module.main()
