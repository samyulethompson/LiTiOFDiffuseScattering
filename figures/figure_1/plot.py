#!/usr/bin/env python3
"""Regenerate this article figure using only the bundled sources and data."""
from pathlib import Path
import importlib.util
import sys
import matplotlib
matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'source/LiTiOF graphing/intro_experimental_overview/plot_intro_experimental_overview_3x3_integrated_model_3dpdf_transposed.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("article_figure", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
output = ROOT / "output"
output.mkdir(exist_ok=True)
module.OUTPUT_STEM = output / 'LiTiOF_intro_experimental_overview_3x3_integrated_model_3dpdf_transposed'
module.main()
