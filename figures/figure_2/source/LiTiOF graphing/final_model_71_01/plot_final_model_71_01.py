#!/usr/bin/env python3
"""Plot six experimental/model half-plane comparisons for final model 71_01."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAIR_ROOT = HERE.parents[1]
TEMPLATE = (
    PAIR_ROOT
    / "LiTiOF graphing"
    / "final_model_61_02"
    / "plot_final_model_61_02.py"
)
MODEL_ROOT = (
    PAIR_ROOT
    / "3d-deltaPDF_informed"
    / "run71"
    / "01_run61_02_displacements"
)


def load_template():
    spec = importlib.util.spec_from_file_location("final_model_halfplane_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting workflow from {TEMPLATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    plotting = load_template()
    plotting.MODEL_NAME = "71_01"
    plotting.MODEL_DESCRIPTION = (
        "run 71 case 01; J1/kBT=-0.5, J2/kBT=+0.5; run 61_02 displacements"
    )
    plotting.MODEL_ROOT = MODEL_ROOT
    plotting.MODEL_SCATTERING = (
        MODEL_ROOT
        / "scatty_structures"
        / "rocks_run71_01_run61_02_displacements_kbt1_q10_exp20_bragg_removed_hk0_sc_list.npz"
    )
    plotting.OUTPUT_STEM = HERE / "final_model_71_01_halfplane_comparison"
    plotting.main()


if __name__ == "__main__":
    main()
