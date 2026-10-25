#!/usr/bin/env python3
"""
Master Runner: Generates all thesis figures inside `aoi_final_plan/thesis_artifacts/figures/`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR))

from gen_fig1_homography import generate_fig1
from gen_fig2_tolerance_heatmap import generate_fig2
from gen_fig3_box_fusion_recall import generate_fig3
from gen_fig4_openset_defect import generate_fig4
from gen_fig5_ablation_charts import generate_fig5


def main() -> None:
    t0 = time.perf_counter()
    print("=== Generating Thesis High-Resolution Figures (300 DPI) ===")
    p1 = generate_fig1()
    p2a, p2b = generate_fig2()
    p3 = generate_fig3()
    p4 = generate_fig4()
    p5 = generate_fig5()
    dt = time.perf_counter() - t0
    print(f"=== Completed 6 figures in {dt:.2f}s ===")
    for p in [p1, p2a, p2b, p3, p4, p5]:
        print(" ->", p)


if __name__ == "__main__":
    main()
