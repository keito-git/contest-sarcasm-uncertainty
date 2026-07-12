#!/usr/bin/env python3
"""
Bold-only version of fig_capacity_curves.pdf.

Keeps original figsize (~9.6 x 2.9 to match PDF 680.9 x 206.1 pts)
and original font sizes (9-11 pt). Adds font.weight='bold' throughout
and slightly thicker lines/markers.

Data: results/llm_e3/capacity_curve_data.csv
  columns: ds, size, low, high
Out : figures/fig_capacity_curves.pdf
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.0,
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(BASE, "results", "llm_e3", "capacity_curve_data.csv")
OUT  = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
CORPORA = ["CSC", "MultiPICo", "EPIC"]
LABELS  = {"CSC": "(a) CSC", "MultiPICo": "(b) MultiPICo", "EPIC": "(c) EPIC"}

# figsize chosen to reproduce original PDF 680.9 x 206.1 pts (9.46 x 2.86 in)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)

for ax, corpus in zip(axes, CORPORA):
    sub   = df[df["ds"] == corpus].sort_values("size")
    sizes = sub["size"].values
    low   = sub["low"].values
    high  = sub["high"].values

    ax.fill_between(sizes, low, high, color="0.85", zorder=0)
    ax.plot(sizes, low,  "-o",
            color="#1f4e9c", lw=2.5, ms=6,
            label="Low disagreement (epistemic)")
    ax.plot(sizes, high, "--s",
            color="#c62828", lw=2.5, ms=6,
            label="High disagreement (aleatoric)")
    ax.axhline(0.5, color="0.4", lw=1.0, ls=":", zorder=1)

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.5, 3, 7, 14, 32])
    ax.set_xticklabels(
        ["0.5", "1.5", "3", "7", "14", "32"],
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Qwen2.5 model size (B, log scale)",
                  fontsize=10, fontweight="bold")
    ax.set_title(LABELS[corpus], fontsize=11, fontweight="bold")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=9, width=1.2)

axes[0].set_ylabel("Balanced accuracy", fontsize=10, fontweight="bold")
axes[0].legend(
    fontsize=8,
    loc="upper left",
    framealpha=0.9,
    prop={"weight": "bold", "size": 8},
)

fig.tight_layout()
out_path = os.path.join(OUT, "fig_capacity_curves.pdf")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"wrote  {out_path}")
