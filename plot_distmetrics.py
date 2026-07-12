#!/usr/bin/env python3
"""
Bold-only version of distmetric figures.

Keeps original figsize (9.6 x 2.7) and font sizes (8-11 pt) exactly as the
original plot_metareview_distmetrics.py; adds font.weight='bold' throughout
and slightly thicker lines/markers. No other changes.

Data: results/metareview/dist_metrics_curves.csv
Out : figures/
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
    "font.weight": "bold",           # << bold added
    "axes.labelweight": "bold",      # << bold added
    "axes.titleweight": "bold",      # << bold added
    "axes.linewidth": 1.0,           # slightly thicker than original 0.8
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(BASE, "results", "metareview", "dist_metrics_curves.csv")
OUT  = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

c = pd.read_csv(CSV)
CORPORA = ["CSC", "MultiPICo", "EPIC"]
LABELS  = {"CSC": "(a) CSC", "MultiPICo": "(b) MultiPICo", "EPIC": "(c) EPIC"}

for metric, ylab, fname in [
    ("jsd",   "JS divergence to human dist. (bits)", "fig_distmetric_jsd.pdf"),
    ("brier", "Brier vs human distribution",          "fig_distmetric_brier.pdf"),
]:
    # Original figsize preserved exactly
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), sharey=True)

    for ax, corpus in zip(axes, CORPORA):
        sub = c[(c.corpus == corpus) & (c.metric == metric)]
        piv = sub.pivot(index="size", columns="stratum", values="value")

        ax.fill_between(piv.index, piv["low"], piv["high"], color="0.85", zorder=0)
        ax.plot(piv.index, piv["low"],  "-o",
                color="#1f4e9c", lw=2.5, ms=6,
                label="Low disagreement (epistemic)")
        ax.plot(piv.index, piv["high"], "--s",
                color="#c62828", lw=2.5, ms=6,
                label="High disagreement (aleatoric)")

        ax.set_xscale("log")
        ax.set_xticks([0.5, 1.5, 3, 7, 14, 32])
        ax.set_xticklabels(
            ["0.5", "1.5", "3", "7", "14", "32"],
            fontsize=9, fontweight="bold",          # original size=9, + bold
        )
        ax.set_xlabel("Qwen2.5 model size (B, log scale)",
                      fontsize=10, fontweight="bold")  # original size=10, + bold
        ax.set_title(LABELS[corpus], fontsize=11, fontweight="bold")  # original 11, + bold
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
        ax.tick_params(axis="y", labelsize=9, width=1.2)

    axes[0].set_ylabel(ylab, fontsize=10, fontweight="bold")
    axes[0].legend(
        fontsize=8,
        loc="best",
        framealpha=0.9,
        prop={"weight": "bold", "size": 8},         # original size=8, + bold
    )

    fig.tight_layout()
    out_path = os.path.join(OUT, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote  {out_path}")
