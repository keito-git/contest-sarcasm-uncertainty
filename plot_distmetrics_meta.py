#!/usr/bin/env python3
"""Plot distribution-sensitive (JSD to human distribution) capacity curves per
disagreement stratum, 3 panels (CSC/MultiPICo/EPIC). Times font per lab rule."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})

OUT = "paper/en/figures"
c = pd.read_csv("results/metareview/dist_metrics_curves.csv")
CORPORA = ["CSC", "MultiPICo", "EPIC"]
LABELS = {"CSC": "(a) CSC", "MultiPICo": "(b) MultiPICo", "EPIC": "(c) EPIC"}

for metric, ylab, fname in [
    ("jsd", "JS divergence to human dist. (bits)", "fig_distmetric_jsd.pdf"),
    ("brier", "Brier vs human distribution", "fig_distmetric_brier.pdf"),
]:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), sharey=True)
    for ax, corpus in zip(axes, CORPORA):
        sub = c[(c.corpus == corpus) & (c.metric == metric)]
        piv = sub.pivot(index="size", columns="stratum", values="value")
        ax.plot(piv.index, piv["low"], "-o", color="#1f4e9c", lw=2, ms=5,
                label="Low disagreement (epistemic)")
        ax.plot(piv.index, piv["high"], "--s", color="#c62828", lw=2, ms=5,
                label="High disagreement (aleatoric)")
        ax.fill_between(piv.index, piv["low"], piv["high"], color="0.85", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1.5, 3, 7, 14, 32])
        ax.set_xticklabels(["0.5", "1.5", "3", "7", "14", "32"], fontsize=9)
        ax.set_xlabel("Qwen2.5 model size (B, log scale)", fontsize=10)
        ax.set_title(LABELS[corpus], fontsize=11)
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel(ylab, fontsize=10)
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    print("wrote", f"{OUT}/{fname}")
