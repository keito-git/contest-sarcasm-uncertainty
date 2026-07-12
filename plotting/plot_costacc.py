#!/usr/bin/env python3
"""
Bold-only version of fig_costacc.pdf.

Keeps original figsize (~9.6 x 3.0 to match PDF 680.5 x 213.3 pts)
and original font sizes (8-11 pt). Adds font.weight='bold' throughout
and slightly thicker lines.

Data: results/llm_e3/cost_accuracy.pkl
  Format: (budgets_array, {corpus: {method: acc_array}})
  x-axis = budgets * 100  (Escalation budget, % routed to 32B)
Out : figures/fig_costacc.pdf
"""
import os
import pickle
import numpy as np
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

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(BASE, "results", "llm_e3", "cost_accuracy.pkl")
OUT      = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

budgets, corpus_dict = data
x = budgets * 100  # 0-100 %

CORPORA = ["CSC", "MultiPICo", "EPIC"]
LABELS  = {"CSC": "(a) CSC", "MultiPICo": "(b) MultiPICo", "EPIC": "(c) EPIC"}

# (method key, color, linestyle, linewidth) — visual style matches original
LINE_SPECS = [
    ("oracle",                "0.55",    "-",    1.6),
    ("human-dis (ref)",       "#2ca02c", "-.",   2.0),
    ("contraction (ours)",    "#1f77b4", "-",    2.8),
    ("conf (FrugalGPT-style)","#d62728", "--",   2.0),
    ("entropy",               "#ff7f0e", ":",    2.0),
    ("random",                "0.15",    ":",    1.6),
]

# figsize chosen to reproduce original PDF 680.5 x 213.3 pts (9.45 x 2.96 in)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)

for ax, corpus in zip(axes, CORPORA):
    methods = corpus_dict[corpus]
    for (key, color, ls, lw) in LINE_SPECS:
        if key not in methods:
            continue
        ax.plot(x, methods[key],
                color=color, linestyle=ls, linewidth=lw,
                label=key)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"],
                       fontsize=9, fontweight="bold")
    ax.set_xlabel("Escalation budget (% routed to 32B)",
                  fontsize=10, fontweight="bold")
    ax.set_title(LABELS[corpus], fontsize=11, fontweight="bold")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=9, width=1.2)

axes[0].set_ylabel("Routed-system balanced accuracy",
                   fontsize=10, fontweight="bold")

fig.tight_layout()

# Legend below the panels, in a single horizontal row.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=len(labels),
    fontsize=8,
    framealpha=0.9,
    prop={"weight": "bold", "size": 8},
    handlelength=2.0,
    columnspacing=1.2,
    bbox_to_anchor=(0.5, -0.02),
)
fig.subplots_adjust(bottom=0.24)

out_path = os.path.join(OUT, "fig_costacc.pdf")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"wrote  {out_path}")
