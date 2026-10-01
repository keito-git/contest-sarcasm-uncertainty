#!/usr/bin/env python3
"""
make_figures.py — kbs/paper/make_figures.py
Generates all paper figures from JSON result files.

Figures produced:
  $CONTEST_PAPER_OUT/figures/fig_capacity.pdf    — BalAcc vs Qwen2.5 size (6 panels × 2 tercile lines)
  $CONTEST_PAPER_OUT/figures/fig_delta_asym.pdf  — ΔAsym per corpus with bootstrap 95% CIs
  $CONTEST_PAPER_OUT/figures/fig_auroc_probes.pdf — Contraction-probe AUROC vs diversity probes per corpus

Style rules:
  - Font: Times New Roman throughout (fallback: DejaVu Serif)
  - pdf.fonttype = 42 (Type 1 / TrueType embedding, no Type 3)
  - Minimum label size 8pt, tick 7pt
  - Colors: colorblind-safe
  - No parenthetical captions; figure text is minimal
  - Figure placement [t] in LaTeX (handled in .tex, not here)

Usage:
    python kbs/paper/make_figures.py   (from the repository root)

Dependencies:
    matplotlib, numpy (standard scientific stack)
    results/revision_gpu_2026-09/regen_analysis/C7_REGEN_20260925T182937.json
    results/revision_gpu_2026-09/regen_analysis/C8_REGEN_20260925T183127.json
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- Font setup ----
import matplotlib.font_manager as fm
_FONT_CANDIDATES = ["Times New Roman", "DejaVu Serif", "serif"]
_FOUND_FONT = None
for _f in _FONT_CANDIDATES:
    if any(_f.lower() in ff.name.lower() for ff in fm.fontManager.ttflist):
        _FOUND_FONT = _f
        break
if _FOUND_FONT is None:
    _FOUND_FONT = "serif"

plt.rcParams.update({
    "font.family":       _FOUND_FONT,
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
})

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR, PAPER_OUT  # noqa: E402  (repo-root path config)
BASE   = WORK_DIR
FIGDIR = PAPER_OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

C7_PATH  = BASE / "results/revision_gpu_2026-09/regen_analysis/C7_REGEN_20260925T182937.json"
C8_PATH  = BASE / "results/revision_gpu_2026-09/regen_analysis/C8_REGEN_20260925T183127.json"
# C1: actual bootstrap CIs for ΔAsym figure — no fallback allowed
C1_PATH  = BASE / "results/revision_gpu_2026-09/round1_analysis/C1_DASYM_CI.json"

# Qwen2.5 size labels in log scale (M params)
SIZE_TAGS    = ["q0p5b", "q1p5b", "q3b",  "q7b",  "q14b", "q32b"]
SIZE_LABELS  = ["0.5B",  "1.5B",  "3B",   "7B",   "14B",  "32B"]
SIZE_MPARAMS = [500, 1500, 3000, 7000, 14000, 32000]

# Corpus display names (full, for all figures)
CORPUS_ORDER = [
    ("CSC",          "CSC"),
    ("EPIC",         "EPIC"),
    ("MultiPICo_es", "MultiPICo-es"),
    ("MultiPICo_de", "MultiPICo-de"),
    ("MultiPICo_fr", "MultiPICo-fr"),
    ("MultiPICo_it", "MultiPICo-it"),
]

COL_LOW  = "#2166AC"   # blue  — low-disagreement tercile
COL_HIGH = "#D6604D"   # red   — high-disagreement tercile
CI_ALPHA = 0.20


def load_json(path):
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    with open(path) as f:
        return json.load(f)


# ================================================================
# Figure 1: BalAcc vs Qwen2.5 capacity — 6 panels (2×3)
# Legend placed ABOVE panels; full corpus names on panel titles
# ================================================================
def make_fig_capacity():
    c7 = load_json(C7_PATH)
    tc = c7["per_corpus_tercile_acc"]

    # Legend placed in figure-level space above all panels;
    # extra top margin created with top= parameter
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.2),
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.50, wspace=0.28,
                        top=0.82, bottom=0.18, left=0.09, right=0.98)

    x = np.log10(SIZE_MPARAMS)
    x_ticks = x
    x_tick_labels = SIZE_LABELS

    handles_legend = None

    for ax, (corpus_key, label) in zip(axes.flat, CORPUS_ORDER):
        bal = tc[corpus_key]["bal_acc"]
        lo_vals, hi_vals = [], []
        for tag in SIZE_TAGS:
            lo_vals.append(bal[tag]["low_dis"]["bal_acc"])
            hi_vals.append(bal[tag]["high_dis"]["bal_acc"])

        lo_vals = np.array(lo_vals)
        hi_vals = np.array(hi_vals)

        h_lo, = ax.plot(x, lo_vals, color=COL_LOW,  marker="o", ms=3, lw=1.2,
                        label="Low disagreement")
        h_hi, = ax.plot(x, hi_vals, color=COL_HIGH, marker="s", ms=3, lw=1.2,
                        label="High disagreement")
        if handles_legend is None:
            handles_legend = [h_lo, h_hi]

        ax.set_title(label, fontsize=8, pad=3)
        ax.set_ylim(0.30, 0.90)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.set_xticks(x_ticks)
        # Alternating labels to avoid collisions at print width
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=6)

        ax.fill_between(x, lo_vals, hi_vals,
                        where=(lo_vals >= hi_vals),
                        alpha=CI_ALPHA, color=COL_LOW)

    # Shared y-label on left panels only
    for row in range(2):
        axes[row, 0].set_ylabel("Balanced accuracy", fontsize=8)

    # Shared x-axis label centred below all panels
    fig.text(0.535, 0.03, "Qwen2.5 model size", ha="center", fontsize=8)

    # Legend above all panels
    fig.legend(handles_legend,
               ["Low disagreement", "High disagreement"],
               loc="upper center",
               ncol=2,
               frameon=False,
               fontsize=7.5,
               bbox_to_anchor=(0.535, 0.97))

    outpath = FIGDIR / "fig_capacity.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ================================================================
# Figure 2: ΔAsym per corpus with 95% bootstrap CIs
# CIs read from C1_DASYM_CI.json — no fallback
# ================================================================
def make_fig_delta_asym():
    # Load C1 for CIs; C7 for asym_acc values (they match, but C1 has CI keys)
    c1 = load_json(C1_PATH)
    c1_results = c1["n_bootstrap"]  # validate key exists
    c1_rows = {(r["corpus"], r["subset"]): r for r in c1["results"]}

    labels, centers, lo_err, hi_err = [], [], [], []
    for corpus_key, label in CORPUS_ORDER:
        row = c1_rows.get((corpus_key, "full"))
        if row is None:
            raise KeyError(f"C1_DASYM_CI: no full entry for corpus={corpus_key}")
        da    = row["asym_acc"]
        # Raise immediately if CI keys missing — no fallback
        ci_lo = row["ci_95_lo"]
        ci_hi = row["ci_95_hi"]
        labels.append(label)
        centers.append(da)
        lo_err.append(da - ci_lo)
        hi_err.append(ci_hi - da)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    y = np.arange(len(labels))

    ax.barh(y, centers,
            xerr=[lo_err, hi_err],
            color=COL_LOW, alpha=0.75,
            error_kw=dict(ecolor="black", capsize=3, lw=1.0),
            height=0.55)

    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\Delta\mathrm{Asym}$ (balanced accuracy)", fontsize=8)
    ax.set_xlim(-0.05, 0.45)

    fig.tight_layout()
    outpath = FIGDIR / "fig_delta_asym.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ================================================================
# Figure 3: Contraction-probe AUROC vs ALL THREE diversity probes
#           + sampling-noise control per corpus with CIs.
# Legend entries name the actual models (from C8 env JSONs).
# ================================================================
def make_fig_auroc_probes():
    c8 = load_json(C8_PATH)
    pc = c8["per_corpus"]

    # Model tags → display labels (verified from env_c8_*.json):
    #   ll3b   = microsoft/Phi-3-mini-4k-instruct   (3.8 B)
    #   ll1b   = HuggingFaceTB/SmolLM2-1.7B-Instruct (1.7 B)
    #   gm2_2b = stabilityai/stablelm-zephyr-3b      (3 B)
    probe_keys = [
        ("contraction_probes", "contraction_q3b_q0p5b",
         "Capacity gap (Qwen2.5 3B vs 0.5B)",  COL_LOW,    "o"),
        ("diversity_probes",   "diversity_3b",
         "Diversity: Phi-3-mini-3.8B",          "#4DAC26",  "s"),
        ("diversity_probes",   "diversity_1b",
         "Diversity: SmolLM2-1.7B",             "#9970AB",  "^"),
        ("diversity_probes",   "diversity_gm2",
         "Diversity: StableLM-zephyr-3B",       "#E08214",  "D"),
        ("diversity_probes",   "sampling_noise",
         "Sampling noise (same model)",          "#BABABA",  "v"),
    ]

    labels = [label for _, label in CORPUS_ORDER]
    n = len(labels)
    x = np.arange(n)
    n_probes = len(probe_keys)
    total_width = 0.80
    width = total_width / n_probes

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    fig.subplots_adjust(bottom=0.28)

    for i, (section_key, probe_key, probe_label, color, marker) in enumerate(probe_keys):
        vals, ci_lo_arr, ci_hi_arr = [], [], []
        for corpus_key, _ in CORPUS_ORDER:
            probe_data = pc[corpus_key][section_key][probe_key]
            auroc = probe_data["AUROC"]
            ci_lo = probe_data["CI_low_95"]
            ci_hi = probe_data["CI_high_95"]
            vals.append(auroc)
            ci_lo_arr.append(auroc - ci_lo)
            ci_hi_arr.append(ci_hi - auroc)

        offset = (i - (n_probes - 1) / 2.0) * width
        ax.bar(x + offset, vals, width * 0.9,
               color=color, alpha=0.82, label=probe_label)
        ax.errorbar(x + offset, vals,
                    yerr=[ci_lo_arr, ci_hi_arr],
                    fmt="none", ecolor="black", capsize=2, lw=0.8)

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("AUROC", fontsize=8)
    ax.set_ylim(0.30, 0.80)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(frameon=False, fontsize=6.0, loc="upper right",
              ncol=2, bbox_to_anchor=(1.0, 1.0))

    outpath = FIGDIR / "fig_auroc_probes.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    print(f"Using font: {_FOUND_FONT}")
    make_fig_capacity()
    make_fig_delta_asym()
    make_fig_auroc_probes()
    print("All figures generated.")
