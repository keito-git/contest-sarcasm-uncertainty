"""
make_figures_r1.py -- result figures of the KBS R1 manuscript as 600-dpi PNG.

Inputs : results/revision_r1_2026-11/R1_CPU.json (Qwen2.5, all corpora)
         results/revision_r1_2026-11/R1_GPU.json (OLMo-2), optional
Outputs: paper/journal_kbs/revision_R1/manuscript_R1/figures/
           fig_capacity_r1.png   balanced accuracy per stratum across sizes
           fig_delta_asym_r1.png DeltaAsym with 95% bootstrap intervals
KBS does not accept PDF figures, so only PNG is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "results" / "revision_r1_2026-11"
OUT = PROJ / "results" / "revision_r1_2026-11" / "figures"
CORPORA = ["CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]
NAMES = {"CSC": "CSC", "EPIC": "EPIC", "MultiPICo_es": "MultiPICo-es",
         "MultiPICo_de": "MultiPICo-de", "MultiPICo_fr": "MultiPICo-fr",
         "MultiPICo_it": "MultiPICo-it"}
QWEN = {"q0p5b": 0.5, "q1p5b": 1.5, "q3b": 3, "q7b": 7, "q14b": 14, "q32b": 32}
OLMO = {"o1b": 1, "o7b": 7, "o13b": 13, "o32b": 32}
BLUE, RED = "#1f4e9c", "#c62828"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(name: str) -> dict | None:
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def fig_capacity(cpu: dict, gpu: dict | None) -> None:
    fam = (gpu or {}).get("family", {})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, CORPORA):
        r = cpu["S5_full_range"][c]
        x = [QWEN[t] for t in QWEN]
        ax.plot(x, [r["balacc_low_by_size"][t] for t in QWEN], "-o", color=BLUE,
                lw=1.6, ms=4, label="Qwen2.5, low disagreement")
        ax.plot(x, [r["balacc_high_by_size"][t] for t in QWEN], "-s", color=RED,
                lw=1.6, ms=4, label="Qwen2.5, high disagreement")
        if c in fam:
            s = fam[c]["slope"]
            xo = [OLMO[t] for t in OLMO]
            ax.plot(xo, s["balacc_low"], "--o", color=BLUE, mfc="white", lw=1.2, ms=4,
                    label="OLMo-2, low disagreement")
            ax.plot(xo, s["balacc_high"], "--s", color=RED, mfc="white", lw=1.2, ms=4,
                    label="OLMo-2, high disagreement")
        ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
        ax.minorticks_off()
        ax.set_title(NAMES[c], fontsize=10)
        ax.grid(axis="y", color="0.9", lw=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Parameters (B)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced accuracy")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_capacity_r1.png", dpi=600)
    plt.close(fig)


def fig_delta(cpu: dict, gpu: dict | None) -> None:
    """(a) DeltaAsym on balanced accuracy; (b) threshold-free AUROC variant."""
    fam = (gpu or {}).get("family", {})
    auc = (gpu or {}).get("auroc_variant", {})
    y = np.arange(len(CORPORA))
    two = bool(fam) and bool(auc)
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(7.2 if two else 4.8, 3.1),
                             sharey=True, squeeze=False)
    off = 0.17 if fam else 0.0

    def draw(ax, get_q, get_o, xlabel, title):
        for i, c in enumerate(CORPORA):
            for val, ci_, yy, sty in ((get_q(c), None, i - off, "q"), (get_o(c), None, i + off, "o")):
                if val is None:
                    continue
                d, (lo, hi) = val
                if sty == "q":
                    ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="o", color=BLUE, ms=5,
                                capsize=3, lw=1.2, label="Qwen2.5, 0.5B to 32B" if i == 0 else None)
                else:
                    ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="s", color="0.35",
                                mfc="white", ms=5, capsize=3, lw=1.2,
                                label="OLMo-2, 1B to 32B" if i == 0 else None)
        ax.axvline(0, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="0.9", lw=0.6)
        if title:
            ax.set_title(title, fontsize=10)

    def q_ba(c):
        e = cpu["S2_alt_strat"][c]["tercile_primary"]
        return e["dasym"], e["ci95"]

    def o_ba(c):
        if c not in fam:
            return None
        f = fam[c]["primary_1b_32b"]
        return f["dasym"], f["ci95"]

    draw(axes[0, 0], q_ba, o_ba, r"$\Delta\mathrm{Asym}$", "(a) Balanced accuracy" if two else None)
    if two:
        draw(axes[0, 1],
             lambda c: (auc[c]["Qwen25"]["dasym_auroc"], auc[c]["Qwen25"]["ci95"]),
             lambda c: (auc[c]["OLMo2"]["dasym_auroc"], auc[c]["OLMo2"]["ci95"]),
             r"$\Delta\mathrm{Asym}$", "(b) Within-stratum AUROC")
    ax0 = axes[0, 0]
    ax0.set_yticks(y)
    ax0.set_yticklabels([NAMES[c] for c in CORPORA])
    ax0.invert_yaxis()
    if fam:
        h, lab = ax0.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(OUT / "fig_delta_asym_r1.png", dpi=600)
    plt.close(fig)


def main() -> None:
    cpu, gpu = load("R1_CPU.json"), load("R1_GPU.json")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_capacity(cpu, gpu)
    fig_delta(cpu, gpu)
    print("wrote figures; OLMo-2 included:", bool((gpu or {}).get("family")))


if __name__ == "__main__":
    main()
