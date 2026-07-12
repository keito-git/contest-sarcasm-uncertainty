"""
Post-hoc visualisation for MHS hate-speech generalisation experiment.

Generates:
  results/llm_hate/figures/
    hate_e1_err_by_dis.png      — E1: error rate by disagreement stratum
    hate_e2_rho_bar.png         — E2: Spearman rho bar chart (with 95% CI)
    hate_vs_sarcasm_summary.png — Side-by-side low_dis_err + rho across tasks

All figures are colourblind-safe (CB palette), 300 dpi, font >= 11pt.
"""
from __future__ import annotations
from pathlib import Path

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = str(Path(__file__).parent.parent)
HATE_OUT = f"{BASE}/results/llm_hate"
FIG_DIR = f"{HATE_OUT}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---- colourblind-safe palette (Wong 2011) ------------------------------------
CB = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
}

MODEL_COLORS = {
    "gpt-4o-mini":       CB["blue"],
    "gpt-4.1":           CB["sky"],
    "claude-haiku-4.5":  CB["green"],
    "claude-opus-4.8":   CB["orange"],
}

# ---- load summary ------------------------------------------------------------

def load_summary(phase: str = "hate1") -> list[dict]:
    path = f"{HATE_OUT}/e1_hate_summary_{phase}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["results"]


# ---- figure 1: E1 error rate by disagreement stratum -----------------------

def plot_e1_err(summ: list[dict], phase: str) -> None:
    models = [a for a in summ if "acc" in a]
    if not models:
        print("[plot_e1_err] no valid models"); return

    labels = [a["model_display"] for a in models]
    agree_err = [a.get("agree_err", float("nan")) for a in models]
    disagree_err = [a.get("disagree_err", float("nan")) for a in models]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w / 2, agree_err, w,
                   color=[MODEL_COLORS.get(l, CB["black"]) for l in labels],
                   alpha=0.9, label="Low dis (humans all agree)")
    bars2 = ax.bar(x + w / 2, disagree_err, w,
                   color=[MODEL_COLORS.get(l, CB["black"]) for l in labels],
                   alpha=0.45, hatch="//", label="High dis (humans disagree)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Error rate", fontsize=12)
    ax.set_ylim(0, 0.6)
    ax.set_title("E1 — Error rate by human-disagreement stratum (MHS)", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(0.1, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.35)

    # value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = f"{FIG_DIR}/hate_e1_err_by_dis.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {path}")


# ---- figure 2: E2 Spearman rho -----------------------------------------------

def plot_e2_rho(summ: list[dict], phase: str) -> None:
    models = [a for a in summ if "rho_Uverbal_dis" in a]
    if not models:
        print("[plot_e2_rho] no valid models"); return

    labels = [a["model_display"] for a in models]
    rhos = [a["rho_Uverbal_dis"]["rho"] for a in models]
    ci_lo = [a["rho_Uverbal_dis"]["ci"][0] for a in models]
    ci_hi = [a["rho_Uverbal_dis"]["ci"][1] for a in models]
    err_lo = [r - l for r, l in zip(rhos, ci_lo)]
    err_hi = [h - r for r, h in zip(rhos, ci_hi)]

    x = np.arange(len(labels))
    colors = [MODEL_COLORS.get(l, CB["black"]) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(x, rhos, xerr=[err_lo, err_hi], color=colors, alpha=0.85,
            error_kw={"ecolor": "black", "capsize": 4})
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Spearman rho(U_verbal, dis_mi)", fontsize=12)
    ax.set_title("E2 — Single-signal failure: U_verbal vs human disagreement (MHS)",
                 fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.30, color=CB["red"], linewidth=1.2, linestyle="--", alpha=0.7,
               label="ρ = 0.30 threshold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.35)

    for i, (r, lo, hi) in enumerate(zip(rhos, ci_lo, ci_hi)):
        ax.text(r + 0.01, i, f"{r:+.2f}", va="center", fontsize=9)

    fig.tight_layout()
    path = f"{FIG_DIR}/hate_e2_rho_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {path}")


# ---- figure 3: cross-task summary -------------------------------------------

SARCASM_REFS = {
    # (dataset, model_display): (low_dis_err, rho_Uverbal_dis)
    ("CSC", "gpt-4.1"):          (0.119, +0.123),
    ("CSC", "claude-opus-4.8"):  (0.170, +0.142),
    ("MultiPICo", "gpt-4.1"):    (0.109, +0.261),
    ("MultiPICo", "claude-opus-4.8"): (0.080, +0.290),
}


def plot_cross_task(summ: list[dict], phase: str) -> None:
    models_of_interest = {"gpt-4.1", "claude-opus-4.8"}
    hate_models = {a["model_display"]: a for a in summ
                   if "acc" in a and a["model_display"] in models_of_interest}
    if not hate_models:
        print("[plot_cross_task] missing gpt-4.1 or claude-opus-4.8 results; skipping")
        return

    # build comparison data
    rows = []
    for (ds, md), (err, rho) in SARCASM_REFS.items():
        rows.append({"task": ds, "model": md, "low_dis_err": err, "rho": rho})
    for md, a in hate_models.items():
        err = a.get("agree_err", a.get("low_dis_err", float("nan")))
        rho = a["rho_Uverbal_dis"]["rho"]
        rows.append({"task": "MHS (hate)", "model": md, "low_dis_err": err, "rho": rho})

    tasks = sorted(set(r["task"] for r in rows))
    model_names = sorted(set(r["model"] for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_i, (col, ylabel, title) in enumerate([
        ("low_dis_err", "Error rate (low-dis)", "E1: LLM error on human-agreed items"),
        ("rho", "Spearman rho", "E2: Single-signal failure rho"),
    ]):
        ax = axes[ax_i]
        x = np.arange(len(tasks))
        n_models = len(model_names)
        width = 0.8 / n_models
        for j, mn in enumerate(model_names):
            vals = []
            for t in tasks:
                hit = [r[col] for r in rows if r["task"] == t and r["model"] == mn]
                vals.append(hit[0] if hit else float("nan"))
            offset = (j - n_models / 2 + 0.5) * width
            c = MODEL_COLORS.get(mn, CB["black"])
            ax.bar(x + offset, vals, width, label=mn, color=c, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=10, rotation=10, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.35)
        if col == "rho":
            ax.axhline(0.30, color=CB["red"], linewidth=1.2, linestyle="--", alpha=0.7)
            ax.set_ylim(-0.1, 0.5)

    fig.suptitle("Cross-task generalisation: sarcasm (CSC/MultiPICo) vs hate speech (MHS)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = f"{FIG_DIR}/hate_vs_sarcasm_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {path}")


# ---- main --------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="hate1")
    args = ap.parse_args()

    summ = load_summary(args.phase)
    print(f"Loaded {len(summ)} model result(s) from phase {args.phase}")

    plot_e1_err(summ, args.phase)
    plot_e2_rho(summ, args.phase)
    plot_cross_task(summ, args.phase)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
