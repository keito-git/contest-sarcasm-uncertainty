#!/usr/bin/env python3
"""
Visualization for strong baseline comparison (AAAI 2028).

Produces:
  results/strong_baselines/figures/auroc_comparison.pdf   (paper-quality)
  results/strong_baselines/figures/auroc_comparison.png   (300 dpi)
  results/strong_baselines/figures/aucac_comparison.png   (300 dpi)

Design: Cleveland dot plot with 95% CI error bars.
- Horizontal layout: AUROC on x-axis, methods on y-axis.
- Three panels (one per corpus), arranged vertically.
- Color coding by method group (emphasis pattern):
    "ours" (contraction probe): blue  #2a78d6
    "oracle/reference" (oracle, peale_oracle): gray  #898781
    "strong baselines": aqua  #1baf7a
    "weak baselines": red  #e34948
- Contraction probe row highlighted with a shaded band.
- 95% CI shown as horizontal error bars.
- Vertical dashed line at AUROC=0.5 (chance).
- Degenerate semantic_entropy marked with † and excluded from main rows
  (identical to H_small by construction for binary tasks).
- Colorblind-friendly: shape (filled circle for deployable,
  open square for annotation-required / non-deployable) as secondary encoding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed fonts for PDF
        "ps.fonttype": 42,
    }
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "strong_baselines"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette (from validated reference palette)
# ---------------------------------------------------------------------------
COLOR_OURS = "#2a78d6"       # blue — contraction probe (ours)
COLOR_ORACLE = "#898781"     # muted gray — oracle/annotation-required (not deployable)
COLOR_STRONG = "#1baf7a"     # aqua — strong deployable baselines
COLOR_WEAK = "#e34948"       # red — weak baselines
COLOR_HIGHLIGHT_BG = "#e8f2fd"  # very light blue band for "ours" row
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_DASHED = "#c3c2b7"    # reference line at 0.5

# ---------------------------------------------------------------------------
# Method display names and grouping
# ---------------------------------------------------------------------------
# Group assignment: deployable strong / weak / ours / oracle
METHOD_META: dict[str, dict] = {
    "H_small":          {"label": "H_small (sampling entropy)", "group": "weak"},
    "conf_small":       {"label": "conf_small (|2p−1|)",         "group": "weak"},
    "semantic_entropy": {"label": "Semantic entropy (†=H_small)", "group": "weak"},
    "UCCI_calibrated":  {"label": "UCCI-calibrated (Kotte 2026 type)", "group": "strong"},
    "deep_ensemble_std":{"label": "Deep ensemble std",           "group": "strong"},
    "peale_annfree":    {"label": "Peale annotation-free",       "group": "strong"},
    "phillips_probe":   {"label": "Phillips probe (supervised GB)", "group": "strong"},
    "contraction":      {"label": "Contraction probe [ours]",    "group": "ours"},
    "peale_oracle":     {"label": "Peale oracle (†annot. req.)", "group": "oracle"},
    "oracle":           {"label": "Oracle y_fix (upper bound)",  "group": "oracle"},
}

# Display order (bottom → top in horizontal plot)
DISPLAY_ORDER = [
    "oracle",
    "peale_oracle",
    "contraction",
    "phillips_probe",
    "peale_annfree",
    "deep_ensemble_std",
    "UCCI_calibrated",
    "conf_small",
    "H_small",
    "semantic_entropy",
]

GROUP_COLOR = {
    "weak":   COLOR_WEAK,
    "strong": COLOR_STRONG,
    "ours":   COLOR_OURS,
    "oracle": COLOR_ORACLE,
}

# Marker shape: deployable → filled circle; not-deployable → open square
DEPLOYABLE = {"H_small", "conf_small", "semantic_entropy",
              "UCCI_calibrated", "deep_ensemble_std",
              "peale_annfree", "phillips_probe", "contraction"}
MARKER_STYLE = {m: "o" if m in DEPLOYABLE else "s" for m in METHOD_META}
MARKER_FILLED = {m: True if m in DEPLOYABLE else False for m in METHOD_META}


def plot_auroc(auroc_df: pd.DataFrame) -> None:
    """Three-panel horizontal dot plot (one panel per corpus)."""
    corpora = ["CSC", "MultiPICo", "EPIC"]
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 5.2), sharey=True,
        gridspec_kw={"wspace": 0.06}
    )

    # Build y-positions
    # Exclude oracle (AUROC=1.0) from visible range; include as top row with clip
    n_methods = len(DISPLAY_ORDER)
    y_pos = {m: i for i, m in enumerate(DISPLAY_ORDER)}

    for ax, corpus in zip(axes, corpora):
        grp = auroc_df[auroc_df.corpus == corpus].set_index("signal")

        # Draw grid
        ax.set_axisbelow(True)
        for y in y_pos.values():
            ax.axhline(y, color=COLOR_GRID, linewidth=0.4, zorder=0)
        ax.axvline(0.5, color=COLOR_DASHED, linewidth=0.8, linestyle="--",
                   zorder=1, label="Chance (0.5)")

        # Highlight "ours" row
        ours_y = y_pos["contraction"]
        ax.axhspan(ours_y - 0.45, ours_y + 0.45,
                   facecolor=COLOR_HIGHLIGHT_BG, edgecolor="none", zorder=0)

        for method in DISPLAY_ORDER:
            if method not in grp.index:
                continue
            row = grp.loc[method]
            auroc = float(row["auroc"])
            ci_lo = float(row["ci_lo"])
            ci_hi = float(row["ci_hi"])
            y = y_pos[method]
            meta = METHOD_META[method]
            color = GROUP_COLOR[meta["group"]]
            marker = MARKER_STYLE[method]
            is_filled = MARKER_FILLED[method]

            # Error bar
            ax.plot(
                [ci_lo, ci_hi], [y, y],
                color=color, linewidth=1.2, alpha=0.7, zorder=2
            )
            # Dot
            ms = 7 if method == "contraction" else 6
            mfc = color if is_filled else "white"
            mec = color
            ax.plot(
                auroc, y,
                marker=marker, markersize=ms,
                markerfacecolor=mfc, markeredgecolor=mec,
                markeredgewidth=1.5, color=color, zorder=3,
                linewidth=0
            )
            # Value label for contraction probe and strong baselines
            if method in {"contraction", "phillips_probe", "deep_ensemble_std"}:
                ha = "left" if auroc < 0.75 else "right"
                offset = 0.01 if ha == "left" else -0.01
                ax.text(
                    auroc + offset, y + 0.2,
                    f"{auroc:.3f}",
                    ha=ha, va="bottom", fontsize=7,
                    color=color, fontweight="bold" if method == "contraction" else "normal"
                )

        # Axis formatting
        ax.set_title(corpus, fontsize=10, fontweight="bold", pad=5,
                     color=COLOR_TEXT_PRIMARY)
        ax.set_xlim(0.35, 1.05)
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                           color=COLOR_TEXT_SECONDARY)
        ax.tick_params(axis="x", length=3, color=COLOR_GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_edgecolor(COLOR_GRID)
        ax.set_xlabel("AUROC (↑ better)", color=COLOR_TEXT_SECONDARY)

    # Y-tick labels on leftmost panel
    axes[0].set_yticks(list(y_pos.values()))
    axes[0].set_yticklabels(
        [METHOD_META[m]["label"] for m in DISPLAY_ORDER],
        color=COLOR_TEXT_PRIMARY,
    )
    axes[0].tick_params(axis="y", length=0)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_OURS, label="Contraction probe [ours]"),
        mpatches.Patch(facecolor=COLOR_STRONG, label="Strong baseline (deployable)"),
        mpatches.Patch(facecolor=COLOR_WEAK, label="Weak baseline"),
        mpatches.Patch(facecolor=COLOR_ORACLE, label="Oracle / annotation-req."),
        plt.Line2D([0], [0], color=COLOR_DASHED, linewidth=0.8,
                   linestyle="--", label="Chance (0.5)"),
        plt.Line2D([0], [0], marker="s", markersize=6, markerfacecolor="white",
                   markeredgecolor=COLOR_TEXT_SECONDARY, linewidth=0,
                   label="Not deployable (open marker)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        labelcolor=COLOR_TEXT_PRIMARY,
    )

    fig.suptitle(
        "AUROC: Strong Baseline Comparison — Contraction Probe (AAAI 2028)\n"
        "Population: 0.5B errors; y_fix = 1 if Qwen-32B correct. 95% CI bootstrap (n=2000).",
        fontsize=9.5, color=COLOR_TEXT_PRIMARY, y=1.01
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    for ext in ["pdf", "png"]:
        path = FIG / f"auroc_comparison.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_aucac(aucac_df: pd.DataFrame) -> None:
    """Three-panel horizontal dot plot for AUCAC."""
    corpora = ["CSC", "MultiPICo", "EPIC"]
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 5.2), sharey=True,
        gridspec_kw={"wspace": 0.06}
    )
    # Exclude oracle from main display (very high)
    disp = [m for m in DISPLAY_ORDER if m != "oracle"]
    n = len(disp)
    y_pos = {m: i for i, m in enumerate(disp)}

    # Get per-corpus random AUCAC references from diff_vs_random
    # random_aucac[corpus] = aucac - diff_vs_random
    for ax, corpus in zip(axes, corpora):
        grp = aucac_df[aucac_df.corpus == corpus].set_index("signal")
        # Compute random AUCAC reference
        if "contraction" in grp.index:
            random_aucac = float(grp.loc["contraction", "aucac"]) - float(grp.loc["contraction", "diff_vs_random"])
        else:
            random_aucac = 0.58

        ax.set_axisbelow(True)
        for y in y_pos.values():
            ax.axhline(y, color=COLOR_GRID, linewidth=0.4, zorder=0)
        ax.axvline(random_aucac, color=COLOR_DASHED, linewidth=0.8,
                   linestyle="--", zorder=1, label=f"Random ({random_aucac:.3f})")

        # Highlight "ours" row
        ours_y = y_pos["contraction"]
        ax.axhspan(ours_y - 0.45, ours_y + 0.45,
                   facecolor=COLOR_HIGHLIGHT_BG, edgecolor="none", zorder=0)

        for method in disp:
            if method not in grp.index:
                continue
            row = grp.loc[method]
            aucac = float(row["aucac"])
            ci_lo = float(row["aucac_ci_lo"]) if "aucac_ci_lo" in row else float(row["ci_lo"])
            ci_hi = float(row["aucac_ci_hi"]) if "aucac_ci_hi" in row else float(row["ci_hi"])
            y = y_pos[method]
            meta = METHOD_META[method]
            color = GROUP_COLOR[meta["group"]]
            marker = MARKER_STYLE[method]
            is_filled = MARKER_FILLED[method]

            ax.plot([ci_lo, ci_hi], [y, y],
                    color=color, linewidth=1.2, alpha=0.7, zorder=2)
            mfc = color if is_filled else "white"
            ms = 7 if method == "contraction" else 6
            ax.plot(aucac, y, marker=marker, markersize=ms,
                    markerfacecolor=mfc, markeredgecolor=color,
                    markeredgewidth=1.5, linewidth=0, zorder=3)
            if method in {"contraction", "phillips_probe", "peale_oracle"}:
                ax.text(aucac + 0.003, y + 0.2, f"{aucac:.4f}",
                        ha="left", va="bottom", fontsize=7, color=color,
                        fontweight="bold" if method == "contraction" else "normal")

        ax.set_title(corpus, fontsize=10, fontweight="bold", pad=5,
                     color=COLOR_TEXT_PRIMARY)
        ax.set_xlim(0.52, 0.70)
        ax.set_xticks([0.55, 0.58, 0.61, 0.64, 0.67, 0.70])
        ax.set_xticklabels(["0.55", "0.58", "0.61", "0.64", "0.67", "0.70"],
                           color=COLOR_TEXT_SECONDARY)
        ax.tick_params(axis="x", length=3, color=COLOR_GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_edgecolor(COLOR_GRID)
        ax.set_xlabel("AUCAC (↑ better)", color=COLOR_TEXT_SECONDARY)

    axes[0].set_yticks(list(y_pos.values()))
    axes[0].set_yticklabels(
        [METHOD_META[m]["label"] for m in disp], color=COLOR_TEXT_PRIMARY
    )
    axes[0].tick_params(axis="y", length=0)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_OURS, label="Contraction probe [ours]"),
        mpatches.Patch(facecolor=COLOR_STRONG, label="Strong baseline (deployable)"),
        mpatches.Patch(facecolor=COLOR_WEAK, label="Weak baseline"),
        mpatches.Patch(facecolor=COLOR_ORACLE, label="Oracle / annotation-req."),
        plt.Line2D([0], [0], color=COLOR_DASHED, linewidth=0.8,
                   linestyle="--", label="Random baseline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05),
               labelcolor=COLOR_TEXT_PRIMARY)

    fig.suptitle(
        "AUCAC: Strong Baseline Comparison — Contraction Probe (AAAI 2028)\n"
        "Cascade: Qwen-0.5B → 32B; routing by each signal. "
        "Area under balanced-acc curve. 95% CI bootstrap (n=2000).",
        fontsize=9.5, color=COLOR_TEXT_PRIMARY, y=1.01
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = FIG / "aucac_comparison.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    auroc_df = pd.read_csv(OUT / "auroc_comparison.csv")
    aucac_df = pd.read_csv(OUT / "aucac_comparison.csv")

    print("Plotting AUROC comparison...")
    plot_auroc(auroc_df)

    print("Plotting AUCAC comparison...")
    plot_aucac(aucac_df)

    print(f"\nAll figures saved to: {FIG}")


if __name__ == "__main__":
    main()
