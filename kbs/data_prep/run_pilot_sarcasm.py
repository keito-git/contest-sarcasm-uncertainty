"""
Sarcasm/Irony "Ideal Testbed" Pilot — go/no-go for δ × Annotator Disagreement.

Purpose
-------
This is the primary-testbed validation of Claim C2:
    "δ (literal-intended discrepancy) predicts LLM miscalibration localization."

The prerequisite (tested here):
    δ correlates with HUMAN irreducible disagreement (aleatoric uncertainty proxy).

We use sarcasm/irony datasets where annotators genuinely disagree because the
meaning is contested, not because they are uniformed.

Datasets
--------
  CSC    Conversational Sarcasm Corpus (Jang & Frassinelli, NAACL 2024)
         GitHub: https://github.com/CoPsyN/CSC  (MIT license)
         7,036 context-response items / up to 360 annotators / 6-pt Likert.
         Unique: speaker self-rating (intent) + 3rd-party ratings.

  MultiPICo English  (Casola et al., ACL 2024)
         HuggingFace: Multilingual-Perspectivist-NLU/MultiPICo  (English subset)
         2,999 items / 74 annotators / binary irony label.
         Has `post` (context) + `reply` columns.

Delta definitions
-----------------
  delta_ctx  |VADER(text) - (P(pos)-P(neg))_RoBERTa(text)| / 2     [PRIMARY, label-free]
  delta_lex  |VADER(response) - VADER(context)| / 2                 [context mismatch, label-free]
  delta_self (CSC only) |VADER(response)| × self_sarcasm_norm       [uses speaker self-rating, NOT 3rd-party]

go/no-go criterion
------------------
  GO  iff  ρ ≥ 0.30 AND bootstrap 95% CI lower > 0.10
           for at least 1 (dataset × delta_def) combination.
  NO-GO   if ALL ρ < 0.10.
  MARGINAL otherwise.

Collision avoidance
-------------------
  Does NOT touch: run_delta_ctx.py, data/processed/goemotions_processed.parquet,
                  data/processed/mhs_processed.parquet,
                  results/pilot_deltactx_*, results/pilot_gonogo_*.
  New outputs:    results/sarcasm/*, data/raw/csc_data_full.csv,
                  data/raw/multipico_en.parquet.

GPU / API policy
----------------
  CUDA_VISIBLE_DEVICES="" set BEFORE any torch import — CPU only.
  No frontier-LLM API calls (local RoBERTa CPU only).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

# CRITICAL: set before any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pilot_gonogo/

from pilot_gonogo.data_loader_sarcasm import (
    aggregate_csc,
    aggregate_multipico,
    compute_delta_ctx,
    compute_delta_lex_csc,
    compute_delta_lex_csc_self,
    compute_delta_lex_multipico,
    load_csc_raw,
    load_multipico_en_raw,
)

RESULTS_DIR = BASE / "results" / "sarcasm"
FIGURES_DIR = RESULTS_DIR / "figures"
SEED = 42
N_BOOTSTRAP = 10_000
CTX_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Max items for delta_ctx (RoBERTa on CPU; all items if smaller)
MAX_CTX_SAMPLE = 7000

RHO_GO_THRESHOLD = 0.30
CI_LO_THRESHOLD = 0.10


# ── Statistics ────────────────────────────────────────────────────────────────


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float]:
    """Spearman ρ and 95% bootstrap CI (pairwise NaN-dropped)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    rho, pval = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = spearmanr(x[idx], y[idx]).statistic
    return {
        "rho": float(rho),
        "pvalue": float(pval),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n": n,
    }


def partial_spearman_len(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Spearman partial correlation of x, y controlling for z (rank-OLS residuals)."""
    from scipy.stats import rankdata

    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    rx, ry, rz = rankdata(x[mask]), rankdata(y[mask]), rankdata(z[mask])

    def _resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        B = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef

    return float(spearmanr(_resid(rx, rz), _resid(ry, rz)).statistic)


def gonogo_verdict(results: list[dict]) -> str:
    """GO / MARGINAL / NO-GO based on all results."""
    any_go = any(
        r["rho"] >= RHO_GO_THRESHOLD and r["ci_lo"] > CI_LO_THRESHOLD
        for r in results
    )
    all_low = all(r["rho"] < 0.10 for r in results)
    if any_go:
        return "GO"
    elif all_low:
        return "NO-GO"
    return "MARGINAL"


# ── Figures ───────────────────────────────────────────────────────────────────

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "csc_ctx": "#0072B2",
    "csc_lex": "#E69F00",
    "csc_self": "#009E73",
    "multi_ctx": "#CC79A7",
    "multi_lex": "#56B4E9",
    "neutral": "#999999",
}


def _scatter_fig(
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    ci_lo: float,
    ci_hi: float,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, alpha=0.15, s=6, color=color, rasterized=True)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"{title}\nρ={rho:+.3f}  95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}]  n={len(x)}",
        fontsize=10,
    )
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {save_path}")


def _quartile_fig(
    df: pd.DataFrame,
    delta_col: str,
    disagree_col: str,
    color: str,
    title: str,
    save_path: Path,
) -> None:
    sub = df[[delta_col, disagree_col]].dropna().copy()
    # Use duplicates="drop" to handle many ties (e.g., many zeros in delta_self).
    # Compute bin edges first, then assign labels matching actual number of bins.
    q_result, bin_edges = pd.qcut(
        sub[delta_col], q=4, retbins=True, duplicates="drop"
    )
    n_bins = int(q_result.cat.codes.max() + 1)  # actual number of non-duplicate bins
    bin_labels = [f"Q{i+1}" for i in range(n_bins)]
    sub["Q"] = pd.Categorical.from_codes(q_result.cat.codes, categories=bin_labels)
    grp = (
        sub.groupby("Q", observed=False)[disagree_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        grp["Q"].astype(str),
        grp["mean"],
        yerr=grp["std"] / np.sqrt(grp["count"].astype(float)),
        color=color,
        alpha=0.8,
        capsize=4,
    )
    ax.set_xlabel("Delta quartile", fontsize=11)
    ax.set_ylabel("Mean annotator disagreement (±SE)", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {save_path}")


def _rho_summary_fig(results: list[dict], save_path: Path) -> None:
    """Heatmap-like bar chart of ρ across all (dataset, delta_def)."""
    labels = [f"{r['dataset']}\n{r['delta_def']}" for r in results]
    rhos = [r["rho"] for r in results]
    ci_lo = [r["ci_lo"] for r in results]
    ci_hi = [r["ci_hi"] for r in results]
    colors_bar = [
        "#0072B2"
        if (r["rho"] >= RHO_GO_THRESHOLD and r["ci_lo"] > CI_LO_THRESHOLD)
        else "#E69F00"
        if r["rho"] >= 0.10
        else "#CC79A7"
        for r in results
    ]
    err_lo = [r - lo for r, lo in zip(rhos, ci_lo)]
    err_hi = [hi - r for hi, r in zip(ci_hi, rhos)]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.2), 4))
    x = np.arange(len(results))
    ax.bar(x, rhos, color=colors_bar, alpha=0.85)
    ax.errorbar(
        x, rhos, yerr=[err_lo, err_hi], fmt="none", color="black", capsize=4
    )
    ax.axhline(RHO_GO_THRESHOLD, color="green", linestyle="--", lw=1.2, label=f"GO threshold (ρ={RHO_GO_THRESHOLD})")
    ax.axhline(CI_LO_THRESHOLD, color="orange", linestyle=":", lw=1.0, label=f"CI threshold ({CI_LO_THRESHOLD})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title("All (dataset × δ definition) — Sarcasm Pilot", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {save_path}")


def _self_3rd_fig(df_csc: pd.DataFrame, save_path: Path) -> None:
    """CSC-specific: scatter of delta_self_3rd vs disagreement."""
    x = df_csc["delta_self_3rd"].values
    y = df_csc["disagreement"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    rho_val = float(spearmanr(x[mask], y[mask]).statistic)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, alpha=0.2, s=6, color="#009E73", rasterized=True)
    ax.set_xlabel("|self_sarcasm_norm - mean_3rd_norm|", fontsize=10)
    ax.set_ylabel("Evaluator disagreement (var)", fontsize=10)
    ax.set_title(
        f"CSC: self vs 3rd-party gap × evaluator disagreement\nρ={rho_val:+.3f}  n={mask.sum()}",
        fontsize=10,
    )
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {save_path}")


# ── Environment log ───────────────────────────────────────────────────────────


def build_env_log(data_hashes: dict, git_commit: str = "unknown") -> dict:
    """Build reproducibility manifest."""
    import torch

    try:
        import transformers
        tf_ver = transformers.__version__
    except Exception:
        tf_ver = "N/A"

    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": tf_ver,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "ctx_model": CTX_MODEL,
        "max_ctx_sample": MAX_CTX_SAMPLE,
        "data_hashes": data_hashes,
        "git_commit": git_commit,
        "rho_go_threshold": RHO_GO_THRESHOLD,
        "ci_lo_threshold": CI_LO_THRESHOLD,
    }


# ── Analysis runner ───────────────────────────────────────────────────────────


def run_correlation(
    dataset: str,
    delta_def: str,
    delta_vals: np.ndarray,
    disagree_vals: np.ndarray,
    text_len: np.ndarray,
    df_plot: pd.DataFrame,
    delta_col: str,
    disagree_col: str,
    color: str,
) -> dict:
    """Run one (dataset × delta_def) correlation analysis."""
    print(f"\n[{dataset}] delta_def={delta_def} ...")
    stats = bootstrap_spearman(delta_vals, disagree_vals)
    partial = partial_spearman_len(delta_vals, disagree_vals, text_len)
    print(
        f"  rho={stats['rho']:+.4f} CI=[{stats['ci_lo']:+.4f},{stats['ci_hi']:+.4f}] "
        f"p={stats['pvalue']:.2e} partial_rho(len)={partial:+.4f} n={stats['n']}"
    )
    # Scatter plot
    mask = ~(np.isnan(delta_vals) | np.isnan(disagree_vals))
    _scatter_fig(
        delta_vals[mask],
        disagree_vals[mask],
        stats["rho"],
        stats["ci_lo"],
        stats["ci_hi"],
        xlabel=f"δ ({delta_def})",
        ylabel="Annotator disagreement",
        title=f"{dataset}",
        color=color,
        save_path=FIGURES_DIR / f"scatter_{dataset.lower()}_{delta_def}.png",
    )
    # Quartile plot
    _quartile_fig(
        df_plot, delta_col, disagree_col,
        color=color,
        title=f"{dataset} — quartile monotonicity ({delta_def})",
        save_path=FIGURES_DIR / f"quartile_{dataset.lower()}_{delta_def}.png",
    )
    return {
        "dataset": dataset,
        "delta_def": delta_def,
        "n": stats["n"],
        "rho": stats["rho"],
        "pvalue": stats["pvalue"],
        "ci_lo": stats["ci_lo"],
        "ci_hi": stats["ci_hi"],
        "partial_rho_len": partial,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main(dry_run: bool = False) -> None:
    """Run sarcasm pilot.

    Parameters
    ----------
    dry_run : bool
        If True, skip RoBERTa (delta_ctx) computation and report only delta_lex.
        Useful to verify data loading and delta_lex pipeline end-to-end quickly.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Sarcasm Pilot — go/no-go (δ × annotator disagreement)")
    print(f"dry_run={dry_run}  seed={SEED}  n_bootstrap={N_BOOTSTRAP}")
    print("=" * 70)

    data_hashes: dict[str, str] = {}
    results: list[dict] = []

    # ── CSC ────────────────────────────────────────────────────────────────────
    print("\n--- CSC: Conversational Sarcasm Corpus ---")
    df_csc_raw, hash_csc = load_csc_raw()
    data_hashes["CSC"] = hash_csc
    df_csc = aggregate_csc(df_csc_raw)

    # delta_lex (context mismatch, label-free)
    print("[CSC] Computing delta_lex (context-sentiment mismatch) ...")
    df_csc["delta_lex"] = compute_delta_lex_csc(df_csc)

    # delta_self (self-rating × |VADER|, uses speaker intent NOT 3rd-party)
    print("[CSC] Computing delta_self (|VADER| × self_sarcasm_norm) ...")
    df_csc["delta_self"] = compute_delta_lex_csc_self(df_csc)

    # delta_ctx (label-free RoBERTa vs VADER; skip in dry_run)
    if not dry_run:
        print("[CSC] Computing delta_ctx (VADER vs RoBERTa, CPU only) ...")
        # Sample if needed (CPU budget)
        n_csc = min(MAX_CTX_SAMPLE, len(df_csc))
        rng = np.random.default_rng(SEED)
        idx_csc = rng.choice(len(df_csc), size=n_csc, replace=False)
        df_csc_sub = df_csc.iloc[idx_csc].copy()
        ctx_vals = compute_delta_ctx(df_csc_sub["response_text"].tolist(), model_name=CTX_MODEL)
        df_csc["delta_ctx"] = np.nan
        df_csc.loc[df_csc_sub.index, "delta_ctx"] = ctx_vals
    else:
        print("[CSC] dry_run: skipping delta_ctx (RoBERTa)")
        df_csc["delta_ctx"] = np.nan

    # Disagreement column
    df_csc_dropna = df_csc.dropna(subset=["disagreement", "text_length"]).copy()

    # Correlation analyses
    for delta_col, color_key in [
        ("delta_lex", "csc_lex"),
        ("delta_self", "csc_self"),
        ("delta_ctx", "csc_ctx"),
    ]:
        if dry_run and delta_col == "delta_ctx":
            continue
        sub = df_csc_dropna.dropna(subset=[delta_col]).copy()
        if len(sub) < 20:
            print(f"[CSC] {delta_col}: too few non-NaN items ({len(sub)}), skipping.")
            continue
        r = run_correlation(
            dataset="CSC",
            delta_def=delta_col,
            delta_vals=sub[delta_col].values,
            disagree_vals=sub["disagreement"].values,
            text_len=sub["text_length"].values,
            df_plot=sub,
            delta_col=delta_col,
            disagree_col="disagreement",
            color=COLORS[color_key],
        )
        results.append(r)

    # CSC special: self vs 3rd-party gap × evaluator disagreement
    print("\n[CSC] Computing self vs 3rd-party gap × evaluator disagreement ...")
    mask_self3rd = ~(df_csc["delta_self_3rd"].isna() | df_csc["disagreement"].isna())
    self3rd_sub = df_csc[mask_self3rd]
    stats_self3rd = bootstrap_spearman(
        self3rd_sub["delta_self_3rd"].values,
        self3rd_sub["disagreement"].values,
    )
    partial_self3rd = partial_spearman_len(
        self3rd_sub["delta_self_3rd"].values,
        self3rd_sub["disagreement"].values,
        self3rd_sub["text_length"].values,
    )
    print(
        f"  [CSC self-3rd] rho={stats_self3rd['rho']:+.4f} "
        f"CI=[{stats_self3rd['ci_lo']:+.4f},{stats_self3rd['ci_hi']:+.4f}] "
        f"partial={partial_self3rd:+.4f} n={stats_self3rd['n']}"
    )
    _self_3rd_fig(df_csc, FIGURES_DIR / "csc_self_vs_3rd_disagreement.png")

    # Save CSC aggregated parquet (input to the KBS pipeline)
    csc_out_path = RESULTS_DIR / "csc_aggregated.parquet"
    df_csc.to_parquet(csc_out_path, index=False)
    print(f"[CSC] Aggregated data saved: {csc_out_path}")

    # ── MultiPICo English ──────────────────────────────────────────────────────
    print("\n--- MultiPICo English (Casola et al., ACL 2024) ---")
    df_multi_raw, hash_multi = load_multipico_en_raw()
    data_hashes["MultiPICo_EN"] = hash_multi
    df_multi = aggregate_multipico(df_multi_raw)

    # delta_lex (context mismatch)
    print("[MultiPICo] Computing delta_lex ...")
    df_multi["delta_lex"] = compute_delta_lex_multipico(df_multi)

    # delta_ctx
    if not dry_run:
        print("[MultiPICo] Computing delta_ctx (CPU only) ...")
        n_multi = len(df_multi)
        ctx_vals_multi = compute_delta_ctx(
            df_multi["reply"].tolist(), model_name=CTX_MODEL
        )
        df_multi["delta_ctx"] = ctx_vals_multi
    else:
        print("[MultiPICo] dry_run: skipping delta_ctx")
        df_multi["delta_ctx"] = np.nan

    df_multi_dropna = df_multi.dropna(subset=["disagreement", "text_length"]).copy()

    for delta_col, color_key in [("delta_lex", "multi_lex"), ("delta_ctx", "multi_ctx")]:
        if dry_run and delta_col == "delta_ctx":
            continue
        sub = df_multi_dropna.dropna(subset=[delta_col]).copy()
        if len(sub) < 20:
            print(f"[MultiPICo] {delta_col}: too few items, skipping.")
            continue
        r = run_correlation(
            dataset="MultiPICo_EN",
            delta_def=delta_col,
            delta_vals=sub[delta_col].values,
            disagree_vals=sub["disagreement"].values,
            text_len=sub["text_length"].values,
            df_plot=sub,
            delta_col=delta_col,
            disagree_col="disagreement",
            color=COLORS[color_key],
        )
        results.append(r)

    # Save MultiPICo aggregated
    multi_out_path = RESULTS_DIR / "multipico_aggregated.parquet"
    df_multi.to_parquet(multi_out_path, index=False)
    print(f"[MultiPICo] Aggregated data saved: {multi_out_path}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    verdict = gonogo_verdict(results)
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    # ── Summary figure ─────────────────────────────────────────────────────────
    if results:
        _rho_summary_fig(results, FIGURES_DIR / "rho_summary.png")

    # ── Markdown summary ───────────────────────────────────────────────────────
    fig_paths = sorted(FIGURES_DIR.glob("*.png"))

    lines = [
        "# Sarcasm Pilot — go/no-go Summary",
        "",
        f"**Generated:** {datetime.datetime.utcnow().isoformat()}Z  ",
        f"**GPU:** CPU-only (CUDA_VISIBLE_DEVICES=\"\")  ",
        f"**Frontier-LLM API:** None  ",
        f"**RoBERTa model:** {CTX_MODEL}  ",
        f"**dry_run:** {dry_run}  ",
        "",
        f"## VERDICT: **{verdict}**",
        "",
        f"GO criterion: Spearman ρ ≥ {RHO_GO_THRESHOLD} AND 95% CI lower > {CI_LO_THRESHOLD}"
        " for ≥1 (dataset × delta_def).",
        "NO-GO: ALL ρ < 0.10.",
        "",
        "---",
        "",
        "## Full Results Table (all dataset × delta_def — cherry-picking prohibited)",
        "",
        "| Dataset | delta_def | n | Spearman ρ | 95% CI | p-value | Partial ρ (len) |",
        "|:--------|:----------|--:|----------:|:-------|--------:|----------------:|",
    ]
    for r in results:
        go_mark = " **(GO)**" if (r["rho"] >= RHO_GO_THRESHOLD and r["ci_lo"] > CI_LO_THRESHOLD) else ""
        lines.append(
            f"| {r['dataset']} | {r['delta_def']} | {r['n']:,} | "
            f"{r['rho']:+.4f}{go_mark} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{r['pvalue']:.2e} | "
            f"{r['partial_rho_len']:+.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## CSC: Self vs 3rd-party gap (bonus indicator)",
        "",
        "| Metric | Value |",
        "|:-------|------:|",
        f"| delta_self_3rd × disagreement: Spearman ρ | {stats_self3rd['rho']:+.4f} |",
        f"| 95% CI | [{stats_self3rd['ci_lo']:+.4f}, {stats_self3rd['ci_hi']:+.4f}] |",
        f"| partial ρ (controlling text length) | {partial_self3rd:+.4f} |",
        f"| n | {stats_self3rd['n']:,} |",
        "",
        "Interpretation: when speakers rate their response as more sarcastic than",
        "3rd-party evaluators perceive it (larger self-3rd gap), do evaluators also",
        "disagree more among themselves?",
        "",
        "---",
        "",
        "## Dataset Notes",
        "",
        "- **CSC**: 7,036 items (context × speaker pairs). Self-rating = speaker's sarcasm",
        "  INTENT (1-6). Evaluator ratings = 3rd-party perception (4-6 raters/item).",
        "  disagreement = variance of evaluator ratings per item.",
        "- **MultiPICo English**: 2,999 items. Binary label (0=not ironic, 1=ironic).",
        "  disagreement = normalized binary entropy of annotator label distribution.",
        "- **EPIC** dropped: 88% text overlap with MultiPICo EN; MultiPICo preferred for",
        "  `post` (context) column needed for delta_lex.",
        "",
        "---",
        "",
        "## Delta Definitions",
        "",
        "| delta_def | Formula | Labels used? | Datasets |",
        "|:----------|:--------|:------------:|:---------|",
        "| delta_ctx | \\|VADER(text) - (P(pos)-P(neg))_RoBERTa(text)\\| / 2 | No | all |",
        "| delta_lex | \\|VADER(response) - VADER(context)\\| / 2 | No | all |",
        "| delta_self | \\|VADER(response)\\| × self_sarcasm_norm | Self-rating only | CSC |",
        "",
        "Note: delta_self uses speaker's SELF-rating (NOT the 3rd-party labels used for",
        "disagreement), so there is no label contamination of the disagreement target.",
        "",
        "---",
        "",
        "## Figures",
        "",
    ]
    for fp in fig_paths:
        lines.append(f"- `{fp}`")

    lines += [
        "",
        "---",
        "",
        "## Data Hashes (reproducibility)",
        "",
        "```json",
        json.dumps(data_hashes, indent=2),
        "```",
        "",
        "---",
        "",
        "## Environment Log",
        "",
        f"env_log path: `{RESULTS_DIR / 'env_log_sarcasm.json'}`",
        "",
        "---",
        "",
        "## Blockers / Notes",
        "",
        "- GPU: CPU-only per design spec. All RoBERTa runs on CPU.",
        "- No frontier-LLM API calls.",
        "- EPIC: dropped (see Dataset Notes).",
    ]

    summary_path = RESULTS_DIR / "pilot_sarcasm_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[summary] Written: {summary_path}")

    # ── JSON results ────────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "pilot_sarcasm_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "verdict": verdict,
                "results": results,
                "csc_self_3rd": {
                    "rho": stats_self3rd["rho"],
                    "ci_lo": stats_self3rd["ci_lo"],
                    "ci_hi": stats_self3rd["ci_hi"],
                    "pvalue": stats_self3rd["pvalue"],
                    "partial_rho_len": partial_self3rd,
                    "n": stats_self3rd["n"],
                },
                "data_hashes": data_hashes,
                "dry_run": dry_run,
            },
            f,
            indent=2,
        )
    print(f"[results] Written: {json_path}")

    # ── Environment log ─────────────────────────────────────────────────────────
    env_log = build_env_log(data_hashes)
    env_path = RESULTS_DIR / "env_log_sarcasm.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_log, f, indent=2)
    print(f"[env_log] Written: {env_path}")

    print("\n[run_pilot_sarcasm] DONE")
    print(f"  Verdict: {verdict}")
    for r in results:
        print(
            f"  {r['dataset']} × {r['delta_def']}: ρ={r['rho']:+.4f} "
            f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] n={r['n']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sarcasm pilot: δ × annotator disagreement go/no-go."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip RoBERTa (delta_ctx) computation; verify data loading and delta_lex only.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
