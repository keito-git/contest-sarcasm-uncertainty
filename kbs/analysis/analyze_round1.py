"""
analyze_round1.py — extended analyses C1–C9 on the regenerated sweep.

All results written to:
  results/revision_gpu_2026-09/round1_analysis/

Output files (all new, never overwrites existing):
  C1_DASYM_CI.json          — ΔAsym with 95% percentile bootstrap CIs (B=4000)
  C2_SOFTLABEL.json         — Soft-label scoring (Brier/JSD) + noise null simulation
  C3_ANNOTATOR_COUNTS.json  — Actual k_i per language from HF parquet + numbers.tex comparison
  C4_INSTABILITY.json       — Stratum instability + ΔAsym with k_i≥4/≥5 filter
  C5_ENDPOINT.json          — ΔAsym for multiple endpoint pairs + slope regression
  C6_PROBE_14B.json         — |p_14b - p_0.5b| AUROC variant (C6)
  C7_COSTACC_RANDOM.json    — Cost-accuracy: random(≥100 seeds), 3B-conf, 1-dis oracle
  C8_MEAN_DEP.json          — Spearman ρ(dis_mi, mean_p_yes) for CSC with CI
  C9_COMPARISONS.json       — Closest-method comparisons (ensemble variance, calib conf, grouping loss)
  ENV_ROUND1.json           — Run provenance

Constraints:
  - CPU only, no GPU, no API
  - B=4000 for all bootstrap CIs (unless otherwise stated)
  - SEED=42 throughout
  - All bootstrap counts stated explicitly in each JSON
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

# ── Paths ────────────────────────────────────────────────────────────────── #
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR
REGEN_DIR = BASE / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
FULL_INPUT_DIR = BASE / "results" / "revision_gpu_2026-09"
OUT_DIR = BASE / "results" / "revision_gpu_2026-09" / "round1_analysis"
PARQUET_PATH = BASE / "results" / "revision_gpu_2026-09" / "multipico_hf_raw.parquet"
NUMBERS_TEX = BASE / "paper_out" / "numbers.tex"  # optional; only used for a side-by-side comparison

SEED = 42
B = 4000  # bootstrap resamples (stated explicitly in each JSON)

C7_TAGS_CAPS: dict[str, float] = {
    "q0p5b": 0.5,
    "q1p5b": 1.5,
    "q3b":   3.0,
    "q7b":   7.0,
    "q14b": 14.0,
    "q32b": 32.0,
}

DEFAULT_CORPORA = [
    "CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"
]

EXPECTED_ROWS: dict[str, int] = {
    "CSC":          7036,
    "EPIC":         2999,
    "MultiPICo_es": 4681,
    "MultiPICo_de": 2370,
    "MultiPICo_fr": 1760,
    "MultiPICo_it":  998,
}


# ── Utilities ────────────────────────────────────────────────────────────── #
def binary_entropy(p: float) -> float:
    p = max(1e-12, min(1 - 1e-12, float(p)))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def tercile_labels(a: np.ndarray) -> np.ndarray:
    """0=low, 1=mid, 2=high disagreement (rank-based)."""
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append(float((pred[m] == c).mean()))
    return float(np.mean(recs)) if recs else float("nan")


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    pos, neg = score[y == 1], score[y == 0]
    r = rankdata(np.concatenate([pos, neg]))
    return float(r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg)
    )


def boot_ci(stat_fn, data: tuple, nb: int = B, seed: int = SEED) -> tuple[float, float, float]:
    """Generic item-level bootstrap CI. data = (arr1, arr2, ...) same length."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(d) for d in data]
    n = len(arrs[0])
    obs = stat_fn(*arrs)
    bs = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        v = stat_fn(*[a[idx] for a in arrs])
        if not math.isnan(v):
            bs.append(v)
    if not bs:
        return obs, float("nan"), float("nan")
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_auroc_ci(y: np.ndarray, score: np.ndarray, nb: int = B) -> tuple[float, float, float]:
    def fn(ya, sa):
        return auroc(ya, sa)
    return boot_ci(fn, (y, score), nb=nb)


def das_ym_fn(pl: np.ndarray, ps: np.ndarray, y: np.ndarray, terc: np.ndarray) -> float:
    """ΔAsym = Δbal_acc(low) − Δbal_acc(high)."""
    dal = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
    dah = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])
    return dal - dah


def holm_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    corrected = np.array(p_values, float)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank))
    for i in range(1, n):
        corrected[order[i]] = max(corrected[order[i]], corrected[order[i - 1]])
    return corrected.tolist()


# ── Data loading ─────────────────────────────────────────────────────────── #
def load_c7(tag: str, corpus: str) -> Optional[pd.DataFrame]:
    path = REGEN_DIR / f"c7_{tag}_{corpus}.csv"
    need = {"item_id", "p_yes", "n_valid", "H", "dis_mi", "y_true"}
    if path.exists():
        df = pd.read_csv(path)
        if need.issubset(df.columns):
            return df[list(need)].copy()
    return None


def load_c8_diversity(tag: str, corpus: str, mode: str = "c8") -> Optional[pd.DataFrame]:
    path = REGEN_DIR / f"{mode}_{tag}_{corpus}.csv"
    need = {"item_id", "p_yes"}
    if path.exists():
        df = pd.read_csv(path)
        if need.issubset(df.columns):
            return df[["item_id", "p_yes"]].copy()
    return None


def load_full_input(corpus: str) -> Optional[pd.DataFrame]:
    # Try REGEN_DIR first, then FULL_INPUT_DIR
    for d in [REGEN_DIR, FULL_INPUT_DIR]:
        p = d / f"full_input_{corpus}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


def load_all_sizes(corpus: str) -> Optional[pd.DataFrame]:
    """Load all 6 Qwen size predictions merged on item_id."""
    dfs = {}
    for tag in C7_TAGS_CAPS:
        df = load_c7(tag, corpus)
        if df is not None:
            dfs[tag] = df.rename(columns={"p_yes": f"p_{tag}"})
    if len(dfs) < 2:
        return None
    tags_sorted = sorted(dfs.keys(), key=lambda t: C7_TAGS_CAPS[t])
    base_cols = ["item_id", "dis_mi", "y_true"]
    merged = dfs[tags_sorted[0]][base_cols + [f"p_{tags_sorted[0]}"]].copy()
    for t in tags_sorted[1:]:
        merged = merged.merge(
            dfs[t][["item_id", f"p_{t}"]],
            on="item_id", how="inner",
        )
    return merged


# ── MultiPICo k_i and p_human from parquet ───────────────────────────────── #
def compute_multipico_human_stats() -> dict:
    """Compute per-item k_i and p_human for each MultiPICo language from parquet."""
    if not PARQUET_PATH.exists():
        return {"available": False, "reason": "parquet not found"}
    try:
        df_par = pd.read_parquet(str(PARQUET_PATH))
    except Exception as e:
        return {"available": False, "reason": str(e)}

    result: dict = {"available": True, "source": str(PARQUET_PATH)}
    per_lang: dict = {}

    for lang in ["es", "de", "fr", "it"]:
        df_fi = load_full_input(f"MultiPICo_{lang}")
        df_lang_par = df_par[df_par["language"] == lang].copy()

        if df_fi is None or len(df_lang_par) == 0:
            per_lang[lang] = {"available": False}
            continue

        # Per-item stats from parquet
        per_item = df_lang_par.groupby("reply_id").agg(
            k_i=("label", "count"),
            p_human=("label", "mean"),
        ).reset_index()
        per_item["reply_id"] = per_item["reply_id"].astype(str)

        # Match to full_input by item_id
        df_fi = df_fi.copy()
        df_fi["reply_id"] = df_fi["item_id"].str.replace(f"mp_{lang}_", "", regex=False)
        merged = df_fi.merge(per_item, on="reply_id", how="inner")

        ki_arr = merged["k_i"].to_numpy()
        ph_arr = merged["p_human"].to_numpy()
        n_matched = len(merged)

        from collections import Counter
        ki_vc = Counter(ki_arr.tolist())
        frac_ge4 = float((ki_arr >= 4).mean())
        frac_ge5 = float((ki_arr >= 5).mean())

        per_lang[lang] = {
            "n_matched": n_matched,
            "n_fi": len(df_fi),
            "ki_min": int(ki_arr.min()),
            "ki_max": int(ki_arr.max()),
            "ki_median": float(np.median(ki_arr)),
            "ki_mean": round(float(ki_arr.mean()), 4),
            "ki_std": round(float(ki_arr.std()), 4),
            "ki_value_counts": {str(k): int(v) for k, v in sorted(ki_vc.items())},
            "frac_k_ge4": round(frac_ge4, 4),
            "frac_k_ge5": round(frac_ge5, 4),
            "p_human_mean": round(float(ph_arr.mean()), 4),
            "p_human_std": round(float(ph_arr.std()), 4),
            "merged_item_ids": merged["item_id"].tolist(),  # for later use
            "ki_arr": ki_arr.tolist(),
            "p_human_arr": ph_arr.tolist(),
            "dis_mi_arr": merged["dis_mi"].tolist(),
            "y_true_arr": merged["y_true"].tolist(),
        }

    result["per_lang"] = per_lang
    return result


# ── p_human back-inference for CSC/EPIC ─────────────────────────────────── #
def infer_p_human(dis_mi: float, y_true: int, k: int = 5) -> float:
    """
    Numerically invert dis_mi = H(p)/log(k) to recover p_human.
    Subject to: p > 0.5 if y_true=1, p <= 0.5 if y_true=0.
    Uses bisection search.
    Limitation: k is assumed (not measured for CSC/EPIC).
    """
    if dis_mi <= 1e-9:
        return 1.0 if y_true == 1 else 0.0
    if k < 2:
        k = 2
    log_k = math.log(k)
    target_h = dis_mi * log_k
    # Clamp target_h to valid range
    target_h = min(target_h, math.log(2) - 1e-9)

    def h_fn(p):
        return binary_entropy(p)

    if y_true == 1:
        # p in [0.5, 1]: h decreases as p -> 1
        lo, hi = 0.5, 1.0 - 1e-9
    else:
        # p in [0, 0.5]: h decreases as p -> 0
        lo, hi = 1e-9, 0.5

    for _ in range(60):
        mid = (lo + hi) / 2
        hm = h_fn(mid)
        if y_true == 1:
            if hm > target_h:
                lo = mid
            else:
                hi = mid
        else:
            if hm > target_h:
                hi = mid
            else:
                lo = mid

    return (lo + hi) / 2


# ── C1: ΔAsym with 95% bootstrap CIs ────────────────────────────────────── #
def run_c1(corpora: list[str]) -> dict:
    """
    C1: 95% percentile bootstrap CIs for ΔAsym (B=4000) on all corpora
    including orig400 regenerated subsets for CSC/EPIC.
    """
    print("=== C1: ΔAsym bootstrap CIs ===", flush=True)
    n_corpora = len(corpora)
    # family size for Holm: n_corpora tests (full only; orig400 are secondary)
    n_tests_holm = n_corpora
    rng = np.random.default_rng(SEED)

    records: list[dict] = []
    p_values_full: list[float] = []

    for corpus in corpora:
        merged = load_all_sizes(corpus)
        if merged is None:
            print(f"  [{corpus}] no data, skip", flush=True)
            continue

        sm_tag = min(C7_TAGS_CAPS, key=lambda t: C7_TAGS_CAPS[t])  # q0p5b
        lg_tag = max(C7_TAGS_CAPS, key=lambda t: C7_TAGS_CAPS[t])  # q32b
        if f"p_{sm_tag}" not in merged.columns or f"p_{lg_tag}" not in merged.columns:
            continue

        n = len(merged)
        terc = tercile_labels(merged["dis_mi"].to_numpy())
        y = merged["y_true"].to_numpy()
        ps = (merged[f"p_{sm_tag}"].to_numpy() >= 0.5).astype(int)
        pl = (merged[f"p_{lg_tag}"].to_numpy() >= 0.5).astype(int)

        def asym_fn(pl_b, ps_b, y_b, t_b):
            return das_ym_fn(pl_b, ps_b, y_b, t_b)

        subsets: list[tuple[str, np.ndarray]] = [("full", np.ones(n, bool))]
        # orig400 subset: items with in_original_400==1 if available, else fall back to repro flag
        fi = load_full_input(corpus.split("_")[0] if "_" in corpus else corpus)
        if fi is None:
            fi = load_full_input(corpus)
        if fi is not None and "in_original_400" in fi.columns:
            orig_ids = set(fi[fi["in_original_400"] == 1]["item_id"].astype(str))
            orig_mask = merged["item_id"].astype(str).isin(orig_ids).to_numpy()
            if orig_mask.sum() >= 20:
                subsets.append(("orig400", orig_mask))

        for subset_label, mask in subsets:
            pl_m = pl[mask]
            ps_m = ps[mask]
            y_m = y[mask]
            t_m = terc[mask]
            nm = int(mask.sum())

            obs = asym_fn(pl_m, ps_m, y_m, t_m)
            bs = []
            for _ in range(B):
                idx = rng.integers(0, nm, nm)
                v = asym_fn(pl_m[idx], ps_m[idx], y_m[idx], t_m[idx])
                bs.append(v)
            bs_arr = np.array(bs)
            p_raw = float((bs_arr <= 0).mean())
            ci_lo = float(np.percentile(bs_arr, 2.5))
            ci_hi = float(np.percentile(bs_arr, 97.5))

            records.append({
                "corpus": corpus,
                "subset": subset_label,
                "n": nm,
                "sm_model": sm_tag,
                "lg_model": lg_tag,
                "sm_cap_B": C7_TAGS_CAPS[sm_tag],
                "lg_cap_B": C7_TAGS_CAPS[lg_tag],
                "asym_acc": round(obs, 4),
                "p_boot_raw": round(p_raw, 4),
                "ci_95_lo": round(ci_lo, 4),
                "ci_95_hi": round(ci_hi, 4),
                "n_bootstrap": B,
            })
            if subset_label == "full":
                p_values_full.append(p_raw)

    # Holm correction over full-subset tests only
    if p_values_full:
        corrected = holm_correct(p_values_full)
        full_recs = [r for r in records if r["subset"] == "full"]
        for rec, pc in zip(full_recs, corrected):
            rec["p_holm"] = round(pc, 4)
            rec["sig_05_holm"] = bool(pc < 0.05)

    return {
        "analysis": "C1: ΔAsym 95% percentile bootstrap CI",
        "estimand": "ΔAsym = Δbal_acc(low_dis_tercile, 0.5B→32B) − Δbal_acc(high_dis_tercile, 0.5B→32B)",
        "n_bootstrap": B,
        "bootstrap_seed": SEED,
        "resample_unit": "item",
        "holm_family_size": n_corpora,
        "holm_note": "Holm correction over full-subset tests only; orig400 are secondary disclosures",
        "corpora": corpora,
        "results": records,
    }


# ── C2: Soft-label scoring + annotator-noise null ─────────────────────────── #
def run_c2(corpora: list[str], multipico_stats: dict) -> dict:
    """
    C2a: Brier score and JSD(model, human) per tercile, 0.5B and 32B, ΔAsym analogue.
    C2b: Annotator-noise null — 1000 sims of ΔAsym under null of stratum-independent model.
    """
    print("=== C2: Soft-label scoring + noise null ===", flush=True)
    rng = np.random.default_rng(SEED)
    N_SIM = 1000

    def brier(p_model: np.ndarray, p_human: np.ndarray) -> float:
        """Mean Brier score: E[(p_model - p_human)^2]."""
        return float(np.mean((p_model - p_human) ** 2))

    def jsd(p_model: np.ndarray, p_human: np.ndarray) -> float:
        """Mean JSD between [p_model, 1-p_model] and [p_human, 1-p_human]."""
        eps = 1e-12
        pm = np.clip(p_model, eps, 1 - eps)
        ph = np.clip(p_human, eps, 1 - eps)
        m1 = (pm + ph) / 2
        m2 = (1 - pm + 1 - ph) / 2
        jsd_vals = (
            ph * np.log(ph / m1) + (1 - ph) * np.log((1 - ph) / m2) +
            pm * np.log(pm / m1) + (1 - pm) * np.log((1 - pm) / m2)
        ) / 2
        return float(np.mean(jsd_vals))

    per_corpus: dict = {}

    for corpus in corpora:
        lang = None
        if corpus.startswith("MultiPICo_"):
            lang = corpus.split("_")[1]

        # Load model predictions for 0.5B and 32B
        df_sm = load_c7("q0p5b", corpus)
        df_lg = load_c7("q32b", corpus)
        if df_sm is None or df_lg is None:
            per_corpus[corpus] = {"available": False, "reason": "model data missing"}
            continue

        merged = df_sm[["item_id", "dis_mi", "y_true"]].copy()
        merged["p_q0p5b"] = df_sm["p_yes"].to_numpy()
        merged = merged.merge(
            df_lg[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q32b"}),
            on="item_id", how="inner",
        )

        # Get p_human
        if lang and multipico_stats.get("available") and lang in multipico_stats.get("per_lang", {}):
            ls = multipico_stats["per_lang"][lang]
            if ls.get("available", True) and "merged_item_ids" in ls:
                ph_df = pd.DataFrame({
                    "item_id": ls["merged_item_ids"],
                    "p_human": ls["p_human_arr"],
                    "k_i": ls["ki_arr"],
                })
                merged = merged.merge(ph_df, on="item_id", how="inner")
                p_human_source = "parquet_exact"
            else:
                per_corpus[corpus] = {"available": False, "reason": "lang stats not available"}
                continue
        else:
            # Back-infer p_human from dis_mi for CSC/EPIC
            # Use k=5 (common assumption for crowdsourced binary annotation)
            # This is an approximation; documented as limitation
            k_assumed = 5
            merged["p_human"] = [
                infer_p_human(dis, yt, k_assumed)
                for dis, yt in zip(merged["dis_mi"].tolist(), merged["y_true"].tolist())
            ]
            merged["k_i"] = k_assumed
            p_human_source = f"back_inferred_k{k_assumed}"

        n = len(merged)
        terc = tercile_labels(merged["dis_mi"].to_numpy())
        merged["terc"] = terc

        p_human = merged["p_human"].to_numpy()
        p_q0p5b = merged["p_q0p5b"].to_numpy()
        p_q32b = merged["p_q32b"].to_numpy()
        y = merged["y_true"].to_numpy()
        ki = merged["k_i"].to_numpy()

        # C2a: Brier/JSD per tercile per model
        soft_scores: dict = {}
        for model_tag, p_model in [("q0p5b", p_q0p5b), ("q32b", p_q32b)]:
            entry: dict = {}
            for t_lbl, t_name in [(0, "low"), (1, "mid"), (2, "high")]:
                m = terc == t_lbl
                if m.sum() < 5:
                    entry[t_name] = {"n": int(m.sum()), "skipped": True}
                    continue
                br = brier(p_model[m], p_human[m])
                js = jsd(p_model[m], p_human[m])
                # Bootstrap CI for Brier
                obs_br, br_lo, br_hi = boot_ci(
                    lambda pm, ph: brier(pm, ph), (p_model[m], p_human[m]), nb=B
                )
                obs_js, js_lo, js_hi = boot_ci(
                    lambda pm, ph: jsd(pm, ph), (p_model[m], p_human[m]), nb=B
                )
                entry[t_name] = {
                    "n": int(m.sum()),
                    "brier": round(br, 4),
                    "brier_ci": [round(br_lo, 4), round(br_hi, 4)],
                    "jsd": round(js, 4),
                    "jsd_ci": [round(js_lo, 4), round(js_hi, 4)],
                }
            soft_scores[model_tag] = entry

        # ΔAsym analogue for Brier score: Δbrier(low) - Δbrier(high)
        # where Δbrier(stratum) = brier(32B, human) - brier(0.5B, human)
        asym_brier_vals = []
        for t_lbl in [0, 2]:
            m = terc == t_lbl
            if m.sum() < 5:
                asym_brier_vals.append(float("nan"))
            else:
                delta = brier(p_q32b[m], p_human[m]) - brier(p_q0p5b[m], p_human[m])
                asym_brier_vals.append(delta)
        if not any(math.isnan(v) for v in asym_brier_vals):
            dasym_brier = asym_brier_vals[0] - asym_brier_vals[1]

            def fn_brier_asym(p32, p05, ph, t):
                d0 = brier(p32[t == 0], ph[t == 0]) - brier(p05[t == 0], ph[t == 0])
                d2 = brier(p32[t == 2], ph[t == 2]) - brier(p05[t == 2], ph[t == 2])
                return d0 - d2

            bs_ba = []
            for _ in range(B):
                idx = rng.integers(0, n, n)
                v = fn_brier_asym(p_q32b[idx], p_q0p5b[idx], p_human[idx], terc[idx])
                bs_ba.append(v)
            bs_ba = np.array(bs_ba)
            p_brier_asym = float((bs_ba <= 0).mean())
            dasym_brier_ci = [round(float(np.percentile(bs_ba, 2.5)), 4),
                              round(float(np.percentile(bs_ba, 97.5)), 4)]
        else:
            dasym_brier = float("nan")
            p_brier_asym = float("nan")
            dasym_brier_ci = [float("nan"), float("nan")]

        # C2b: Annotator-noise null simulation
        # Model: y_model = 1 iff model predicts >=0.5, correctness independent of stratum
        # Null: overall accuracy (balanced) matches observed, stratum-independent
        # Observed majority label = resample ki annotations from Bernoulli(p_human)
        # Compute ΔAsym from resimulated data

        # Observed overall accuracies per model
        def overall_bal_acc(pred, y):
            return bal_acc(pred, y)

        ps_obs = (p_q0p5b >= 0.5).astype(int)
        pl_obs = (p_q32b >= 0.5).astype(int)
        acc_sm = overall_bal_acc(ps_obs, y)
        acc_lg = overall_bal_acc(pl_obs, y)

        # For null: model correct = Bernoulli(acc) independent of item
        null_dasym_dist: list[float] = []
        for _ in range(N_SIM):
            # Resample observed majority labels from annotators
            # y_null_majority ~ Bernoulli(p_human) with k_i draws, majority vote
            ki_int = np.round(ki).astype(int)
            ki_int = np.maximum(ki_int, 1)
            y_null = np.array([
                int(rng.binomial(ki_int[i], p_human[i]) > ki_int[i] / 2)
                for i in range(n)
            ])

            # Null model accuracy independent of stratum
            correct_sm = rng.binomial(1, acc_sm, n)
            correct_lg = rng.binomial(1, acc_lg, n)
            y_pred_sm = np.where(correct_sm == 1, y_null, 1 - y_null)
            y_pred_lg = np.where(correct_lg == 1, y_null, 1 - y_null)

            null_terc = tercile_labels(merged["dis_mi"].to_numpy())
            null_asym = das_ym_fn(y_pred_lg, y_pred_sm, y_null, null_terc)
            null_dasym_dist.append(null_asym)

        null_arr = np.array(null_dasym_dist)

        # Observed ΔAsym with original labels
        orig_asym = das_ym_fn(pl_obs, ps_obs, y, terc)

        # How often does null exceed observed?
        p_null_exceeds = float((null_arr >= orig_asym).mean())
        null_pct = {
            "mean": round(float(null_arr.mean()), 4),
            "std": round(float(null_arr.std()), 4),
            "p2.5": round(float(np.percentile(null_arr, 2.5)), 4),
            "p97.5": round(float(np.percentile(null_arr, 97.5)), 4),
        }

        per_corpus[corpus] = {
            "n": n,
            "p_human_source": p_human_source,
            "n_bootstrap_soft": B,
            "n_sim_null": N_SIM,
            "c2a_soft_scores": soft_scores,
            "c2a_dasym_brier": {
                "value": round(dasym_brier, 4) if not math.isnan(dasym_brier) else None,
                "ci_95": dasym_brier_ci,
                "p_boot_raw": round(p_brier_asym, 4) if not math.isnan(p_brier_asym) else None,
                "interpretation": (
                    "ΔAsym(Brier) < 0 means 32B reduces Brier more in low-dis stratum: "
                    "consistent with paper's claim"
                ),
            },
            "c2b_null": {
                "observed_dasym": round(orig_asym, 4),
                "null_dasym": null_pct,
                "p_null_exceeds_observed": round(p_null_exceeds, 4),
                "n_sims": N_SIM,
                "null_seed": SEED,
                "interpretation": (
                    "p_null_exceeds_observed is the fraction of noise-only sims "
                    "achieving ΔAsym >= observed. Small value -> noise cannot explain ΔAsym."
                ),
            },
        }

    return {
        "analysis": "C2: Reference-label reliability control",
        "c2a_description": (
            "Brier score and JSD between [p_yes, 1-p_yes] and human proportion per tercile; "
            "ΔAsym analogue with bootstrap CI"
        ),
        "c2b_description": (
            "Annotator-noise null: synthetic model with stratum-independent accuracy, "
            "majority label resampled from k_i annotators; ≥1000 sims"
        ),
        "n_bootstrap": B,
        "n_sims_null": N_SIM,
        "bootstrap_seed": SEED,
        "per_corpus": per_corpus,
    }


# ── C3: Actual annotator counts from HF parquet ───────────────────────────── #
def run_c3(multipico_stats: dict) -> dict:
    """
    C3: Actual k_i per language from HF parquet; compare with numbers.tex macros.
    """
    print("=== C3: Actual annotator counts ===", flush=True)

    # Read kiMP macros from numbers.tex
    ki_macros: dict = {}
    if NUMBERS_TEX.exists():
        for line in NUMBERS_TEX.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            for macro in ["kiMPMin", "kiMPMax", "kiMPMed", "kiMPMeanSE"]:
                if f"\\newcommand{{\\{macro}}}" in line:
                    try:
                        val_str = line.split("}{")[1].rstrip("}")
                        ki_macros[macro] = float(val_str)
                    except Exception:
                        ki_macros[macro] = line

    # Compile per-language stats
    per_lang: dict = {}
    corrections_needed: list[str] = []

    if multipico_stats.get("available"):
        for lang in ["es", "de", "fr", "it"]:
            ls = multipico_stats.get("per_lang", {}).get(lang, {})
            if not ls.get("available", True):
                per_lang[lang] = {"available": False}
                continue
            per_lang[lang] = {
                "ki_min": ls.get("ki_min"),
                "ki_max": ls.get("ki_max"),
                "ki_median": ls.get("ki_median"),
                "ki_mean": ls.get("ki_mean"),
                "ki_std": ls.get("ki_std"),
                "ki_value_counts": ls.get("ki_value_counts"),
                "frac_k_ge4": ls.get("frac_k_ge4"),
                "frac_k_ge5": ls.get("frac_k_ge5"),
                "n_items": ls.get("n_matched"),
                "source": "parquet_exact",
            }
    else:
        # Fallback: estimate from dis_mi
        for lang in ["es", "de", "fr", "it"]:
            df = load_full_input(f"MultiPICo_{lang}")
            if df is None:
                per_lang[lang] = {"available": False}
                continue
            est_k = [
                _estimate_k_from_dis(float(row["dis_mi"]), int(row["y_true"]))
                for _, row in df.iterrows()
            ]
            from collections import Counter
            vc = Counter(est_k)
            arr = np.array(est_k)
            per_lang[lang] = {
                "ki_min": int(arr.min()),
                "ki_max": int(arr.max()),
                "ki_median": float(np.median(arr)),
                "ki_mean": round(float(arr.mean()), 4),
                "ki_std": round(float(arr.std()), 4),
                "ki_value_counts": {str(k): int(v) for k, v in sorted(vc.items())},
                "frac_k_ge4": round(float((arr >= 4).mean()), 4),
                "frac_k_ge5": round(float((arr >= 5).mean()), 4),
                "n_items": len(arr),
                "source": "estimated_from_dis_mi",
                "limitation": "k_i estimated via MLE from H(p)/log(k) formula; not exact",
            }

    # Compare with numbers.tex macros
    # kiMPMed in paper applies to all 4 languages (aggregated)
    if per_lang:
        all_medians = [v.get("ki_median") for v in per_lang.values()
                       if v.get("ki_median") is not None]
        all_mins = [v.get("ki_min") for v in per_lang.values()
                    if v.get("ki_min") is not None]
        all_maxs = [v.get("ki_max") for v in per_lang.values()
                    if v.get("ki_max") is not None]

        paper_med = ki_macros.get("kiMPMed")
        paper_min = ki_macros.get("kiMPMin")
        paper_max = ki_macros.get("kiMPMax")

        if paper_med is not None and all_medians:
            agg_median = float(np.median(all_medians))
            if abs(agg_median - paper_med) > 0.5:
                corrections_needed.append(
                    f"kiMPMed: paper has {paper_med}, actual aggregate median = {agg_median:.1f}"
                )
        if paper_min is not None and all_mins:
            actual_min = int(min(all_mins))
            if actual_min != int(paper_min):
                corrections_needed.append(
                    f"kiMPMin: paper has {int(paper_min)}, actual min = {actual_min}"
                )
        if paper_max is not None and all_maxs:
            actual_max = int(max(all_maxs))
            if actual_max != int(paper_max):
                corrections_needed.append(
                    f"kiMPMax: paper has {int(paper_max)}, actual max = {actual_max}"
                )

    return {
        "analysis": "C3: Actual annotator counts for MultiPICo es/de/fr/it from HF parquet",
        "source": "multipico_hf_raw.parquet (Multilingual-Perspectivist-NLU/MultiPICo)",
        "numbers_tex_path": str(NUMBERS_TEX),
        "ki_macros_in_paper": ki_macros,
        "per_lang": per_lang,
        "corrections_needed": corrections_needed,
        "note": (
            "k_i computed from parquet by grouping per-annotation records by reply_id and language; "
            "matched to full_input by item_id (format: mp_{lang}_{reply_id})"
        ),
    }


def _estimate_k_from_dis(dis_mi: float, y_true: int) -> int:
    """Fallback: estimate k from dis_mi via brute force."""
    if dis_mi <= 1e-9:
        return 5
    best_k, best_err = 5, float("inf")
    for k in range(2, 9):
        log_k = math.log(k)
        for j in range(0, k + 1):
            p = j / k
            if p <= 0 or p >= 1:
                continue
            d = binary_entropy(p) / log_k
            if abs(d - dis_mi) < best_err:
                best_err = abs(d - dis_mi)
                best_k = k
    return best_k


# ── C4: Stratum instability + k_i≥4 filter ────────────────────────────────── #
def run_c4(corpora: list[str], multipico_stats: dict) -> dict:
    """
    C4: Fraction of items changing tercile under 500 annotator resamplings,
        and ΔAsym restricted to k_i ≥ 4 (and ≥ 5 where available).
    """
    print("=== C4: Stratum instability ===", flush=True)
    rng_inst = np.random.default_rng(SEED)
    N_INST = 500

    per_corpus: dict = {}

    for corpus in corpora:
        merged = load_all_sizes(corpus)
        if merged is None:
            per_corpus[corpus] = {"available": False}
            continue

        sm_tag = "q0p5b"
        lg_tag = "q32b"
        if f"p_{sm_tag}" not in merged.columns or f"p_{lg_tag}" not in merged.columns:
            per_corpus[corpus] = {"available": False, "reason": "models missing"}
            continue

        n = len(merged)
        dis = merged["dis_mi"].to_numpy()
        y = merged["y_true"].to_numpy()
        ps = (merged[f"p_{sm_tag}"].to_numpy() >= 0.5).astype(int)
        pl = (merged[f"p_{lg_tag}"].to_numpy() >= 0.5).astype(int)
        terc_orig = tercile_labels(dis)

        # Get p_human and k_i for resampling
        lang = corpus.split("_")[1] if corpus.startswith("MultiPICo_") else None
        if lang and multipico_stats.get("available"):
            ls = multipico_stats.get("per_lang", {}).get(lang, {})
            if ls.get("available", True) and "merged_item_ids" in ls:
                ph_df = pd.DataFrame({
                    "item_id": ls["merged_item_ids"],
                    "p_human": ls["p_human_arr"],
                    "k_i": ls["ki_arr"],
                })
                m2 = merged.merge(ph_df, on="item_id", how="left")
                p_human = m2["p_human"].fillna(0.5).to_numpy()
                ki = m2["k_i"].fillna(5).to_numpy().astype(int)
            else:
                p_human = np.array([infer_p_human(d, t, 5)
                                    for d, t in zip(dis, y)])
                ki = np.full(n, 5)
        else:
            p_human = np.array([infer_p_human(d, t, 5)
                                for d, t in zip(dis, y)])
            ki = np.full(n, 5)

        # Stratum instability: resample k_i labels, recompute dis_mi surrogate
        # Surrogate: resampled disagreement = H(j/k)/log(k)
        n_changed = 0
        for _ in range(N_INST):
            ki_int = np.maximum(ki, 1)
            j_resamp = rng_inst.binomial(ki_int, p_human)
            p_resamp = j_resamp / ki_int
            p_resamp = np.clip(p_resamp, 1e-9, 1 - 1e-9)
            dis_resamp = np.array([
                binary_entropy(float(p)) / math.log(max(int(k_), 2))
                for p, k_ in zip(p_resamp, ki_int)
            ])
            terc_resamp = tercile_labels(dis_resamp)
            changed = int((terc_resamp != terc_orig).sum())
            n_changed += changed
        frac_unstable = round(n_changed / (N_INST * n), 4)

        # ΔAsym with k_i filters
        asym_results: dict = {}
        for min_k, label in [(1, "all"), (4, "ki_ge4"), (5, "ki_ge5")]:
            if min_k == 1:
                mask = np.ones(n, bool)
            else:
                mask = ki >= min_k

            nm = int(mask.sum())
            if nm < 20:
                asym_results[label] = {"n": nm, "skipped": True}
                continue

            pl_m = pl[mask]
            ps_m = ps[mask]
            y_m = y[mask]
            t_m = terc_orig[mask]

            obs = das_ym_fn(pl_m, ps_m, y_m, t_m)
            bs = []
            rng_b = np.random.default_rng(SEED)
            for _ in range(B):
                idx = rng_b.integers(0, nm, nm)
                bs.append(das_ym_fn(pl_m[idx], ps_m[idx], y_m[idx], t_m[idx]))
            bs_arr = np.array(bs)
            asym_results[label] = {
                "n": nm,
                "asym_acc": round(obs, 4),
                "ci_95": [round(float(np.percentile(bs_arr, 2.5)), 4),
                          round(float(np.percentile(bs_arr, 97.5)), 4)],
                "p_boot_raw": round(float((bs_arr <= 0).mean()), 4),
                "n_bootstrap": B,
            }

        per_corpus[corpus] = {
            "n": n,
            "stratum_instability": {
                "n_resamplings": N_INST,
                "frac_changing_tercile": frac_unstable,
                "note": (
                    "Tercile changes under annotator resampling; "
                    "uses back-inferred p_human for CSC/EPIC"
                ),
            },
            "dasym_by_ki_filter": asym_results,
        }

    return {
        "analysis": "C4: Stratum instability + ΔAsym with k_i filter",
        "n_inst_resamplings": N_INST,
        "n_bootstrap": B,
        "bootstrap_seed": SEED,
        "per_corpus": per_corpus,
    }


# ── C5: Endpoint dependence ───────────────────────────────────────────────── #
def run_c5(corpora: list[str]) -> dict:
    """
    C5: ΔAsym for endpoint pairs (0.5B→32B, 1.5B→32B, 3B→32B, 0.5B→14B)
        and slope-based version.
    """
    print("=== C5: Endpoint dependence ===", flush=True)
    rng = np.random.default_rng(SEED)

    pairs: list[tuple[str, str]] = [
        ("q0p5b", "q32b"),
        ("q1p5b", "q32b"),
        ("q3b", "q32b"),
        ("q0p5b", "q14b"),
    ]

    per_corpus: dict = {}

    for corpus in corpora:
        merged = load_all_sizes(corpus)
        if merged is None:
            per_corpus[corpus] = {"available": False}
            continue

        n = len(merged)
        terc = tercile_labels(merged["dis_mi"].to_numpy())
        y = merged["y_true"].to_numpy()

        pair_results: dict = {}
        for sm_tag, lg_tag in pairs:
            if f"p_{sm_tag}" not in merged.columns or f"p_{lg_tag}" not in merged.columns:
                pair_results[f"{sm_tag}→{lg_tag}"] = {"available": False}
                continue

            ps = (merged[f"p_{sm_tag}"].to_numpy() >= 0.5).astype(int)
            pl = (merged[f"p_{lg_tag}"].to_numpy() >= 0.5).astype(int)

            obs = das_ym_fn(pl, ps, y, terc)
            bs = []
            for _ in range(B):
                idx = rng.integers(0, n, n)
                bs.append(das_ym_fn(pl[idx], ps[idx], y[idx], terc[idx]))
            bs_arr = np.array(bs)
            pair_results[f"{sm_tag}→{lg_tag}"] = {
                "sm_cap_B": C7_TAGS_CAPS[sm_tag],
                "lg_cap_B": C7_TAGS_CAPS[lg_tag],
                "asym_acc": round(obs, 4),
                "ci_95": [round(float(np.percentile(bs_arr, 2.5)), 4),
                          round(float(np.percentile(bs_arr, 97.5)), 4)],
                "p_boot_raw": round(float((bs_arr <= 0).mean()), 4),
                "n_bootstrap": B,
            }

        # Slope-based: regress bal_acc on log(cap) per tercile; compare slopes
        tags_sorted = sorted(
            [t for t in C7_TAGS_CAPS if f"p_{t}" in merged.columns],
            key=lambda t: C7_TAGS_CAPS[t]
        )
        if len(tags_sorted) >= 3:
            log_caps = np.array([math.log(C7_TAGS_CAPS[t]) for t in tags_sorted])
            slope_results: dict = {}
            for t_lbl, t_name in [(0, "low"), (2, "high")]:
                m = terc == t_lbl
                y_m = y[m]
                baccs = []
                for tag in tags_sorted:
                    preds = (merged.loc[m, f"p_{tag}"].to_numpy() >= 0.5).astype(int)
                    baccs.append(bal_acc(preds, y_m))
                baccs = np.array(baccs)
                valid = ~np.isnan(baccs)
                if valid.sum() >= 2:
                    x = log_caps[valid]
                    y_v = baccs[valid]
                    slope = float(np.cov(x, y_v)[0, 1] / np.var(x)) if np.var(x) > 0 else float("nan")
                    slope_results[t_name] = {
                        "n_tercile": int(m.sum()),
                        "slope_balAcc_per_logCap": round(slope, 6),
                        "bacc_by_size": {
                            t: round(float(b), 4)
                            for t, b in zip(tags_sorted, baccs.tolist())
                        },
                    }

            if "low" in slope_results and "high" in slope_results:
                slope_diff = (slope_results["low"]["slope_balAcc_per_logCap"]
                              - slope_results["high"]["slope_balAcc_per_logCap"])

                def fn_slope_diff(arr_all):
                    sd = {}
                    for t_lbl, t_name in [(0, "low"), (2, "high")]:
                        m_b = arr_all[:, -1] == t_lbl  # last col = terc
                        y_b = arr_all[m_b, -2]         # second-to-last = y
                        bacs = []
                        for ti, tag in enumerate(tags_sorted):
                            preds_b = (arr_all[m_b, ti] >= 0.5).astype(int)
                            bacs.append(bal_acc(preds_b, y_b))
                        bacs = np.array(bacs)
                        valid_b = ~np.isnan(bacs)
                        if valid_b.sum() >= 2:
                            xb = log_caps[valid_b]
                            yb = bacs[valid_b]
                            slp = float(np.cov(xb, yb)[0, 1] / np.var(xb)) if np.var(xb) > 0 else float("nan")
                        else:
                            slp = float("nan")
                        sd[t_name] = slp
                    if math.isnan(sd.get("low", float("nan"))) or math.isnan(sd.get("high", float("nan"))):
                        return float("nan")
                    return sd["low"] - sd["high"]

                # Build data array for bootstrap
                data_cols = [merged[f"p_{t}"].to_numpy() for t in tags_sorted]
                data_cols.append(y.astype(float))
                data_cols.append(terc.astype(float))
                data_arr = np.column_stack(data_cols)
                bs_slope = []
                for _ in range(B):
                    idx = rng.integers(0, n, n)
                    bs_slope.append(fn_slope_diff(data_arr[idx]))
                bs_slope_arr = np.array([v for v in bs_slope if not math.isnan(v)])

                slope_results["diff_low_minus_high"] = {
                    "value": round(slope_diff, 6),
                    "ci_95": [round(float(np.percentile(bs_slope_arr, 2.5)), 6),
                              round(float(np.percentile(bs_slope_arr, 97.5)), 6)],
                    "p_boot_raw": round(float((bs_slope_arr <= 0).mean()), 4),
                    "n_bootstrap": B,
                    "interpretation": (
                        "slope(low_dis) > slope(high_dis) "
                        "→ capacity matters more for epistemic items"
                    ),
                }
        else:
            slope_results = {"skipped": True, "reason": "too few model sizes"}

        per_corpus[corpus] = {
            "n": n,
            "pair_dasym": pair_results,
            "slope_based": slope_results,
        }

    return {
        "analysis": "C5: Endpoint dependence of ΔAsym",
        "pairs": [f"{sm}→{lg}" for sm, lg in pairs],
        "n_bootstrap": B,
        "bootstrap_seed": SEED,
        "per_corpus": per_corpus,
    }


# ── C6: 14B probe variant ─────────────────────────────────────────────────── #
def run_c6(corpora: list[str]) -> dict:
    """
    C6: |p_14b - p_0.5b| among 0.5B errors predicting 32B fixes — AUROC with CI.
    This uses a model closer to the target (32B) as middle model.
    """
    print("=== C6: 14B probe variant ===", flush=True)
    per_corpus: dict = {}

    for corpus in corpora:
        df_0p5b = load_c7("q0p5b", corpus)
        df_32b = load_c7("q32b", corpus)
        df_14b = load_c7("q14b", corpus)

        if df_0p5b is None or df_32b is None or df_14b is None:
            per_corpus[corpus] = {"available": False, "reason": "data missing"}
            continue

        base = df_0p5b[["item_id", "dis_mi", "y_true"]].copy()
        base["p_q0p5b"] = df_0p5b["p_yes"].to_numpy()
        base = base.merge(
            df_32b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q32b"}),
            on="item_id", how="inner",
        ).merge(
            df_14b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q14b"}),
            on="item_id", how="inner",
        )

        small_wrong = (base["p_q0p5b"].to_numpy() >= 0.5).astype(int) != base["y_true"].to_numpy()
        large_right = (base["p_q32b"].to_numpy() >= 0.5).astype(int) == base["y_true"].to_numpy()

        sub = base[small_wrong].reset_index(drop=True)
        y_fix = large_right[small_wrong].astype(int)

        n_all = len(base)
        n_wrong = int(small_wrong.sum())
        n_fix = int(y_fix.sum())

        if n_wrong < 20 or n_fix < 5 or (n_wrong - n_fix) < 5:
            per_corpus[corpus] = {
                "available": False,
                "n_total": n_all,
                "n_small_wrong": n_wrong,
                "n_fix": n_fix,
                "reason": "too few items",
            }
            continue

        probe_14b = (sub["p_q14b"] - sub["p_q0p5b"]).abs().to_numpy()
        # Also compute primary probe for comparison
        df_3b = load_c7("q3b", corpus)
        if df_3b is not None:
            sub_3b = sub.merge(
                df_3b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q3b"}),
                on="item_id", how="inner",
            )
            if len(sub_3b) >= 20:
                y_3b = y_fix[sub["item_id"].isin(sub_3b["item_id"]).to_numpy()]
                probe_3b = (sub_3b["p_q3b"] - sub_3b["p_q0p5b"]).abs().to_numpy()
                a_3b, lo_3b, hi_3b = boot_auroc_ci(y_3b, probe_3b, nb=B)
                primary_result = {
                    "probe": "|p_3b - p_0.5b|",
                    "AUROC": round(a_3b, 4),
                    "ci_95": [round(lo_3b, 4), round(hi_3b, 4)],
                    "n_bootstrap": B,
                }
            else:
                primary_result = {"available": False}
        else:
            primary_result = {"available": False, "reason": "q3b not found"}

        a_14b, lo_14b, hi_14b = boot_auroc_ci(y_fix, probe_14b, nb=B)

        per_corpus[corpus] = {
            "n_total": n_all,
            "n_q0p5b_wrong": n_wrong,
            "n_fix": n_fix,
            "base_rate": round(n_fix / n_wrong, 4),
            "probe_14b": {
                "probe": "|p_14b - p_0.5b|  (C6 variant: middle model closer to target)",
                "AUROC": round(a_14b, 4),
                "ci_95": [round(lo_14b, 4), round(hi_14b, 4)],
                "n_bootstrap": B,
                "warning": (
                    "14B is close to 32B; this variant uses a middle model nearly as large "
                    "as the target, which makes the probe more powerful but less informative "
                    "as a routing signal in practice"
                ),
            },
            "primary_probe_q3b": primary_result,
        }

    return {
        "analysis": "C6: Probe variant with 14B as middle model",
        "description": "|p_14b - p_0.5b| among 0.5B errors predicting 32B fixes",
        "n_bootstrap": B,
        "bootstrap_seed": SEED,
        "caution": (
            "14B model is close to 32B in capacity; higher AUROC expected but "
            "has limited practical relevance for cheap routing. Labeled as variant in paper."
        ),
        "per_corpus": per_corpus,
    }


# ── C7: Cost-accuracy cascade with random baseline ≥100 seeds ────────────── #
def run_c7(corpora: list[str]) -> dict:
    """
    C7: Cost-accuracy cascade.
    Random baseline: averaged over ≥100 seeds (mean + 95% band).
    3B-confidence routing: existing COSTACC_REGEN.json values.
    1−dis oracle: sort items by dis_mi ascending → high-dis to 3B first.
    """
    print("=== C7: Cost-accuracy cascade ===", flush=True)
    N_RAND_SEEDS = 100
    BUDGETS = np.linspace(0, 1, 21)

    def aubac_for_order(order: np.ndarray, y: np.ndarray,
                        bacc_3b: float, bacc_32b: float,
                        p_3b: np.ndarray, p_32b: np.ndarray) -> float:
        """
        Budget axis: fraction of items routed to 3B (0 → all use 32B, 1 → all use 3B).
        For each budget b, the cheapest b items (by order) go to 3B, rest to 32B.
        Performance = bal_acc of merged predictions.
        """
        n = len(y)
        sorted_idx = np.argsort(order)  # items sorted by routing priority (low = use 3B first)
        accs = []
        for budget in BUDGETS:
            n_3b = int(round(budget * n))
            use_3b = np.zeros(n, bool)
            use_3b[sorted_idx[:n_3b]] = True
            preds = np.where(use_3b, (p_3b >= 0.5).astype(int), (p_32b >= 0.5).astype(int))
            accs.append(bal_acc(preds, y))
        accs = np.array(accs)
        return float(np.trapz(accs, BUDGETS))

    per_corpus: dict = {}

    for corpus in corpora:
        df_3b_raw = load_c7("q3b", corpus)
        df_32b_raw = load_c7("q32b", corpus)
        if df_3b_raw is None or df_32b_raw is None:
            per_corpus[corpus] = {"available": False}
            continue

        base = df_3b_raw[["item_id", "dis_mi", "y_true"]].copy()
        base["p_3b"] = df_3b_raw["p_yes"].to_numpy()
        base["p_conf_3b"] = df_3b_raw["p_conf_mean"].to_numpy() if "p_conf_mean" in df_3b_raw.columns else df_3b_raw["p_yes"].to_numpy()
        base = base.merge(
            df_32b_raw[["item_id", "p_yes"]].rename(columns={"p_yes": "p_32b"}),
            on="item_id", how="inner",
        )

        n = len(base)
        y = base["y_true"].to_numpy()
        p_3b = base["p_3b"].to_numpy()
        p_32b = base["p_32b"].to_numpy()
        dis = base["dis_mi"].to_numpy()
        conf_3b = base["p_conf_3b"].to_numpy()

        # Load 0.5B for contraction probe
        df_0p5b = load_c7("q0p5b", corpus)
        if df_0p5b is not None:
            base = base.merge(
                df_0p5b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_0p5b"}),
                on="item_id", how="left",
            )
            p_0p5b = base["p_0p5b"].to_numpy()
            # Contraction probe: |p_3b - p_0.5b| — higher = route to 32B
            cont_probe = np.abs(p_3b - p_0p5b)
            # For routing: high contraction → use 32B (high capacity gap)
            # → order_contraction = -cont_probe (items with smallest contraction go to 3B first)
            order_cont = -cont_probe
        else:
            order_cont = None

        # 1-dis oracle: items with high dis_mi go to 3B first (they're aleatoric, 32B won't help)
        # items with low dis_mi go to 32B (they're epistemic, 32B can fix them)
        order_oracle = dis  # high dis = high routing priority to 3B = goes to 3B first

        # 3B confidence routing: confident items (high conf) go to 3B first
        # (if 3B is confident, no need for 32B)
        order_conf = -conf_3b  # high conf → use 3B → low order value

        # Random baseline: ≥100 seeds
        rand_aubacs = []
        for s in range(N_RAND_SEEDS):
            rng_rand = np.random.default_rng(SEED + s)
            order_rand = rng_rand.uniform(0, 1, n)
            rand_aubacs.append(aubac_for_order(order_rand, y, 0, 0, p_3b, p_32b))
        rand_arr = np.array(rand_aubacs)

        # Oracle AUBAC
        aubac_oracle = aubac_for_order(order_oracle, y, 0, 0, p_3b, p_32b)

        # Confidence routing AUBAC
        aubac_conf = aubac_for_order(order_conf, y, 0, 0, p_3b, p_32b)

        # Contraction routing AUBAC
        aubac_cont = None if order_cont is None else aubac_for_order(
            order_cont, y, 0, 0, p_3b, p_32b
        )

        # Static 3B and 32B
        bacc_3b_only = bal_acc((p_3b >= 0.5).astype(int), y)
        bacc_32b_only = bal_acc((p_32b >= 0.5).astype(int), y)

        per_corpus[corpus] = {
            "n": n,
            "n_random_seeds": N_RAND_SEEDS,
            "aubac_random_mean": round(float(rand_arr.mean()), 4),
            "aubac_random_p2.5": round(float(np.percentile(rand_arr, 2.5)), 4),
            "aubac_random_p97.5": round(float(np.percentile(rand_arr, 97.5)), 4),
            "aubac_oracle_1minus_dis": round(aubac_oracle, 4),
            "aubac_3b_conf_routing": round(aubac_conf, 4),
            "aubac_contraction_routing": round(aubac_cont, 4) if aubac_cont is not None else None,
            "aubac_static_3b_only": round(bacc_3b_only, 4),
            "aubac_static_32b_only": round(bacc_32b_only, 4),
            "budget_axis_note": (
                "budget=0 → all items use 32B; "
                "budget=1 → all items use 3B; "
                "AUBAC integrates bal_acc over budget ∈ [0,1]"
            ),
        }

    return {
        "analysis": "C7: Cost-accuracy cascade",
        "n_random_seeds": N_RAND_SEEDS,
        "random_seed_base": SEED,
        "budget_points": BUDGETS.tolist(),
        "routing_methods": [
            "random (mean ± 95% band over 100 seeds)",
            "3B confidence routing (-conf_3b score)",
            "1-dis oracle (sort by dis_mi ascending = high-dis to 3B first)",
            "contraction probe routing (-|p_3b - p_0.5b|)",
        ],
        "per_corpus": per_corpus,
    }


# ── C8: CSC mean-dependence ───────────────────────────────────────────────── #
def run_c8(corpora: list[str]) -> dict:
    """
    C8: Spearman ρ(dis_mi, mean_p_yes_across_qwen) for CSC with bootstrap CI.
    Also computes ρ(dis_mi, mean_p_yes) for other corpora for context.
    """
    print("=== C8: Mean-dependence ===", flush=True)
    per_corpus: dict = {}

    for corpus in corpora:
        merged = load_all_sizes(corpus)
        if merged is None:
            per_corpus[corpus] = {"available": False}
            continue

        tags = [t for t in C7_TAGS_CAPS if f"p_{t}" in merged.columns]
        if not tags:
            per_corpus[corpus] = {"available": False}
            continue

        n = len(merged)
        dis = merged["dis_mi"].to_numpy()

        # Mean p_yes across Qwen models
        p_cols = np.column_stack([merged[f"p_{t}"].to_numpy() for t in tags])
        mean_p_yes = p_cols.mean(axis=1)

        # Spearman rho
        rho, p_val = spearmanr(dis, mean_p_yes)
        rho = float(rho)
        p_val = float(p_val)

        # Bootstrap CI for Spearman rho
        def spear_fn(d, m):
            r, _ = spearmanr(d, m)
            return float(r)

        _, rho_lo, rho_hi = boot_ci(spear_fn, (dis, mean_p_yes), nb=B)

        # Also per-model
        per_model: dict = {}
        for t in tags:
            r_m, p_m = spearmanr(dis, merged[f"p_{t}"].to_numpy())
            per_model[t] = {
                "rho": round(float(r_m), 4),
                "p_value_parametric": round(float(p_m), 4),
            }

        per_corpus[corpus] = {
            "n": n,
            "rho_dis_vs_mean_p_yes": round(rho, 4),
            "ci_95": [round(rho_lo, 4), round(rho_hi, 4)],
            "p_value_parametric": round(p_val, 6),
            "n_bootstrap": B,
            "models_averaged": tags,
            "per_model_rho": per_model,
            "interpretation": (
                "ρ > 0: items with higher dis tend to have higher mean model probability → "
                "model probability correlates with annotation disagreement (dis is NOT mean-independent)"
            ),
        }

    return {
        "analysis": "C8: CSC mean-dependence: Spearman ρ(dis_mi, mean_p_yes_across_qwen)",
        "note": (
            "mean_p_yes = average of p_yes across all available Qwen model sizes. "
            "ρ ≠ 0 → dis measure is not mean-independent. "
            "Paper should not claim 'mean independence'."
        ),
        "n_bootstrap": B,
        "bootstrap_seed": SEED,
        "per_corpus": per_corpus,
    }


# ── C9: Comparisons with closest methods ──────────────────────────────────── #
def run_c9(corpora: list[str]) -> dict:
    """
    C9: Compare dis_mi probe with:
    (i) Multi-model variance across Qwen sizes / diversity models
    (ii) Calibrated confidence (isotonic/Platt CV on 0.5B p_yes)
    (iii) Grouping-loss-style estimator (cross-fitted calibration residual variance)
    """
    print("=== C9: Comparisons with closest methods ===", flush=True)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression

    per_corpus: dict = {}

    for corpus in corpora:
        df_0p5b = load_c7("q0p5b", corpus)
        df_32b = load_c7("q32b", corpus)
        if df_0p5b is None or df_32b is None:
            per_corpus[corpus] = {"available": False}
            continue

        base = df_0p5b[["item_id", "dis_mi", "y_true"]].copy()
        base["p_q0p5b"] = df_0p5b["p_yes"].to_numpy()
        base = base.merge(
            df_32b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q32b"}),
            on="item_id", how="inner",
        )

        # Estimand: among 0.5B wrong items, does 32B fix?
        small_wrong = (base["p_q0p5b"].to_numpy() >= 0.5).astype(int) != base["y_true"].to_numpy()
        large_right = (base["p_q32b"].to_numpy() >= 0.5).astype(int) == base["y_true"].to_numpy()

        sub = base[small_wrong].reset_index(drop=True)
        y_fix = large_right[small_wrong].astype(int)
        n_all = len(base)
        n_wrong = int(small_wrong.sum())
        n_fix = int(y_fix.sum())

        if n_wrong < 30 or n_fix < 5 or (n_wrong - n_fix) < 5:
            per_corpus[corpus] = {
                "available": False,
                "n_total": n_all, "n_wrong": n_wrong, "n_fix": n_fix,
                "reason": "too few items in estimand subset",
            }
            continue

        p_0p5b_sub = sub["p_q0p5b"].to_numpy()
        dis_sub = sub["dis_mi"].to_numpy()

        # Reference: primary probe (|p_3b - p_0.5b|)
        df_3b = load_c7("q3b", corpus)
        if df_3b is not None:
            sub_3b = sub.merge(
                df_3b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q3b"}),
                on="item_id", how="inner",
            )
            if len(sub_3b) >= 20:
                yf_3b = y_fix[sub["item_id"].isin(sub_3b["item_id"]).to_numpy()]
                probe_prim = (sub_3b["p_q3b"] - sub_3b["p_q0p5b"]).abs().to_numpy()
                auroc_prim, lo_prim, hi_prim = boot_auroc_ci(yf_3b, probe_prim, nb=B)
                primary_auroc = {
                    "AUROC": round(auroc_prim, 4),
                    "ci_95": [round(lo_prim, 4), round(hi_prim, 4)],
                    "n_bootstrap": B,
                }
            else:
                primary_auroc = {"available": False}
                yf_3b = y_fix
                probe_prim = None
        else:
            primary_auroc = {"available": False}
            yf_3b = y_fix
            probe_prim = None

        # ── (i) Multi-model variance across Qwen sizes ───────────────────── #
        qwen_dfs: dict = {}
        for tag in C7_TAGS_CAPS:
            df = load_c7(tag, corpus)
            if df is not None:
                qwen_dfs[tag] = df

        if len(qwen_dfs) >= 2:
            # Merge all sizes on the sub (0.5B-wrong) items
            tmp = sub[["item_id"]].copy()
            for tag, df in qwen_dfs.items():
                tmp = tmp.merge(
                    df[["item_id", "p_yes"]].rename(columns={"p_yes": f"p_{tag}"}),
                    on="item_id", how="inner",
                )
            if len(tmp) >= 20:
                y_var = y_fix[sub["item_id"].isin(tmp["item_id"]).to_numpy()]
                p_cols = np.column_stack([tmp[f"p_{t}"].to_numpy() for t in qwen_dfs])
                var_score = p_cols.var(axis=1)
                auroc_var, lo_var, hi_var = boot_auroc_ci(y_var, var_score, nb=B)
                qwen_var_result = {
                    "description": "Variance of p_yes across Qwen model sizes (ensemble variance)",
                    "n_models": len(qwen_dfs),
                    "model_sizes": list(qwen_dfs.keys()),
                    "n_items": len(y_var),
                    "AUROC": round(auroc_var, 4),
                    "ci_95": [round(lo_var, 4), round(hi_var, 4)],
                    "n_bootstrap": B,
                    "limitation": (
                        "Uses all 6 Qwen sizes including 32B (target), "
                        "which creates data leakage in the predictor"
                    ),
                }
            else:
                qwen_var_result = {"available": False, "reason": "too few items after merge"}
        else:
            qwen_var_result = {"available": False, "reason": "fewer than 2 Qwen sizes available"}

        # Diversity model variance
        div_models = {
            "q3b": ("c7", corpus),
            "q0p5b": ("c7", corpus),
        }
        div_dfs: dict = {}
        for tag, mode in [("q3b", "c7"), ("ll3b", "c8"), ("ll1b", "c8"), ("gm2_2b", "c8")]:
            df_d = load_c8_diversity(tag, corpus, mode)
            if df_d is not None:
                div_dfs[tag] = df_d

        if len(div_dfs) >= 2:
            tmp_div = sub[["item_id"]].copy()
            for tag, df_d in div_dfs.items():
                tmp_div = tmp_div.merge(
                    df_d.rename(columns={"p_yes": f"pd_{tag}"}),
                    on="item_id", how="inner",
                )
            if len(tmp_div) >= 20:
                y_div_var = y_fix[sub["item_id"].isin(tmp_div["item_id"]).to_numpy()]
                pd_cols = np.column_stack([tmp_div[f"pd_{t}"].to_numpy() for t in div_dfs])
                div_var_score = pd_cols.var(axis=1)
                auroc_div_var, lo_dv, hi_dv = boot_auroc_ci(y_div_var, div_var_score, nb=B)
                div_var_result = {
                    "description": "Variance across diversity models (Qwen3B + Llama3B/1B + Gemma2B)",
                    "n_models": len(div_dfs),
                    "models": list(div_dfs.keys()),
                    "n_items": len(y_div_var),
                    "AUROC": round(auroc_div_var, 4),
                    "ci_95": [round(lo_dv, 4), round(hi_dv, 4)],
                    "n_bootstrap": B,
                }
            else:
                div_var_result = {"available": False, "reason": "too few after merge"}
        else:
            div_var_result = {
                "available": False,
                "reason": f"only {len(div_dfs)} diversity models available",
                "available_models": list(div_dfs.keys()),
            }

        # ── (ii) Calibrated confidence: isotonic CV on 0.5B p_yes ────────── #
        # Use 5-fold CV; fit isotonic regression calibrator; apply to holdout
        X_cal = p_0p5b_sub.reshape(-1, 1)
        y_cal = y_fix.to_numpy() if hasattr(y_fix, "to_numpy") else np.array(y_fix)

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        calib_pred = np.full(len(y_cal), float("nan"))

        try:
            for train_idx, test_idx in kf.split(X_cal):
                X_tr, X_te = p_0p5b_sub[train_idx], p_0p5b_sub[test_idx]
                y_tr = y_cal[train_idx]
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(X_tr, y_tr)
                calib_pred[test_idx] = ir.predict(X_te)
            valid_calib = ~np.isnan(calib_pred)
            if valid_calib.sum() >= 20:
                auroc_calib, lo_calib, hi_calib = boot_auroc_ci(
                    y_cal[valid_calib], calib_pred[valid_calib], nb=B
                )
                calib_result = {
                    "method": "isotonic regression 5-fold CV on p_yes(0.5B)",
                    "n_items": int(valid_calib.sum()),
                    "AUROC": round(auroc_calib, 4),
                    "ci_95": [round(lo_calib, 4), round(hi_calib, 4)],
                    "n_bootstrap": B,
                    "limitation": (
                        "Calibrated confidence uses only the 0.5B model's output; "
                        "does not use any cross-model comparison"
                    ),
                }
            else:
                calib_result = {"available": False, "reason": "CV produced NaN predictions"}
        except Exception as e:
            calib_result = {"available": False, "reason": f"calibration failed: {e}"}

        # ── (iii) Grouping-loss-style estimator ───────────────────────────── #
        # Cross-fitted calibration residual variance within bins
        # Inspired by Melo et al. 2026's grouping-loss approach
        # Implementation: cross-fit variance of calibration residuals
        try:
            n_bins = 10
            kf2 = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)
            gl_score = np.full(len(y_cal), float("nan"))

            for train_idx, test_idx in kf2.split(X_cal):
                X_tr, X_te = p_0p5b_sub[train_idx], p_0p5b_sub[test_idx]
                y_tr = y_cal[train_idx]
                # Fit calibrator on training fold
                ir_gl = IsotonicRegression(out_of_bounds="clip")
                ir_gl.fit(X_tr, y_tr)
                p_cal_te = ir_gl.predict(X_te)
                # Calibration residual on test fold
                residuals = y_cal[test_idx] - p_cal_te
                # Bin test items by p_0.5b_te
                bins = np.quantile(X_te, np.linspace(0, 1, n_bins + 1))
                bins[0] -= 1e-9
                bin_labels = np.digitize(X_te, bins) - 1
                bin_labels = np.clip(bin_labels, 0, n_bins - 1)
                # Within-bin residual variance
                for bi in range(n_bins):
                    m_bin = bin_labels == bi
                    if m_bin.sum() >= 2:
                        gl_score[test_idx[m_bin]] = np.var(residuals[m_bin])
                    elif m_bin.sum() == 1:
                        gl_score[test_idx[m_bin]] = float(residuals[m_bin][0] ** 2)

            valid_gl = ~np.isnan(gl_score)
            if valid_gl.sum() >= 20:
                auroc_gl, lo_gl, hi_gl = boot_auroc_ci(
                    y_cal[valid_gl], gl_score[valid_gl], nb=B
                )
                gl_result = {
                    "method": (
                        "cross-fitted within-bin calibration residual variance "
                        "(grouping-loss-style; 5-fold CV, 10 bins, inspired by Melo et al. 2026)"
                    ),
                    "n_items": int(valid_gl.sum()),
                    "AUROC": round(auroc_gl, 4),
                    "ci_95": [round(lo_gl, 4), round(hi_gl, 4)],
                    "n_bootstrap": B,
                    "limitation": (
                        "Uses only 0.5B model output; no true training-data-leakage protection "
                        "beyond the cross-fitting. Not an exact implementation of Melo et al. 2026."
                    ),
                }
            else:
                gl_result = {"available": False, "reason": "too few valid items after CV"}
        except Exception as e:
            gl_result = {"available": False, "reason": f"GL estimator failed: {e}"}

        per_corpus[corpus] = {
            "n_total": n_all,
            "n_q0p5b_wrong": n_wrong,
            "n_fix": n_fix,
            "primary_probe_q3b_q0p5b": primary_auroc,
            "c9i_qwen_variance": qwen_var_result,
            "c9i_diversity_variance": div_var_result,
            "c9ii_calibrated_conf": calib_result,
            "c9iii_grouping_loss_style": gl_result,
        }

    return {
        "analysis": "C9: Comparisons with closest methods",
        "estimand": (
            "AMONG items where 0.5B is wrong: does 32B fix them? "
            "Primary probe: |p_3b - p_0.5b|"
        ),
        "methods": {
            "c9i_qwen_variance": "variance of p_yes across Qwen model sizes",
            "c9i_diversity_variance": "variance across diverse-family models (Qwen + Llama + Gemma)",
            "c9ii_calibrated_conf": "isotonic-regression calibrated 0.5B confidence (5-fold CV)",
            "c9iii_grouping_loss_style": (
                "cross-fitted within-bin calibration residual variance "
                "(grouping-loss style, inspired by Melo et al. 2026)"
            ),
        },
        "n_bootstrap": B,
        "bootstrap_seed": SEED,
        "per_corpus": per_corpus,
    }




# ── Main ─────────────────────────────────────────────────────────────────── #
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default=",".join(DEFAULT_CORPORA))
    ap.add_argument("--skip", default="", help="comma-separated analysis IDs to skip (c1,c2,...)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    print(f"Corpora: {corpora}", flush=True)
    print(f"Skip: {skip}", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)

    # Pre-compute MultiPICo parquet stats (used by C2, C3, C4)
    print("=== Loading MultiPICo parquet stats ===", flush=True)
    mp_stats = compute_multipico_human_stats()

    env: dict = {
        "script": "kbs/analysis/analyze_round1.py",
        "version": "v1 (2026-09-27)",
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "corpora": corpora,
        "n_bootstrap_all": B,
        "bootstrap_seed": SEED,
        "regen_dir": str(REGEN_DIR),
        "out_dir": str(OUT_DIR),
        "multipico_parquet_available": mp_stats.get("available", False),
    }

    def save_json(obj: dict, name: str) -> Path:
        p = OUT_DIR / name
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  → {p}", flush=True)
        return p

    # Run analyses
    c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = {}

    if "c1" not in skip:
        c1 = run_c1(corpora)
        save_json(c1, "C1_DASYM_CI.json")

    if "c2" not in skip:
        c2 = run_c2(corpora, mp_stats)
        save_json(c2, "C2_SOFTLABEL.json")

    if "c3" not in skip:
        c3 = run_c3(mp_stats)
        save_json(c3, "C3_ANNOTATOR_COUNTS.json")

    if "c4" not in skip:
        c4 = run_c4(corpora, mp_stats)
        save_json(c4, "C4_INSTABILITY.json")

    if "c5" not in skip:
        c5 = run_c5(corpora)
        save_json(c5, "C5_ENDPOINT.json")

    if "c6" not in skip:
        c6 = run_c6(corpora)
        save_json(c6, "C6_PROBE_14B.json")

    if "c7" not in skip:
        c7 = run_c7(corpora)
        save_json(c7, "C7_COSTACC_RANDOM.json")

    if "c8" not in skip:
        c8 = run_c8(corpora)
        save_json(c8, "C8_MEAN_DEP.json")

    if "c9" not in skip:
        c9 = run_c9(corpora)
        save_json(c9, "C9_COMPARISONS.json")

    # Env log
    env["utc_end"] = datetime.now(timezone.utc).isoformat()
    save_json(env, "ENV_ROUND1.json")

    print(f"\n[analyze_round1] done — {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
