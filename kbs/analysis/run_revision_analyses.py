"""
CONTEST supplementary analyses (rc3, rc4, rc6, rc9)
===================================================
CPU-only, no API calls. All numbers from saved per-item parquet/CSV files.
Output: results/revision_2026-09/*.json
"""

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
ROOT = WORK_DIR
OUT  = ROOT / "results" / "revision_2026-09"
OUT.mkdir(exist_ok=True)

SARCASM_DIR = ROOT / "results" / "sarcasm"
E1_DIR      = ROOT / "results" / "llm_e1"
E3_DIR      = ROOT / "results" / "llm_e3"
HATE_DIR    = ROOT / "results" / "llm_hate"
RAW_DIR     = ROOT / "data" / "raw"

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho with bootstrap 95% CI (1000 resamples)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    rho, pval = stats.spearmanr(x, y)
    # Bootstrap CI
    rhos = []
    for _ in range(1000):
        idx = RNG.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(r)
    rhos = np.array(rhos)
    ci_low, ci_high = float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))
    return {"rho": float(rho), "p": float(pval), "ci_95": [ci_low, ci_high], "n": int(n)}


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Partial Spearman correlation of x and y controlling for z.
    Method: rank-then-partial-pearson on residuals (standard approximation).
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # Regress rx on rz, take residuals; same for ry
    def resid(v, cov):
        beta = np.cov(v, cov)[0,1] / np.var(cov, ddof=1)
        return v - beta * cov
    rx_r = resid(rx, rz)
    ry_r = resid(ry, rz)
    rho, pval = stats.pearsonr(rx_r, ry_r)
    return {"rho": float(rho), "p": float(pval), "n": int(n)}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC; returns NaN if only one class present."""
    mask = np.isfinite(scores)
    scores, labels = scores[mask], labels[mask]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def permutation_mi(x: np.ndarray, y: np.ndarray, n_bins: int = 10,
                   n_perm: int = 1000) -> dict:
    """Bias-corrected MI via permutation baseline."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    # Discretise x into quantile bins
    qx = pd.qcut(pd.Series(x), q=n_bins, labels=False, duplicates='drop')
    if y.dtype == float:
        qy = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates='drop')
    else:
        qy = y.astype(int)
    joint = pd.crosstab(qx, qy, normalize=True).values + 1e-10
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mi_obs = float(np.sum(joint * np.log(joint / (px * py))))
    # Permutation baseline
    perm_mi = []
    qx_arr = np.array(qx)
    qy_arr = np.array(qy)
    for _ in range(n_perm):
        qx_perm = RNG.permutation(qx_arr)
        jt = pd.crosstab(qx_perm, qy_arr, normalize=True).values + 1e-10
        px_ = jt.sum(axis=1, keepdims=True)
        py_ = jt.sum(axis=0, keepdims=True)
        perm_mi.append(float(np.sum(jt * np.log(jt / (px_ * py_)))))
    perm_mi = np.array(perm_mi)
    mi_bc = mi_obs - float(perm_mi.mean())
    p_val = float(np.mean(perm_mi >= mi_obs))
    return {"mi_observed": mi_obs, "mi_bias_corrected": mi_bc,
            "permutation_mean": float(perm_mi.mean()),
            "p_permutation": p_val, "n": int(len(x))}


def isotonic_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Isotonic regression R² of y ~ f(x) where f is monotone increasing."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    ir = IsotonicRegression(increasing=True)
    y_hat = ir.fit_transform(x[order], y[order])
    ss_res = np.sum((y[order] - y_hat) ** 2)
    ss_tot = np.sum((y[order] - y[order].mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def bootstrap_delta_asym(df_low: pd.DataFrame, df_high: pd.DataFrame,
                          col_small: str, col_large: str,
                          n_boot: int = 5000, seed: int = 42) -> dict:
    """Bootstrap ΔAsym = ΔBalAcc_low - ΔBalAcc_high."""
    rng = np.random.default_rng(seed)

    def bal_acc(preds: np.ndarray, labels: np.ndarray) -> float:
        if len(preds) == 0:
            return float("nan")
        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0.5 * (tpr + tnr)

    def delta_asym(df_l, df_h):
        pred_s_l = (df_l[col_small] >= 0.5).astype(int).values
        pred_l_l = (df_l[col_large] >= 0.5).astype(int).values
        pred_s_h = (df_h[col_small] >= 0.5).astype(int).values
        pred_l_h = (df_h[col_large] >= 0.5).astype(int).values
        y_l = df_l["y_true"].values
        y_h = df_h["y_true"].values
        ba_small_low  = bal_acc(pred_s_l, y_l)
        ba_large_low  = bal_acc(pred_l_l, y_l)
        ba_small_high = bal_acc(pred_s_h, y_h)
        ba_large_high = bal_acc(pred_l_h, y_h)
        delta_low  = ba_large_low  - ba_small_low
        delta_high = ba_large_high - ba_small_high
        return delta_low - delta_high, delta_low, delta_high, ba_small_low, ba_large_low, ba_small_high, ba_large_high

    obs_da, obs_dl, obs_dh, bsl, bll, bsh, blh = delta_asym(df_low, df_high)
    boot_da = []
    for _ in range(n_boot):
        idx_l = rng.integers(0, len(df_low),  len(df_low))
        idx_h = rng.integers(0, len(df_high), len(df_high))
        da, *_ = delta_asym(df_low.iloc[idx_l], df_high.iloc[idx_h])
        boot_da.append(da)
    boot_da = np.array(boot_da)
    p_one_sided = float(np.mean(boot_da <= 0))
    ci = [float(np.percentile(boot_da, 2.5)), float(np.percentile(boot_da, 97.5))]
    return {
        "delta_asym": float(obs_da),
        "delta_low": float(obs_dl),
        "delta_high": float(obs_dh),
        "bal_acc_small_low": float(bsl),
        "bal_acc_large_low": float(bll),
        "bal_acc_small_high": float(bsh),
        "bal_acc_large_high": float(blh),
        "boot_p_le0": p_one_sided,
        "ci_95": ci,
        "n_boot": n_boot,
    }


# ===========================================================================
# ENVIRONMENT LOG
# ===========================================================================

def make_env_log() -> dict:
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": stats.__version__ if hasattr(stats, '__version__') else "unknown",
        "platform": platform.platform(),
        "gpu": "CPU-only (no GPU used)",
        "script": str(Path(__file__).name),
        "note": "Supplementary analyses rc3, rc4, rc6, rc9",
    }


env_log = make_env_log()
print("Environment:", env_log["python"][:40], env_log["utc"])


# ===========================================================================
# LOAD DATA
# ===========================================================================

print("\n--- Loading data ---")

# ---- Sarcasm corpora ----
csc  = pd.read_parquet(SARCASM_DIR / "csc_aggregated.parquet")
mp   = pd.read_parquet(SARCASM_DIR / "multipico_aggregated.parquet")
epic = pd.read_parquet(SARCASM_DIR / "epic_aggregated.parquet")

# ---- MHS per-item (aggregate dis_i) ----
mhs_raw = pd.read_parquet(RAW_DIR / "measuring_hate_speech.parquet")
# Aggregate per comment: k_i, hate_speech_score mean, binary dis (std/range)
mhs_agg = (
    mhs_raw.groupby("comment_id")
    .agg(
        k_i=("annotator_id", "nunique"),
        hate_score_mean=("hate_speech_score", "mean"),
        hate_score_std=("hate_speech_score", "std"),
        hate_frac=("hatespeech", lambda x: (x > 0).mean()),
        n_ann=("annotator_id", "count"),
    )
    .reset_index()
)
# Binary dis for hate: normalized entropy of hate binary label
def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

mhs_agg["dis_mi"] = mhs_agg["hate_frac"].apply(
    lambda p: binary_entropy(p) / np.log(2)  # log(2) = max entropy for binary
)
mhs_agg = mhs_agg[np.isfinite(mhs_agg["dis_mi"])]
print(f"CSC: {len(csc)} items | MultiPICo: {len(mp)} | EPIC: {len(epic)} | MHS: {len(mhs_agg)} items")

# ---- E1 phase1 (signal vs dis correlation data) ----
e1_csc_gpt     = pd.read_parquet(E1_DIR / "e1_CSC_gpt_phase1.parquet")
e1_csc_claude  = pd.read_parquet(E1_DIR / "e1_CSC_claude_phase1.parquet")
e1_mp_gpt      = pd.read_parquet(E1_DIR / "e1_MultiPICo_gpt_phase1.parquet")
e1_mp_claude   = pd.read_parquet(E1_DIR / "e1_MultiPICo_claude_phase1.parquet")

# EPIC frontier
e1_epic_gpt    = pd.read_parquet(E1_DIR / "e1_EPIC_gpt_phaseepic1.parquet")
e1_epic_claude = pd.read_parquet(E1_DIR / "e1_EPIC_claude_phaseepic1.parquet")

# Merge k_i into E1 items
csc_ki  = csc[["item_id", "n_evaluators"]].rename(columns={"n_evaluators": "k_i"})
mp_ki   = mp[["item_id", "n_annotators"]].rename(columns={"n_annotators": "k_i"})
epic_ki = epic[["item_id", "n_annotators"]].rename(columns={"n_annotators": "k_i"})

for df in [e1_csc_gpt, e1_csc_claude]:
    df.drop(columns=["k_i"], errors="ignore", inplace=True)
    df["item_id"] = df["item_id"].astype(str)
    csc_ki["item_id"] = csc_ki["item_id"].astype(str)
    df_m = df.merge(csc_ki, on="item_id", how="left")
    df["k_i"] = df_m["k_i"].values

for df in [e1_mp_gpt, e1_mp_claude]:
    df.drop(columns=["k_i"], errors="ignore", inplace=True)
    df["item_id"] = df["item_id"].astype(str)
    mp_ki["item_id"] = mp_ki["item_id"].astype(str)
    df_m = df.merge(mp_ki, on="item_id", how="left")
    df["k_i"] = df_m["k_i"].values

for df in [e1_epic_gpt, e1_epic_claude]:
    df.drop(columns=["k_i"], errors="ignore", inplace=True)
    df["item_id"] = df["item_id"].astype(str)
    epic_ki["item_id"] = epic_ki["item_id"].astype(str)
    df_m = df.merge(epic_ki, on="item_id", how="left")
    df["k_i"] = df_m["k_i"].values

print("E1 data loaded (with k_i merged)")

# ---- Load E3 xfamily data ----
e3_models = {
    "qwen3":  {"CSC": "e3_q3x8b_CSC.csv",  "MultiPICo": "e3_q3x8b_MultiPICo.csv",
               "EPIC": "e3_q3x8b_EPIC.csv",  "MHS": "e3_q3x8b_MHS.csv",
               "small_key": "q3x8b", "large_key": "q3x32b",
               "small_B": 8.0, "large_B": 32.0},
    "gemma3": {"CSC": "e3_g3x4b_CSC.csv",  "MultiPICo": "e3_g3x4b_MultiPICo.csv",
               "EPIC": "e3_g3x4b_EPIC.csv",  "MHS": "e3_g3x4b_MHS.csv",
               "small_key": "g3x4b", "large_key": "g3x27b",
               "small_B": 4.0, "large_B": 27.0},
    "llama":  {"CSC": "e3_llx1b_CSC.csv",  "MultiPICo": "e3_llx1b_MultiPICo.csv",
               "EPIC": "e3_llx1b_EPIC.csv",  "MHS": "e3_llx1b_MHS.csv",
               "small_key": "llx1b", "large_key": "llx70b",
               "small_B": 1.0, "large_B": 70.0},
}

# Also load Qwen2.5 (main paper family)
qwen25_sizes = {
    "0.5B": "e3_q0p5b", "1.5B": "e3_q1p5b", "3B": "e3_q3b",
    "7B": "e3_q7b", "14B": "e3_q14b", "32B": "e3_q32b",
}

print("E3 model registry built")


# ===========================================================================
# rc3: ANNOTATOR COUNT k_i ANALYSIS
# ===========================================================================

print("\n=== rc3: k_i distribution and reliability ===")

rc3 = {}

# ---- k_i distributions ----
ki_stats = {}
for name, ki_arr in [
    ("CSC",       csc["n_evaluators"].values),
    ("MultiPICo", mp["n_annotators"].values),
    ("EPIC",      epic["n_annotators"].values),
    ("MHS",       mhs_agg["k_i"].values),
]:
    vc = dict(zip(*np.unique(ki_arr, return_counts=True)))
    vc = {int(k): int(v) for k, v in vc.items()}
    ki_stats[name] = {
        "min": int(ki_arr.min()), "median": float(np.median(ki_arr)),
        "mean": float(ki_arr.mean()), "max": int(ki_arr.max()),
        "std": float(ki_arr.std()),
        "value_counts": vc,
        "n_items": int(len(ki_arr)),
        "frac_k_ge4": float(np.mean(ki_arr >= 4)),
        "frac_k_ge5": float(np.mean(ki_arr >= 5)),
    }
    print(f"  {name}: min={vc.get(min(vc.keys()))}, n_items={len(ki_arr)}, "
          f"min_k={ki_arr.min()}, median={np.median(ki_arr):.1f}, max={ki_arr.max()}")

rc3["ki_distributions"] = ki_stats


# ---- CSC split-half reliability of dis_i ----
print("  Computing CSC split-half reliability of dis_i ...")

csc_raw = pd.read_csv(RAW_DIR / "csc_data_full.csv")
print("  CSC raw cols:", list(csc_raw.columns)[:15])

# Find annotator-level rating columns
ann_cols = [c for c in csc_raw.columns if c.startswith("eval_") or c.startswith("rating_") or
            c.startswith("annot")]
print("  Candidate annotator cols:", ann_cols[:10])

# If no individual annotator columns, fall back to parametric split-half via moments
if not ann_cols:
    print("  No individual annotator columns found — using moment-based Spearman-Brown estimate")
    # For Likert: SB = (k * r_item) / (1 + (k-1)*r_item)
    # where r_item = intraclass correlation ≈ var_between / (var_between + var_within)
    # Approximate: ICC(1,1) from known mean/std
    # We use SB on the mean: r_splithalf ≈ alpha
    # Cronbach's alpha for dis_i across k raters:
    # We'll compute an approximate alpha from the standard error of dis_i
    # dis_i = Var(r_i) / denom  → std_dis ≈ std_i / (k-1)^0.5
    # This is an approximation only
    k_vals = csc["n_evaluators"].values
    # Spearman-Brown for reliability of sum with k raters
    # Using r_ii = 0 (no within-item ICC estimate available without raw ratings)
    # We'll estimate bootstrap SE approach instead
    rng_sb = np.random.default_rng(42)
    # Approximate: dis_i ~ Var(r) / denom
    # std_eval from csc is available
    n_sim = 500
    sb_corrs = []
    for sim in range(n_sim):
        perm = rng_sb.permutation(len(csc))
        half1_dis = csc["disagreement"].values[perm[:len(perm)//2]]
        half2_dis = csc["disagreement"].values[perm[len(perm)//2:]]
        # These are different items; can't do split-half on items
        # Actually: split-half should be on annotators WITHIN item
        # Without raw annotator data, we can't do this properly
        sb_corrs.append(float("nan"))
    # Flag this limitation
    rc3["CSC_splithalf_reliability"] = {
        "status": "not_computed",
        "reason": "Individual annotator ratings not recoverable from aggregated CSC data (csc_data_full.csv has no per-annotator columns)",
        "available_cols": list(csc_raw.columns),
        "note": "dis_i = relative variance from aggregated statistics (std_eval, mean_eval); individual rater data not saved."
    }
    print("  CSC individual ratings unavailable — flagging")
else:
    # Full split-half computation
    # ... (would implement here if columns found)
    pass

# ---- MultiPICo/EPIC split-half via parametric approximation ----
# dis_i = H(p_i) / log(k_i)  where p_i = fraction ironic
# For binary annotations, dis_i is FULLY determined by (n_ironic, k_i)
# Split-half reliability: split k_i annotators into two halves
print("  MultiPICo/EPIC: dis_i is deterministic from (p_ironic, k_i)")
print("  Computing parametric SE of dis_i via delta method ...")

mp_orig = pd.read_parquet(RAW_DIR / "multipico_en.parquet")
print("  MultiPICo raw cols:", list(mp_orig.columns)[:15])


# ===========================================================================
# rc3b: Annotator-bootstrap SE of dis_i (via delta method for binary case)
# ===========================================================================

def binary_dis_se(p: float, k: int) -> float:
    """SE of normalized H via delta method: dH/dp * SE(p_hat).
    H(p) = -p*log(p) - (1-p)*log(1-p); SE(p_hat) = sqrt(p*(1-p)/k).
    dH/dp = -log(p/(1-p)).
    normalized_H = H(p) / log(k) if k >= 2 else nan.
    """
    if k < 2 or p <= 0 or p >= 1:
        return 0.0
    dHdp = -np.log(p / (1 - p))
    se_p = np.sqrt(p * (1 - p) / k)
    se_H = abs(dHdp) * se_p
    return se_H / np.log(k)


mp["dis_se"] = [binary_dis_se(row["p_ironic"], int(row["n_annotators"]))
                for _, row in mp.iterrows()]
epic["dis_se"] = [binary_dis_se(row["p_ironic"], int(row["n_annotators"]))
                  for _, row in epic.iterrows()]

rc3["MultiPICo_dis_se"] = {
    "method": "delta method SE of H(p)/log(k)",
    "mean_se": float(mp["dis_se"].mean()),
    "median_se": float(mp["dis_se"].median()),
    "max_se": float(mp["dis_se"].max()),
    "frac_se_gt_0.1": float((mp["dis_se"] > 0.1).mean()),
}
rc3["EPIC_dis_se"] = {
    "method": "delta method SE of H(p)/log(k)",
    "mean_se": float(epic["dis_se"].mean()),
    "median_se": float(epic["dis_se"].median()),
    "max_se": float(epic["dis_se"].max()),
    "frac_se_gt_0.1": float((epic["dis_se"] > 0.1).mean()),
}
print(f"  MultiPICo dis SE: mean={rc3['MultiPICo_dis_se']['mean_se']:.4f}, "
      f"frac>0.1={rc3['MultiPICo_dis_se']['frac_se_gt_0.1']:.3f}")
print(f"  EPIC dis SE: mean={rc3['EPIC_dis_se']['mean_se']:.4f}")


# ---- Stratum stability under annotator resampling (bootstrap) ----
print("  Stratum stability: how often does item change tercile under k_i bootstrap? ...")

def stratum_stability_bootstrap(dis: np.ndarray, k_arr: np.ndarray,
                                  p_arr: np.ndarray, n_boot: int = 500) -> dict:
    """For binary corpora: resample n annotations from Binomial(k_i, p_i), recompute dis, check tercile."""
    tau_low  = np.percentile(dis, 33.3)
    tau_high = np.percentile(dis, 66.7)
    orig_tercile = np.where(dis <= tau_low, 0, np.where(dis >= tau_high, 2, 1))

    n_items = len(dis)
    change_frac_low  = np.zeros(n_items)
    change_frac_high = np.zeros(n_items)

    rng_stab = np.random.default_rng(42)
    changes = np.zeros(n_items)

    for _ in range(n_boot):
        # Resample n_ironic from Binomial(k_i, p_i) for each item
        new_n = rng_stab.binomial(k_arr.astype(int), p_arr)
        new_p = np.where(k_arr > 0, new_n / k_arr, 0.5)
        new_dis = np.array([
            binary_entropy(float(p_)) / np.log(k_) if k_ >= 2 else 0.0
            for p_, k_ in zip(new_p, k_arr)
        ])
        new_tau_low  = np.percentile(new_dis, 33.3)
        new_tau_high = np.percentile(new_dis, 66.7)
        new_terc = np.where(new_dis <= new_tau_low, 0, np.where(new_dis >= new_tau_high, 2, 1))
        changes += (new_terc != orig_tercile).astype(float)

    change_rate = changes / n_boot
    return {
        "mean_change_rate": float(change_rate.mean()),
        "frac_items_ever_change": float((changes > 0).mean()),
        "frac_items_change_gt50pct": float((change_rate > 0.5).mean()),
        "by_tercile": {
            "low":  float(change_rate[orig_tercile == 0].mean()),
            "mid":  float(change_rate[orig_tercile == 1].mean()),
            "high": float(change_rate[orig_tercile == 2].mean()),
        }
    }

for corpus_name, corpus_df in [("MultiPICo", mp), ("EPIC", epic)]:
    stab = stratum_stability_bootstrap(
        corpus_df["disagreement"].values,
        corpus_df["n_annotators"].values,
        corpus_df["p_ironic"].values,
    )
    rc3[f"{corpus_name}_stratum_stability"] = stab
    print(f"  {corpus_name} stability: mean_change={stab['mean_change_rate']:.4f}, "
          f"ever_change={stab['frac_items_ever_change']:.3f}")


# ---- rc3c: Partial Spearman correlations controlling for k_i ----
print("  Computing partial Spearman correlations controlling k_i ...")

def partial_spearman_signal_dis(e1_df: pd.DataFrame, signal_col: str,
                                 corpus_name: str, provider: str) -> dict:
    """Signal vs dis controlling for k_i."""
    df = e1_df.dropna(subset=[signal_col, "dis_mi", "k_i"])
    x = df[signal_col].values
    y = df["dis_mi"].values
    z = df["k_i"].values

    # Raw Spearman
    raw = spearman(x, y)
    # Partial Spearman (controlling k_i)
    partial = partial_spearman(x, y, z)
    return {
        "corpus": corpus_name, "provider": provider, "signal": signal_col,
        "raw_spearman": raw,
        "partial_spearman_ctrl_ki": partial,
    }

partial_corrs = []

# CSC GPT (verbalized: 1 - conf/100; logprob: 1 - p_yes_lp)
for e1_df, corp, prov, sig_cols in [
    (e1_csc_gpt,    "CSC",       "gpt",    [("U_verbal", lambda df: 1 - df["conf"]/100),
                                             ("U_logprob", lambda df: 1 - df["p_yes_lp"])]),
    (e1_csc_claude, "CSC",       "claude", [("U_verbal", lambda df: 1 - df["conf"]/100)]),
    (e1_mp_gpt,     "MultiPICo", "gpt",    [("U_verbal", lambda df: 1 - df["conf"]/100),
                                             ("U_logprob", lambda df: 1 - df["p_yes_lp"])]),
    (e1_mp_claude,  "MultiPICo", "claude", [("U_verbal", lambda df: 1 - df["conf"]/100)]),
    (e1_epic_gpt,   "EPIC",      "gpt",    [("U_verbal", lambda df: 1 - df["conf"]/100),
                                             ("U_logprob", lambda df: 1 - df["p_yes_lp"])]),
    (e1_epic_claude,"EPIC",      "claude", [("U_verbal", lambda df: 1 - df["conf"]/100)]),
]:
    for sig_name, sig_fn in sig_cols:
        df_work = e1_df.copy()
        df_work["_signal"] = sig_fn(df_work)
        result = partial_spearman_signal_dis(df_work, "_signal", corp, prov)
        result["signal"] = sig_name
        partial_corrs.append(result)
        print(f"  {corp}/{prov}/{sig_name}: rho={result['raw_spearman']['rho']:+.3f} "
              f"→ partial={result['partial_spearman_ctrl_ki']['rho']:+.3f}")

rc3["partial_spearman_ctrl_ki"] = partial_corrs


# ---- rc3d: Spearman within k_i strata ----
print("  Computing Spearman within k_i strata ...")
within_ki_corrs = []
for e1_df, corp, prov, sig_fn in [
    (e1_csc_gpt,    "CSC",       "gpt",    lambda df: 1 - df["conf"]/100),
    (e1_csc_claude, "CSC",       "claude", lambda df: 1 - df["conf"]/100),
    (e1_mp_gpt,     "MultiPICo", "gpt",    lambda df: 1 - df["conf"]/100),
    (e1_mp_claude,  "MultiPICo", "claude", lambda df: 1 - df["conf"]/100),
]:
    df_work = e1_df.copy()
    df_work["_signal"] = sig_fn(df_work)
    df_work = df_work.dropna(subset=["_signal", "dis_mi", "k_i"])
    strata_result = {}
    for k_val in sorted(df_work["k_i"].dropna().unique()):
        sub = df_work[df_work["k_i"] == k_val]
        if len(sub) < 10:
            continue
        r = spearman(sub["_signal"].values, sub["dis_mi"].values)
        strata_result[int(k_val)] = {"rho": r["rho"], "n": r["n"]}
    within_ki_corrs.append({
        "corpus": corp, "provider": prov, "signal": "U_verbal",
        "within_ki_strata": strata_result,
    })

rc3["spearman_within_ki_strata"] = within_ki_corrs


# ---- rc3e: ΔAsym within k_i bins ----
print("  Computing ΔAsym within k_i bins ...")

# For this we need per-item bal_acc from E3 (Qwen2.5 small and large)
# Load Qwen2.5 0.5B and 32B for CSC and MultiPICo
def load_e3_model(prefix: str, corpus: str) -> pd.DataFrame:
    fpath = E3_DIR / f"{prefix}_{corpus}.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    df["pred"] = (df["p_yes"] >= 0.5).astype(int)
    return df

q05b_csc  = load_e3_model("e3_q0p5b", "CSC")
q32b_csc  = load_e3_model("e3_q32b",  "CSC")
q05b_mp   = load_e3_model("e3_q0p5b", "MultiPICo")
q32b_mp   = load_e3_model("e3_q32b",  "MultiPICo")

delta_asym_ki_bins = {}
for corpus_name, small_df, large_df, ki_src in [
    ("CSC",       q05b_csc, q32b_csc, csc_ki),
    ("MultiPICo", q05b_mp,  q32b_mp,  mp_ki),
]:
    if small_df is None or large_df is None:
        delta_asym_ki_bins[corpus_name] = {"status": "data_unavailable"}
        continue

    # Merge small, large, k_i
    merged = small_df[["item_id", "p_yes", "y_true", "dis_mi"]].copy()
    merged = merged.rename(columns={"p_yes": "p_small"})
    merged["item_id"] = merged["item_id"].astype(str)

    lrg = large_df[["item_id", "p_yes"]].copy()
    lrg["item_id"] = lrg["item_id"].astype(str)
    merged = merged.merge(lrg.rename(columns={"p_yes": "p_large"}), on="item_id")

    ki_src2 = ki_src.copy()
    ki_src2["item_id"] = ki_src2["item_id"].astype(str)
    merged = merged.merge(ki_src2, on="item_id", how="left")

    # Full ΔAsym
    tau_low  = np.percentile(merged["dis_mi"].values, 33.3)
    tau_high = np.percentile(merged["dis_mi"].values, 66.7)
    df_low  = merged[merged["dis_mi"] <= tau_low]
    df_high = merged[merged["dis_mi"] >= tau_high]
    full_da = bootstrap_delta_asym(df_low, df_high, "p_small", "p_large", n_boot=2000)

    # ΔAsym within k_i bins
    ki_bins_result = {}
    for k_val in sorted(merged["k_i"].dropna().unique()):
        sub = merged[merged["k_i"] == k_val]
        if len(sub) < 30:
            continue
        t_low  = np.percentile(sub["dis_mi"].values, 33.3)
        t_high = np.percentile(sub["dis_mi"].values, 66.7)
        sub_low  = sub[sub["dis_mi"] <= t_low]
        sub_high = sub[sub["dis_mi"] >= t_high]
        if len(sub_low) < 5 or len(sub_high) < 5:
            continue
        da = bootstrap_delta_asym(sub_low, sub_high, "p_small", "p_large", n_boot=1000)
        ki_bins_result[int(k_val)] = {"n": int(len(sub)),
                                       "n_low": int(len(sub_low)),
                                       "n_high": int(len(sub_high)),
                                       "delta_asym": da}

    delta_asym_ki_bins[corpus_name] = {
        "full_delta_asym": full_da,
        "within_ki_bins": ki_bins_result,
    }
    print(f"  {corpus_name} full ΔAsym={full_da['delta_asym']:+.3f} (p_le0={full_da['boot_p_le0']:.3f})")
    for k_val, res in ki_bins_result.items():
        print(f"    k={k_val}: n={res['n']}, ΔAsym={res['delta_asym']['delta_asym']:+.3f}")

rc3["delta_asym_within_ki_bins"] = delta_asym_ki_bins


# ---- rc3f: AUROC (routing) with k_i ≥ 4 restriction ----
print("  Computing routing AUROC with k_i filtering ...")

def routing_auroc_ki_filter(e1_df: pd.DataFrame, q05b_df: pd.DataFrame,
                             q32b_df: pd.DataFrame, ki_threshold: int,
                             corpus_name: str) -> dict:
    """AUROC of 1-dis_i for predicting y_fix = small_wrong & large_right."""
    if q05b_df is None or q32b_df is None:
        return {"status": "data_unavailable"}
    # Merge everything
    e1_work = e1_df[["item_id", "dis_mi", "k_i", "y_true", "conf"]].copy()
    e1_work["item_id"] = e1_work["item_id"].astype(str)
    e1_work["U_verbal"] = 1 - e1_work["conf"] / 100

    q0 = q05b_df[["item_id", "p_yes"]].copy()
    q0["item_id"] = q0["item_id"].astype(str)
    q0 = q0.rename(columns={"p_yes": "p_small"})
    q3 = q32b_df[["item_id", "p_yes"]].copy()
    q3["item_id"] = q3["item_id"].astype(str)
    q3 = q3.rename(columns={"p_yes": "p_large"})

    merged = e1_work.merge(q0, on="item_id").merge(q3, on="item_id")
    merged["pred_small"] = (merged["p_small"] >= 0.5).astype(int)
    merged["pred_large"] = (merged["p_large"] >= 0.5).astype(int)
    merged["y_fix"] = (
        (merged["pred_small"] != merged["y_true"]) &
        (merged["pred_large"] == merged["y_true"])
    ).astype(int)

    results = {}
    for label, subset in [("all", merged),
                           (f"k_ge_{ki_threshold}", merged[merged["k_i"] >= ki_threshold])]:
        if subset["y_fix"].sum() < 5:
            results[label] = {"status": "too_few_positive"}
            continue
        auc_dis  = auroc(1 - subset["dis_mi"].values,    subset["y_fix"].values)
        auc_verb = auroc(subset["U_verbal"].values,       subset["y_fix"].values)
        results[label] = {
            "n": int(len(subset)),
            "n_yfix": int(subset["y_fix"].sum()),
            "auroc_1minus_dis": auc_dis,
            "auroc_U_verbal":   auc_verb,
        }
    return results

auroc_ki_results = {}
for corp, e1_df, q0, q3 in [
    ("CSC",       e1_csc_gpt,  q05b_csc, q32b_csc),
    ("MultiPICo", e1_mp_gpt,   q05b_mp,  q32b_mp),
]:
    if q0 is None or q3 is None:
        auroc_ki_results[corp] = {"status": "data_unavailable"}
        continue
    res = routing_auroc_ki_filter(e1_df, q0, q3, ki_threshold=4, corpus_name=corp)
    auroc_ki_results[corp] = res
    print(f"  {corp} routing AUROC:")
    for label, r in res.items():
        if isinstance(r, dict) and "auroc_1minus_dis" in r:
            print(f"    [{label}] n={r['n']}, AUROC(1-dis)={r['auroc_1minus_dis']:.3f}, "
                  f"AUROC(U_verbal)={r['auroc_U_verbal']:.3f}")

rc3["routing_auroc_ki_filter"] = auroc_ki_results

# Save rc3 results
with open(OUT / "rc3_ki_reliability.json", "w") as f:
    json.dump({"env": env_log, "rc3": rc3}, f, indent=2, default=str)
print("  -> Saved rc3_ki_reliability.json")


# ===========================================================================
# rc4: MEAN-INDEPENDENCE AND Eq.1 ANALYSIS
# ===========================================================================

print("\n=== rc4: Mean-independence and Eq.1 analysis ===")

rc4 = {}

# ---- Binary corpora: dis_i is deterministic function of p_ironic and k_i ----
print("  Verifying dis_i = f(p_ironic, k_i) for MultiPICo/EPIC ...")

def compute_dis_from_p_k(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Reconstruct dis_i from p_ironic and k_i (binary corpus formula)."""
    dis = np.zeros(len(p))
    for i, (pi, ki) in enumerate(zip(p, k)):
        if ki < 2:
            dis[i] = 0.0
        elif pi <= 0 or pi >= 1:
            dis[i] = 0.0
        else:
            dis[i] = binary_entropy(float(pi)) / np.log(float(ki))
    return dis

# For MultiPICo
mp_dis_recon = compute_dis_from_p_k(mp["p_ironic"].values, mp["n_annotators"].values)
mp_corr = float(np.corrcoef(mp_dis_recon, mp["disagreement"].values)[0, 1])
mp_max_err = float(np.max(np.abs(mp_dis_recon - mp["disagreement"].values)))
print(f"  MultiPICo: corr(recon, stored)={mp_corr:.6f}, max_err={mp_max_err:.2e}")

epic_dis_recon = compute_dis_from_p_k(epic["p_ironic"].values, epic["n_annotators"].values)
epic_corr = float(np.corrcoef(epic_dis_recon, epic["disagreement"].values)[0, 1])
epic_max_err = float(np.max(np.abs(epic_dis_recon - epic["disagreement"].values)))
print(f"  EPIC:      corr(recon, stored)={epic_corr:.6f}, max_err={epic_max_err:.2e}")

rc4["binary_corpora_dis_determinism"] = {
    "claim": "For binary-label corpora (MultiPICo, EPIC), dis_i = H(p_ironic) / log(k_i) is a deterministic function of (p_ironic, k_i). When k_i is fixed, dis_i is a monotone function of |p_ironic - 0.5|.",
    "MultiPICo": {
        "corr_reconstructed_vs_stored": mp_corr,
        "max_abs_err": mp_max_err,
        "unique_k_values": sorted([int(x) for x in mp["n_annotators"].unique()]),
        "interpretation": "dis_i is NOT mean-independent in the same sense as CSC relative variance; it IS a function of p_ironic. Mean-independence claim is valid only for CSC (Likert).",
    },
    "EPIC": {
        "corr_reconstructed_vs_stored": epic_corr,
        "max_abs_err": epic_max_err,
        "unique_k_values": sorted([int(x) for x in epic["n_annotators"].unique()]),
    }
}

# ---- CSC: mean-independence check ----
print("  CSC: checking mean-independence of dis_i ...")
# dis_CSC = Var(r) / max(0.5, (mean_r - 1)(6 - mean_r))
# Under perfect mean-independence: corr(dis_i, mean_eval) should be near 0
rho_dis_mean, p_dis_mean = stats.spearmanr(csc["disagreement"].values,
                                            csc["mean_eval"].values)
# Also check with denominator floor
csc["denom_theoretical"] = (csc["mean_eval"] - 1) * (6 - csc["mean_eval"])
csc["denom_floored"] = csc["denom_theoretical"].clip(lower=0.5)
frac_floored = float((csc["denom_theoretical"] < 0.5).mean())
print(f"  CSC corr(dis, mean_eval): rho={rho_dis_mean:.4f} (p={p_dis_mean:.4f})")
print(f"  CSC fraction with denom < 0.5 (floored): {frac_floored:.4f} ({int(frac_floored * len(csc))} items)")

# Eq.1 range check
dis_min = float(csc["disagreement"].min())
dis_max = float(csc["disagreement"].max())
print(f"  CSC dis_i range: [{dis_min:.6f}, {dis_max:.6f}] (formula: bounded in (0,1])")

# Check if any dis_i > 1 (formula violation)
n_gt1 = int((csc["disagreement"] > 1.0).sum())
print(f"  CSC items with dis_i > 1: {n_gt1}")

# Denominator floor analysis
csc["dis_recon"] = csc["std_eval"]**2 / csc["denom_floored"]
corr_recon = float(np.corrcoef(csc["dis_recon"].values, csc["disagreement"].values)[0,1])
print(f"  CSC corr(dis_recon, stored): {corr_recon:.6f}")

rc4["CSC_mean_independence"] = {
    "spearman_dis_vs_mean_eval": {"rho": float(rho_dis_mean), "p": float(p_dis_mean)},
    "dis_range": [dis_min, dis_max],
    "n_items_dis_gt1": n_gt1,
    "frac_denom_floored": frac_floored,
    "n_denom_floored": int(frac_floored * len(csc)),
    "corr_reconstructed_vs_stored": corr_recon,
    "interpretation": "CSC: Spearman(dis, mean_eval) should be near 0 if mean-independent. Relative variance with floor is approximately mean-independent but not perfectly so.",
    "denominator_floor": 0.5,
}

# ---- MHS mean-independence ----
# MHS: dis_i = H(p_hate) / log(2) which IS a function of p_hate (the mean of binary hate label)
# This confirms mean-independence only meaningful for CSC
rho_mhs = spearman(mhs_agg["hate_frac"].values, mhs_agg["dis_mi"].values)
rc4["MHS_dis_vs_mean"] = {
    "spearman_dis_vs_hate_frac": rho_mhs,
    "interpretation": "MHS: dis_i = H(p_hate)/log(2) is monotone in |p_hate - 0.5|. The claim of mean-independence applies only to CSC (Likert scale)."
}
print(f"  MHS: corr(dis, hate_frac)={rho_mhs['rho']:.4f} (expected non-zero)")

with open(OUT / "rc4_mean_independence.json", "w") as f:
    json.dump({"env": env_log, "rc4": rc4}, f, indent=2, default=str)
print("  -> Saved rc4_mean_independence.json")


# ===========================================================================
# rc6: BEYOND SPEARMAN — AUROC, MI, ISOTONIC R², CLASS-CONDITIONAL SPEARMAN
# ===========================================================================

print("\n=== rc6: Beyond Spearman ===")

rc6 = {}

def beyond_spearman(signal: np.ndarray, dis: np.ndarray,
                     labels: np.ndarray, corpus: str,
                     provider: str, signal_name: str) -> dict:
    """Full suite of beyond-Spearman metrics."""
    mask = np.isfinite(signal) & np.isfinite(dis)
    s, d, y = signal[mask], dis[mask], labels[mask]
    n = len(s)
    if n < 20:
        return {"status": "too_few_items", "n": n}

    # 1. Raw Spearman
    raw_sp = spearman(s, d)

    # 2. AUROC for discriminating high vs low dis tercile
    tau_low  = np.percentile(d, 33.3)
    tau_high = np.percentile(d, 66.7)
    is_high  = (d >= tau_high).astype(int)
    is_low   = (d <= tau_low).astype(int)
    # AUROC: signal for predicting high-dis tercile (score = signal, 1 = high-dis)
    in_extreme = (d <= tau_low) | (d >= tau_high)
    s_ext, is_high_ext = s[in_extreme], is_high[in_extreme]
    if len(np.unique(is_high_ext)) == 2:
        auc_highlow = auroc(s_ext, is_high_ext)
    else:
        auc_highlow = float("nan")

    # 3. Mutual Information (permutation-corrected)
    mi_res = permutation_mi(s, d)

    # 4. Isotonic R²
    iso_r2 = isotonic_r2(s, d)

    # 5. Class-conditional Spearman (conditioned on predicted label)
    preds_binary = (signal >= 0.5).astype(int)
    cc_spearman = {}
    for pred_label in [0, 1]:
        sub_mask = (preds_binary[mask] == pred_label)
        if sub_mask.sum() < 10:
            cc_spearman[str(pred_label)] = {"status": "too_few", "n": int(sub_mask.sum())}
            continue
        r = spearman(s[sub_mask], d[sub_mask])
        cc_spearman[str(pred_label)] = r

    # 6. Logprob normalization check (for GPT with p_yes_lp)
    # Normalization: p_yes_normalized = p_yes_lp / (p_yes_lp + p_no_lp)
    # We only have p_yes_lp saved (not p_no_lp), so we cannot redo this.
    # Flag this.
    logprob_note = (
        "p_yes_lp saved but p_no_lp not saved in e1 parquet files. "
        "Two-token normalization p_yes/(p_yes+p_no) cannot be computed from saved data."
        if "p_yes_lp" not in signal_name else
        "Two-token normalization not computable: only p_yes_lp saved, not p_no_lp."
    )

    return {
        "corpus": corpus, "provider": provider, "signal": signal_name, "n": n,
        "raw_spearman": raw_sp,
        "auroc_signal_for_high_vs_low_dis_tercile": auc_highlow,
        "mutual_information": mi_res,
        "isotonic_r2_signal_vs_dis": iso_r2,
        "class_conditional_spearman": cc_spearman,
        "logprob_normalization_status": logprob_note if signal_name == "U_logprob" else "N/A",
    }


rc6_results = []

# Process all E1 corpora × providers × signals
for e1_df, corp, prov in [
    (e1_csc_gpt,     "CSC",       "gpt"),
    (e1_csc_claude,  "CSC",       "claude"),
    (e1_mp_gpt,      "MultiPICo", "gpt"),
    (e1_mp_claude,   "MultiPICo", "claude"),
    (e1_epic_gpt,    "EPIC",      "gpt"),
    (e1_epic_claude, "EPIC",      "claude"),
]:
    # U_verbal
    sig_v = 1 - e1_df["conf"].values / 100
    res = beyond_spearman(sig_v, e1_df["dis_mi"].values, e1_df["y_true"].values,
                          corp, prov, "U_verbal")
    rc6_results.append(res)
    if "auroc_signal_for_high_vs_low_dis_tercile" in res:
        print(f"  {corp}/{prov}/U_verbal: rho={res['raw_spearman']['rho']:+.3f}, "
              f"AUROC(H/L)={res['auroc_signal_for_high_vs_low_dis_tercile']:.3f}, "
              f"MI_bc={res['mutual_information']['mi_bias_corrected']:.4f}, "
              f"iso_R2={res['isotonic_r2_signal_vs_dis']:.4f}")

    # U_logprob (GPT only)
    if "p_yes_lp" in e1_df.columns and prov == "gpt":
        sig_lp = 1 - e1_df["p_yes_lp"].values
        res_lp = beyond_spearman(sig_lp, e1_df["dis_mi"].values, e1_df["y_true"].values,
                                  corp, prov, "U_logprob")
        rc6_results.append(res_lp)
        if "auroc_signal_for_high_vs_low_dis_tercile" in res_lp:
            print(f"  {corp}/{prov}/U_logprob: rho={res_lp['raw_spearman']['rho']:+.3f}, "
                  f"AUROC(H/L)={res_lp['auroc_signal_for_high_vs_low_dis_tercile']:.3f}")

# Also add Sampling Entropy (from e3 Qwen2.5 data)
for corpus_name, corpus_files in [
    ("CSC",       ["e3_q14b_CSC.csv"]),
    ("MultiPICo", ["e3_q14b_MultiPICo.csv"]),
    ("EPIC",      ["e3_q14b_EPIC.csv"]),
]:
    for fpath in corpus_files:
        ffile = E3_DIR / fpath
        if not ffile.exists():
            continue
        df_e3 = pd.read_csv(ffile)
        # SE = sampling entropy H = -p*log(p) - (1-p)*log(1-p) stored as 'H' column
        if "H" not in df_e3.columns:
            continue
        # Merge dis_mi
        if "dis_mi" not in df_e3.columns:
            continue
        sig_se = df_e3["H"].values
        dis_se = df_e3["dis_mi"].values
        y_se   = df_e3["y_true"].values
        res_se = beyond_spearman(sig_se, dis_se, y_se, corpus_name, "qwen14b", "U_sampling_entropy")
        rc6_results.append(res_se)
        if "auroc_signal_for_high_vs_low_dis_tercile" in res_se:
            print(f"  {corpus_name}/qwen14b/SE: rho={res_se['raw_spearman']['rho']:+.3f}, "
                  f"AUROC(H/L)={res_se['auroc_signal_for_high_vs_low_dis_tercile']:.3f}")

# Logprob normalization note
rc6["logprob_normalization"] = {
    "status": "not_computable",
    "reason": "Only p_yes_lp (log-prob of predicted label token) is saved in e1 parquet files. "
               "The complementary p_no_lp was not saved. "
               "Two-label normalized probability p_yes/(p_yes+p_no) cannot be recomputed from saved data. "
               "The original U_logprob = 1 - exp(log_p_yes_lp) uses only the top-logprob of the predicted token.",
    "recommendation": "Superseded by run_corrections.py (rc6_logprob_fixed.json): p_yes_lp is already two-label normalized."
}

rc6["signal_metrics"] = rc6_results

with open(OUT / "rc6_beyond_spearman.json", "w") as f:
    json.dump({"env": env_log, "rc6": rc6}, f, indent=2, default=str)
print("  -> Saved rc6_beyond_spearman.json")


# ===========================================================================
# rc9: CAPABILITY GRADIENT MONOTONICITY
# ===========================================================================

print("\n=== rc9: Capability gradient monotonicity ===")

rc9 = {}

def bal_acc_vec(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return 0.5 * (tpr + tnr)


def analyze_family(family_name: str, corpus_name: str,
                   model_files: list, sizes_B: list,
                   ki_df=None) -> dict:
    """
    model_files: list of (filename, size_B) pairs
    Returns ΔAsym, monotonicity check, Spearman(size, bal_acc).
    """
    dfs = []
    for fname, sB in model_files:
        fpath = E3_DIR / fname
        if not fpath.exists():
            return {"status": f"file_missing: {fname}"}
        df = pd.read_csv(fpath)
        if "dis_mi" not in df.columns or "y_true" not in df.columns:
            return {"status": "missing_columns"}
        df["size_B"] = sB
        df["pred"] = (df["p_yes"] >= 0.5).astype(int)
        dfs.append(df)

    if len(dfs) < 2:
        return {"status": "insufficient_models"}

    # Compute overall bal_acc per model size (all items)
    base_df = dfs[0][["item_id", "dis_mi", "y_true"]].copy()
    base_df["item_id"] = base_df["item_id"].astype(str)

    bal_accs = {}
    for df in dfs:
        sB = df["size_B"].iloc[0]
        ba = bal_acc_vec(df["pred"].values, df["y_true"].values)
        bal_accs[sB] = ba

    sizes = sorted(bal_accs.keys())
    ba_vals = [bal_accs[s] for s in sizes]

    # Monotonicity: Spearman(log_size, bal_acc)
    log_sizes = np.log(sizes)
    if len(sizes) >= 2:
        mono_rho, mono_p = stats.spearmanr(log_sizes, ba_vals)
    else:
        mono_rho, mono_p = float("nan"), float("nan")

    # Is strictly monotone?
    is_monotone = all(ba_vals[i] <= ba_vals[i+1] for i in range(len(ba_vals)-1))

    # ΔAsym: smallest vs largest
    df_small = [d for d in dfs if d["size_B"].iloc[0] == sizes[0]][0]
    df_large = [d for d in dfs if d["size_B"].iloc[0] == sizes[-1]][0]

    # Compute tercile thresholds from base dis_mi
    dis_all = base_df["dis_mi"].values
    tau_low  = np.percentile(dis_all, 33.3)
    tau_high = np.percentile(dis_all, 66.7)

    # Merge small and large on item_id
    sm = df_small[["item_id", "p_yes", "y_true", "dis_mi"]].copy()
    sm["item_id"] = sm["item_id"].astype(str)
    lg = df_large[["item_id", "p_yes"]].copy()
    lg["item_id"] = lg["item_id"].astype(str)
    merged = sm.merge(lg.rename(columns={"p_yes": "p_large"}), on="item_id")
    merged = merged.rename(columns={"p_yes": "p_small"})

    df_low  = merged[merged["dis_mi"] <= tau_low]
    df_high = merged[merged["dis_mi"] >= tau_high]

    if len(df_low) < 10 or len(df_high) < 10:
        da = {"status": "insufficient_items_per_stratum"}
    else:
        da = bootstrap_delta_asym(df_low, df_high, "p_small", "p_large", n_boot=2000)

    result = {
        "family": family_name, "corpus": corpus_name,
        "sizes_B": sizes, "bal_accs": ba_vals,
        "monotone": is_monotone,
        "spearman_log_size_bal_acc": {"rho": float(mono_rho), "p": float(mono_p)},
        "delta_asym_small_to_large": da,
        "n_low": int(len(df_low)),
        "n_high": int(len(df_high)),
    }
    return result


# ---- Qwen3 family (main xfamily model) ----
corpora_for_q3 = ["CSC", "MultiPICo", "EPIC", "MHS"]
q3_models_per_corpus = {
    "CSC":       [("e3_q3x8b_CSC.csv", 8), ("e3_q3x14b_CSC.csv", 14), ("e3_q3x32b_CSC.csv", 32)],
    "MultiPICo": [("e3_q3x8b_MultiPICo.csv", 8), ("e3_q3x14b_MultiPICo.csv", 14), ("e3_q3x32b_MultiPICo.csv", 32)],
    "EPIC":      [("e3_q3x8b_EPIC.csv", 8), ("e3_q3x14b_EPIC.csv", 14), ("e3_q3x32b_EPIC.csv", 32)],
    "MHS":       [("e3_q3x8b_MHS.csv", 8), ("e3_q3x14b_MHS.csv", 14), ("e3_q3x32b_MHS.csv", 32)],
}
g3_models_per_corpus = {
    "CSC":       [("e3_g3x4b_CSC.csv", 4), ("e3_g3x12b_CSC.csv", 12), ("e3_g3x27b_CSC.csv", 27)],
    "MultiPICo": [("e3_g3x4b_MultiPICo.csv", 4), ("e3_g3x12b_MultiPICo.csv", 12), ("e3_g3x27b_MultiPICo.csv", 27)],
    "EPIC":      [("e3_g3x4b_EPIC.csv", 4), ("e3_g3x12b_EPIC.csv", 12), ("e3_g3x27b_EPIC.csv", 27)],
    "MHS":       [("e3_g3x4b_MHS.csv", 4), ("e3_g3x12b_MHS.csv", 12), ("e3_g3x27b_MHS.csv", 27)],
}
llx_models_per_corpus = {
    "CSC":       [("e3_llx1b_CSC.csv", 1), ("e3_llx3b_CSC.csv", 3), ("e3_llx8b_CSC.csv", 8), ("e3_llx70b_CSC.csv", 70)],
    "MultiPICo": [("e3_llx1b_MultiPICo.csv", 1), ("e3_llx3b_MultiPICo.csv", 3), ("e3_llx8b_MultiPICo.csv", 8), ("e3_llx70b_MultiPICo.csv", 70)],
    "EPIC":      [("e3_llx1b_EPIC.csv", 1), ("e3_llx3b_EPIC.csv", 3), ("e3_llx8b_EPIC.csv", 8), ("e3_llx70b_EPIC.csv", 70)],
    "MHS":       [("e3_llx1b_MHS.csv", 1), ("e3_llx3b_MHS.csv", 3), ("e3_llx8b_MHS.csv", 8), ("e3_llx70b_MHS.csv", 70)],
}

# Qwen2.5 (main paper family) — also check monotonicity
q25_models_per_corpus = {
    "CSC":       [("e3_q0p5b_CSC.csv", 0.5), ("e3_q1p5b_CSC.csv", 1.5), ("e3_q3b_CSC.csv", 3),
                  ("e3_q7b_CSC.csv", 7), ("e3_q14b_CSC.csv", 14), ("e3_q32b_CSC.csv", 32)],
    "MultiPICo": [("e3_q0p5b_MultiPICo.csv", 0.5), ("e3_q1p5b_MultiPICo.csv", 1.5), ("e3_q3b_MultiPICo.csv", 3),
                  ("e3_q7b_MultiPICo.csv", 7), ("e3_q14b_MultiPICo.csv", 14), ("e3_q32b_MultiPICo.csv", 32)],
    "EPIC":      [("e3_q0p5b_EPIC.csv", 0.5), ("e3_q1p5b_EPIC.csv", 1.5), ("e3_q3b_EPIC.csv", 3),
                  ("e3_q7b_EPIC.csv", 7), ("e3_q14b_EPIC.csv", 14), ("e3_q32b_EPIC.csv", 32)],
}

rc9_results = {}
for family_name, models_map in [
    ("qwen3",  q3_models_per_corpus),
    ("gemma3", g3_models_per_corpus),
    ("llama_instruct", llx_models_per_corpus),
    ("qwen2.5", q25_models_per_corpus),
]:
    rc9_results[family_name] = {}
    for corpus_name, model_list in models_map.items():
        res = analyze_family(family_name, corpus_name, model_list, [s for _, s in model_list])
        rc9_results[family_name][corpus_name] = res
        if "bal_accs" in res:
            sizes_str = "→".join(f"{s:.0f}B:{ba:.3f}" for s, ba in zip(res["sizes_B"], res["bal_accs"]))
            da_val = res["delta_asym_small_to_large"]
            if isinstance(da_val, dict) and "delta_asym" in da_val:
                print(f"  {family_name}/{corpus_name}: {sizes_str} | "
                      f"mono={res['monotone']} rho={res['spearman_log_size_bal_acc']['rho']:+.2f} | "
                      f"ΔAsym={da_val['delta_asym']:+.3f} p={da_val['boot_p_le0']:.3f}")
            else:
                print(f"  {family_name}/{corpus_name}: {sizes_str} | mono={res['monotone']}")

rc9["family_results"] = rc9_results

# ---- Summary: monotonicity vs ΔAsym significance ----
print("\n  Summary: families with ΔAsym significant (p < 0.1) by monotonicity:")
mono_summary = []
for family, corpus_dict in rc9_results.items():
    for corpus, res in corpus_dict.items():
        if "delta_asym_small_to_large" not in res:
            continue
        da = res["delta_asym_small_to_large"]
        if not isinstance(da, dict) or "delta_asym" not in da:
            continue
        mono_summary.append({
            "family": family, "corpus": corpus,
            "monotone": res.get("monotone", None),
            "spearman_rho": res.get("spearman_log_size_bal_acc", {}).get("rho", float("nan")),
            "delta_asym": da["delta_asym"],
            "boot_p_le0": da["boot_p_le0"],
            "significant_p10": da["boot_p_le0"] < 0.1,
        })

rc9["monotonicity_vs_delta_asym_summary"] = mono_summary

# Count significant vs not by monotone status
mono_sig   = [r for r in mono_summary if r["monotone"] and r["significant_p10"]]
mono_ns    = [r for r in mono_summary if r["monotone"] and not r["significant_p10"]]
nonmono_sig = [r for r in mono_summary if not r["monotone"] and r["significant_p10"]]
nonmono_ns  = [r for r in mono_summary if not r["monotone"] and not r["significant_p10"]]

print(f"  Monotone + sig: {len(mono_sig)}, Monotone + ns: {len(mono_ns)}")
print(f"  Non-mono + sig: {len(nonmono_sig)}, Non-mono + ns: {len(nonmono_ns)}")

rc9["contingency_monotone_vs_significance"] = {
    "monotone_and_significant_p10":     len(mono_sig),
    "monotone_and_nonsignificant_p10":  len(mono_ns),
    "nonmonotone_and_significant_p10":  len(nonmono_sig),
    "nonmonotone_and_nonsignificant_p10": len(nonmono_ns),
}

with open(OUT / "rc9_capability_gradient.json", "w") as f:
    json.dump({"env": env_log, "rc9": rc9}, f, indent=2, default=str)
print("  -> Saved rc9_capability_gradient.json")


# ===========================================================================
# HEADLINE NUMBER SANITY CHECK (reproduce paper numbers from saved data)
# ===========================================================================

print("\n=== Sanity check: reproduce headline numbers ===")
sanity = {}

# Table 3: Spearman(U_verbal, dis) for CSC/claude = +0.115
s_csc_cl = 1 - e1_csc_claude["conf"].values / 100
d_csc = e1_csc_claude["dis_mi"].values
rho_sanity_csc_cl, _ = stats.spearmanr(s_csc_cl[np.isfinite(s_csc_cl) & np.isfinite(d_csc)],
                                         d_csc[np.isfinite(s_csc_cl) & np.isfinite(d_csc)])
sanity["rho_Uverbal_dis_CSC_claude"] = {
    "paper_value": 0.115, "computed": float(rho_sanity_csc_cl),
    "match": abs(float(rho_sanity_csc_cl) - 0.115) < 0.002,
}

# MultiPICo/claude = +0.300
s_mp_cl = 1 - e1_mp_claude["conf"].values / 100
d_mp = e1_mp_claude["dis_mi"].values
rho_sanity_mp_cl, _ = stats.spearmanr(s_mp_cl[np.isfinite(s_mp_cl) & np.isfinite(d_mp)],
                                        d_mp[np.isfinite(s_mp_cl) & np.isfinite(d_mp)])
sanity["rho_Uverbal_dis_MultiPICo_claude"] = {
    "paper_value": 0.300, "computed": float(rho_sanity_mp_cl),
    "match": abs(float(rho_sanity_mp_cl) - 0.300) < 0.002,
}

# Qwen2.5: ΔAsym CSC (main paper: +0.272 p=0.000 or similar from xfamily)
q05_csc = load_e3_model("e3_q0p5b", "CSC")
q32_csc = load_e3_model("e3_q32b",  "CSC")
if q05_csc is not None and q32_csc is not None:
    merged_qw = q05_csc[["item_id", "p_yes", "dis_mi", "y_true"]].copy()
    merged_qw["item_id"] = merged_qw["item_id"].astype(str)
    q32_s = q32_csc[["item_id", "p_yes"]].copy()
    q32_s["item_id"] = q32_s["item_id"].astype(str)
    merged_qw = merged_qw.merge(q32_s.rename(columns={"p_yes": "p_large"}), on="item_id")
    merged_qw = merged_qw.rename(columns={"p_yes": "p_small"})
    t_low  = np.percentile(merged_qw["dis_mi"].values, 33.3)
    t_high = np.percentile(merged_qw["dis_mi"].values, 66.7)
    dlow  = merged_qw[merged_qw["dis_mi"] <= t_low]
    dhigh = merged_qw[merged_qw["dis_mi"] >= t_high]
    da_qw = bootstrap_delta_asym(dlow, dhigh, "p_small", "p_large", n_boot=2000)
    sanity["delta_asym_qwen25_CSC"] = {
        "computed": da_qw,
        "note": "Qwen2.5 0.5B→32B; CSC; paper reports ΔAsym from xfamily summary"
    }
    print(f"  Sanity ΔAsym Qwen2.5 CSC: {da_qw['delta_asym']:+.3f} p={da_qw['boot_p_le0']:.3f}")

print(f"  Sanity rho CSC/claude: {rho_sanity_csc_cl:.3f} (paper: 0.115, match={sanity['rho_Uverbal_dis_CSC_claude']['match']})")
print(f"  Sanity rho MP/claude:  {rho_sanity_mp_cl:.3f} (paper: 0.300, match={sanity['rho_Uverbal_dis_MultiPICo_claude']['match']})")

with open(OUT / "sanity_check.json", "w") as f:
    json.dump({"env": env_log, "sanity": sanity}, f, indent=2, default=str)
print("  -> Saved sanity_check.json")




# ===========================================================================
# ENV LOG
# ===========================================================================

with open(OUT / "env_log_revision.json", "w") as f:
    json.dump(env_log, f, indent=2)
print("  -> Saved env_log_revision.json")

print("\n=== All revision analyses complete ===")
print(f"Output directory: {OUT}")
for fpath in sorted(OUT.glob("*.json")):
    print(f"  {fpath.name}")
