"""
analyze_round2_v3.py — noise-null, per-tercile soft-label, AUBAC and CSC
floor-sensitivity analyses.

Tasks
-----
C2b_v3  : Annotator-noise null simulation (superseded by run_c2b_v4/v5.py).
           - Latent label z_i ~ Bernoulli(p_i) from real annotator proportions.
           - Null model with CONSTANT balanced accuracy across strata.
           - Observed reference = majority_vote of k_i resampled annotators.
           - Strata from resampled dis. >=1000 simulations per corpus.
           - Report null mean/SD/95% band, empirical p for observed ΔAsym.
           - Positive control: accuracy deliberately correlated with stratum.

C2c_v3  : Tercile-level soft-label table.
           - Brier and JSD per tercile (low/mid/high) × size (0.5B / 32B)
             with bootstrap CIs (B=4000).
           - Per-tercile change drop_t = mean_metric(0.5B,t) - mean_metric(32B,t)
             with CIs. Explicit note: whether high-dis gets worse (overconfidence).

C7_v3   : AUBAC with unified definition and bootstrap CIs.
           - Definition: budget=0=all 3B, budget=1=all 32B; AUBAC = mean/21.
           - Routing: contraction (|p3B-p0.5B| desc → 32B),
                      confidence (|2*p3B-1| asc → 32B),
                      oracle (dis asc → 32B, i.e. low-dis gets 32B),
                      random (mean over 100 seeds, 95% band).
           - Bootstrap CIs (B=2000) for (contraction - random) and
             (confidence - random) per corpus. State B.
           - All 6 corpora.

C9_floor : CSC denominator-floor sensitivity.
           - ΔAsym (0.5B → 32B) on CSC with:
             (a) standard (include all, floor=0.5 on denominator),
             (b) drop floored items (557 items where (me-1)*(6-me)=0),
             (c) drop top-clipped items (dis_raw > 1 before clip, 166 items,
                 but only those where denom>=0.5 so they were affected by
                 the upper clip rather than lower),
             (d) drop both.
           - Bootstrap CIs (B=4000) for each condition.

Outputs (all _v3 suffix, in round1_analysis/):
  C2b_v3.json
  C2c_v3.json   (tercile-level soft-label table — separate from C2a)
  C7_v3.json
  C9_floor_v3.json

Usage
-----
  python kbs/analysis/analyze_round2_v3.py

CPU only. No GPU or API calls. Expected runtime: ~20 min.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────── #
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR
REGEN = BASE / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
OUT = BASE / "results" / "revision_gpu_2026-09" / "round1_analysis"
SARCASM = BASE / "results" / "sarcasm"
MP_HF = BASE / "results" / "revision_gpu_2026-09" / "multipico_hf_raw.parquet"

# ── Constants ────────────────────────────────────────────────────────────── #
SEED = 42
N_SIM = 1000   # C2b null simulations per corpus
B_BOOT_C2 = 4000  # bootstrap samples for C2c
B_BOOT_C7 = 2000  # bootstrap samples for C7
B_BOOT_C9 = 4000  # bootstrap samples for C9
N_RAND_SEEDS_C7 = 100  # random baseline seeds for AUBAC


# ── Helpers ─────────────────────────────────────────────────────────────── #
def _sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def balanced_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Balanced accuracy: (TPR + TNR) / 2. Handles all-one-class strata."""
    mask0 = (y_true == 0)
    mask1 = (y_true == 1)
    if mask0.sum() == 0 or mask1.sum() == 0:
        return float(np.mean(y_pred == y_true))
    tnr = float(np.mean(y_pred[mask0] == 0))
    tpr = float(np.mean(y_pred[mask1] == 1))
    return (tpr + tnr) / 2.0


def dasym_from_arrays(
    p_small: np.ndarray,
    p_large: np.ndarray,
    y_ref: np.ndarray,
    terc: np.ndarray,
) -> float:
    """ΔAsym = (gain_low) − (gain_high).

    gain_t = BAC(large, y_ref, t) − BAC(small, y_ref, t)
    terc ∈ {0=low, 1=mid, 2=high}
    """
    def gain(t: int) -> float:
        mask = terc == t
        if mask.sum() < 2:
            return 0.0
        pred_s = (p_small[mask] >= 0.5).astype(int)
        pred_l = (p_large[mask] >= 0.5).astype(int)
        return balanced_accuracy(pred_l, y_ref[mask]) - balanced_accuracy(pred_s, y_ref[mask])

    return gain(0) - gain(2)


def tercile_assign(dis: np.ndarray) -> np.ndarray:
    """Assign tercile labels {0=low, 1=mid, 2=high} by quantile."""
    q33 = np.percentile(dis, 100 / 3)
    q67 = np.percentile(dis, 200 / 3)
    terc = np.where(dis <= q33, 0, np.where(dis <= q67, 1, 2))
    return terc.astype(int)


def brier_score(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    """Mean Brier score: E[(p_pred - p_true)^2]."""
    return float(np.mean((p_pred - p_true) ** 2))


def jsd_score(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    """Mean JSD for Bernoulli distributions.

    JSD(p || q) = 0.5 * [H(m) - 0.5 * H(p) - 0.5 * H(q)]
    where m = (p+q)/2 and H(x) = -x log2(x) - (1-x) log2(1-x).
    """
    def h(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))

    m = (p_pred + p_true) / 2
    jsd = h(m) - 0.5 * h(p_pred) - 0.5 * h(p_true)
    return float(np.mean(np.clip(jsd, 0, None)))


# ── Data loaders ────────────────────────────────────────────────────────── #
def load_csc_per_item() -> pd.DataFrame:
    """Load CSC aggregated and compute dis_mi/y_true as in the pipeline."""
    csc_agg = pd.read_parquet(SARCASM / "csc_aggregated.parquet").copy()
    me = csc_agg["mean_eval"].to_numpy(float)
    var = csc_agg["std_eval"].to_numpy(float) ** 2
    denom = np.clip((me - 1.0) * (6.0 - me), 0.5, None)
    csc_agg["dis_mi"] = np.clip(var / denom, 0.0, 1.0)
    csc_agg["y_true"] = (csc_agg["mean_eval_norm"].to_numpy(float) >= 0.5).astype(int)
    # p_human = mean_eval_norm (already in [0,1])
    csc_agg["p_human"] = csc_agg["mean_eval_norm"].to_numpy(float)
    # k_i = number of evaluators per item
    csc_agg["k_i"] = csc_agg["n_evaluators"].to_numpy(int)
    return csc_agg[["item_id", "dis_mi", "y_true", "p_human", "k_i"]].copy()


def load_epic_per_item() -> pd.DataFrame:
    """Load EPIC aggregated with k_i (n_annotators) and p_human."""
    epic = pd.read_parquet(SARCASM / "epic_aggregated.parquet").copy()
    epic["dis_mi"] = epic["disagreement"].to_numpy(float)
    epic["y_true"] = (epic["p_ironic"].to_numpy(float) >= 0.5).astype(int)
    epic["p_human"] = epic["p_ironic"].to_numpy(float)
    epic["k_i"] = epic["n_annotators"].to_numpy(int)
    return epic[["item_id", "dis_mi", "y_true", "p_human", "k_i"]].copy()


def load_multipico_per_item(lang: str) -> pd.DataFrame:
    """Load MultiPICo per-item from HF raw parquet, aggregating by reply_id."""
    mp_hf = pd.read_parquet(MP_HF)
    mp_lang = mp_hf[mp_hf["language"] == lang].copy()
    # Aggregate per reply_id
    grp = mp_lang.groupby("reply_id").agg(
        n_annotators=("label", "count"),
        p_ironic=("label", "mean"),
    ).reset_index()
    grp["reply_id"] = grp["reply_id"].astype(str)
    grp["item_id"] = "mp_" + lang + "_" + grp["reply_id"]
    # Binary entropy for dis
    p = grp["p_ironic"].to_numpy(float)
    p_c = np.clip(p, 1e-12, 1 - 1e-12)
    grp["dis_mi"] = -(p_c * np.log2(p_c) + (1 - p_c) * np.log2(1 - p_c))
    grp["y_true"] = (p >= 0.5).astype(int)
    grp["p_human"] = p
    grp["k_i"] = grp["n_annotators"].to_numpy(int)
    return grp[["item_id", "dis_mi", "y_true", "p_human", "k_i"]].copy()


def load_c7_pair(corpus: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_0p5b, p_32b, y_true) for a corpus from the c7 CSVs."""
    df0 = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
    df32 = pd.read_csv(REGEN / f"c7_q32b_{corpus}.csv")
    # Align on item_id
    merged = df0.merge(df32, on="item_id", suffixes=("_0", "_32"))
    p0 = merged["p_yes_0"].to_numpy(float)
    p32 = merged["p_yes_32"].to_numpy(float)
    y = merged["y_true_0"].to_numpy(int)
    return p0, p32, y


def load_c7_dis(corpus: str) -> np.ndarray:
    """Load dis_mi from the 0.5B CSV (same for all sizes per item)."""
    df = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
    return df["dis_mi"].to_numpy(float)


def load_c7_p3b(corpus: str) -> np.ndarray:
    """Load p_yes from 3B CSV for AUBAC contraction signal."""
    df = pd.read_csv(REGEN / f"c7_q3b_{corpus}.csv")
    # Ensure alignment with 0.5B order
    df0 = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
    merged = df0[["item_id"]].merge(df[["item_id", "p_yes"]], on="item_id")
    return merged["p_yes"].to_numpy(float)


CORPORA = ["CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]

# ── C2b: Corrected annotator-noise null ─────────────────────────────────── #
def run_c2b_v3() -> dict:
    """
    Null: when model accuracy is CONSTANT across strata, can annotator sampling
    noise in the reference labels alone produce ΔAsym > 0 of the observed size?

    Construction per corpus:
    - Latent true label z_i ~ Bernoulli(p_i) per simulation.
    - Null model of size s: P(correct | item) = BAC_obs_s (constant across items).
      Prediction: y_pred_i = z_i with prob BAC_s, else (1-z_i).
    - Observed reference label: y_ref_i = majority_vote(Bernoulli(p_i)^{k_i}).
      Tie rule: ties (equal yes/no) → y_ref = 1 (same as pipeline >= 0.5 rule).
    - dis_resampled_i = H(p_resamp_i) for binary, relative-variance formula for CSC.
      Strata from dis_resampled (same tercile formula).
    - ΔAsym computed using y_ref as ground truth.

    Positive control: accuracy = BAC_obs_s + gamma * (1 - dis_normalized_i).
    If gamma > 0, model is MORE accurate on low-dis items → should produce ΔAsym > 0.
    """
    print("C2b_v3: annotator-noise null simulation ...", flush=True)
    rng = np.random.default_rng(SEED)

    def _load_human(corpus: str) -> pd.DataFrame:
        if corpus == "CSC":
            return load_csc_per_item()
        elif corpus == "EPIC":
            return load_epic_per_item()
        else:
            lang = corpus.split("_", 1)[1]
            return load_multipico_per_item(lang)

    results: dict = {}
    for corpus in CORPORA:
        print(f"  {corpus} ...", flush=True)
        p0, p32, y_obs = load_c7_pair(corpus)
        dis_obs = load_c7_dis(corpus)
        terc_obs = tercile_assign(dis_obs)
        observed_dasym = dasym_from_arrays(p0, p32, y_obs, terc_obs)

        # Observed BAC for calibration of null model
        y_pred_0 = (p0 >= 0.5).astype(int)
        y_pred_32 = (p32 >= 0.5).astype(int)
        bac_0b_obs = balanced_accuracy(y_pred_0, y_obs)
        bac_32b_obs = balanced_accuracy(y_pred_32, y_obs)

        # Human data for resampling
        hdf = _load_human(corpus)
        # Align item order to c7 CSV order
        df0 = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
        hdf = hdf.set_index("item_id")
        item_ids = df0["item_id"].astype(str).tolist()
        hdf = hdf.reindex(item_ids)
        p_human = hdf["p_human"].to_numpy(float)
        k_i = hdf["k_i"].to_numpy(int)

        # Fill missing (should be none, but safety)
        valid = ~(np.isnan(p_human) | (k_i < 1))
        if valid.sum() < len(p_human) * 0.95:
            print(f"    WARNING: {(~valid).sum()} items without human data", flush=True)

        n = len(p_human)
        is_csc = (corpus == "CSC")

        def compute_dis_from_resamp(p_resamp: np.ndarray) -> np.ndarray:
            """Compute dis from resampled p values using corpus-appropriate formula."""
            if is_csc:
                # CSC: use same relative-variance formula
                # p_resamp is proportion of >=4 votes (binary on resampled scores)
                # But in resampled setting, scores are binary (0/1 based on p_human >= 4?)
                # Actually for the null model, we treat CSC as binary as well
                # (the original scores are converted to binary via >=0.5 threshold)
                # For the resampling, use binary entropy on the binary proportion
                p_c = np.clip(p_resamp, 1e-12, 1 - 1e-12)
                return -(p_c * np.log2(p_c) + (1 - p_c) * np.log2(1 - p_c))
            else:
                p_c = np.clip(p_resamp, 1e-12, 1 - 1e-12)
                return -(p_c * np.log2(p_c) + (1 - p_c) * np.log2(1 - p_c))

        def majority_vote(counts_yes: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
            """Majority vote; tie (equal yes/no) → 1 (>= 0.5 rule)."""
            p_resamp = counts_yes / k_vec
            return (p_resamp >= 0.5).astype(int)

        # ---- Null simulation ----
        null_dasym_list: list[float] = []
        for sim_idx in range(N_SIM):
            # Step 1: Latent true labels
            z = (rng.uniform(size=n) < p_human).astype(int)

            # Step 2: Null model predictions (constant BAC)
            def null_pred(bac: float, z_lat: np.ndarray) -> np.ndarray:
                correct = rng.uniform(size=n) < bac
                pred = np.where(correct, z_lat, 1 - z_lat)
                return pred.astype(int)

            pred_0b = null_pred(bac_0b_obs, z)
            pred_32b = null_pred(bac_32b_obs, z)
            p_null_0b = pred_0b.astype(float)
            p_null_32b = pred_32b.astype(float)

            # Step 3: Resample reference labels
            counts_yes = rng.binomial(k_i, p_human)
            p_resamp = np.where(k_i > 0, counts_yes / k_i, p_human)
            y_ref = majority_vote(counts_yes, k_i)

            # Step 4: Compute dis from resampled proportions
            dis_resamp = compute_dis_from_resamp(p_resamp)
            terc_resamp = tercile_assign(dis_resamp)

            # Step 5: ΔAsym
            da = dasym_from_arrays(p_null_0b, p_null_32b, y_ref, terc_resamp)
            null_dasym_list.append(da)

        null_arr = np.array(null_dasym_list)
        p_emp = float(np.mean(null_arr >= observed_dasym))

        # ---- Positive control: accuracy depends on stratum ----
        # gamma makes model more accurate on low-dis items
        gamma = 0.15
        pc_null: list[float] = []
        for sim_idx in range(200):
            z = (rng.uniform(size=n) < p_human).astype(int)
            counts_yes = rng.binomial(k_i, p_human)
            p_resamp = np.where(k_i > 0, counts_yes / k_i, p_human)
            dis_resamp = compute_dis_from_resamp(p_resamp)
            dis_norm = dis_resamp / (dis_resamp.max() + 1e-12)
            # bac depends on stratum (low-dis → higher bac)
            bac_item_0b = np.clip(bac_0b_obs + gamma * (1 - dis_norm), 0, 1)
            bac_item_32b = np.clip(bac_32b_obs + gamma * (1 - dis_norm), 0, 1)
            correct_0b = rng.uniform(size=n) < bac_item_0b
            correct_32b = rng.uniform(size=n) < bac_item_32b
            pred_0b_pc = np.where(correct_0b, z, 1 - z).astype(float)
            pred_32b_pc = np.where(correct_32b, z, 1 - z).astype(float)
            y_ref = majority_vote(counts_yes, k_i)
            terc_resamp = tercile_assign(dis_resamp)
            da = dasym_from_arrays(pred_0b_pc, pred_32b_pc, y_ref, terc_resamp)
            pc_null.append(da)

        pc_arr = np.array(pc_null)

        results[corpus] = {
            "n": int(n),
            "observed_dasym": round(float(observed_dasym), 4),
            "bac_0p5b_obs": round(float(bac_0b_obs), 4),
            "bac_32b_obs": round(float(bac_32b_obs), 4),
            "n_sims": N_SIM,
            "null_mean": round(float(null_arr.mean()), 4),
            "null_sd": round(float(null_arr.std()), 4),
            "null_ci95": [
                round(float(np.percentile(null_arr, 2.5)), 4),
                round(float(np.percentile(null_arr, 97.5)), 4),
            ],
            "empirical_p_null_ge_observed": round(float(p_emp), 4),
            "positive_control_gamma": gamma,
            "positive_control_n_sims": 200,
            "positive_control_mean": round(float(pc_arr.mean()), 4),
            "positive_control_p97.5": round(float(np.percentile(pc_arr, 97.5)), 4),
            "note": (
                "Null: constant BAC model + resampled reference labels. "
                "Positive control: BAC increases by gamma*(1-dis_norm) for lower-dis items."
            ),
        }
        print(
            f"    observed={observed_dasym:.4f}  null_mean={null_arr.mean():.4f}  "
            f"p={p_emp:.4f}  pc_mean={pc_arr.mean():.4f}",
            flush=True,
        )

    return {
        "analysis": "C2b_v3: corrected annotator-noise null simulation",
        "description": (
            "Tests whether annotator sampling noise in the REFERENCE labels alone "
            "can produce ΔAsym>0. Null: constant-BAC model calibrated to observed "
            "overall BAC per size; reference labels resampled from real p_human and k_i; "
            "strata from resampled dis. Tie rule: p_resamp>=0.5 → y_ref=1."
        ),
        "seed": SEED,
        "n_sims": N_SIM,
        "per_corpus": results,
    }


# ── C2c: Tercile-level soft-label table with CIs ─────────────────────────── #
def run_c2c_v3() -> dict:
    """Re-express C2a data: Brier/JSD per tercile × size, with bootstrap CIs.

    Uses the existing c7 CSV files (p_yes = model probability) and human
    p_human distributions for Brier/JSD target.
    """
    print("C2c_v3: tercile-level soft-label table ...", flush=True)
    rng = np.random.default_rng(SEED)

    def _p_human_array(corpus: str, item_ids: list[str]) -> np.ndarray:
        """Return p_human aligned to item_ids order."""
        if corpus == "CSC":
            hdf = load_csc_per_item().set_index("item_id")
            return hdf.reindex(item_ids)["p_human"].to_numpy(float)
        elif corpus == "EPIC":
            hdf = load_epic_per_item().set_index("item_id")
            return hdf.reindex(item_ids)["p_human"].to_numpy(float)
        else:
            lang = corpus.split("_", 1)[1]
            hdf = load_multipico_per_item(lang).set_index("item_id")
            return hdf.reindex(item_ids)["p_human"].to_numpy(float)

    results: dict = {}
    for corpus in CORPORA:
        print(f"  {corpus} ...", flush=True)
        df0 = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
        df32 = pd.read_csv(REGEN / f"c7_q32b_{corpus}.csv")
        merged = df0.merge(df32, on="item_id", suffixes=("_0", "_32"))
        item_ids = merged["item_id"].astype(str).tolist()
        p0 = merged["p_yes_0"].to_numpy(float)
        p32 = merged["p_yes_32"].to_numpy(float)
        dis = merged["dis_mi_0"].to_numpy(float)
        terc = tercile_assign(dis)
        p_human = _p_human_array(corpus, item_ids)
        n = len(p0)

        tercile_results: dict = {}
        for t_label, t_idx in [("low", 0), ("mid", 1), ("high", 2)]:
            mask = terc == t_idx
            if mask.sum() < 5:
                tercile_results[t_label] = None
                continue
            ph_t = p_human[mask]
            p0_t = p0[mask]
            p32_t = p32[mask]
            nt = int(mask.sum())

            # Observed values
            brier_0b = brier_score(p0_t, ph_t)
            brier_32b = brier_score(p32_t, ph_t)
            jsd_0b = jsd_score(p0_t, ph_t)
            jsd_32b = jsd_score(p32_t, ph_t)
            drop_brier = brier_0b - brier_32b  # positive = 32B improved
            drop_jsd = jsd_0b - jsd_32b

            # Bootstrap CIs for each quantity
            idx_all = np.arange(nt)
            boot_b0, boot_b32, boot_j0, boot_j32 = [], [], [], []
            boot_db, boot_dj = [], []
            for _ in range(B_BOOT_C2):
                bi = rng.choice(idx_all, size=nt, replace=True)
                boot_b0.append(brier_score(p0_t[bi], ph_t[bi]))
                boot_b32.append(brier_score(p32_t[bi], ph_t[bi]))
                boot_j0.append(jsd_score(p0_t[bi], ph_t[bi]))
                boot_j32.append(jsd_score(p32_t[bi], ph_t[bi]))
                boot_db.append(boot_b0[-1] - boot_b32[-1])
                boot_dj.append(boot_j0[-1] - boot_j32[-1])

            def _ci(arr: list[float]) -> list[float]:
                return [round(float(np.percentile(arr, 2.5)), 4),
                        round(float(np.percentile(arr, 97.5)), 4)]

            tercile_results[t_label] = {
                "n": nt,
                "brier_0p5b": round(brier_0b, 4),
                "brier_0p5b_ci95": _ci(boot_b0),
                "brier_32b": round(brier_32b, 4),
                "brier_32b_ci95": _ci(boot_b32),
                "drop_brier": round(drop_brier, 4),
                "drop_brier_ci95": _ci(boot_db),
                "jsd_0p5b": round(jsd_0b, 4),
                "jsd_0p5b_ci95": _ci(boot_j0),
                "jsd_32b": round(jsd_32b, 4),
                "jsd_32b_ci95": _ci(boot_j32),
                "drop_jsd": round(drop_jsd, 4),
                "drop_jsd_ci95": _ci(boot_dj),
                "high_dis_32b_worse_brier": bool(brier_32b > brier_0b) if t_idx == 2 else None,
                "high_dis_32b_worse_jsd": bool(jsd_32b > jsd_0b) if t_idx == 2 else None,
            }

        results[corpus] = {
            "n": n,
            "tercile_scores": tercile_results,
        }

    return {
        "analysis": "C2c_v3: tercile-level soft-label Brier/JSD with bootstrap CIs",
        "description": (
            "Brier score and JSD between model probability and human proportion p_human, "
            "per tercile (low/mid/high dis), per model size (0.5B / 32B). "
            "drop_t = metric(0.5B, t) - metric(32B, t); positive = 32B improved. "
            "Bootstrap CIs (B=4000). high_dis_32b_worse_brier: True if 32B Brier > 0.5B Brier."
        ),
        "bootstrap_n": B_BOOT_C2,
        "bootstrap_seed": SEED,
        "per_corpus": results,
    }


# ── C7: AUBAC with unified definition and bootstrap CIs ─────────────────── #
def run_c7_v3() -> dict:
    """Unified AUBAC with bootstrap CIs for contraction-random and conf-random."""
    print("C7_v3: AUBAC with bootstrap CIs ...", flush=True)
    rng = np.random.default_rng(SEED)

    budgets = np.linspace(0, 1, 21)

    def aubac_from_schedule(
        p0b: np.ndarray,
        p3b: np.ndarray,
        p32b: np.ndarray,
        y: np.ndarray,
        dis: np.ndarray,
        routing: str,
        budget_grid: np.ndarray,
    ) -> float:
        """Compute AUBAC = mean balanced accuracy over budget grid.

        budget=0: all items use 3B.
        budget=1: all items use 32B.
        At budget b: top b*n items (by routing score) use 32B, rest use 3B.
        routing: 'contraction' | 'confidence' | 'oracle' | 'random_seed'
        """
        n = len(y)
        if routing == "contraction":
            score = np.abs(p3b - p0b)  # high = large capacity gap → route to 32B
        elif routing == "confidence":
            score = -np.abs(2 * p3b - 1)  # low confidence → route to 32B (high score = low conf)
        elif routing == "oracle":
            score = -dis  # low dis → route to 32B (high score = low dis)
        else:
            raise ValueError(f"Unknown routing: {routing}")

        order_32b = np.argsort(-score)  # highest score gets 32B first

        balacc_list = []
        for b in budget_grid:
            n32 = int(round(b * n))
            use_32b = np.zeros(n, dtype=bool)
            if n32 > 0:
                use_32b[order_32b[:n32]] = True
            p_pred = np.where(use_32b, p32b, p3b)
            y_pred = (p_pred >= 0.5).astype(int)
            balacc_list.append(balanced_accuracy(y_pred, y))

        return float(np.mean(balacc_list))

    results: dict = {}
    for corpus in CORPORA:
        print(f"  {corpus} ...", flush=True)
        df0 = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
        df3 = pd.read_csv(REGEN / f"c7_q3b_{corpus}.csv")
        df32 = pd.read_csv(REGEN / f"c7_q32b_{corpus}.csv")
        merged = (
            df0[["item_id", "p_yes", "dis_mi", "y_true"]]
            .merge(df3[["item_id", "p_yes"]].rename(columns={"p_yes": "p3b"}), on="item_id")
            .merge(df32[["item_id", "p_yes"]].rename(columns={"p_yes": "p32b"}), on="item_id")
        )
        p0b = merged["p_yes"].to_numpy(float)
        p3b = merged["p3b"].to_numpy(float)
        p32b = merged["p32b"].to_numpy(float)
        y = merged["y_true"].to_numpy(int)
        dis = merged["dis_mi"].to_numpy(float)
        n = len(y)

        # Observed routing scores
        aubac_cont = aubac_from_schedule(p0b, p3b, p32b, y, dis, "contraction", budgets)
        aubac_conf = aubac_from_schedule(p0b, p3b, p32b, y, dis, "confidence", budgets)
        aubac_oracle = aubac_from_schedule(p0b, p3b, p32b, y, dis, "oracle", budgets)

        # Random baseline: mean + 95% band over 100 seeds
        rand_aucs: list[float] = []
        for s in range(N_RAND_SEEDS_C7):
            rng_rand = np.random.default_rng(s)
            rand_score = rng_rand.permutation(n).astype(float)  # random order = random score
            balacc_list = []
            for b in budgets:
                n32 = int(round(b * n))
                use_32b = np.zeros(n, dtype=bool)
                if n32 > 0:
                    use_32b[:n32] = True  # first n32 in random order get 32B
                balacc_list.append(
                    balanced_accuracy(
                        (np.where(use_32b[np.argsort(rand_score)], p32b, p3b) >= 0.5).astype(int),
                        y,
                    )
                )
            rand_aucs.append(float(np.mean(balacc_list)))
        rand_mean = float(np.mean(rand_aucs))
        rand_p25 = float(np.percentile(rand_aucs, 2.5))
        rand_p975 = float(np.percentile(rand_aucs, 97.5))

        # Bootstrap CIs for (contraction - random) and (confidence - random)
        idx_all = np.arange(n)
        boot_delta_cont: list[float] = []
        boot_delta_conf: list[float] = []
        for _ in range(B_BOOT_C7):
            bi = rng.choice(idx_all, size=n, replace=True)
            p0b_b = p0b[bi]
            p3b_b = p3b[bi]
            p32b_b = p32b[bi]
            y_b = y[bi]
            dis_b = dis[bi]
            # Single random seed for bootstrap (faster)
            rng_br = np.random.default_rng(int(rng.integers(1000000)))
            rand_perm = rng_br.permutation(len(bi)).astype(float)
            rand_balacc: list[float] = []
            for bval in budgets:
                n32 = int(round(bval * len(bi)))
                use_32b = np.zeros(len(bi), dtype=bool)
                if n32 > 0:
                    use_32b[:n32] = True
                rand_balacc.append(
                    balanced_accuracy(
                        (np.where(use_32b[np.argsort(rand_perm)], p32b_b, p3b_b) >= 0.5).astype(int),
                        y_b,
                    )
                )
            rand_b = float(np.mean(rand_balacc))
            cont_b = aubac_from_schedule(p0b_b, p3b_b, p32b_b, y_b, dis_b, "contraction", budgets)
            conf_b = aubac_from_schedule(p0b_b, p3b_b, p32b_b, y_b, dis_b, "confidence", budgets)
            boot_delta_cont.append(cont_b - rand_b)
            boot_delta_conf.append(conf_b - rand_b)

        def _ci2(arr: list[float]) -> list[float]:
            return [round(float(np.percentile(arr, 2.5)), 4),
                    round(float(np.percentile(arr, 97.5)), 4)]

        results[corpus] = {
            "n": n,
            "aubac_random_mean": round(rand_mean, 4),
            "aubac_random_ci95_band": [round(rand_p25, 4), round(rand_p975, 4)],
            "aubac_contraction": round(aubac_cont, 4),
            "aubac_confidence": round(aubac_conf, 4),
            "aubac_oracle_low_dis": round(aubac_oracle, 4),
            "aubac_3b_only": round(
                balanced_accuracy((p3b >= 0.5).astype(int), y), 4
            ),
            "aubac_32b_only": round(
                balanced_accuracy((p32b >= 0.5).astype(int), y), 4
            ),
            "delta_contraction_minus_random": round(aubac_cont - rand_mean, 4),
            "delta_contraction_minus_random_ci95": _ci2(boot_delta_cont),
            "delta_confidence_minus_random": round(aubac_conf - rand_mean, 4),
            "delta_confidence_minus_random_ci95": _ci2(boot_delta_conf),
            "n_random_seeds": N_RAND_SEEDS_C7,
            "bootstrap_n": B_BOOT_C7,
        }
        print(
            f"    cont={aubac_cont:.4f}  conf={aubac_conf:.4f}  "
            f"oracle={aubac_oracle:.4f}  rand={rand_mean:.4f}",
            flush=True,
        )

    return {
        "analysis": "C7_v3: AUBAC unified definition with bootstrap CIs",
        "definition": (
            "budget=0=all_3B, budget=1=all_32B. AUBAC=mean/21pts. "
            "Contraction: high |p3B-p0.5B| -> 32B. Confidence: low |2*p3B-1| -> 32B. "
            "Oracle: low dis -> 32B. Random: mean of 100 seeds."
        ),
        "bootstrap_n": B_BOOT_C7,
        "n_random_seeds": N_RAND_SEEDS_C7,
        "seed": SEED,
        "per_corpus": results,
    }


# ── C9_floor: CSC denominator-floor sensitivity ─────────────────────────── #
def run_c9_floor_v3() -> dict:
    """
    CSC ΔAsym sensitivity to the denominator floor in dis_mi.

    Conditions:
      standard : all 7036 items, floor=0.5 on denominator (as per pipeline).
      drop_denom_floor: drop 557 items where (me-1)*(6-me) < 0.5 (mean=1 or 6).
                        These items have dis_mi=0 (var=0 too); dropping them
                        removes unanimously-agreed items from the low-dis stratum.
      drop_top_clip: drop 166 items where dis_raw = var/denom > 1 (clipped to 1
                     by the upper clip). These are high-dis items at the ceiling.
                     Note: only items where denom >= 0.5 AND var/denom > 1.
      drop_both: drop union of the above two sets.
    """
    print("C9_floor_v3: CSC floor sensitivity ...", flush=True)
    rng = np.random.default_rng(SEED)

    csc_agg = pd.read_parquet(SARCASM / "csc_aggregated.parquet").copy()
    me = csc_agg["mean_eval"].to_numpy(float)
    var_arr = csc_agg["std_eval"].to_numpy(float) ** 2
    denom_raw = (me - 1.0) * (6.0 - me)
    denom_floored = np.clip(denom_raw, 0.5, None)
    dis_raw = var_arr / denom_floored
    dis_standard = np.clip(dis_raw, 0.0, 1.0)

    # Flag floored items
    is_denom_floored = denom_raw < 0.5  # 557 items
    is_top_clipped = dis_raw > 1.0       # items where raw dis > 1 (clipped to 1)
    # Note: is_top_clipped includes is_denom_floored cases where var>0,
    # but we showed all denom_floored items have var=0 so dis_raw=0, not >1.
    # So is_top_clipped is purely the 114 items where var/denom > 1 and denom>=0.5.
    print(
        f"  denom_floored: {is_denom_floored.sum()}, top_clipped: {is_top_clipped.sum()}",
        flush=True,
    )

    csc_agg["dis_standard"] = dis_standard
    csc_agg["y_true"] = (csc_agg["mean_eval_norm"].to_numpy(float) >= 0.5).astype(int)
    csc_agg["is_denom_floored"] = is_denom_floored
    csc_agg["is_top_clipped"] = is_top_clipped

    # Load model predictions
    df0 = pd.read_csv(REGEN / "c7_q0p5b_CSC.csv")
    df32 = pd.read_csv(REGEN / "c7_q32b_CSC.csv")
    merged_model = df0.merge(df32, on="item_id", suffixes=("_0", "_32"))

    # Merge with CSC annotations
    merged = merged_model.merge(
        csc_agg[["item_id", "dis_standard", "y_true", "is_denom_floored", "is_top_clipped"]],
        on="item_id",
        how="inner",
    )
    p0 = merged["p_yes_0"].to_numpy(float)
    p32 = merged["p_yes_32"].to_numpy(float)
    y = merged["y_true"].to_numpy(int)
    dis = merged["dis_standard"].to_numpy(float)
    is_df = merged["is_denom_floored"].to_numpy(bool)
    is_tc = merged["is_top_clipped"].to_numpy(bool)
    n_total = len(y)

    conditions = {
        "standard": ~np.zeros(n_total, dtype=bool),  # all items
        "drop_denom_floor": ~is_df,
        "drop_top_clip": ~is_tc,
        "drop_both": ~(is_df | is_tc),
    }

    results: dict = {}
    for cond_name, keep_mask in conditions.items():
        p0_c = p0[keep_mask]
        p32_c = p32[keep_mask]
        y_c = y[keep_mask]
        dis_c = dis[keep_mask]
        nc = int(keep_mask.sum())
        if nc < 100:
            results[cond_name] = {"n": nc, "skipped": True}
            continue

        terc_c = tercile_assign(dis_c)
        obs_dasym = dasym_from_arrays(p0_c, p32_c, y_c, terc_c)

        # Bootstrap CI for ΔAsym
        idx_all = np.arange(nc)
        boot_da: list[float] = []
        for _ in range(B_BOOT_C9):
            bi = rng.choice(idx_all, size=nc, replace=True)
            terc_b = tercile_assign(dis_c[bi])
            da_b = dasym_from_arrays(p0_c[bi], p32_c[bi], y_c[bi], terc_b)
            boot_da.append(da_b)

        ci_lo = round(float(np.percentile(boot_da, 2.5)), 4)
        ci_hi = round(float(np.percentile(boot_da, 97.5)), 4)
        p_pos = float(np.mean(np.array(boot_da) > 0))

        results[cond_name] = {
            "n": nc,
            "n_dropped": int(n_total - nc),
            "dasym": round(float(obs_dasym), 4),
            "ci95": [ci_lo, ci_hi],
            "ci_excludes_zero": bool(ci_lo > 0),
            "bootstrap_p_positive": round(p_pos, 4),
        }
        print(
            f"  {cond_name}: n={nc}  ΔAsym={obs_dasym:.4f}  CI=[{ci_lo},{ci_hi}]",
            flush=True,
        )

    return {
        "analysis": "C9_floor_v3: CSC denominator-floor sensitivity",
        "description": (
            "ΔAsym (0.5B→32B) on CSC under 4 conditions. "
            "denom_floor: 557 items with (me-1)*(6-me)<0.5 (mean_eval=1 or 6, unanimous). "
            "top_clip: items where var/denom > 1 before upper clip to 1 (166 items). "
            "Note: all denom_floored items have var=0 so dis=0 regardless of floor value; "
            "the 'floor sensitivity' here tests whether excluding unanimous items changes ΔAsym."
        ),
        "bootstrap_n": B_BOOT_C9,
        "seed": SEED,
        "n_total_CSC": int(n_total),
        "n_denom_floored": int(is_df.sum()),
        "n_top_clipped": int(is_tc.sum()),
        "per_condition": results,
    }




# ── Main ────────────────────────────────────────────────────────────────── #
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT}", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"Host: {socket.gethostname()}", flush=True)
    print(f"Platform: {platform.platform()}", flush=True)
    print("=" * 60, flush=True)

    # Run analyses
    c2b = run_c2b_v3()
    c2c = run_c2c_v3()
    c7 = run_c7_v3()
    c9 = run_c9_floor_v3()

    # Write JSONs
    for name, data in [
        ("C2b_v3.json", c2b),
        ("C2c_v3.json", c2c),
        ("C7_v3.json", c7),
        ("C9_floor_v3.json", c9),
    ]:
        out_path = OUT / name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
