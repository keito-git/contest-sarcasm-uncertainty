"""
run_c2b_v4.py — C2b corrected to match C1 pipeline exactly.

Fixes over C2b_v3:
1. tercile_labels: rank-based (rankdata / (n+1), digitize at [1/3, 2/3])
   — exactly as in analyze_round1.py (the script that produced C1_DASYM_CI.json).
2. bal_acc: same two-class loop as C1.
3. das_ym_fn: same sign convention as C1.
4. Observed ΔAsym: matches C1_DASYM_CI.json exactly for all 6 corpora.
5. Null strata: also use rank-based tercile_labels on the resampled dis.
6. Added: noise-corrected excess = observed − null_mean, with a
   bootstrap-over-simulations 95% CI.

Output: round1_analysis/C2b_v4.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
PROJ = WORK_DIR
REGEN   = PROJ / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
MP_HF   = PROJ / "results" / "revision_gpu_2026-09" / "multipico_hf_raw.parquet"
SARCASM = PROJ / "results" / "sarcasm"
OUT     = PROJ / "results" / "revision_gpu_2026-09" / "round1_analysis"

SEED   = 42
N_SIM  = 1000
CORPORA = ["CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]


# ── Helpers exactly matching analyze_round1.py ──────────────────────────── #
def tercile_labels(a: np.ndarray) -> np.ndarray:
    """0=low, 1=mid, 2=high disagreement (rank-based, same as C1 pipeline)."""
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append(float((pred[m] == c).mean()))
    return float(np.mean(recs)) if recs else float("nan")


def das_ym_fn(pl: np.ndarray, ps: np.ndarray, y: np.ndarray, terc: np.ndarray) -> float:
    """ΔAsym = Δbal_acc(low) − Δbal_acc(high)."""
    dal = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
    dah = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])
    return dal - dah


def das_ym_from_probs(p_small: np.ndarray, p_large: np.ndarray,
                       y: np.ndarray, terc: np.ndarray) -> float:
    ps_bin = (p_small >= 0.5).astype(int)
    pl_bin = (p_large >= 0.5).astype(int)
    return das_ym_fn(pl_bin, ps_bin, y, terc)


# ── Data loaders ────────────────────────────────────────────────────────── #
def load_c7_pair(corpus: str):
    df0  = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
    df32 = pd.read_csv(REGEN / f"c7_q32b_{corpus}.csv")
    m = df0.merge(df32, on="item_id", suffixes=("_0", "_32"))
    p0   = m["p_yes_0"].to_numpy(float)
    p32  = m["p_yes_32"].to_numpy(float)
    y    = m["y_true_0"].to_numpy(int)
    dis  = m["dis_mi_0"].to_numpy(float)
    return p0, p32, y, dis, m["item_id"].astype(str).tolist()


def load_human(corpus: str, item_ids: list[str]):
    """Return (p_human, k_i) arrays aligned to item_ids."""
    if corpus == "CSC":
        csc = pd.read_parquet(SARCASM / "csc_aggregated.parquet")
        csc["p_human"] = csc["mean_eval_norm"].to_numpy(float)
        csc["k_i"]     = csc["n_evaluators"].to_numpy(int)
        h = csc.set_index("item_id")[["p_human", "k_i"]].reindex(item_ids)
    elif corpus == "EPIC":
        ep = pd.read_parquet(SARCASM / "epic_aggregated.parquet")
        ep["p_human"] = ep["p_ironic"].to_numpy(float)
        ep["k_i"]     = ep["n_annotators"].to_numpy(int)
        h = ep.set_index("item_id")[["p_human", "k_i"]].reindex(item_ids)
    else:
        lang = corpus.split("_", 1)[1]
        mh = pd.read_parquet(MP_HF)
        ml = mh[mh["language"] == lang].copy()
        g = ml.groupby("reply_id").agg(
            k_i=("label", "count"), p_human=("label", "mean")
        ).reset_index()
        g["reply_id"] = g["reply_id"].astype(str)
        g["item_id"]  = f"mp_{lang}_" + g["reply_id"]
        h = g.set_index("item_id")[["p_human", "k_i"]].reindex(item_ids)
    return h["p_human"].to_numpy(float), h["k_i"].fillna(1).to_numpy(int)


def resamp_dis(p_resamp: np.ndarray) -> np.ndarray:
    """Binary entropy (same formula for all corpora in the null sim)."""
    pc = np.clip(p_resamp, 1e-12, 1 - 1e-12)
    return -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))


# ── Main ────────────────────────────────────────────────────────────────── #
def main() -> None:
    rng = np.random.default_rng(SEED)
    results: dict = {}

    # Load C1 reference values
    with open(OUT / "C1_DASYM_CI.json") as f:
        c1 = json.load(f)
    c1_vals = {r["corpus"]: r["asym_acc"]
               for r in c1["results"] if r["subset"] == "full"}

    for corpus in CORPORA:
        print(f"{corpus} ...", flush=True)
        p0, p32, y_obs, dis_obs, item_ids = load_c7_pair(corpus)
        p_human, k_i = load_human(corpus, item_ids)
        n = len(y_obs)

        # ── Observed ΔAsym (must equal C1) ──
        terc_obs = tercile_labels(dis_obs)
        observed = das_ym_from_probs(p0, p32, y_obs, terc_obs)
        c1_ref = c1_vals.get(corpus, float("nan"))
        match_ok = abs(observed - c1_ref) < 1e-4
        print(f"  observed={observed:.4f}  C1={c1_ref:.4f}  match={'OK' if match_ok else 'MISMATCH'}")

        # ── BAC calibration for null model ──
        ps_bin = (p0  >= 0.5).astype(int)
        pl_bin = (p32 >= 0.5).astype(int)
        bac_0b  = bal_acc(ps_bin, y_obs)
        bac_32b = bal_acc(pl_bin, y_obs)

        # ── Null simulation (1000 runs) ──
        null_vals: list[float] = []
        for _ in range(N_SIM):
            # Step 1: latent true label
            z = (rng.uniform(size=n) < p_human).astype(int)
            # Step 2: null model predictions (constant BAC per size)
            correct_0  = rng.uniform(size=n) < bac_0b
            correct_32 = rng.uniform(size=n) < bac_32b
            pred_0_f  = np.where(correct_0,  z, 1 - z).astype(float)
            pred_32_f = np.where(correct_32, z, 1 - z).astype(float)
            # Step 3: resample reference labels (k_i votes from p_human)
            counts_yes = rng.binomial(k_i, p_human)
            p_resamp   = np.where(k_i > 0, counts_yes / k_i, p_human)
            y_ref      = (p_resamp >= 0.5).astype(int)  # tie → 1 (same as pipeline)
            # Step 4: dis from resampled proportions, strata via rank-based tercile
            dis_r  = resamp_dis(p_resamp)
            terc_r = tercile_labels(dis_r)   # ← rank-based, same as C1
            # Step 5: ΔAsym
            da = das_ym_fn(pred_32_f, pred_0_f, y_ref, terc_r)
            null_vals.append(da)

        null_arr   = np.array(null_vals)
        null_mean  = float(null_arr.mean())
        null_sd    = float(null_arr.std())
        null_ci95  = [float(np.percentile(null_arr, 2.5)),
                      float(np.percentile(null_arr, 97.5))]
        emp_p      = float(np.mean(null_arr >= observed))

        # ── Noise-corrected excess = observed − null_mean ──
        excess = observed - null_mean
        # Bootstrap-over-simulations CI: subsample null_vals to estimate
        # variability of the mean; CI on (observed - mean_of_subsample)
        B_exc = 2000
        boot_excess: list[float] = []
        idx_sim = np.arange(N_SIM)
        for _ in range(B_exc):
            bi = rng.choice(idx_sim, size=N_SIM, replace=True)
            boot_excess.append(observed - float(null_arr[bi].mean()))
        exc_ci95 = [float(np.percentile(boot_excess, 2.5)),
                    float(np.percentile(boot_excess, 97.5))]

        # ── Asymmetric positive control (only for validation, 200 sims) ──
        gamma = 0.15
        pc_vals: list[float] = []
        for _ in range(200):
            z       = (rng.uniform(size=n) < p_human).astype(int)
            counts_yes = rng.binomial(k_i, p_human)
            p_resamp   = np.where(k_i > 0, counts_yes / k_i, p_human)
            dis_r      = resamp_dis(p_resamp)
            dis_norm   = dis_r / (dis_r.max() + 1e-12)
            y_ref      = (p_resamp >= 0.5).astype(int)
            terc_r     = tercile_labels(dis_r)
            # 0.5B constant, 32B gets extra accuracy on low-dis items
            bac_32_item = np.clip(bac_32b + gamma * (1 - dis_norm), 0, 1)
            correct_0   = rng.uniform(size=n) < bac_0b
            correct_32  = rng.uniform(size=n) < bac_32_item
            pred_0_f    = np.where(correct_0,  z, 1 - z).astype(float)
            pred_32_f   = np.where(correct_32, z, 1 - z).astype(float)
            pc_vals.append(das_ym_fn(pred_32_f, pred_0_f, y_ref, terc_r))

        pc_arr = np.array(pc_vals)

        results[corpus] = {
            "n":                  n,
            "observed_dasym":     round(observed, 4),
            "c1_ref_dasym":       round(c1_ref, 4),
            "c1_match":           match_ok,
            "terc_dist_obs":      np.bincount(terc_obs).tolist(),
            "bac_0p5b":           round(bac_0b, 4),
            "bac_32b":            round(bac_32b, 4),
            "n_sims":             N_SIM,
            "null_mean":          round(null_mean, 4),
            "null_sd":            round(null_sd, 4),
            "null_ci95":          [round(x, 4) for x in null_ci95],
            "empirical_p":        round(emp_p, 4),
            "noise_corrected_excess":       round(excess, 4),
            "noise_corrected_excess_ci95":  [round(x, 4) for x in exc_ci95],
            "positive_control_gamma":        gamma,
            "positive_control_n_sims":       200,
            "positive_control_asym_mean":    round(float(pc_arr.mean()), 4),
            "positive_control_asym_p97.5":   round(float(np.percentile(pc_arr, 97.5)), 4),
        }
        print(
            f"  null_mean={null_mean:.4f} SD={null_sd:.4f} "
            f"emp_p={emp_p:.4f} excess={excess:.4f} exc_CI={exc_ci95}",
            flush=True,
        )

    out_data = {
        "analysis":    "C2b_v4: corrected annotator-noise null (rank-based terciles, C1-aligned)",
        "description": (
            "Null: latent z_i~Bern(p_i), constant-BAC model, reference=majority_vote of k_i "
            "resampled from p_i. Strata from resampled dis via rank-based tercile_labels "
            "(same as C1 pipeline). Observed ΔAsym = C1_DASYM_CI.json values. "
            "Noise-corrected excess = observed - null_mean; CI from bootstrap over simulations."
        ),
        "seed":        SEED,
        "n_sims":      N_SIM,
        "bootstrap_excess_n": 2000,
        "generated":   datetime.now(timezone.utc).isoformat(),
        "per_corpus":  results,
    }
    out_path = OUT / "C2b_v4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
