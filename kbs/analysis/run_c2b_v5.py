"""
run_c2b_v5.py — C2b noise null; the CSC null uses actual Likert-rating resampling.

Changes over v4 (run_c2b_v4.py):
-----------------------------------------------------------------------
1. CSC null: annotator Likert ratings (1-6) are resampled WITH REPLACEMENT
   from data/raw/csc_data_full.csv (per item, k_i actual annotators).
   dis_mi is recomputed from the resampled ratings using the EXACT pipeline
   formula: sample_var(ddof=1) / max(0.5, (mean-1)*(6-mean)), clipped [0,1].
   Reference label: int(mean_resamp >= 3.5), i.e. the pipeline rule
   (mean_eval_norm = (mean_eval-1)/5 >= 0.5  <=>  mean_eval >= 3.5).
   v4 approximated CSC by resampling binary votes from p_human=mean_eval_norm
   and used binary entropy for dis; that approximation is removed here.

2. Binary corpora (EPIC, MultiPICo): unchanged from v4 (binomial resample +
   binary entropy = matches the pipeline's H(p)/log(2) exactly). Verified.

3. Report per corpus:
   - observed_dasym (= C1)
   - null_mean, null_sd, null_ci95 (95th-percentile band of simulations)
   - empirical_p (fraction of sims >= observed)
   - null_explained_frac (= null_mean / observed_dasym; "share explained by null")
   - noise_corrected_excess (= observed - null_mean; NOTE: no CI reported —
     the bootstrap-over-simulations CI is NOT a CI for the observed effect)
   - p_holm (Holm correction over 6-corpus family)
   - positive_control: mean and p97.5 (validation that gamma=0.15 shifts null up)

4. p < 1/1001 (= 0 hits in 1000 sims) is reported as "< 0.001" via
   empirical_p = 0.0000; the JSON stores the raw float (0.0).

Output: round1_analysis/C2b_v5.json
Seed: 42 throughout.  N_SIM = 1000 (main) + 200 (positive control).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ── Paths ────────────────────────────────────────────────────────────────── #
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
PROJ    = WORK_DIR
REGEN   = PROJ / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
MP_HF   = PROJ / "results" / "revision_gpu_2026-09" / "multipico_hf_raw.parquet"
SARCASM = PROJ / "results" / "sarcasm"
CSC_RAW = PROJ / "data" / "raw" / "csc_data_full.csv"
OUT     = PROJ / "results" / "revision_gpu_2026-09" / "round1_analysis"

SEED    = 42
N_SIM   = 1000
CORPORA = ["CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]

# CSC Likert scale: 1 to 6
CSC_SCALE_MIN: float = 1.0
CSC_SCALE_MAX: float = 6.0
CSC_BINARY_THRESH: float = 3.5   # mean >= 3.5  <==>  mean_norm >= 0.5


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


def das_ym_fn(pl: np.ndarray, ps: np.ndarray,
              y: np.ndarray, terc: np.ndarray) -> float:
    """ΔAsym = Δbal_acc(low) − Δbal_acc(high)."""
    dal = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
    dah = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])
    return dal - dah


def das_ym_from_probs(p_small: np.ndarray, p_large: np.ndarray,
                      y: np.ndarray, terc: np.ndarray) -> float:
    ps_bin = (p_small >= 0.5).astype(int)
    pl_bin = (p_large >= 0.5).astype(int)
    return das_ym_fn(pl_bin, ps_bin, y, terc)


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    corrected = np.array(p_values, float)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, float(p_values[idx]) * (n - rank))
    # enforce monotonicity
    for i in range(1, n):
        corrected[order[i]] = max(corrected[order[i]], corrected[order[i - 1]])
    return corrected.tolist()


# ── CSC Likert resampling helpers ────────────────────────────────────────── #
def load_csc_likert_per_item(item_ids: list[str]) -> dict[str, np.ndarray]:
    """Load per-item Likert scores from csc_data_full.csv.

    Returns {item_id: np.ndarray of int scores (1-6)}.
    item_id = f"{global_context_name}_{global_speaker}" (same as csc_aggregated).
    """
    raw = pd.read_csv(CSC_RAW)
    raw["item_id"] = (raw["global_context_name"].astype(str)
                      + "_" + raw["global_speaker"].astype(str))
    grp = (raw.groupby("item_id")["sarcasm_score_by_evaluator"]
             .apply(lambda x: x.to_numpy(dtype=int))
             .to_dict())
    # reindex to item_ids (all should be present)
    result = {}
    for iid in item_ids:
        scores = grp.get(iid)
        if scores is None or len(scores) == 0:
            # fallback: single vote = 3 (mid-scale), contributes 0 dis
            result[iid] = np.array([3], dtype=int)
        else:
            result[iid] = scores
    return result


def csc_dis_mi_from_scores(scores: np.ndarray) -> float:
    """Compute CSC dis_mi from an array of 1-6 Likert scores.

    Formula (pipeline-exact):
        var  = sample_variance(scores, ddof=1)
        denom = clip((mean - 1) * (6 - mean), 0.5, None)
        dis_mi = clip(var / denom, 0.0, 1.0)
    """
    if len(scores) == 1:
        return 0.0
    me = float(scores.mean())
    var = float(scores.var(ddof=1))
    denom = max(0.5, (me - CSC_SCALE_MIN) * (CSC_SCALE_MAX - me))
    return float(np.clip(var / denom, 0.0, 1.0))


def csc_y_ref_from_scores(scores: np.ndarray) -> int:
    """Binary reference label from resampled Likert scores.

    Pipeline rule: y_true = int(mean_eval_norm >= 0.5)
                           = int((mean_eval - 1) / 5 >= 0.5)
                           = int(mean_eval >= 3.5)
    Tie: mean_eval = 3.5 → y_ref = 1 (>= is strict >=, so 3.5 → True).
    """
    return int(float(scores.mean()) >= CSC_BINARY_THRESH)


# ── Data loaders ────────────────────────────────────────────────────────── #
def load_c7_pair(corpus: str):
    """Load 0.5B and 32B predictions for a corpus; return aligned arrays."""
    df0  = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
    df32 = pd.read_csv(REGEN / f"c7_q32b_{corpus}.csv")
    m = df0.merge(df32, on="item_id", suffixes=("_0", "_32"))
    p0   = m["p_yes_0"].to_numpy(float)
    p32  = m["p_yes_32"].to_numpy(float)
    y    = m["y_true_0"].to_numpy(int)
    dis  = m["dis_mi_0"].to_numpy(float)
    return p0, p32, y, dis, m["item_id"].astype(str).tolist()


def load_human_binary(corpus: str, item_ids: list[str]):
    """Return (p_human, k_i) for binary corpora (EPIC, MultiPICo_*).

    p_human = proportion of annotators who voted ironic/sarcastic.
    k_i     = number of annotators per item.
    """
    if corpus == "EPIC":
        ep = pd.read_parquet(SARCASM / "epic_aggregated.parquet")
        ep["p_human"] = ep["p_ironic"].to_numpy(float)
        ep["k_i"]     = ep["n_annotators"].to_numpy(int)
        h = ep.set_index("item_id")[["p_human", "k_i"]].reindex(item_ids)
    elif corpus.startswith("MultiPICo_"):
        lang = corpus.split("_", 1)[1]
        mh = pd.read_parquet(MP_HF)
        ml = mh[mh["language"] == lang].copy()
        g = ml.groupby("reply_id").agg(
            k_i=("label", "count"), p_human=("label", "mean")
        ).reset_index()
        g["reply_id"] = g["reply_id"].astype(str)
        g["item_id"]  = f"mp_{lang}_" + g["reply_id"]
        h = g.set_index("item_id")[["p_human", "k_i"]].reindex(item_ids)
    else:
        raise ValueError(f"Not a binary corpus: {corpus}")
    return h["p_human"].to_numpy(float), h["k_i"].fillna(1).to_numpy(int)


def binary_entropy_arr(p: np.ndarray) -> np.ndarray:
    """Binary entropy H(p) / log(2) in bits — matches EPIC/MultiPICo pipeline."""
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    return -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))


def load_csc_agg(item_ids: list[str]):
    """Return (p_human, k_i) for CSC from csc_aggregated.parquet.

    p_human = mean_eval_norm = (mean_eval - 1) / 5  (used for latent z).
    k_i     = n_evaluators.
    """
    csc = pd.read_parquet(SARCASM / "csc_aggregated.parquet")
    csc["p_human"] = csc["mean_eval_norm"].to_numpy(float)
    csc["k_i"]     = csc["n_evaluators"].to_numpy(int)
    h = csc.set_index("item_id")[["p_human", "k_i"]].reindex(item_ids)
    return h["p_human"].to_numpy(float), h["k_i"].fillna(4).to_numpy(int)


# ── Main ────────────────────────────────────────────────────────────────── #
def main() -> None:
    rng = np.random.default_rng(SEED)

    # Load C1 reference values (observed ΔAsym)
    with open(OUT / "C1_DASYM_CI.json") as f:
        c1 = json.load(f)
    c1_vals = {r["corpus"]: r["asym_acc"]
               for r in c1["results"] if r["subset"] == "full"}

    # Per-corpus results
    per_corpus: dict = {}
    emp_p_list: list[float] = []  # for Holm correction

    for corpus in CORPORA:
        print(f"\n{corpus} ...", flush=True)
        p0, p32, y_obs, dis_obs, item_ids = load_c7_pair(corpus)
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
        print(f"  bac_0.5b={bac_0b:.4f}  bac_32b={bac_32b:.4f}", flush=True)

        # ── Corpus-specific data loading ──
        if corpus == "CSC":
            # Latent-label probability (for model prediction generation)
            p_human, _ = load_csc_agg(item_ids)
            # Per-item Likert scores (for dis_mi and y_ref resampling)
            likert_lookup = load_csc_likert_per_item(item_ids)
            likert_arrays = [likert_lookup[iid] for iid in item_ids]
        else:
            p_human, k_i = load_human_binary(corpus, item_ids)

        # ── Null simulation (N_SIM runs) ──
        null_vals: list[float] = []
        for _ in range(N_SIM):
            # Step 1: latent true label z_i ~ Bern(p_i)
            z = (rng.uniform(size=n) < p_human).astype(int)

            # Step 2: null model predictions (constant BAC per size)
            correct_0  = rng.uniform(size=n) < bac_0b
            correct_32 = rng.uniform(size=n) < bac_32b
            pred_0_f  = np.where(correct_0,  z, 1 - z).astype(float)
            pred_32_f = np.where(correct_32, z, 1 - z).astype(float)

            # Step 3+4: reference label and dis from resampled annotations
            if corpus == "CSC":
                # Resample Likert ratings per item; recompute dis_mi (pipeline-exact)
                dis_r  = np.empty(n, dtype=float)
                y_ref  = np.empty(n, dtype=int)
                for i, scores_i in enumerate(likert_arrays):
                    resamp = rng.choice(scores_i, size=len(scores_i), replace=True)
                    dis_r[i] = csc_dis_mi_from_scores(resamp)
                    y_ref[i] = csc_y_ref_from_scores(resamp)
            else:
                # Binary corpora: resample k_i votes; binary entropy matches pipeline
                counts_yes = rng.binomial(k_i, p_human)
                p_resamp   = np.where(k_i > 0, counts_yes / k_i, p_human)
                y_ref      = (p_resamp >= 0.5).astype(int)   # tie → 1
                dis_r      = binary_entropy_arr(p_resamp)

            terc_r = tercile_labels(dis_r)   # rank-based, same as C1

            # Step 5: ΔAsym
            da = das_ym_fn(pred_32_f, pred_0_f, y_ref, terc_r)
            null_vals.append(da)

        null_arr  = np.array(null_vals)
        null_mean = float(null_arr.mean())
        null_sd   = float(null_arr.std())
        null_ci95 = [float(np.percentile(null_arr, 2.5)),
                     float(np.percentile(null_arr, 97.5))]
        emp_p     = float(np.mean(null_arr >= observed))
        emp_p_list.append(emp_p)

        excess    = observed - null_mean
        expl_frac = (null_mean / observed) if abs(observed) > 1e-9 else float("nan")

        print(f"  null_mean={null_mean:.4f} SD={null_sd:.4f} "
              f"emp_p={emp_p:.4f} null%={expl_frac*100:.1f}%", flush=True)

        # ── Positive control (gamma=0.15, 200 sims) ──
        gamma = 0.15
        pc_vals: list[float] = []
        for _ in range(200):
            z = (rng.uniform(size=n) < p_human).astype(int)

            if corpus == "CSC":
                dis_r_pc = np.empty(n, dtype=float)
                y_ref_pc = np.empty(n, dtype=int)
                for i, scores_i in enumerate(likert_arrays):
                    resamp = rng.choice(scores_i, size=len(scores_i), replace=True)
                    dis_r_pc[i] = csc_dis_mi_from_scores(resamp)
                    y_ref_pc[i] = csc_y_ref_from_scores(resamp)
            else:
                counts_yes = rng.binomial(k_i, p_human)
                p_resamp   = np.where(k_i > 0, counts_yes / k_i, p_human)
                y_ref_pc   = (p_resamp >= 0.5).astype(int)
                dis_r_pc   = binary_entropy_arr(p_resamp)

            dis_norm     = dis_r_pc / (dis_r_pc.max() + 1e-12)
            y_ref_pc_arr = np.asarray(y_ref_pc)
            terc_r_pc    = tercile_labels(dis_r_pc)

            bac_32_item = np.clip(bac_32b + gamma * (1 - dis_norm), 0.0, 1.0)
            correct_0   = rng.uniform(size=n) < bac_0b
            correct_32  = rng.uniform(size=n) < bac_32_item
            pred_0_f    = np.where(correct_0,  z, 1 - z).astype(float)
            pred_32_f   = np.where(correct_32, z, 1 - z).astype(float)
            pc_vals.append(das_ym_fn(pred_32_f, pred_0_f, y_ref_pc_arr, terc_r_pc))

        pc_arr = np.array(pc_vals)
        print(f"  pos_ctrl: mean={pc_arr.mean():.4f} p97.5={np.percentile(pc_arr,97.5):.4f}",
              flush=True)

        per_corpus[corpus] = {
            "n":                 n,
            "observed_dasym":    round(observed, 4),
            "c1_ref_dasym":      round(c1_ref, 4),
            "c1_match":          match_ok,
            "terc_dist_obs":     np.bincount(terc_obs).tolist(),
            "bac_0p5b":          round(bac_0b, 4),
            "bac_32b":           round(bac_32b, 4),
            "n_sims":            N_SIM,
            "null_mean":         round(null_mean, 4),
            "null_sd":           round(null_sd, 4),
            "null_ci95":         [round(x, 4) for x in null_ci95],
            "empirical_p":       round(emp_p, 4),
            "null_explained_frac":       round(expl_frac, 4) if not (isinstance(expl_frac, float) and expl_frac != expl_frac) else None,
            "noise_corrected_excess":    round(excess, 4),
            # NOTE: No CI on excess — bootstrap-over-simulations CI is NOT a
            # CI for the observed effect. See N3-3(a).
            "positive_control_gamma":       gamma,
            "positive_control_n_sims":      200,
            "positive_control_asym_mean":   round(float(pc_arr.mean()), 4),
            "positive_control_asym_p97_5":  round(float(np.percentile(pc_arr, 97.5)), 4),
        }

    # ── Holm correction over 6 corpora ──
    holm_p = holm_correct(emp_p_list)
    for corpus, hp in zip(CORPORA, holm_p):
        if corpus in per_corpus:
            per_corpus[corpus]["p_holm"]         = round(hp, 4)
            per_corpus[corpus]["sig_05_holm"]    = bool(hp < 0.05)
            per_corpus[corpus]["holm_family_n"]  = len(CORPORA)

    # ── Summary table ──
    print("\n=== Summary ===", flush=True)
    header = f"{'Corpus':<20} {'Obs':>6} {'NullM':>6} {'NullSD':>6} "
    header += f"{'95% band':>18} {'EmpP':>6} {'Null%':>6} {'pHolm':>7} {'Sig':>4}"
    print(header)
    for corpus in CORPORA:
        v = per_corpus[corpus]
        band = f"[{v['null_ci95'][0]:.3f},{v['null_ci95'][1]:.3f}]"
        sig = "*" if v.get("sig_05_holm", False) else ""
        frac = v.get("null_explained_frac")
        frac_s = f"{frac*100:.0f}%" if frac is not None else "—"
        print(f"{corpus:<20} {v['observed_dasym']:>6.4f} {v['null_mean']:>6.4f} "
              f"{v['null_sd']:>6.4f} {band:>18} {v['empirical_p']:>6.4f} "
              f"{frac_s:>6} {v.get('p_holm', float('nan')):>7.4f} {sig:>4}")

    # ── Save ──
    out_data = {
        "analysis":   "C2b_v5: CSC null uses Likert-rating resampling (pipeline-exact dis_mi)",
        "description": (
            "Null: latent z_i~Bern(p_i), constant-BAC model. "
            "CSC: k_i Likert ratings (1-6) are resampled WITH REPLACEMENT from "
            "csc_data_full.csv; dis_mi = sample_var(ddof=1)/max(0.5,(mean-1)(6-mean)) "
            "clipped [0,1]; y_ref = int(mean_resamp >= 3.5) [pipeline rule]. "
            "EPIC/MultiPICo: k_i binary votes resampled from p_human; "
            "dis_mi = H(p_resamp)/log(2) [matches pipeline binary entropy]. "
            "Terciles: rank-based (same as C1). "
            "CI of noise-corrected excess is NOT reported: bootstrap-over-simulations "
            "CI is not a CI for the observed effect. "
            "Holm family: 6 corpora."
        ),
        "seed":      SEED,
        "n_sims":    N_SIM,
        "csc_null_note": (
            "v4 approximated CSC by resampling binary votes from p_human=mean_eval_norm "
            "and using binary entropy. v5 uses actual Likert-score resampling and the "
            "pipeline's relative-variance formula, closing the approximation."
        ),
        "generated": datetime.now(timezone.utc).isoformat(),
        "per_corpus": per_corpus,
    }

    out_path = OUT / "C2b_v5.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
