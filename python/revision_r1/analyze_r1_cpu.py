"""
analyze_r1_cpu.py -- KBS R1 revision (KNOSYS-D-26-23467): CPU-only additional analyses.

Every number added to the revised manuscript in response to the reviewers is produced
here from the per-item result files of the regenerated Qwen2.5 sweep (C7/C8) and the
frontier audit (E1). No value is typed by hand.

Analyses (reviewer item in brackets):
  S1  stratum composition: sizes, distinct dis values, ties, class balance   [R1-2, R2-4, R2-5]
  S2  alternative stratifications (unanimous vs split; median split)        [R2-4]
  S3  ΔAsym CI that includes stratum-definition uncertainty                  [R1-5]
  S4  ΔAsym after reweighting strata to a common (label x length) mix        [R1-2]
  S5  full-capacity-range interaction (slope difference) and common-ref
      lower endpoints; 0.5B output bias and parse failures                  [R2-3]
  S6  CSC disagreement bound: sample vs population variance                  [R2-5]
  S7  400-item subsampling distribution of ΔAsym (explains 0.242 -> 0.136)   [R1-1]
  S8  escalation predictors under one pre-specified orientation; all-item
      (deployment-observable) AUROC; CSC reversal diagnostics               [R2-6, R2-7]
  S9  global Holm over secondary tests                                       [R1-5]
  S10 frontier audit: balanced accuracy / per-class errors with CIs, ECE,
      complete single-signal table                                          [R1-5, R2-9]
  S11 sharpness vs calibration on high-disagreement items                    [R1-5]
  S12 sampling-run variance of ΔAsym from the repeated 0.5B run              [R1-1]

Output: results/revision_r1_2026-11/R1_CPU.json (timestamped if it exists).
Seed 42 throughout.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PROJ = Path(__file__).resolve().parents[2]
REGEN = PROJ / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
SARC = PROJ / "results" / "sarcasm"
MP_HF = PROJ / "results" / "revision_gpu_2026-09" / "multipico_hf_raw.parquet"
CSC_RAW = PROJ / "data" / "raw" / "csc_data_full.csv"
E1 = PROJ / "results" / "llm_e1"
E3 = PROJ / "results" / "llm_e3"
OUT = PROJ / "results" / "revision_r1_2026-11"

SEED = 42
import os
SMOKE = os.environ.get("R1_SMOKE") == "1"
B_TEST = 50 if SMOKE else 4000      # bootstrap resamples for ΔAsym tests (as in the paper)
B_CI = 50 if SMOKE else 2000        # bootstrap resamples for AUROC CIs (as in the paper)
CORPORA = ["CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]
if SMOKE:
    OUT = PROJ / "results" / "revision_r1_2026-11" / "smoke"
SIZES = {"q0p5b": 0.5, "q1p5b": 1.5, "q3b": 3.0, "q7b": 7.0, "q14b": 14.0, "q32b": 32.0}


# ---------------------------------------------------------------------------
# Helpers (identical definitions to the submitted pipeline)
# ---------------------------------------------------------------------------
def tercile_labels(a: np.ndarray) -> np.ndarray:
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])


def bal_acc(pred: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    recs = []
    for c in (0, 1):
        m = y == c
        if m.sum() == 0:
            continue
        if w is None:
            recs.append(float((pred[m] == c).mean()))
        else:
            ww = w[m]
            if ww.sum() <= 0:
                continue
            recs.append(float(np.sum(ww * (pred[m] == c)) / ww.sum()))
    return float(np.mean(recs)) if recs else float("nan")


def dasym(pl, ps, y, low, high, w=None) -> float:
    wl = None if w is None else w[low]
    wh = None if w is None else w[high]
    dl = bal_acc(pl[low], y[low], wl) - bal_acc(ps[low], y[low], wl)
    dh = bal_acc(pl[high], y[high], wh) - bal_acc(ps[high], y[high], wh)
    return dl - dh


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    npos = y.sum()
    if npos == 0 or npos == len(y):
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def boot_auroc(y, s, nb=B_CI, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, n):
            continue
        bs.append(auroc(y[i], s[i]))
    bs = np.asarray(bs)
    a = auroc(y, s)
    # two-sided bootstrap p for H0: AUROC = 0.5
    p2 = float(min(1.0, 2 * min(np.mean(bs <= 0.5), np.mean(bs >= 0.5))))
    return {"AUROC": round(a, 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                           round(float(np.percentile(bs, 97.5)), 4)],
            "p_two_sided_vs_0.5": p2, "n": int(n), "n_pos": int(y.sum())}


def holm(ps: list[float]) -> list[float]:
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    adj = np.empty(n)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, ps[idx] * (n - rank)))
        adj[idx] = run
    return adj.tolist()


def p_floor(p: float, b: int) -> str:
    """Bootstrap p-values cannot be smaller than 1/(B+1)."""
    return f"<{1/(b+1):.5f}" if p < 1 / (b + 1) else f"{p:.4f}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sweep(corpus: str, tags=tuple(SIZES)) -> pd.DataFrame:
    base = None
    for t in tags:
        d = pd.read_csv(REGEN / f"c7_{t}_{corpus}.csv")
        d["item_id"] = d["item_id"].astype(str)
        keep = d[["item_id", "p_yes", "H", "n_valid"]].rename(
            columns={"p_yes": f"p_{t}", "H": f"H_{t}", "n_valid": f"nv_{t}"})
        if base is None:
            base = d[["item_id", "dis_mi", "y_true"]].merge(keep, on="item_id")
        else:
            base = base.merge(keep, on="item_id", how="inner")
    inp = pd.read_csv(REGEN / f"full_input_{corpus}.csv")
    inp["item_id"] = inp["item_id"].astype(str)
    inp["len_words"] = inp["txt"].astype(str).str.split().str.len()
    base = base.merge(inp[["item_id", "len_words"]], on="item_id", how="left")
    return base.reset_index(drop=True)


def load_c8(tag: str, corpus: str, mode: str = "c8") -> pd.Series:
    d = pd.read_csv(REGEN / f"{mode}_{tag}_{corpus}.csv")
    d["item_id"] = d["item_id"].astype(str)
    return d.set_index("item_id")["p_yes"]


def human_votes(corpus: str, ids: list[str]):
    """Per-item annotation data: binary -> (k_i, n_yes); CSC -> list of Likert scores."""
    if corpus == "CSC":
        raw = pd.read_csv(CSC_RAW)
        raw["item_id"] = raw["global_context_name"].astype(str) + "_" + raw["global_speaker"].astype(str)
        g = raw.groupby("item_id")["sarcasm_score_by_evaluator"].apply(lambda x: x.to_numpy(int)).to_dict()
        return [g.get(i, np.array([3])) for i in ids]
    if corpus == "EPIC":
        ep = pd.read_parquet(SARC / "epic_aggregated.parquet")
        ep["item_id"] = ep["item_id"].astype(str)
        h = ep.set_index("item_id").reindex(ids)
        k = h["n_annotators"].to_numpy(int)
        nyes = np.rint(h["p_ironic"].to_numpy(float) * k).astype(int)
        return k, nyes
    lang = corpus.split("_", 1)[1]
    mh = pd.read_parquet(MP_HF)
    ml = mh[mh["language"] == lang]
    g = ml.groupby("reply_id").agg(k=("label", "count"), nyes=("label", "sum")).reset_index()
    g["item_id"] = "mp_" + lang + "_" + g["reply_id"].astype(str)
    h = g.set_index("item_id").reindex(ids)
    return h["k"].to_numpy(int), h["nyes"].to_numpy(int)


def csc_dis(scores: np.ndarray, ddof: int = 1) -> float:
    m = scores.mean()
    var = scores.var(ddof=ddof) if len(scores) > ddof else 0.0
    den = max(0.5, (m - 1) * (6 - m))
    return float(min(1.0, max(0.0, var / den)))


def bin_entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    h = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
    return np.where((p <= 0) | (p >= 1), 0.0, h)


# ---------------------------------------------------------------------------
# Bootstrap ΔAsym test given fixed strata
# ---------------------------------------------------------------------------
def dasym_test(pl, ps, y, low, high, w=None, nb=B_TEST, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = dasym(pl, ps, y, low, high, w)
    bs = np.empty(nb)
    for b in range(nb):
        i = rng.integers(0, n, n)
        bs[b] = dasym(pl[i], ps[i], y[i], low[i], high[i], None if w is None else w[i])
    p = float(np.mean(bs <= 0))
    return {"dasym": round(obs, 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "p_one_sided": p, "p_str": p_floor(p, nb)}


# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"meta": {"utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                          "B_test": B_TEST, "B_ci": B_CI,
                          "script": "python/revision_r1/analyze_r1_cpu.py"}}
    S = {k: {} for k in ["S1_strata", "S2_alt_strat", "S3_strat_uncertainty", "S4_reweight",
                         "S5_full_range", "S6_csc_bound", "S7_sub400", "S8_escalation",
                         "S11_sharpness", "S12_run_variance"]}
    secondary_tests: list[dict] = []

    for corpus in CORPORA:
        print(f"== {corpus}", flush=True)
        df = load_sweep(corpus)
        n = len(df)
        y = df["y_true"].to_numpy(int)
        dis = df["dis_mi"].to_numpy(float)
        terc = tercile_labels(dis)
        low, high = terc == 0, terc == 2
        P = {t: (df[f"p_{t}"].to_numpy(float) >= 0.5).astype(int) for t in SIZES}
        ps, pl = P["q0p5b"], P["q32b"]

        # ---------------- S1 strata composition ----------------
        uniq = np.unique(np.round(dis, 10))
        s1 = {"n": n, "n_distinct_dis": int(len(uniq)),
              "distinct_dis_values": [round(float(u), 4) for u in uniq] if len(uniq) <= 30 else None,
              "frac_dis_zero": round(float(np.mean(dis <= 1e-12)), 4),
              "base_rate": round(float(y.mean()), 4)}
        for lab, m in (("low", low), ("mid", terc == 1), ("high", high)):
            s1[lab] = {"n": int(m.sum()), "dis_min": round(float(dis[m].min()), 4),
                       "dis_max": round(float(dis[m].max()), 4),
                       "n_distinct_dis": int(len(np.unique(np.round(dis[m], 10)))),
                       "pos_rate": round(float(y[m].mean()), 4), "n_pos": int(y[m].sum()),
                       "n_neg": int((1 - y[m]).sum()),
                       "mean_len_words": round(float(df["len_words"].to_numpy()[m].mean()), 2)}
        S["S1_strata"][corpus] = s1

        # ---------------- annotation data ----------------
        ids = df["item_id"].tolist()
        if corpus == "CSC":
            scores = human_votes(corpus, ids)
            k_i = np.array([len(s) for s in scores])
            dis_chk = np.array([csc_dis(s, 1) for s in scores])
            unanimous = np.array([len(np.unique(s)) == 1 for s in scores])
        else:
            k_i, nyes = human_votes(corpus, ids)
            dis_chk = bin_entropy(nyes / k_i)
            unanimous = (nyes == 0) | (nyes == k_i)
        S["S1_strata"][corpus]["pipeline_dis_reproduced_max_abs_diff"] = round(
            float(np.nanmax(np.abs(dis_chk - dis))), 6)
        S["S1_strata"][corpus]["frac_unanimous"] = round(float(unanimous.mean()), 4)

        # ---------------- S2 alternative stratifications ----------------
        s2 = {}
        s2["unanimous_vs_split"] = dasym_test(pl, ps, y, unanimous, ~unanimous)
        s2["unanimous_vs_split"].update({"n_unanimous": int(unanimous.sum()),
                                         "n_split": int((~unanimous).sum()),
                                         "pos_rate_unanimous": round(float(y[unanimous].mean()), 4),
                                         "pos_rate_split": round(float(y[~unanimous].mean()), 4)})
        med = np.median(dis)
        lo_m, hi_m = dis <= med, dis > med
        if hi_m.sum() < 30:  # ties at the median: use rank halves
            r = rankdata(dis) / (n + 1)
            lo_m, hi_m = r <= 0.5, r > 0.5
        s2["median_split"] = dasym_test(pl, ps, y, lo_m, hi_m)
        s2["tercile_primary"] = dasym_test(pl, ps, y, low, high)
        S["S2_alt_strat"][corpus] = s2

        # ---------------- S3 stratum-definition uncertainty ----------------
        rng = np.random.default_rng(SEED)
        bs = np.empty(B_TEST)
        if corpus == "CSC":
            kmax = int(k_i.max())
            M = np.full((n, kmax), np.nan)
            for j, s in enumerate(scores):
                M[j, :len(s)] = s
            col = np.arange(kmax)[None, :]
        for b in range(B_TEST):
            i = rng.integers(0, n, n)
            # re-draw annotators for each sampled item, recompute dis and terciles
            if corpus == "CSC":
                kk = k_i[i][:, None]
                pick = np.floor(rng.random((n, kmax)) * kk).astype(int)
                vals = np.take_along_axis(M[i], pick, axis=1)
                valid = col < kk
                vals = np.where(valid, vals, np.nan)
                mu = np.nanmean(vals, axis=1)
                var = np.where(kk[:, 0] > 1, np.nanvar(vals, axis=1, ddof=1), 0.0)
                den = np.maximum(0.5, (mu - 1) * (6 - mu))
                d_b = np.clip(var / den, 0.0, 1.0)
            else:
                ny_b = rng.binomial(k_i[i], nyes[i] / k_i[i])
                d_b = bin_entropy(ny_b / k_i[i])
            t_b = tercile_labels(d_b)
            bs[b] = dasym(pl[i], ps[i], y[i], t_b == 0, t_b == 2)
        p3 = float(np.mean(bs <= 0))
        S["S3_strat_uncertainty"][corpus] = {
            "dasym_point": round(dasym(pl, ps, y, low, high), 4),
            "mean_boot": round(float(bs.mean()), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "p_one_sided": p3, "p_str": p_floor(p3, B_TEST),
            "note": "item bootstrap + annotator resampling within item; strata re-derived each draw"}

        # ---------------- S4 reweighting to a common label x length mix ----------------
        lenq = pd.qcut(df["len_words"].rank(method="first"), 5, labels=False).to_numpy()
        cell = y * 10 + lenq
        pooled = pd.Series(cell).value_counts(normalize=True)
        w = np.ones(n)
        for m in (low, terc == 1, high):
            vc = pd.Series(cell[m]).value_counts(normalize=True)
            ratio = (pooled / vc).reindex(cell[m]).to_numpy()
            w[m] = np.nan_to_num(ratio, nan=0.0)
        s4 = dasym_test(pl, ps, y, low, high, w=w)
        # length-only: high-dis items reweighted to the length distribution of low-dis items
        S["S4_reweight"][corpus] = {"label_x_length_reweighted": s4,
                                    "note": "post-stratification of each stratum to the pooled "
                                            "(label x length-quintile) distribution; balanced accuracy "
                                            "is already invariant to the within-stratum class mix"}

        # ---------------- S5 full range ----------------
        logs = np.log2([SIZES[t] for t in SIZES])
        logs_c = logs - logs.mean()

        def slope_diff(idx):
            bl = np.array([bal_acc(P[t][idx][low[idx]], y[idx][low[idx]]) for t in SIZES])
            bh = np.array([bal_acc(P[t][idx][high[idx]], y[idx][high[idx]]) for t in SIZES])
            sl = float(np.sum(logs_c * bl) / np.sum(logs_c ** 2))
            sh = float(np.sum(logs_c * bh) / np.sum(logs_c ** 2))
            return sl, sh, sl - sh, bl, bh
        sl, sh, sd, bl, bh = slope_diff(np.arange(n))
        rng = np.random.default_rng(SEED)
        bsd = np.empty(B_TEST)
        for b in range(B_TEST):
            i = rng.integers(0, n, n)
            bsd[b] = slope_diff(i)[2]
        p5 = float(np.mean(bsd <= 0))
        s5 = {"slope_low_per_doubling": round(sl, 4), "slope_high_per_doubling": round(sh, 4),
              "slope_diff": round(sd, 4),
              "slope_diff_ci95": [round(float(np.percentile(bsd, 2.5)), 4), round(float(np.percentile(bsd, 97.5)), 4)],
              "p_one_sided": p5, "p_str": p_floor(p5, B_TEST),
              "balacc_low_by_size": {t: round(float(v), 4) for t, v in zip(SIZES, bl)},
              "balacc_high_by_size": {t: round(float(v), 4) for t, v in zip(SIZES, bh)}}
        # endpoint pairs as primary contrasts
        s5["endpoints"] = {}
        for a in ("q0p5b", "q1p5b", "q3b"):
            s5["endpoints"][f"{a}->q32b"] = dasym_test(P["q32b"], P[a], y, low, high)
        # common reference: smallest size whose low-stratum BA CI lies above 0.5
        ref = None
        for t in SIZES:
            rng = np.random.default_rng(SEED)
            yl, pt = y[low], P[t][low]
            bb = [bal_acc(pt[j], yl[j]) for j in (rng.integers(0, low.sum(), low.sum()) for _ in range(1000))]
            if np.percentile(bb, 2.5) > 0.5:
                ref = t
                break
        s5["common_reference_lower"] = ref
        if ref is not None and ref != "q32b":
            s5["common_reference_test"] = dasym_test(P["q32b"], P[ref], y, low, high)
        # output bias / parse failure per size
        s5["output_profile"] = {t: {"yes_rate": round(float(P[t].mean()), 4),
                                    "frac_n_valid_lt10": round(float((df[f"nv_{t}"] < 10).mean()), 4),
                                    "frac_n_valid_0": round(float((df[f"nv_{t}"] == 0).mean()), 4),
                                    "balacc_low": round(bal_acc(P[t][low], y[low]), 4),
                                    "recall_pos_low": round(float((P[t][low & (y == 1)] == 1).mean()), 4),
                                    "recall_neg_low": round(float((P[t][low & (y == 0)] == 0).mean()), 4)}
                                for t in SIZES}
        # restrict to items with fully parsed 0.5B and 32B outputs
        ok = (df["nv_q0p5b"] == 10).to_numpy() & (df["nv_q32b"] == 10).to_numpy()
        s5["fully_parsed_only"] = dasym_test(pl[ok], ps[ok], y[ok], low[ok], high[ok])
        s5["fully_parsed_only"]["n"] = int(ok.sum())
        S["S5_full_range"][corpus] = s5
        secondary_tests.append({"family": "full_range_slope", "corpus": corpus, "p": p5})

        # ---------------- S6 CSC bound ----------------
        if corpus == "CSC":
            d1 = np.array([csc_dis(s, 1) for s in scores])
            raw1 = np.array([(s.var(ddof=1) if len(s) > 1 else 0.0) / max(0.5, (s.mean() - 1) * (6 - s.mean()))
                             for s in scores])
            d0 = np.array([csc_dis(s, 0) for s in scores])
            raw0 = np.array([s.var(ddof=0) / max(0.5, (s.mean() - 1) * (6 - s.mean())) for s in scores])
            floor_hit = np.array([(s.mean() - 1) * (6 - s.mean()) < 0.5 for s in scores])
            t0 = tercile_labels(d0)
            S["S6_csc_bound"] = {
                "n_items": n,
                "sample_var_ratio_gt1": int((raw1 > 1 + 1e-12).sum()),
                "population_var_ratio_gt1": int((raw0 > 1 + 1e-12).sum()),
                "max_population_ratio_without_floor": round(float(max(
                    s.var(ddof=0) / ((s.mean() - 1) * (6 - s.mean())) for s in scores
                    if (s.mean() - 1) * (6 - s.mean()) > 0)), 6),
                "n_floor_hit": int(floor_hit.sum()),
                "frac_floor_hit": round(float(floor_hit.mean()), 4),
                "spearman_ddof1_vs_ddof0": round(float(spearmanr(d1, d0)[0]), 4),
                "tercile_agreement_ddof1_vs_ddof0": round(float(np.mean(t0 == terc)), 4),
                "dasym_population_variance": dasym_test(pl, ps, y, t0 == 0, t0 == 2),
                "k_i_min": int(k_i.min()), "k_i_max": int(k_i.max()),
                "k_i_median": float(np.median(k_i)),
            }

        # ---------------- S7 400-item subsampling distribution ----------------
        rng = np.random.default_rng(SEED)
        sub = []
        for _ in range(50 if SMOKE else 2000):
            i = rng.choice(n, 400, replace=False)
            t_i = tercile_labels(dis[i])
            sub.append(dasym(pl[i], ps[i], y[i], t_i == 0, t_i == 2))
        sub = np.asarray(sub)
        s7 = {"full_dasym": round(dasym(pl, ps, y, low, high), 4),
              "sub400_mean": round(float(sub.mean()), 4), "sub400_sd": round(float(sub.std()), 4),
              "sub400_p2.5": round(float(np.percentile(sub, 2.5)), 4),
              "sub400_p97.5": round(float(np.percentile(sub, 97.5)), 4)}
        orig = pd.read_csv(REGEN / f"full_input_{corpus}.csv")
        if "in_original_400" in orig.columns and orig["in_original_400"].sum() > 0:
            oid = set(orig.loc[orig["in_original_400"] == 1, "item_id"].astype(str))
            mo = df["item_id"].isin(oid).to_numpy()
            t_o = tercile_labels(dis[mo])
            s7["regen_on_original_400"] = dasym_test(pl[mo], ps[mo], y[mo], t_o == 0, t_o == 2)
            s7["regen_on_original_400"]["n"] = int(mo.sum())
            s7["frac_sub400_ge_0.242"] = round(float(np.mean(sub >= 0.242)), 4)
        S["S7_sub400"][corpus] = s7

        # ---------------- S8 escalation predictors ----------------
        wrong = ps != y
        yfix_all = (wrong & (pl == y)).astype(int)
        sw = np.where(wrong)[0]
        yfix = yfix_all[sw]
        p0 = df["p_q0p5b"].to_numpy(float)
        p3 = df["p_q3b"].to_numpy(float)
        H0 = df["H_q0p5b"].to_numpy(float)
        preds = {
            # every score is oriented a priori so that a larger value means "escalate"
            "contraction_probe": np.abs(p3 - p0),
            "entropy_0p5b": H0,
            "one_minus_dis": 1.0 - dis,
        }
        s8 = {"n_items": n, "n_small_wrong": int(wrong.sum()), "n_fix": int(yfix.sum()),
              "orientation": "pre-specified: larger score = escalate (higher model uncertainty, "
                             "larger capacity gap, lower human disagreement)",
              "on_small_wrong": {}, "all_items": {}}
        for k, v in preds.items():
            s8["on_small_wrong"][k] = boot_auroc(yfix, v[sw])
            s8["all_items"][k] = boot_auroc(yfix_all, v)
        # supervised comparators, cross-fitted on y_fix, natural orientation
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import KFold
        xs = p0[sw]
        calib = np.full(len(sw), np.nan)
        gl = np.full(len(sw), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(xs):
            ir = IsotonicRegression(out_of_bounds="clip").fit(xs[tr], yfix[tr])
            calib[te] = ir.predict(xs[te])
        for tr, te in KFold(5, shuffle=True, random_state=SEED + 1).split(xs):
            ir = IsotonicRegression(out_of_bounds="clip").fit(xs[tr], yfix[tr])
            resid = yfix[te] - ir.predict(xs[te])
            edges = np.quantile(xs[te], np.linspace(0, 1, 11))
            edges[0] -= 1e-9
            bl_ = np.clip(np.digitize(xs[te], edges) - 1, 0, 9)
            for bi in range(10):
                mb = bl_ == bi
                if mb.sum() >= 2:
                    gl[te[mb]] = np.var(resid[mb])
                elif mb.sum() == 1:
                    gl[te[mb]] = resid[mb][0] ** 2
        s8["on_small_wrong"]["supervised_calibrated_confidence"] = boot_auroc(yfix, calib)
        s8["on_small_wrong"]["supervised_grouping_loss_natural"] = boot_auroc(yfix, gl)
        # diversity controls (pairwise vs capacity-gap probe; one-sided bootstrap)
        ctrl_defs = {"phi3_mini": ("q3b", "ll3b"), "smollm2_1p7b": ("q1p5b", "ll1b"),
                     "stablelm_3b": ("q3b", "gm2_2b"), "sampling_noise": ("q0p5b", "q0p5b_s2")}
        s8["diversity"] = {}
        idx = df["item_id"]
        for name, (a, bt) in ctrl_defs.items():
            pa = df[f"p_{a}"].to_numpy(float)
            mode = "c8_noise" if bt == "q0p5b_s2" else "c8"
            pb = load_c8(bt, corpus, mode).reindex(idx).to_numpy(float)
            ctrl = np.abs(pa - pb)[sw]
            prim = preds["contraction_probe"][sw]
            ok_ = ~np.isnan(ctrl)
            rng = np.random.default_rng(SEED)
            m_ = ok_.sum()
            yy, aa, cc = yfix[ok_], prim[ok_], ctrl[ok_]
            diffs = []
            for _ in range(B_TEST):
                i = rng.integers(0, m_, m_)
                if yy[i].sum() in (0, m_):
                    continue
                diffs.append(auroc(yy[i], aa[i]) - auroc(yy[i], cc[i]))
            pdiff = float(np.mean(np.asarray(diffs) <= 0))
            s8["diversity"][name] = {"AUROC_control": round(auroc(yy, cc), 4),
                                     "AUROC_probe": round(auroc(yy, aa), 4),
                                     "p_probe_gt_control": pdiff, "n": int(m_)}
            secondary_tests.append({"family": "capacity_vs_control", "corpus": corpus,
                                    "control": name, "p": pdiff})
        secondary_tests.append({"family": "probe_auroc_vs_0.5", "corpus": corpus,
                                "p": s8["on_small_wrong"]["contraction_probe"]["p_two_sided_vs_0.5"]})
        # CSC reversal diagnostics: error type and confidence of 0.5B errors
        fp = (ps == 1) & (y == 0)
        fn = (ps == 0) & (y == 1)
        s8["error_types"] = {
            "n_false_pos": int(fp.sum()), "n_false_neg": int(fn.sum()),
            "fix_rate_false_pos": round(float(yfix_all[fp].mean()), 4) if fp.sum() else None,
            "fix_rate_false_neg": round(float(yfix_all[fn].mean()), 4) if fn.sum() else None,
            "mean_entropy_false_pos": round(float(H0[fp].mean()), 4) if fp.sum() else None,
            "mean_entropy_false_neg": round(float(H0[fn].mean()), 4) if fn.sum() else None,
            "mean_entropy_fixed": round(float(H0[yfix_all == 1].mean()), 4),
            "mean_entropy_unfixed_errors": round(float(H0[wrong & (yfix_all == 0)].mean()), 4),
            "mean_dis_fixed": round(float(dis[yfix_all == 1].mean()), 4),
            "mean_dis_unfixed_errors": round(float(dis[wrong & (yfix_all == 0)].mean()), 4),
            "mean_contr_fixed": round(float(preds["contraction_probe"][yfix_all == 1].mean()), 4),
            "mean_contr_unfixed_errors": round(float(preds["contraction_probe"][wrong & (yfix_all == 0)].mean()), 4),
        }
        S["S8_escalation"][corpus] = s8

        # ---------------- S11 sharpness on strata ----------------
        if corpus == "CSC":
            ph = np.array([(s.mean() - 1) / 5 for s in scores])
        else:
            ph = nyes / k_i
        s11 = {}
        for lab, m in (("low", low), ("high", high)):
            row = {}
            for t in ("q0p5b", "q32b"):
                pt = df[f"p_{t}"].to_numpy(float)[m]
                row[t] = {"brier_vs_human": round(float(np.mean((pt - ph[m]) ** 2)), 4),
                          "sharpness_mean_abs_p_minus_half": round(float(np.mean(np.abs(pt - 0.5))), 4),
                          "mean_signed_bias_p_minus_ph": round(float(np.mean(pt - ph[m])), 4),
                          "var_term": round(float(np.var(pt - ph[m])), 4),
                          "bias_sq_term": round(float(np.mean(pt - ph[m]) ** 2), 4),
                          "human_sharpness": round(float(np.mean(np.abs(ph[m] - 0.5))), 4)}
            s11[lab] = row
        S["S11_sharpness"][corpus] = s11

        # ---------------- S12 repeated 0.5B run ----------------
        p0b = load_c8("q0p5b_s2", corpus, "c8_noise").reindex(df["item_id"]).to_numpy(float)
        okr = ~np.isnan(p0b)
        ps2 = (p0b >= 0.5).astype(int)
        S["S12_run_variance"][corpus] = {
            "dasym_run1": round(dasym(pl[okr], ps[okr], y[okr], low[okr], high[okr]), 4),
            "dasym_run2": round(dasym(pl[okr], ps2[okr], y[okr], low[okr], high[okr]), 4),
            "item_agreement_0p5b_runs": round(float(np.mean(ps[okr] == ps2[okr])), 4),
            "n": int(okr.sum())}

    # ---------------- AUBAC probe-vs-random p-values (recomputed, C7_v3 definition) -------
    budgets = np.linspace(0, 1, 21)

    def aubac(score, p3b, p32b, y):
        order = np.argsort(-score)
        nn = len(y)
        out = []
        for b in budgets:
            n32 = int(round(b * nn))
            use = np.zeros(nn, bool)
            use[order[:n32]] = True
            out.append(bal_acc((np.where(use, p32b, p3b) >= 0.5).astype(int), y))
        return float(np.mean(out))

    S["S9_aubac"] = {}
    for corpus in CORPORA:
        df = load_sweep(corpus, tags=("q0p5b", "q3b", "q32b"))
        p0 = df["p_q0p5b"].to_numpy(float)
        p3 = df["p_q3b"].to_numpy(float)
        p32 = df["p_q32b"].to_numpy(float)
        y = df["y_true"].to_numpy(int)
        dis = df["dis_mi"].to_numpy(float)
        nn = len(y)
        rng = np.random.default_rng(SEED)
        rand_obs = np.mean([aubac(np.random.default_rng(s).permutation(nn).astype(float), p3, p32, y)
                            for s in range(100)])
        obs = {"contraction": aubac(np.abs(p3 - p0), p3, p32, y),
               "confidence": aubac(-np.abs(2 * p3 - 1), p3, p32, y),
               "one_minus_dis": aubac(-dis, p3, p32, y)}
        boots = {k: [] for k in obs}
        for _ in range(20 if SMOKE else 1000):
            i = rng.integers(0, nn, nn)
            r_b = aubac(rng.permutation(nn).astype(float), p3[i], p32[i], y[i])
            boots["contraction"].append(aubac(np.abs(p3[i] - p0[i]), p3[i], p32[i], y[i]) - r_b)
            boots["confidence"].append(aubac(-np.abs(2 * p3[i] - 1), p3[i], p32[i], y[i]) - r_b)
            boots["one_minus_dis"].append(aubac(-dis[i], p3[i], p32[i], y[i]) - r_b)
        row = {"random_mean": round(float(rand_obs), 4)}
        for k in obs:
            bb = np.asarray(boots[k])
            p2 = float(min(1.0, 2 * min(np.mean(bb <= 0), np.mean(bb >= 0))))
            row[k] = {"aubac": round(obs[k], 4), "delta_vs_random": round(obs[k] - float(rand_obs), 4),
                      "delta_ci95": [round(float(np.percentile(bb, 2.5)), 4), round(float(np.percentile(bb, 97.5)), 4)],
                      "p_two_sided": p2}
        S["S9_aubac"][corpus] = row
        secondary_tests.append({"family": "aubac_probe_vs_random", "corpus": corpus,
                                "p": row["contraction"]["p_two_sided"]})
        print(f"  AUBAC {corpus} done", flush=True)

    # noise-null excess p-values from the submitted analysis (C2b_v5)
    c2b = json.load(open(PROJ / "results/revision_gpu_2026-09/round1_analysis/C2b_v5.json"))
    for corpus in CORPORA:
        pc = c2b["per_corpus"][corpus]
        secondary_tests.append({"family": "noise_null_excess", "corpus": corpus,
                                "p": float(pc["empirical_p"])})

    adj = holm([t["p"] for t in secondary_tests])
    for t, a in zip(secondary_tests, adj):
        t["p_holm_global"] = round(float(a), 4)
        t["sig_global_05"] = bool(a < 0.05)
    S["S9_global_holm"] = {"n_tests": len(secondary_tests), "tests": secondary_tests,
                           "note": "single Holm family over all secondary tests: probe AUROC vs 0.5 (6), "
                                   "capacity-vs-control pairwise (24), AUBAC probe vs random (6), "
                                   "noise-null excess (6), full-range slope difference (6)"}

    # ---------------- S10 frontier audit ----------------
    S["S10_frontier"] = frontier_audit()

    res.update(S)
    out = OUT / "R1_CPU.json"
    if out.exists():
        out = OUT / f"R1_CPU_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("wrote", out)


def frontier_audit() -> dict:
    """Balanced accuracy, per-class error, ECE, and all signal correlations for the
    400-item frontier audit (CSC, EPIC). MultiPICo-en is excluded (overlaps EPIC)."""
    files = {
        ("CSC", "GPT-4o-mini"): ("e1_CSC_gpt_phase1.parquet"),
        ("CSC", "Claude-Haiku-4.5"): ("e1_CSC_claude_phase1.parquet"),
        ("CSC", "GPT-4.1"): ("e1_CSC_gpt_phasefrontier400.parquet"),
        ("CSC", "Claude-Opus-4.8"): ("e1_CSC_claude_phasefrontier400.parquet"),
        ("EPIC", "GPT-4o-mini"): ("e1_EPIC_gpt_phaseepic1.parquet"),
        ("EPIC", "Claude-Haiku-4.5"): ("e1_EPIC_claude_phaseepic1.parquet"),
        ("EPIC", "GPT-4.1"): ("e1_EPIC_gpt_phaseepicF.parquet"),
        ("EPIC", "Claude-Opus-4.8"): ("e1_EPIC_claude_phaseepicF.parquet"),
    }
    out: dict = {}
    for (corpus, model), f in files.items():
        d = pd.read_parquet(E1 / f)
        d = d[d["pred"].notna() & d["conf"].notna()].copy()
        y = d["y_true"].to_numpy(int)
        pr = d["pred"].to_numpy(int)
        conf = d["conf"].to_numpy(float) / 100.0
        dis = d["dis_mi"].to_numpy(float)
        terc = tercile_labels(dis)
        low = terc == 0
        rng = np.random.default_rng(SEED)
        nl = int(low.sum())
        yl, pl_ = y[low], pr[low]
        bb, e0, e1_ = [], [], []
        for _ in range(B_CI):
            i = rng.integers(0, nl, nl)
            bb.append(bal_acc(pl_[i], yl[i]))
            m0, m1 = yl[i] == 0, yl[i] == 1
            e0.append(float((pl_[i][m0] != 0).mean()) if m0.sum() else np.nan)
            e1_.append(float((pl_[i][m1] != 1).mean()) if m1.sum() else np.nan)
        # ECE of verbalized confidence for the predicted label (10 equal-width bins)
        correct = (pr == y).astype(float)
        edges = np.linspace(0, 1, 11)
        bi = np.clip(np.digitize(conf, edges) - 1, 0, 9)
        ece = sum(abs(correct[bi == b].mean() - conf[bi == b].mean()) * (bi == b).mean()
                  for b in range(10) if (bi == b).sum())
        Uvc = 1 - conf
        row = {"n": int(len(y)), "n_low": nl, "pos_rate_low": round(float(yl.mean()), 4),
               "n_pos_low": int(yl.sum()),
               "err_rate_low": round(float((pl_ != yl).mean()), 4),
               "balacc_low": round(bal_acc(pl_, yl), 4),
               "balacc_low_ci95": [round(float(np.nanpercentile(bb, 2.5)), 4), round(float(np.nanpercentile(bb, 97.5)), 4)],
               "fpr_low": round(float((pl_[yl == 0] != 0).mean()), 4),
               "fpr_low_ci95": [round(float(np.nanpercentile(e0, 2.5)), 4), round(float(np.nanpercentile(e0, 97.5)), 4)],
               "fnr_low": round(float((pl_[yl == 1] != 1).mean()), 4) if (yl == 1).sum() else None,
               "fnr_low_ci95": [round(float(np.nanpercentile(e1_, 2.5)), 4), round(float(np.nanpercentile(e1_, 97.5)), 4)],
               "balacc_all": round(bal_acc(pr, y), 4),
               "ece_verbal": round(float(ece), 4),
               "rho_Uvc_dis": round(float(spearmanr(Uvc, dis)[0]), 4)}
        # AUROC of U_vc for identifying high-disagreement (top tercile) items
        row["auroc_Uvc_high_dis"] = round(auroc((terc == 2).astype(int), Uvc), 4)
        rng = np.random.default_rng(SEED)
        rb = [spearmanr(Uvc[i], dis[i])[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(B_CI))]
        row["rho_Uvc_dis_ci95"] = [round(float(np.nanpercentile(rb, 2.5)), 4), round(float(np.nanpercentile(rb, 97.5)), 4)]
        if model == "GPT-4o-mini" and d["p_yes_lp"].notna().sum() > 0:
            dd = d[d["p_yes_lp"].notna()]
            ulp = 1 - np.abs(2 * dd["p_yes_lp"].astype(float).to_numpy() - 1)
            dl = dd["dis_mi"].to_numpy(float)
            row["rho_Ulp_dis"] = round(float(spearmanr(ulp, dl)[0]), 4)
            row["auroc_Ulp_high_dis"] = round(auroc((tercile_labels(dl) == 2).astype(int), ulp), 4)
            row["n_lp"] = int(len(dd))
        out[f"{corpus}|{model}"] = row
    return out


if __name__ == "__main__":
    main()
