#!/usr/bin/env python3
"""
Meta-review response experiment #1 (highest priority):
Re-score the capacity sweep with DISTRIBUTION-SENSITIVE metrics (Brier against the
human label distribution, and Jensen-Shannon divergence) instead of majority-label
balanced accuracy, to test whether the "aleatoric ceiling" in the high-disagreement
stratum is genuine irreducibility or an artifact of scoring a distributed answer
against a single reference label.

CPU-only; uses existing predictions in results/llm_e3/ and human distributions in
results/sarcasm/. No GPU / no API calls.

Interpretation:
  - Human distribution p^h: binary corpora -> annotator ironic fraction (p_ironic);
    CSC (Likert 1-6) -> normalized mean rating (mean_eval_norm) as P(sarcastic).
  - Model predicted prob p_hat = p_yes (fraction of K=10 samples = "sarcastic").
  - Brier (vs distribution) = (p_hat - p^h)^2   [lower = closer to humans]
  - JSD = Jensen-Shannon divergence between [p_hat,1-p_hat] and [p^h,1-p^h] (bits).
  - Stratify by dis_mi terciles (same as the paper): low = bottom third, high = top third.

If, as capacity grows, Brier/JSD DROP on low-dis but stay FLAT (or rise) on high-dis,
the ceiling is genuine (scaling cannot resolve contested items even distributionally)
-> supports the paper and rebuts the single-label-artifact concern.
If Brier/JSD drop on high-dis too, the balanced-accuracy ceiling was an artifact
-> undermines the headline; report honestly either way.
"""
import os
import numpy as np
import pandas as pd

E3 = "results/llm_e3"
SARC = "results/sarcasm"
OUT = "results/metareview"
os.makedirs(OUT, exist_ok=True)

SIZES = [("q0p5b", 0.5), ("q1p5b", 1.5), ("q3b", 3.0),
         ("q7b", 7.0), ("q14b", 14.0), ("q32b", 32.0)]
CORPORA = ["CSC", "MultiPICo", "EPIC"]
EPS = 1e-12


def human_dist(corpus):
    """Return DataFrame[item_id, p_h] = human P(sarcastic/ironic)."""
    if corpus == "CSC":
        a = pd.read_parquet(f"{SARC}/csc_aggregated.parquet")
        # Likert mean normalized to [0,1] as the human P(sarcastic) proxy.
        df = a[["item_id", "mean_eval_norm"]].rename(columns={"mean_eval_norm": "p_h"})
    else:
        fn = "multipico" if corpus == "MultiPICo" else "epic"
        a = pd.read_parquet(f"{SARC}/{fn}_aggregated.parquet")
        df = a[["item_id", "p_ironic"]].rename(columns={"p_ironic": "p_h"})
    df["item_id"] = df["item_id"].astype(str)
    return df


def jsd_binary(p, q):
    """Jensen-Shannon divergence (bits) between Bernoulli(p) and Bernoulli(q).
    JSD = 0.5*KL(P||M) + 0.5*KL(Q||M), M = 0.5(P+Q); in [0,1] with log base 2."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    q = np.clip(np.asarray(q, float), EPS, 1 - EPS)
    m = 0.5 * (p + q)

    def kl(a, b):
        return a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_corpus(corpus):
    """Join per-size predictions with dis_mi and human p_h. Returns long DataFrame."""
    hd = human_dist(corpus)
    frames = []
    # dis_mi is size-independent; take it from the 0.5B file.
    base = pd.read_csv(f"{E3}/e3_q0p5b_{corpus}.csv")
    base["item_id"] = base["item_id"].astype(str)
    dis = base[["item_id", "dis_mi", "y_true"]].drop_duplicates("item_id")
    for tag, size in SIZES:
        fn = f"{E3}/e3_{tag}_{corpus}.csv"
        if not os.path.exists(fn):
            continue
        d = pd.read_csv(fn)[["item_id", "p_yes"]].copy()
        d["item_id"] = d["item_id"].astype(str)
        d = d.merge(dis, on="item_id", how="inner").merge(hd, on="item_id", how="inner")
        d["size"] = size
        d["brier"] = (d["p_yes"] - d["p_h"]) ** 2
        d["jsd"] = jsd_binary(d["p_yes"].values, d["p_h"].values)
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    # Stratify by dis_mi terciles (33rd/67th), consistent per corpus.
    lo, hi = np.percentile(dis["dis_mi"], [33, 67])
    long["stratum"] = np.where(long["dis_mi"] <= lo, "low",
                        np.where(long["dis_mi"] >= hi, "high", "mid"))
    return long, (lo, hi)


def boot_asym(sub, metric, n=10000, seed=42):
    """Bootstrap one-sided p that the metric DROP is larger on low than high.
    delta_low = metric(0.5B) - metric(32B) on low-dis (positive = improved).
    delta_high likewise on high-dis. asym = delta_low - delta_high.
    Null: asym <= 0. Resample items within each stratum."""
    rng = np.random.default_rng(seed)
    low = sub[sub.stratum == "low"]
    high = sub[sub.stratum == "high"]

    def piv(df):
        return df.pivot_table(index="item_id", columns="size", values=metric)
    plow, phigh = piv(low), piv(high)
    s_small, s_big = 0.5, 32.0
    if s_small not in plow.columns or s_big not in plow.columns:
        return np.nan, np.nan
    dlow0 = (plow[s_small] - plow[s_big]).dropna()
    dhigh0 = (phigh[s_small] - phigh[s_big]).dropna()
    asym0 = dlow0.mean() - dhigh0.mean()
    cnt = 0
    for _ in range(n):
        bl = dlow0.sample(len(dlow0), replace=True, random_state=rng.integers(1 << 30))
        bh = dhigh0.sample(len(dhigh0), replace=True, random_state=rng.integers(1 << 30))
        if (bl.mean() - bh.mean()) <= 0:
            cnt += 1
    return asym0, cnt / n


def main():
    rows = []
    curve_rows = []
    for corpus in CORPORA:
        long, (lo, hi) = load_corpus(corpus)
        for metric in ["brier", "jsd"]:
            # Mean metric per (size, stratum)
            g = (long[long.stratum.isin(["low", "high"])]
                 .groupby(["size", "stratum"])[metric].mean().reset_index())
            for _, r in g.iterrows():
                curve_rows.append(dict(corpus=corpus, metric=metric,
                                       size=r["size"], stratum=r["stratum"],
                                       value=r[metric]))
            piv = g.pivot(index="size", columns="stratum", values=metric)
            s_small, s_big = 0.5, 32.0
            dlow = piv.loc[s_small, "low"] - piv.loc[s_big, "low"]     # >0 => improved
            dhigh = piv.loc[s_small, "high"] - piv.loc[s_big, "high"]
            asym, p = boot_asym(long, metric)
            rows.append(dict(
                corpus=corpus, metric=metric,
                low_small=round(piv.loc[s_small, "low"], 4),
                low_big=round(piv.loc[s_big, "low"], 4),
                high_small=round(piv.loc[s_small, "high"], 4),
                high_big=round(piv.loc[s_big, "high"], 4),
                d_low=round(dlow, 4), d_high=round(dhigh, 4),
                asym=round(asym, 4), p_onesided=round(p, 4),
            ))
    res = pd.DataFrame(rows)
    curve = pd.DataFrame(curve_rows)
    res.to_csv(f"{OUT}/dist_metrics_asymmetry.csv", index=False)
    curve.to_csv(f"{OUT}/dist_metrics_curves.csv", index=False)

    print("=" * 78)
    print("DISTRIBUTION-SENSITIVE RE-SCORING (lower metric = closer to human dist)")
    print("d_low/d_high = metric(0.5B) - metric(32B); POSITIVE = improved with scale")
    print("asym = d_low - d_high; p = one-sided bootstrap that asym<=0 (10k)")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.to_string(index=False))
    print()
    print("INTERPRETATION per row:")
    for _, r in res.iterrows():
        hi_improves = r["d_high"] > 0.01
        lo_improves = r["d_low"] > 0.01
        msg = f"  {r['corpus']:9s} {r['metric']:5s}: "
        if lo_improves and not hi_improves:
            msg += "low-dis improves, high-dis FLAT -> ceiling GENUINE (supports paper)"
        elif lo_improves and hi_improves and r["d_high"] < r["d_low"]:
            msg += "both improve but low>high -> partial ceiling (mixed)"
        elif hi_improves:
            msg += "high-dis ALSO improves -> ceiling may be artifact (undermines)"
        else:
            msg += "neither improves clearly"
        print(msg)


if __name__ == "__main__":
    main()
