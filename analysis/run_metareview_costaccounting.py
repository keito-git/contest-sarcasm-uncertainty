#!/usr/bin/env python3
"""
Meta-review response experiment #2: honest cost accounting for the contraction probe.

Concern: the probe runs BOTH 0.5B and 3B to decide whether to escalate to 32B, so it
pays extra compute. Is it worth it vs simply using the 3B answer, or vs the static
single-model frontier?

We build a compute-accuracy frontier. Cost is proxied by parameter count (inference
FLOPs scale ~linearly with params for a forward pass). For each corpus:

  Static single models: (cost=size, balacc(size))  -- "just use one model".

  Cascade(base=3B): default answer = 3B; escalate the top-b fraction of items to 32B
  by a routing signal, non-escalated items keep the 3B answer.
    - contraction  : route by |p_3B - p_0.5B|   (needs 0.5B+3B -> +0.5 overhead)
        cost(b) = 0.5 + 3 + b*32
    - 3B-confidence: route by 3B predictive entropy (needs only 3B)
        cost(b) = 3 + b*32
    - random       : cost(b) = 3 + b*32
    - oracle       : route by true fixability (upper bound), cost = 0.5+3+b*32

Verdict: does the contraction cascade sit ABOVE the static frontier and above the
3B-confidence cascade *including* its +0.5 overhead? Report honestly.
CPU-only; existing predictions. No GPU / API.
"""
import os
import numpy as np
import pandas as pd

E3 = "results/llm_e3"
OUT = "results/metareview"
os.makedirs(OUT, exist_ok=True)
SIZES = [("q0p5b", 0.5), ("q1p5b", 1.5), ("q3b", 3.0),
         ("q7b", 7.0), ("q14b", 14.0), ("q32b", 32.0)]
CORPORA = ["CSC", "MultiPICo", "EPIC"]


def balacc(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))


def load(corpus):
    d = {}
    for tag, size in SIZES:
        f = f"{E3}/e3_{tag}_{corpus}.csv"
        if os.path.exists(f):
            t = pd.read_csv(f)
            t["item_id"] = t["item_id"].astype(str)
            d[size] = t.set_index("item_id")
    ids = d[0.5].index
    y = d[0.5]["y_true"].values
    return d, ids, y


def bin_entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def cascade_curve(d, y, signal, base_size=3.0, big_size=32.0, budgets=None):
    """Return (costs, accs) for escalating top-b by signal (higher=escalate)."""
    if budgets is None:
        budgets = np.linspace(0, 1, 21)
    base_pred = (d[base_size]["p_yes"].values >= 0.5).astype(int)
    big_pred = (d[big_size]["p_yes"].values >= 0.5).astype(int)
    order = np.argsort(-signal)  # escalate largest-signal first
    accs = []
    n = len(y)
    for b in budgets:
        k = int(round(b * n))
        esc = np.zeros(n, bool)
        esc[order[:k]] = True
        pred = np.where(esc, big_pred, base_pred)
        accs.append(balacc(pred, y))
    return np.array(budgets), np.array(accs)


def main():
    summary = []
    all_curves = {}
    for corpus in CORPORA:
        d, ids, y = load(corpus)
        p05 = d[0.5]["p_yes"].values
        p3 = d[3.0]["p_yes"].values
        contraction = np.abs(p3 - p05)
        conf3 = bin_entropy(p3)                      # 3B uncertainty
        rng = np.random.default_rng(42)
        rnd = rng.random(len(y))
        big_pred = (d[32.0]["p_yes"].values >= 0.5).astype(int)
        base_pred = (d[3.0]["p_yes"].values >= 0.5).astype(int)
        fixable = ((base_pred != y) & (big_pred == y)).astype(int)  # oracle target

        budgets = np.linspace(0, 1, 21)
        curves = {}
        for name, sig, overhead in [
            ("contraction", contraction, 0.5 + 3.0),
            ("conf3", conf3, 3.0),
            ("random", rnd, 3.0),
            ("oracle", fixable + rng.random(len(y)) * 1e-6, 0.5 + 3.0),
        ]:
            bud, acc = cascade_curve(d, y, sig, budgets=budgets)
            cost = overhead + bud * 32.0
            curves[name] = (cost, acc, bud)
        # static single-model frontier
        static = [(sz, balacc((d[sz]["p_yes"].values >= 0.5).astype(int), y))
                  for sz in [0.5, 1.5, 3.0, 7.0, 14.0, 32.0]]
        all_curves[corpus] = dict(curves=curves, static=static)

        # ---- Verdict metrics ----
        # 1) At matched TOTAL COST, does contraction cascade beat the static frontier?
        #    Interpolate static frontier acc at the cascade's cost points.
        scost = np.array([s[0] for s in static]); sacc = np.array([s[1] for s in static])
        ccost, cacc, cbud = curves["contraction"]
        static_at = np.interp(ccost, scost, sacc)
        gain_vs_static = cacc - static_at            # >0 => cascade dominates
        # 2) contraction vs 3B-confidence cascade at matched cost
        #    conf3 cost = 3 + b*32; contraction cost = 3.5 + b*32 (shifted by 0.5)
        concost, conacc, _ = curves["conf3"]
        conf_at_ccost = np.interp(ccost, concost, conacc)
        gain_vs_conf = cacc - conf_at_ccost
        # 3) 3B-alone reference
        ba3 = balacc(base_pred, y)
        ba32 = balacc(big_pred, y)

        summary.append(dict(
            corpus=corpus, balacc_3B=round(ba3, 3), balacc_32B=round(ba32, 3),
            max_gain_vs_static=round(np.nanmax(gain_vs_static), 3),
            mean_gain_vs_static=round(np.nanmean(gain_vs_static[1:]), 3),
            max_gain_vs_conf3=round(np.nanmax(gain_vs_conf), 3),
            mean_gain_vs_conf3=round(np.nanmean(gain_vs_conf[1:]), 3),
        ))

    res = pd.DataFrame(summary)
    res.to_csv(f"{OUT}/cost_accounting_summary.csv", index=False)
    import pickle
    pickle.dump(all_curves, open(f"{OUT}/cost_accounting_curves.pkl", "wb"))

    print("=" * 84)
    print("COST ACCOUNTING: contraction cascade (base=3B, escalate to 32B by |p3B-p0.5B|)")
    print("cost = params-weighted calls; static frontier = single models")
    print("gain_vs_static  : cascade balacc - static-frontier balacc at MATCHED total cost")
    print("gain_vs_conf3   : cascade balacc - 3B-entropy cascade balacc at matched cost")
    print("(positive => the probe's extra compute IS worth it)")
    print("=" * 84)
    with pd.option_context("display.width", 200):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
