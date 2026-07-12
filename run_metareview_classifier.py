#!/usr/bin/env python3
"""
Meta-review response experiment #3: learned contraction probe.

Replace the hand-crafted 1-D signal |p_3B - p_0.5B| with a discriminative classifier
trained on features from small models only (<= 3B; non-overlap with the 32B target),
to predict scale-fixability  y_fix = 1[y_hat_0.5B != y] * 1[y_hat_32B == y].

Report 5-fold cross-validated AUROC and compare to:
  - hand-crafted contraction |p_3B - p_0.5B| (paper's probe; sanity target
    CSC .553 / MP .689 / EPIC .724)
  - human-disagreement reference 1 - dis
CPU-only, existing predictions. No GPU / API.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

E3 = "results/llm_e3"
OUT = "results/metareview"
os.makedirs(OUT, exist_ok=True)
CORPORA = ["CSC", "MultiPICo", "EPIC"]
SMALL = [("q0p5b", 0.5), ("q1p5b", 1.5), ("q3b", 3.0)]  # <= 3B, non-overlap w/ 32B


def H(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def load(corpus):
    """Match the paper's tab:e5 definition EXACTLY: population = items the 0.5B model
    got WRONG; target y_fix = 1 if 32B is correct on that item (fixable by scaling)."""
    base = pd.read_csv(f"{E3}/e3_q0p5b_{corpus}.csv").set_index("item_id")
    y = base["y_true"].values
    dis = base["dis_mi"].values
    feats = {}
    for tag, sz in SMALL:
        t = pd.read_csv(f"{E3}/e3_{tag}_{corpus}.csv").set_index("item_id")
        feats[sz] = t["p_yes"].reindex(base.index).values
    big = pd.read_csv(f"{E3}/e3_q32b_{corpus}.csv").set_index("item_id")
    p32 = big["p_yes"].reindex(base.index).values
    pred05 = (feats[0.5] >= 0.5).astype(int)
    pred32 = (p32 >= 0.5).astype(int)
    wrong = pred05 != y                      # small-model errors only (paper's subset)
    feats = {sz: v[wrong] for sz, v in feats.items()}
    dis = dis[wrong]
    yfix = (pred32[wrong] == y[wrong]).astype(int)
    return feats, dis, yfix


def make_X(feats):
    p05, p15, p3 = feats[0.5], feats[1.5], feats[3.0]
    return np.column_stack([
        p05, p15, p3,
        H(p05), H(p15), H(p3),
        np.abs(p3 - p05), np.abs(p3 - p15), np.abs(p15 - p05),
        np.std(np.column_stack([p05, p15, p3]), axis=1),
        np.mean(np.column_stack([p05, p15, p3]), axis=1),
    ])


def cv_auroc(X, y, model_fn, seed=42, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def boot_ci(y, score, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], score[b]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    rows = []
    for corpus in CORPORA:
        feats, dis, yfix = load(corpus)
        base_rate = yfix.mean()
        X = make_X(feats)
        contraction = np.abs(feats[3.0] - feats[0.5])
        humandis = 1 - dis

        auroc_hand = roc_auc_score(yfix, contraction)
        auroc_hd = roc_auc_score(yfix, humandis)
        auroc_lr = cv_auroc(X, yfix, lambda: LogisticRegression(max_iter=1000, C=1.0))
        auroc_gb = cv_auroc(X, yfix, lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0))

        # CV out-of-fold scores for the LR, for a CI
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        oof = np.zeros(len(yfix))
        for tr, te in skf.split(X, yfix):
            sc = StandardScaler().fit(X[tr])
            oof[te] = LogisticRegression(max_iter=1000).fit(
                sc.transform(X[tr]), yfix[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        lo, hi = boot_ci(yfix, oof)

        rows.append(dict(
            corpus=corpus, n=len(yfix), fixable_rate=round(base_rate, 3),
            hand_contraction=round(auroc_hand, 3),
            learned_LR=round(auroc_lr, 3), learned_LR_CI=f"[{lo:.2f},{hi:.2f}]",
            learned_GB=round(auroc_gb, 3),
            human_dis_ref=round(auroc_hd, 3),
        ))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/learned_probe_auroc.csv", index=False)
    print("=" * 92)
    print("LEARNED CONTRACTION PROBE (5-fold CV AUROC for predicting scale-fixability)")
    print("features: p_0.5/1.5/3B, their entropies, pairwise |contractions|, mean/std (<=3B only)")
    print("=" * 92)
    with pd.option_context("display.width", 220):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
