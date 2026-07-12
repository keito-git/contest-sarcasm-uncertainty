"""
E3-method — turning the diagnosis into a usable result (AAAI constructive contribution).

Claim: a single model's own uncertainty CANNOT tell an epistemic (fixable-by-scaling)
error from an aleatoric (irreducible) one; the epistemic/aleatoric structure — grounded
in human disagreement, revealed by capacity contraction — can.

Concrete task ("selective escalation"): you run a small model; among the items it gets
WRONG, which should you escalate to a large model? Only the epistemic (fixable) ones.
  y_fix = 1 if LARGE model is correct on an item the SMALL model got wrong.
Predictors available at the small model:
  - H_small        : sampling predictive entropy (single-model uncertainty)   [baseline]
  - conf_small     : |2 p_small - 1|                                          [baseline]
  - contraction    : |p_mid - p_small| (does a mid model already move?)       [cheap 2-model probe]
  - low_humandis   : 1 - dis_mi (aleatoric-grounded; humans agree => epistemic)[structure]
We report AUROC(predictor -> y_fix) and an escalation risk-coverage curve.
Expected: single-model uncertainty ~0.5 (uninformative); human-disagreement / contraction
clearly > 0.5 => single signals miss the epistemic/aleatoric split.

Runs on the E3 sweep CSVs. SMALL/MID/LARGE tags configurable.
"""
from __future__ import annotations
from pathlib import Path
import os, glob, numpy as np, pandas as pd
from scipy.stats import rankdata

BASE = str(Path(__file__).parent.parent)
IN = f"{BASE}/results/llm_e3"


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); ok = ~np.isnan(s); y, s = y[ok], s[ok]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(y, s, nb=2000):
    y = np.asarray(y); s = np.asarray(s); ok = ~np.isnan(s); y, s = y[ok], s[ok]
    rng = np.random.default_rng(42); n = len(y)
    bs = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        bs.append(auc(y[idx], s[idx]))
    return auc(y, s), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan, np.nan)


def load(tag, name):
    f = f"{IN}/e3_{tag}_{name}.csv"
    return pd.read_csv(f)[["item_id", "p_yes", "H", "dis_mi", "y_true"]].rename(
        columns={"p_yes": f"p_{tag}", "H": f"H_{tag}"}) if os.path.exists(f) else None


def main(SMALL="q1p5b", MID="q7b", LARGE="q14b"):
    lines = ["# E3-method — selective escalation: single-model uncertainty vs epistemic/aleatoric structure",
             "", f"SMALL={SMALL} MID={MID} LARGE={LARGE}. y_fix=1 if LARGE correct on a SMALL error.", ""]
    for name in ["CSC", "MultiPICo"]:
        s, mid, lg = load(SMALL, name), load(MID, name), load(LARGE, name)
        if s is None or lg is None:
            lines.append(f"## {name}: missing tags"); continue
        m = s.merge(lg[["item_id", f"p_{LARGE}", f"H_{LARGE}"]], on="item_id")
        if mid is not None: m = m.merge(mid[["item_id", f"p_{MID}"]], on="item_id")
        corr_s = ((m[f"p_{SMALL}"] >= 0.5).astype(int) == m["y_true"]).to_numpy()
        corr_l = ((m[f"p_{LARGE}"] >= 0.5).astype(int) == m["y_true"]).to_numpy()
        wrong = ~corr_s
        sub = m[wrong].copy()
        y_fix = corr_l[wrong].astype(int)   # among small-errors, did large fix it?
        if y_fix.sum() < 5 or (1 - y_fix).sum() < 5:
            lines.append(f"## {name}: too few fixable/persistent"); continue
        preds = {
            "H_small (baseline)": sub[f"H_{SMALL}"].to_numpy(),
            "conf_small=|2p-1| (baseline)": np.abs(2 * sub[f"p_{SMALL}"].to_numpy() - 1),
            "low_humandis=1-dis (structure)": 1 - sub["dis_mi"].to_numpy(),
        }
        if f"p_{MID}" in sub:
            preds["contraction=|p_mid-p_small| (2-model probe)"] = np.abs(
                sub[f"p_{MID}"].to_numpy() - sub[f"p_{SMALL}"].to_numpy())
        lines.append(f"## {name}  (small errors n={len(sub)}, fixable={int(y_fix.sum())}, "
                     f"persistent={int((1-y_fix).sum())})")
        lines.append(f"- base rate fixable = {y_fix.mean():.3f}")
        lines.append("- AUROC(predictor -> y_fix):  (0.5=uninformative; >0.5 finds epistemic errors)")
        for k, v in preds.items():
            a, ci = auc_ci(y_fix, v)
            lines.append(f"    {k}: AUROC={a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
        # escalation efficiency: escalate top-k% by each predictor, fraction of fixable captured
        lines.append("- escalation efficiency (escalate top-30% of small-errors; frac of all fixable captured):")
        k = max(1, int(0.30 * len(sub)))
        for kk, v in preds.items():
            order = np.argsort(-np.nan_to_num(v, nan=-np.inf))
            cap = y_fix[order[:k]].sum() / max(1, y_fix.sum())
            lines.append(f"    {kk}: {cap:.3f}")
        rng = np.random.default_rng(0)
        rand = np.mean([y_fix[rng.permutation(len(sub))[:k]].sum() / max(1, y_fix.sum()) for _ in range(500)])
        lines.append(f"    random: {rand:.3f}  | oracle: {min(1.0, k/max(1,y_fix.sum())):.3f}")
        lines.append("")
    lines += ["## Reading",
              "- If H_small/conf_small AUROC ~ 0.5 but low_humandis/contraction AUROC > 0.6:",
              "  the small model's OWN uncertainty cannot identify which of its errors are",
              "  epistemic (fixable); the epistemic/aleatoric structure (human-disagreement",
              "  grounding, or a cheap contraction probe) can. => single-signal routing is",
              "  insufficient; separating epistemic from aleatoric needs the structure."]
    txt = "\n".join(lines)
    open(f"{IN}/e3_method_summary.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
