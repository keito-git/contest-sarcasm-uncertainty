"""
E3 analysis v2 — asymmetric-contraction with BALANCED accuracy over the full Qwen
capacity curve (0.5B -> 32B). Balanced accuracy (mean of per-class recall) removes the
class-imbalance confound in raw accuracy (CSC base rate ~0.26 'sarcastic', so a
'No'-biased model inflates raw accuracy). Reports per-capacity/per-tercile balanced acc,
mean p_yes (to expose 'No'-bias), and a set-level bootstrap for the asymmetry
  contraction_low(bal) - contraction_high(bal).
"""
from __future__ import annotations
from pathlib import Path
import os, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

BASE = str(Path(__file__).parent.parent)
IN = f"{BASE}/results/llm_e3"
CAP = {"q0p5b": 0.5, "q1p5b": 1.5, "q3b": 3.0, "q7b": 7.0, "q14b": 14.0, "q32b": 32.0,
       "l1b": 1.0, "l3b": 3.0, "l8b": 8.0}


def tercile(a):
    return np.digitize(rankdata(a) / (len(a) + 1), [1/3, 2/3])


def bal_acc(pred, y):
    y = np.asarray(y); pred = np.asarray(pred)
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0: recs.append((pred[m] == c).mean())
    return float(np.mean(recs)) if recs else float("nan")


def load(tag, name):
    f = f"{IN}/e3_{tag}_{name}.csv"
    if not os.path.exists(f): return None
    d = pd.read_csv(f)[["item_id", "p_yes", "H", "dis_mi", "y_true"]]
    return d.rename(columns={"p_yes": f"p_{tag}", "H": f"H_{tag}"})


def main(family_tags):
    lines = ["# E3 v2 — balanced-accuracy asymmetric contraction (full capacity curve)", ""]
    for name in ["CSC", "MultiPICo"]:
        present = [t for t in family_tags if load(t, name) is not None]
        if len(present) < 2:
            lines.append(f"## {name}: <2 capacities"); continue
        base = load(present[0], name)[["item_id", "dis_mi", "y_true"]]
        m = base.copy()
        for t in present:
            d = load(t, name)
            m = m.merge(d[["item_id", f"p_{t}", f"H_{t}"]], on="item_id")
        m["terc"] = tercile(m["dis_mi"].to_numpy())
        y = m["y_true"].to_numpy()
        lines.append(f"## {name}  (n={len(m)}, base_rate_pos={y.mean():.3f})")
        lines.append("| cap | tercile | n | raw_acc | bal_acc | mean_pyes | meanH |")
        lines.append("|--|--|--|--|--|--|--|")
        for t in present:
            pred = (m[f"p_{t}"].to_numpy() >= 0.5).astype(int)
            for g, lab in [(0, "low"), (1, "mid"), (2, "high")]:
                gi = m.terc.to_numpy() == g
                lines.append(f"| {CAP[t]}B | {lab} | {gi.sum()} | "
                             f"{(pred[gi]==y[gi]).mean():.3f} | {bal_acc(pred[gi],y[gi]):.3f} | "
                             f"{m.loc[gi,f'p_{t}'].mean():.3f} | {m.loc[gi,f'H_{t}'].mean():.3f} |")
        # contraction of balanced acc, small->large, per tercile + bootstrap asymmetry
        sm, lg = present[0], present[-1]
        ps = (m[f"p_{sm}"].to_numpy() >= 0.5).astype(int)
        pl = (m[f"p_{lg}"].to_numpy() >= 0.5).astype(int)
        terc = m.terc.to_numpy()
        def contr(idx, g):
            gi = idx[terc[idx] == g]
            return bal_acc(pl[gi], y[gi]) - bal_acc(ps[gi], y[gi])
        allidx = np.arange(len(m))
        dlow, dhigh = contr(allidx, 0), contr(allidx, 2)
        rng = np.random.default_rng(42)
        bs = []
        for _ in range(4000):
            bi = rng.integers(0, len(m), len(m))
            bs.append(contr(bi, 0) - contr(bi, 2))
        bs = np.asarray(bs); p_asym = np.mean(bs <= 0)
        lines.append(f"- **balanced-acc contraction ({CAP[sm]}B->{CAP[lg]}B)**: "
                     f"low={dlow:+.3f} high={dhigh:+.3f} | "
                     f"**ASYMMETRY low-high={dlow-dhigh:+.3f}** (boot p(<=0)={p_asym:.3f})")
        r_last = spearmanr(m[f"H_{lg}"], m["dis_mi"]).statistic
        lines.append(f"- single-capacity rho(H_{CAP[lg]}B, human_dis)={r_last:+.3f} "
                     f"(near 0 => one H value can't separate epistemic vs aleatoric)")
        lines.append("")
    lines += ["## Reading",
              "- Balanced-acc contraction should be LARGER on low-dis (humans agree, epistemic)",
              "  than high-dis (aleatoric). mean_pyes columns expose whether gains are real or",
              "  just majority('No')-bias tracking the imbalanced base rate."]
    txt = "\n".join(lines)
    open(f"{IN}/e3_v2_summary.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    QWEN = ["q0p5b", "q1p5b", "q3b", "q7b", "q14b", "q32b"]
    main(QWEN)
