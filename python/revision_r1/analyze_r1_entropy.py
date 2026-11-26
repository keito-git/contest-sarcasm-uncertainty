"""
analyze_r1_entropy.py -- KBS R1: alignment of sampled label entropy with annotator
disagreement (Section 5.2, Appendix C).

For each corpus and for Qwen2.5-0.5B and Qwen2.5-32B, the sampled label entropy
U^ent_i (column H of the regenerated sweep, K = 10 samples) is compared with dis_i by
  - the Spearman correlation, and
  - the AUROC for identifying the high-disagreement tercile,
each with a 95% item-bootstrap interval (B = 2000, seed 42).

Input : results/revision_gpu_2026-09/server_outputs/regen/c7_{tag}_{corpus}.csv
Output: results/revision_r1_2026-11/R1_ENT.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PROJ = Path(__file__).resolve().parents[2]
REGEN = PROJ / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
OUT = PROJ / "results" / "revision_r1_2026-11" / "R1_ENT.json"
CORPORA = ["CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]
TAGS = ["q0p5b", "q32b"]
B, SEED = 2000, 42


def tercile_labels(a: np.ndarray) -> np.ndarray:
    """Same definition as analyze_r1_cpu.py."""
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    r = pd.Series(s).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    res: dict = {"meta": {"utc": datetime.now(timezone.utc).isoformat(), "B": B, "seed": SEED}}
    for c in CORPORA:
        res[c] = {}
        for t in TAGS:
            d = pd.read_csv(REGEN / f"c7_{t}_{c}.csv")
            d = d.dropna(subset=["H", "dis_mi"])
            h = d["H"].to_numpy(float)
            dis = d["dis_mi"].to_numpy(float)
            top = (tercile_labels(dis) == 2).astype(int)
            rho = float(spearmanr(h, dis).statistic)
            auc = auroc(top, h)
            rng = np.random.default_rng(SEED)
            n = len(h)
            br, ba = np.empty(B), np.empty(B)
            for b in range(B):
                i = rng.integers(0, n, n)
                br[b] = spearmanr(h[i], dis[i]).statistic
                ba[b] = auroc(top[i], h[i])
            res[c][t] = {"n": n, "rho": round(rho, 4),
                         "rho_ci95": [round(float(np.nanpercentile(br, 2.5)), 4),
                                      round(float(np.nanpercentile(br, 97.5)), 4)],
                         "auroc_top": round(auc, 4),
                         "auroc_ci95": [round(float(np.nanpercentile(ba, 2.5)), 4),
                                        round(float(np.nanpercentile(ba, 97.5)), 4)]}
            print(c, t, res[c][t], flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
