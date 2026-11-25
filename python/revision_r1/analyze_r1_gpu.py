"""
analyze_r1_gpu.py -- aggregate the KBS R1 GPU runs (seeds, OLMo-2, item-language prompts).

Inputs : results/revision_r1_2026-11/gpu_outputs/{c7seed*,fam_*,native_*}.csv
         results/revision_gpu_2026-09/server_outputs/regen/c7_*.csv (seed 42, original)
Output : results/revision_r1_2026-11/R1_GPU.json, including a "macros" dict that
         make_numbers_r1.py writes into numbers_r1.tex.
Definitions are identical to analyze_r1_cpu.py (terciles of the corpus dis_mi,
balanced accuracy, item bootstrap B=4000, seed 42).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_r1_cpu import (B_TEST, CORPORA, SEED, auroc, bal_acc, boot_auroc,  # noqa: E402
                            dasym, dasym_test, p_floor, tercile_labels)

PROJ = Path(__file__).resolve().parents[2]
GPU = PROJ / "results" / "revision_r1_2026-11" / "gpu_outputs"
REGEN = PROJ / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
OUT = PROJ / "results" / "revision_r1_2026-11" / "R1_GPU.json"
TAG = {"CSC": "CSC", "EPIC": "EPIC", "MultiPICo_es": "ES", "MultiPICo_de": "DE",
       "MultiPICo_fr": "FR", "MultiPICo_it": "IT"}
OLMO = {"o1b": 1.0, "o7b": 7.0, "o13b": 13.0, "o32b": 32.0}


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(c) -> str:
    return f"[{f3(c[0])}, {f3(c[1])}]"


def load(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d["item_id"] = d["item_id"].astype(str)
    return d.set_index("item_id")


def base_frame(corpus: str) -> pd.DataFrame:
    d = pd.read_csv(REGEN / f"c7_q0p5b_{corpus}.csv")
    d["item_id"] = d["item_id"].astype(str)
    return d.set_index("item_id")[["dis_mi", "y_true"]]


def slope_test(P: dict, y, low, high, sizes: dict):
    logs = np.log2([sizes[t] for t in P])
    lc = logs - logs.mean()

    def sd(idx):
        bl = np.array([bal_acc(P[t][idx][low[idx]], y[idx][low[idx]]) for t in P])
        bh = np.array([bal_acc(P[t][idx][high[idx]], y[idx][high[idx]]) for t in P])
        return float(np.sum(lc * bl) / np.sum(lc ** 2) - np.sum(lc * bh) / np.sum(lc ** 2)), bl, bh

    obs, bl, bh = sd(np.arange(len(y)))
    rng = np.random.default_rng(SEED)
    bs = np.array([sd(rng.integers(0, len(y), len(y)))[0] for _ in range(B_TEST)])
    p = float(np.mean(bs <= 0))
    return {"slope_diff": round(obs, 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                  round(float(np.percentile(bs, 97.5)), 4)],
            "p_one_sided": p, "p_str": p_floor(p, B_TEST),
            "balacc_low": [round(float(v), 4) for v in bl],
            "balacc_high": [round(float(v), 4) for v in bh]}


def main() -> None:
    res: dict = {"meta": {"utc": datetime.now(timezone.utc).isoformat(), "B": B_TEST, "seed": SEED},
                 "seeds": {}, "native": {}, "family": {}}
    M: dict[str, str] = {}

    # ---------------- seeds ----------------
    seed_sds, seed_means, have_all_seeds = [], [], True
    for c in CORPORA:
        b = base_frame(c)
        y_all = b["y_true"].to_numpy(int)
        terc = tercile_labels(b["dis_mi"].to_numpy(float))
        rows = {}
        for s in (42, 101, 202):
            pref = "c7" if s == 42 else f"c7seed{s}"
            src = REGEN if s == 42 else GPU
            f0, f3_, f32 = (load(src / f"{pref}_{t}_{c}.csv") for t in ("q0p5b", "q3b", "q32b"))
            if f0 is None or f32 is None:
                continue
            ids = b.index.intersection(f0.index).intersection(f32.index)
            m = b.index.isin(ids)
            ps = (f0.loc[b.index[m], "p_yes"].to_numpy() >= 0.5).astype(int)
            pl = (f32.loc[b.index[m], "p_yes"].to_numpy() >= 0.5).astype(int)
            y, t = y_all[m], terc[m]
            row = {"n": int(m.sum()), "dasym": round(dasym(pl, ps, y, t == 0, t == 2), 4)}
            if f3_ is not None:
                p0 = f0.loc[b.index[m], "p_yes"].to_numpy(float)
                p3 = f3_["p_yes"].reindex(b.index[m]).to_numpy(float)
                yfix = ((ps != y) & (pl == y)).astype(int)
                row["probe_all_auroc"] = round(auroc(yfix, np.abs(p3 - p0)), 4)
            rows[str(s)] = row
        res["seeds"][c] = dict(rows)
        t_ = TAG[c]
        if len(rows) == 3:
            v = np.array([rows[k]["dasym"] for k in rows])
            res["seeds"][c]["mean"] = round(float(v.mean()), 4)  # rows itself stays seed-only
            res["seeds"][c]["sd"] = round(float(v.std(ddof=1)), 4)
            M[f"SeedMean{t_}"] = f3(v.mean())
            M[f"SeedSD{t_}"] = f3(v.std(ddof=1))
            seed_sds.append(v.std(ddof=1))
            seed_means.append(v.mean())
            for k in rows:
                M[f"SeedDA{t_}S{ {'42': 'A', '101': 'B', '202': 'C'}[k]}"] = f3(rows[k]["dasym"])
                if "probe_all_auroc" in rows[k]:
                    M[f"SeedProbe{t_}S{ {'42': 'A', '101': 'B', '202': 'C'}[k]}"] = f3(rows[k]["probe_all_auroc"])
        else:
            have_all_seeds = False
    if have_all_seeds:
        M["SeedSDMax"] = f3(max(seed_sds))
        M["SeedMeanMin"] = f3(min(seed_means))

    # ---------------- native prompts ----------------
    for c in CORPORA[2:]:
        b = base_frame(c)
        f0, f32 = load(GPU / f"native_q0p5b_{c}.csv"), load(GPU / f"native_q32b_{c}.csv")
        if f0 is None or f32 is None:
            continue
        y = b["y_true"].to_numpy(int)
        terc = tercile_labels(b["dis_mi"].to_numpy(float))
        ps = (f0["p_yes"].reindex(b.index).to_numpy() >= 0.5).astype(int)
        pl = (f32["p_yes"].reindex(b.index).to_numpy() >= 0.5).astype(int)
        r = dasym_test(pl, ps, y, terc == 0, terc == 2)
        r["mean_n_valid_0p5b"] = round(float(f0["n_valid"].mean()), 3)
        r["mean_n_valid_32b"] = round(float(f32["n_valid"].mean()), 3)
        r["yes_rate_0p5b"] = round(float(ps.mean()), 4)
        r["yes_rate_32b"] = round(float(pl.mean()), 4)
        r["balacc_all_32b"] = round(bal_acc(pl, y), 4)
        res["native"][c] = r
        t_ = TAG[c]
        sig = r["ci95"][0] > 0
        M[f"NatDA{t_}"] = f3(r["dasym"]) + ("$^\\dagger$" if sig else "")
        M[f"NatDACI{t_}"] = ci(r["ci95"])

    # ---------------- OLMo-2 family ----------------
    for c in CORPORA:
        b = base_frame(c)
        fr = {t: load(GPU / f"fam_{t}_{c}.csv") for t in OLMO}
        if any(v is None for v in fr.values()):
            continue
        y = b["y_true"].to_numpy(int)
        terc = tercile_labels(b["dis_mi"].to_numpy(float))
        low, high = terc == 0, terc == 2
        P = {t: (fr[t]["p_yes"].reindex(b.index).to_numpy() >= 0.5).astype(int) for t in OLMO}
        r = {"primary_1b_32b": dasym_test(P["o32b"], P["o1b"], y, low, high),
             "endpoint_7b_32b": dasym_test(P["o32b"], P["o7b"], y, low, high),
             "slope": slope_test(P, y, low, high, OLMO),
             "yes_rate": {t: round(float(P[t].mean()), 4) for t in OLMO},
             "mean_n_valid": {t: round(float(fr[t]["n_valid"].mean()), 3) for t in OLMO},
             "balacc_all": {t: round(bal_acc(P[t], y), 4) for t in OLMO}}
        # contraction probe within OLMo-2: |p7B - p1B| for errors of 1B fixed by 32B
        p1 = fr["o1b"]["p_yes"].reindex(b.index).to_numpy(float)
        p7 = fr["o7b"]["p_yes"].reindex(b.index).to_numpy(float)
        yfix = ((P["o1b"] != y) & (P["o32b"] == y)).astype(int)
        r["probe_all"] = boot_auroc(yfix, np.abs(p7 - p1))
        res["family"][c] = r
        t_ = TAG[c]
        sig = r["primary_1b_32b"]["ci95"][0] > 0
        M[f"FamDA{t_}"] = f3(r["primary_1b_32b"]["dasym"]) + ("$^\\dagger$" if sig else "")
        M[f"FamDACI{t_}"] = ci(r["primary_1b_32b"]["ci95"])
        sig = r["slope"]["ci95"][0] > 0
        M[f"FamSlope{t_}"] = f3(r["slope"]["slope_diff"]) + ("$^\\dagger$" if sig else "")
        M[f"FamSlopeCI{t_}"] = ci(r["slope"]["ci95"])
        M[f"FamEnd{t_}"] = f3(r["endpoint_7b_32b"]["dasym"])
        M[f"FamEndCI{t_}"] = ci(r["endpoint_7b_32b"]["ci95"])
        pa = r["probe_all"]
        M[f"FamProbe{t_}"] = f3(pa["AUROC"]) + ("$^\\dagger$" if (pa["ci95"][0] > 0.5 or pa["ci95"][1] < 0.5) else "")
        for i, t in enumerate(OLMO):
            M[f"FamBALow{t_}{'ABCD'[i]}"] = f3(r["slope"]["balacc_low"][i])
            M[f"FamBAHigh{t_}{'ABCD'[i]}"] = f3(r["slope"]["balacc_high"][i])
            M[f"FamYes{t_}{'ABCD'[i]}"] = f3(r["yes_rate"][t])

    # ---------------- threshold-free variant: DeltaAsym on within-stratum AUROC ----------------
    # AUROC of p_yes against the reference label in each stratum; insensitive to an output
    # bias that shifts every p_yes below the 0.5 decision threshold.
    def auc_dasym(ps, pl, y, t):
        def stat(ix):
            lo, hi = ix[t[ix] == 0], ix[t[ix] == 2]
            return (auroc(y[lo], pl[lo]) - auroc(y[lo], ps[lo])) - (auroc(y[hi], pl[hi]) - auroc(y[hi], ps[hi]))
        n = len(y)
        obs = stat(np.arange(n))
        rng = np.random.default_rng(SEED)
        bs = np.array([stat(rng.integers(0, n, n)) for _ in range(B_TEST)])
        return {"dasym_auroc": round(float(obs), 4),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}

    res["auroc_variant"] = {}
    for c in CORPORA:
        b = base_frame(c)
        y = b["y_true"].to_numpy(int)
        t = tercile_labels(b["dis_mi"].to_numpy(float))
        row = {}
        for fam_name, fs, fl in (("OLMo2", GPU / f"fam_o1b_{c}.csv", GPU / f"fam_o32b_{c}.csv"),
                                 ("Qwen25", REGEN / f"c7_q0p5b_{c}.csv", REGEN / f"c7_q32b_{c}.csv")):
            s, l_ = load(fs), load(fl)
            if s is None or l_ is None:
                continue
            row[fam_name] = auc_dasym(s["p_yes"].reindex(b.index).to_numpy(float),
                                      l_["p_yes"].reindex(b.index).to_numpy(float), y, t)
        res["auroc_variant"][c] = row
        t_ = TAG[c]
        for fam_name, key in (("OLMo2", "FamAucDA"), ("Qwen25", "QwenAucDA")):
            if fam_name in row:
                r = row[fam_name]
                M[f"{key}{t_}"] = f3(r["dasym_auroc"]) + ("$^\\dagger$" if r["ci95"][0] > 0 else "")
                M[f"{key}CI{t_}"] = ci(r["ci95"])
    fa = [v["OLMo2"] for v in res["auroc_variant"].values() if "OLMo2" in v]
    if len(fa) == len(CORPORA):
        sig = [r["dasym_auroc"] for r in fa if r["ci95"][0] > 0]
        M["FamAucNsig"] = str(len(sig))
        M["FamAucSigMin"], M["FamAucSigMax"] = f3(min(sig)), f3(max(sig))

    # ---------------- summary ranges used in the prose ----------------
    bin_c = CORPORA[1:]
    if res["seeds"] and all("probe_all_auroc" in v for c in CORPORA for k, v in res["seeds"].get(c, {}).items()
                            if isinstance(v, dict)):
        pr = [v["probe_all_auroc"] for c in CORPORA for v in res["seeds"].get(c, {}).values() if isinstance(v, dict)]
        M["SeedProbeMin"], M["SeedProbeMax"] = f3(min(pr)), f3(max(pr))
    if res["native"]:
        nat = res["native"]
        M["NatDAMin"] = f3(min(v["dasym"] for v in nat.values()))
        M["NatDAMax"] = f3(max(v["dasym"] for v in nat.values()))
        M["NatCILoMin"] = f3(min(v["ci95"][0] for v in nat.values()))
        M["NatYesSmallMax"] = f3(max(v["yes_rate_0p5b"] for v in nat.values()))
        M["NatValidMin"] = f"{min(min(v['mean_n_valid_0p5b'], v['mean_n_valid_32b']) for v in nat.values()):.2f}"
    fam = res["family"]
    if len(fam) == len(CORPORA):
        yr = [fam[c]["yes_rate"][t] for c in bin_c for t in ("o1b", "o7b", "o32b")]
        M["FamYesBinMax"] = f3(max(yr))
        y13 = [fam[c]["yes_rate"]["o13b"] for c in bin_c]
        M["FamYesThirteenMin"], M["FamYesThirteenMax"] = f3(min(y13)), f3(max(y13))
        bl = [fam[c]["slope"]["balacc_low"][3] for c in bin_c]
        M["FamBALowLargeBinMin"], M["FamBALowLargeBinMax"] = f3(min(bl)), f3(max(bl))
        pa = [fam[c]["probe_all"]["AUROC"] for c in CORPORA]
        M["FamProbeMin"], M["FamProbeMax"] = f3(min(pa)), f3(max(pa))
        M["FamYesLargeCSC"] = f3(fam["CSC"]["yes_rate"]["o32b"])
        M["FamNsigDA"] = str(sum(fam[c]["primary_1b_32b"]["ci95"][0] > 0 for c in CORPORA))
        M["FamNsigSlope"] = str(sum(fam[c]["slope"]["ci95"][0] > 0 for c in CORPORA))
    res["macros"] = M
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT, len(M), "macros")


if __name__ == "__main__":
    main()
