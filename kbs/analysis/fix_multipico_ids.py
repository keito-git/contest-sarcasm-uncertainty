"""
fix_multipico_ids.py  —  Fix C4 (MultiPICo instability) and C2 (MultiPICo human dist).

Problem: item_id in c7 CSV is "mp_de_32863009042841225" but ann_lookup keys
were just str(reply_id). Fix: use "mp_{lang}_{reply_id}" as the key.

Also updates C2_SOFTLABEL_v2.json with correct MultiPICo_de/fr/it data.
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR
REGEN_DIR = BASE / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
SARCASM_DIR = BASE / "results" / "sarcasm"
OUT_DIR = BASE / "results" / "revision_gpu_2026-09" / "round1_analysis"
MULTIPICO_HF = BASE / "results" / "revision_gpu_2026-09" / "multipico_hf_raw.parquet"

SEED = 42
B = 4000
N_INST = 200
BUDGETS = np.linspace(0, 1, 21)


def binary_entropy_scalar(p: float) -> float:
    p = max(1e-12, min(1 - 1e-12, p))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def tercile_labels(a: np.ndarray) -> np.ndarray:
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append(float((pred[m] == c).mean()))
    return float(np.mean(recs)) if recs else float("nan")


def boot_ci(stat_fn, data: tuple, nb: int = B, seed: int = SEED):
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(d) for d in data]
    n = len(arrs[0])
    obs = stat_fn(*arrs)
    bs = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        v = stat_fn(*[a[idx] for a in arrs])
        if not math.isnan(v):
            bs.append(v)
    if not bs:
        return obs, float("nan"), float("nan")
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_c7(tag: str, corpus: str) -> "pd.DataFrame | None":
    path = REGEN_DIR / f"c7_{tag}_{corpus}.csv"
    need = {"item_id", "p_yes", "n_valid", "H", "dis_mi", "y_true"}
    if path.exists():
        df = pd.read_csv(path)
        if need.issubset(df.columns):
            return df[list(need)].copy()
    return None


def jsd_binary_arr(p_model: np.ndarray, p_human: np.ndarray) -> np.ndarray:
    m = (p_model + p_human) / 2
    m = np.clip(m, 1e-12, 1 - 1e-12)
    pm = np.clip(p_model, 1e-12, 1 - 1e-12)
    ph = np.clip(p_human, 1e-12, 1 - 1e-12)
    def _kl(p, q):
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return 0.5 * _kl(pm, m) + 0.5 * _kl(ph, m)


# ── Rebuild C4 for MultiPICo with correct item_id keys ────────────────────── #
def fix_c4_multipico() -> dict:
    print("=== Fixing C4 MultiPICo (item_id key fix) ===", flush=True)

    # Load existing C4
    c4_path = OUT_DIR / "C4_INSTABILITY_v2.json"
    with open(c4_path) as f:
        c4 = json.load(f)

    hf = pd.read_parquet(MULTIPICO_HF)

    def binary_dis(labels: np.ndarray) -> float:
        if len(labels) == 0: return 0.0
        return binary_entropy_scalar(float(labels.mean()))

    def dasym_fn(ps, pl, y_arr, t):
        dal = bal_acc((pl[t==0]>=0.5).astype(int), y_arr[t==0]) - \
              bal_acc((ps[t==0]>=0.5).astype(int), y_arr[t==0])
        dah = bal_acc((pl[t==2]>=0.5).astype(int), y_arr[t==2]) - \
              bal_acc((ps[t==2]>=0.5).astype(int), y_arr[t==2])
        return dal - dah

    for lang in ["es", "de", "fr", "it"]:
        corpus = f"MultiPICo_{lang}"
        print(f"  {corpus}...", flush=True)

        # Build ann_lookup with key = "mp_{lang}_{reply_id}" to match c7 item_id
        sub = hf[hf["language"] == lang]
        ann_lookup: dict[str, np.ndarray] = {}
        for reply_id, grp in sub.groupby("reply_id"):
            key = f"mp_{lang}_{reply_id}"
            ann_lookup[key] = grp["label"].values.astype(float)

        df_base = load_c7("q0p5b", corpus)
        df_32b = load_c7("q32b", corpus)
        if df_base is None:
            continue

        item_ids = df_base["item_id"].astype(str).tolist()
        dis_obs = df_base["dis_mi"].to_numpy(float)

        # Check coverage
        n_found = sum(1 for iid in item_ids if iid in ann_lookup)
        print(f"    {corpus}: ann_lookup coverage = {n_found}/{len(item_ids)}", flush=True)

        # Compute instability
        rng = np.random.default_rng(SEED)
        terc_obs = tercile_labels(dis_obs)
        n_items = len(item_ids)
        change_frac_total = 0.0

        for _ in range(N_INST):
            dis_new = np.empty(n_items, float)
            for i, iid in enumerate(item_ids):
                scores = ann_lookup.get(iid)
                if scores is None or len(scores) == 0:
                    dis_new[i] = dis_obs[i]
                    continue
                resampled = rng.choice(scores, size=len(scores), replace=True)
                dis_new[i] = binary_dis(resampled)
            terc_new = tercile_labels(dis_new)
            change_frac_total += float((terc_new != terc_obs).mean())
        frac_unstable = change_frac_total / N_INST
        print(f"    {corpus}: frac_unstable = {frac_unstable:.4f}", flush=True)

        # ΔAsym with ki filters
        if df_32b is not None:
            merged = df_base[["item_id", "dis_mi", "y_true"]].copy()
            merged["p_0p5b"] = df_base["p_yes"].to_numpy()
            merged = merged.merge(
                df_32b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_32b"}),
                on="item_id", how="inner"
            )
            n_full = len(merged)
            dis_f  = merged["dis_mi"].to_numpy(float)
            y_f    = merged["y_true"].to_numpy(int)
            p0b_f  = merged["p_0p5b"].to_numpy(float)
            p32_f  = merged["p_32b"].to_numpy(float)
            terc_f = tercile_labels(dis_f)

            ki_filters: dict = {}
            for ki_min, fname in [(None, "all"), (4, "ki_ge4"), (5, "ki_ge5")]:
                if ki_min is None:
                    mask = np.ones(n_full, bool)
                else:
                    keep = []
                    for iid in merged["item_id"].astype(str).tolist():
                        s = ann_lookup.get(iid)
                        keep.append((len(s) if s is not None else 0) >= ki_min)
                    mask = np.array(keep, bool)

                if mask.sum() < 10:
                    ki_filters[fname] = {"n": int(mask.sum()), "note": "too few"}
                    continue

                obs_da, ci_lo, ci_hi = boot_ci(
                    dasym_fn, (p0b_f[mask], p32_f[mask], y_f[mask], terc_f[mask])
                )
                rng_pv = np.random.default_rng(SEED + 7)
                nm = mask.sum()
                cnt = sum(
                    1 for _ in range(B)
                    if dasym_fn(
                        p0b_f[mask][rng_pv.integers(0, nm, nm)],
                        p32_f[mask][rng_pv.integers(0, nm, nm)],
                        y_f[mask][rng_pv.integers(0, nm, nm)],
                        terc_f[mask][rng_pv.integers(0, nm, nm)]
                    ) <= 0
                )
                ki_filters[fname] = {
                    "n": int(nm), "asym_acc": round(obs_da, 4),
                    "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                    "p_boot_raw": round(cnt / B, 4), "n_bootstrap": B,
                }
        else:
            ki_filters = {"note": "no 32B data"}
            n_full = len(df_base)

        c4["per_corpus"][corpus] = {
            "n": n_full,
            "stratum_instability": {
                "n_resamplings": N_INST,
                "frac_changing_tercile": round(frac_unstable, 4),
                "note": "annotator-level resampling from multipico_hf_raw.parquet (item_id fixed)",
                "source": "multipico_hf_raw.parquet",
                "item_id_key_format": f"mp_{lang}_{{reply_id}}",
            },
            "dasym_by_ki_filter": ki_filters,
        }

    # Overwrite C4_INSTABILITY_v2.json (allowed since we're updating our own output)
    with open(c4_path, "w") as f:
        json.dump(c4, f, indent=2, ensure_ascii=False)
    print(f"  Updated: {c4_path}", flush=True)
    return c4


# ── Rebuild C2 for MultiPICo de/fr/it with correct item_id keys ───────────── #
def fix_c2_multipico() -> dict:
    print("=== Fixing C2 MultiPICo de/fr/it (item_id key fix) ===", flush=True)

    c2_path = OUT_DIR / "C2_SOFTLABEL_v2.json"
    with open(c2_path) as f:
        c2 = json.load(f)

    hf = pd.read_parquet(MULTIPICO_HF)

    def dasym_fn(ps, pl, y_arr, t):
        dal = bal_acc((pl[t==0]>=0.5).astype(int), y_arr[t==0]) - \
              bal_acc((ps[t==0]>=0.5).astype(int), y_arr[t==0])
        dah = bal_acc((pl[t==2]>=0.5).astype(int), y_arr[t==2]) - \
              bal_acc((ps[t==2]>=0.5).astype(int), y_arr[t==2])
        return dal - dah

    for lang in ["es", "de", "fr", "it"]:
        corpus = f"MultiPICo_{lang}"
        print(f"  {corpus}...", flush=True)

        # Build per-item human p_ironic with key = "mp_{lang}_{reply_id}"
        sub = hf[hf["language"] == lang]
        p_human_map: dict[str, float] = {}
        for reply_id, grp in sub.groupby("reply_id"):
            key = f"mp_{lang}_{reply_id}"
            p_human_map[key] = float(grp["label"].values.mean())

        # Load LLM data
        dfs_llm = {}
        for tag in ["q0p5b", "q1p5b", "q3b", "q7b", "q14b", "q32b"]:
            path = REGEN_DIR / f"c7_{tag}_{corpus}.csv"
            need = {"item_id", "p_yes", "n_valid", "H", "dis_mi", "y_true"}
            if path.exists():
                df = pd.read_csv(path)
                if need.issubset(df.columns):
                    dfs_llm[tag] = df[list(need)].copy()

        if not dfs_llm:
            continue

        base_tag = "q0p5b"
        merged = dfs_llm[base_tag].rename(columns={"p_yes": f"p_{base_tag}"})
        merged = merged[["item_id", "dis_mi", "y_true", f"p_{base_tag}"]].copy()
        for tag, df_tag in dfs_llm.items():
            if tag != base_tag:
                merged = merged.merge(
                    df_tag[["item_id", "p_yes"]].rename(columns={"p_yes": f"p_{tag}"}),
                    on="item_id", how="left"
                )
        merged["item_id"] = merged["item_id"].astype(str)
        merged["p_human"] = merged["item_id"].map(p_human_map)
        merged = merged.dropna(subset=["p_human"])
        n = len(merged)
        print(f"    {corpus}: merged n={n}", flush=True)

        if n < 10:
            c2["per_corpus"][corpus] = {"available": False, "note": f"merged n={n}"}
            continue

        dis = merged["dis_mi"].to_numpy(float)
        y = merged["y_true"].to_numpy(int)
        p_human = merged["p_human"].to_numpy(float)
        terc = tercile_labels(dis)

        # C2a
        c2a: dict = {}
        for tag in dfs_llm.keys():
            col = f"p_{tag}"
            if col not in merged.columns:
                continue
            p_llm = merged[col].to_numpy(float)
            c2a_tag: dict = {}
            for tname, tidx in [("low", 0), ("mid", 1), ("high", 2)]:
                mask = terc == tidx
                if mask.sum() == 0:
                    c2a_tag[tname] = {}
                    continue
                brier = float(np.mean((p_llm[mask] - p_human[mask]) ** 2))
                jsd_vals = jsd_binary_arr(p_llm[mask], p_human[mask])
                c2a_tag[tname] = {
                    "n": int(mask.sum()),
                    "brier": round(brier, 4),
                    "jsd": round(float(np.mean(jsd_vals)), 4),
                }
            c2a[tag] = c2a_tag

        # C2b null
        rng = np.random.default_rng(SEED)
        p0b = merged.get("p_q0p5b") if "p_q0p5b" in merged.columns else None
        if p0b is None and "p_q0p5b" in merged.columns:
            p0b = merged["p_q0p5b"].to_numpy(float)
        elif "p_q0p5b" in merged.columns:
            p0b = merged["p_q0p5b"].to_numpy(float)
        p32b_arr = merged["p_q32b"].to_numpy(float) if "p_q32b" in merged.columns else None

        observed_dasym = None
        if p0b is not None and p32b_arr is not None:
            observed_dasym = round(dasym_fn(p0b, p32b_arr, y, terc), 4)

        null_vals = []
        for _ in range(1000):
            shuffled = rng.permutation(terc)
            if p0b is not None and p32b_arr is not None:
                null_vals.append(dasym_fn(p0b, p32b_arr, y, shuffled))
        null_arr = np.array(null_vals)
        p_null = float(np.mean(null_arr >= observed_dasym)) if observed_dasym is not None and len(null_vals) > 0 else float("nan")

        c2["per_corpus"][corpus] = {
            "n": n,
            "p_human_source": "real_annotators_from_multipico_hf_raw_parquet",
            "p_human_col_used": "p_ironic",
            "dis_deterministic_of_p_human": (
                "dis_mi = H(p_human): deterministic function of human proportion"
            ),
            "c2a_soft_scores": c2a,
            "c2b_null": {
                "observed_dasym_hard": observed_dasym,
                "null_dasym": {
                    "mean": round(float(null_arr.mean()), 4) if len(null_vals) > 0 else None,
                    "std": round(float(null_arr.std()), 4) if len(null_vals) > 0 else None,
                    "p2.5": round(float(np.percentile(null_arr, 2.5)), 4) if len(null_vals) > 0 else None,
                    "p97.5": round(float(np.percentile(null_arr, 97.5)), 4) if len(null_vals) > 0 else None,
                },
                "p_null_exceeds_observed": round(p_null, 4),
                "n_sims": len(null_vals),
            },
        }

    with open(c2_path, "w") as f:
        json.dump(c2, f, indent=2, ensure_ascii=False)
    print(f"  Updated: {c2_path}", flush=True)
    return c2


if __name__ == "__main__":
    c4 = fix_c4_multipico()
    c2 = fix_c2_multipico()

    # Print key C4 numbers
    print("\n=== C4 instability summary (corrected) ===")
    for corpus, cdata in c4["per_corpus"].items():
        si = cdata.get("stratum_instability", {})
        print(f"  {corpus}: {si.get('frac_changing_tercile', 'N/A')}")

    print("\nDone.")
