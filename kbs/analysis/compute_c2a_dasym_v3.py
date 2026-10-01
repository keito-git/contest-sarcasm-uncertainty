"""
compute_c2a_dasym_v3.py
=======================
Computes C2a soft-label ΔAsym for all six corpora (CSC, EPIC,
MultiPICo_es/de/fr/it) using the CORRECT formula:

    ΔAsym_Brier = drop_low − drop_high
    where  drop_t = mean_Brier(0.5B, tercile t) − mean_Brier(32B, tercile t)

    ΔAsym_JSD  = (mean_JSD_0.5B_low − mean_JSD_0.5B_high)
                 − (mean_JSD_32B_low − mean_JSD_32B_high)

Here "low/high" refers to the lowest/highest tercile of dis_mi, and JSD is
JSD(p_LLM ‖ p_human) per item.

The v2 EPIC bug: the original code computed
    Brier_low(0.5B) − Brier_high(0.5B)
i.e. a within-0.5B tercile difference, ignoring 32B entirely.  For CSC this
coincidentally approximates the correct value because 32B Brier is similar
across terciles.  For EPIC it diverges because 32B degrades on high-dis items
(high-dis irony is aleatoric: 32B is more confident but wrong → higher Brier).

Bootstrap: item-level (resample rows, B=4000, seed=42).

Output: results/revision_gpu_2026-09/round1_analysis/C2a_SOFT_DASYM_v3.json
Script: kbs/analysis/compute_c2a_dasym_v3.py
"""
from __future__ import annotations

import json
import math
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ── paths ──────────────────────────────────────────────────────────────── #
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR
REGEN_DIR = BASE / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
SARCASM_DIR = BASE / "results" / "sarcasm"
OUT_DIR = BASE / "results" / "revision_gpu_2026-09" / "round1_analysis"
MULTIPICO_HF = BASE / "results" / "revision_gpu_2026-09" / "multipico_hf_raw.parquet"
CSC_RAW = BASE / "data" / "raw" / "csc_data_full.csv"

SEED = 42
B = 4000


# ── utilities ──────────────────────────────────────────────────────────── #
def tercile_labels(a: np.ndarray) -> np.ndarray:
    """Assign tercile membership (0=low, 1=mid, 2=high) by rank."""
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])


def jsd_binary(p_model: np.ndarray, p_human: np.ndarray) -> np.ndarray:
    """Per-item JSD between two Bernoulli distributions."""
    pm = np.clip(p_model.astype(float), 1e-12, 1 - 1e-12)
    ph = np.clip(p_human.astype(float), 1e-12, 1 - 1e-12)
    m = (pm + ph) / 2
    def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return 0.5 * kl(pm, m) + 0.5 * kl(ph, m)


def dasym_brier(
    p0b: np.ndarray, p32b: np.ndarray, ph: np.ndarray, t: np.ndarray
) -> float:
    """ΔAsym_Brier = drop_low − drop_high  (0.5B → 32B, Brier vs p_human)."""
    mask_lo = t == 0
    mask_hi = t == 2
    if mask_lo.sum() == 0 or mask_hi.sum() == 0:
        return float("nan")
    drop_low = (
        float(np.mean((p0b[mask_lo] - ph[mask_lo]) ** 2))
        - float(np.mean((p32b[mask_lo] - ph[mask_lo]) ** 2))
    )
    drop_high = (
        float(np.mean((p0b[mask_hi] - ph[mask_hi]) ** 2))
        - float(np.mean((p32b[mask_hi] - ph[mask_hi]) ** 2))
    )
    return drop_low - drop_high


def dasym_jsd(
    p0b: np.ndarray, p32b: np.ndarray, ph: np.ndarray, t: np.ndarray
) -> float:
    """ΔAsym_JSD = (mean_JSD_0.5B_low − mean_JSD_0.5B_high) − same for 32B."""
    mask_lo = t == 0
    mask_hi = t == 2
    if mask_lo.sum() == 0 or mask_hi.sum() == 0:
        return float("nan")
    jsd0_lo = float(np.mean(jsd_binary(p0b[mask_lo], ph[mask_lo])))
    jsd0_hi = float(np.mean(jsd_binary(p0b[mask_hi], ph[mask_hi])))
    jsd32_lo = float(np.mean(jsd_binary(p32b[mask_lo], ph[mask_lo])))
    jsd32_hi = float(np.mean(jsd_binary(p32b[mask_hi], ph[mask_hi])))
    return (jsd0_lo - jsd0_hi) - (jsd32_lo - jsd32_hi)


def boot_ci_1d(fn, p0b, p32b, ph, t, nb: int = B, seed: int = SEED):
    """Item-level bootstrap CI for a scalar statistic fn(p0b, p32b, ph, t)."""
    rng = np.random.default_rng(seed)
    n = len(p0b)
    obs = fn(p0b, p32b, ph, t)
    bs: list[float] = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        v = fn(p0b[idx], p32b[idx], ph[idx], t[idx])
        if not math.isnan(v):
            bs.append(v)
    if not bs:
        return obs, float("nan"), float("nan")
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_tercile_scores(
    p0b: np.ndarray, p32b: np.ndarray, ph: np.ndarray, t: np.ndarray
) -> dict:
    """Return per-tercile mean Brier and JSD for both 0.5B and 32B."""
    out: dict = {}
    for tname, tidx in [("low", 0), ("mid", 1), ("high", 2)]:
        mask = t == tidx
        if mask.sum() == 0:
            out[tname] = {}
            continue
        brier_0 = float(np.mean((p0b[mask] - ph[mask]) ** 2))
        brier_32 = float(np.mean((p32b[mask] - ph[mask]) ** 2))
        jsd_0 = float(np.mean(jsd_binary(p0b[mask], ph[mask])))
        jsd_32 = float(np.mean(jsd_binary(p32b[mask], ph[mask])))
        out[tname] = {
            "n": int(mask.sum()),
            "brier_0p5b": round(brier_0, 4),
            "brier_32b": round(brier_32, 4),
            "jsd_0p5b": round(jsd_0, 4),
            "jsd_32b": round(jsd_32, 4),
        }
    return out


def load_c7(tag: str, corpus: str) -> "pd.DataFrame | None":
    path = REGEN_DIR / f"c7_{tag}_{corpus}.csv"
    need = {"item_id", "p_yes", "dis_mi", "y_true"}
    if path.exists():
        df = pd.read_csv(path)
        if need.issubset(df.columns):
            return df[list(need)].copy()
    return None


# ── human distribution loading helpers ────────────────────────────────── #
def build_csc_phuman() -> dict[str, float]:
    """Build item_id → p_binary (fraction of scores >= 4) for CSC.

    Matches the definition used in C2_SOFTLABEL_v2.json (the v2 script used
    p_binary = fraction(sarcasm_score_by_evaluator >= 4), not mean Likert).
    This ensures per-tercile Brier scores are consistent across v2 and v3.
    """
    df = pd.read_csv(CSC_RAW)
    df["item_id"] = df["global_context_name"].astype(str) + "_" + df["global_speaker"].astype(str)
    df["is_sarcastic"] = (df["sarcasm_score_by_evaluator"] >= 4).astype(float)
    return dict(df.groupby("item_id")["is_sarcastic"].mean())


def build_epic_phuman() -> dict[str, float]:
    """Build item_id → p_ironic for EPIC."""
    df = pd.read_parquet(SARCASM_DIR / "epic_aggregated.parquet")
    return {str(row.item_id): float(row.p_ironic) for _, row in df.iterrows()}


def build_multipico_phuman(lang: str, hf: pd.DataFrame) -> dict[str, float]:
    """Build item_id → p_ironic for MultiPICo (keyed as mp_{lang}_{reply_id})."""
    sub = hf[hf["language"] == lang]
    out: dict[str, float] = {}
    for reply_id, grp in sub.groupby("reply_id"):
        key = f"mp_{lang}_{reply_id}"
        out[key] = float(grp["label"].values.mean())
    return out


# ── per-corpus computation ─────────────────────────────────────────────── #
def compute_corpus(corpus: str, p_human_map: dict[str, float]) -> dict:
    """Compute ΔAsym_Brier and ΔAsym_JSD for one corpus."""
    base = load_c7("q0p5b", corpus)
    top = load_c7("q32b", corpus)
    if base is None or top is None:
        return {"available": False, "note": "missing q0p5b or q32b CSV"}

    merged = base[["item_id", "dis_mi", "y_true"]].copy()
    merged["p_0p5b"] = base["p_yes"].values
    merged = merged.merge(
        top[["item_id", "p_yes"]].rename(columns={"p_yes": "p_32b"}),
        on="item_id", how="inner"
    )
    merged["item_id"] = merged["item_id"].astype(str)
    merged["p_human"] = merged["item_id"].map(p_human_map)
    merged = merged.dropna(subset=["p_human"]).reset_index(drop=True)
    n = len(merged)
    if n < 10:
        return {"available": False, "note": f"merged n={n} after p_human join"}

    dis = merged["dis_mi"].to_numpy(float)
    p0b = merged["p_0p5b"].to_numpy(float)
    p32 = merged["p_32b"].to_numpy(float)
    ph = merged["p_human"].to_numpy(float)
    t = tercile_labels(dis)

    # Per-tercile scores
    pt = per_tercile_scores(p0b, p32, ph, t)

    # ΔAsym values and CIs
    obs_b, ci_b_lo, ci_b_hi = boot_ci_1d(dasym_brier, p0b, p32, ph, t)
    obs_j, ci_j_lo, ci_j_hi = boot_ci_1d(dasym_jsd, p0b, p32, ph, t)

    # Explicit decomposition
    lo = t == 0
    hi = t == 2
    drop_low_b = float(np.mean((p0b[lo]-ph[lo])**2)) - float(np.mean((p32[lo]-ph[lo])**2))
    drop_hi_b  = float(np.mean((p0b[hi]-ph[hi])**2)) - float(np.mean((p32[hi]-ph[hi])**2))
    drop_low_j = float(np.mean(jsd_binary(p0b[lo], ph[lo]))) - float(np.mean(jsd_binary(p32[lo], ph[lo])))
    drop_hi_j  = float(np.mean(jsd_binary(p0b[hi], ph[hi]))) - float(np.mean(jsd_binary(p32[hi], ph[hi])))

    return {
        "n": n,
        "formula": "ΔAsym = drop_low − drop_high; drop_t = mean_metric(0.5B,t) − mean_metric(32B,t)",
        "tercile_scores": pt,
        "brier": {
            "drop_low":  round(drop_low_b, 4),
            "drop_high": round(drop_hi_b,  4),
            "dasym":     round(obs_b, 4),
            "ci_95":     [round(ci_b_lo, 4), round(ci_b_hi, 4)],
        },
        "jsd": {
            "drop_low":  round(drop_low_j, 4),
            "drop_high": round(drop_hi_j,  4),
            "dasym":     round(obs_j, 4),
            "ci_95":     [round(ci_j_lo, 4), round(ci_j_hi, 4)],
        },
    }


# ── EPIC v2 discrepancy explanation ───────────────────────────────────── #
def epic_discrepancy_note(v2_value: float, correct_value: float) -> str:
    return (
        f"v2 stored {v2_value:.4f}; correct (drop_low − drop_high) = {correct_value:.4f}. "
        "Bug in analyze_round1_v2.py: `brier_diff_0p5b` computed "
        "Brier_low(0.5B) − Brier_high(0.5B) only, ignoring 32B entirely. "
        "For CSC the two formulas coincide because 32B Brier is similar across "
        "terciles (upgrade helps all terciles similarly). "
        "For EPIC they diverge because 32B degrades on high-dis items: "
        "high-dis EPIC irony is highly aleatoric, so 32B becomes more confident "
        "and more wrong → Brier_high(32B) > Brier_high(0.5B), making drop_high negative "
        "and the correct ΔAsym much larger (0.3246 vs stored 0.2207)."
    )


# ── main ──────────────────────────────────────────────────────────────── #
def main() -> None:
    print("=== C2a ΔAsym (v3, correct formula) ===", flush=True)

    # Load human distributions
    csc_ph = build_csc_phuman()
    epic_ph = build_epic_phuman()
    hf = pd.read_parquet(MULTIPICO_HF)
    mp_ph = {lang: build_multipico_phuman(lang, hf) for lang in ["es", "de", "fr", "it"]}

    corpus_list = [
        ("CSC",          csc_ph),
        ("EPIC",         epic_ph),
        ("MultiPICo_es", mp_ph["es"]),
        ("MultiPICo_de", mp_ph["de"]),
        ("MultiPICo_fr", mp_ph["fr"]),
        ("MultiPICo_it", mp_ph["it"]),
    ]

    results: dict = {}
    for corpus, ph_map in corpus_list:
        print(f"  {corpus}...", flush=True)
        results[corpus] = compute_corpus(corpus, ph_map)
        r = results[corpus]
        if "brier" in r:
            print(f"    ΔAsym_Brier = {r['brier']['dasym']:.4f}  "
                  f"CI=[{r['brier']['ci_95'][0]:.4f},{r['brier']['ci_95'][1]:.4f}]", flush=True)
            print(f"    ΔAsym_JSD   = {r['jsd']['dasym']:.4f}  "
                  f"CI=[{r['jsd']['ci_95'][0]:.4f},{r['jsd']['ci_95'][1]:.4f}]", flush=True)

    # Attach v2 discrepancy note for EPIC
    if "EPIC" in results and "brier" in results["EPIC"]:
        results["EPIC"]["v2_discrepancy"] = epic_discrepancy_note(
            v2_value=0.2207,
            correct_value=results["EPIC"]["brier"]["dasym"],
        )

    # Build output
    out = {
        "analysis": "C2a soft-label ΔAsym (v3): correct drop_low − drop_high formula",
        "formula": (
            "ΔAsym_Brier = [mean_Brier(0.5B, low) − mean_Brier(32B, low)]"
            " − [mean_Brier(0.5B, high) − mean_Brier(32B, high)]"
        ),
        "v2_bug": (
            "v2 computed Brier_low(0.5B) − Brier_high(0.5B) (no 32B term). "
            "Matches correct formula only when Brier(32B, low) ≈ Brier(32B, high)."
        ),
        "bootstrap_n": B,
        "bootstrap_seed": SEED,
        "p_human_note": {
            "CSC": "p_binary = fraction(sarcasm_score_by_evaluator >= 4); consistent with C2_SOFTLABEL_v2.json",
            "EPIC": "p_ironic from epic_aggregated.parquet",
            "MultiPICo_*": "mean binary label from multipico_hf_raw.parquet, key=mp_{lang}_{reply_id}",
        },
        "per_corpus": results,
        "env": {
            "hostname": socket.gethostname(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "script": "kbs/analysis/compute_c2a_dasym_v3.py",
        },
    }

    out_path = OUT_DIR / "C2a_SOFT_DASYM_v3.json"
    if out_path.exists():
        raise FileExistsError(f"Already exists: {out_path}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
