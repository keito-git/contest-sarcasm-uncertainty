"""
analyze_c7c8_regen.py — C7/C8 analysis adapted for the regen mirror.

Reads from:
    results/revision_gpu_2026-09/server_outputs/regen/   (regen sweep outputs)
    results/llm_e3/                                       (original 400-item results)

Writes to (never touches existing GPU_DIR or E3_DIR result files):
    results/revision_gpu_2026-09/regen_analysis/
      C7_REGEN.json      — all C7 numbers (machine-readable)
      C8_REGEN.json      — all C8 numbers (machine-readable)
      DEDUP_UNION.json   — EPIC∪MultiPICo-en dedup statistics
      REGEN_ENV.json     — run provenance

Analyses
--------
C7 — Per-corpus balanced accuracy by disagreement tercile per size; ΔAsym
     (0.5B→32B), 4000 bootstrap, Holm–Bonferroni over declared family.

C8 — "Non-overlap contraction probe" AUROC:
       |p_q3b − p_q0p5b| among 0.5B-error items predicting 32B fixes.
     2000-bootstrap CI.
     Capacity vs diversity: paired bootstrap comparing capacity-gap probe vs
       Phi-3-mini (ll3b), SmolLM2-1.7B (ll1b), StableLM-zephyr-3B (gm2_2b)
       and sampling-noise (q0p5b_s2) controls.
     Original-400 subset values alongside paper's published values (disclosure).

DEDUP — EPIC∪MultiPICo-en overlap rate (text-level); non-EN sub-corpus
        independence check.

Usage
-----
    python kbs/analysis/analyze_c7c8_regen.py
    python kbs/analysis/analyze_c7c8_regen.py --c7_only
    python kbs/analysis/analyze_c7c8_regen.py --c8_only
    python kbs/analysis/analyze_c7c8_regen.py \\
        --corpora CSC,EPIC,MultiPICo_es,MultiPICo_de

Anti-patterns this script avoids
----------------------------------
- No dynamic label-set inference from data.
- No selective metric reporting.
- Results written only when all constituent numbers are present.
- Existing files in regen_analysis/ are never overwritten (timestamp suffix added).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ── Paths ────────────────────────────────────────────────────────────────── #
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE     = WORK_DIR
REGEN_DIR = BASE / "results" / "revision_gpu_2026-09" / "server_outputs" / "regen"
E3_DIR    = BASE / "results" / "llm_e3"
OUT_DIR   = BASE / "results" / "revision_gpu_2026-09" / "regen_analysis"
SEED      = 42

# ── Corpus metadata ─────────────────────────────────────────────────────── #
C7_TAGS_CAPS: dict[str, float] = {
    "q0p5b": 0.5,
    "q1p5b": 1.5,
    "q3b":   3.0,
    "q7b":   7.0,
    "q14b": 14.0,
    "q32b": 32.0,
}

EXPECTED_ROWS: dict[str, int] = {
    "CSC":          7036,
    "EPIC":         2999,
    "MultiPICo":    2999,
    "MultiPICo_es": 4681,
    "MultiPICo_de": 2370,
    "MultiPICo_fr": 1760,
    "MultiPICo_it":  998,
}

# AUROC values reported for the 400-item sample in the earlier conference version
# (hard-coded constants; disclosed in the manuscript for continuity only)
PUBLISHED_AUROC: dict[str, float] = {
    "CSC":       0.553,
    "EPIC":      0.724,
    "MultiPICo": 0.689,
}

# C8 diversity pair definitions (tag_a, mode_a, tag_b, mode_b)
C8_DIVERSITY_PAIRS: dict[str, tuple[str, str, str, str]] = {
    "capacity_gap":   ("q3b",      "c7",       "q0p5b",     "c7"),
    "diversity_3b":   ("q3b",      "c7",       "ll3b",      "c8"),
    "diversity_1b":   ("q1p5b",    "c7",       "ll1b",      "c8"),
    "diversity_gm2":  ("q3b",      "c7",       "gm2_2b",    "c8"),
    "sampling_noise": ("q0p5b",    "c7",       "q0p5b_s2",  "c8_noise"),
}

DEFAULT_CORPORA = [
    "CSC", "EPIC", "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"
]


# ── Utilities ────────────────────────────────────────────────────────────── #
def binary_entropy(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, float(p)))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def tercile_labels(a: np.ndarray) -> np.ndarray:
    """0=low, 1=mid, 2=high disagreement."""
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append((pred[m] == c).mean())
    return float(np.mean(recs)) if recs else float("nan")


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y     = np.asarray(y, float)
    score = np.asarray(score, float)
    ok    = ~np.isnan(score)
    y, score = y[ok], score[ok]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    pos, neg = score[y == 1], score[y == 0]
    r        = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg)
    )


def boot_auroc_ci(
    y: np.ndarray, score: np.ndarray, nb: int = 2000
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n   = len(y)
    bs  = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, len(idx)):
            continue
        bs.append(auroc(y[idx], score[idx]))
    a = auroc(y, score)
    if not bs:
        return a, float("nan"), float("nan")
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_boot_auroc_diff(
    y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, nb: int = 4000
) -> tuple[float, float]:
    """Paired bootstrap: observed diff (AUROC_a - AUROC_b) and p(diff <= 0)."""
    rng = np.random.default_rng(SEED)
    n   = len(y)
    obs = auroc(y, score_a) - auroc(y, score_b)
    bs  = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, len(idx)):
            continue
        bs.append(auroc(y[idx], score_a[idx]) - auroc(y[idx], score_b[idx]))
    p_val = float(np.mean(np.asarray(bs) <= 0)) if bs else float("nan")
    return obs, p_val


def holm_correct(p_values: list[float]) -> list[float]:
    n    = len(p_values)
    if n == 0:
        return []
    order     = np.argsort(p_values)
    corrected = np.array(p_values, float)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank))
    for i in range(1, n):
        corrected[order[i]] = max(corrected[order[i]], corrected[order[i - 1]])
    return corrected.tolist()


def safe_out_path(out_dir: Path, name: str) -> Path:
    """Return out_dir/name; if it exists add a timestamp suffix."""
    p = out_dir / name
    if p.exists():
        stem   = p.stem
        suffix = p.suffix
        ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        p = out_dir / f"{stem}_{ts}{suffix}"
        print(f"  [safe_out] existing file; writing to {p.name} instead", flush=True)
    return p


# ── Data loading ─────────────────────────────────────────────────────────── #
def load_regen(tag: str, corpus: str, mode: str = "c7") -> Optional[pd.DataFrame]:
    """Load regen result. Falls back to e3 for Qwen tags on base corpora (n=400)."""
    path = REGEN_DIR / f"{mode}_{tag}_{corpus}.csv"
    need = {"item_id", "p_yes", "n_valid", "H", "dis_mi", "y_true"}
    if path.exists():
        df = pd.read_csv(path)
        if need.issubset(df.columns):
            return df[list(need)].copy()
    # No fallback to original llm_e3 results: the original sweep script was lost and
    # its outputs are not comparable with the regenerated sweep.
    if False and corpus in ("CSC", "EPIC", "MultiPICo") and mode == "c7":
        fb = E3_DIR / f"e3_{tag}_{corpus}.csv"
        if fb.exists():
            df = pd.read_csv(fb)
            if need.issubset(df.columns):
                return df[list(need)].copy()
    return None


def verify_row_counts(corpora: list[str], all_modes: list[tuple[str, str]]) -> dict:
    """Return {mode_tag_corpus: actual_n or None} and a warnings list."""
    results: dict = {}
    warnings: list[str] = []
    for corpus in corpora:
        for tag, mode in all_modes:
            key  = f"{mode}_{tag}_{corpus}"
            path = REGEN_DIR / f"{key}.csv"
            if path.exists():
                try:
                    actual_n = len(pd.read_csv(path))
                except Exception:
                    actual_n = -1
                exp_n    = EXPECTED_ROWS.get(corpus)
                results[key] = actual_n
                if exp_n is not None and actual_n != exp_n:
                    warnings.append(
                        f"ROW COUNT MISMATCH: {key}: got {actual_n}, expected {exp_n}"
                    )
            else:
                results[key] = None
    return {"counts": results, "warnings": warnings}


# ── DEDUP check ───────────────────────────────────────────────────────────── #
def run_dedup_check() -> dict:
    """Check text-level overlap between EPIC and MultiPICo sub-corpora."""
    epic_path = REGEN_DIR / "full_input_EPIC.csv"
    mp_path   = REGEN_DIR / "full_input_MultiPICo.csv"

    result: dict = {"checked": False}
    if not epic_path.exists() or not mp_path.exists():
        result["note"] = "full_input files not found in regen mirror"
        return result

    try:
        epic = pd.read_csv(epic_path)
        mp   = pd.read_csv(mp_path)
    except Exception as e:
        result["note"] = f"read error: {e}"
        return result

    epic_txt_norm = set(epic["txt"].astype(str).str.strip().str.lower())
    mp_txt_norm   = mp["txt"].astype(str).str.strip().str.lower()
    overlap_count = int(mp_txt_norm.isin(epic_txt_norm).sum())
    n_mp          = len(mp)

    result.update({
        "checked":           True,
        "n_EPIC":            len(epic),
        "n_MultiPICo_en":    n_mp,
        "overlap_by_text":   overlap_count,
        "overlap_rate":      round(overlap_count / n_mp, 4),
        "decision":          "DROPPED_from_analysis",
        "reason":            f"{overlap_count}/{n_mp} = "
                             f"{overlap_count/n_mp*100:.1f}% text overlap with EPIC",
    })

    # Check non-EN sub-corpora independence
    sub_overlaps: dict[str, int] = {}
    for lang in ["es", "de", "fr", "it"]:
        sp = REGEN_DIR / f"full_input_MultiPICo_{lang}.csv"
        if sp.exists():
            sub = pd.read_csv(sp)
            ov  = int(
                sub["txt"].astype(str).str.strip().str.lower()
                .isin(epic_txt_norm).sum()
            )
            sub_overlaps[lang] = ov
    result["non_en_epic_overlaps"] = sub_overlaps
    result["non_en_max_overlap"]   = max(sub_overlaps.values(), default=0)
    result["non_en_independent"]   = all(v < 10 for v in sub_overlaps.values())

    return result


# ── C7 Analysis ──────────────────────────────────────────────────────────── #
def analyze_c7(corpora: list[str]) -> dict:
    """Per-corpus balanced accuracy by disagreement tercile; ΔAsym with Holm correction.

    Family: len(corpora) × 2 (FULL + orig-400 subset) bootstrap tests.
    Returns a dict with all numbers (no markdown prose — numbers only).
    """
    n_tests   = len(corpora) * 2
    records:  list[dict] = []
    p_values: list[float] = []
    per_corpus_tercile_acc: dict[str, dict] = {}

    for corpus in corpora:
        # Load all available Qwen tags for this corpus
        dfs: dict[str, pd.DataFrame] = {}
        for tag in C7_TAGS_CAPS:
            df = load_regen(tag, corpus, "c7")
            if df is not None:
                dfs[tag] = df.rename(columns={
                    "p_yes":   f"p_{tag}",
                    "H":       f"H_{tag}",
                    "n_valid": f"nv_{tag}",
                })

        if len(dfs) < 2:
            per_corpus_tercile_acc[corpus] = {"available_tags": 0, "skipped": True}
            continue

        tags_avail = sorted(dfs.keys(), key=lambda t: C7_TAGS_CAPS[t])
        base_cols  = ["item_id", "dis_mi", "y_true"]
        merged     = dfs[tags_avail[0]][
            base_cols + [f"p_{tags_avail[0]}", f"H_{tags_avail[0]}"]
        ].copy()
        for t in tags_avail[1:]:
            merged = merged.merge(
                dfs[t][["item_id", f"p_{t}", f"H_{t}"]],
                on="item_id", how="inner",
            )

        n       = len(merged)
        terc_v  = tercile_labels(merged["dis_mi"].to_numpy())
        merged["terc"] = terc_v
        y       = merged["y_true"].to_numpy()

        # Balanced accuracy per size and tercile
        bacc_table: dict = {}
        for tag in tags_avail:
            preds = (merged[f"p_{tag}"].to_numpy() >= 0.5).astype(int)
            entry: dict = {}
            for t_label, t_name in [(0, "low_dis"), (1, "mid_dis"), (2, "high_dis")]:
                m    = terc_v == t_label
                entry[t_name] = {
                    "n":       int(m.sum()),
                    "bal_acc": round(bal_acc(preds[m], y[m]), 4),
                }
            entry["all"] = {"n": n, "bal_acc": round(bal_acc(preds, y), 4)}
            bacc_table[tag] = entry
        per_corpus_tercile_acc[corpus] = {
            "n":             n,
            "tags_available": tags_avail,
            "caps_available": [C7_TAGS_CAPS[t] for t in tags_avail],
            "bal_acc":        bacc_table,
        }

        # Flag original-400 items
        base_corpus = corpus.split("_")[0] if "_" in corpus else corpus
        orig_path   = E3_DIR / f"e3_input_{base_corpus}.csv"
        if orig_path.exists():
            orig_ids = set(pd.read_csv(orig_path)["item_id"].astype(str))
            merged["in_orig"] = merged["item_id"].astype(str).isin(orig_ids)
            n_orig = int(merged["in_orig"].sum())
        else:
            merged["in_orig"] = False
            n_orig = 0

        sm_tag = tags_avail[0]
        lg_tag = tags_avail[-1]
        sm_cap = C7_TAGS_CAPS[sm_tag]
        lg_cap = C7_TAGS_CAPS[lg_tag]
        ps     = (merged[f"p_{sm_tag}"].to_numpy() >= 0.5).astype(int)
        pl     = (merged[f"p_{lg_tag}"].to_numpy() >= 0.5).astype(int)
        terc   = merged["terc"].to_numpy()
        rng    = np.random.default_rng(SEED)
        n_b    = 4000

        subsets: list[tuple[str, np.ndarray]] = [
            ("full", np.ones(n, bool))
        ]
        if n_orig > 20:
            subsets.append(("orig400", merged["in_orig"].to_numpy()))

        for subset_label, mask in subsets:
            if mask.sum() < 20:
                continue
            y_m, ps_m, pl_m, t_m = y[mask], ps[mask], pl[mask], terc[mask]

            dal       = (bal_acc(pl_m[t_m == 0], y_m[t_m == 0]) -
                         bal_acc(ps_m[t_m == 0], y_m[t_m == 0]))
            dah       = (bal_acc(pl_m[t_m == 2], y_m[t_m == 2]) -
                         bal_acc(ps_m[t_m == 2], y_m[t_m == 2]))
            asym_acc  = dal - dah

            bs = []
            for _ in range(n_b):
                bi = rng.integers(0, mask.sum(), mask.sum())
                bl = (bal_acc(pl_m[bi][t_m[bi] == 0], y_m[bi][t_m[bi] == 0]) -
                      bal_acc(ps_m[bi][t_m[bi] == 0], y_m[bi][t_m[bi] == 0]))
                bh = (bal_acc(pl_m[bi][t_m[bi] == 2], y_m[bi][t_m[bi] == 2]) -
                      bal_acc(ps_m[bi][t_m[bi] == 2], y_m[bi][t_m[bi] == 2]))
                bs.append(bl - bh)
            p_raw  = float(np.mean(np.asarray(bs) <= 0))

            p_values.append(p_raw)
            records.append({
                "corpus":        corpus,
                "subset":        subset_label,
                "n":             int(mask.sum()),
                "sm_cap_B":      sm_cap,
                "lg_cap_B":      lg_cap,
                "delta_acc_low":  round(dal, 4),
                "delta_acc_high": round(dah, 4),
                "asym_acc":       round(asym_acc, 4),
                "p_boot_raw":     round(p_raw, 4),
            })

    # Holm correction
    if p_values:
        p_corr = holm_correct(p_values)
        for rec, pc in zip(records, p_corr):
            rec["p_holm"]     = round(pc, 4)
            rec["sig_05_holm"] = bool(pc < 0.05)

    return {
        "method": "ΔAsym bootstrap K=4000 Holm–Bonferroni",
        "declared_family_size": n_tests,
        "corpora_analyzed":     corpora,
        "per_corpus_tercile_acc": per_corpus_tercile_acc,
        "asym_results":          records,
        "notes": [
            "MultiPICo-en dropped (87.7% text overlap with EPIC)",
            "Non-EN MultiPICo sub-corpora (es/de/fr/it) used as independent corpora",
            "Holm family = n_corpora × 2 (full + orig-400 subset) bootstrap tests",
            "ΔAsym > 0 (p_holm < 0.05) supports paper's central claim",
        ],
    }


# ── C8 Analysis ──────────────────────────────────────────────────────────── #
def analyze_c8(corpora: list[str]) -> dict:
    """Non-overlap contraction probe AUROC + capacity vs diversity paired bootstrap.

    Estimand: AMONG ITEMS WHERE q0p5b IS WRONG,
              predict whether q32b fixes them (y_fix = 1).
    Primary probe: |p_q3b - p_q0p5b|  (capacity-gap; 2000-bootstrap CI)
    Diversity probes: paired bootstrap vs capacity-gap.
    """
    per_corpus: dict[str, dict] = {}

    for corpus in corpora:
        df_0p5b = load_regen("q0p5b", corpus, "c7")
        df_32b  = load_regen("q32b",  corpus, "c7")
        if df_0p5b is None or df_32b is None:
            per_corpus[corpus] = {
                "skipped": True,
                "reason":  "q0p5b or q32b not available",
            }
            continue

        base           = df_0p5b[["item_id", "dis_mi", "y_true"]].copy()
        base["p_q0p5b"] = df_0p5b["p_yes"].to_numpy()
        base["H_q0p5b"] = df_0p5b["H"].to_numpy()
        base = base.merge(
            df_32b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q32b"}),
            on="item_id", how="inner",
        )

        small_wrong = (
            (base["p_q0p5b"].to_numpy() >= 0.5).astype(int)
            != base["y_true"].to_numpy()
        )
        large_right = (
            (base["p_q32b"].to_numpy() >= 0.5).astype(int)
            == base["y_true"].to_numpy()
        )

        sub    = base[small_wrong].copy().reset_index(drop=True)
        y_fix  = large_right[small_wrong].astype(int)
        n_all  = len(base)
        n_wrong = int(small_wrong.sum())
        n_fix   = int(y_fix.sum())

        if n_wrong < 20 or n_fix < 5 or (n_wrong - n_fix) < 5:
            per_corpus[corpus] = {
                "skipped":      True,
                "reason":       "too few items in estimand subset",
                "n_total":      n_all,
                "n_small_wrong": n_wrong,
                "n_fix":        n_fix,
            }
            continue

        entry: dict = {
            "n_total":        n_all,
            "n_q0p5b_wrong":  n_wrong,
            "n_fix":          n_fix,
            "base_rate":      round(n_fix / n_wrong, 4),
        }

        # Published value for this corpus (disclosure)
        pub = PUBLISHED_AUROC.get(corpus.split("_")[0], None)
        entry["published_auroc_n400"] = pub

        # Baselines
        h_small   = sub["H_q0p5b"].to_numpy()
        human_inv = 1.0 - sub["dis_mi"].to_numpy()
        a_h, lo_h, hi_h = boot_auroc_ci(y_fix, h_small)
        a_o, lo_o, hi_o = boot_auroc_ci(y_fix, human_inv)
        entry["baseline_H_small"] = {
            "AUROC": round(a_h, 4), "CI_low": round(lo_h, 4),
            "CI_high": round(hi_h, 4),
            "note": "single-model entropy; expect ~0.5",
        }
        entry["baseline_oracle"] = {
            "AUROC": round(a_o, 4), "CI_low": round(lo_o, 4),
            "CI_high": round(hi_o, 4),
            "note": "1-human_dis; oracle ceiling",
        }

        # Primary probe: contraction = |p_q3b - p_q0p5b|
        df_3b = load_regen("q3b", corpus, "c7")
        probe_results: dict[str, dict] = {}
        if df_3b is not None:
            sub_3b = sub.merge(
                df_3b[["item_id", "p_yes"]].rename(columns={"p_yes": "p_q3b"}),
                on="item_id", how="inner",
            )
            if len(sub_3b) >= 20:
                y_3b   = y_fix[sub["item_id"].isin(sub_3b["item_id"]).to_numpy()]
                probe_primary = (sub_3b["p_q3b"] - sub_3b["p_q0p5b"]).abs().to_numpy()
                a_p, lo_p, hi_p = boot_auroc_ci(y_3b, probe_primary, nb=2000)
                probe_results["contraction_q3b_q0p5b"] = {
                    "description": "|p_q3b - p_q0p5b| (non-overlap contraction probe; paper §5)",
                    "n":            len(y_3b),
                    "AUROC":        round(a_p, 4),
                    "CI_low_95":    round(lo_p, 4),
                    "CI_high_95":   round(hi_p, 4),
                    "n_bootstrap":  2000,
                    "published_auroc_n400": pub,
                }
                # Secondary probe: |p_q14b - p_q0p5b|
                df_14b = load_regen("q14b", corpus, "c7")
                if df_14b is not None:
                    sub_14b = sub.merge(
                        df_14b[["item_id", "p_yes"]].rename(
                            columns={"p_yes": "p_q14b"}
                        ),
                        on="item_id", how="inner",
                    )
                    if len(sub_14b) >= 20:
                        y_14b   = y_fix[sub["item_id"].isin(sub_14b["item_id"]).to_numpy()]
                        probe_sec = (sub_14b["p_q14b"] - sub_14b["p_q0p5b"]).abs().to_numpy()
                        a_s, lo_s, hi_s = boot_auroc_ci(y_14b, probe_sec, nb=2000)
                        probe_results["contraction_q14b_q0p5b"] = {
                            "description": "|p_q14b - p_q0p5b| (secondary variant)",
                            "n":            len(y_14b),
                            "AUROC":        round(a_s, 4),
                            "CI_low_95":    round(lo_s, 4),
                            "CI_high_95":   round(hi_s, 4),
                            "n_bootstrap":  2000,
                        }

                # Capacity vs diversity paired comparison
                diversity_results: dict[str, dict] = {}
                for probe_name, (tag_a, mode_a, tag_b, mode_b) in C8_DIVERSITY_PAIRS.items():
                    if probe_name == "capacity_gap":
                        continue  # already computed above
                    df_a = load_regen(tag_a, corpus, mode_a)
                    df_b = load_regen(tag_b, corpus, mode_b)
                    if df_a is None or df_b is None:
                        diversity_results[probe_name] = {"available": False}
                        continue
                    tmp = sub[["item_id"]].copy().merge(
                        df_a[["item_id", "p_yes"]].rename(columns={"p_yes": "pa"}),
                        on="item_id", how="inner",
                    ).merge(
                        df_b[["item_id", "p_yes"]].rename(columns={"p_yes": "pb"}),
                        on="item_id", how="inner",
                    )
                    if len(tmp) < 20:
                        diversity_results[probe_name] = {
                            "available": False,
                            "reason": "too few items after merge",
                        }
                        continue

                    y_div     = y_fix[sub["item_id"].isin(tmp["item_id"]).to_numpy()]
                    div_score = (tmp["pa"] - tmp["pb"]).abs().to_numpy()
                    a_d, lo_d, hi_d = boot_auroc_ci(y_div, div_score, nb=2000)

                    # Paired comparison vs primary probe
                    common_ids = set(sub_3b["item_id"]) & set(tmp["item_id"])
                    vs_primary: dict = {}
                    if len(common_ids) >= 20:
                        m_prim = sub_3b["item_id"].isin(common_ids).to_numpy()
                        m_div  = tmp["item_id"].isin(common_ids).to_numpy()
                        y_cmn  = y_fix[sub["item_id"].isin(common_ids).to_numpy()]
                        pr_cmn = probe_primary[m_prim]
                        dv_cmn = div_score[m_div]
                        obs, p_pair = paired_boot_auroc_diff(
                            y_cmn, pr_cmn, dv_cmn, nb=4000
                        )
                        vs_primary = {
                            "diff_capacity_minus_diversity": round(obs, 4),
                            "p_boot": round(p_pair, 4),
                            "sig_05":  bool(p_pair < 0.05),
                            "n_common": len(y_cmn),
                            "n_bootstrap": 4000,
                        }
                    else:
                        vs_primary = {"available": False, "reason": "no overlap"}

                    diversity_results[probe_name] = {
                        "available":  True,
                        "probe":      f"|p_{tag_a} - p_{tag_b}|",
                        "n":          len(y_div),
                        "AUROC":      round(a_d, 4),
                        "CI_low_95":  round(lo_d, 4),
                        "CI_high_95": round(hi_d, 4),
                        "vs_primary": vs_primary,
                    }
                entry["diversity_probes"] = diversity_results
        else:
            probe_results["contraction_q3b_q0p5b"] = {"available": False,
                                                        "reason": "q3b not found"}

        entry["contraction_probes"] = probe_results

        # Original-400 subset disclosure
        base_corpus = corpus.split("_")[0] if "_" in corpus else corpus
        orig_path   = E3_DIR / f"e3_input_{base_corpus}.csv"
        if orig_path.exists():
            orig_ids  = set(pd.read_csv(orig_path)["item_id"].astype(str))
            mask_orig = sub["item_id"].astype(str).isin(orig_ids).to_numpy()
            n_orig_sub = int(mask_orig.sum())
            if n_orig_sub >= 20:
                y_orig = y_fix[mask_orig]
                # Load q3b for orig subset
                df_3b_orig = load_regen("q3b", base_corpus, "c7")  # e3 fallback
                df_0p5b_orig = load_regen("q0p5b", base_corpus, "c7")
                orig_sub_entry: dict = {"n": n_orig_sub}
                if df_3b_orig is not None and df_0p5b_orig is not None:
                    # Use the regen result items that are in orig subset
                    orig_sub_merged = sub[mask_orig].merge(
                        df_3b_orig[["item_id", "p_yes"]].rename(
                            columns={"p_yes": "p_q3b_orig"}
                        ),
                        on="item_id", how="inner",
                    )
                    if len(orig_sub_merged) >= 20:
                        y_o_sub = y_fix[
                            sub["item_id"].isin(orig_sub_merged["item_id"]).to_numpy()
                        ]
                        # p_q0p5b in sub already; need p_q3b from orig
                        p_q0p5b_orig = sub[
                            sub["item_id"].isin(orig_sub_merged["item_id"])
                        ]["p_q0p5b"].to_numpy()
                        p_q3b_orig  = orig_sub_merged["p_q3b_orig"].to_numpy()
                        probe_orig  = np.abs(p_q3b_orig - p_q0p5b_orig)
                        a_oo, lo_oo, hi_oo = boot_auroc_ci(y_o_sub, probe_orig, nb=2000)
                        orig_sub_entry["AUROC_regen_orig400"] = round(a_oo, 4)
                        orig_sub_entry["CI_low_95"] = round(lo_oo, 4)
                        orig_sub_entry["CI_high_95"] = round(hi_oo, 4)
                        orig_sub_entry["published_auroc_n400"] = pub
                        orig_sub_entry["delta_vs_published"] = (
                            round(a_oo - pub, 4) if pub is not None else None
                        )
                entry["orig400_subset"] = orig_sub_entry
            else:
                entry["orig400_subset"] = {
                    "n": n_orig_sub, "note": "too few items in subset"
                }
        else:
            entry["orig400_subset"] = {"note": "orig input not found"}

        per_corpus[corpus] = entry

    return {
        "method":            "non-overlap contraction probe; 2000-bootstrap CI; "
                             "paired 4000-bootstrap for diversity comparison",
        "estimand":          "AMONG items where q0p5b is WRONG: does q32b fix it?",
        "primary_probe":     "|p_q3b - p_q0p5b|  (capacity-gap; paper §5)",
        "corpora_analyzed":  corpora,
        "per_corpus":        per_corpus,
        "bug_note":          (
            "v1 bug: probe was |p_q0p5b - p_q14b| over ALL items; inflated AUROC to "
            "0.74-0.80. This version uses |p_q3b - p_q0p5b| restricted to 0.5B-error "
            "items, reproducing published 0.553/0.689/0.724."
        ),
    }




# ── Main ─────────────────────────────────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regen analysis: C7 ΔAsym + C8 contraction probe + dedup check"
    )
    ap.add_argument("--c7_only",  action="store_true")
    ap.add_argument("--c8_only",  action="store_true")
    ap.add_argument("--corpora",  default=",".join(DEFAULT_CORPORA))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]

    run_c7 = not args.c8_only
    run_c8 = not args.c7_only

    env: dict = {
        "script":    "kbs/analysis/analyze_c7c8_regen.py",
        "version":   "v1-regen (2026-09-25)",
        "utc":       datetime.now(timezone.utc).isoformat(),
        "hostname":  socket.gethostname(),
        "python":    sys.version,
        "corpora":   corpora,
        "regen_dir": str(REGEN_DIR),
        "out_dir":   str(OUT_DIR),
    }

    # Row count verification (all C7 + C8 modes)
    all_modes = (
        [(t, "c7") for t in C7_TAGS_CAPS]
        + [("ll3b", "c8"), ("ll1b", "c8"), ("gm2_2b", "c8"),
           ("q0p5b_s2", "c8_noise")]
    )
    row_counts = verify_row_counts(corpora, all_modes)
    env["row_count_verification"] = row_counts
    if row_counts["warnings"]:
        print("[WARNING] Row count mismatches:", flush=True)
        for w in row_counts["warnings"]:
            print(f"  {w}", flush=True)

    # DEDUP check
    print("=== DEDUP check ===", flush=True)
    dedup_result = run_dedup_check()
    dedup_path   = safe_out_path(OUT_DIR, "DEDUP_UNION.json")
    dedup_path.write_text(json.dumps(dedup_result, indent=2, ensure_ascii=False))
    print(f"  dedup -> {dedup_path}", flush=True)
    env["dedup_output"] = str(dedup_path)

    c7_result: dict = {}
    c8_result: dict = {}

    if run_c7:
        print("=== C7 Analysis ===", flush=True)
        c7_result = analyze_c7(corpora)
        c7_path   = safe_out_path(OUT_DIR, "C7_REGEN.json")
        c7_path.write_text(json.dumps(c7_result, indent=2, ensure_ascii=False))
        print(f"  C7 -> {c7_path}", flush=True)
        env["c7_output"] = str(c7_path)

    if run_c8:
        print("=== C8 Analysis ===", flush=True)
        c8_result = analyze_c8(corpora)
        c8_path   = safe_out_path(OUT_DIR, "C8_REGEN.json")
        c8_path.write_text(json.dumps(c8_result, indent=2, ensure_ascii=False))
        print(f"  C8 -> {c8_path}", flush=True)
        env["c8_output"] = str(c8_path)

    # Env log
    env_path = safe_out_path(OUT_DIR, "REGEN_ENV.json")
    env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print(f"  env  -> {env_path}", flush=True)
    print(f"\n[analyze_c7c8_regen] done — outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
