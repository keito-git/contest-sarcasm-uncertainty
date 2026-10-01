#!/usr/bin/env python3
"""
compute_ki_languages.py
Compute k_i (annotator count) distributions per non-English MultiPICo language
from the raw parquet file. If parquet is unavailable, estimate from dis_mi.

Saves: results/revision_2026-09/ki_per_language.json
"""
import json
import pathlib
import math
import csv
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR

PARQUET = BASE / "results/revision_gpu_2026-09/multipico_hf_raw.parquet"
OUT = BASE / "results/revision_2026-09/ki_per_language.json"

LANG_CSV = {
    "es": BASE / "results/revision_gpu_2026-09/full_input_MultiPICo_es.csv",
    "de": BASE / "results/revision_gpu_2026-09/full_input_MultiPICo_de.csv",
    "fr": BASE / "results/revision_gpu_2026-09/full_input_MultiPICo_fr.csv",
    "it": BASE / "results/revision_gpu_2026-09/full_input_MultiPICo_it.csv",
}


def binary_entropy_norm(p, k):
    """Normalized binary entropy = H(p) / log(k)."""
    if p <= 0 or p >= 1:
        return 0.0
    h = -p * math.log(p) - (1 - p) * math.log(1 - p)
    return h / math.log(k)


def estimate_k_from_dis(dis_mi, y_true):
    """
    For binary MultiPICo: dis_mi = H(j/k)/log(k).
    Given dis_mi and y_true (>= 0.5 majority), find k in {2..8} and j that minimize
    |dis_mi - H(j/k)/log(k)|. Returns best k.
    """
    if dis_mi <= 0:
        return 5  # default: consensus item
    best_k, best_err = 5, float("inf")
    for k in range(2, 9):
        # For y_true==1: majority ironic => j in {ceil(k/2)..k}
        # For y_true==0: majority non-ironic => j in {0..floor(k/2)}
        j_range = range(0, k + 1)
        for j in j_range:
            p = j / k
            pred = binary_entropy_norm(p, k)
            err = abs(pred - dis_mi)
            if err < best_err:
                best_err = err
                best_k = k
    return best_k


def stats_from_list(ks):
    """Compute min/median/mean/max/std from a list."""
    if not ks:
        return {}
    ks_sorted = sorted(ks)
    n = len(ks_sorted)
    mean = sum(ks_sorted) / n
    mid = n // 2
    median = ks_sorted[mid] if n % 2 == 1 else (ks_sorted[mid - 1] + ks_sorted[mid]) / 2.0
    variance = sum((x - mean) ** 2 for x in ks_sorted) / n
    std = math.sqrt(variance)
    from collections import Counter
    vc = Counter(ks_sorted)
    frac_ge4 = sum(v for k2, v in vc.items() if k2 >= 4) / n
    frac_ge5 = sum(v for k2, v in vc.items() if k2 >= 5) / n
    return {
        "min": int(min(ks_sorted)),
        "median": float(median),
        "mean": round(mean, 4),
        "max": int(max(ks_sorted)),
        "std": round(std, 4),
        "n_items": n,
        "frac_k_ge4": round(frac_ge4, 4),
        "frac_k_ge5": round(frac_ge5, 4),
        "value_counts": {str(k2): int(v) for k2, v in sorted(vc.items())},
    }


def delta_method_se(ki_list, dis_list):
    """
    Delta method SE for dis = H(p)/log(k).
    Approximation: SE(dis) ≈ std(dis) / sqrt(n).
    (Full delta method requires per-item SE propagation.)
    """
    if not dis_list:
        return {}
    n = len(dis_list)
    mean_d = sum(dis_list) / n
    var_d = sum((d - mean_d) ** 2 for d in dis_list) / n
    std_d = math.sqrt(var_d)
    se = std_d / math.sqrt(n)
    return {
        "mean_dis": round(mean_d, 6),
        "std_dis": round(std_d, 6),
        "se_dis": round(se, 6),
        "n": n,
    }


def main():
    result = {}
    source_used = "unknown"

    # ----------------------------------------------------------------
    # Attempt 1: read parquet for n_annotators per language
    # ----------------------------------------------------------------
    try:
        import pandas as pd
        print("Attempting parquet read...", flush=True)
        df = pd.read_parquet(str(PARQUET))
        print(f"Parquet loaded: {df.shape}, cols={list(df.columns)[:12]}", flush=True)
        source_used = "parquet"

        # MultiPICo typically has 'language' and 'n_annotators' columns
        lang_col = None
        for c in ["language", "lang", "subset"]:
            if c in df.columns:
                lang_col = c
                break
        ann_col = None
        for c in ["n_annotators", "num_annotators", "k"]:
            if c in df.columns:
                ann_col = c
                break

        print(f"  lang_col={lang_col}, ann_col={ann_col}", flush=True)

        if lang_col and ann_col:
            for lang in ["es", "de", "fr", "it"]:
                sub = df[df[lang_col] == lang]
                ks = sub[ann_col].tolist()
                dis_col = None
                for c in ["dis_mi", "disagreement", "dis"]:
                    if c in sub.columns:
                        dis_col = c
                        break
                dis_list = sub[dis_col].tolist() if dis_col else []
                result[f"MultiPICo_{lang}"] = {
                    **stats_from_list(ks),
                    "delta_method_dis_se": delta_method_se(ks, dis_list),
                    "source": "parquet",
                }
        elif ann_col:
            print("  No language column; trying language-specific subsets", flush=True)
            for lang in ["es", "de", "fr", "it"]:
                if lang_col is None:
                    # Try id-based split
                    sub = df[df.index.astype(str).str.startswith(f"mp_{lang}_")] if True else df
                ks = sub[ann_col].tolist() if not sub.empty else []
                result[f"MultiPICo_{lang}"] = {
                    **stats_from_list(ks),
                    "source": "parquet_no_lang_col",
                }

    except ImportError:
        print("pandas not available", flush=True)
    except Exception as e:
        print(f"Parquet read failed: {e}", flush=True)

    # ----------------------------------------------------------------
    # Attempt 2: estimate from CSV dis_mi if parquet failed
    # ----------------------------------------------------------------
    if not result:
        print("Falling back to CSV dis_mi estimation...", flush=True)
        source_used = "estimated_from_dis_mi"
        for lang, csv_path in LANG_CSV.items():
            if not csv_path.exists():
                print(f"  Missing CSV: {csv_path}", flush=True)
                continue
            ks, dis_list = [], []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dis = float(row["dis_mi"])
                        yt = int(float(row["y_true"]))
                        k = estimate_k_from_dis(dis, yt)
                        ks.append(k)
                        dis_list.append(dis)
                    except (ValueError, KeyError):
                        continue
            result[f"MultiPICo_{lang}"] = {
                **stats_from_list(ks),
                "delta_method_dis_se": delta_method_se(ks, dis_list),
                "source": "estimated_from_dis_mi",
                "note": "k_i estimated via MLE from normalized binary entropy formula",
            }

    # ----------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------
    output = {
        "source_used": source_used,
        "ki_per_language": result,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten: {OUT}", flush=True)
    for lang_key, stats in result.items():
        print(f"  {lang_key}: min={stats.get('min')}, med={stats.get('median')}, "
              f"mean={stats.get('mean')}, max={stats.get('max')}, n={stats.get('n_items')}", flush=True)


if __name__ == "__main__":
    main()
