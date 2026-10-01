"""
prepare_full_input.py — Build full-corpus e3_input CSVs for C7/C8 GPU runs.

Purpose
-------
The original experiments used 400 randomly sampled items per corpus (SEED=42).
This script extends the evaluation to all available annotator-level items.
This script builds:
  - results/revision_gpu_2026-09/full_input_CSC.csv            (n ≈ 7036)
  - results/revision_gpu_2026-09/full_input_EPIC.csv            (n = 2999)
  - results/revision_gpu_2026-09/full_input_MultiPICo_es.csv    (n ≈ 4683)
  - results/revision_gpu_2026-09/full_input_MultiPICo_de.csv    (n ≈ 2375)
  - results/revision_gpu_2026-09/full_input_MultiPICo_fr.csv    (n ≈ 1760)
  - results/revision_gpu_2026-09/full_input_MultiPICo_it.csv    (n ≈ 1000)

MultiPICo-en is NOT included as a corpus: it shares 80 % of items with EPIC
(same 5 English varieties from Frenda et al. 2023b; see rc10_corpus_overlap.json).
The non-English sub-corpora are independent and add cross-lingual replication.

Each CSV is in the exact schema used by the existing sweep:
  item_id, dataset, concept, ctx, txt, dis_mi, y_true

  For CSC/EPIC: in_original_400 flag marks items in the original 400-item set.
  For MultiPICo_xx: in_original_400 = 0 (new corpora; no original baseline).

Preprocessing formulas (from run_llm_e1.py, unchanged):
  CSC:
    dis_mi  = var(eval_scores) / ((mean_eval - 1) * (6 - mean_eval)),  clipped [0,1]
    y_true  = int(mean_eval_norm >= 0.5)   [mean_eval_norm = (mean_eval - 1) / 5]
  EPIC / MultiPICo_xx:
    dis_mi  = normalized_binary_entropy(p_ironic)   [0,1]
    y_true  = int(p_ironic >= 0.5)

Prompt language choice: English instruction template is used for all corpora,
including non-English MultiPICo sub-corpora (cross-lingual zero-shot). This is
documented in the paper as the "English-instruction, native-text" setting.

Data hash is written to PREP_ENV.json for reproducibility.

Usage
-----
  python kbs/data_prep/prepare_full_input.py [--dry_run] [--skip_multipico_dl]

  --dry_run:           print stats only; do not write files.
  --skip_multipico_dl: use cached MultiPICo HF data if already saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────── #
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR
SARCASM = BASE / "results" / "sarcasm"
E3_IN   = BASE / "results" / "llm_e3"
OUT_DIR = BASE / "results" / "revision_gpu_2026-09"
SEED    = 42  # same seed as original experiments

# Non-English MultiPICo language codes to include.
# Chosen for item count and annotator coverage (≥3 ann/item for nearly all):
#   es (Spanish):  ~4683 items,  5.1 ann/item avg
#   de (German):   ~2375 items,  5.3 ann/item avg
#   fr (French):   ~1760 items,  5.0 ann/item avg
#   it (Italian):  ~1000 items,  4.8 ann/item avg
MULTIPICO_LANGUAGES: list[str] = ["es", "de", "fr", "it"]
MULTIPICO_HF_ID = "Multilingual-Perspectivist-NLU/MultiPICo"
MULTIPICO_CACHE = OUT_DIR / "multipico_hf_raw.parquet"   # cached HF download


# ── Helpers ──────────────────────────────────────────────────────────────── #
def _sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def _check_original_400(full_df: pd.DataFrame, corpus_name: str) -> pd.Series:
    """Return boolean Series: True if item was in the original 400-item input file."""
    orig_path = E3_IN / f"e3_input_{corpus_name}.csv"
    if not orig_path.exists():
        return pd.Series(False, index=full_df.index)
    orig_ids = set(pd.read_csv(orig_path)["item_id"].astype(str))
    return full_df["item_id"].astype(str).isin(orig_ids)


# ── CSC ──────────────────────────────────────────────────────────────────── #
def build_csc() -> pd.DataFrame:
    """Build full CSC input CSV.

    Source: results/sarcasm/csc_aggregated.parquet
    dis_mi  = std_eval^2 / ((mean_eval - 1) * (6 - mean_eval))  capped [0,1]
    y_true  = int(mean_eval_norm >= 0.5)                         [>= 3.5 on 1-6 scale]
    ctx     = context_text
    txt     = response_text
    concept = "sarcastic"
    """
    df = pd.read_parquet(SARCASM / "csc_aggregated.parquet").copy()

    # Require non-empty text
    df = df.dropna(subset=["response_text", "context_text"])
    df = df[df["response_text"].str.strip().str.len() > 0]

    # dis_mi: relative variance (mean-normalized variance for an ordered Likert scale)
    me = df["mean_eval"].to_numpy(float)
    var = df["std_eval"].to_numpy(float) ** 2
    denom = np.clip((me - 1.0) * (6.0 - me), 0.5, None)  # avoid div-by-zero
    df["dis_mi"] = np.clip(var / denom, 0.0, 1.0)

    # y_true: majority-vote binarization on normalized 1-6 scale
    df["y_true"] = (df["mean_eval_norm"].to_numpy(float) >= 0.5).astype(int)

    # Rename to standard schema
    df = df.rename(columns={
        "context_text":  "ctx",
        "response_text": "txt",
    })
    df["dataset"] = "CSC"
    df["concept"] = "sarcastic"

    out = df[["item_id", "dataset", "concept", "ctx", "txt", "dis_mi", "y_true"]].copy()
    out["in_original_400"] = _check_original_400(out, "CSC").astype(int)
    return out.reset_index(drop=True)


# ── EPIC ─────────────────────────────────────────────────────────────────── #
def _build_irony_corpus(corpus_name: str, concept: str) -> pd.DataFrame:
    """Builder for EPIC (parquet already aggregated locally).

    Source: results/sarcasm/{corpus_name.lower()}_aggregated.parquet
    dis_mi  = disagreement   [normalized binary entropy, already [0,1]]
    y_true  = int(p_ironic >= 0.5)
    ctx     = post
    txt     = reply

    Note: MultiPICo-en uses the same underlying data as EPIC (80 % overlap,
    same 5 English varieties from Frenda et al. 2023b).  It is NOT built here.
    Non-English MultiPICo sub-corpora are built via build_multipico_lang().
    """
    fname = f"{corpus_name.lower()}_aggregated.parquet"
    df = pd.read_parquet(SARCASM / fname).copy()

    df = df.dropna(subset=["reply", "post"])
    df = df[df["reply"].str.strip().str.len() > 0]

    df["dis_mi"] = np.clip(df["disagreement"].to_numpy(float), 0.0, 1.0)
    df["y_true"] = (df["p_ironic"].to_numpy(float) >= 0.5).astype(int)

    df = df.rename(columns={"post": "ctx", "reply": "txt"})
    df["dataset"] = corpus_name
    df["concept"] = concept

    out = df[["item_id", "dataset", "concept", "ctx", "txt", "dis_mi", "y_true"]].copy()
    out["in_original_400"] = _check_original_400(out, corpus_name).astype(int)
    return out.reset_index(drop=True)


# ── MultiPICo non-English ─────────────────────────────────────────────────── #
def _load_multipico_hf(skip_dl: bool = False) -> pd.DataFrame:
    """Download or load cached MultiPICo full HF dataset (annotator level).

    Returns a dataframe with columns: reply_id, language, label, reply, post.
    Caches to MULTIPICO_CACHE to avoid repeated downloads.
    """
    if skip_dl and MULTIPICO_CACHE.exists():
        print("  [MultiPICo] using cached HF data", flush=True)
        return pd.read_parquet(MULTIPICO_CACHE)

    print(f"  [MultiPICo] downloading from HuggingFace: {MULTIPICO_HF_ID}", flush=True)
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` package required. "
            "Install with: pip install datasets"
        )
    ds = load_dataset(MULTIPICO_HF_ID, trust_remote_code=True)
    df = ds["train"].to_pandas()
    print(f"  [MultiPICo HF] {len(df)} annotator-level rows; "
          f"languages: {df['language'].value_counts().to_dict()}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MULTIPICO_CACHE, index=False)
    print(f"  [MultiPICo HF] cached -> {MULTIPICO_CACHE}", flush=True)
    return df


def build_multipico_lang(lang: str, hf_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate annotator-level MultiPICo rows into item-level for one language.

    item_id  = f"mp_{lang}_{reply_id}"
    dis_mi   = normalized binary entropy of p_ironic  (same formula as EPIC)
    y_true   = int(p_ironic >= 0.5)
    ctx      = post
    txt      = reply
    concept  = "ironic"   (English instruction — cross-lingual zero-shot)
    in_original_400 = 0   (new corpus; no prior baseline)

    Minimum annotators per item: 3 (items with <3 are dropped, typically <0.3 %).
    """
    sub = hf_df[hf_df["language"] == lang].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for language='{lang}'")

    # Aggregate per reply_id
    grp = sub.groupby("reply_id").agg(
        n_ann=("label", "count"),
        p_ironic=("label", "mean"),
        post=("post", "first"),
        reply=("reply", "first"),
    ).reset_index()

    # Drop items with fewer than 3 annotators
    grp = grp[grp["n_ann"] >= 3].copy()

    # dis_mi: normalized binary entropy (same formula as EPIC/multipico_aggregated)
    p = np.clip(grp["p_ironic"].to_numpy(float), 1e-9, 1 - 1e-9)
    ln2 = float(np.log(2))
    grp["dis_mi"] = np.clip(
        (-p * np.log(p) - (1 - p) * np.log(1 - p)) / ln2,
        0.0, 1.0,
    )
    grp["y_true"] = (grp["p_ironic"].to_numpy(float) >= 0.5).astype(int)

    # Drop items with empty text
    grp = grp.dropna(subset=["reply", "post"])
    grp = grp[grp["reply"].str.strip().str.len() > 0]

    corpus_name = f"MultiPICo_{lang}"
    grp["item_id"]  = "mp_" + lang + "_" + grp["reply_id"].astype(str)
    grp["dataset"]  = corpus_name
    grp["concept"]  = "ironic"  # English instruction; cross-lingual zero-shot
    grp = grp.rename(columns={"post": "ctx", "reply": "txt"})

    out = grp[["item_id", "dataset", "concept", "ctx", "txt",
               "dis_mi", "y_true"]].copy()
    out["in_original_400"] = 0  # no prior baseline for these new corpora
    return out.reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="print stats only; do not write files")
    ap.add_argument("--skip_multipico_dl", action="store_true",
                    help="use cached HF MultiPICo parquet if available")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env: dict = {
        "script":   "kbs/data_prep/prepare_full_input.py",
        "version":  "v2 (2026-09-25): dropped MultiPICo-en; added non-EN sub-corpora",
        "utc":      datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python":   platform.python_version(),
        "purpose":  "C7 — build full-corpus input CSVs",
        "note_multipico_en": (
            "MultiPICo-en dropped: 80 % item overlap with EPIC "
            "(same 5 English varieties; see rc10_corpus_overlap.json)."
        ),
        "prompt_language_choice": (
            "English instruction template used for all corpora including "
            "non-English MultiPICo sub-corpora (cross-lingual zero-shot). "
            "Documented in paper as 'English-instruction, native-text' setting."
        ),
        "corpora":  {},
    }

    # ── Fixed English corpora ──────────────────────────────────────────────── #
    base_corpora: list[tuple[str, "Callable[[], pd.DataFrame]"]] = [
        ("CSC",  build_csc),
        ("EPIC", lambda: _build_irony_corpus("EPIC", "ironic")),
    ]

    for corpus_name, builder in base_corpora:
        print(f"\n[{corpus_name}] building ...", flush=True)
        df = builder()
        _record_and_save(corpus_name, df, env, args.dry_run)

    # ── Non-English MultiPICo sub-corpora ─────────────────────────────────── #
    print("\n[MultiPICo non-EN] loading HuggingFace data ...", flush=True)
    try:
        hf_df = _load_multipico_hf(skip_dl=args.skip_multipico_dl)
    except Exception as e:
        print(f"  [ERROR] Could not load MultiPICo HF data: {e}", flush=True)
        print("  Skipping non-EN MultiPICo. Re-run without --skip_multipico_dl "
              "or with network access.", flush=True)
        hf_df = None

    if hf_df is not None:
        for lang in MULTIPICO_LANGUAGES:
            corpus_name = f"MultiPICo_{lang}"
            print(f"\n[{corpus_name}] building ...", flush=True)
            try:
                df = build_multipico_lang(lang, hf_df)
                _record_and_save(corpus_name, df, env, args.dry_run)
            except Exception as e:
                print(f"  [ERROR] {corpus_name}: {e}", flush=True)
                env["corpora"][corpus_name] = {"error": str(e)}

    if not args.dry_run:
        env_path = OUT_DIR / "PREP_ENV.json"
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2)
        print(f"\nEnv log -> {env_path}", flush=True)

    print("\nDone.", flush=True)


def _record_and_save(
    corpus_name: str, df: pd.DataFrame, env: dict, dry_run: bool
) -> None:
    """Print stats, record in env dict, and optionally write CSV."""
    n_total = len(df)
    n_orig  = int(df["in_original_400"].sum())
    n_pos   = int(df["y_true"].sum())
    n_neg   = n_total - n_pos
    sha     = _sha256(df)

    print(f"  n_total={n_total}  n_pos={n_pos}  n_neg={n_neg}  "
          f"in_original_400={n_orig}", flush=True)
    print(f"  dis_mi: mean={df['dis_mi'].mean():.3f} "
          f"std={df['dis_mi'].std():.3f} "
          f"min={df['dis_mi'].min():.3f} "
          f"max={df['dis_mi'].max():.3f}", flush=True)
    print(f"  sha256(df)={sha}", flush=True)

    env["corpora"][corpus_name] = {
        "n_total":         n_total,
        "n_pos":           n_pos,
        "n_neg":           n_neg,
        "in_original_400": n_orig,
        "sha256":          sha,
    }

    if not dry_run:
        out_path = OUT_DIR / f"full_input_{corpus_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}", flush=True)
    else:
        print("  [dry_run] skipping write", flush=True)


if __name__ == "__main__":
    from typing import Callable
    main()
