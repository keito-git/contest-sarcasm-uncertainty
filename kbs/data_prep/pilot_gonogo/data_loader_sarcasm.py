"""
Data loaders for the sarcasm/irony "ideal testbed" pilot.

Datasets used:
  1. CSC  — Conversational Sarcasm Corpus (Jang & Frassinelli, NAACL 2024)
            GitHub: https://github.com/CoPsyN/CSC  (MIT license)
            7,036 context-response items / up to 360 annotators / 6-pt Likert.
            Unique feature: BOTH self-rating (speaker intent) AND 3rd-party ratings.
  2. MultiPICo English — Casola et al. ACL 2024.
            HuggingFace: Multilingual-Perspectivist-NLU/MultiPICo (English only).
            ~2,999 items / 74 annotators / binary irony label.
            Has `post` (context) + `reply` (text) columns.

NOTE: EPIC (Frenda et al. ACL 2023, Multilingual-Perspectivist-NLU/EPIC) overlaps ~88%
with MultiPICo English and lacks the `post` context column needed for delta_lex.
EPIC is therefore dropped to avoid data duplication.

Output files:
  - This module writes ONLY to data/raw/csc_data_full.csv and data/raw/multipico_en.parquet.
  - Does NOT touch goemotions_raw.parquet, measuring_hate_speech.parquet, or any
    processed/*.parquet created by the existing pilot_gonogo pipeline.

GPU policy: CUDA_VISIBLE_DEVICES="" is set before any torch import in run_pilot_sarcasm.py.
No frontier-LLM API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE = WORK_DIR
DATA_RAW = BASE / "data" / "raw"
DATA_PROCESSED = BASE / "data" / "processed"

CSC_URL = "https://raw.githubusercontent.com/CoPsyN/CSC/main/data_full.csv"
CSC_RAW_PATH = DATA_RAW / "csc_data_full.csv"
MULTIPICO_RAW_PATH = DATA_RAW / "multipico_en.parquet"

# Sarcasm scale for CSC: 1 (not sarcastic) – 6 (very sarcastic)
CSC_SARCASM_MIN = 1
CSC_SARCASM_MAX = 6


# ── Utilities ────────────────────────────────────────────────────────────────


def _sha256_df(df: pd.DataFrame) -> str:
    """Compute 16-hex SHA-256 of a DataFrame for reproducibility logging."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def _normalize_sarcasm(score: float | np.ndarray) -> float | np.ndarray:
    """Normalize CSC sarcasm score from [1, 6] to [0, 1]."""
    return (score - CSC_SARCASM_MIN) / (CSC_SARCASM_MAX - CSC_SARCASM_MIN)


# ── CSC ──────────────────────────────────────────────────────────────────────


def load_csc_raw(
    cache_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, str]:
    """Download and cache the CSC per-annotator CSV.

    Columns of interest:
        global_context_name     : int, context scenario ID
        context_text            : str, scenario text
        response_text           : str, speaker's response (the text we analyze)
        global_speaker          : int, speaker (author) ID
        global_evaluator        : int, 3rd-party evaluator ID
        sarcasm_score_by_speaker: int (1-6), speaker's OWN sarcasm intent
        sarcasm_score_by_evaluator: int (1-6), evaluator's perceived sarcasm

    Returns
    -------
    df : pd.DataFrame  (per-annotator rows)
    data_hash : str
    """
    cache_path = cache_path or CSC_RAW_PATH

    if cache_path.exists():
        print(f"[CSC] Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        print(f"[CSC] Downloading from GitHub ({CSC_URL}) ...")
        result = subprocess.run(
            ["curl", "-L", "-s", CSC_URL, "-o", str(cache_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"curl failed (rc={result.returncode}): {result.stderr.decode()}"
            )
        df = pd.read_csv(cache_path)
        print(f"[CSC] Saved to {cache_path}")

    data_hash = _sha256_df(df)
    print(
        f"[CSC] Loaded: {len(df):,} annotator rows, "
        f"{df['global_context_name'].nunique()} contexts, "
        f"{df.groupby(['global_context_name','global_speaker']).ngroups:,} items, "
        f"hash={data_hash}"
    )
    return df, data_hash


def aggregate_csc(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CSC per-annotator rows to per-item level.

    Item = unique (global_context_name, global_speaker) pair.
    Each item has:
      - One response_text (constant per item)
      - One context_text (constant per context scenario)
      - sarcasm_score_by_speaker: speaker's self-rating of sarcasm intent (1-6)
      - N evaluator scores: sarcasm_score_by_evaluator (1-6)

    Computed fields:
      - mean_eval:  mean of evaluator sarcasm ratings
      - std_eval:   std of evaluator sarcasm ratings
      - mad_eval:   MAD of evaluator sarcasm ratings
      - disagreement: variance of evaluator ratings (primary disagree measure)
      - self_sarcasm_norm: (speaker_rating - 1) / 5  ∈ [0, 1]
      - delta_self_3rd: |speaker_norm - mean_eval_norm| / 1 ∈ [0, 1]
        (self vs 3rd-party perception gap; NOT disagreement but item-level stat)
      - n_evaluators: number of evaluator ratings per item
      - text_length: word count of response_text
    """
    records = []
    for (ctx_id, spk_id), grp in df_raw.groupby(
        ["global_context_name", "global_speaker"]
    ):
        response_text = grp["response_text"].iloc[0]
        context_text = grp["context_text"].iloc[0]
        self_sarcasm = grp["sarcasm_score_by_speaker"].iloc[0]  # constant per item
        eval_scores = grp["sarcasm_score_by_evaluator"].values

        mean_eval = float(eval_scores.mean())
        std_eval = float(eval_scores.std(ddof=1)) if len(eval_scores) > 1 else 0.0
        mad_eval = float(np.mean(np.abs(eval_scores - mean_eval)))
        disagreement = float(np.var(eval_scores, ddof=1)) if len(eval_scores) > 1 else 0.0

        self_norm = float(_normalize_sarcasm(self_sarcasm))
        mean_eval_norm = float(_normalize_sarcasm(mean_eval))
        delta_self_3rd = float(abs(self_norm - mean_eval_norm))

        n_words = len(response_text.split()) if isinstance(response_text, str) else 0

        records.append(
            {
                "item_id": f"{ctx_id}_{spk_id}",
                "global_context_name": ctx_id,
                "global_speaker": spk_id,
                "response_text": response_text,
                "context_text": context_text,
                "n_evaluators": len(eval_scores),
                "self_sarcasm": self_sarcasm,
                "self_sarcasm_norm": self_norm,
                "mean_eval": mean_eval,
                "std_eval": std_eval,
                "mad_eval": mad_eval,
                "disagreement": disagreement,
                "mean_eval_norm": mean_eval_norm,
                "delta_self_3rd": delta_self_3rd,
                "text_length": n_words,
            }
        )

    agg = pd.DataFrame(records)
    print(
        f"[CSC] Aggregated: {len(agg):,} items, "
        f"n_evaluators mean={agg['n_evaluators'].mean():.1f}, "
        f"disagreement (var) mean={agg['disagreement'].mean():.4f} "
        f"std={agg['disagreement'].std():.4f}"
    )
    return agg


# ── MultiPICo English ─────────────────────────────────────────────────────────


def load_multipico_en_raw(
    cache_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, str]:
    """Load MultiPICo English from HuggingFace or local cache.

    Only the English subset is used. EPIC is dropped (88% overlap with
    MultiPICo English; MultiPICo adds `post` context column needed for delta_lex).

    Columns of interest:
        reply_id      : str/int, unique reply identifier (item)
        post          : str, the parent post/context (context text)
        reply         : str, the reply text (what we analyze)
        label         : int (0=not ironic, 1=ironic)
        annotator_id  : int, annotator identifier

    Returns
    -------
    df : pd.DataFrame  (per-annotator rows, English only)
    data_hash : str
    """
    from datasets import load_dataset  # type: ignore

    cache_path = cache_path or MULTIPICO_RAW_PATH

    if cache_path.exists():
        print(f"[MultiPICo] Loading English from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("[MultiPICo] Downloading from HuggingFace ...")
        ds = load_dataset(
            "Multilingual-Perspectivist-NLU/MultiPICo",
            split="train",
            trust_remote_code=True,
        )
        df = ds.to_pandas()
        df_en = df[df["language"] == "en"].copy().reset_index(drop=True)
        df_en.to_parquet(cache_path, index=False)
        df = df_en
        print(f"[MultiPICo] Saved English subset to {cache_path}")

    # Ensure English-only (in case cache was saved before filter)
    if "language" in df.columns:
        df = df[df["language"] == "en"].copy()

    data_hash = _sha256_df(df)
    print(
        f"[MultiPICo] Loaded: {len(df):,} annotator rows, "
        f"{df['reply_id'].nunique():,} unique items, "
        f"{df['annotator_id'].nunique():,} annotators, "
        f"hash={data_hash}"
    )
    return df, data_hash


def aggregate_multipico(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MultiPICo per-annotator rows to per-item level.

    Item = unique reply_id.
    Each item has:
      - One reply text (constant per item)
      - One post text (context; constant per item)
      - N binary labels (0=not ironic, 1=ironic) from different annotators

    Computed fields:
      - p_ironic:     fraction of annotators labeling as ironic
      - n_annotators: number of annotator labels per item
      - disagreement: normalized entropy of binary label distribution
        H = -(p*log2(p) + (1-p)*log2(1-p)), then normalized to [0,1] by dividing by log2(2)=1
        (0 = all agree; 1 = 50/50 split)
      - text_length:  word count of reply
      - context_length: word count of post
    """
    eps = 1e-10
    records = []

    for item_id, grp in df_raw.groupby("reply_id"):
        reply = grp["reply"].iloc[0]
        post = grp["post"].iloc[0]
        labels = grp["label"].values.astype(float)

        n = len(labels)
        p_iro = float(labels.mean())
        # Normalized binary entropy
        h = -(p_iro * np.log2(p_iro + eps) + (1 - p_iro) * np.log2(1 - p_iro + eps))
        h = float(np.clip(h, 0.0, 1.0))  # already in [0,1] since max entropy of binary = 1 bit

        n_words_reply = len(reply.split()) if isinstance(reply, str) else 0
        n_words_post = len(post.split()) if isinstance(post, str) and post else 0

        records.append(
            {
                "item_id": item_id,
                "reply": reply,
                "post": post,
                "n_annotators": n,
                "p_ironic": p_iro,
                "disagreement": h,
                "text_length": n_words_reply,
                "context_length": n_words_post,
            }
        )

    agg = pd.DataFrame(records)
    print(
        f"[MultiPICo] Aggregated: {len(agg):,} items, "
        f"n_annotators mean={agg['n_annotators'].mean():.1f}, "
        f"disagreement (H) mean={agg['disagreement'].mean():.4f} "
        f"std={agg['disagreement'].std():.4f}"
    )
    return agg


# ── Delta computation (lexical) ───────────────────────────────────────────────


def compute_delta_lex_csc(df: pd.DataFrame) -> pd.Series:
    """Compute delta_lex for CSC: context-sentiment-mismatch (label-free).

    delta_lex = |VADER_compound(response_text) - VADER_compound(context_text)| / 2

    Rationale: sarcastic responses have sentiment that diverges from the scenario
    context. Sincere responses echo the context sentiment.
    Range: [0, 1]. Label-free (no annotator labels used).

    Parameters
    ----------
    df : pd.DataFrame
        Output of aggregate_csc(), must have 'response_text' and 'context_text'.

    Returns
    -------
    pd.Series  named 'delta_lex'
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    sia = SentimentIntensityAnalyzer()
    resp_vader = np.array(
        [sia.polarity_scores(str(t))["compound"] for t in df["response_text"].tolist()],
        dtype=float,
    )
    ctx_vader = np.array(
        [sia.polarity_scores(str(t))["compound"] for t in df["context_text"].tolist()],
        dtype=float,
    )
    delta = np.abs(resp_vader - ctx_vader) / 2.0
    return pd.Series(delta, index=df.index, name="delta_lex")


def compute_delta_lex_csc_self(df: pd.DataFrame) -> pd.Series:
    """Compute delta_self for CSC: |VADER(response)| x self_sarcasm_norm.

    Uses the speaker's SELF-RATING of sarcasm intent (not 3rd-party labels).
    self_sarcasm_norm in [0,1] (normalized from 1-6 scale).
    |VADER(response)| measures the 'surface signal strength'.

    High delta_self: response has strong surface sentiment AND speaker intends
    high sarcasm → large literal-intended gap.
    Range: [0, 1]. Uses self-rating (NOT the disagreement target).

    Parameters
    ----------
    df : pd.DataFrame
        Output of aggregate_csc(), must have 'response_text', 'self_sarcasm_norm'.

    Returns
    -------
    pd.Series  named 'delta_self'
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    sia = SentimentIntensityAnalyzer()
    resp_vader = np.abs(
        np.array(
            [sia.polarity_scores(str(t))["compound"] for t in df["response_text"].tolist()],
            dtype=float,
        )
    )
    delta_self = resp_vader * df["self_sarcasm_norm"].values
    return pd.Series(delta_self, index=df.index, name="delta_self")


def compute_delta_lex_multipico(df: pd.DataFrame) -> pd.Series:
    """Compute delta_lex for MultiPICo: context-sentiment-mismatch (label-free).

    delta_lex = |VADER_compound(reply) - VADER_compound(post)| / 2

    Rationale: ironic replies diverge in sentiment from the original post.
    Same formula as CSC to enable cross-dataset comparison.
    Range: [0, 1]. Label-free.

    Parameters
    ----------
    df : pd.DataFrame
        Output of aggregate_multipico(), must have 'reply' and 'post'.

    Returns
    -------
    pd.Series  named 'delta_lex'
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    sia = SentimentIntensityAnalyzer()
    reply_vader = np.array(
        [sia.polarity_scores(str(t))["compound"] for t in df["reply"].tolist()],
        dtype=float,
    )
    post_vader = np.array(
        [sia.polarity_scores(str(t))["compound"] for t in df["post"].fillna("").tolist()],
        dtype=float,
    )
    delta = np.abs(reply_vader - post_vader) / 2.0
    return pd.Series(delta, index=df.index, name="delta_lex")


def compute_delta_ctx(
    texts: list[str],
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute delta_ctx = |VADER_compound - (P(pos)-P(neg))_RoBERTa| / 2.

    Label-free literal-vs-contextual sentiment discrepancy.
    Identical formula to run_delta_ctx.py to ensure consistency.

    CUDA_VISIBLE_DEVICES="" MUST be set before calling (enforced in
    run_pilot_sarcasm.py before any torch import).

    Parameters
    ----------
    texts : list[str]
    model_name : str
        Default: cardiffnlp/twitter-roberta-base-sentiment-latest.
        Label order for this model: 0=negative, 1=neutral, 2=positive.
    batch_size : int

    Returns
    -------
    np.ndarray  delta_ctx values in [0, 1]
    """
    import torch
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    sia = SentimentIntensityAnalyzer()
    lex = np.array([sia.polarity_scores(t)["compound"] for t in texts], dtype=float)

    print(f"[delta_ctx] Loading {model_name} on CPU ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    ctx = np.empty(len(texts), dtype=float)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            ctx[i : i + len(batch)] = probs[:, 2] - probs[:, 0]  # P(pos) - P(neg)
            if i % (batch_size * 50) == 0:
                print(f"    [delta_ctx] {i}/{len(texts)}", flush=True)

    return np.abs(lex - ctx) / 2.0
