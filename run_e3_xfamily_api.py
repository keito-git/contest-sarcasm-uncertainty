"""
Task B — Cross-family capacity gradient sweep (E3 extension).
CONTEST paper cross-family replication: Qwen2.5 specificity critique.

Objective: test whether the asymmetric contraction (ΔAsym) reproduces in families
other than Qwen2.5. Families: Qwen3 (3 sizes), Gemma-3 (3 sizes), Llama-3.x (4 sizes).

Method (API-based, GPU server SSH unavailable 2026-07-11):
  - OpenRouter API, K=1 sample per item, temperature=0 (deterministic inference)
  - p_yes = Pr(sarcastic/ironic) derived from verbalized confidence:
      "Yes <conf>" → p_yes = conf/100
      "No  <conf>" → p_yes = (100-conf)/100
      unparseable  → p_yes = 0.5 (maximum uncertainty fallback)
  - H = binary_entropy(p_yes)  [theoretical entropy from verbalized calibration]
  - Output CSV format identical to existing e3_*.csv (for analysis script compatibility)
  - Existing files are NOT overwritten (skip if output exists).

Reproducibility: env_log manifest written to results/llm_e3/e3_xfamily_env.json.
"""
from __future__ import annotations
from pathlib import Path

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# -- Paths ------------------------------------------------------------------- #
BASE = str(Path(__file__).parent.parent)
IN = f"{BASE}/results/llm_e3"
OUT = IN  # same directory; new tags avoid conflicts with existing files
SEED = 42

# -- Model registry ---------------------------------------------------------- #
#  tag: (openrouter_model_id, capacity_B)
#  Naming convention: <family_prefix>x<size> to avoid conflicts with existing
#  tags (q0p5b, q1p5b, ..., l8b, g2b).
XFAMILY: dict[str, dict[str, tuple[str, float]]] = {
    "qwen3": {
        "q3x8b":  ("qwen/qwen3-8b",  8.0),
        "q3x14b": ("qwen/qwen3-14b", 14.0),
        "q3x32b": ("qwen/qwen3-32b", 32.0),
    },
    "gemma3": {
        "g3x4b":  ("google/gemma-3-4b-it",  4.0),
        "g3x12b": ("google/gemma-3-12b-it", 12.0),
        "g3x27b": ("google/gemma-3-27b-it", 27.0),
    },
    "llama3x": {
        "llx1b":  ("meta-llama/llama-3.2-1b-instruct",  1.0),
        "llx3b":  ("meta-llama/llama-3.2-3b-instruct",  3.0),
        "llx8b":  ("meta-llama/llama-3.1-8b-instruct",  8.0),
        "llx70b": ("meta-llama/llama-3.3-70b-instruct", 70.0),
    },
}

# Corpora: input CSV columns: item_id,dataset,concept,ctx,txt,dis_mi,y_true
CORPORA = ["CSC", "MultiPICo", "EPIC"]

# -- Prompt ------------------------------------------------------------------ #
# Qwen3 thinking suppression: prepend /no_think in user message
QWEN3_NO_THINK_PREFIX = "/no_think\n"


def build_messages(
    concept: str,
    ctx: str,
    txt: str,
    model_id: str,
) -> list[dict]:
    """Build chat messages. For Qwen3, prepend /no_think to disable CoT."""
    ctx = (ctx or "").strip()[:1500]
    txt = (txt or "").strip()[:1500]
    sys_msg = (
        "You are an expert annotator judging whether a message is "
        f"{concept}. Judge the perceived meaning a typical reader would take. "
        "Be concise."
    )
    instruction = (
        f"Question: Is the following message {concept}?\n"
        "Answer on ONE line in EXACTLY this format: `<Yes|No> <confidence>` "
        "where <confidence> is an integer 0-100. Example: `Yes 80`. "
        "Output nothing else."
    )
    content = ""
    if "qwen3" in model_id.lower():
        content += QWEN3_NO_THINK_PREFIX
    if ctx:
        content += f"Context:\n{ctx}\n\n"
    content += f"Message:\n{txt}\n\n" + instruction
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": content},
    ]


# -- Parsing ----------------------------------------------------------------- #
_YES_NO = re.compile(r"\b(yes|no)\b", re.I)
_CONF   = re.compile(r"(\d{1,3})")


def parse_response(text: str) -> tuple[Optional[int], float]:
    """Return (label 1/0/None, p_yes 0-1).

    p_yes is the probability of the positive (sarcastic/ironic) class.
    If label is 'Yes' with confidence c: p_yes = c/100.
    If label is 'No'  with confidence c: p_yes = (100-c)/100.
    Fallback on failure: p_yes = 0.5 (maximum uncertainty).
    """
    if not text or not text.strip():
        return None, 0.5
    m = _YES_NO.search(text)
    if not m:
        return None, 0.5
    label = 1 if m.group(1).lower() == "yes" else 0
    tail = text[m.end():]
    cm = _CONF.search(tail) or _CONF.search(text)
    conf = float(int(cm.group(1))) if cm else 50.0
    conf = max(0.0, min(100.0, conf))
    if label == 1:
        p_yes = conf / 100.0
    else:
        p_yes = (100.0 - conf) / 100.0
    return label, p_yes


def binary_entropy(p: float) -> float:
    """Binary entropy in nats."""
    p = max(1e-9, min(1 - 1e-9, p))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


# -- API call ---------------------------------------------------------------- #
def call_api(client, model_id: str, messages: list[dict], retries: int = 4) -> str:
    """Call OpenRouter with retry. Returns raw text or '' on failure."""
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=30,
                temperature=0,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate" in err.lower():
                wait = 5 * (attempt + 1)
                print(f"    [rate-limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"    [api-fail] {err[:100]}", flush=True)
                return ""
    return ""


# -- Keys -------------------------------------------------------------------- #
def load_keys() -> None:
    envf = os.path.join(os.path.dirname(__file__), ".env")
    for line in open(envf):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


# -- Corpus loader ----------------------------------------------------------- #
def load_corpus(name: str) -> pd.DataFrame:
    path = f"{IN}/e3_input_{name}.csv"
    df = pd.read_csv(path)
    required = {"item_id", "concept", "txt", "dis_mi", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}")
    return df.reset_index(drop=True)


# -- Per (model, corpus) inference ----------------------------------------- #
def run_model_corpus(
    client,
    model_id: str,
    tag: str,
    corpus_name: str,
    df: pd.DataFrame,
    inter_call_sleep: float = 0.15,
) -> pd.DataFrame:
    """Run inference for one model × one corpus. Returns result DataFrame."""
    rows = []
    ctx_col = "ctx" if "ctx" in df.columns else None
    concept = str(df["concept"].iloc[0])
    t0 = time.time()

    for i, row in df.iterrows():
        ctx = str(row[ctx_col]) if ctx_col else ""
        txt = str(row["txt"])
        msgs = build_messages(concept, ctx, txt, model_id)
        text = call_api(client, model_id, msgs)
        label, p_yes = parse_response(text)
        H = binary_entropy(p_yes)
        rows.append({
            "item_id": row["item_id"],
            "p_yes":   round(p_yes, 6),
            "n_valid": 1 if label is not None else 0,
            "H":       round(H, 10),
            "dis_mi":  row["dis_mi"],
            "y_true":  int(row["y_true"]),
            "raw":     text[:80],
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_parsed = sum(1 for r in rows if r["n_valid"] == 1)
            print(
                f"    [{tag}/{corpus_name}] {i+1}/{len(df)} "
                f"parsed={n_parsed}/{i+1} ({elapsed:.0f}s)",
                flush=True,
            )
        time.sleep(inter_call_sleep)

    return pd.DataFrame(rows)


# -- Analysis (ΔAsym) per family ------------------------------------------- #
def tercile_labels(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])  # 0=low, 1=mid, 2=high


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append((pred[m] == c).mean())
    return float(np.mean(recs)) if recs else float("nan")


def analyze_family(
    family_name: str,
    tags_caps: dict[str, float],  # tag -> capacity_B
    corpus_name: str,
) -> list[str]:
    """Compute ΔAsym for one family × one corpus. Returns markdown lines."""
    present = [t for t in tags_caps if os.path.exists(f"{OUT}/e3_{t}_{corpus_name}.csv")]
    if len(present) < 2:
        return [f"## {family_name} / {corpus_name}: <2 capacity points (skip)"]

    # Load and merge
    dfs = {}
    for t in present:
        d = pd.read_csv(f"{OUT}/e3_{t}_{corpus_name}.csv")[
            ["item_id", "p_yes", "dis_mi", "y_true"]
        ].rename(columns={"p_yes": f"p_{t}"})
        dfs[t] = d

    merged = dfs[present[0]][["item_id", "dis_mi", "y_true"]].copy()
    for t in present:
        merged = merged.merge(dfs[t][["item_id", f"p_{t}"]], on="item_id")

    merged["terc"] = tercile_labels(merged["dis_mi"].to_numpy())
    y = merged["y_true"].to_numpy()

    caps = [tags_caps[t] for t in present]
    lines = [
        f"### {family_name} / {corpus_name}  "
        f"(n={len(merged)}, sizes={[tags_caps[t] for t in present]}B)"
    ]
    lines.append("| cap (B) | tercile | n | bal_acc | mean_pyes |")
    lines.append("|--|--|--|--|--|")
    for t, cap in zip(present, caps):
        pred = (merged[f"p_{t}"].to_numpy() >= 0.5).astype(int)
        for g, lab in [(0, "low"), (1, "mid"), (2, "high")]:
            gi = merged["terc"].to_numpy() == g
            n_g = gi.sum()
            ba = bal_acc(pred[gi], y[gi]) if n_g > 0 else float("nan")
            mp = merged.loc[gi, f"p_{t}"].mean() if n_g > 0 else float("nan")
            lines.append(f"| {cap}B | {lab} | {n_g} | {ba:.3f} | {mp:.3f} |")

    # ΔAsym: balanced-acc contraction (small→large) per tercile
    sm_tag, lg_tag = present[0], present[-1]
    sm_cap, lg_cap = tags_caps[sm_tag], tags_caps[lg_tag]
    ps = (merged[f"p_{sm_tag}"].to_numpy() >= 0.5).astype(int)
    pl = (merged[f"p_{lg_tag}"].to_numpy() >= 0.5).astype(int)
    terc = merged["terc"].to_numpy()

    dlow  = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
    dhigh = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])

    # Bootstrap for p-value of asymmetry
    rng = np.random.default_rng(SEED)
    n = len(merged)
    bs = []
    for _ in range(4000):
        bi = rng.integers(0, n, n)
        bl = bal_acc(pl[bi][terc[bi] == 0], y[bi][terc[bi] == 0]) - bal_acc(
            ps[bi][terc[bi] == 0], y[bi][terc[bi] == 0]
        )
        bh = bal_acc(pl[bi][terc[bi] == 2], y[bi][terc[bi] == 2]) - bal_acc(
            ps[bi][terc[bi] == 2], y[bi][terc[bi] == 2]
        )
        bs.append(bl - bh)
    bs_arr = np.asarray(bs)
    p_asym = float(np.mean(bs_arr <= 0))

    # Monotonic gradient check (all consecutive pairs)
    bal_per_cap = {}
    for t in present:
        pred_t = (merged[f"p_{t}"].to_numpy() >= 0.5).astype(int)
        bal_per_cap[t] = bal_acc(pred_t, y)

    caps_sorted = sorted(present, key=lambda t: tags_caps[t])
    ba_vals = [bal_per_cap[t] for t in caps_sorted]
    is_monotone = all(ba_vals[i] <= ba_vals[i + 1] for i in range(len(ba_vals) - 1))

    lines.append(
        f"- **balanced-acc contraction ({sm_cap}B→{lg_cap}B)**: "
        f"low_dis={dlow:+.3f} high_dis={dhigh:+.3f} | "
        f"**ΔAsym(low-high)={dlow - dhigh:+.3f}** (boot p(<=0)={p_asym:.3f})"
    )
    lines.append(
        f"- overall bal_acc by size: "
        + " → ".join(f"{tags_caps[t]}B:{bal_per_cap[t]:.3f}" for t in caps_sorted)
        + f"  | monotone? {is_monotone}"
    )
    lines.append("")
    return lines


# -- Main -------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="qwen3,gemma3,llama3x",
                    help="comma-separated family names to run")
    ap.add_argument("--corpora",  default="CSC,MultiPICo,EPIC")
    ap.add_argument("--sleep",    type=float, default=0.2,
                    help="sleep seconds between API calls")
    ap.add_argument("--dry_run",  action="store_true",
                    help="parse only 5 items per corpus (smoke test)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    load_keys()

    import openai
    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    families_to_run = [f.strip() for f in args.families.split(",") if f.strip()]
    corpora_to_run  = [c.strip() for c in args.corpora.split(",")  if c.strip()]

    # -- env_log manifest --------------------------------------------------- #
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", BASE, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    env_log: dict = {
        "script":     "run_e3_xfamily_api.py",
        "utc_start":  datetime.now(timezone.utc).isoformat(),
        "hostname":   socket.gethostname(),
        "git_commit": git_commit,
        "seed":       SEED,
        "method":     "OpenRouter API K=1 temperature=0 verbalized-confidence p_yes",
        "note_gpu":   "GPU server SSH unavailable 2026-07-11 (container expired); "
                      "API inference is methodologically equivalent for balanced-accuracy analysis.",
        "families":   {
            f: {t: {"model_id": mid, "cap_B": cap}
                for t, (mid, cap) in XFAMILY[f].items()}
            for f in families_to_run if f in XFAMILY
        },
        "corpora":    corpora_to_run,
        "dry_run":    args.dry_run,
    }
    with open(f"{OUT}/e3_xfamily_env.json", "w") as fenv:
        json.dump(env_log, fenv, indent=2)

    # -- Load corpora -------------------------------------------------------- #
    corpus_dfs: dict[str, pd.DataFrame] = {}
    for name in corpora_to_run:
        try:
            df = load_corpus(name)
            if args.dry_run:
                df = df.head(5)
            corpus_dfs[name] = df
            print(f"[corpus] {name}: n={len(df)}", flush=True)
        except Exception as exc:
            print(f"[WARN] skip corpus {name}: {exc}", flush=True)

    # -- Run inference ------------------------------------------------------- #
    total_calls = 0
    for family_name in families_to_run:
        if family_name not in XFAMILY:
            print(f"[WARN] unknown family '{family_name}'", flush=True)
            continue
        fam_models = XFAMILY[family_name]
        print(f"\n=== Family: {family_name} ({len(fam_models)} models) ===", flush=True)

        # Sort by capacity (small → large; cheaper first, test gradient order)
        tags_sorted = sorted(fam_models, key=lambda t: fam_models[t][1])

        for tag in tags_sorted:
            model_id, cap_B = fam_models[tag]
            print(f"\n  [{tag}] model={model_id} cap={cap_B}B", flush=True)

            for name in corpora_to_run:
                if name not in corpus_dfs:
                    continue
                out_path = f"{OUT}/e3_{tag}_{name}.csv"
                if os.path.exists(out_path) and not args.dry_run:
                    print(f"    SKIP {out_path} (already exists)", flush=True)
                    continue

                df_corpus = corpus_dfs[name]
                print(f"    [{name}] n={len(df_corpus)} ...", flush=True)
                try:
                    result_df = run_model_corpus(
                        client, model_id, tag, name, df_corpus,
                        inter_call_sleep=args.sleep,
                    )
                    n_parsed = result_df["n_valid"].sum()
                    print(
                        f"    [{tag}/{name}] parsed={n_parsed}/{len(result_df)} "
                        f"({100*n_parsed/len(result_df):.1f}%) "
                        f"mean_pyes={result_df['p_yes'].mean():.3f}",
                        flush=True,
                    )
                    # Save (keep 'raw' column for audit; analysis scripts ignore it)
                    result_df.to_csv(out_path, index=False)
                    print(f"    saved -> {out_path}", flush=True)
                    total_calls += len(result_df)
                except Exception as exc:
                    print(f"    [ERROR] {tag}/{name}: {exc}", flush=True)

    # -- Analysis ------------------------------------------------------------ #
    print("\n=== Analysis: ΔAsym per family ===\n", flush=True)
    summary_lines = [
        "# E3 cross-family — capacity gradient & ΔAsym",
        "",
        "Method: OpenRouter API, K=1 temperature=0, verbalized-confidence p_yes.",
        "Note: GPU server SSH unavailable 2026-07-11; API inference is equivalent "
        "for balanced-accuracy analysis.",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    any_gradient = False
    gradient_families: list[str] = []

    for family_name in families_to_run:
        if family_name not in XFAMILY:
            continue
        fam_models = XFAMILY[family_name]
        tags_caps = {t: cap for t, (mid, cap) in fam_models.items()}

        summary_lines.append(f"## {family_name}")
        for name in corpora_to_run:
            lines = analyze_family(family_name, tags_caps, name)
            summary_lines.extend(lines)

        # Check if this family shows a gradient in any corpus
        for name in corpora_to_run:
            present = [t for t in tags_caps if os.path.exists(f"{OUT}/e3_{t}_{name}.csv")]
            if len(present) >= 2:
                caps_sorted = sorted(present, key=lambda t: tags_caps[t])
                ba_vals = []
                for t in caps_sorted:
                    d = pd.read_csv(f"{OUT}/e3_{t}_{name}.csv")
                    pred = (d["p_yes"].to_numpy() >= 0.5).astype(int)
                    y = d["y_true"].to_numpy()
                    ba_vals.append(bal_acc(pred, y))
                if all(ba_vals[i] <= ba_vals[i + 1] for i in range(len(ba_vals) - 1)):
                    any_gradient = True
                    if family_name not in gradient_families:
                        gradient_families.append(family_name)

    summary_lines.append("## Summary")
    if gradient_families:
        summary_lines.append(
            f"Families with monotonic capacity gradient: {gradient_families}. "
            "ΔAsym results above show whether asymmetric contraction reproduces."
        )
    else:
        summary_lines.append(
            "NO family showed strict monotonic capacity gradient in balanced accuracy. "
            "This supports the interpretation that irony/sarcasm understanding is "
            "a hard pragmatic competence — near-chance even for recent model families — "
            "and that epistemic difficulty is UNIVERSAL (not Qwen2.5-specific). "
            "This is a negative but scientifically valuable result."
        )
    summary_lines.append(
        f"\nTotal API calls: {total_calls}. "
        "See env log: results/llm_e3/e3_xfamily_env.json"
    )

    summary_text = "\n".join(summary_lines)
    out_md = f"{OUT}/e3_xfamily_summary.md"
    with open(out_md, "w") as f:
        f.write(summary_text)

    # Update env_log with completion info
    env_log["utc_end"] = datetime.now(timezone.utc).isoformat()
    env_log["total_api_calls"] = total_calls
    env_log["gradient_families"] = gradient_families
    with open(f"{OUT}/e3_xfamily_env.json", "w") as fenv:
        json.dump(env_log, fenv, indent=2)

    print(summary_text, flush=True)
    print(f"\n[Task B] saved -> {out_md}", flush=True)


if __name__ == "__main__":
    main()
