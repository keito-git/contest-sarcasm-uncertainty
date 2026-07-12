"""
run_e3_mhs_api.py — MHS hate speech E3 capacity sweep (one model at a time).

Purpose: Test whether asymmetric contraction (C3 claim) generalises to hate speech
detection, extending the CSC/MultiPICo/EPIC findings in run_e3_xfamily_api.py.

Key design choices:
  - Uses the same 10 models (Qwen3/Gemma3/Llama3x) and same prompt format as xfamily.
  - Checkpoint every 50 items → partial runs survive Bash timeout.
  - Output CSVs e3_{tag}_MHS.csv match the exact schema of other e3 files,
    so run_e3_xfamily_api.py::analyze_family() can read them directly.
  - MHS dis_mi is BIMODAL: 64% items have dis_mi=0 (unanimous), 36% have dis_mi>0.
    Tercile labels therefore give: low≈n/3 all-agree items, mid≈tiny, high≈dis>0 items.
    This is documented and reported honestly; ΔAsym is still interpretable as
    "unanimous-human-agree" vs "any-disagree".

Usage:
  python run_e3_mhs_api.py --tag q3x8b          # run one model, checkpoint-resumable
  python run_e3_mhs_api.py --tag q3x14b
  python run_e3_mhs_api.py --analyze             # compute ΔAsym for all completed tags
  python run_e3_mhs_api.py --tag q3x8b --dry    # smoke-test (5 items only)
  python run_e3_mhs_api.py --list                # show which tags are completed

Reproducibility: env_log for each run in results/llm_e3/e3_mhs_env_{tag}.json.
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
from scipy.stats import rankdata

# ── Paths ──────────────────────────────────────────────────────────────────── #
BASE = str(Path(__file__).parent.parent)
IN = f"{BASE}/results/llm_e3"
OUT = IN
SEED = 42
CORPUS = "MHS"
INPUT_CSV = f"{IN}/e3_input_{CORPUS}.csv"

# ── Model registry ─────────────────────────────────────────────────────────── #
# Identical to XFAMILY in run_e3_xfamily_api.py
MODELS: dict[str, tuple[str, float]] = {
    "q3x8b":  ("qwen/qwen3-8b",                        8.0),
    "q3x14b": ("qwen/qwen3-14b",                       14.0),
    "q3x32b": ("qwen/qwen3-32b",                       32.0),
    "g3x4b":  ("google/gemma-3-4b-it",                 4.0),
    "g3x12b": ("google/gemma-3-12b-it",                12.0),
    "g3x27b": ("google/gemma-3-27b-it",                27.0),
    "llx1b":  ("meta-llama/llama-3.2-1b-instruct",     1.0),
    "llx3b":  ("meta-llama/llama-3.2-3b-instruct",     3.0),
    "llx8b":  ("meta-llama/llama-3.1-8b-instruct",     8.0),
    "llx70b": ("meta-llama/llama-3.3-70b-instruct",    70.0),
}

FAMILY_OF: dict[str, str] = {
    "q3x8b": "qwen3", "q3x14b": "qwen3", "q3x32b": "qwen3",
    "g3x4b": "gemma3", "g3x12b": "gemma3", "g3x27b": "gemma3",
    "llx1b": "llama3x", "llx3b": "llama3x", "llx8b": "llama3x", "llx70b": "llama3x",
}

FAMILY_TAGS: dict[str, list[str]] = {
    "qwen3":   ["q3x8b",  "q3x14b", "q3x32b"],
    "gemma3":  ["g3x4b",  "g3x12b", "g3x27b"],
    "llama3x": ["llx1b",  "llx3b",  "llx8b",  "llx70b"],
}

QWEN3_NO_THINK_PREFIX = "/no_think\n"


# ── Prompt ─────────────────────────────────────────────────────────────────── #
def build_messages(concept: str, ctx: str, txt: str, model_id: str) -> list[dict]:
    """Build chat messages (identical structure to run_e3_xfamily_api.py)."""
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


# ── Response parsing ────────────────────────────────────────────────────────── #
_YES_NO = re.compile(r"\b(yes|no)\b", re.I)
_CONF   = re.compile(r"(\d{1,3})")


def parse_response(text: str) -> tuple[Optional[int], float]:
    """Return (label 1/0/None, p_yes 0-1). Identical to xfamily version."""
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
    p_yes = conf / 100.0 if label == 1 else (100.0 - conf) / 100.0
    return label, p_yes


def binary_entropy(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


# ── API call ───────────────────────────────────────────────────────────────── #
def call_api(client, model_id: str, messages: list[dict], retries: int = 4) -> str:
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
                wait = 8 * (attempt + 1)
                print(f"    [rate-limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"    [api-fail] {err[:120]}", flush=True)
                return ""
    return ""


# ── Keys ───────────────────────────────────────────────────────────────────── #
def load_keys() -> None:
    envf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    for line in open(envf):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


# ── Checkpoint helpers ─────────────────────────────────────────────────────── #
def ckpt_path(tag: str) -> str:
    return f"{OUT}/_ckpt_e3_{tag}_MHS.parquet"


def out_path(tag: str) -> str:
    return f"{OUT}/e3_{tag}_MHS.csv"


def load_checkpoint(tag: str) -> tuple[list[dict], int]:
    """Load existing checkpoint. Returns (rows, start_idx)."""
    cp = ckpt_path(tag)
    if not os.path.exists(cp):
        return [], 0
    try:
        df = pd.read_parquet(cp)
        rows = df.to_dict("records")
        start = len(rows)
        print(f"  [checkpoint] resumed: {start} items already done", flush=True)
        return rows, start
    except Exception as exc:
        print(f"  [WARN] checkpoint corrupt, starting fresh: {exc}", flush=True)
        return [], 0


def save_checkpoint(rows: list[dict], tag: str) -> None:
    pd.DataFrame(rows).to_parquet(ckpt_path(tag), index=False)


# ── Inference for one model on MHS ────────────────────────────────────────── #
def run_model_mhs(
    client,
    tag: str,
    df: pd.DataFrame,
    inter_call_sleep: float = 0.2,
    dry: bool = False,
) -> bool:
    """
    Run inference for one tag on MHS.
    Returns True on success, False if interrupted mid-way (checkpoint saved).
    """
    out = out_path(tag)
    if os.path.exists(out) and not dry:
        print(f"  SKIP {out} (already exists)", flush=True)
        return True

    model_id, cap_B = MODELS[tag]
    concept = str(df["concept"].iloc[0])
    ctx_col = "ctx" if "ctx" in df.columns else None

    if dry:
        df = df.head(5)
        print(f"  [DRY] {tag} {model_id} {cap_B}B — first 5 items", flush=True)

    rows, start = load_checkpoint(tag)
    print(
        f"  [{tag}] model={model_id} cap={cap_B}B n={len(df)} start={start}",
        flush=True,
    )
    t0 = time.time()

    for i in range(start, len(df)):
        row = df.iloc[i]
        ctx = str(row[ctx_col]) if ctx_col and pd.notna(row[ctx_col]) else ""
        txt = str(row["txt"])
        msgs = build_messages(concept, ctx, txt, model_id)
        raw = call_api(client, model_id, msgs)
        label, p_yes = parse_response(raw)
        H = binary_entropy(p_yes)
        rows.append({
            "item_id": int(row["item_id"]),
            "p_yes":   round(p_yes, 6),
            "n_valid": 1 if label is not None else 0,
            "H":       round(H, 10),
            "dis_mi":  float(row["dis_mi"]),
            "y_true":  int(row["y_true"]),
            "raw":     raw[:80],
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_parsed = sum(1 for r in rows if r["n_valid"] == 1)
            print(
                f"    [{tag}] {i+1}/{len(df)} "
                f"parsed={n_parsed} ({elapsed:.0f}s)",
                flush=True,
            )
            if not dry:
                save_checkpoint(rows, tag)
        time.sleep(inter_call_sleep)

    # Write final CSV
    final = pd.DataFrame(rows)
    n_parsed = final["n_valid"].sum()
    print(
        f"  [{tag}] DONE  n={len(final)} parsed={n_parsed}/{len(final)} "
        f"({100*n_parsed/len(final):.1f}%) "
        f"mean_pyes={final['p_yes'].mean():.3f}",
        flush=True,
    )
    if not dry:
        final.to_csv(out, index=False)
        print(f"  saved -> {out}", flush=True)
        # Remove checkpoint after successful final write
        cp = ckpt_path(tag)
        if os.path.exists(cp):
            os.remove(cp)
    return True


# ── Analysis: ΔAsym per family ────────────────────────────────────────────── #
def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append((pred[m] == c).mean())
    return float(np.mean(recs)) if recs else float("nan")


def tercile_labels(a: np.ndarray) -> np.ndarray:
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])  # 0=low,1=mid,2=high


def analyze_all() -> str:
    """Analyze all completed e3_*_MHS.csv files. Returns markdown summary."""
    lines = [
        f"# E3 MHS (hate speech) — capacity gradient & ΔAsym",
        "",
        f"Method: OpenRouter API, K=1 temperature=0, verbalized-confidence p_yes.",
        f"Note: MHS dis_mi is bimodal (dis=0: unanimous annotators ≈64%; dis>0: any-disagree ≈36%).",
        f"Tercile split is degenerate: low≈256 all-agree, mid≈tiny, high≈133 any-disagree.",
        f"ΔAsym = dAcc_low(all-agree) - dAcc_high(any-disagree).",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    for family_name, tags in FAMILY_TAGS.items():
        completed = [t for t in tags if os.path.exists(out_path(t))]
        if len(completed) < 2:
            lines.append(
                f"## {family_name}: only {len(completed)}/{len(tags)} tags complete "
                f"(need ≥2 for ΔAsym) — skip"
            )
            lines.append("")
            continue

        # Load and merge
        dfs: dict[str, pd.DataFrame] = {}
        for t in completed:
            d = pd.read_csv(out_path(t))[["item_id", "p_yes", "dis_mi", "y_true"]].rename(
                columns={"p_yes": f"p_{t}"}
            )
            dfs[t] = d

        merged = dfs[completed[0]][["item_id", "dis_mi", "y_true"]].copy()
        for t in completed:
            merged = merged.merge(dfs[t][["item_id", f"p_{t}"]], on="item_id")

        merged["terc"] = tercile_labels(merged["dis_mi"].to_numpy())
        y = merged["y_true"].to_numpy()
        terc = merged["terc"].to_numpy()

        caps_map = {t: MODELS[t][1] for t in completed}
        caps_sorted = sorted(completed, key=lambda t: MODELS[t][1])
        cap_labels = [f"{MODELS[t][1]}B" for t in caps_sorted]

        lines.append(
            f"## {family_name}  "
            f"(n={len(merged)}, pos_rate={y.mean():.3f}, "
            f"caps={cap_labels})"
        )
        lines.append(
            f"Note: tercile distribution: "
            f"low={int((terc==0).sum())}, mid={int((terc==1).sum())}, "
            f"high={int((terc==2).sum())} items."
        )
        lines.append("| cap (B) | tercile | n | bal_acc | yes_rate |")
        lines.append("|--|--|--|--|--|")
        for t in caps_sorted:
            pred = (merged[f"p_{t}"].to_numpy() >= 0.5).astype(int)
            for g, lab in [(0, "low"), (1, "mid"), (2, "high")]:
                gi = terc == g
                n_g = int(gi.sum())
                ba = bal_acc(pred[gi], y[gi]) if n_g > 0 else float("nan")
                yr = pred[gi].mean() if n_g > 0 else float("nan")
                lines.append(
                    f"| {MODELS[t][1]}B | {lab} | {n_g} | "
                    f"{'nan' if np.isnan(ba) else f'{ba:.3f}'} | "
                    f"{'nan' if np.isnan(yr) else f'{yr:.3f}'} |"
                )

        # Overall balanced acc by capacity
        ba_by_cap = {}
        for t in caps_sorted:
            pred = (merged[f"p_{t}"].to_numpy() >= 0.5).astype(int)
            ba_by_cap[t] = bal_acc(pred, y)
        is_mono = all(
            ba_by_cap[caps_sorted[i]] <= ba_by_cap[caps_sorted[i + 1]]
            for i in range(len(caps_sorted) - 1)
        )
        lines.append(
            "- Overall bal_acc: "
            + " → ".join(f"{MODELS[t][1]}B:{ba_by_cap[t]:.3f}" for t in caps_sorted)
            + f"  | monotone? {is_mono}"
        )

        # ΔAsym: small → large, low vs high tercile
        sm_tag, lg_tag = caps_sorted[0], caps_sorted[-1]
        sm_cap, lg_cap = MODELS[sm_tag][1], MODELS[lg_tag][1]
        ps = (merged[f"p_{sm_tag}"].to_numpy() >= 0.5).astype(int)
        pl = (merged[f"p_{lg_tag}"].to_numpy() >= 0.5).astype(int)

        dlow  = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
        dhigh = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])
        dasym = dlow - dhigh

        # Bootstrap p-value (4000 resamples)
        rng = np.random.default_rng(SEED)
        n = len(merged)
        bs_vals = []
        for _ in range(4000):
            bi = rng.integers(0, n, n)
            terc_bi = terc[bi]
            y_bi = y[bi]
            ps_bi = ps[bi]
            pl_bi = pl[bi]
            bl = (
                bal_acc(pl_bi[terc_bi == 0], y_bi[terc_bi == 0])
                - bal_acc(ps_bi[terc_bi == 0], y_bi[terc_bi == 0])
            )
            bh = (
                bal_acc(pl_bi[terc_bi == 2], y_bi[terc_bi == 2])
                - bal_acc(ps_bi[terc_bi == 2], y_bi[terc_bi == 2])
            )
            bs_vals.append(bl - bh)
        bs_arr = np.asarray(bs_vals)
        p_asym = float(np.mean(bs_arr <= 0))

        lines.append(
            f"- **ΔAsym ({sm_cap}B→{lg_cap}B)**: "
            f"low_dis(agree)={dlow:+.3f} high_dis(split)={dhigh:+.3f} | "
            f"**ΔAsym(low-high)={dasym:+.3f}** (boot p(<=0)={p_asym:.3f})"
        )

        # Interpretation
        if dasym > 0 and p_asym < 0.05:
            verdict = "C3 CONFIRMED: asymmetric contraction significant (p<0.05)."
        elif dasym > 0 and p_asym < 0.10:
            verdict = "C3 MARGINAL: positive asymmetry but p≥0.05 (marginal)."
        elif dasym > 0:
            verdict = "C3 DIRECTIONAL: positive asymmetry but not significant."
        else:
            verdict = "C3 NOT REPRODUCED in this family: ΔAsym ≤ 0."
        lines.append(f"- *{verdict}*")
        lines.append("")

    # Which families are complete enough to claim
    complete_families = [
        f for f, tags in FAMILY_TAGS.items()
        if sum(1 for t in tags if os.path.exists(out_path(t))) >= 2
    ]
    all_families = list(FAMILY_TAGS.keys())
    incomplete = [
        f"{f} ({sum(os.path.exists(out_path(t)) for t in FAMILY_TAGS[f])}/{len(FAMILY_TAGS[f])})"
        for f in all_families
        if f not in complete_families
    ]

    lines.append("## Coverage")
    lines.append(
        f"Families with ≥2 completed tags (analyzed): {complete_families}"
    )
    if incomplete:
        lines.append(f"Families incomplete (pending): {incomplete}")

    lines.append("")
    lines.append("## Notes on MHS dis_mi bimodality")
    lines.append(
        "- 256/400 items have dis_mi=0 (all annotators agree on label). "
        "These are concentrated in the low tercile."
    )
    lines.append(
        "- 144/400 items have dis_mi>0 (at least one disagreement). "
        "These form the mid/high tercile."
    )
    lines.append(
        "- The degenerate tercile (low n≈256, mid n≈11, high n≈133) is an inherent "
        "property of MHS annotation design (median 3 annotators, many unanimous). "
        "ΔAsym compares unanimous (all-agree) vs any-disagree strata."
    )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="MHS hate speech E3 sweep — one model at a time."
    )
    ap.add_argument("--tag", type=str, help="model tag (e.g. q3x8b)")
    ap.add_argument(
        "--analyze", action="store_true",
        help="analyze all completed tags and write summary",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="list completed and missing tags",
    )
    ap.add_argument("--dry", action="store_true", help="smoke-test with 5 items")
    ap.add_argument("--sleep", type=float, default=0.2, help="inter-call sleep (s)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if args.list:
        print("=== MHS E3 completion status ===")
        for family_name, tags in FAMILY_TAGS.items():
            for t in tags:
                model_id, cap_B = MODELS[t]
                exists = os.path.exists(out_path(t))
                ckpt   = os.path.exists(ckpt_path(t))
                status = "DONE" if exists else ("PARTIAL(ckpt)" if ckpt else "missing")
                print(f"  {t:10s} {cap_B:5.1f}B  [{family_name}]  {status}")
        sys.exit(0)

    if args.analyze:
        summary = analyze_all()
        out_md = f"{OUT}/e3_mhs_summary.md"
        with open(out_md, "w") as f:
            f.write(summary)
        print(summary, flush=True)
        print(f"\n[analyze] saved -> {out_md}", flush=True)
        sys.exit(0)

    if not args.tag:
        print("ERROR: specify --tag or --analyze or --list", file=sys.stderr)
        sys.exit(1)

    tag = args.tag
    if tag not in MODELS:
        print(
            f"ERROR: unknown tag '{tag}'. Valid: {list(MODELS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    load_keys()
    import openai
    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    # Env log
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", BASE, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    env_log = {
        "script":     "run_e3_mhs_api.py",
        "tag":        tag,
        "model_id":   MODELS[tag][0],
        "cap_B":      MODELS[tag][1],
        "corpus":     CORPUS,
        "utc_start":  datetime.now(timezone.utc).isoformat(),
        "hostname":   socket.gethostname(),
        "git_commit": git_commit,
        "seed":       SEED,
        "dry":        args.dry,
        "note":       "API inference K=1 temperature=0 verbalized-confidence p_yes.",
    }

    # Load corpus
    df = pd.read_csv(INPUT_CSV).reset_index(drop=True)
    print(f"[corpus] {CORPUS}: n={len(df)}", flush=True)

    run_model_mhs(client, tag, df, inter_call_sleep=args.sleep, dry=args.dry)

    env_log["utc_end"] = datetime.now(timezone.utc).isoformat()
    env_path = f"{OUT}/e3_mhs_env_{tag}.json"
    with open(env_path, "w") as f:
        json.dump(env_log, f, indent=2)
    print(f"[env_log] saved -> {env_path}", flush=True)


if __name__ == "__main__":
    main()
