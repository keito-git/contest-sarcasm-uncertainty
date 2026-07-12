"""
E1 + E2 — Generalisation to hate-speech detection (MHS, Kennedy et al. 2022).

Task: binary hate-speech classification (hate vs. not-hate).
Dataset: Measuring Hate Speech corpus (MHS).
  - Only items with >= 3 annotators are used (n_items available = 17,352).
  - N = 400 sampled at seed 42.

Disagreement measure (dis_mi):
  Normalised binary entropy of p_hate = fraction of annotators with hatespeech==2.
  dis_mi = H(p_hate) / log(2) in [0, 1].
  0 = full human consensus; 1 = maximally split (p=0.5).
  NOTE: with 3-5 annotators the distribution is discrete-bimodal
  (64 % items dis_mi=0, 36 % dis_mi>0), equivalent to a binary
  "all-agree" vs "any-disagree" split.  This is explicitly noted in the results.

Models run (temperature 0, seed 42):
  - gpt-4o-mini   : OpenAI direct  (logprobs available)
  - gpt-4.1       : OpenRouter     (verbalized confidence only)
  - claude-haiku-4.5  : OpenRouter (verbalized confidence only)
  - claude-opus-4.8   : OpenRouter (verbalized confidence only)

E1 (epistemic misattribution):
  Error rate on low-dis tercile (humans all agree) — should stay high
  if LLM errors are epistemic rather than aleatoric.

E2 (single-signal failure):
  Spearman rho(U_verbal, dis_mi) — expected weak (< 0.30) if
  LLM confidence fails to track human ambiguity.

Keys: read from .env file in script directory (values never printed externally).
Output: results/llm_hate/
"""
from __future__ import annotations
from pathlib import Path

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# ---- paths -------------------------------------------------------------------
BASE = str(Path(__file__).parent.parent)
RAW_MHS = f"{BASE}/data/raw/measuring_hate_speech.parquet"
OUT = f"{BASE}/results/llm_hate"
E3_OUT = f"{BASE}/results/llm_e3"
SEED = 42
N_SAMPLE = 400
DATASET_NAME = "MHS"
CONCEPT = "hate speech"

# ---- models ------------------------------------------------------------------
MODEL_CONFIGS: list[dict] = [
    {
        "key": "gpt_mini",
        "display": "gpt-4o-mini",
        "model_id": "gpt-4o-mini",
        "client": "openai",   # logprobs available
    },
    {
        "key": "gpt41",
        "display": "gpt-4.1",
        "model_id": "openai/gpt-4.1",
        "client": "openrouter",
    },
    {
        "key": "haiku",
        "display": "claude-haiku-4.5",
        "model_id": "anthropic/claude-haiku-4.5",
        "client": "openrouter",
    },
    {
        "key": "opus",
        "display": "claude-opus-4.8",
        "model_id": "anthropic/claude-opus-4.8",
        "client": "openrouter",
    },
]

# ---- key loading -------------------------------------------------------------

def load_keys() -> None:
    """Read .env file into os.environ (values never printed)."""
    envf = os.path.join(os.path.dirname(__file__), ".env")
    with open(envf) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


# ---- data preparation --------------------------------------------------------

def build_mhs_dataset(n: int, seed: int) -> pd.DataFrame:
    """
    Aggregate MHS raw annotations to per-item stats, compute dis_mi,
    sample n items.

    dis_mi = normalised binary entropy of p_hate (fraction of annotators
    labelling hatespeech==2).
    y_true = (p_hate >= 0.5).
    """
    raw = pd.read_parquet(RAW_MHS)
    n_ann = raw.groupby("comment_id").size()
    valid_ids = n_ann[n_ann >= 3].index

    agg = (
        raw[raw["comment_id"].isin(valid_ids)]
        .groupby("comment_id")
        .agg(
            text=("text", "first"),
            p_hate=("hatespeech", lambda x: (x == 2).mean()),
            n_annotators=("hatespeech", "count"),
        )
        .reset_index()
    )

    # normalised binary entropy
    p = np.clip(agg["p_hate"].values, 1e-9, 1.0 - 1e-9)
    h = (-p * np.log(p) - (1.0 - p) * np.log(1.0 - p)) / math.log(2)
    h[agg["p_hate"].values == 0.0] = 0.0
    h[agg["p_hate"].values == 1.0] = 0.0
    agg["dis_mi"] = np.clip(h, 0.0, 1.0)

    agg["y_true"] = (agg["p_hate"] >= 0.5).astype(int)
    agg["item_id"] = agg["comment_id"].astype(str)
    agg = agg.dropna(subset=["text"])
    agg = agg[agg["text"].astype(str).str.len() > 0]

    df = agg.sample(n=min(n, len(agg)), random_state=seed).reset_index(drop=True)
    print(
        f"[MHS] n={len(df)}  base_rate_pos={df['y_true'].mean():.2f}  "
        f"dis_mi_mean={df['dis_mi'].mean():.3f}  "
        f"dis_mi==0: {(df['dis_mi']==0).sum()}/{len(df)} "
        f"dis_mi>0: {(df['dis_mi']>0).sum()}/{len(df)}",
        flush=True,
    )
    return df


# ---- prompt ------------------------------------------------------------------

def build_messages(txt: str) -> list[dict]:
    sys_msg = (
        "You are an expert content moderator. Determine whether a social-media "
        "post constitutes hate speech targeting a person or group based on "
        "protected characteristics (race, religion, gender, sexual orientation, "
        "nationality, disability, etc.). Focus on the intended message a typical "
        "reader would perceive."
    )
    user = (
        f"Post:\n{txt[:1500]}\n\n"
        "Question: Is this post hate speech?\n"
        "Answer on ONE line in EXACTLY this format: `<Yes|No> <confidence>` "
        "where <confidence> is an integer 0-100 representing how confident you "
        "are in your Yes/No answer.  Example: `Yes 80`.  Output nothing else."
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user},
    ]


_ANS = re.compile(r"\b(yes|no)\b", re.I)
_CONF = re.compile(r"(\d{1,3})")


def parse_text(s: str) -> tuple[int | None, float | None]:
    """Return (label 1/0/None, confidence 0-100 or None)."""
    if not s:
        return None, None
    m = _ANS.search(s)
    label = None
    if m:
        label = 1 if m.group(1).lower() == "yes" else 0
    conf = None
    tail = s[m.end():] if m else s
    c = _CONF.search(tail) or _CONF.search(s)
    if c:
        v = int(c.group(1))
        if 0 <= v <= 100:
            conf = float(v)
    return label, conf


def p_yes_from_logprobs(choice) -> float | None:
    """Probability mass on Yes vs No at the first content token (OpenAI only)."""
    lp = getattr(choice, "logprobs", None)
    if not lp or not getattr(lp, "content", None):
        return None
    for tok in lp.content[:3]:
        cands = list(getattr(tok, "top_logprobs", []) or [])
        yes_mass = no_mass = 0.0
        for c in cands:
            t = c.token.strip().lower()
            p = math.exp(c.logprob)
            if t.startswith("yes") or t == "y":
                yes_mass += p
            elif t.startswith("no") or t == "n":
                no_mass += p
        if yes_mass + no_mass > 1e-6:
            return yes_mass / (yes_mass + no_mass)
    return None


# ---- API calls ---------------------------------------------------------------

def call_openai(client, model_id: str, txt: str) -> dict:
    r = client.chat.completions.create(
        model=model_id,
        messages=build_messages(txt),
        temperature=0,
        max_tokens=12,
        logprobs=True,
        top_logprobs=20,
        seed=SEED,
    )
    ch = r.choices[0]
    text = ch.message.content or ""
    label, conf = parse_text(text)
    p_yes = p_yes_from_logprobs(ch)
    usage = r.usage
    return {
        "raw": text, "label": label, "conf": conf, "p_yes_lp": p_yes,
        "in_tok": usage.prompt_tokens, "out_tok": usage.completion_tokens,
    }


def call_openrouter(client, model_id: str, txt: str) -> dict:
    kw = dict(
        model=model_id,
        messages=build_messages(txt),
        max_tokens=16,
        temperature=0,
    )
    try:
        r = client.chat.completions.create(**kw)
    except Exception:
        kw.pop("temperature", None)
        kw["max_tokens"] = 4000
        r = client.chat.completions.create(**kw)
    ch = r.choices[0]
    text = ch.message.content or ""
    label, conf = parse_text(text)
    usage = getattr(r, "usage", None)
    return {
        "raw": text, "label": label, "conf": conf, "p_yes_lp": None,
        "in_tok": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "out_tok": getattr(usage, "completion_tokens", 0) if usage else 0,
    }


def with_retry(fn, *args, tries: int = 4):
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                return {
                    "error": repr(e)[:200], "label": None, "conf": None,
                    "p_yes_lp": None, "in_tok": 0, "out_tok": 0, "raw": "",
                }
            time.sleep(2 * (i + 1))


# ---- run one model -----------------------------------------------------------

def _build_out(df_base: pd.DataFrame, rows: list[dict], cfg: dict) -> pd.DataFrame:
    """Build result DataFrame from accumulated rows."""
    R = pd.DataFrame(rows)
    n = len(rows)
    out = df_base.iloc[:n].copy()
    out["model"] = cfg["key"]
    out["model_display"] = cfg["display"]
    out["pred"] = R["label"].values
    out["conf"] = R["conf"].values
    out["p_yes_lp"] = R["p_yes_lp"].values
    out["raw"] = R["raw"].values
    out["in_tok"] = R["in_tok"].values
    out["out_tok"] = R["out_tok"].values
    if "error" in R.columns:
        out["error_api"] = R["error"].values
    return out


def run_model(df: pd.DataFrame, cfg: dict, oc, orc,
              checkpoint_path: str | None = None) -> pd.DataFrame:
    """Run one model over df; return per-item result DataFrame.

    If checkpoint_path is set, saves partial results every 50 items
    so they survive a process timeout.
    """
    caller = call_openai if cfg["client"] == "openai" else call_openrouter
    client = oc if cfg["client"] == "openai" else orc
    model_id = cfg["model_id"]
    rows: list[dict] = []
    base = df[["item_id", "y_true", "dis_mi"]].reset_index(drop=True)
    t0 = time.time()
    for i, row in df.iterrows():
        res = with_retry(caller, client, model_id, row["text"])
        rows.append(res)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"    [{cfg['display']}] {i+1}/{len(df)} ({elapsed:.0f}s)",
                flush=True,
            )
            if checkpoint_path:
                _build_out(base, rows, cfg).to_parquet(checkpoint_path)
    return _build_out(base, rows, cfg)


# ---- metrics -----------------------------------------------------------------

def ece(p_pos: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p_pos = np.clip(p_pos, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p_pos >= lo) & (p_pos <= hi if i == bins - 1 else p_pos < hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(p_pos[m].mean() - y[m].mean())
    return float(e)


def brier(p_pos: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.clip(p_pos, 0, 1) - y) ** 2))


def balanced_acc(pred: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy: mean of per-class recall."""
    classes = np.unique(y)
    recalls = []
    for c in classes:
        mask = y == c
        if mask.sum() == 0:
            continue
        recalls.append(float((pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def boot_spearman(x, y, nb: int = 2000, seed: int = SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan"), len(x)
    rng = np.random.default_rng(seed)
    base = spearmanr(x, y).statistic
    rr = np.empty(nb)
    for b in range(nb):
        idx = rng.integers(0, len(x), len(x))
        try:
            rr[b] = spearmanr(x[idx], y[idx]).statistic
        except Exception:
            rr[b] = float("nan")
    return (
        float(base),
        float(np.nanpercentile(rr, 2.5)),
        float(np.nanpercentile(rr, 97.5)),
        len(x),
    )


def tercile(a: np.ndarray) -> np.ndarray:
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])  # 0, 1, 2


# ---- analyse one model result ------------------------------------------------

def analyse(out: pd.DataFrame) -> dict:
    df = out.dropna(subset=["pred"]).copy()
    n = len(df)
    model_key = df["model"].iloc[0] if len(df) else "?"
    model_disp = df["model_display"].iloc[0] if len(df) else "?"

    if n < 5:
        return {"model": model_key, "model_display": model_disp, "n_ok": n,
                "note": "too few parsed"}

    y = df["y_true"].to_numpy(int)
    pred = df["pred"].to_numpy(int)
    err = (pred != y).astype(int)
    dis = df["dis_mi"].to_numpy(float)

    conf = df["conf"].to_numpy(float)
    conf = np.where(np.isnan(conf), 50.0, conf)
    u_verbal = 1.0 - conf / 100.0
    p_pos_verbal = np.where(pred == 1, conf / 100.0, 1.0 - conf / 100.0)

    res: dict = {
        "model": model_key,
        "model_display": model_disp,
        "n_ok": int(n),
        "acc": float((pred == y).mean()),
        "bal_acc": balanced_acc(pred, y),
        "err_rate": float(err.mean()),
        "base_rate_pos": float(y.mean()),
        "ece_verbal": ece(p_pos_verbal, y),
        "brier_verbal": brier(p_pos_verbal, y),
    }

    # E2: Spearman rho(U_verbal, dis_mi)
    rv = boot_spearman(u_verbal, dis)
    res["rho_Uverbal_dis"] = {"rho": rv[0], "ci": [rv[1], rv[2]], "n": rv[3]}

    # GPT logprob arm
    if df["p_yes_lp"].notna().sum() >= 5:
        pl = df["p_yes_lp"].to_numpy(float)
        ok_l = ~np.isnan(pl)
        u_lp = 1.0 - np.abs(2 * pl - 1.0)
        rl = boot_spearman(u_lp[ok_l], dis[ok_l])
        res["rho_Ulogprob_dis"] = {"rho": rl[0], "ci": [rl[1], rl[2]], "n": rl[3]}
        res["ece_logprob"] = ece(pl[ok_l], y[ok_l])
        res["brier_logprob"] = brier(pl[ok_l], y[ok_l])

    # E1: err by human-agreement tercile
    tt = tercile(dis)
    for lab, tg in [("low_dis", 0), ("mid_dis", 1), ("high_dis", 2)]:
        m = tt == tg
        if m.sum() == 0:
            continue
        res[f"{lab}_err"] = float(err[m].mean())
        res[f"{lab}_bal_acc"] = balanced_acc(pred[m], y[m])
        res[f"{lab}_Uverbal"] = float(u_verbal[m].mean())
        res[f"{lab}_n"] = int(m.sum())

    # Binary split: dis_mi==0 (all-agree) vs dis_mi>0 (any-disagree)
    m0 = dis == 0.0
    m1 = dis > 0.0
    if m0.sum() > 0:
        res["agree_err"] = float(err[m0].mean())
        res["agree_bal_acc"] = balanced_acc(pred[m0], y[m0])
        res["agree_n"] = int(m0.sum())
    if m1.sum() > 0:
        res["disagree_err"] = float(err[m1].mean())
        res["disagree_bal_acc"] = balanced_acc(pred[m1], y[m1])
        res["disagree_n"] = int(m1.sum())

    # epistemic cluster: low-dis (tercile 0) AND high-U (tercile 2)
    uT = tercile(u_verbal)
    cluster = (tt == 0) & (uT == 2)
    res["epistemic_cluster_frac"] = float(cluster.mean())
    res["epistemic_cluster_n"] = int(cluster.sum())
    res["epistemic_cluster_err"] = float(err[cluster].mean()) if cluster.sum() else float("nan")

    return res


# ---- cost helpers ------------------------------------------------------------

PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt_mini":   (0.15, 0.60),
    "gpt41":      (2.00, 8.00),
    "haiku":      (1.00, 5.00),
    "opus":       (15.0, 75.0),
}


# ---- reporting ---------------------------------------------------------------

def write_summary(summ: list[dict], env_log: dict, phase: str) -> None:
    with open(f"{OUT}/e1_hate_summary_{phase}.json", "w") as f:
        json.dump({"env": env_log, "results": summ}, f, indent=2)

    L = [
        f"# E1+E2 — Hate Speech (MHS) generalisation (phase {phase})",
        "",
        f"- generated: {env_log['utc']}",
        f"- N/dataset: {env_log['n_sample']}  seed: {SEED}",
        f"- dataset: MHS (Kennedy et al. 2022) — items with >=3 annotators",
        f"- dis_mi: normalised binary entropy of p_hate = fraction(annotators→hate)",
        f"  NOTE: dis distribution is bimodal (64% dis=0 / 36% dis>0) due to",
        f"  few annotators per item (median 3).  Low-dis tercile ≈ all-agree group.",
        "",
        "## Per model",
        "",
    ]
    for a in summ:
        L.append(f"### {a['model_display']}  (n_ok={a.get('n_ok')})")
        if "acc" not in a:
            L.append(f"- {a.get('note', '')}"); L.append(""); continue
        L.append(
            f"- acc={a['acc']:.3f}  bal_acc={a['bal_acc']:.3f}  "
            f"err={a['err_rate']:.3f}  base_rate_pos={a['base_rate_pos']:.3f}"
        )
        rv = a["rho_Uverbal_dis"]
        L.append(
            f"- **E2** rho(U_verbal, dis_mi)={rv['rho']:+.3f} "
            f"[{rv['ci'][0]:+.3f},{rv['ci'][1]:+.3f}]  "
            f"ECE_verbal={a['ece_verbal']:.3f}  Brier_verbal={a['brier_verbal']:.3f}"
        )
        if "rho_Ulogprob_dis" in a:
            rl = a["rho_Ulogprob_dis"]
            L.append(
                f"- **E2** rho(U_logprob, dis_mi)={rl['rho']:+.3f} "
                f"[{rl['ci'][0]:+.3f},{rl['ci'][1]:+.3f}]  "
                f"ECE_logprob={a.get('ece_logprob', float('nan')):.3f}  "
                f"Brier_logprob={a.get('brier_logprob', float('nan')):.3f}"
            )
        L.append(
            f"- **E1 tercile** low_dis err={a.get('low_dis_err', float('nan')):.3f} "
            f"(n={a.get('low_dis_n','?')}) | "
            f"mid={a.get('mid_dis_err', float('nan')):.3f} | "
            f"high={a.get('high_dis_err', float('nan')):.3f}"
        )
        L.append(
            f"- **E1 binary** agree err={a.get('agree_err', float('nan')):.3f} "
            f"(n={a.get('agree_n','?')}) | "
            f"disagree err={a.get('disagree_err', float('nan')):.3f} "
            f"(n={a.get('disagree_n','?')})"
        )
        L.append(
            f"- epistemic cluster (low human-dis & high U_verbal): "
            f"frac={a['epistemic_cluster_frac']:.3f} "
            f"n={a['epistemic_cluster_n']} "
            f"err={a['epistemic_cluster_err']:.3f}"
        )
        L.append("")

    L += [
        "## Interpretation key",
        "- **E1 support** = LLM error stays high in LOW human-disagreement group",
        "  (humans all agree, dis_mi=0) -> errors are epistemic, not aleatoric.",
        "- **E2 support** = rho(U_verbal, dis_mi) < 0.30 -> single verbalized signal",
        "  fails to track human ambiguity.",
        "",
        "## Consistency with sarcasm corpora (phasefrontier400)",
        "  CSC/gpt:        low_dis err=0.119  rho=+0.123",
        "  CSC/claude:     low_dis err=0.170  rho=+0.142",
        "  MultiPICo/gpt:  low_dis err=0.109  rho=+0.261",
        "  MultiPICo/claude: low_dis err=0.080  rho=+0.290",
    ]

    with open(f"{OUT}/e1_hate_summary_{phase}.md", "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[run_llm_hate] wrote {OUT}/e1_hate_summary_{phase}.md")


# ---- E3 input CSV ------------------------------------------------------------

def write_e3_input(df: pd.DataFrame) -> None:
    """Write E3 input CSV for Qwen capacity sweep (write E3 input CSV for capacity sweep)."""
    e3 = pd.DataFrame({
        "item_id": df["item_id"].values,
        "dataset": DATASET_NAME,
        "concept": CONCEPT,
        "ctx": "",          # MHS has no separate context field
        "txt": df["text"].values,
        "dis_mi": df["dis_mi"].values,
        "y_true": df["y_true"].values,
    })
    out_path = f"{E3_OUT}/e3_input_{DATASET_NAME}.csv"
    e3.to_csv(out_path, index=False)
    print(f"[E3] wrote {out_path}  shape={e3.shape}")


# ---- env log -----------------------------------------------------------------

def build_env_log(n_sample: int, phase: str, cost_report: dict) -> dict:
    import platform
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "n_sample": n_sample,
        "seed": SEED,
        "dataset": DATASET_NAME,
        "concept": CONCEPT,
        "models": [c["display"] for c in MODEL_CONFIGS],
        "hostname": platform.node(),
        "python": sys.version,
        "cost": cost_report,
    }


# ---- summarize from saved parquets -------------------------------------------

def summarize_from_disk(phase: str) -> None:
    """Load all per-model parquets for the given phase and write combined summary."""
    summ: list[dict] = []
    cost_report: dict[str, dict] = {}
    found: list[str] = []
    for cfg in MODEL_CONFIGS:
        pq_path = f"{OUT}/e1_hate_{cfg['key']}_phase{phase}.parquet"
        if not os.path.exists(pq_path):
            print(f"[summarize] MISSING: {pq_path}", flush=True)
            continue
        out = pd.read_parquet(pq_path)
        if "model" not in out.columns:
            out["model"] = cfg["key"]
        if "model_display" not in out.columns:
            out["model_display"] = cfg["display"]
        a = analyse(out)
        summ.append(a)
        in_tok = int(out.get("in_tok", pd.Series([0])).sum())
        out_tok = int(out.get("out_tok", pd.Series([0])).sum())
        px_in, px_out = PRICE_PER_1M.get(cfg["key"], (1.0, 5.0))
        usd = in_tok / 1e6 * px_in + out_tok / 1e6 * px_out
        cost_report[cfg["key"]] = {
            "model_display": cfg["display"],
            "in_tok": in_tok, "out_tok": out_tok, "usd": round(usd, 4),
        }
        found.append(cfg["display"])
        print(f"[summarize] loaded {pq_path}  n={len(out)}", flush=True)
    if not summ:
        print("[summarize] No parquets found. Run models first."); return
    total_usd = sum(v["usd"] for v in cost_report.values())
    env_log = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase, "n_sample": N_SAMPLE, "seed": SEED,
        "dataset": DATASET_NAME, "concept": CONCEPT,
        "models_loaded": found, "cost": cost_report, "total_usd": round(total_usd, 4),
    }
    write_summary(summ, env_log, phase)


# ---- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--phase", default="hate1")
    ap.add_argument("--dry", action="store_true",
                    help="Run 5 items only for pipeline check")
    ap.add_argument("--models", default="",
                    help="Comma-sep subset of model keys to run (blank = all)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip models whose parquet already exists for this phase")
    ap.add_argument("--summarize_only", action="store_true",
                    help="Skip all API calls; aggregate saved parquets into summary")
    args = ap.parse_args()

    n = 5 if args.dry else args.n
    phase = (args.phase + "_dry") if args.dry else args.phase

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(E3_OUT, exist_ok=True)

    if args.summarize_only:
        summarize_from_disk(phase)
        return

    load_keys()

    import openai
    oc = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    orc = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    df = build_mhs_dataset(n, SEED)

    # Write E3 input (from full 400-item sample, not dry-run slice)
    if not args.dry:
        write_e3_input(df)

    # filter model subset if requested
    model_filter = set(args.models.split(",")) if args.models else set()

    summ: list[dict] = []
    cost_report: dict[str, dict] = {}
    all_out_frames: list[pd.DataFrame] = []

    for cfg in MODEL_CONFIGS:
        if model_filter and cfg["key"] not in model_filter:
            continue
        pq_path = f"{OUT}/e1_hate_{cfg['key']}_phase{phase}.parquet"
        if args.skip_existing and os.path.exists(pq_path):
            print(f"[skip] {cfg['display']} parquet already exists -> {pq_path}",
                  flush=True)
            out = pd.read_parquet(pq_path)
            a = analyse(out)
            summ.append(a)
            in_tok = int(out.get("in_tok", pd.Series([0])).sum())
            out_tok = int(out.get("out_tok", pd.Series([0])).sum())
            px_in, px_out = PRICE_PER_1M.get(cfg["key"], (1.0, 5.0))
            usd = in_tok / 1e6 * px_in + out_tok / 1e6 * px_out
            cost_report[cfg["key"]] = {
                "model_display": cfg["display"], "in_tok": in_tok,
                "out_tok": out_tok, "usd": round(usd, 4),
            }
            continue
        print(f"\n[{DATASET_NAME}] running {cfg['display']} (client={cfg['client']})...",
              flush=True)
        # Use checkpoint_path so partial results survive a timeout
        ckpt = pq_path.replace(".parquet", "_ckpt.parquet")
        out = run_model(df, cfg, oc, orc, checkpoint_path=ckpt)
        out.to_parquet(pq_path)
        # remove checkpoint once full save is done
        if os.path.exists(ckpt):
            os.remove(ckpt)
        print(f"  saved -> {pq_path}")

        # cost
        in_tok = int(out["in_tok"].sum())
        out_tok = int(out["out_tok"].sum())
        px_in, px_out = PRICE_PER_1M.get(cfg["key"], (1.0, 5.0))
        usd = in_tok / 1e6 * px_in + out_tok / 1e6 * px_out
        cost_report[cfg["key"]] = {
            "model_display": cfg["display"],
            "in_tok": in_tok, "out_tok": out_tok,
            "usd": round(usd, 4),
        }

        a = analyse(out)
        summ.append(a)
        all_out_frames.append(out)
        print(
            f"  parsed: {out['pred'].notna().sum()}/{len(out)} "
            f"acc={a.get('acc', float('nan')):.3f} "
            f"err={a.get('err_rate', float('nan')):.3f} "
            f"low_dis_err={a.get('low_dis_err', float('nan')):.3f} "
            f"rho_E2={a.get('rho_Uverbal_dis', {}).get('rho', float('nan')):+.3f}",
            flush=True,
        )

    total_usd = sum(v["usd"] for v in cost_report.values())
    env_log = build_env_log(n, phase, cost_report)
    env_log["total_usd"] = round(total_usd, 4)

    with open(f"{OUT}/env_log_{phase}.json", "w") as f:
        json.dump(env_log, f, indent=2)

    write_summary(summ, env_log, phase)


if __name__ == "__main__":
    main()
