"""
E1 — LLM item-level uncertainty vs human disagreement (H1: epistemic misattribution).

Central claim (from pilot 03/04): sarcasm/irony is LOW-aleatoric for humans
(delta does NOT predict human disagreement, rho~0.02). If LLMs are nonetheless
uncertain/wrong on human-AGREED sarcasm items, that uncertainty cannot be the
humans' genuine ambiguity (aleatoric) -> it is the model's own ignorance
(epistemic), misattributed as aleatoric.

Providers / uncertainty signals:
  - GPT via native OpenAI  : token logprobs (p_yes) + verbalized confidence.
    Anthropic exposes NO logprobs, so GPT is the quantitative-calibration arm.
  - Claude via OpenRouter  : verbalized confidence only (contrast arm).

Human disagreement is MEAN-INDEPENDENT (Asai gate, learned from pilot confound):
  - CSC       : rel_var = Var / ((mean-1)(6-mean))   [1..6 rating scale]
  - MultiPICo : normalized entropy of p_ironic (already in `disagreement` col)

Keys are read from a .env file in the repository root into os.environ WITHOUT printing values.
CPU only for data; the only network calls are the two chat APIs.
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

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = str(Path(__file__).parent.parent)
OUT = f"{BASE}/results/llm_e1"
SEED = 42

GPT_MODEL = "gpt-4o-mini"                     # OpenAI direct, logprobs
CLAUDE_MODEL = "anthropic/claude-haiku-4.5"   # OpenRouter, verbalized only

DATASETS = {
    "CSC": {
        "path": f"{BASE}/results/sarcasm/csc_aggregated.parquet",
        "ctx": "context_text", "txt": "response_text",
        "concept": "sarcastic",
    },
    "MultiPICo": {
        "path": f"{BASE}/results/sarcasm/multipico_aggregated.parquet",
        "ctx": "post", "txt": "reply",
        "concept": "ironic",
    },
    "EPIC": {
        "path": f"{BASE}/results/sarcasm/epic_aggregated.parquet",
        "ctx": "post", "txt": "reply",
        "concept": "ironic",
    },
}


# ----------------------------------------------------------------------------- keys
def load_keys() -> None:
    envf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    for line in open(envf):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


# ----------------------------------------------------------------------------- prompt
def build_messages(ctx: str, txt: str, concept: str) -> list[dict]:
    ctx = (ctx or "").strip()[:1500]
    txt = (txt or "").strip()[:1500]
    sys_msg = (
        "You are an expert annotator judging whether a message is "
        f"{concept}. Judge the perceived meaning a typical reader would take."
    )
    user = (
        (f"Context:\n{ctx}\n\n" if ctx else "")
        + f"Message:\n{txt}\n\n"
        + f"Question: Is the Message {concept}?\n"
        + "Answer on ONE line in EXACTLY this format: `<Yes|No> <confidence>` "
        + "where <confidence> is an integer 0-100 for how confident you are in "
        + "your Yes/No label. Example: `Yes 80`. Output nothing else."
    )
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": user}]


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
    # take the confidence integer AFTER the yes/no token if possible
    tail = s[m.end():] if m else s
    c = _CONF.search(tail) or _CONF.search(s)
    if c:
        v = int(c.group(1))
        if 0 <= v <= 100:
            conf = float(v)
    return label, conf


def p_yes_from_logprobs(choice) -> float | None:
    """Probability mass on Yes vs No at the first content token."""
    lp = getattr(choice, "logprobs", None)
    if not lp or not getattr(lp, "content", None):
        return None
    # find the first token that looks like yes/no; scan first few
    for tok in lp.content[:3]:
        cands = list(getattr(tok, "top_logprobs", []) or [])
        # include the realized token itself
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


# ----------------------------------------------------------------------------- API
def call_openai(client, ctx, txt, concept):
    r = client.chat.completions.create(
        model=GPT_MODEL,
        messages=build_messages(ctx, txt, concept),
        temperature=0, max_tokens=12,
        logprobs=True, top_logprobs=20,
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


def call_openrouter(client, ctx, txt, concept):
    kw = dict(model=CLAUDE_MODEL, messages=build_messages(ctx, txt, concept),
              max_tokens=16, temperature=0)
    try:
        r = client.chat.completions.create(**kw)
    except Exception:  # frontier/reasoning models reject temperature
        kw.pop("temperature", None)
        kw["max_tokens"] = 4000  # reasoning models need room
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


def call_openrouter_as_gpt(client, ctx, txt, concept):
    """GPT-family model routed via OpenRouter, verbalized-only (frontier check)."""
    kw = dict(model=GPT_MODEL, messages=build_messages(ctx, txt, concept),
              max_tokens=16, temperature=0)
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
    return {"raw": text, "label": label, "conf": conf, "p_yes_lp": None,
            "in_tok": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "out_tok": getattr(usage, "completion_tokens", 0) if usage else 0}


def with_retry(fn, *a, tries=4):
    for i in range(tries):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                return {"error": repr(e)[:200], "label": None, "conf": None,
                        "p_yes_lp": None, "in_tok": 0, "out_tok": 0, "raw": ""}
            time.sleep(2 * (i + 1))


# ----------------------------------------------------------------------------- data
def load_dataset(name: str, n: int) -> pd.DataFrame:
    d = DATASETS[name]
    df = pd.read_parquet(d["path"]).copy()
    df = df.dropna(subset=[d["txt"]])
    df = df[df[d["txt"]].astype(str).str.len() > 0]

    if name == "CSC":
        me = df["mean_eval"].to_numpy(float)
        var = df["std_eval"].to_numpy(float) ** 2
        denom = np.clip((me - 1.0) * (6.0 - me), 0.5, None)
        df["dis_mi"] = np.clip(var / denom, 0.0, 1.0)
        df["y_true"] = (df["mean_eval_norm"].to_numpy(float) >= 0.5).astype(int)
    else:  # MultiPICo — disagreement col is normalized entropy already
        df["dis_mi"] = np.clip(df["disagreement"].to_numpy(float), 0.0, 1.0)
        df["y_true"] = (df["p_ironic"].to_numpy(float) >= 0.5).astype(int)

    df = df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- metrics
def ece(p_pos: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p_pos = np.clip(p_pos, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p_pos >= lo) & (p_pos < hi if i < bins - 1 else p_pos <= hi)
        if m.sum() == 0:
            continue
        conf = p_pos[m].mean()
        acc = y[m].mean()
        e += (m.sum() / n) * abs(acc - conf)
    return float(e)


def brier(p_pos: np.ndarray, y: np.ndarray) -> float:
    p_pos = np.clip(p_pos, 0, 1)
    return float(np.mean((p_pos - y) ** 2))


def boot_spearman(x, y, nb=2000, seed=SEED):
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
        rr[b] = spearmanr(x[idx], y[idx]).statistic
    return float(base), float(np.nanpercentile(rr, 2.5)), float(np.nanpercentile(rr, 97.5)), len(x)


def tercile(a: np.ndarray) -> np.ndarray:
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])  # 0,1,2


# ----------------------------------------------------------------------------- run
def run_provider(name, df, provider, client):
    d = DATASETS[name]
    rows = []
    caller = call_openai if provider == "gpt" else call_openrouter
    t0 = time.time()
    for i, row in df.iterrows():
        res = with_retry(caller, client, row[d["ctx"]] if d["ctx"] in df else "",
                         row[d["txt"]], d["concept"])
        rows.append(res)
        if (i + 1) % 25 == 0:
            print(f"    [{name}/{provider}] {i+1}/{len(df)}  ({time.time()-t0:.0f}s)",
                  flush=True)
    R = pd.DataFrame(rows)
    out = df[["item_id", "y_true", "dis_mi"]].reset_index(drop=True).copy()
    out["provider"] = provider
    out["pred"] = R["label"].values
    out["conf"] = R["conf"].values
    out["p_yes_lp"] = R["p_yes_lp"].values
    out["raw"] = R["raw"].values
    out["in_tok"] = R["in_tok"].values
    out["out_tok"] = R["out_tok"].values
    if "error" in R:
        out["error_api"] = R["error"].values
    return out


def analyse(name, provider, out: pd.DataFrame) -> dict:
    df = out.dropna(subset=["pred"]).copy()
    n = len(df)
    if n < 5:
        return {"dataset": name, "provider": provider, "n_ok": n, "note": "too few parsed"}
    y = df["y_true"].to_numpy(int)
    pred = df["pred"].to_numpy(int)
    err = (pred != y).astype(int)
    dis = df["dis_mi"].to_numpy(float)

    # verbalized uncertainty (both providers)
    conf = df["conf"].to_numpy(float)
    conf = np.where(np.isnan(conf), 50.0, conf)
    u_verbal = 1.0 - conf / 100.0
    # p_pos from verbalized (for calibration): yes->conf, no->1-conf
    p_pos_verbal = np.where(pred == 1, conf / 100.0, 1.0 - conf / 100.0)

    res = {"dataset": name, "provider": provider, "n_ok": int(n),
           "acc": float((pred == y).mean()), "err_rate": float(err.mean()),
           "base_rate_pos": float(y.mean())}

    # correlations U vs human disagreement (mean-independent)
    rv = boot_spearman(u_verbal, dis)
    res["rho_Uverbal_dis"] = {"rho": rv[0], "ci": [rv[1], rv[2]], "n": rv[3]}
    res["ece_verbal"] = ece(p_pos_verbal, y)
    res["brier_verbal"] = brier(p_pos_verbal, y)

    if provider == "gpt" and df["p_yes_lp"].notna().sum() >= 5:
        pl = df["p_yes_lp"].to_numpy(float)
        okl = ~np.isnan(pl)
        u_lp = 1.0 - np.abs(2 * pl - 1.0)   # 0 (certain) .. 1 (p=.5)
        rl = boot_spearman(u_lp[okl], dis[okl])
        res["rho_Ulogprob_dis"] = {"rho": rl[0], "ci": [rl[1], rl[2]], "n": rl[3]}
        res["ece_logprob"] = ece(pl[okl], y[okl])
        res["brier_logprob"] = brier(pl[okl], y[okl])

    # ---- H1 core: within LOW human-disagreement (humans agree), is LLM still uncertain/wrong?
    tt = tercile(dis)
    for lab, tg in [("low_dis", 0), ("mid_dis", 1), ("high_dis", 2)]:
        m = tt == tg
        if m.sum() == 0:
            continue
        res[f"{lab}_err"] = float(err[m].mean())
        res[f"{lab}_Uverbal"] = float(u_verbal[m].mean())
        res[f"{lab}_n"] = int(m.sum())

    # epistemic-misattribution cluster: low disagreement AND high U_llm
    uT = tercile(u_verbal)
    cluster = (tt == 0) & (uT == 2)
    res["epistemic_cluster_frac"] = float(cluster.mean())      # low-dis & high-U among all
    res["epistemic_cluster_n"] = int(cluster.sum())
    res["epistemic_cluster_err"] = float(err[cluster].mean()) if cluster.sum() else float("nan")
    return res


def main():
    global GPT_MODEL, CLAUDE_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--phase", default="0")
    ap.add_argument("--datasets", default="CSC,MultiPICo")
    ap.add_argument("--providers", default="gpt,claude")
    ap.add_argument("--gpt_model", default=GPT_MODEL)
    ap.add_argument("--claude_model", default=CLAUDE_MODEL)
    ap.add_argument("--gpt_client", default="openai", choices=["openai", "openrouter"])
    args = ap.parse_args()
    GPT_MODEL, CLAUDE_MODEL = args.gpt_model, args.claude_model

    os.makedirs(OUT, exist_ok=True)
    load_keys()
    import openai
    oc = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    orc = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1")
    # gpt provider can be routed via OpenRouter (verbalized-only) for frontier models
    clients = {"gpt": oc if args.gpt_client == "openai" else orc, "claude": orc}
    if args.gpt_client == "openrouter":
        globals()["call_openai"] = call_openrouter_as_gpt

    ds_names = [d for d in args.datasets.split(",") if d]
    provs = [p for p in args.providers.split(",") if p]

    all_out, summ, cost = [], [], {"gpt": [0, 0], "claude": [0, 0]}
    for name in ds_names:
        df = load_dataset(name, args.n)
        print(f"[{name}] n={len(df)} loaded "
              f"(base_rate_pos={df['y_true'].mean():.2f}, "
              f"dis_mi mean={df['dis_mi'].mean():.3f})", flush=True)
        for prov in provs:
            print(f"  -> {prov} ({GPT_MODEL if prov=='gpt' else CLAUDE_MODEL})", flush=True)
            out = run_provider(name, df, prov, clients[prov])
            out.to_parquet(f"{OUT}/e1_{name}_{prov}_phase{args.phase}.parquet")
            cost[prov][0] += int(out["in_tok"].sum())
            cost[prov][1] += int(out["out_tok"].sum())
            a = analyse(name, prov, out)
            summ.append(a)
            all_out.append(out)
            print("     parsed:", out["pred"].notna().sum(), "/", len(out),
                  "| acc=", round(a.get("acc", float('nan')), 3),
                  "| err=", round(a.get("err_rate", float('nan')), 3), flush=True)

    # cost ($ per 1M): gpt-4o-mini 0.15/0.60 ; claude-haiku-4.5 ~1.0/5.0
    px = {"gpt": (0.15, 0.60), "claude": (1.0, 5.0)}
    total = 0.0
    cost_report = {}
    for p in provs:
        ci, co = cost[p]
        usd = ci / 1e6 * px[p][0] + co / 1e6 * px[p][1]
        total += usd
        cost_report[p] = {"in_tok": ci, "out_tok": co, "usd": round(usd, 4)}

    from datetime import datetime, timezone
    env_log = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase, "n_per_dataset": args.n,
        "gpt_model": GPT_MODEL, "claude_model": CLAUDE_MODEL,
        "seed": SEED, "cost": cost_report, "total_usd": round(total, 4),
    }
    with open(f"{OUT}/e1_summary_phase{args.phase}.json", "w") as f:
        json.dump({"env": env_log, "results": summ}, f, indent=2)

    # markdown
    cost_str = ", ".join(f"{k}=${v['usd']}" for k, v in cost_report.items())
    L = [f"# E1 — LLM uncertainty vs human disagreement (phase {args.phase})", "",
         f"- generated: {env_log['utc']}",
         f"- N/dataset: {args.n}  seed: {SEED}",
         f"- GPT (logprob arm): `{GPT_MODEL}`  |  Claude (verbalized arm): `{CLAUDE_MODEL}`",
         f"- **cost this run: ${env_log['total_usd']}** ({cost_str})",
         "",
         "## Per (dataset, provider)", ""]
    for a in summ:
        L.append(f"### {a['dataset']} / {a['provider']}  (n_ok={a.get('n_ok')})")
        if "acc" not in a:
            L.append(f"- {a.get('note','')}"); L.append(""); continue
        L.append(f"- acc={a['acc']:.3f}  err={a['err_rate']:.3f}  "
                 f"base_rate_pos={a['base_rate_pos']:.3f}")
        rv = a["rho_Uverbal_dis"]
        L.append(f"- rho(U_verbal, human_dis)={rv['rho']:+.3f} "
                 f"[{rv['ci'][0]:+.3f},{rv['ci'][1]:+.3f}]  "
                 f"ECE_verbal={a['ece_verbal']:.3f}  Brier_verbal={a['brier_verbal']:.3f}")
        if "rho_Ulogprob_dis" in a:
            rl = a["rho_Ulogprob_dis"]
            L.append(f"- rho(U_logprob, human_dis)={rl['rho']:+.3f} "
                     f"[{rl['ci'][0]:+.3f},{rl['ci'][1]:+.3f}]  "
                     f"ECE_logprob={a['ece_logprob']:.3f}  Brier_logprob={a['brier_logprob']:.3f}")
        L.append(f"- **H1 (err by human-agreement tercile)**: "
                 f"low_dis err={a.get('low_dis_err',float('nan')):.3f} "
                 f"(n={a.get('low_dis_n','?')}) | "
                 f"mid={a.get('mid_dis_err',float('nan')):.3f} | "
                 f"high={a.get('high_dis_err',float('nan')):.3f}")
        L.append(f"- epistemic cluster (low human-dis & high U): "
                 f"frac={a['epistemic_cluster_frac']:.3f} "
                 f"n={a['epistemic_cluster_n']} err={a['epistemic_cluster_err']:.3f}")
        L.append("")
    L += ["## Reading",
          "- **H1 support** = LLM error rate stays high in the LOW human-disagreement",
          "  tercile (humans agree) — the LLM is wrong where humans are not split,",
          "  so that uncertainty is epistemic, not aleatoric.",
          "- rho(U, human_dis) near 0 = LLM uncertainty is DECOUPLED from human",
          "  disagreement (expected under epistemic misattribution)."]
    with open(f"{OUT}/e1_summary_phase{args.phase}.md", "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[run_llm_e1] wrote {OUT}/e1_summary_phase{args.phase}.md")


if __name__ == "__main__":
    sys.exit(main())
