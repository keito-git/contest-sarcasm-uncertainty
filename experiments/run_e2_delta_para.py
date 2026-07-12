"""
E2 — delta_para (literal-vs-intended paraphrase distance) predicts LLM error (H2).

delta_para is an automated, label-free discrepancy score:
  for each item, an LLM writes (a) a LITERAL paraphrase (words at face value,
  ignoring sarcasm) and (b) an INTENDED paraphrase (what the speaker means in
  context). delta_para = 1 - cos( emb(literal), emb(intended) ).
  Large delta_para = big literal-vs-intended gap = high discrepancy (the sarcasm
  signal that the dual-branch discrepancy model (DBDA) is meant to capture).

H2 test (per E1 arm, merged by item_id on the SAME N=400 sample):
  - Spearman(delta_para, LLM_error) controlling for human disagreement
    -> does delta predict WHICH items the LLM gets wrong, beyond shared difficulty?
  - Spearman(delta_para, human_disagreement)  [sanity: pilot found ~0]
    -> if delta predicts LLM error but NOT human disagreement, delta captures
       LLM-specific (epistemic) difficulty = the thesis.

Paraphrase generation: OpenRouter cheap model (fast). Embeddings: OpenAI
text-embedding-3-small. Keys read from .env file (values never printed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import run_e1 as e1  # reuse load_dataset / DATASETS / load_keys

BASE = e1.BASE
OUT = f"{BASE}/results/llm_e2"
SEED = 42
PARA_MODEL = "anthropic/claude-haiku-4.5"   # OpenRouter, paraphrase generation
EMB_MODEL = "text-embedding-3-small"        # OpenAI


def para_messages(ctx: str, txt: str, concept: str) -> list[dict]:
    ctx = (ctx or "").strip()[:1500]
    txt = (txt or "").strip()[:1500]
    sysm = ("You rewrite messages. You will be given a context and a message. "
            "Produce two short paraphrases of the MESSAGE.")
    user = (
        (f"Context:\n{ctx}\n\n" if ctx else "")
        + f"Message:\n{txt}\n\n"
        + "Write TWO one-sentence paraphrases of the Message:\n"
        + f"1) LITERAL: what the words say at face value, IGNORING any {concept} "
        + "intent (take it completely straight).\n"
        + f"2) INTENDED: what the speaker actually MEANS here, accounting for "
        + f"any {concept} intent given the context.\n"
        + 'Reply with strict JSON only: {"literal": "...", "intended": "..."}'
    )
    return [{"role": "system", "content": sysm}, {"role": "user", "content": user}]


def gen_paraphrases(client, ctx, txt, concept) -> dict:
    kw = dict(model=PARA_MODEL, messages=para_messages(ctx, txt, concept),
              max_tokens=250, temperature=0)
    try:
        r = client.chat.completions.create(**kw)
    except Exception:
        kw.pop("temperature", None)
        r = client.chat.completions.create(**kw)
    text = r.choices[0].message.content or ""
    lit = intd = None
    try:
        s = text[text.find("{"): text.rfind("}") + 1]
        d = json.loads(s)
        lit, intd = str(d.get("literal", "")), str(d.get("intended", ""))
    except Exception:
        pass
    u = getattr(r, "usage", None)
    return {"literal": lit, "intended": intd, "raw": text,
            "in_tok": getattr(u, "prompt_tokens", 0) if u else 0,
            "out_tok": getattr(u, "completion_tokens", 0) if u else 0}


def embed_batch(client, texts: list[str]) -> np.ndarray:
    vecs = []
    B = 100
    for i in range(0, len(texts), B):
        chunk = [t if t and t.strip() else " " for t in texts[i:i + B]]
        r = client.embeddings.create(model=EMB_MODEL, input=chunk)
        vecs.extend([d.embedding for d in r.data])
        print(f"    embed {i+len(chunk)}/{len(texts)}", flush=True)
    return np.asarray(vecs, dtype=float)


def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return np.sum(An * Bn, axis=1)


def partial_spearman(x, y, z) -> float:
    x, y, z = map(lambda a: rankdata(np.asarray(a, float)), (x, y, z))

    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    return float(spearmanr(resid(x, z), resid(y, z)).statistic)


def boot_sp(x, y, nb=2000, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan"), len(x)
    rng = np.random.default_rng(seed)
    rr = np.array([spearmanr(x[i], y[i]).statistic
                   for i in (rng.integers(0, len(x), len(x)) for _ in range(nb))])
    return (float(spearmanr(x, y).statistic),
            float(np.nanpercentile(rr, 2.5)), float(np.nanpercentile(rr, 97.5)), len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--phase", default="1")
    ap.add_argument("--datasets", default="CSC,MultiPICo")
    ap.add_argument("--e1_phase", default="1")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    e1.load_keys()
    import openai
    oc = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])          # embeddings
    orc = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1")       # paraphrase

    summ, cost = [], [0, 0]
    for name in [d for d in args.datasets.split(",") if d]:
        d = e1.DATASETS[name]
        df = e1.load_dataset(name, args.n)  # same sample as E1 (seed=42)
        print(f"[{name}] n={len(df)} — generating paraphrases...", flush=True)
        recs = []
        t0 = time.time()
        for i, row in df.iterrows():
            ctx = row[d["ctx"]] if d["ctx"] in df else ""
            res = e1.with_retry(gen_paraphrases, orc, ctx, row[d["txt"]], d["concept"])
            recs.append(res)
            cost[0] += res.get("in_tok", 0)
            cost[1] += res.get("out_tok", 0)
            if (i + 1) % 50 == 0:
                print(f"    para {i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
        P = pd.DataFrame(recs)
        df = df.reset_index(drop=True)
        df["literal"] = P["literal"].values
        df["intended"] = P["intended"].values

        print(f"[{name}] embedding literal+intended...", flush=True)
        el = embed_batch(oc, df["literal"].fillna("").tolist())
        ei = embed_batch(oc, df["intended"].fillna("").tolist())
        df["delta_para"] = 1.0 - cosine_rows(el, ei)

        df[["item_id", "y_true", "dis_mi", "delta_para", "literal", "intended"]].to_parquet(
            f"{OUT}/e2_dpara_{name}_phase{args.phase}.parquet")

        # sanity: delta_para vs human disagreement (pilot expected ~0)
        s_dis = boot_sp(df["delta_para"], df["dis_mi"])

        # merge with each E1 arm's error/uncertainty
        for prov in ["gpt", "claude"]:
            f1 = f"{BASE}/results/llm_e1/e1_{name}_{prov}_phase{args.e1_phase}.parquet"
            if not os.path.exists(f1):
                continue
            e = pd.read_parquet(f1)[["item_id", "pred", "y_true", "conf"]].dropna(subset=["pred"])
            m = df.merge(e, on="item_id", how="inner", suffixes=("", "_e1"))
            if len(m) < 20:
                continue
            err = (m["pred"].astype(int) != m["y_true"].astype(int)).astype(int).to_numpy()
            dp = m["delta_para"].to_numpy(float)
            dis = m["dis_mi"].to_numpy(float)
            conf = np.where(np.isnan(m["conf"].to_numpy(float)), 50.0, m["conf"].to_numpy(float))
            u = 1.0 - conf / 100.0

            s_err = boot_sp(dp, err)
            s_err_partial = partial_spearman(dp, err, dis)
            s_u = boot_sp(dp, u)
            # error rate by delta_para tercile
            tp = np.digitize(rankdata(dp) / (len(dp) + 1), [1 / 3, 2 / 3])
            err_by_t = {f"t{t}": float(err[tp == t].mean()) for t in [0, 1, 2] if (tp == t).any()}
            summ.append({
                "dataset": name, "provider": prov, "n": int(len(m)),
                "rho_delta_error": {"rho": s_err[0], "ci": [s_err[1], s_err[2]]},
                "partial_delta_error_ctrl_humandis": round(s_err_partial, 4),
                "rho_delta_Uverbal": {"rho": s_u[0], "ci": [s_u[1], s_u[2]]},
                "rho_delta_humandis": {"rho": s_dis[0], "ci": [s_dis[1], s_dis[2]]},
                "err_by_delta_tercile": err_by_t,
                "delta_para_mean": float(np.nanmean(dp)),
            })
            print(f"  [{name}/{prov}] rho(delta,err)={s_err[0]:+.3f} "
                  f"partial(ctrl dis)={s_err_partial:+.3f} "
                  f"rho(delta,humandis)={s_dis[0]:+.3f} n={len(m)}", flush=True)

    px_in, px_out = 1.0, 5.0  # haiku per 1M (approx)
    usd = cost[0] / 1e6 * px_in + cost[1] / 1e6 * px_out
    from datetime import datetime, timezone
    env = {"utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
           "n": args.n, "para_model": PARA_MODEL, "emb_model": EMB_MODEL,
           "para_in_tok": cost[0], "para_out_tok": cost[1], "para_usd": round(usd, 4)}
    with open(f"{OUT}/e2_summary_phase{args.phase}.json", "w") as f:
        json.dump({"env": env, "results": summ}, f, indent=2)

    L = [f"# E2 — delta_para -> LLM error (H2)  phase {args.phase}", "",
         f"- {env['utc']}  N={args.n}  para={PARA_MODEL}  emb={EMB_MODEL}",
         f"- paraphrase cost ~${env['para_usd']} (embeddings negligible)", "",
         "delta_para = 1 - cos(emb(literal paraphrase), emb(intended paraphrase)).", ""]
    for a in summ:
        re_ = a["rho_delta_error"]; ru = a["rho_delta_Uverbal"]; rd = a["rho_delta_humandis"]
        L += [f"### {a['dataset']} / {a['provider']}  (n={a['n']})",
              f"- rho(delta_para, LLM_error)={re_['rho']:+.3f} "
              f"[{re_['ci'][0]:+.3f},{re_['ci'][1]:+.3f}]  "
              f"| partial ctrl human_dis={a['partial_delta_error_ctrl_humandis']:+.3f}",
              f"- rho(delta_para, U_verbal)={ru['rho']:+.3f} "
              f"[{ru['ci'][0]:+.3f},{ru['ci'][1]:+.3f}]",
              f"- rho(delta_para, human_dis)={rd['rho']:+.3f} "
              f"[{rd['ci'][0]:+.3f},{rd['ci'][1]:+.3f}]  (sanity: pilot ~0)",
              f"- err by delta tercile: {a['err_by_delta_tercile']}", ""]
    L += ["## Reading",
          "- H2 support = rho(delta_para, LLM_error) > 0 AND survives controlling",
          "  for human disagreement, WHILE rho(delta_para, human_dis) ~ 0.",
          "  => delta captures LLM-specific (epistemic) difficulty, not human ambiguity."]
    with open(f"{OUT}/e2_summary_phase{args.phase}.md", "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
