"""
run_r1_gpu.py -- KBS R1 revision GPU runs (KNOSYS-D-26-23467).

Reuses the exact inference loop, prompt, parser, and decoding settings of
run_gpu_sweep_regen_v3.py (HF transformers, K=10, temperature 1.0, top_p 0.95,
max_new_tokens 30). Three run types:

  --mode seed    : Qwen2.5 replicate runs with a different inference seed
                   (answers Reviewer 1: run-to-run variance across seeds).
                   Output prefix: c7seed{SEED}
  --mode family  : OLMo-2 Instruct capacity sweep (1B, 7B, 13B, 32B), a second,
                   fully open model family (Reviewers 1 and 2).
                   Output prefix: fam
  --mode native  : Qwen2.5 on the four non-English MultiPICo subcorpora with the
                   prompt translated into the item language (Reviewer 1).
                   Output prefix: native

Run from the directory that holds run_gpu_sweep_regen_v3.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["seed", "family", "native"], required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--corpora", default="CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it")
ap.add_argument("--device", default="0")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--delete_after", action=argparse.BooleanOptionalAction, default=False)
ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items only")
args = ap.parse_args()

# GPU rule: mask devices before torch is imported anywhere
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import pandas as pd  # noqa: E402

import run_gpu_sweep_regen_v3 as base  # noqa: E402

FAMILY = {
    "o1b": ("allenai/OLMo-2-0425-1B-Instruct", 1.0),
    "o7b": ("allenai/OLMo-2-1124-7B-Instruct", 7.0),
    "o13b": ("allenai/OLMo-2-1124-13B-Instruct", 13.0),
    "o32b": ("allenai/OLMo-2-0325-32B-Instruct", 32.0),
}
QWEN = {t: (m, c) for t, (m, c, _) in base.C7_MODELS.items()}

# ---------------------------------------------------------------------------
# Native-language prompts (same content and answer format as the English prompt)
# ---------------------------------------------------------------------------
NATIVE = {
    "es": {"concept": "irónico",
           "system": "Eres un anotador experto que juzga si un mensaje es {c}. "
                     "Juzga el significado percibido que tomaría un lector típico. Sé conciso.",
           "ctx": "Contexto", "msg": "Mensaje",
           "instr": "Pregunta: ¿Es {c} el siguiente mensaje?\n"
                    "Responde en UNA línea con EXACTAMENTE este formato: `<Sí|No> <confianza>` "
                    "donde <confianza> es un número entero de 0 a 100. Ejemplo: `Sí 80`. "
                    "No escribas nada más."},
    "de": {"concept": "ironisch",
           "system": "Du bist ein erfahrener Annotator und beurteilst, ob eine Nachricht {c} ist. "
                     "Beurteile die wahrgenommene Bedeutung, die ein typischer Leser annehmen würde. "
                     "Antworte knapp.",
           "ctx": "Kontext", "msg": "Nachricht",
           "instr": "Frage: Ist die folgende Nachricht {c}?\n"
                    "Antworte in EINER Zeile in GENAU diesem Format: `<Ja|Nein> <Konfidenz>`, "
                    "wobei <Konfidenz> eine ganze Zahl von 0 bis 100 ist. Beispiel: `Ja 80`. "
                    "Gib nichts anderes aus."},
    "fr": {"concept": "ironique",
           "system": "Tu es un annotateur expert qui juge si un message est {c}. "
                     "Juge le sens perçu qu'un lecteur typique comprendrait. Sois concis.",
           "ctx": "Contexte", "msg": "Message",
           "instr": "Question : le message suivant est-il {c} ?\n"
                    "Réponds sur UNE seule ligne EXACTEMENT dans ce format : `<Oui|Non> <confiance>` "
                    "où <confiance> est un nombre entier de 0 à 100. Exemple : `Oui 80`. "
                    "N'écris rien d'autre."},
    "it": {"concept": "ironico",
           "system": "Sei un annotatore esperto che valuta se un messaggio è {c}. "
                     "Valuta il significato percepito che un lettore tipico coglierebbe. Sii conciso.",
           "ctx": "Contesto", "msg": "Messaggio",
           "instr": "Domanda: il seguente messaggio è {c}?\n"
                    "Rispondi su UNA sola riga ESATTAMENTE in questo formato: `<Sì|No> <confidenza>` "
                    "dove <confidenza> è un numero intero da 0 a 100. Esempio: `Sì 80`. "
                    "Non scrivere nient'altro."},
}
_YES = {"yes", "sí", "si", "sì", "ja", "oui"}
_NO = {"no", "nein", "non"}
_ANS = re.compile(r"(?<!\w)(yes|no|sí|sì|si|ja|nein|oui|non)(?!\w)", re.I)
_CONF = re.compile(r"(\d{1,3})")


def make_native(lang: str):
    P = NATIVE[lang]

    def build_messages(concept: str, ctx: str, txt: str) -> list[dict]:
        c = P["concept"]
        ctx = (ctx or "").strip()[:1500]
        txt = (txt or "").strip()[:1500]
        content = ""
        if ctx:
            content += f"{P['ctx']}:\n{ctx}\n\n"
        content += f"{P['msg']}:\n{txt}\n\n" + P["instr"].format(c=c)
        return [{"role": "system", "content": P["system"].format(c=c)},
                {"role": "user", "content": content}]

    def parse_response(text: str):
        if not text or not text.strip():
            return None, 0.5, 0.5
        m = _ANS.search(text)
        if not m:
            return None, 0.5, 0.5
        w = m.group(1).lower()
        label = 1 if w in _YES else 0
        cm = _CONF.search(text[m.end():]) or _CONF.search(text)
        conf = max(0.0, min(100.0, float(int(cm.group(1))) if cm else 50.0))
        return label, (1.0 if label else 0.0), conf / 100.0

    return build_messages, parse_response


def main() -> None:
    if args.mode == "family":
        model_id, cap = FAMILY[args.tag]
        prefix = "fam"
        seed = 42
    else:
        model_id, cap = QWEN[args.tag]
        seed = args.seed if args.mode == "seed" else 42
        prefix = f"c7seed{seed}" if args.mode == "seed" else "native"
    corpora = [c for c in args.corpora.split(",") if c]
    if args.mode == "native":
        corpora = [c for c in corpora if c.startswith("MultiPICo_")]

    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = {"script": "run_r1_gpu.py", "mode": args.mode, "tag": args.tag, "model_id": model_id,
           "cap_B": cap, "seed": seed, "K": 10, "temperature": 1.0, "top_p": 0.95,
           "utc_start": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "corpora": corpora,
           "limit": args.limit, "nvidia_smi_before": base.nvidia_smi_log("before")}
    for corpus in corpora:
        out = base.OUT_DIR / f"{prefix}_{args.tag}_{corpus}.csv"
        if out.exists():
            print(f"[skip] {out.name} exists", flush=True)
            continue
        df = pd.read_csv(base.IN_DIR / f"full_input_{corpus}.csv")
        if args.limit:
            df = df.head(args.limit)
        if args.mode == "native":
            bm, pr = make_native(corpus.split("_", 1)[1])
            base.build_messages, base.parse_response = bm, pr
        ckpt = base.OUT_DIR / f"_ckpt_{prefix}_{args.tag}_{corpus}.csv"
        res = base.run_inference_hf(model_id=model_id, df=df, K=10, temperature=1.0,
                                    max_new_tokens=30, batch_size=args.batch_size,
                                    checkpoint_path=ckpt, checkpoint_every=200,
                                    inference_seed=seed, trust_remote_code=False)
        res.to_csv(out, index=False)
        if ckpt.exists():
            ckpt.unlink()
        env[f"{corpus}_n"] = len(res)
        env[f"{corpus}_mean_n_valid"] = float(res["n_valid"].mean())
        print(f"[saved] {out.name} n={len(res)} mean_pyes={res['p_yes'].mean():.3f} "
              f"mean_nvalid={res['n_valid'].mean():.2f}", flush=True)
    env["utc_end"] = datetime.now(timezone.utc).isoformat()
    env["nvidia_smi_after"] = base.nvidia_smi_log("after")
    json.dump(env, open(base.OUT_DIR / f"env_{prefix}_{args.tag}.json", "w"), indent=2)
    if args.delete_after:
        base.delete_model_cache_full(model_id)


if __name__ == "__main__":
    main()
