"""
run_gpu_sweep.py — GPU K=10 temperature=1.0 sweep for C7 and C8.

Covers
------
C7  Qwen2.5 6-size sweep (0.5B – 32B) on FULL corpus
    Tags: q0p5b q1p5b q3b q7b q14b q32b
    Input: results/revision_gpu_2026-09/full_input_{corpus}.csv
    Output: results/revision_gpu_2026-09/c7_{tag}_{corpus}.csv

C8  Same-capacity different-weights probe (~0.5–3B tier; matched to capacity pair)
    Tags (gated models substituted by ungated models of similar scale):
      ll3b   (Phi-3-mini-4k-instruct 3.8B)   — matched to Qwen2.5-3B; was Llama-3.2-3B (gated)
      ll1b   (SmolLM2-1.7B-Instruct 1.7B)    — matched to Qwen2.5-1.5B; was Llama-3.2-1B (gated)
      gm2_2b (stablelm-zephyr-3b 3B)         — matched to Qwen2.5-1.5B alt; was gemma-2-2b (gated)
    Note: Qwen2.5-3B (q3b) and Qwen2.5-1.5B (q1p5b) are shared from C7.
    Output: results/revision_gpu_2026-09/c8_{tag}_{corpus}.csv

C8-noise  Sampling noise control (same model, different seed)
    Tag: q0p5b_s2 — Qwen2.5-0.5B-Instruct re-run with seed=1234
    Output: results/revision_gpu_2026-09/c8_noise_q0p5b_s2_{corpus}.csv
    Use to estimate |p_run1 - p_run2| = sampling variance floor.

C8-ensemble (optional)
    Tag: ens30_q7b — Qwen2.5-7B K=30 passes for ensemble variance baseline
    Output: results/revision_gpu_2026-09/c8_ens30_q7b_{corpus}.csv

Corpora
----------------------------
  - MultiPICo-en is DROPPED (80 % overlap with EPIC; see rc10_corpus_overlap.json).
  - Non-English MultiPICo sub-corpora added:
      MultiPICo_es (n≈4681), MultiPICo_de (n≈2370),
      MultiPICo_fr (n≈1760), MultiPICo_it (n≈998)
  - CSC (n=7036) and EPIC (n=2999) unchanged.

Design constraints
------------------
  - K=10, temperature=1.0  (exact replication of original Qwen2.5 sweep)
  - Same prompt/parsing as run_e3_xfamily_api.py::build_messages()
  - CUDA_VISIBLE_DEVICES set from --device arg BEFORE torch import
  - batch_size default 16 (reduce if OOM)
  - Models deleted from disk after each run (--delete_after default True)
  - Checkpoint: partial results written every --checkpoint_every items
  - Output CSV schema: item_id, p_yes, p_conf_mean, n_valid, H, dis_mi, y_true
  - Original 400-item results in results/llm_e3/ are never touched

Container (RTX PRO 6000 Blackwell, 96 GB VRAM, 4 CPU cores, 8 GB RAM cgroup)
-----------------------------------------------------------------------------
  CUDA 12.8  /  sm_120 (Blackwell).  Requires torch built for CUDA 12.8;
  if container comes with an older wheel, run:
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
  or use nvcr.io/nvidia/pytorch:25.xx-py3 which ships torch+CUDA 12.8.

  vLLM MAY FAIL at 8 GB RAM (cgroup): vLLM's CPU workers use ~2–4 GB RAM at
  startup, and its Triton JIT compilation can exceed the remaining headroom.
  This script uses HF transformers + device_map="auto" as the primary backend,
  which typically needs only ~300 MB RAM for loading logic.  If RAM OOMs, add
  --batch_size 1 and set TRANSFORMERS_OFFLINE=1 after the first model download.

GPU rules (shared server)
  - Set CUDA_VISIBLE_DEVICES BEFORE any torch import (done in main())
  - Never hardcode cuda:0 / cuda:1; use device_map='auto'
  - nvidia-smi check before and after each model
  - Do NOT kill other processes on the server

Usage (from the repository root; see also run_all_v2.sh)
--------------
  # Full C7 sweep:
  CORPORA="CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it"
  for TAG in q0p5b q1p5b q3b q7b q14b q32b; do
    python kbs/gpu_sweep/run_gpu_sweep_regen.py \\
        --mode c7 --tag $TAG --corpora $CORPORA --device 0
  done

  # C8 diversity models (matched ~3B / ~1B scale):
  for TAG in ll3b ll1b gm2_2b; do
    python kbs/gpu_sweep/run_gpu_sweep_regen.py \\
        --mode c8 --tag $TAG --corpora $CORPORA --device 0
  done

  # C8 sampling noise control:
  python kbs/gpu_sweep/run_gpu_sweep_regen.py \\
      --mode c8_noise --tag q0p5b_s2 --corpora $CORPORA --device 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────── #
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR  # noqa: E402  (repo-root path config)
BASE   = WORK_DIR
IN_DIR = BASE / "results" / "revision_gpu_2026-09"
OUT_DIR = IN_DIR / "server_outputs" / "regen"  # read by kbs/analysis/*
SEED   = 42

# ── Model registry ───────────────────────────────────────────────────────── #
# tag -> (HuggingFace model_id, capacity_B)
C7_MODELS: dict[str, tuple[str, float]] = {
    "q0p5b": ("Qwen/Qwen2.5-0.5B-Instruct",  0.5),
    "q1p5b": ("Qwen/Qwen2.5-1.5B-Instruct",  1.5),
    "q3b":   ("Qwen/Qwen2.5-3B-Instruct",    3.0),
    "q7b":   ("Qwen/Qwen2.5-7B-Instruct",    7.0),
    "q14b":  ("Qwen/Qwen2.5-14B-Instruct",  14.0),
    "q32b":  ("Qwen/Qwen2.5-32B-Instruct",  32.0),
}

C8_MODELS: dict[str, tuple[str, float]] = {
    # NOTE: the originally planned gated models (Llama-3.2, gemma-2-2b) were not accessible
    # (server has no HF token; 401 GatedRepoError). Substituted with ungated models
    # of similar scale and different architecture family.
    # Tags UNCHANGED so analyze_c7c8.py can find the files.
    "ll3b":   ("microsoft/Phi-3-mini-4k-instruct",       3.8),  # ~3B, Microsoft; was Llama-3.2-3B
    "ll1b":   ("HuggingFaceTB/SmolLM2-1.7B-Instruct",   1.7),  # ~1.5B, HuggingFace; was Llama-3.2-1B
    "gm2_2b": ("stabilityai/stablelm-zephyr-3b",         3.0),  # ~3B, StabilityAI; was gemma-2-2b
}

# C8 noise control: same model as q0p5b, different seed (K=10, T=1.0)
# stored under mode="c8_noise", tag="q0p5b_s2"
C8_NOISE_MODELS: dict[str, tuple[str, float]] = {
    "q0p5b_s2": ("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
}
C8_NOISE_SEED = 1234  # different from SEED=42 used in C7

C8_ENS_MODELS: dict[str, tuple[str, float]] = {
    "ens30_q7b": ("Qwen/Qwen2.5-7B-Instruct", 7.0),
}

ALL_MODELS = {**C7_MODELS, **C8_MODELS, **C8_NOISE_MODELS, **C8_ENS_MODELS}

# Default corpora: MultiPICo-en dropped; non-English sub-corpora added.
CORPORA = ["CSC", "EPIC",
           "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]


# ── Prompt (exact copy from original sweep) ─────────────────────────────── #
def build_prompt_text(concept: str, ctx: str, txt: str) -> str:
    """Returns the full user-turn text (no system message for tokenizer use)."""
    ctx = (ctx or "").strip()[:1500]
    txt = (txt or "").strip()[:1500]
    instruction = (
        f"Question: Is the following message {concept}?\n"
        "Answer on ONE line in EXACTLY this format: `<Yes|No> <confidence>` "
        "where <confidence> is an integer 0-100. Example: `Yes 80`. "
        "Output nothing else."
    )
    content = ""
    if ctx:
        content += f"Context:\n{ctx}\n\n"
    content += f"Message:\n{txt}\n\n" + instruction
    return content


def build_messages(concept: str, ctx: str, txt: str) -> list[dict]:
    sys_msg = (
        "You are an expert annotator judging whether a message is "
        f"{concept}. Judge the perceived meaning a typical reader would take. "
        "Be concise."
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": build_prompt_text(concept, ctx, txt)},
    ]


# ── Response parsing (exact copy from original sweep) ───────────────────── #
_YES_NO = re.compile(r"\b(yes|no)\b", re.I)
_CONF   = re.compile(r"(\d{1,3})")


def parse_response(text: str) -> tuple[Optional[int], float, float]:
    """Parse model output. Return (label 1/0/None, p_yes_binary ∈ {0,1}, conf_raw 0-1).

    p_yes_binary: 1.0 if Yes, 0.0 if No (used for K-sampling aggregate p_yes).
    conf_raw: verbalized confidence / 100 (0-1 scale, direction-agnostic).
              For p_conf_mean: use conf_raw if Yes, 1-conf_raw if No.
    """
    if not text or not text.strip():
        return None, 0.5, 0.5
    m = _YES_NO.search(text)
    if not m:
        return None, 0.5, 0.5
    label = 1 if m.group(1).lower() == "yes" else 0
    tail  = text[m.end():]
    cm    = _CONF.search(tail) or _CONF.search(text)
    conf  = float(int(cm.group(1))) if cm else 50.0
    conf  = max(0.0, min(100.0, conf))
    # For K-sampling, we convert to binary: just return the label as p
    # parse_response here returns label_prob NOT verbalized_conf because
    # in K=10 mode we aggregate by majority.  p_yes from a single sample
    # is either 1.0 (Yes) or 0.0 (No); the aggregator averages K such values.
    p_yes = 1.0 if label == 1 else 0.0
    return label, p_yes, conf / 100.0


def binary_entropy(p: float) -> float:
    """Binary entropy in nats (same as original sweep)."""
    p = max(1e-9, min(1 - 1e-9, p))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


# ── GPU inference ────────────────────────────────────────────────────────── #
def run_inference_hf(
    model_id: str,
    df: pd.DataFrame,
    K: int = 10,
    temperature: float = 1.0,
    max_new_tokens: int = 30,
    batch_size: int = 16,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 200,
    inference_seed: int = 42,
) -> "pd.DataFrame":
    """
    Run K-sample inference on df using HuggingFace transformers.

    GPU rule: CUDA_VISIBLE_DEVICES must be set BEFORE this is called (done in
    main() before 'import torch'). Never references cuda:0 / cuda:1 directly;
    uses device_map='auto' which maps layers to available GPU automatically.

    Container note (RTX PRO 6000 Blackwell, CUDA 12.8 / sm_120):
    - Requires torch compiled for CUDA 12.8. If the wheel is older, install:
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    - BF16 supported natively on Blackwell.
    - vLLM may OOM with 8 GB RAM cgroup; this function avoids vLLM entirely.

    inference_seed: sets torch seed before generate() for reproducibility.
                    Use a different value for c8_noise runs (second independent pass).

    Returns DataFrame with columns:
        item_id, p_yes, n_valid, H, dis_mi, y_true
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  [hf] loading {model_id} (seed={inference_seed}) ...", flush=True)
    # Use BF16 on GPU (saves memory on Blackwell/Ada); FP32 on CPU fallback
    _dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "left"  # required for decoder-only batched generation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=_dtype,          # transformers >= 4.46: 'dtype' replaces 'torch_dtype'
        device_map="auto",     # maps to GPU automatically; never hardcodes cuda:0/1
        trust_remote_code=True,
    )
    model.eval()

    # Set seed for this run (different between c7 and c8_noise)
    torch.manual_seed(inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed)

    # Load checkpoint if resuming
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if checkpoint_path and checkpoint_path.exists():
        import pandas as _pd
        ckpt = _pd.read_csv(checkpoint_path)
        done_ids = set(ckpt["item_id"].astype(str))
        existing_rows = ckpt.to_dict(orient="records")
        print(f"  [resume] {len(done_ids)} items already done", flush=True)

    rows: list[dict] = list(existing_rows)
    concept = str(df["concept"].iloc[0])
    ctx_col = "ctx" if "ctx" in df.columns else None
    n_total = len(df)
    t0 = time.time()

    # Process in batches
    todo = df[~df["item_id"].astype(str).isin(done_ids)].copy()
    print(f"  [hf] {len(todo)} items to run (K={K}, T={temperature})", flush=True)

    for batch_start in range(0, len(todo), batch_size):
        batch = todo.iloc[batch_start : batch_start + batch_size]
        batch_prompts: list[list[dict]] = []
        for _, row in batch.iterrows():
            ctx_val = str(row[ctx_col]) if ctx_col else ""
            msgs = build_messages(concept, ctx_val, str(row["txt"]))
            batch_prompts.append(msgs)

        # Tokenize with chat template
        inputs_text = [
            tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batch_prompts
        ]
        enc = tok(inputs_text, return_tensors="pt", padding=True,
                  truncation=True, max_length=1500).to(model.device)

        # Generate K samples per item
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=K,
                pad_token_id=tok.eos_token_id,
            )
        # gen_ids shape: (batch_size * K, seq_len)
        input_len = enc["input_ids"].shape[1]
        # Decode only new tokens
        gen_new = gen_ids[:, input_len:]
        texts = tok.batch_decode(gen_new, skip_special_tokens=True)
        # texts is a flat list of length batch_size * K
        # Re-group: texts[i*K : (i+1)*K] for item i

        for idx_in_batch, (_, row) in enumerate(batch.iterrows()):
            item_texts = texts[idx_in_batch * K : (idx_in_batch + 1) * K]
            labels: list[int] = []
            conf_p_yes_list: list[float] = []
            for txt_out in item_texts:
                lbl, _, conf_raw = parse_response(txt_out)
                if lbl is not None:
                    labels.append(lbl)
                    # Verbalized p_yes: conf if Yes, 1-conf if No
                    conf_p_yes_list.append(conf_raw if lbl == 1 else 1.0 - conf_raw)

            n_valid = len(labels)
            p_yes = float(sum(labels) / max(n_valid, 1))
            H = binary_entropy(p_yes) if n_valid > 0 else float("nan")
            p_conf_mean = (float(sum(conf_p_yes_list) / len(conf_p_yes_list))
                           if conf_p_yes_list else float("nan"))

            rows.append({
                "item_id":      row["item_id"],
                "p_yes":        round(p_yes, 6),
                "p_conf_mean":  round(p_conf_mean, 6),
                "n_valid":      n_valid,
                "H":            round(H, 10),
                "dis_mi":       float(row["dis_mi"]),
                "y_true":       int(row["y_true"]),
            })

        # Checkpoint
        if checkpoint_path and (batch_start + batch_size) % checkpoint_every == 0:
            import pandas as _pd
            _pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

        if (batch_start + batch_size) % (batch_size * 10) == 0 or \
                batch_start + batch_size >= len(todo):
            elapsed = time.time() - t0
            print(f"  [{batch_start + batch_size}/{len(todo)}] "
                  f"elapsed={elapsed:.0f}s", flush=True)

    import pandas as _pd
    result = _pd.DataFrame(rows)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ── nvidia-smi check ─────────────────────────────────────────────────────── #
def nvidia_smi_log(label: str) -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
        print(f"  [nvidia-smi {label}] {out}", flush=True)
        return out
    except Exception:
        return "nvidia-smi unavailable"


# ── Main ─────────────────────────────────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    choices=["c7", "c8", "c8_ens", "c8_noise"], default="c7")
    ap.add_argument("--tag",     required=True, help="model tag (see model registry)")
    ap.add_argument("--corpora",
                    default="CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it")
    ap.add_argument("--device",  default="0",
                    help="CUDA_VISIBLE_DEVICES value; use 'cpu' to force CPU")
    ap.add_argument("--K",        type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--checkpoint_every", type=int, default=200)
    ap.add_argument("--delete_after", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="delete model weights from HF cache after run (save disk)")
    ap.add_argument("--dry_run",  action="store_true",
                    help="skip actual inference (for pipeline testing)")
    args = ap.parse_args()

    # GPU rule: set CUDA_VISIBLE_DEVICES BEFORE any torch import
    if args.device.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Now safe to import torch
    import pandas as pd

    # Verify model tag
    registry = {**C7_MODELS, **C8_MODELS, **C8_NOISE_MODELS, **C8_ENS_MODELS}
    if args.tag not in registry:
        print(f"[ERROR] unknown tag '{args.tag}'. Available: {list(registry.keys())}")
        sys.exit(1)
    model_id, cap_B = registry[args.tag]

    # Sampling noise mode uses a different seed to produce a second independent run
    inference_seed = C8_NOISE_SEED if args.mode == "c8_noise" else SEED

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.mode  # "c7", "c8", "c8_ens", or "c8_noise"

    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]

    # Env log
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    env: dict = {
        "script":      "kbs/gpu_sweep/run_gpu_sweep_regen.py",
        "mode":        args.mode,
        "tag":         args.tag,
        "model_id":    model_id,
        "cap_B":       cap_B,
        "K":           args.K,
        "temperature": args.temperature,
        "batch_size":  args.batch_size,
        "utc_start":   datetime.now(timezone.utc).isoformat(),
        "hostname":    socket.gethostname(),
        "git_commit":  git_commit,
        "seed":        inference_seed,
        "corpora":     corpora,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "dry_run":     args.dry_run,
    }

    smi_before = nvidia_smi_log("before")
    env["nvidia_smi_before"] = smi_before

    K_actual = 30 if args.mode == "c8_ens" else args.K
    print(f"[run_gpu_sweep] mode={args.mode} tag={args.tag} model={model_id} "
          f"K={K_actual} T={args.temperature} seed={inference_seed}", flush=True)

    for corpus_name in corpora:
        input_path = IN_DIR / f"full_input_{corpus_name}.csv"
        if not input_path.exists():
            # Fall back to original 400-item input if full input not prepared yet
            fallback = BASE / "results" / "llm_e3" / f"e3_input_{corpus_name}.csv"
            if fallback.exists():
                print(f"  [WARN] full_input_{corpus_name}.csv not found; "
                      f"using original 400-item input", flush=True)
                input_path = fallback
            else:
                print(f"  [SKIP] no input file for {corpus_name}", flush=True)
                continue

        out_path = OUT_DIR / f"{prefix}_{args.tag}_{corpus_name}.csv"
        if out_path.exists():
            print(f"  [SKIP] {out_path.name} already exists", flush=True)
            continue

        df = pd.read_csv(input_path)
        print(f"\n[{corpus_name}] n={len(df)} items -> {out_path.name}", flush=True)

        if args.dry_run:
            print(f"  [dry_run] skipping inference", flush=True)
            # Write a placeholder with 5 rows so downstream can detect a dry run
            df_fake = df.head(5).copy()
            df_fake["p_yes"]   = 0.5
            df_fake["n_valid"] = 0
            df_fake["H"]       = binary_entropy(0.5)
            df_fake = df_fake[["item_id", "p_yes", "n_valid", "H", "dis_mi", "y_true"]]
            df_fake["_dry_run"] = True
            df_fake.to_csv(out_path, index=False)
            continue

        ckpt_path = OUT_DIR / f"_ckpt_{prefix}_{args.tag}_{corpus_name}.csv"
        result_df = run_inference_hf(
            model_id         = model_id,
            df               = df,
            K                = K_actual,
            temperature      = args.temperature,
            max_new_tokens   = args.max_new_tokens,
            batch_size       = args.batch_size,
            checkpoint_path  = ckpt_path,
            checkpoint_every = args.checkpoint_every,
            inference_seed   = inference_seed,
        )

        # Save final
        result_df.to_csv(out_path, index=False)
        print(f"  [saved] {out_path}  n={len(result_df)} "
              f"mean_pyes={result_df['p_yes'].mean():.3f} "
              f"mean_nvalid={result_df['n_valid'].mean():.1f}", flush=True)

        # Remove checkpoint file
        if ckpt_path.exists():
            ckpt_path.unlink()

        env[f"{corpus_name}_n"] = len(result_df)

    smi_after = nvidia_smi_log("after")
    env["nvidia_smi_after"] = smi_after
    env["utc_end"] = datetime.now(timezone.utc).isoformat()

    env_path = OUT_DIR / f"env_{prefix}_{args.tag}.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"\n[env log] {env_path}", flush=True)

    # Optionally delete HF cache for this model to save disk
    if args.delete_after and not args.dry_run:
        import subprocess as _sp
        try:
            hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
            cache_dir = hf_home / "hub"
            # Identify model folder by org/name slugification
            slug = model_id.replace("/", "--")
            model_cache = cache_dir / f"models--{slug}"
            if model_cache.exists():
                shutil.rmtree(model_cache)
                print(f"  [cleanup] deleted {model_cache}", flush=True)
            else:
                print(f"  [cleanup] cache not found at {model_cache}", flush=True)
        except Exception as e:
            print(f"  [cleanup WARNING] {e}", flush=True)
        print("  Run `nvidia-smi` manually to confirm GPU is free", flush=True)


if __name__ == "__main__":
    main()
