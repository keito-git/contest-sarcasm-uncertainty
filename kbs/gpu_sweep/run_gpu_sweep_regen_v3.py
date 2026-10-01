"""
run_gpu_sweep_regen_v3.py — C7/C8 GPU sweep (regen revision, v3).

Changes from run_gpu_sweep_regen.py (v2):
  1. trust_remote_code per model: False for Phi-3-mini (model registry 4th element).
     AutoTokenizer + AutoModelForCausalLM now pass trust_remote_code=False for ll3b.
  2. OOM-adaptive batch: on torch.cuda.OutOfMemoryError, halve batch_size and
     retry the same batch (minimum batch_size=1). Initial q32b default is --batch_size 4.
  3. Full cache cleanup: delete model snapshot directory AND orphaned blobs in
     $HF_HOME/hub/blobs/ (previous --delete_after left ~55 GB of blobs).
  4. Disk guard: `du_check()` before each model; log warning if WORK_DIR exceeds 140 GB.
  5. Resume-from-checkpoint is preserved (identical logic to v2).

Usage (server):
  CUDA_VISIBLE_DEVICES=0 (set via --device 0; done before torch import)

  # q32b — missing EPIC + MultiPICo sub-corpora; checkpoint for CSC already consumed.
  setsid nohup python kbs/gpu_sweep/run_gpu_sweep_regen_v3.py \\
      --mode c7 --tag q32b \\
      --corpora EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it \\
      --device 0 --batch_size 4 > results/revision_gpu_2026-09/run_c7_q32b_v3.log 2>&1 < /dev/null &

  # ll3b — ALL corpora (trust_remote_code=False applied automatically from registry)
  setsid nohup python kbs/gpu_sweep/run_gpu_sweep_regen_v3.py \\
      --mode c8 --tag ll3b \\
      --corpora CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it \\
      --device 0 --batch_size 16 > results/revision_gpu_2026-09/run_c8_ll3b_v3.log 2>&1 < /dev/null &

  # gm2_2b — missing MultiPICo sub-corpora; checkpoint for MultiPICo_es resumes.
  setsid nohup python kbs/gpu_sweep/run_gpu_sweep_regen_v3.py \\
      --mode c8 --tag gm2_2b \\
      --corpora MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it \\
      --device 0 --batch_size 8 > results/revision_gpu_2026-09/run_c8_gm2_2b_v3.log 2>&1 < /dev/null &

GPU rules (shared server):
  - CUDA_VISIBLE_DEVICES set BEFORE torch import (done in main()).
  - Never hardcode cuda:0/cuda:1; use device_map='auto'.
  - Do NOT kill other GPU processes.
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
BASE    = WORK_DIR
IN_DIR  = BASE / "results" / "revision_gpu_2026-09"
OUT_DIR = IN_DIR / "server_outputs" / "regen"  # read by kbs/analysis/*
SEED    = 42

# ── Model registry ───────────────────────────────────────────────────────── #
# tag -> (HuggingFace model_id, capacity_B, trust_remote_code)
C7_MODELS: dict[str, tuple[str, float, bool]] = {
    "q0p5b": ("Qwen/Qwen2.5-0.5B-Instruct",  0.5,  True),
    "q1p5b": ("Qwen/Qwen2.5-1.5B-Instruct",  1.5,  True),
    "q3b":   ("Qwen/Qwen2.5-3B-Instruct",    3.0,  True),
    "q7b":   ("Qwen/Qwen2.5-7B-Instruct",    7.0,  True),
    "q14b":  ("Qwen/Qwen2.5-14B-Instruct",  14.0,  True),
    "q32b":  ("Qwen/Qwen2.5-32B-Instruct",  32.0,  True),
}

C8_MODELS: dict[str, tuple[str, float, bool]] = {
    # ll3b: Phi-3-mini-4k-instruct — rope_scaling KeyError 'type' in transformers 5.x
    # with trust_remote_code=True; must use native transformers Phi3 implementation.
    "ll3b":   ("microsoft/Phi-3-mini-4k-instruct",       3.8,  False),
    # ll1b: SmolLM2-1.7B-Instruct (confirmed: env json says HuggingFaceTB/SmolLM2-1.7B-Instruct)
    "ll1b":   ("HuggingFaceTB/SmolLM2-1.7B-Instruct",   1.7,  False),
    # gm2_2b: stablelm-zephyr-3b
    "gm2_2b": ("stabilityai/stablelm-zephyr-3b",         3.0,  False),
}

C8_NOISE_MODELS: dict[str, tuple[str, float, bool]] = {
    "q0p5b_s2": ("Qwen/Qwen2.5-0.5B-Instruct", 0.5, True),
}
C8_NOISE_SEED = 1234

C8_ENS_MODELS: dict[str, tuple[str, float, bool]] = {
    "ens30_q7b": ("Qwen/Qwen2.5-7B-Instruct", 7.0, True),
}

ALL_MODELS: dict[str, tuple[str, float, bool]] = {
    **C7_MODELS, **C8_MODELS, **C8_NOISE_MODELS, **C8_ENS_MODELS
}

CORPORA = ["CSC", "EPIC",
           "MultiPICo_es", "MultiPICo_de", "MultiPICo_fr", "MultiPICo_it"]

# Expected row counts for post-run verification
EXPECTED_ROWS: dict[str, int] = {
    "CSC":          7036,
    "EPIC":         2999,
    "MultiPICo":    2999,
    "MultiPICo_es": 4681,
    "MultiPICo_de": 2370,
    "MultiPICo_fr": 1760,
    "MultiPICo_it":  998,
}


# ── Prompt (exact copy from regen script — do not change) ────────────────── #
def build_prompt_text(concept: str, ctx: str, txt: str) -> str:
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


# ── Response parsing (exact copy from regen script) ─────────────────────── #
_YES_NO = re.compile(r"\b(yes|no)\b", re.I)
_CONF   = re.compile(r"(\d{1,3})")


def parse_response(text: str) -> tuple[Optional[int], float, float]:
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
    p_yes = 1.0 if label == 1 else 0.0
    return label, p_yes, conf / 100.0


def binary_entropy(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


# ── Disk guard ───────────────────────────────────────────────────────────── #
def du_check(label: str = "") -> int:
    """Return disk usage of WORK_DIR in GB; warn if >= 140 GB."""
    try:
        out = subprocess.check_output(
            ["du", "-sb", str(WORK_DIR)], stderr=subprocess.DEVNULL, timeout=30
        ).decode().split()[0]
        gb = int(out) // (1024 ** 3)
        msg = f"  [du {label}] {WORK_DIR} = {gb} GB"
        if gb >= 140:
            msg += "  *** WARNING: approaching 150 GB limit ***"
        print(msg, flush=True)
        return gb
    except Exception as e:
        print(f"  [du WARNING] {e}", flush=True)
        return 0


# ── nvidia-smi check ─────────────────────────────────────────────────────── #
def nvidia_smi_log(label: str) -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
        print(f"  [nvidia-smi {label}] {out}", flush=True)
        return out
    except Exception:
        return "nvidia-smi unavailable"


# ── Cache cleanup (snapshot + orphaned blobs) ─────────────────────────────── #
def delete_model_cache_full(model_id: str) -> None:
    """Delete model snapshot dir AND orphaned top-level blobs.

    HF hub structure:
      hub/
        models--{org}--{name}/
          snapshots/{rev}/{file} -> ../../blobs/{sha}  (symlinks)
          blobs/{sha}            -> ../blobs/{sha}     (also symlinks on some versions)
          refs/
        blobs/{sha}              (actual content-addressed files, shared)

    Steps:
      1. Collect all blob paths referenced by this model's symlinks.
      2. Delete the model's directory.
      3. Collect all blob paths still referenced by any remaining model.
      4. Delete blobs that are no longer referenced.
    """
    hf_home = Path(
        os.environ.get("HF_HOME",
                       str(Path.home() / ".cache" / "huggingface"))
    )
    cache_dir = hf_home / "hub"
    slug      = model_id.replace("/", "--")
    model_dir = cache_dir / f"models--{slug}"

    if not model_dir.exists():
        print(f"  [cleanup] {model_dir.name} not found — skipping", flush=True)
        return

    # Collect all real blob files referenced by this model
    def collect_blob_targets(root: Path) -> set[Path]:
        targets: set[Path] = set()
        for p in root.rglob("*"):
            if p.is_symlink():
                try:
                    t = p.resolve()
                    if t.exists():
                        targets.add(t)
                except Exception:
                    pass
            elif p.is_file() and not p.is_symlink():
                # Blobs stored directly in model dir (non-symlink layout)
                targets.add(p.resolve())
        return targets

    blobs_this_model = collect_blob_targets(model_dir)
    print(f"  [cleanup] {model_dir.name}: {len(blobs_this_model)} blobs referenced", flush=True)

    # Delete the model directory
    try:
        shutil.rmtree(model_dir)
        print(f"  [cleanup] deleted {model_dir}", flush=True)
    except Exception as e:
        print(f"  [cleanup WARNING] rmtree failed: {e}", flush=True)
        return

    # Collect blobs still referenced by OTHER remaining model dirs
    still_referenced: set[Path] = set()
    for other in cache_dir.iterdir():
        if other.is_dir() and other.name.startswith("models--"):
            still_referenced |= collect_blob_targets(other)

    # Delete orphaned blobs
    n_del = 0
    n_skip = 0
    for b in blobs_this_model:
        if b not in still_referenced and b.exists():
            try:
                b.unlink()
                n_del += 1
            except Exception as e:
                print(f"  [cleanup WARNING] cannot delete {b}: {e}", flush=True)
                n_skip += 1
        else:
            n_skip += 1

    print(f"  [cleanup] orphaned blobs deleted={n_del}  kept(still-ref)={n_skip}",
          flush=True)

    # Also delete the top-level 'blobs' subdir inside model dir (already gone via rmtree)
    # but verify hub/blobs if it exists as a separate top-level dir
    hub_blobs = cache_dir / "blobs"
    if hub_blobs.exists() and hub_blobs.is_dir():
        all_remaining_refs: set[Path] = set()
        for other in cache_dir.iterdir():
            if other.is_dir() and other.name.startswith("models--"):
                all_remaining_refs |= collect_blob_targets(other)
        orphans_hub = []
        for b in hub_blobs.iterdir():
            if b.is_file() and b.resolve() not in all_remaining_refs:
                orphans_hub.append(b)
        for b in orphans_hub:
            try:
                b.unlink()
                n_del += 1
            except Exception as e:
                print(f"  [cleanup WARNING] hub/blobs {b.name}: {e}", flush=True)
        if orphans_hub:
            print(f"  [cleanup] hub/blobs orphans deleted: {len(orphans_hub)}", flush=True)


# ── GPU inference ──────────────────────────────────────────────────────────── #
def run_inference_hf(
    model_id: str,
    df: "pd.DataFrame",
    K: int = 10,
    temperature: float = 1.0,
    max_new_tokens: int = 30,
    batch_size: int = 16,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 200,
    inference_seed: int = 42,
    trust_remote_code: bool = True,
) -> "pd.DataFrame":
    """Run K-sample inference on df using HuggingFace transformers.

    trust_remote_code: set to False for Phi-3-mini (native transformers Phi3 impl).

    OOM-adaptive: on torch.cuda.OutOfMemoryError, halve batch_size and retry the
    same batch in-place (minimum batch_size=1). Reports the reduced batch_size.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  [hf] loading {model_id} "
          f"(seed={inference_seed}, trust_remote_code={trust_remote_code}) ...",
          flush=True)
    _dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id,
                                        trust_remote_code=trust_remote_code)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    torch.manual_seed(inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed)

    # Resume from checkpoint
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if checkpoint_path and checkpoint_path.exists():
        import pandas as _pd
        ckpt = _pd.read_csv(checkpoint_path)
        done_ids = set(ckpt["item_id"].astype(str))
        existing_rows = ckpt.to_dict(orient="records")
        print(f"  [resume] {len(done_ids)} items already done", flush=True)

    rows: list[dict] = list(existing_rows)
    concept  = str(df["concept"].iloc[0])
    ctx_col  = "ctx" if "ctx" in df.columns else None
    todo     = df[~df["item_id"].astype(str).isin(done_ids)].copy()
    n_todo   = len(todo)
    print(f"  [hf] {n_todo} items to run (K={K}, T={temperature})", flush=True)
    t0 = time.time()

    current_batch_size = batch_size
    batch_start        = 0

    while batch_start < n_todo:
        batch = todo.iloc[batch_start : batch_start + current_batch_size]

        # Tokenize
        batch_prompts: list[list[dict]] = []
        for _, row in batch.iterrows():
            ctx_val = str(row[ctx_col]) if ctx_col else ""
            batch_prompts.append(build_messages(concept, ctx_val, str(row["txt"])))

        inputs_text = [
            tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batch_prompts
        ]
        enc = tok(inputs_text, return_tensors="pt", padding=True,
                  truncation=True, max_length=1500).to(model.device)

        # OOM-adaptive generate
        while True:
            try:
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
                break  # success
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    raise RuntimeError(
                        f"CUDA OOM with batch_size=1; cannot reduce further"
                    )
                current_batch_size = max(1, current_batch_size // 2)
                print(f"  [OOM] batch_size halved to {current_batch_size}; "
                      "retrying ...", flush=True)
                del enc
                torch.cuda.empty_cache()
                # Re-slice and re-tokenize with new batch size
                batch = todo.iloc[batch_start : batch_start + current_batch_size]
                batch_prompts = []
                for _, row in batch.iterrows():
                    ctx_val = str(row[ctx_col]) if ctx_col else ""
                    batch_prompts.append(
                        build_messages(concept, ctx_val, str(row["txt"]))
                    )
                inputs_text = [
                    tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    for msgs in batch_prompts
                ]
                enc = tok(inputs_text, return_tensors="pt", padding=True,
                          truncation=True, max_length=1500).to(model.device)

        # Decode
        input_len = enc["input_ids"].shape[1]
        gen_new   = gen_ids[:, input_len:]
        texts     = tok.batch_decode(gen_new, skip_special_tokens=True)
        actual_bs = len(batch)

        for idx_in_batch, (_, row) in enumerate(batch.iterrows()):
            item_texts  = texts[idx_in_batch * K : (idx_in_batch + 1) * K]
            labels: list[int]  = []
            conf_p_yes_list: list[float] = []
            for txt_out in item_texts:
                lbl, _, conf_raw = parse_response(txt_out)
                if lbl is not None:
                    labels.append(lbl)
                    conf_p_yes_list.append(
                        conf_raw if lbl == 1 else 1.0 - conf_raw
                    )

            n_valid      = len(labels)
            p_yes        = float(sum(labels) / max(n_valid, 1))
            H            = binary_entropy(p_yes) if n_valid > 0 else float("nan")
            p_conf_mean  = (float(sum(conf_p_yes_list) / len(conf_p_yes_list))
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
        done_so_far = batch_start + actual_bs
        if checkpoint_path and done_so_far % checkpoint_every < actual_bs:
            import pandas as _pd
            _pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

        if done_so_far % (current_batch_size * 10) < actual_bs or done_so_far >= n_todo:
            elapsed = time.time() - t0
            print(f"  [{done_so_far}/{n_todo}] elapsed={elapsed:.0f}s "
                  f"batch_size={current_batch_size}", flush=True)

        batch_start += actual_bs

    import pandas as _pd
    result = _pd.DataFrame(rows)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ── Main ─────────────────────────────────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="GPU sweep v3: resume + trust_remote_code + OOM-adaptive"
    )
    ap.add_argument("--mode",    choices=["c7", "c8", "c8_ens", "c8_noise"],
                    default="c7")
    ap.add_argument("--tag",     required=True)
    ap.add_argument("--corpora",
                    default="CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it")
    ap.add_argument("--device",  default="0")
    ap.add_argument("--K",        type=int,   default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size",  type=int,   default=16)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--checkpoint_every", type=int, default=200)
    ap.add_argument("--delete_after", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--dry_run",  action="store_true")
    args = ap.parse_args()

    # GPU rule: set CUDA_VISIBLE_DEVICES BEFORE any torch import
    if args.device.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import pandas as pd  # safe to import now

    # Verify tag
    registry = ALL_MODELS
    if args.tag not in registry:
        print(f"[ERROR] unknown tag '{args.tag}'. Available: {list(registry.keys())}")
        sys.exit(1)
    model_id, cap_B, trust_rc = registry[args.tag]

    inference_seed = C8_NOISE_SEED if args.mode == "c8_noise" else SEED
    prefix         = args.mode
    corpora        = [c.strip() for c in args.corpora.split(",") if c.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Env log
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    env: dict = {
        "script":           "run_gpu_sweep_regen_v3.py",
        "mode":             args.mode,
        "tag":              args.tag,
        "model_id":         model_id,
        "cap_B":            cap_B,
        "trust_remote_code": trust_rc,
        "K":                args.K,
        "temperature":      args.temperature,
        "batch_size_init":  args.batch_size,
        "utc_start":        datetime.now(timezone.utc).isoformat(),
        "hostname":         socket.gethostname(),
        "git_commit":       git_commit,
        "seed":             inference_seed,
        "corpora":          corpora,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "dry_run":          args.dry_run,
    }

    # Disk check before loading model
    du_check(f"before {args.tag}")
    smi_before = nvidia_smi_log("before")
    env["nvidia_smi_before"] = smi_before
    env["du_before_GB"]      = du_check(f"{args.tag}_init")

    K_actual = 30 if args.mode == "c8_ens" else args.K
    print(f"[run_gpu_sweep_regen_v3] mode={args.mode} tag={args.tag} "
          f"model={model_id} K={K_actual} T={args.temperature} "
          f"seed={inference_seed} trust_remote_code={trust_rc}",
          flush=True)

    for corpus_name in corpora:
        input_path = IN_DIR / f"full_input_{corpus_name}.csv"
        if not input_path.exists():
            fallback = BASE / "results" / "llm_e3" / f"e3_input_{corpus_name}.csv"
            if fallback.exists():
                print(f"  [WARN] full_input_{corpus_name}.csv not found; "
                      "using original 400-item input", flush=True)
                input_path = fallback
            else:
                print(f"  [SKIP] no input file for {corpus_name}", flush=True)
                continue

        out_path = OUT_DIR / f"{prefix}_{args.tag}_{corpus_name}.csv"
        if out_path.exists():
            print(f"  [SKIP] {out_path.name} already exists", flush=True)
            continue

        df = pd.read_csv(input_path)
        print(f"\n[{corpus_name}] n={len(df)} -> {out_path.name}", flush=True)

        if args.dry_run:
            df_fake = df.head(5).copy()
            df_fake["p_yes"]   = 0.5
            df_fake["n_valid"] = 0
            df_fake["H"]       = binary_entropy(0.5)
            df_fake = df_fake[["item_id", "p_yes", "n_valid", "H",
                                "dis_mi", "y_true"]]
            df_fake["_dry_run"] = True
            df_fake.to_csv(out_path, index=False)
            continue

        ckpt_path = OUT_DIR / f"_ckpt_{prefix}_{args.tag}_{corpus_name}.csv"
        result_df = run_inference_hf(
            model_id           = model_id,
            df                 = df,
            K                  = K_actual,
            temperature        = args.temperature,
            max_new_tokens     = args.max_new_tokens,
            batch_size         = args.batch_size,
            checkpoint_path    = ckpt_path,
            checkpoint_every   = args.checkpoint_every,
            inference_seed     = inference_seed,
            trust_remote_code  = trust_rc,
        )

        result_df.to_csv(out_path, index=False)
        n_rows  = len(result_df)
        exp_n   = EXPECTED_ROWS.get(corpus_name, "?")
        status  = "OK" if n_rows == exp_n else f"WARN: expected {exp_n}"
        print(f"  [saved] {out_path}  n={n_rows} ({status})  "
              f"mean_pyes={result_df['p_yes'].mean():.3f}  "
              f"mean_nvalid={result_df['n_valid'].mean():.1f}", flush=True)

        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"  [ckpt] removed {ckpt_path.name}", flush=True)

        env[f"{corpus_name}_n"] = n_rows

    smi_after          = nvidia_smi_log("after")
    env["nvidia_smi_after"] = smi_after
    env["utc_end"]          = datetime.now(timezone.utc).isoformat()

    env_path = OUT_DIR / f"env_{prefix}_{args.tag}_v3.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"\n[env log] {env_path}", flush=True)

    # Delete model weights + orphaned blobs
    if args.delete_after and not args.dry_run:
        print(f"\n[cleanup] deleting {model_id} cache (snapshot + orphaned blobs) ...",
              flush=True)
        delete_model_cache_full(model_id)

    du_check(f"after {args.tag} cleanup")
    nvidia_smi_log("final")


if __name__ == "__main__":
    main()
