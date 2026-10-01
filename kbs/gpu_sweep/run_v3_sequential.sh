#!/usr/bin/env bash
# run_v3_sequential.sh — run q32b, ll3b, gm2_2b sequentially on GPU 0
# Each model loaded, run, then cache deleted before next model.
set -e
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${CONTEST_VENV:-${REPO_DIR}/venv}"
CODE="${REPO_DIR}/kbs/gpu_sweep"

echo '=== [v3 sequential] start' $(date -u) '==='

echo '--- [1/3] c7 q32b (EPIC + MultiPICo sub-corpora, batch_size=4) ---'
$VENV/bin/python $CODE/run_gpu_sweep_regen_v3.py \
    --mode c7 --tag q32b \
    --corpora EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it \
    --device 0 --batch_size 4

echo '--- [2/3] c8 ll3b (all corpora, trust_remote_code=False auto, batch_size=16) ---'
$VENV/bin/python $CODE/run_gpu_sweep_regen_v3.py \
    --mode c8 --tag ll3b \
    --corpora CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it \
    --device 0 --batch_size 16

echo '--- [3/3] c8 gm2_2b (missing MultiPICo sub-corpora; resume from ckpt_es) ---'
$VENV/bin/python $CODE/run_gpu_sweep_regen_v3.py \
    --mode c8 --tag gm2_2b \
    --corpora MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it \
    --device 0 --batch_size 8

echo '=== [v3 sequential] ALL DONE' $(date -u) '==='
