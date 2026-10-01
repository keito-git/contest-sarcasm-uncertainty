#!/bin/bash
# run_all_v2.sh — C7 + C8 full sweep (v2: regen script, substituted C8 models)
# Runs models sequentially; each model writes checkpoints every 200 items.
# All times in UTC. Run from the repository root.
# IMPORTANT: CUDA_VISIBLE_DEVICES is set inside the Python script before torch import.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
WORK_DIR="${CONTEST_WORK_DIR:-$REPO_DIR}"
LOGDIR="${WORK_DIR}/results/revision_gpu_2026-09/server_outputs/regen"
SCRIPT="${REPO_DIR}/kbs/gpu_sweep/run_gpu_sweep_regen.py"
VENV="${CONTEST_VENV:-${REPO_DIR}/venv}/bin/activate"
mkdir -p "$LOGDIR"

# C7: 6 main non-EN corpora + MultiPICo (full EN) for disclosure comparison
CORPORA_C7="CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it,MultiPICo"
# C8: 6 main corpora (no EN MultiPICo: 80% overlap with EPIC)
CORPORA_C8="CSC,EPIC,MultiPICo_es,MultiPICo_de,MultiPICo_fr,MultiPICo_it"

source "$VENV"

log() {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOGDIR}/run_all_v2.log"
}

run_model() {
    local mode="$1"
    local tag="$2"
    local corpora="$3"
    log "START mode=$mode tag=$tag"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null | tee -a "${LOGDIR}/run_all_v2.log" || true
    python "$SCRIPT" --mode "$mode" --tag "$tag" --corpora "$corpora" \
        --device 0 --batch_size 16 --K 10 --temperature 1.0 --delete_after \
        2>&1 | tee -a "${LOGDIR}/run_${mode}_${tag}.log"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null | tee -a "${LOGDIR}/run_all_v2.log" || true
    log "DONE mode=$mode tag=$tag"
}

# Clean up smoke test artifacts
rm -f "${LOGDIR}/c7_q0p5b_SMOKE5.csv" "${LOGDIR}/full_input_SMOKE5.csv"

# ==== C7: Qwen2.5 6-size sweep ====
log "=== C7 START ==="
for TAG in q0p5b q1p5b q3b q7b q14b q32b; do
    run_model c7 "$TAG" "$CORPORA_C7"
done
log "=== C7 DONE ==="

# ==== C8: diversity models (ungated substitutes; same tags for downstream compat) ====
# ll3b = Phi-3-mini-4k-instruct (3.8B; was Llama-3.2-3B, gated)
# ll1b = SmolLM2-1.7B-Instruct (1.7B; was Llama-3.2-1B, gated)
# gm2_2b = stablelm-zephyr-3b (3B; was gemma-2-2b-it, gated)
log "=== C8 START ==="
for TAG in ll3b ll1b gm2_2b; do
    run_model c8 "$TAG" "$CORPORA_C8"
done
log "=== C8 DONE ==="

# ==== C8 noise control: Qwen2.5-0.5B, seed=1234 (independent from C7 seed=42) ====
log "=== C8_noise START ==="
run_model c8_noise q0p5b_s2 "$CORPORA_C8"
log "=== C8_noise DONE ==="

log "=== ALL RUNS COMPLETE ==="
df -h / | tail -1
ls -la "${LOGDIR}"/c7_*.csv "${LOGDIR}"/c8_*.csv 2>/dev/null | wc -l
log "Result file count above"
