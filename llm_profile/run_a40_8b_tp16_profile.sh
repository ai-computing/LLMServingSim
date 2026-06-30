#!/bin/bash
# In-container A40 / Llama-3.1-8B profiling at TP=16 (logical TP on a single GPU, 1 layer).
# tp16 > num_key_value_heads(8) -> profiler replicates KV heads (>=1/rank). Offline (random
# weights; only the gated config.json is needed, seeded into the HF cache below).
# Run inside profiler24 (nvcr.io/nvidia/pytorch:24.09-py3, torch 2.5.1, transformers 4.57.3),
# workdir = /app/LLMServingSim/llm_profile.
set -euo pipefail

HARDWARE="A40"
MODEL="meta-llama/Llama-3.1-8B"
TP="${1:-16}"

# --- Seed the gated config.json offline (no 16GB weight download) ---
HUB="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B"
mkdir -p "$HUB/refs" "$HUB/snapshots/local"
printf 'local' > "$HUB/refs/main"
cp /app/LLMServingSim/model_config/meta-llama/Llama-3.1-8B.json "$HUB/snapshots/local/config.json"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "==> [1/3] Layer latency profiling (tp=$TP)"
CUDA_VISIBLE_DEVICES=0 python3 -m profiler.layers.main \
  --hardware "$HARDWARE" --model "$MODEL" \
  --num-layers 1 --tp-size "$TP" \
  --warmup 10 --repeat 30 --max-len 10 --device cuda \
  --profile-method cuda_event

echo "==> [2/3] Attention latency profiling (tp=$TP)"
CUDA_VISIBLE_DEVICES=0 python3 -m profiler.attention.main \
  --hardware "$HARDWARE" --model "$MODEL" \
  --max-len 2048 --tp-size "$TP" \
  --warmup 10 --repeat 50 --device cuda \
  --profile-method cuda_event

echo "==> [3/3] Building attention predictor (tp=$TP)"
python3 -m profiler.predictor.main \
  --model "$MODEL" --hardware "$HARDWARE" \
  --tp-size "$TP" --kv-granularity 64 --chunk-granularity 32 \
  --max-len 2048 --max-batch 256

echo "==> DONE. Outputs under perf_models/$HARDWARE/$MODEL/tp$TP/"
