#!/bin/bash
# In-container A40 / Llama-3.1-70B profiling, TP=4 (logical TP on a single GPU, 1 layer).
# Invoked inside nvcr.io/nvidia/pytorch:25.01-py3 with /workspace = llm_profile.
set -euo pipefail

HARDWARE="A40"
MODEL="meta-llama/Llama-3.1-70B"
TP="${1:-4}"

echo "==> Installing deps (transformers 4.57.3 + sklearn)"
pip install -q -U pip setuptools wheel packaging transformers==4.57.3 scikit-learn 2>&1 | tail -2

if [ -z "${HF_TOKEN:-}" ] && [ -s /root/.cache/huggingface/token ]; then
  export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

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
