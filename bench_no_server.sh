#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

MODEL=${MODEL:-/mnt/data4/models/Qwen/Qwen3-8B}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
ENDPOINT=${ENDPOINT:-/v1/chat/completions}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_WARMUPS=${NUM_WARMUPS:-5}
NUM_PROMPTS=${NUM_PROMPTS:-150}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-16}


vllm bench serve \
  --backend openai-chat \
  --base-url "$BASE_URL" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --trust-remote-code \
  --dataset-name random \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --num-warmups "$NUM_WARMUPS" \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate inf \
  --max-concurrency "$MAX_CONCURRENCY" \
  --temperature 1.0 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result \
  --result-dir ./
