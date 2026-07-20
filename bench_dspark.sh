#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/mnt/data4/models/Qwen/Qwen3-8B}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
ENDPOINT=${ENDPOINT:-/v1/chat/completions}

# Spec-decode-oriented datasets. These JSONL files use {"turns": [...]} and
# are compatible with vLLM's built-in spec_bench dataset loader.
DATASET_PATH=${DATASET_PATH:-/root/vllm-workspace/DeepSpec/eval_datasets/gsm8k.jsonl}
DATASET_NAME=${DATASET_NAME:-spec_bench}
SPEC_BENCH_CATEGORY=${SPEC_BENCH_CATEGORY:-}

OUTPUT_LEN=${OUTPUT_LEN:-256}
NUM_WARMUPS=${NUM_WARMUPS:-5}
NUM_PROMPTS=${NUM_PROMPTS:-150}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-16}
REQUEST_RATE=${REQUEST_RATE:-inf}
TEMPERATURE=${TEMPERATURE:-1.0}

VLLM_BIN=${VLLM_BIN:-.venv/bin/vllm}

args=(
  bench serve
  --backend openai-chat
  --base-url "$BASE_URL"
  --endpoint "$ENDPOINT"
  --model "$MODEL"
  --trust-remote-code
  --dataset-name "$DATASET_NAME"
  --dataset-path "$DATASET_PATH"
  --spec-bench-output-len "$OUTPUT_LEN"
  --num-warmups "$NUM_WARMUPS"
  --num-prompts "$NUM_PROMPTS"
  --request-rate "$REQUEST_RATE"
  --max-concurrency "$MAX_CONCURRENCY"
  --temperature "$TEMPERATURE"
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,99
  --save-result
  --result-dir ./
)

if [[ -n "$SPEC_BENCH_CATEGORY" ]]; then
  args+=(--spec-bench-category "$SPEC_BENCH_CATEGORY")
fi

"$VLLM_BIN" "${args[@]}"
