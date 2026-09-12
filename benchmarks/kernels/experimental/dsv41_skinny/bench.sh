#!/usr/bin/env bash
set -euo pipefail

BENCH_INPUT="${1:-8192}"
BENCH_OUTPUT="${2:-1024}"
BENCH_CONCURRENCY="${3:-16}"
BENCH_REQUESTS="${4:-100}"
BENCH_RESULT="${5:-random-8k-1k-c16.json}"
BENCH_RESULT_DIR="${RESULT_DIR:-$PWD/manual}"
mkdir -p "$BENCH_RESULT_DIR"
test ! -e "$BENCH_RESULT_DIR/$BENCH_RESULT"

# Run the installed vLLM CLI with the active venv's interpreter.
uv run --active --no-project python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm \
  --base-url "${BASE_URL:-http://127.0.0.1:8032}" \
  --endpoint /v1/completions \
  --model dsv41-bench \
  --tokenizer "${DSV41_MODEL:-/gpfs/mszn/models/deepseek-ai/DeepSeek-V4.1-Flash}" \
  --tokenizer-mode deepseek_v41 \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len "$BENCH_INPUT" \
  --random-output-len "$BENCH_OUTPUT" \
  --random-range-ratio 0 \
  --random-prefix-len 0 \
  --num-prompts "$BENCH_REQUESTS" \
  --max-concurrency "$BENCH_CONCURRENCY" \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --seed 42 \
  --num-warmups 2 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed --disable-tqdm \
  --result-dir "$BENCH_RESULT_DIR" \
  --result-filename "$BENCH_RESULT"
