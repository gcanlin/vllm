#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: $0 <rank> <label> <local-concurrency> <global-concurrency> <num-prompts>}"
label="${2:?usage: $0 <rank> <label> <local-concurrency> <global-concurrency> <num-prompts>}"
local_concurrency="${3:?usage: $0 <rank> <label> <local-concurrency> <global-concurrency> <num-prompts>}"
global_concurrency="${4:?usage: $0 <rank> <label> <local-concurrency> <global-concurrency> <num-prompts>}"
num_prompts="${5:?usage: $0 <rank> <label> <local-concurrency> <global-concurrency> <num-prompts>}"

model="${MODEL_PATH:-/gpfs/mszn/models/moonshotai/Kimi-K3}"
output_dir="${OUTPUT_DIR:-/workspace/vllm/ab-results}"
result_name="$label-c$global_concurrency-node$((rank + 3))"

mkdir -p "$output_dir"
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model Kimi-K3 \
  --tokenizer "$model" \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 256 \
  --random-range-ratio 0 \
  --num-prompts "$num_prompts" \
  --max-concurrency "$local_concurrency" \
  --seed 20260821 \
  --ignore-eos \
  --save-result \
  --result-dir "$output_dir" \
  --result-filename "$result_name.json" \
  >"$output_dir/$result_name.log" 2>&1
