#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/mnt/data3/models/Qwen/Qwen3-235B-A22B-Instruct-2507}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-9256}
TP=${TP:-8}

INPUT_LEN=${INPUT_LEN:-8192}
OUTPUT_LEN=${OUTPUT_LEN:-1}
NUM_PROMPTS=${NUM_PROMPTS:-128}
NUM_WARMUPS=${NUM_WARMUPS:-16}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8193}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1048576}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
PROFILE_DIR=${PROFILE_DIR:-/tmp/profile_vllm_qwen3_235b_pr46635}
SERVER_LOG=${SERVER_LOG:-/tmp/vllm_qwen3_235b_pr46635_server.log}

mkdir -p "$PROFILE_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --enable-expert-parallel \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --generation-config vllm \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="$PROFILE_DIR" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Started vLLM server pid=$SERVER_PID, log=$SERVER_LOG"
echo "Waiting for http://$HOST:$PORT/health ..."
for _ in $(seq 1 300); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM server exited early. Last 120 log lines:" >&2
    tail -n 120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done

curl -fsS "http://$HOST:$PORT/health" >/dev/null

vllm bench serve \
  --model "$MODEL" \
  --dataset-name random \
  --host "$HOST" \
  --port "$PORT" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --request-rate inf \
  --num-prompts "$NUM_PROMPTS" \
  --num-warmups "$NUM_WARMUPS"
