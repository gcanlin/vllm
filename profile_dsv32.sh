#!/usr/bin/env bash
# Capture a torch-profiler trace of DeepSeek-V3.2 steady-state decode.
#
# Prerequisite: the server must be launched WITH the profiler config, e.g.:
#
  VLLM_USE_V2_MODEL_RUNNER=1 vllm serve \
    /mnt/data4/models/deepseek-ai/DeepSeek-V3.2 \
    --trust-remote-code --tensor-parallel-size 8 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/vllm-workspace/bench-results/prof_dsv32","ignore_frontend":true,"delay_iterations":20,"max_iterations":100}'
#
# Without --profiler-config the /start_profile route does not exist.
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
MODEL=${MODEL:-/mnt/data4/models/deepseek-ai/DeepSeek-V3.2}
TRACE_DIR=${TRACE_DIR:-/root/vllm-workspace/bench-results/prof_dsv32}
STEADY_DELAY=${STEADY_DELAY:-25}   # seconds of load before starting capture
CAPTURE_WAIT=${CAPTURE_WAIT:-30}   # covers delay_iterations + max_iterations

cd "$(dirname "$0")"
mkdir -p "$TRACE_DIR"

echo "[1/5] Waiting for server health..."
until curl -sf "$BASE_URL/health" > /dev/null; do sleep 5; done

echo "[2/5] Starting background load (bench_dspark.sh)..."
MODEL="$MODEL" NUM_PROMPTS=64 MAX_CONCURRENCY=8 OUTPUT_LEN=256 \
  bash bench_dspark.sh > "$TRACE_DIR/bench.log" 2>&1 &
BENCH_PID=$!
trap 'kill $BENCH_PID 2>/dev/null || true' EXIT

echo "[3/5] Letting load reach steady state (${STEADY_DELAY}s)..."
sleep "$STEADY_DELAY"
if ! kill -0 "$BENCH_PID" 2>/dev/null; then
  echo "ERROR: bench exited early, see $TRACE_DIR/bench.log" >&2
  exit 1
fi

echo "[4/5] Capturing profile..."
curl -sf -X POST "$BASE_URL/start_profile" \
  || { echo "ERROR: /start_profile failed - was the server started with --profiler-config?" >&2; exit 1; }
sleep "$CAPTURE_WAIT"
echo "Stopping profiler (trace dump can take a minute or two)..."
curl -sf -X POST --max-time 600 "$BASE_URL/stop_profile"

echo "[5/5] Waiting for bench to finish..."
wait "$BENCH_PID" || true
trap - EXIT

echo "Done. Traces:"
ls -lh "$TRACE_DIR"/*.json.gz 2>/dev/null || echo "  (no traces found - check server log)"
