#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: $0 <rank> <enabled> <label>}"
enabled="${2:?usage: $0 <rank> <enabled> <label>}"
label="${3:?usage: $0 <rank> <enabled> <label>}"

model="${MODEL_PATH:-/gpfs/mszn/models/moonshotai/Kimi-K3}"
master_addr="${DP_ADDRESS:-192.168.5.3}"
rpc_port="${DP_RPC_PORT:-13348}"
output_dir="${OUTPUT_DIR:-/workspace/vllm/ab-results}"

mkdir -p "$output_dir"
export VLLM_KIMI_K3_ROUTED_DOWN_MXFP8="$enabled"
export VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1

exec vllm serve "$model" \
  --served-model-name Kimi-K3 \
  --trust-remote-code \
  --max-model-len 32768 \
  --kv-cache-memory 26843545600 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8 \
  --attention-config \
  '{"use_prefill_query_quantization":true,"mla_prefill_backend":"flashinfer"}' \
  --no-disable-hybrid-kv-cache-manager \
  --no-enable-flashinfer-autotune \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank "$rank" \
  --data-parallel-address "$master_addr" \
  --data-parallel-rpc-port "$rpc_port" \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --all2all-backend deepep_v2 \
  --enable-prefix-caching \
  --max-num-seqs 512 \
  --max-num-batched-tokens 1024 \
  --max-cudagraph-capture-size 512 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  >"$output_dir/$label-node$((rank + 3)).log" 2>&1
