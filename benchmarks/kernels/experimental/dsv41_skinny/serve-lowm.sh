#!/usr/bin/env bash
set -euo pipefail
exec uv run --active --no-project python -m vllm.entrypoints.cli.main serve \
  "${DSV41_MODEL:-/gpfs/mszn/models/deepseek-ai/DeepSeek-V4.1-Flash}" \
  --served-model-name dsv41-bench --host 127.0.0.1 --port 8032 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --distributed-executor-backend mp --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV41 \
  --kv-cache-dtype fp8 --block-size 128 \
  --max-model-len 16384 --max-num-batched-tokens 8192 --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 --seed 2026 \
  --no-enable-prefix-caching --no-enable-flashinfer-autotune \
  --generation-config vllm --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256],"max_cudagraph_capture_size":256}' \
  --worker-cls lowm_worker.LowMWorker
