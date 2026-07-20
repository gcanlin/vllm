export TARGET_MODEL=/mnt/data4/models/Qwen/Qwen3-8B
export DRAFT_MODEL=/root/.cache/huggingface/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
export PROFILE_DIR=/root/vllm-workspace/vllm/vllm-prof/qwen3_dflash_$(date +%Y%m%d_%H%M%S)

mkdir -p "$PROFILE_DIR"

SPEC_CONFIG='{"method":"dflash","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":16,"max_model_len":32768}'

CUDA_VISIBLE_DEVICES=7 \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 \
vllm bench latency \
--model "$TARGET_MODEL" \
--trust-remote-code \
--max-model-len 32768 \
--max-num-seqs 128 \
--gpu-memory-utilization 0.85 \
--speculative-config "$SPEC_CONFIG" \
--input-len 1024 \
--output-len 128 \
--batch-size 8 \
--num-iters-warmup 3 \
--num-iters 1 \
--profile \
--profiler-config.profiler=torch \
--profiler-config.torch_profiler_dir="$PROFILE_DIR" \
--profiler-config.torch_profiler_record_shapes=true \
--profiler-config.torch_profiler_with_stack=true \
--profiler-config.torch_profiler_with_memory=false \
--profiler-config.torch_profiler_use_gzip=true
