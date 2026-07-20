export TARGET_MODEL="/mnt/data4/models/Qwen/Qwen3-8B"
export DRAFT_MODEL="/root/.cache/huggingface/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4"

CUDA_VISIBLE_DEVICES=7 \
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$TARGET_MODEL" \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--max-num-seqs 128 \
--max-model-len 32768 \
--max-num-batched-tokens 32768 \
--generation-config vllm \
--speculative-config '{"method":"dflash","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":16,"max_model_len":32768,"attention_backend":"FLASH_ATTN"}'