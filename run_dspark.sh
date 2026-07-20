export TARGET_MODEL=/mnt/data4/models/Qwen/Qwen3-8B
export DRAFT_MODEL=/root/.cache/huggingface/hub/models--Dogacel--Qwen3-8B-DSpark/snapshots/d9895dcabc6e91d5d49dd064902958f374f44bdc
export SPEC_CONFIG='{"method":"dspark","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":16,"max_model_len":32768,"attention_backend":"FLASH_ATTN"}'

CUDA_VISIBLE_DEVICES=7 \
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$TARGET_MODEL" \
--host 0.0.0.0 \
--port 8000 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--max-num-seqs 128 \
--max-model-len 32768 \
--max-num-batched-tokens 32768 \
--generation-config vllm \
--default-chat-template-kwargs '{"enable_thinking": false}' \
--speculative-config "$SPEC_CONFIG"