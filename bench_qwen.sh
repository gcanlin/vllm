vllm serve /mnt/data1/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/dc4d348443bc740c68e2d77492492c11606384d5 \
    --host 127.0.0.1 \
    --port 9256 \
    --trust-remote-code \
    --enable-expert-parallel \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8193 \
    --max-num-seqs 128 

vllm bench serve \
    --model /mnt/data1/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/dc4d348443bc740c68e2d77492492c11606384d5 \
    --dataset-name random \
    --host 127.0.0.1 \
    --port 9256 \
    --input-len 8192 \
    --output-len 1 \
    --request-rate inf \
    --num-prompts 128 \
    --num-warmups 16



 CUDA_VISIBLE_DEVICES=1,4,5,6 VLLM_DISABLE_COMPILE_CACHE=1 vllm serve /mnt/data1/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/dc4d348443bc740c68e2d77492492c11606384d5 \
      --served-model-name qwen35-mtp \
      --host 127.0.0.1 \
      --port 9256 \
      --trust-remote-code \
      --enable-expert-parallel \
      --tensor-parallel-size 4 \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.95 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

MODEL_PATH=/mnt/data1/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/dc4d348443bc740c68e2d77492492c11606384d5

lm_eval \
      --model local-completions \
      --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=qwen35-mtp,tokenizer=${MODEL_PATH},tokenizer_backend=huggingface,trust_remote_code=True,num_concurrent=1024" \
      --tasks gsm8k