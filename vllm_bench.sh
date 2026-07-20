 VLLM_USE_V2_MODEL_RUNNER=1 vllm bench latency \
     --model /mnt/data4/models/deepseek-ai/DeepSeek-V3.2 \
     --trust-remote-code \
     --tensor-parallel-size 8 \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.9 \
     --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
     --input-len 1024 --output-len 32 --batch-size 8 \
     --num-iters-warmup 3 --num-iters 3 \
     --profile \
     --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/vllm-workspace/prof_dsv32"}' \
     > /root/vllm-workspace/prof_dsv32_run.log