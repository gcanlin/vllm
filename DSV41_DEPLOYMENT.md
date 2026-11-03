# DeepSeek-V4.1-Flash 2P1D 部署

本分支在 vLLM main 上合入了 PD 分离用到的优化。下面在两台 8×B200（IB 互联）上部署 2P1D：

- **P 机器**：etcd、Mooncake master、两个 prefill 实例（DEP4，GPU 0–3 和 4–7），每个实例一个 Dynamo sidecar。
- **D 机器**：Mooncake client、一个 decode 实例（DEP8）及其 sidecar、Dynamo frontend（对外端口 8100）。

请求经 frontend 路由到某个 P 计算 prompt，KV 由 D 经 NIXL 拉走后在 D 上生成。
两个 P 算出的 prefix KV 写进 Mooncake，下一轮无论落到哪个 P 都能从 Mooncake 读回。

启动顺序：P 的 etcd 和 master → D 的 Mooncake client → 三个引擎 → 引擎 `/health` 正常后起各自的 sidecar → frontend。

## 变量

每台机器按实际情况设好，后面的命令都用到：

```bash
P_HOST=<P 机器的 IB 网段地址>
D_HOST=<D 机器的 IB 网段地址>
MY_HOST=<本机的 IB 网段地址>                # P 机器上等于 P_HOST，D 机器上等于 D_HOST
IB_DEVS=<本机 8 张 RDMA 网卡，逗号分隔>        # ibv_devinfo -l 查看
MODEL_DIR=<本机 NVMe 上的模型目录>           # 不要放网络文件系统上，mmap 读权重会卡住
```

## 1. 容器和安装（两台）

用 vLLM 官方 nightly 镜像 `vllm/vllm-openai:nightly-40941f2c4ba96fba0a7e282a5355e89cd74d9ac5`（它和本分支基座之间
没有原生代码改动），把本分支的 Python 代码装进去。本分支改了 Rust frontend 和引擎之间的协议，所以 `vllm-rs` 也要从本分支编译。

```bash
docker run -d --name dsv41 --privileged --network host --ipc host \
  --device nvidia.com/gpu=all --device /dev/infiniband \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --cap-add SYS_NICE --cap-add IPC_LOCK \
  -v $MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro \
  --entrypoint sleep vllm/vllm-openai:nightly-40941f2c4ba96fba0a7e282a5355e89cd74d9ac5 infinity
docker exec -it dsv41 bash
```

容器内：

```bash
git clone -b dsv41-1103 https://github.com/VoiceAgentGroup/vllm-private.git /workspace/vllm
cd /workspace/vllm
SITE=$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')
cp -r $SITE /root/vllm-image-backup                 # 原文件备份
(cd vllm && find . -type f ! -name '*.so' ! -name vllm-rs ! -name _version.py -exec cp --parents {} $SITE/ \;)

# vllm-rs：需要 Rust 1.96、protoc、libssl-dev、pkg-config
cargo build --release --manifest-path rust/src/cmd/Cargo.toml
export VLLM_RUST_FRONTEND_PATH=$PWD/rust/target/release/vllm-rs   # 起引擎前设置

uv pip install --system --no-deps ai-dynamo==1.6.0.dev20261010 ai-dynamo-runtime==1.6.0.dev20261010
```

## 2. etcd 和 Mooncake master（P 机器）

etcd 给 Dynamo 做服务发现，master 只管 Mooncake 的元数据。

```bash
etcd --name pd --data-dir /tmp/etcd-pd \
  --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://$P_HOST:2379 \
  --listen-peer-urls http://127.0.0.1:2380 --initial-advertise-peer-urls http://127.0.0.1:2380 \
  --initial-cluster pd=http://127.0.0.1:2380 &
mooncake_master --port=50051 --eviction_high_watermark_ratio=0.95 --default_kv_lease_ttl=10000 --rpc_thread_num=16 &
```

## 3. Mooncake client（D 机器）

Mooncake 的内存由每个 P rank（各 80 GB）和 D 机器上的这个 client 共同提供。D 不读写 Mooncake，只出内存。

```bash
mooncake_client -master_server_address=$P_HOST:50051 -metadata_server=P2PHANDSHAKE -protocol=rdma \
  -device_names=$IB_DEVS -host=$MY_HOST -global_segment_size=1192GB -local_buffer_size=1GB &
```

## 4. 引擎共用的设置（两台）

```bash
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=1 \
  VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 PYTHONHASHSEED=0 \
  VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_RPC_TIMEOUT=600000 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1
export UCX_TLS=cuda_copy,cuda_ipc,rc,tcp UCX_NET_DEVICES=$(echo $IB_DEVS | sed 's/,/:1,/g'):1 \
  UCX_MEMTYPE_CACHE=n UCX_MEMTYPE_REG_WHOLE=n UCX_RNDV_PIPELINE_ERROR_HANDLING=y NIXL_UCX_SGL_ENABLE=1 \
  UCX_MODULE_DIR=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/ucx
COMMON=(/models/DeepSeek-V4.1-Flash --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0
  --tensor-parallel-size 1 --enable-expert-parallel --enable-ep-weight-filter --tokenizer-mode deepseek_v41
  --kv-cache-dtype fp8 --max-model-len 1048576 --gpu-memory-utilization 0.95
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
  --kernel-config '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}'
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
  --enable-prefix-caching --prefix-cache-retention-interval 32768 --prefix-caching-hash-algo sha256_int32
  --disable-uvicorn-access-log)
NIXL='"kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["UCX"],"kv_lease_duration":300}'
```

## 5. 两个 prefill 实例（P 机器）

P1 用下面的值；P2 换成 GPU `4,5,6,7`、NUMA `1 1 1 1`、端口 8043 / gRPC 9043 / RPC 29750 / NIXL 侧信道 5800。

```bash
cat > /tmp/mooncake.json <<EOF
{"mode":"embedded","metadata_server":"P2PHANDSHAKE","master_server_address":"$P_HOST:50051",
 "global_segment_size":"80GB","local_buffer_size":"4GB","protocol":"rdma","device_name":"$IB_DEVS"}
EOF
export MOONCAKE_CONFIG_PATH=/tmp/mooncake.json MOONCAKE_REQUESTER_LOCAL_HOSTNAME=$P_HOST \
  MC_STORE_MEMCPY=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4 \
  VLLM_MOONCAKE_LOAD_RECV_THREADS=4 VLLM_NIXL_SIDE_CHANNEL_HOST=$P_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=5600
STORE='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":true,"lookup_prefetch_limit":256,"lookup_prefetch_ttl_s":60}}'

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "${COMMON[@]}" \
  --port 8041 --grpc-port 9041 --data-parallel-size 4 --data-parallel-rpc-port 29550 \
  --numa-bind --numa-bind-nodes 0 0 0 0 \
  --max-num-seqs 64 --max-num-batched-tokens 16384 --long-prefill-token-threshold 8192 --prefill-schedule-interval 1 \
  --scheduling-policy residual_sjf --residual-sjf-max-wait-ms 10000 \
  --prefill-step-budget '{"ctx_budget_ms":40,"ctx_balance":[1.15,10,60]}' \
  --prefill-token-cap '{"low_waiting":8,"hold_s":2.0}' \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":true,"use_thp":false,"host_prefetch_threads":8}' \
  --enable-eplb --eplb-config '{"num_redundant_experts":16,"step_interval":5000,"window_size":1000,"use_async":false,"communicator":"pynccl"}' \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","enable_early_recv":true,
    "kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_producer",'"$NIXL"'},'"$STORE"']}}' \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":2046,
    "cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046],
    "decoder_replay_cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2048]}'
```

## 6. decode 实例（D 机器）

`--max-num-seqs` 是每个 DP rank 的上限，按总并发 ÷ 8 设；下面按总并发 2048。

```bash
MS=256
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_NIXL_SIDE_CHANNEL_HOST=$D_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
vllm serve "${COMMON[@]}" \
  --port 8042 --grpc-port 9042 --data-parallel-size 8 --data-parallel-rpc-port 29650 \
  --numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1 \
  --max-num-seqs $MS --max-num-batched-tokens $((10 * MS)) --prefill-schedule-interval 2 \
  --stream-interval 8 --dp-sync-interval 256 \
  --engram-config '{"cpu_offload":false,"embedding_across_dp":true,"use_thp":false}' \
  --enable-eplb --eplb-config '{"num_redundant_experts":16,"step_interval":12000,"window_size":1000,"use_async":false,"communicator":"pynccl"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","enable_early_recv":true,'"$NIXL"'}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":'$((6 * MS))',
    "cudagraph_capture_sizes":['$(seq -s, 6 6 $((6 * MS)))']}'
```

## 7. Dynamo sidecar 和 frontend

所有 Dynamo 进程都要设下面的变量，地址显式写成本机 IB 网段地址：

```bash
export DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS=http://$P_HOST:2379 DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq \
  DYN_TCP_RPC_HOST=$MY_HOST DYN_TCP_RESPONSE_STREAM_HOST=$MY_HOST DYN_EVENT_PLANE_HOST=$MY_HOST
```

引擎 `/health` 返回 200 后，每个引擎起一个 sidecar：

```bash
# P 机器，P1（P2 用 9043）
python3 -m dynamo.vllm.sidecar --grpc-endpoint 127.0.0.1:9041 --disaggregation-mode prefill --component prefill \
  --grpc-startup-deadline-secs 3600 --dyn-tool-call-parser deepseek_v41 --dyn-reasoning-parser deepseek_v41 &
# D 机器
python3 -m dynamo.vllm.sidecar --grpc-endpoint 127.0.0.1:9042 --disaggregation-mode decode \
  --grpc-startup-deadline-secs 3600 --dyn-tool-call-parser deepseek_v41 --dyn-reasoning-parser deepseek_v41 &
```

frontend（D 机器）：

```bash
cat > /tmp/router_policy.yaml <<'EOF'
worker_selection:
  decode: decode-least-loaded
  instances:
    - name: decode-least-loaded
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_threshold: 1.0
EOF
export DYN_ROUTER_TEMPERATURE=0 DYN_ROUTER_QUEUE_THRESHOLD=None \
  DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=10000000 DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS=3600 \
  DYN_TOKENIZER=fastokens DYN_TOKENIZER_CACHE=1 DYN_TOKENIZER_CACHE_BYTES=8000000000 \
  DYN_TCP_CHANNEL_BUFFER=128 DYN_TCP_REQUEST_TIMEOUT=60
python3 -m dynamo.frontend --http-port 8100 --router-mode kv --no-kv-events \
  --active-decode-blocks-threshold None --active-prefill-tokens-threshold None \
  --active-prefill-tokens-threshold-frac None --router-policy-config /tmp/router_policy.yaml &
```

`/v1/models` 里出现模型后即可服务：

```bash
curl http://$D_HOST:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","max_tokens":256,"messages":[{"role":"user","content":"What is 17 * 23?"}]}'
```

## 本分支新增的参数

| 参数 | 位置 | 作用 |
| --- | --- | --- |
| `--scheduling-policy residual_sjf`、`--residual-sjf-max-wait-ms` | P | 按扣除缓存命中后还要算的 token 数排队，等满上限的请求按到达顺序优先 |
| `--prefill-step-budget` | P | 按代价模型限制每步的 prefill 量；`ctx_balance` 让各 DP rank 的上下文预算跟随平均需求，`dp_budget` 让每步不超过其他 rank 发布的最大需求 |
| `--prefill-token-cap` | P | 各 rank 排队都短时用较小的步长和分块，任一 rank 排队变长时恢复满步 |
| `enable_early_recv` | P、D 的 kv-transfer-config | KV 接收完成后在下一次输出前就上报；P 上 Mooncake 加载在调度时直接下发 |
| `lookup_prefetch_limit`、`lookup_prefetch_ttl_s` | P 的 Mooncake 配置 | 请求进引擎时就开始查 Mooncake |
| `host_prefetch_threads` | P 的 engram-config | Engram 表放主机内存时，后台线程提前取好下一步要用的行 |
| `--prefix-caching-hash-algo sha256_int32` | P、D | 直接对 int32 token 计算前缀哈希；共用 Mooncake 的实例必须用同一种算法 |

## 注意

- Mooncake 的内存段注册后常驻，按网卡所在的 NUMA 节点拆开放，规划内存时按每个 NUMA 节点算。
- `kv_lease_duration` 设 300 s，默认 30 s 时 P 可能在 D 读走之前释放 KV。
- 用 containerd/ctr 起容器时，除了 IPC namespace 还要 bind mount 主机的 `/dev/shm`；prefill 的 DP rank 之间经它共享调度状态。
- 不要靠调低 `--gpu-memory-utilization` 绕过启动时的 OOM。

## 实测（2026-11-04）

按上面的部署，AgentX（aiperf，InferenceX 参数）并发 2048，正式测量 1 小时，压测时投机解码用合成接受长度 3.51：

| 指标 | 数值 |
| --- | --- |
| 每卡输出（16 卡） | 5,350 tok/s（总 85,598 tok/s） |
| TTFT p50 / p90 / p99 | 0.64 / 1.61 / 9.71 s |
| ITL p10 / p50 / p90 | 6.71 / 7.03 / 7.85 ms |
| 每用户 tok/s p10 / p50 | 127 / 142 |
| 请求数 / 错误 | 304,010 / 1 |
