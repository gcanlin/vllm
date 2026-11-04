# GLM-5.3-Flash 1P1D 部署

本分支在 `dsv41-1103`（vLLM main 加 PD 分离的优化）上合入了 GLM-5.3-Flash 用到的改动。下面在一台 8×B200 上部署 1P1D：

- **prefill**：GPU 0–3，DEP4，MegaMoE FP8。prefix KV 存进 Mooncake，4 个 rank 共用。
- **decode**：GPU 4–7，DEP4，KDA 层用 RecoverSSM（float32 状态，CUDA kernel）。
- **Mooncake**：master 和 client 在单独的容器里，提供 1300 GB 内存加 6 TiB SSD（LRU）。
- **Dynamo**：两个 sidecar 和 frontend（对外端口 8100）。

请求经 frontend 到 prefill，prefill 从 Mooncake 读回已有的前缀，只算剩下的部分；KV 经 NIXL（同机 NVLink）到 decode 生成。

启动顺序：Mooncake master → Mooncake client → prefill → decode → 两个引擎 `/health` 正常后起 sidecar → frontend。

## 变量

```bash
HOST_IP=<本机 IB 网段地址>
IB_DEVS=<本机 8 张 RDMA 网卡，逗号分隔>        # ibv_devinfo -l 查看
MODEL_DIR=<本机 NVMe 上的模型目录>           # 不要放网络文件系统上，mmap 读权重会卡住
MC_SSD_DIR=<本机 NVMe 上给 Mooncake SSD 层的目录>  # XFS，至少 6 TiB 空闲
```

## 1. 容器和安装

用 vLLM 官方 nightly 镜像 `vllm/vllm-openai:nightly-40941f2c4ba96fba0a7e282a5355e89cd74d9ac5`（它和本分支基座之间
没有原生代码改动），把本分支的 Python 代码装进去。本分支改了 Rust frontend 和引擎之间的协议，`vllm-rs` 也要从本分支编译。

两个容器各设内存上限，合计给宿主机留出至少 250 GiB，超用时只在自己的容器里回收或 OOM：

```bash
IMG=vllm/vllm-openai:nightly-40941f2c4ba96fba0a7e282a5355e89cd74d9ac5
docker run -d --name glm53-flash --privileged --network host --ipc host --memory 400g \
  --device nvidia.com/gpu=all --device /dev/infiniband \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --cap-add SYS_NICE --cap-add IPC_LOCK \
  -v $MODEL_DIR:/models/GLM-5.3-Flash:ro --entrypoint sleep $IMG infinity
docker run -d --name glm53-flash-mooncake --privileged --network host --ipc host --memory 1360g \
  --device /dev/infiniband --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  -v $MC_SSD_DIR:/data/mooncake-kv --entrypoint sleep $IMG infinity
```

`glm53-flash` 容器内：

```bash
git clone -b glm53-flash-1104 https://github.com/VoiceAgentGroup/vllm-private.git /workspace/vllm
cd /workspace/vllm
SITE=$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')
cp -r $SITE /root/vllm-image-backup                 # 原文件备份
(cd vllm && find . -type f ! -name '*.so' ! -name vllm-rs ! -name _version.py -exec cp --parents {} $SITE/ \;)

# vllm-rs：需要 Rust 1.96、protoc、libssl-dev、pkg-config
cargo build --release --manifest-path rust/src/cmd/Cargo.toml
export VLLM_RUST_FRONTEND_PATH=$PWD/rust/target/release/vllm-rs   # 起引擎前设置

uv pip install --system --no-deps ai-dynamo==1.6.0.dev20261010 ai-dynamo-runtime==1.6.0.dev20261010
```

## 2. Mooncake master 和 client（`glm53-flash-mooncake` 容器）

内存满到 90% 时，master 把对象写到 client 的 SSD 层，而不是直接丢掉。

```bash
mooncake_master --port=50051 --eviction_high_watermark_ratio=0.90 \
  --default_kv_lease_ttl=10000 --rpc_thread_num=16 \
  --enable_offload=true --offload_on_evict=true --allocation_strategy=ssd_free_ratio_first \
  --offloading_queue_limit=500000 --offload_cap_ratio=0.8 &

SSD=$((6 * 1024**4))
mkdir -p /data/mooncake-kv/glm
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/data/mooncake-kv/glm \
  MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru \
  MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$SSD MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES=$SSD \
  MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$SSD MOONCAKE_OFFLOAD_TOTAL_KEYS_LIMIT=80000000 \
  MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=$((20 * 1024**3))
mooncake_client -master_server_address=$HOST_IP:50051 -metadata_server=P2PHANDSHAKE -protocol=rdma \
  -device_names=$IB_DEVS -host=$HOST_IP -global_segment_size=1300GB -local_buffer_size=1GB -enable_offload=true &
```

master 日志出现 `Mem Storage: 0 B / 1.27 TB` 后再起 prefill（注册 1.3 TB 约 1 分钟）。SSD 读缓冲要 20 GiB，默认的 1.25 GB
在 4 个 prefill rank 同时读时会溢出。

## 3. 引擎共用的设置（`glm53-flash` 容器）

```bash
export VLLM_USE_RUST_FRONTEND=1 PYTHONHASHSEED=0 VLLM_SSM_CONV_STATE_LAYOUT=DS \
  VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_RPC_TIMEOUT=600000
# 同机 NVLink；不设 UCX_CUDA_COPY_DMABUF=no 时，cuda_copy 每次查询都导出再关闭 dma-buf，单次传输会拖到上百秒
export UCX_TLS=cuda_copy,cuda_ipc,tcp UCX_MEMTYPE_CACHE=n UCX_MEMTYPE_REG_WHOLE=n UCX_RNDV_PIPELINE_ERROR_HANDLING=y \
  UCX_CUDA_COPY_DMABUF=no NIXL_UCX_SGL_ENABLE=1 UCX_MODULE_DIR=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/ucx
COMMON=(/models/GLM-5.3-Flash --served-model-name zai-org/GLM-5.3-Flash --host 0.0.0.0
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
  --language-model-only --tokenizer-mode glm5_next
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32
  --max-model-len 1048576 --gpu-memory-utilization 0.95
  --no-disable-hybrid-kv-cache-manager --prefix-match-unit 16 --disable-extensible-kv-cache
  --kernel-config '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe"}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' --disable-uvicorn-access-log)
NIXL='"kv_buffer_device":"cuda","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["UCX"],"enforce_handshake_compat":false,"kv_lease_duration":300}'
```

## 4. prefill（GPU 0–3）

prefill 自己不出内存（`global_segment_size` 0），4 个 rank 共用 client 的池子，会话不需要粘在某个 rank 上。
Mooncake 读失败时重算前缀，请求不报错。

```bash
cat > /tmp/mooncake_glm53.json <<EOF
{"mode":"standalone-store","metadata_server":"P2PHANDSHAKE","master_server_address":"$HOST_IP:50051",
 "global_segment_size":0,"local_buffer_size":"4GB","protocol":"rdma","device_name":"$IB_DEVS","enable_offload":true}
EOF
STORE='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":true}}'

CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
MOONCAKE_CONFIG_PATH=/tmp/mooncake_glm53.json MOONCAKE_REQUESTER_LOCAL_HOSTNAME=$HOST_IP \
MC_STORE_MEMCPY=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4 \
VLLM_MOONCAKE_LOAD_RECV_THREADS=4 \
vllm serve "${COMMON[@]}" --port 8041 --grpc-port 9041 --data-parallel-rpc-port 29550 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --compilation-config '{"cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092,6144,8190]}' \
  --max-cudagraph-capture-size 8190 \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_producer",'"$NIXL"'},'"$STORE"']}}'
```

## 5. decode（GPU 4–7）

`--use-replayssm` 打开 RecoverSSM：每个请求只占一个 KDA 状态槽，被接受的 token 在采样后再并入状态。
verify 和 commit 的 CUDA kernel 在 decode 第一次用到时编译，产物在 `/tmp/kda_cuda_build`（`KDA_CUDA_BUILD` 可改）。
MTP 5 时每个请求每步 6 个 token，`--max-num-seqs` 64 对应 CUDA graph 到 384。prefill 起来后再起 decode。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
vllm serve "${COMMON[@]}" --port 8042 --grpc-port 9042 --data-parallel-rpc-port 29650 \
  --enable-prefix-caching --max-num-seqs 64 --max-num-batched-tokens 640 --use-replayssm \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":384,
    "cudagraph_capture_sizes":['$(seq -s, 6 6 384)']}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer",'"$NIXL"'}'
```

## 6. Dynamo sidecar 和 frontend

单机用文件做服务发现。不要加 `--dyn-reasoning-parser`：Dynamo 没有 glm47 的 reasoning parser，回退的解析会把带 stop
的回答当成 reasoning。

```bash
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/glm53-flash-dynkv DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
python3 -m dynamo.vllm.sidecar --grpc-endpoint 127.0.0.1:9041 --disaggregation-mode prefill --component prefill \
  --grpc-startup-deadline-secs 3600 --dyn-tool-call-parser glm47 &
python3 -m dynamo.vllm.sidecar --grpc-endpoint 127.0.0.1:9042 --disaggregation-mode decode \
  --grpc-startup-deadline-secs 3600 --dyn-tool-call-parser glm47 &

cat > /tmp/router_policy.yaml <<'EOF'
worker_selection:
  decode: decode-least-loaded
  instances:
    - name: decode-least-loaded
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_threshold: 1.0
EOF
export DYN_ROUTER_SESSION_AFFINITY_MODE=soft DYN_ROUTER_TEMPERATURE=0 DYN_ROUTER_QUEUE_THRESHOLD=None \
  DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=10000000 DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS=3600 \
  DYN_TOKENIZER=fastokens DYN_TOKENIZER_CACHE=1 DYN_TOKENIZER_CACHE_BYTES=8000000000 \
  DYN_TCP_CHANNEL_BUFFER=128 DYN_TCP_REQUEST_TIMEOUT=60
python3 -m dynamo.frontend --http-port 8100 --router-mode kv --router-session-affinity-ttl-secs 14400 --no-kv-events \
  --active-decode-blocks-threshold None --active-prefill-tokens-threshold None \
  --active-prefill-tokens-threshold-frac None --router-policy-config /tmp/router_policy.yaml &
```

`/v1/models` 里出现模型后即可服务：

```bash
curl http://$HOST_IP:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3-Flash","max_tokens":300,"messages":[{"role":"user","content":"What is 17 * 23?"}]}'
```

## 本分支相对 dsv41-1103 的改动

| commit | 作用 |
| --- | --- |
| vllm-project/vllm#60895 | GLM-5.3-Flash 的 FP8 MoE 走 DeepGEMM MegaMoE（共享专家融合进来）；块 FP8 权重在加载时转成 MXFP8 |
| vllm-project/vllm#65137 | extensible KV 在没有 IMEX channel 时默认关闭，同机 NIXL 保持走 NVLink，不会悄悄退到 TCP |
| RecoverSSM for the KDA layers | decode 的 KDA 层不再为每个投机 token 存状态，每个请求一个状态槽，D 的 KV 容量 +10% |
| CUDA RecoverSSM kernels（gcanlin/vllm#12 的 JIT 版） | float32 状态的 verify / commit，比 Triton 快 2.8 倍 / 1.4 倍 |
| Run the sampler warm-up eager | main 上 #62560 的 sampler 预热在 MTP + DP 时让 prefill 启动失败（`reusing a DP sync taken over a different batch`），改成和 V1 一样 eager |

## 注意

- 两个容器的内存上限不能省。没有上限时，host 缓存加 NUMA 严格绑定曾把节点的系统进程 OOM 掉。
- prefill、decode 都不要用 `--numa-bind-nodes` 严格绑 NUMA；内存超出一个 NUMA 节点时会直接 OOM，不会溢出到另一个节点。
- 不要在 prefill 上打开 `kv-transfer-config` 的 `enable_early_recv`：GLM 的 prefill 上它会把同一次 Mooncake 加载报成完成两次，
  scheduler 断言退出。
- extensible KV 关掉：B200 节点没有 IMEX channel，开了以后同机 NIXL 会退到 TCP（#65137 在这种情况下默认就关）；
  有 channel 时 decode 也只多 1.3% KV。
- `kv_lease_duration` 设 300 s，默认 30 s 时 prefill 可能在 decode 读走之前释放 KV。
- 状态精度用 float32；bf16 状态有精度风险。CuTe kernel 只支持 bf16 状态，所以 decode 用本分支的 CUDA kernel。
- MegaMoE 要求 288 个专家能被 EP 整除，单个引擎只能 4/4、6/2 这样切。

## 实测（2026-11-03）

AgentX（aiperf，InferenceX 参数）并发 384，正式测量 1 小时，压测时投机解码用合成接受长度 3.61：

| 指标 | 数值 |
| --- | --- |
| 每卡输出（8 卡） | 1,569 tok/s（总 12,552 tok/s） |
| 每卡总吞吐 | 204,280 tok/s |
| TTFT avg / p50 / p90 / p99 | 8.89 / 6.35 / 18.82 / 37.35 s |
| ITL avg | 9.20 ms |
| 每用户 tok/s p10 / p50 | 100 / 107 |
| 请求数 / 错误 / 抢占 | 44,820 / 0 / 0 |
| prefix 命中 | 93.9%（GPU 23.6%，Mooncake 70.3%） |

真实 MTP 下 GSM8K 5-shot 0.9719，16 段约 2 万 token 上下文的读回 16/16。这组数据是在 nightly `e0d00e4b8e` 加上表里前四个改动时
测的；本分支（`40941f2c4b`）在两台机器的 2P1D 上测过 GSM8K 0.9742、读回 48/48。
