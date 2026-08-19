# DeepSeek V4 NVIDIA SM100 Shared Expert 原生融合优化报告

## 结论

本次优化已经在 8 张 NVIDIA B200 上用完整的
DeepSeek-V4-Flash-0731 模型验证成功，并且属于 DeepSeek V4 的
`vllm/models/deepseek_v4/nvidia/` 专有执行路径，不是通用 MoE 路径切换或参数微调。

核心改动是把原先串行执行的 FP8 shared expert 下沉到 DeepGEMM 的 SM100
persistent MegaMoE kernel 内，与 FP4 routed experts 在同一个原生调度器中执行并融合累加。
最终结果如下：

- 1K 输入 / 128 输出的均衡负载：输出吞吐提升 **9.44%**。
- 8K 输入 / 32 输出的 prefill-heavy 负载：输出吞吐提升 **6.55%**。
- GSM8K 固定 200 题：83.0% -> 83.5%，没有可测量的精度回退。
- 模型加载显存仅增加 0.03 GiB/卡，KV 容量降低 0.10%。
- FULL_AND_PIECEWISE CUDA Graph 的 1 到 256 batch capture 均成功。

这是第一次优化尝试即成功，因此没有触发“三次无效后停止”的条件。

## 1. 调研范围与版本

调研和实现基于 2026-08-20 的本地代码：

- vLLM 基线：`583a00257d4c5d1a54063d956057df1df6822b06`
- TokenSpeed：`978ed2cfdc870cabab20289c46329f5b744aaec2`
- SGLang：`5f128395910dafb98c34083dc26cb790c7674d34`
- vLLM 当前固定的 DeepGEMM：
  `8b1392b978f5a03c828dd1711090d7fb50958b8a`
- 模型：`/mnt/models/deepseek-ai/DeepSeek-V4-Flash-0731`
- 硬件：8x NVIDIA B200，driver 590.48.01，CUDA 13.2
- 软件：PyTorch 2.13.0+cu130，FlashInfer 0.6.17

模型有 43 层、256 个 routed experts、top-k 6、1 个 shared expert；routed
expert 为 FP4，shared expert 和其他线性层为 block-FP8。生产测试采用 TP8+EP8
和 sequence parallel。

## 2. 原有瓶颈

vLLM 的 NVIDIA DeepSeek V4 路径已经使用 DeepGEMM 的 SM100 persistent
MegaMoE kernel 计算 routed experts，但每层 MoE 的执行关系仍然是：

```text
hidden states
  -> routed FP4 persistent MegaMoE
  -> shared FP8 gate/up GEMM
  -> shared activation
  -> shared FP8 down GEMM
  -> routed + shared add
```

shared expert 是每个 token 都执行的固定计算。它位于 routed kernel 之后，形成每个
MoE 层都必须支付的串行尾部；模型有大量 MoE 层，因此这不是单次小 kernel 的问题，
而是会累积到端到端延迟和吞吐的结构性空洞。

## 3. TokenSpeed 与 SGLang 的做法

### 3.1 TokenSpeed：双 CUDA stream 重叠

TokenSpeed 在提交
[`4e51bed`](https://github.com/lightseekorg/tokenspeed/commit/4e51bed840911e483d8ddf5f24d61061cf60bc4d)
中让 routed 和 shared 两条 MoE 分支在不同 CUDA stream 上并行，再同步结果。

这个方案能隐藏部分 shared 计算，但仍然保留两套 kernel pipeline、stream
同步和最终加法；当 routed 与 shared 同时竞争 SM、Tensor Core 和带宽时，也不保证
两者能够理想重叠。

### 3.2 SGLang Waterfill：把 shared expert 变成额外 routed expert

SGLang 在提交
[`eb31b53`](https://github.com/sgl-project/sglang/commit/eb31b5310c8bf076f5ac9624269697e299d0865f)
中支持 MegaMoE Waterfill。它把 shared expert 作为额外 expert 放入调度和负载均衡；
对应的 checkpoint 处理路径可以把 shared 的 FP8 权重转换为 FP4。

这个做法把工作纳入统一调度，但 FP8 shared -> FP4 会改变 shared 分支的权重量化
精度，并引入转换后的权重表示。

## 4. 选择的优化：SM100 原生 shared-expert MegaMoE fusion

vLLM 已固定的 DeepGEMM 版本已经具有 SM100 MegaMoE native shared-expert
能力，但 vLLM 的 DeepSeek V4 NVIDIA 路径没有把 shared 权重、激活 scale 和调用参数
接入该能力。本次实现补齐了这条完整链路。

优化后的执行关系是：

```text
                              +-> shared FP8 L1 -> activation -> shared FP8 L2 --+
hidden states -> one SM100 persistent scheduler                              FP32 sum
                              +-> dispatch -> routed FP4 L1/L2 -> top-k weight --+
                                                     -> one BF16 output store
```

相比 stream overlap，它让 DeepGEMM 自己在 persistent scheduler 内协调两类工作，
不再执行独立的 shared MLP pipeline 和最后的 PyTorch add。相比 Waterfill 的一种
实现方式，它保留 shared expert 的 FP8 权重，不把权重重新量化成 FP4。

## 5. 实现细节

### 5.1 shared FP8 权重 scale 转换

模型 checkpoint 中 shared expert 的 scale 是 UE8M0 128x128 block scale；
DeepGEMM native shared MMA 需要每一行、每 32 列一个 scale，即 1x32 粒度。

实现会把 checkpoint scale 在行方向和 K 方向复制展开，得到数学等价的 1x32
scale，而不是重新估计或重新量化权重。随后通过 DeepGEMM 的布局转换接口打包成
TMA 需要的 stride/layout，再将 gate/up 和 down 权重转换成 native MegaMoE 格式。

转换发生在模型的 post-load hook。vLLM 的 DeepSeek V4 loader 会在通用 FP8 linear
post-process 之前调用自己的 hook，因此能够读取 checkpoint 的原始 scale 形状。

### 5.2 shared activation scale 的动态 TMA 布局

这是接入中最容易出错、也最关键的一部分。

routed FP4 和 shared FP8 虽然使用同一份量化激活，但 scale buffer 的内存排列不同。
shared L1 的 TMA row permutation 又依赖 DeepGEMM 根据 token 数动态选择的
`BLOCK_M`。只把 routed 使用的 `x_sf` 传入 native kernel 会产生严重错误输出；固定
一种 `BLOCK_M` 的转置也无法覆盖 CUDA Graph 和动态 batching。

实现先查询当前 MegaMoE 调度选择的 `BLOCK_M`，然后在现有
`prepare_megamoe_inputs` Triton kernel 中，同时写出 shared L1 所需的 MN-major/TMA
视图：

```text
aligned_block_m = ceil(BLOCK_M / 128) * 128
transposed_m = floor(m / 128) * 128 + (m mod 32) * 4 + floor((m mod 128) / 32)
```

此时 packed UE8M0 scale 已经在寄存器中，因此只增加必要的 store，不需要新 kernel、
临时 tensor 或额外量化过程。测试覆盖 `BLOCK_M = 8, 32, 96, 128, 192`，并覆盖跨
block 的 token 范围。

### 5.3 融合输出与防止重复加法

native DeepGEMM kernel 在 FP32 中组合 routed 的 top-k weighted output 和不加权的
shared output，最后只写一次 BF16。vLLM 在 `has_fused_shared_experts` 为真时跳过原来的
`shared_experts(hidden_states)` 和 `final_hidden_states += shared_output`，避免重复计算和
重复加法。单元测试同时覆盖 fused 和 fallback 两个分支。

### 5.4 显存生命周期优化

DeepGEMM 对 shared gate/up 做 interleave 时会创建完整副本。直接保留原 loader
Parameter 会让 DSV4-Flash 每卡额外占用约 0.70 GiB。

实现将 loader Parameter 的 storage 重新绑定到 interleaved tensor。forward 已经不会
再调用通用 shared MLP，因此原 gate/up storage 可以安全释放，同时 Parameter 仍然拥有
native tensor，避免悬空引用。最终模型加载显存从 baseline 的 20.73 GiB 变为
20.76 GiB，而不是最初实现的约 21.43 GiB。

### 5.5 旧 wheel、并行模式与紧急回滚

用户环境原有 0.27.0 wheel 中的 `_deep_gemm_C` 比当前源码旧。实现不假设 wheel 与
源码同步，而是在建立 symmetric buffer 前检查以下 native API 能力：

- symmetric buffer 的 `num_shared_experts` 参数；
- MegaMoE kernel 的 shared L1/L2 参数；
- 动态 `BLOCK_M` 查询；
- native weight transform。

如果 API 过旧，程序会给出明确 rebuild 提示并自动保留串行 shared 路径，不会在多 rank
symmetric-memory 初始化一半时失败。完整优化需要重建当前 vLLM pin 对应的 vendored
DeepGEMM extension；本次实测已完成该重建，但没有修改源码 pin。

并行支持矩阵：

| 模式 | native fusion | 原因 |
| --- | --- | --- |
| TP8+EP8 + sequence parallel | 开启 | shared 权重复制，token 本地分片 |
| TP1 | 开启 | shared 权重天然完整 |
| 非 SP 的 PP+TP | 自动回退 | shared MLP 仍是 TP shard，不能直接交给完整 native shared MMA |
| 旧 DeepGEMM API | 自动回退 | 缺少 native shared 接口 |
| 其他 MoE backend | 不受影响 | 改动仅在 NVIDIA DSV4 MegaMoE 路径 |

紧急回滚开关：

```bash
VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1
```

设置后无需修改 `--moe-backend`，即可恢复原来的 routed MegaMoE + 串行 shared MLP。

## 6. 性能基准

### 6.1 服务配置

```bash
vllm serve /mnt/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name dsv4 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 --block-size 256 \
  --attention-config.indexer_kv_dtype=mxfp4 \
  --max-model-len 16384 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 --seed 2026 \
  --compilation-config \
  '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256],"max_cudagraph_capture_size":256}'
```

baseline 增加 rollback 环境变量，after 使用默认开启的 native fusion。每组数据都是
随机种子 2027、2028、2029 三次的算术平均；before/after 使用配对种子。每次使用不同
随机 prompt，避免 prefix cache 污染。首轮冷启动和重复同种子的缓存命中结果没有纳入
统计。

### 6.2 均衡负载

参数：128 个请求，精确 1024 input tokens、128 output tokens，最大并发 64。

| 指标 | Before | After | 变化 |
| --- | ---: | ---: | ---: |
| Output throughput | 3,415.45 tok/s | 3,737.98 tok/s | **+9.44%** |
| Total token throughput | 30,739.04 tok/s | 33,641.81 tok/s | **+9.44%** |
| Mean TTFT | 588.52 ms | 530.15 ms | **-9.92%** |
| Mean TPOT | 14.095 ms | 12.935 ms | **-8.23%** |
| Median ITL | 9.644 ms | 8.939 ms | **-7.31%** |

三个配对种子的吞吐提升分别为 +11.38%、+11.24%、+5.80%，没有负样本。

### 6.3 Prefill-heavy 负载

参数：32 个请求，精确 8192 input tokens、32 output tokens，最大并发 16。

| 指标 | Before | After | 变化 |
| --- | ---: | ---: | ---: |
| Output throughput | 234.09 tok/s | 249.42 tok/s | **+6.55%** |
| Total token throughput | 60,160.73 tok/s | 64,099.90 tok/s | **+6.55%** |
| Mean TTFT | 901.81 ms | 836.36 ms | **-7.26%** |
| Mean TPOT | 40.584 ms | 38.458 ms | **-5.24%** |
| Median ITL | 8.256 ms | 7.493 ms | **-9.24%** |

三个配对种子的吞吐提升分别为 +2.65%、+7.52%、+9.52%，同样没有负样本。

### 6.4 显存

| 每卡指标 | Before | After | 变化 |
| --- | ---: | ---: | ---: |
| 模型加载 | 20.73 GiB | 20.76 GiB | +0.03 GiB |
| runtime consumed | 23.04 GiB | 23.17 GiB | +0.13 GiB |
| KV cache 容量 | 386,897 tokens | 386,512 tokens | -385（-0.10%） |

## 7. 精度验证

使用官方 `openai/gsm8k` test split 的固定前 200 题，采用 DeepSeek V4 官方 prompt
encoding、low thinking、temperature 0、最大输出 512 tokens。baseline 与 after 的输入、
顺序和评分规则完全一致。

| 版本 | 正确数 | 准确率 | 无法解析 |
| --- | ---: | ---: | ---: |
| Before：串行 shared expert | 166 / 200 | 83.0% | 0 |
| After：native fused shared expert | 167 / 200 | 83.5% | 0 |

最终 after 与 baseline 的配对结果为：双方都正确 156 题、仅 baseline 正确 10 题、仅
after 正确 11 题、双方都错误 23 题。exact McNemar `p = 1.0`，没有发现统计上可测量的
精度回退。

两条路径不保证 token-by-token 完全相同。原通用 FP8 shared linear 使用 per-128
activation quant，native shared kernel 使用 per-32 activation quant；浮点舍入发生变化
是预期行为。权重 scale 的 128x128 -> 1x32 展开是数学等价的，没有把 FP8 权重重新
量化。任务级准确率是这里更有意义的生产验收指标。

## 8. 测试结果

最终执行：

```text
ruff format: passed
ruff check: passed
git diff --check: passed
pytest -q tests/models/test_deepseek_v4_mega_moe.py
17 passed, 14 warnings
```

新增测试覆盖：

- native shared 权重和 scale finalization；
- interleaved gate/up storage 的生命周期；
- fused 时不再执行/累加串行 shared MLP，fallback 时行为不变；
- routed staging 原有 bitwise exact 测试；
- 五种动态 `BLOCK_M` 的 shared TMA scale 布局和跨 block token；
- 真实 8 卡模型加载、推理、CUDA Graph capture、性能、显存和 GSM8K。

## 9. 生产使用建议与边界

这项优化适合当前测试的 DSV4 NVIDIA SM100 生产配置：FP4 routed experts、FP8
shared expert、DeepGEMM MegaMoE、EP 和 replicated shared weights。它不影响 DSpark、
通用 FusedMoE 或其他模型路径。

上线建议：

1. wheel 与源码一起构建，确保 vendored DeepGEMM extension 来自 vLLM 当前 pin。
2. 先在 canary 使用默认 native fusion，保留 rollback 环境变量。
3. 监控 TTFT、TPOT、output throughput 和模型任务级指标，而不是要求逐 token bitwise
   等价。
4. 如果部署拓扑从 sequence-parallel TP+EP 改为非 SP 的 tensor-sharded shared MLP，
   当前实现会自动回退；未来若要支持，可增加 shared weight all-gather 或原生 TP-aware
   shared MMA，但这不应混入本次已验证的生产改动。

本次优化没有更改 DeepGEMM pin，也没有通过切换 TP/EP 参数制造表面收益；收益来自
SM100 native persistent kernel 内部实际消除 shared expert 串行尾部。
