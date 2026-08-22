# DeepSeek V4 Flash：加速 FlashMLA 单请求解码元数据生成

## 背景

DeepSeek V4 Flash 的解码路径混合使用 SWA、C4A 和 C128A。vLLM 已经按注意力类型复用 FlashMLA 调度元数据，但在单请求、逐 token 解码时，每一步仍会调用 4 次
`smxx::decode::get_mla_metadata_kernel`。在 8 张 NVIDIA B200 上，每次调用平均耗时
28.378 微秒；4 次合计约占一次 6.148 毫秒 GPU decode step 的 1.85%。

原实现是面向任意 batch size 的通用调度器。它只启动一个 warp，并由 thread 0 串行遍历每个请求和约 148 个 SM partition。batch size 为 1 时，负载划分实际上可以直接计算，没有必要执行这段串行规划。

## 优化

本优化为 `batch_size == 1` 增加一个单独的 fast path：

1. 根据 dense sequence length，或者 sparse top-k、dynamic top-k length 与 extra cache length，计算当前 key 长度。
2. 直接计算 key block 数、每个 partition 的 block 数和实际使用的 partition 数。
3. 让一个 warp 的 32 个线程并行写入所有 partition metadata。
4. 保留原来的多请求调度器，`batch_size > 1` 的行为完全不变。

fast path 同时覆盖 dense MLA、普通 sparse MLA、带 extra cache 的 C4A/C128A，以及动态 top-k length。实现位于 FlashMLA 的
`csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`。

当前分支通过 CMake patch 把修改应用到 vLLM 固定的 FlashMLA commit
`a8f794d1251cbfd88a5011445dd5582289c727e4`，以便独立复现。正式合入应先向
[vllm-project/FlashMLA](https://github.com/vllm-project/FlashMLA) 提交 kernel 修改，再在 vLLM 中更新 commit pin；vLLM 不应长期维护这个 downstream patch。

## 性能结果

测试环境：8× NVIDIA B200，DeepSeek V4 Flash BF16/FP8/FP4，TP8 + EP，FlashMLA sparse decode，输入 1024 tokens、输出 128 tokens、concurrency 1。每组启动独立服务，预热 2 个请求后连续测试 5 轮，每轮 20 个请求。

### Kernel microbenchmark

| 实现 | `get_mla_metadata` 平均耗时 | 加速比 |
| --- | ---: | ---: |
| 原实现 | 28.378 µs | 1.00× |
| 单请求 fast path | 2.054 µs | **13.82×** |

DeepSeek V4 Flash 每个 decode step 调用该 kernel 4 次，因此每步减少约
105.3 微秒。

### vLLM 端到端

| 指标 | 原实现 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| 平均 TPOT | 6.6786 ms | 6.5819 ms | **-1.47%** |
| Output throughput | 133.61 tok/s | 134.46 tok/s | **+0.64%** |

五轮 TPOT 原始数据（毫秒）：

- 原实现：6.67917、6.68016、6.67884、6.67802、6.67673
- 优化后：6.58339、6.58168、6.58242、6.58051、6.58171

平均节省 96.64 微秒/token，配对差值的 95% 置信区间为
[94.93, 98.35] 微秒，因此提升显著高于测试噪声。Output throughput 包含 prefill，提升幅度比 TPOT 小是预期结果。TTFT 容易受到跨服务启动状态影响，本次不把它作为优化结论。

## 精度验证

FlashMLA 官方 PyTorch reference harness 的 5 组单请求用例全部通过：

- DeepSeek V4 head 64：SWA、C4A、C128A，包含动态 top-k/extra length；
- DeepSeek V4 head 128：C4A；
- DeepSeek V3.2 head 64、head dimension 576。

此外，固定 GSM8K 前 200 题、temperature 0、seed 2026 的结果如下：

| 实现 | 正确数 | 准确率 | 无法解析 |
| --- | ---: | ---: | ---: |
| 原实现 | 168/200 | 84.0% | 0 |
| 优化后 | 173/200 | 86.5% | 0 |

推理框架的动态 batching 会使跨服务输出存在少量非确定性，因此这组结果只能说明没有观察到精度退化，不能解读为模型精度得到提升。数值等价性的主要依据是 FlashMLA reference harness。

## 调研中的其他方向

这次还验证了 W4A4 MegaGEMM、MegaMoE SM 配额、低 M quant GEMM、C128A metadata 和 GEMM reduce-scatter 等方向。其中有的在相同模型与 B200 上没有收益或存在精度风险，有的端到端收益落在噪声范围。profile 最终表明，单请求 FlashMLA metadata 是一个小而稳定、且不改变主计算路径的瓶颈。

相关背景可参考 [vLLM DeepSeek V4 性能跟踪](https://github.com/vllm-project/vllm/issues/45861)。
