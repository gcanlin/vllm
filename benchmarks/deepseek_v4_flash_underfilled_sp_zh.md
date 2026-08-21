# 在 8×B200 上优化 DeepSeek V4 Flash 的小批量序列并行通信

本文记录一次面向 DeepSeek V4 Flash 解码路径的 vLLM 优化。改动很小：当序列并行（SP）的 token 数不足 TP world size 的一半时，用只处理有效行的 all-reduce 代替“补零后 reduce-scatter”。在 8×NVIDIA B200、TP8+EP 的实测中，batch 1/2/4 的输出吞吐分别提升 **1.78% / 0.92% / 1.92%**，mean TPOT 分别下降 **2.00% / 1.98% / 2.20%**；阈值外的 C64 吞吐变化为 +0.05%，属于噪声。GSM8K 5-shot 的准确率保持在 1235/1319（93.6315%）。

## 背景与定位过程

这次工作的目标不是为模型增加特殊调度或绕过通用执行路径，而是消除一个可以证明没有必要的通信开销。研究时主要参考了以下几类工作：

- [PR #53040](https://github.com/vllm-project/vllm/pull/53040) 把 shared experts 纳入 DeepGEMM MegaMoE，说明框架应尽可能完整地利用已有算子能力。
- [PR #52079](https://github.com/vllm-project/vllm/pull/52079) 使用 CUTLASS GEMM-RS 优化通信与计算，是融合通信原语的代表。
- TensorRT-LLM 的 DeepSeek V4 路径包含 fused Q RMSNorm/RoPE/FP8、adaptive top-k、多流 indexer 等 Blackwell 优化；SGLang、TokenSpeed、FlashInfer、CUTLASS 和 DeepGEMM 的本地实现也用于交叉检查 vLLM 已有能力和缺口。

GPU trace 显示，当前 DeepSeek V4 Flash 的 B1 decode 已经大量使用融合 kernel。每层约 44 us 的 MegaMoE 仍是主项，而注意力后的 MNNVL Lamport reduce-scatter 约为 22.35 us。模型有 43 层，因此这个 collective 即使只缩短少量时间，也会在每个 decode step 中重复累积。

问题出现在 token 数小于 TP size 时。以 TP8、B1 为例，原路径把一行 `[1, 4096]` 补成 `[8, 4096]`，再执行 reduce-scatter；其中七行在每个 rank 上都是零。这里的补零是为了让每个 SP rank 获得固定的一行输出，但不代表这些零必须进入 collective。

## 等价变换

令 TP size 为 `P`，有效 token 数为 `M < P`，第 `r` 个 TP rank 的注意力部分结果为：

```text
X_r ∈ R^(M×H)
Y   = sum(X_r), r = 0 ... P-1
```

MoE 的 SP 输入要求 rank `q` 保存 `Y[q:q+1]`；当 `q >= M` 时保存一行零。两条路径分别是：

```text
原路径: pad X_r to P rows -> reduce-scatter -> rank q gets Y[q] or zero
新路径: all-reduce M real rows -> local slice   -> rank q gets Y[q] or zero
```

二者在代数上相同。浮点 collective 的归约顺序可能不同，所以不声称 bitwise 等价；后文用完整 GSM8K 测试验证任务精度。

在常见 ring 成本模型中，补到 `P` 行的 reduce-scatter 每 rank 传输量与 `(P-1)H` 成正比，而 `M` 行 all-reduce 与 `2(P-1)MH/P` 成正比。因此 `2M <= P` 是一个保守边界：all-reduce 不会因为 payload 本身比 reduce-scatter 更大。实现只在 `0 < M <= P//2` 时切换，其余 shape 保留原 custom reduce-scatter 路径。

## 实现

通用的 `sp_reduce_scatter` 新增一个默认关闭的 keyword 参数。启用且满足阈值时：

1. 对未补零的 `[M, H]` 执行 TP all-reduce；
2. `rank < M` 时返回该 rank 对应的一行；
3. 其余 rank 返回同 shape 的零；
4. `M > P//2` 时继续使用原有 custom collective 或标准 reduce-scatter。

只有 NVIDIA DeepSeek V4 decoder 显式启用该选项，因此 Kimi K3、DeepSeek V3.2 和其他平台的默认行为不变。使用的是 vLLM 现有 TP all-reduce dispatcher，能继续选择当前平台支持的通信后端，并可被 CUDA graph capture。

## 性能测试

### 环境

- GPU：8×NVIDIA B200 183 GB，driver 590.48.01
- PyTorch：2.13.0+cu130；CUDA 13.0；FlashInfer 0.6.17
- vLLM 基线：`ba53da60bb1aeec200d05101936a5474ee46c4eb`
- 模型：`DeepSeek-V4-Flash-0731`，43 layers，TP8 + EP
- MoE：`deep_gemm_mega_moe`
- KV cache：FP8；indexer KV：MXFP4；block size：256
- 每组正式结果取三个 seed 的中位数。C64 在正式样本前额外运行一次并丢弃，用于覆盖该 shape 的 TileLang JIT warmup。

### 结果

输入/输出长度在 B1、B2、B4 均为 1024/512；C64 为 1024/128。吞吐量单位为 output tok/s，TPOT 单位为 ms。

| 场景 | 主线吞吐（3 次原始值） | 优化吞吐（3 次原始值） | 中位数变化 | 主线 mean TPOT | 优化 mean TPOT | TPOT 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1/C1 | 155.129, 155.144, 155.077 | 157.971, 157.885, 157.888 | **+1.779%** | 6.2848 | 6.1593 | **-1.996%** |
| B2/C2 | 286.599, 287.138, 287.395 | 289.629, 289.778, 290.116 | **+0.919%** | 6.7000 | 6.5676 | **-1.977%** |
| B4/C4 | 566.258, 556.037, 568.675 | 577.144, 576.971, 579.390 | **+1.922%** | 6.8579 | 6.7069 | **-2.202%** |
| 128 prompts/C64 | 3890.731, 3816.677, 3819.756 | 3837.438, 3821.749, 3800.706 | +0.052%（噪声） | 12.6886 | 12.6814 | -0.057%（噪声） |

B1 的另一轮最终复核得到 157.05 tok/s，仍高于主线三次结果。TTFT 包含 prefill、请求排队和小样本启动抖动，不是本优化的目标指标，因此没有据其宣称收益。B1/B2/B4 的 mean TPOT 约 2% 改善更直接地反映了 decode 路径变化。

C64 主要落在阈值外，继续使用原 reduce-scatter；+0.05% 验证的是没有可测回退，而不是额外收益。

### 精度

使用仓库自带 `tests/evals/gsm8k/gsm8k_eval.py`，完整 1319 题、5-shot、temperature 0、seed 42、max tokens 256、并发 64，主线和优化使用完全相同的服务参数。

| 版本 | 正确数 | Accuracy | Invalid rate |
| --- | ---: | ---: | ---: |
| 主线 | 1235/1319 | 0.9363153904473086 | 0.0 |
| 优化 | 1235/1319 | 0.9363153904473086 | 0.0 |

collective 归约顺序改变后，生成文本不保证逐 token 相同；这次任务级准确率和 invalid rate 均无变化。

## 复现命令

服务端的关键参数如下（主线和优化分支分别重启服务）：

```bash
vllm serve /mnt/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name dsv4 \
  --host 127.0.0.1 --port 8100 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 --block-size 256 \
  --attention-config.indexer_kv_dtype=mxfp4 \
  --max-model-len 16384 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 --seed 2026 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256],"max_cudagraph_capture_size":256}'
```

B1 示例：

```bash
vllm bench serve --backend vllm \
  --base-url http://127.0.0.1:8100 --endpoint /v1/completions \
  --model dsv4 \
  --tokenizer /mnt/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --dataset-name random --random-input-len 1024 \
  --random-output-len 512 --random-range-ratio 0 \
  --num-prompts 4 --max-concurrency 1 --ignore-eos
```

精度测试：

```bash
python tests/evals/gsm8k/gsm8k_eval.py \
  --host http://127.0.0.1 --port 8100 \
  --num-questions 1319 --num-shots 5 --max-tokens 256 \
  --temperature 0 --seed 42 --max-concurrency 64
```

## 其他尝试

在找到本优化前还检查了两个方向，它们都没有进入最终代码：

1. **短上下文 indexer scoring 快路径。** 做了三轮实现，尝试在 CUDA graph 外跳过无效 slot 的 score/top-k 或复用 buffer。B1 和均衡负载均为持平或负向。原因是昂贵的 Q projection/compressor 已在多流 capture 中，剩余 kernel 很短，而额外 eager fill/同步抵消了节省。
2. **global top-k 的 local-to-paged slot 转换调优。** 对 512/1024 threads 的微基准约为 14.6–14.9 us，表现为 launch bound；单纯改 block size 没有收益。进一步的 kernel fusion 已有 [PR #41105](https://github.com/vllm-project/vllm/pull/41105)，因此没有重复实现。
3. **underfilled SP collective。** profile 指向明确、可证明等价，并在 B1–B4 得到超出噪声的稳定收益，因此成为最终方案。

本次在第三个方向获得有效优化，没有触发“四个方向都无非噪声收益后停止”的条件。

## 限制与后续

- 当前数据来自单机 8×B200 NVLink 域；其他拓扑、TP size 和通信后端需要单独测量。
- 当前阈值来自通信量上界和 TP8 实测，尚未引入按后端或按拓扑的 autotune。
- 只在 NVIDIA DeepSeek V4 路径启用，便于控制风险。若后续在其他 SP 模型上重复验证，可将策略提升为通用启发式。
