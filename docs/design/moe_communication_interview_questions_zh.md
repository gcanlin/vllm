# 从 Qwen MoE Layer 出发的通信面试题

本文围绕一段精简后的 Qwen3.5/Qwen3-Next decoder layer 展开。面试官先给候选人
代码，再从 TP/EP 基础逐步追问到通信、layout、正确性和 profiling，最后要求候选人
完成 `AllReduce + SP chunk → ReduceScatter` 的 coding 优化。

参考答案供面试官使用，不建议和题目同时展示。

## 面试代码

代码保留了与问题相关的真实结构，删除了 QKV 细节、量化、layer scale、shared
expert 和 EP dispatch/combine 的具体实现。

```python
class Qwen3NextAttention(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.attn = Attention(config)
        self.o_proj = RowParallelLinear(
            config.attention_output_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(hidden_states, positions)
        attention_output = self.attn(q, k, v)
        output, _ = self.o_proj(attention_output)
        return output


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.is_sequence_parallel = config.use_sequence_parallel_moe
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts)
        self.experts = FusedMoE(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        if self.is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(
                hidden_states,
                dim=0,
            )
            hidden_states = hidden_states[:num_tokens]

        return hidden_states.view(original_shape)


class Qwen3NextDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
            )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

Qwen3.5 的 GDN linear-attention layer 虽然 attention 实现不同，但 output projection
同样使用 `RowParallelLinear`，可以按相同契约分析。

## 问题一：解释 TP 和 EP 的基本原理

### 问题

结合这段 Qwen layer，解释 Tensor Parallel 和 Expert Parallel 分别切分什么、为什么
需要通信，以及二者组合时 TP group 和 EP group 的职责有什么不同。

### 参考答案

TP 切分单个 dense operator 的参数和计算。以 linear 为例：

- Column Parallel Linear 按输出维切权重，每个 rank 产生不同的输出 channel；
- Row Parallel Linear 按输入维切权重，每个 rank 产生同一输出 tensor 的 partial
  sum，通常需要 AllReduce 得到完整结果；
- attention 的 heads、QKV projection 和 output projection 都可以按 TP 切分。

代码中的 `o_proj` 是 Row Parallel Linear。设第 `i` 个 TP rank 持有权重分片
`W_i` 和对应输入分片 `X_i`，则该 rank 只计算：

```text
Y_i = X_i W_i
Y = sum_i(Y_i)
```

默认的 `reduce_results=True` 会在 `RowParallelLinear` 内部执行 TP AllReduce，使每个
TP rank 都得到完整的 `Y`。

EP 切分的是 experts。假设有 256 个 experts、EP size 为 4，每个 rank 通常只持有
其中 64 个。Router 为每个 token 选择 top-k experts 后，token 必须被发送到目标
expert 所在 rank；计算结束后，再把多个 expert contribution 合并并送回 token owner。

二者的区别可以概括为：

| 并行方式 | 切分对象 | 典型通信 | 主要目的 |
| --- | --- | --- | --- |
| TP | 单个算子的权重/hidden dimension | AllReduce、AllGather、ReduceScatter | 让多卡共同完成一个大算子 |
| EP | experts | Dispatch、All-to-All、Combine | 让不同 rank 承载不同 experts |

TP group 负责合并同一算子的 partial results；EP group 负责按照 routing 结果移动
token。某些配置中两个 group 的 rank 集合可能相同，但通信语义仍然不同。

### 可选追问

如果 TP group 和 EP group 恰好都是同四张 GPU，能否认为 TP collective 和 EP
collective 可以随意合并？

不能。Group 成员相同不代表 tensor layout、reduction 语义和执行位置相同。是否能
融合取决于两次通信之间的数据依赖和算子是否可交换。

## 问题二：MoE 通信后端有哪些，区别是什么？

### 问题

vLLM 中 MoE 的 dispatch/combine 可以使用哪些通信后端？它们的实现思路、适用场景
和主要 trade-off 是什么？

### 参考答案

当前代码中的后端可以按实现思路分为以下几类：

| 后端 | 核心思路 | 适用场景与 trade-off |
| --- | --- | --- |
| `allgather_reducescatter` | Dispatch 用 AllGather/AllGatherv，Combine 用 ReduceScatter/ReduceScatterv | 默认通用路径，依赖少、容易调试；会临时复制所有 token，通信量和显存流量较大 |
| `deepep_high_throughput` | DeepEP 高吞吐 dispatch/combine kernel | 更偏大 token 数、prefill 或吞吐场景；需要 DeepEP，并占用一定通信资源 |
| `deepep_low_latency` | DeepEP 低延迟 kernel，可使用 RDMA/NVLink | 更偏 decode 和小消息延迟；需要预分配 buffer，并受网络与部署能力约束 |
| `deepep_v2` | DeepEP ElasticBuffer 统一接口，支持 hybrid/overlap 配置 | 面向更新的 DeepEP/NCCL 环境，可统一多种模式；依赖版本和部署要求更高 |
| `flashinfer_nvlink_two_sided` | FlashInfer two-sided NVLink MoE All-to-All | 适合受支持的 NVIDIA NVLink/MNNVL 拓扑；需要相应 workspace 和 FlashInfer kernel |
| `flashinfer_nvlink_one_sided` | FlashInfer/TRT-LLM one-sided NVLink kernel | 更新的单边通信实现，目标是降低同步和 dispatch/combine 开销；对 topology、top-k、expert 配置有约束 |
| `nixl_ep` | NIXL EP kernel 和动态连接管理 | 支持 elastic EP、rank 动态连接/断开等场景；需要 NIXL/RIXL 运行环境 |
| `mori_high_throughput` / `mori_low_latency` | MoRI dispatch/combine kernel | 面向受支持的 AMD ROCm 架构；分别针对吞吐和低延迟路径 |

`naive` 和 `pplx` 仍出现在类型定义中，但当前配置校验会将它们视为已移除的旧选项，
回退到 `allgather_reducescatter`，不应作为新部署方案。

High-throughput 和 low-latency 不是绝对的“一个更快”。选择后端时至少需要考虑：

- prefill 还是 decode，以及每个 rank 的 token 数；
- 单机 NVLink、MNNVL 还是跨机 RDMA；
- hidden size、top-k、expert 数和 dispatch dtype；
- 通信 kernel 占用的 SM 是否会挤压 expert GEMM；
- 是否支持 CUDA Graph、quantization、fault tolerance 和 elastic EP；
- 安装依赖、workspace 显存和运维复杂度。

合理的回答不应只背后端名字，而应把它们归纳为“全量 gather 的通用实现”和“根据
routing 做定向 token dispatch 的专用实现”，并说明 workload 与硬件拓扑决定结果。

## 问题三：描述一次 MoE forward 的完整过程

### 问题

从 `[num_tokens, hidden_size]` 的 hidden states 开始，描述一次带 top-k routing 和
shared expert 的 MoE forward。指出哪些步骤是本地计算，哪些步骤可能发生 EP 通信。

### 参考答案

一个典型 MoE forward 包含：

1. **Router/Gate**：对每个 token 计算所有 experts 的 logits；
2. **Top-k selection**：选择 top-k experts，并得到 expert ids 和 routing weights；
3. **Dispatch**：按 expert id 对 token 排序、打包，并发送到 expert 所在 EP rank；
4. **Expert compute**：每个 rank 对收到的 token 运行本地 experts 的 gate/up/down
   projections；
5. **Weighted combine**：根据 routing weights 合并同一 token 的多个 expert 输出；
6. **Return/Combine communication**：把 contribution 送回 token owner，并完成必要的
   reduction；
7. **Shared expert**：所有 token 额外经过一个或多个共享 expert，其输出与 routed
   expert 输出相加。

在 `allgather_reducescatter` 后端中，流程具体表现为：

```text
local token shard
→ router，并根据具体实现选择在 gather 前或 gather 后执行 top-k
→ EP AllGather hidden states 和 router logits/top-k metadata
→ local routed experts
→ weighted combine
→ EP ReduceScatter
→ 返回 local token shard
```

Shared expert 不需要根据 routing 把 token 发往不同 expert rank。它通常可以在辅助
CUDA stream 上与 routed expert 的 router、dispatch 或计算重叠，最后再同步并相加。

### 可选追问

为什么 expert load imbalance 会影响性能？

不同 token 可能集中选择少数热门 experts，使某些 rank 的 token 数和 GEMM 工作量
更大。Collective 通常需要等待最慢 rank，最终 step latency 由 straggler 决定。常见
缓解方法包括合理的 expert placement、EPLB、冗余 expert、动态重排，以及训练阶段的
load-balancing loss。

## 问题四：顺着代码还原 tensor layout 和隐藏通信

### 问题

不修改代码，描述一个 hidden state 从 attention output projection 到 MoE 输出经历的
layout 变化和 collective。哪些通信没有直接写在 decoder layer 的 `forward` 中？

### 参考答案

完整路径是：

```text
attention output projection partial result
→ RowParallelLinear 内部 TP AllReduce
→ replicated attention output [T, H]
→ residual add + RMSNorm，仍为 replicated [T, H]
→ sequence_parallel_chunk
→ local SP shard [ceil(T / TP), H]
→ router + FusedMoE
→ EP dispatch/combine
→ local SP shard
→ TP AllGather
→ replicated MoE output [T, H]
```

隐藏的通信至少有三处：

- `Qwen3NextAttention.o_proj` 的 `RowParallelLinear` 默认在内部执行 TP AllReduce；
- `FusedMoE` 内部根据 EP backend 执行 dispatch/combine；
- `Qwen3NextSparseMoeBlock` 在出口显式执行 TP AllGather。

这道题考察候选人能否跨模块理解数据流，而不是只看 decoder layer 表面上有没有
collective 调用。

## 问题五：为什么 MoE 要使用 Sequence Parallel？

### 问题

如果 attention AllReduce 后每个 TP rank 已经拥有相同的完整 hidden states，为什么
进入 EP MoE 前还要执行 `sequence_parallel_chunk`？FusedMoE 前后的 AllGather 和
ReduceScatter 分别是什么？

### 参考答案

当 TP 和 EP 组合时，如果每个 TP rank 都把相同的完整 token 集合送进 router 和 MoE，
会产生重复 routing、重复 dispatch 和重复 expert computation。SP 让每个 rank 只拥有
一段 token，使 token owner 唯一，从而避免把相同 token 重复送入 EP 路径。

需要区分三个 collective：

| Collective | 位置 | 语义 |
| --- | --- | --- |
| TP AllReduce | attention output projection | 合并不同 TP rank 的 output partial sums |
| EP AllGather | FusedMoE dispatch | 收集不同 token shards 和 routing metadata，使本地 experts 能收到目标 token |
| EP ReduceScatter | FusedMoE combine | 合并 expert contributions，并把结果返回 token owner |
| TP AllGather | MoE block 出口 | 恢复下一层所期望的完整 token layout |

EP AllGather 收集的是各 rank 的不同 token shard，不是重复收集相同 tensor。它与
attention TP AllReduce 的 reduction 语义不同，即使使用同一组 GPU 也不能混为一谈。

专用 All-to-All backend 会把全量 EP AllGather 换成定向 token dispatch，但 SP 的
“每个 token 有明确 owner”这一 layout 仍然重要。

## 问题六：Residual、RMSNorm 和 layout 有哪些正确性约束？

### 问题

如果希望让 MoE 直接接收 token shard，residual 和
`post_attention_layernorm(hidden_states, residual)` 应如何处理？哪些看似可行的重排
实际上会改变数值？

### 参考答案

`post_attention_layernorm` 的核心语义是：

```text
residual_out = residual + attention_output
moe_input = RMSNorm(residual_out)
```

如果 `attention_output` 已经是 SP shard，`residual` 必须使用完全相同的 token padding
和分片规则。二者 shape 相同还不够，它们必须表示相同的 token 范围。

RMSNorm 沿 hidden dimension 独立处理每个 token，因此 token slicing 可以与它交换：

```text
chunk(RMSNorm(residual + reduced_attention))
    == RMSNorm(chunk(residual) + chunk(reduced_attention))
```

但 RMSNorm 不能移动到 TP reduction 之前：

```text
RMSNorm(sum_i(partial_i)) != sum_i(RMSNorm(partial_i))
```

还需要满足：

- token 数不能被 TP size 整除时，各 rank 使用相同的零 padding；
- 恢复 full layout 时移除 padding；
- 显式维护 `is_sequence_parallel`，不能根据 shape 猜测 layout；
- MoE 已收到 shard 时不能再次 chunk；
- final norm、logits、aux hidden states 和 PP boundary 必须恢复其接口约定的 layout。

单 token decode 是一个重要反例：padding 后每个 rank 的 shard 可能仍是 `[1, H]`，
与原始 tensor shape 相同，因此 shape 无法证明 tensor 是 full layout 还是 SP layout。

## 问题七：如何用 profiling 区分 TP 和 EP 通信？

### 问题

PyTorch Profiler 中同时出现多个 AllGather、AllReduce 和 ReduceScatter。如何判断某个
kernel 属于 attention TP、MoE EP dispatch/combine，还是 SP layout restoration？如何
验证一次通信优化真正生效？

### 参考答案

首先每次只加载同一个 rank 的 before/after trace，避免把多个 worker 的 stream 混在
一起。然后同时使用四类证据：

1. **CPU op/call stack**：
   - `RowParallelLinear → tensor_model_parallel_all_reduce` 通常是 attention TP；
   - `AgRsAll2AllManager.dispatch[_router_logits] → all_gatherv` 是 EP dispatch；
   - `AgRsAll2AllManager.combine → reduce_scatterv` 是 EP combine；
   - `tensor_model_parallel_all_gather` 是 TP layout restoration。
2. **Perfetto flow**：从 NCCL GPU kernel 回溯到 CUDA runtime 和 CPU caller；
3. **相对位置**：attention output GEMM 后是 TP reduction，router 后、expert GEMM 前是
   EP dispatch，expert combine 后是 EP return；
4. **调用次数**：结合 layer 数和 decode step 数验证。例如 48 层、32 个 decode step
   对应 `48 × 32 = 1536` 次稳态目标 collective。

优化验证不能只看 kernel 名称是否变化，还应比较：

- collective 自身 GPU duration；
- attention collective 开始到第一个 MoE kernel 开始的边界时间；
- full tensor materialization、chunk kernel 和 RMSNorm 工作量是否减少；
- 四个 rank 中最慢 rank 的关键路径；
- 关闭 profiler 后的 prefill、decode 和端到端 latency。

Profiler 会引入开销，CUDA Graph capture 和 replay 的显示方式也可能不同，所以最终
结论必须由相同配置下的无 profiler benchmark 支撑。

## 最后 Coding 题：优化这段 Qwen Layer

### 题目

现在回到最开始的三个类。已知：

- `RowParallelLinear` 默认在 output projection 内执行 TP AllReduce；
- MoE 开启 SP 后，会立即执行 `sequence_parallel_chunk`；
- 当前 layer 的 attention 后一定紧跟 sparse MoE；
- 暂时只考虑输入为 replicated layout、PP size 为 1 的分支。

请修改接口和 forward 数据流，消除不必要的通信与完整 tensor materialization。要求：

1. 使用合适的 collective 替换当前 attention-to-MoE 边界；
2. 保证 residual add 和 RMSNorm 数值等价；
3. 正确处理 token padding；
4. 防止 MoE 对已经是 SP layout 的输入再次 chunk；
5. 说明连续 layer 和模型输出边界还需要怎样扩展。

候选人可以写接近真实代码的伪代码，不要求记住 vLLM 的全部 API 名称。

### 参考答案

问题路径是：

```text
TP partial output
→ AllReduce，所有 rank 物化完整 tensor
→ residual add + RMSNorm 在所有 rank 上处理完整 tokens
→ sequence_parallel_chunk，只保留本 rank shard
```

可以改成：

```text
TP partial output
→ token-dimension ReduceScatter
→ local reduced attention shard
→ local residual add + RMSNorm
→ MoE 直接消费 SP shard
```

首先让 attention output projection 支持跳过内部 reduction：

```python
class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.attn = Attention(config)
        self.o_proj = RowParallelLinear(
            config.attention_output_size,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
        )
```

Decoder layer 构造 attention 时，需要根据当前 layer 是否紧跟 SP MoE 来设置开关：

```python
self.use_attn_reduce_scatter_for_moe = (
    parallel_config.use_sequence_parallel_moe
    and parallel_config.pipeline_parallel_size == 1
    and self.is_moe_layer
)
self.self_attn = Qwen3NextAttention(
    config,
    reduce_results=not self.use_attn_reduce_scatter_for_moe,
)
```

由同时知道 attention 和 MoE layout 的 decoder layer 执行转换：

```python
attention_partial = self.self_attn(
    hidden_states=hidden_states,
    positions=positions,
)

num_tokens = attention_partial.shape[0]
pad_tokens = (-num_tokens) % self.tp_size
attention_partial = F.pad(
    attention_partial,
    (0, 0, 0, pad_tokens),
)

hidden_states = tensor_model_parallel_reduce_scatter(
    attention_partial,
    dim=0,
)
residual = sequence_parallel_chunk(residual)

hidden_states, residual = self.post_attention_layernorm(
    hidden_states,
    residual,
)
hidden_states = self.mlp(
    hidden_states,
    already_sequence_parallel=True,
)
```

MoE block 需要知道输入是否已经是 SP layout：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    already_sequence_parallel: bool = False,
) -> torch.Tensor:
    should_convert_layout = (
        self.is_sequence_parallel and not already_sequence_parallel
    )

    if should_convert_layout:
        hidden_states = sequence_parallel_chunk(hidden_states)

    hidden_states = self.run_router_and_experts(hidden_states)

    if should_convert_layout:
        hidden_states = tensor_model_parallel_all_gather(
            hidden_states,
            dim=0,
        )

    return hidden_states
```

核心等价关系是：

```text
chunk(AllReduce(attention_partial))
    == ReduceScatter(attention_partial, dim=token)
```

这个单层答案仍不是完整的跨层实现。继续扩展时需要：

- 显式跟踪 hidden states 和 residual 当前是否为 SP layout；
- 连续 SP layer 在 attention 前只 gather attention 真正需要的 normalized input；
- residual stream 尽量保持 sharded；
- 非 MoE layer、final norm、logits、aux output 和 PP boundary 恢复 full layout；
- full attention 与 GDN output projection 使用一致的 `reduce_results` 契约。

### Correctness tests

至少覆盖：

- token 数能和不能被 TP size 整除；
- 单 token decode；
- TP size 2 和 4；
- 连续 MoE layers 和 MoE/非 MoE 交界；
- final norm 和 logits 前的 layout restoration；
- full attention 和 GDN linear attention；
- 优化前后 hidden states、residual 和 logits 的数值对比。

### 判断标准

- **基础**：发现 AllReduce 后很快只保留一个 token chunk；
- **合格**：提出 token-dimension ReduceScatter，并正确切分 residual；
- **良好**：处理 padding、RMSNorm 顺序和 MoE 双重 chunk；
- **优秀**：设计显式 layout 状态、跨层 gather 边界、完整 correctness tests 和
  before/after profiling。
