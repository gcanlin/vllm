# 把 Shared Expert 真正塞进 SM100 MegaMoE：一次 DeepSeek V4 B200 内核级优化实录

> 本文记录的是 vLLM NVIDIA DeepSeek V4 专有路径上的一次生产优化：在
> 8 张 B200 上，把原本串行执行的 FP8 shared expert 接入 DeepGEMM 的 SM100
> persistent MegaMoE kernel。它不是切换 backend，也不是调一个 batch 参数，而是从
> checkpoint 权重布局、Triton 输入量化、SM100 UMMA 指令选择，一直改到 kernel 内部
> scheduler 和最终 combine。

## 先看结果

测试模型是 `DeepSeek-V4-Flash-0731`，生产配置为 8x B200、TP8+EP8、sequence
parallel、FP4 routed experts、FP8 shared expert 和 DeepGEMM MegaMoE。

| 负载 | 关键指标 | 串行 shared | native fused | 变化 |
| --- | --- | ---: | ---: | ---: |
| Batch 1，128 in / 256 out | output throughput | 130.02 tok/s | 149.53 tok/s | **+15.00%** |
| Batch 1，1024 in / 128 out | output throughput | 125.19 tok/s | 142.41 tok/s | **+13.75%** |
| 128 requests，1024/128，concurrency 64 | output throughput | 3,415.45 tok/s | 3,737.98 tok/s | **+9.44%** |
| 32 requests，8192/32，concurrency 16 | output throughput | 234.09 tok/s | 249.42 tok/s | **+6.55%** |

配对的 200 题 GSM8K 回归从 83.0% 变为 83.5%，exact McNemar `p=1.0`，没有发现
可测量的精度回退。完整 1319 题、5-shot 的 `lm-eval 0.4.12` fused-only sanity run
得到 94.8446% strict/flexible exact match。后者没有同协议 baseline，因此只作为完整
数据集上的正确性检查，不拿来计算优化收益。

这次优化的核心不是“让两个 GEMM 同时跑”，而是同时消掉了四层开销：

1. 独立 shared MLP 的 kernel pipeline；
2. shared 输入的第二次量化；
3. shared 分支与 routed 分支之间的框架级同步；
4. `routed + shared` 的独立 PyTorch add 及其全量显存读写。

下面从原始瓶颈开始拆。

## 1. 一层 DSV4 MoE 原来到底在做什么

对一个 token 的 MoE 输出，可以写成：

```text
y = sum(topk_weight[i] * routed_expert[i](x)) + shared_expert(x)
```

DeepSeek V4 Flash 的两类 expert 并不是同一种数据类型：

- routed expert 是 FP4 权重，激活是 FP8；
- shared expert 保留 block-FP8 权重；
- shared expert 对每个有效 token 都执行，不参加 top-k 路由。

优化前，vLLM 的 NVIDIA 专有路径虽然已经把 routed experts 放进 SM100 persistent
MegaMoE，但 shared expert 仍然走普通 dense FP8 MLP：

```text
hidden_states
    |
    +--> DeepGEMM MegaMoE: routed FP4 L1 -> SwiGLU -> routed FP4 L2 -> combine
    |                                                                      |
    +--> 等 routed 返回后: shared FP8 L1 -> SwiGLU -> shared FP8 L2 -------+
                                                                           |
                                                              PyTorch add -> y
```

原来的 Python 关系等价于：

```python
final_hidden_states = self.experts(
    hidden_states,
    topk_weights,
    topk_ids,
    activation_clamp=activation_clamp,
)
shared_output = self.shared_experts(hidden_states)
final_hidden_states += shared_output
```

这是一个结构性串行尾巴。DSV4 有大量 MoE 层，因此即使每层只多出几次 kernel launch、
一次全 hidden-size add 和一点同步，累积到一整个 decode step 后也会很明显。Batch 1
上 +15% 的结果正说明这不是只在大 batch 下才成立的吞吐优化。

## 2. 为什么 B200 上可以做得比“双流并发”更深

vLLM 当前固定的 DeepGEMM 已经包含一条 SM100 native shared-expert 路径，只是 vLLM
此前没有把 DSV4 的权重、scale、buffer 和调用参数接进去。

这条 kernel 的关键硬件基础包括：

- SM100 的 block-scaled UMMA，可以在同一个 persistent kernel 中针对任务动态选择
  FP8×FP4 或 FP8×FP8 指令描述；
- TMA 负责 global/shared memory 搬运；
- TMEM 保存 GEMM accumulator；
- 2-CTA cooperative kernel 和 persistent scheduler 可以交错安排 dispatch、routed L1/L2
  与 shared L1/L2；
- 同一个 combine 阶段可以把 routed top-k slot 与 shared slot 在 FP32 寄存器中累加。

因此目标不是在外面再开一条 CUDA stream，而是让 shared expert 成为同一个调度器认识的
原生 task phase：

```text
                       +-- SharedLinear1 -- SwiGLU -- SharedLinear2 -- slot top_k --+
                       |                                                            |
x -> quant/stage once -+-- dispatch -- Linear1 -- SwiGLU -- Linear2 -- slots 0..k-1 +
                                                                                    |
                                                     FP32 register reduction -> BF16 y
```

注意这里的 shared 仍然不是 routed expert。它没有 fake expert id，没有 top-k gate
weight，也不需要跨 EP rank dispatch；它只是复用了同一个 persistent kernel 的装载、
MMA、epilogue 和 combine 基础设施。

## 3. 先看另外两种实现

这里对比的是 2026-08-20 本地源码快照：TokenSpeed
`978ed2cfdc870cabab20289c46329f5b744aaec2`、SGLang
`5f128395910dafb98c34083dc26cb790c7674d34`。对比结论来自源码，不是跨框架性能跑分。
不同框架的 scheduler、attention、graph 和通信栈都不同，直接拿端到端数字比较反而会
掩盖 shared expert 方案本身的差异。

### 3.1 TokenSpeed：双 CUDA stream 重叠两条独立 pipeline

TokenSpeed 的提交
[`4e51bed`](https://github.com/lightseekorg/tokenspeed/commit/4e51bed840911e483d8ddf5f24d61061cf60bc4d)
引入了 routed/shared overlap。它的 `StreamFork` 本质上由两个 CUDA event 建立 fork/join：

```python
@contextmanager
def scope(self, *, enable: bool, overlap: bool = True):
    self._active = enable and self.aux_stream is not None
    if self._active:
        self._current = torch.cuda.current_stream()
        self.fork_event.record(self._current)
    try:
        yield self
    finally:
        if self._active:
            self.join_event.wait(self._current)

@contextmanager
def branch(self):
    if not self._active:
        yield
        return
    with torch.cuda.stream(self.aux_stream):
        self.fork_event.wait(self.aux_stream)
        yield
        self.join_event.record(self.aux_stream)
```

MoE forward 再把 routed 放在当前 stream，把 shared 放进 branch：

```python
with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
    routed = self.experts(hidden_states, topk_weights, topk_ids, ...)
    with fork.branch():
        shared = self._forward_shared_experts(
            hidden_states,
            ctx=ctx,
            comm_manager=comm_manager,
        )
return routed + shared if shared is not None else routed
```

这个设计有两个明显优点：

- 改动位于模型编排层，不要求 routed backend 原生理解 mixed-precision shared expert；
- shared dense TP 可以通过 `pre_dense_comm/post_dense_comm`，或原始提交中的 token
  all-gather/reduce-scatter，覆盖比本次 native fusion 更宽的并行拓扑。

但它仍然是两套独立 pipeline：

- routed 和 shared 各自量化、launch GEMM 和执行 epilogue；
- 两条 stream 会竞争同一张 B200 的 Tensor Core、SM 和显存带宽，并发不等于计算时间
  可以全部隐藏；
- join 后仍要执行 `routed + shared`；
- dense TP 通信仍然存在；
- 代码中 `enable=get_is_capture_mode()` 意味着这条 overlap 主要服务于 CUDA Graph
  capture 路径，非 capture 时 branch 会退化到当前 stream。

它解决的是“串行执行”，但没有解决“重复 pipeline”和“最终独立 add”。

### 3.2 SGLang Waterfill：把 shared 变成额外 routed slot

SGLang Waterfill 的思路完全不同：复制 shared expert 到每个 EP rank，然后把它当作一个
额外 routed expert。源码注释以 DeepSeek V3/R1 的 top-k 8 为例写成 `8 -> 9`；对本文
测试的 DSV4 top-k 6，同样的逻辑就是 `6 -> 7`。

其低 batch 本地路径会直接扩展 top-k：

```python
old_epr = num_routed_experts // world_size
new_epr = old_epr + 1

# 原 routed id 因每个 rank 插入一个 shared slot 而重映射。
expanded_topk_ids[:, :topk] = torch.where(
    valid_mask, topk_ids + old_ranks, topk_ids
)

shared_id = source_rank * new_epr + old_epr
expanded_topk_ids[:, topk] = torch.where(
    has_valid, shared_id, LOCAL_SHARED_MARKER
)
expanded_topk_weights[:, topk] = shared_weight
```

较大 batch 下，Waterfill 先统计每个 rank 的 routed load，再把 shared slot 发送到较空的
rank。当前阈值是：

```python
class WaterfillBalancer:
    MIN_BATCH_FOR_BALANCE = 64
```

它的 Triton `_waterfill_expand_kernel` 在一次 pass 中完成目的 rank 选择、expert id
重映射和 top-k 扩展。动态模式还会 all-reduce 各 rank 的 routed count 和 active token
count；低 batch 的 static 模式则把 shared 留在本地，避免为了一个 decode token 支付
额外通信。

模型层相应地把 expert 空间改成：

```python
num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
top_k_for_moe = config.num_experts_per_tok + 1
```

loader 再把 checkpoint 中的 shared 权重重写到 fused expert namespace：

```python
if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
    name = name.replace(
        "mlp.shared_experts",
        f"mlp.experts.{self.config.n_routed_experts}",
    )
```

SGLang 的 MegaMoE 调用仍然只传一组同构 expert weights：

```python
deep_gemm.fp8_fp4_mega_moe(
    y,
    moe.experts.mega_l1_weights,
    moe.experts.mega_l2_weights,
    buf,
    ...,
)
```

也就是说，Waterfill 的 shared 在执行语义上已经成为 routed expert，而不是通过
DeepGEMM 的 `shared_l1_weights/shared_l2_weights` mixed-precision 接口进入 kernel。

这个设计的强项是负载均衡：当 routed expert 分布偏斜时，shared 这份固定工作可以填到
较空的 rank，甚至复用 token 已经要去的目标 rank。代价则是：

- 每个 token 的 top-k 多一个元素，dispatch/combine 元数据随之扩大；
- 大 batch 动态 Waterfill 需要统计甚至 all-reduce load；
- shared 可能跨 rank，增加通信；
- routed/shared 必须进入同一套 expert weight 表示。

最后一点对本文的 NVIDIA DSV4 checkpoint 尤其关键：routed 是 FP4，shared 是 FP8。
SGLang 当前 DSV4 代码明确检查这类精度差异：

```python
if quant_blocks_shared_experts_fusion(quant_config):
    return (
        "Quantization keeps shared experts at a higher precision than the "
        "routed experts, so they cannot be fused into the quantized "
        "routed-expert path."
    )
```

某些量化工作流可以先把 shared 转成 routed 所需格式，或在本来就同精度的模型上直接
融合；但对“FP4 routed + FP8 shared”这个目标，Waterfill 本身没有表达两种 weight dtype
的原生 phase。它优化的是 expert placement，而本文的方案优化的是 mixed-precision
kernel integration，两者并不是同一个问题的重复实现。

## 4. 本次设计：保留 shared 语义，也保留 FP8 精度

本次实现给自己设了五条约束：

1. shared checkpoint 权重保持 FP8，不重新量化成 FP4；
2. replicated shared weights 的生产拓扑中不新增通信；
3. routed 和 shared 共用一次输入量化；
4. shared 进入现有 MegaMoE combine，不再调用框架级 add；
5. 旧 wheel、非复制权重和不兼容 shape 必须安全 fallback，不能把生产启动变成硬失败。

最终接入点只有 DSV4 的 NVIDIA MegaMoE 专有路径，不修改通用 FusedMoE，也不改变
DeepGEMM source pin。

### 4.1 先判断这个 rank 是否真的拥有完整 shared MLP

native shared MMA 需要每个执行 rank 上都有完整 shared L1/L2。TP8+EP8 + sequence
parallel 会复制 shared weights、切分 token，因此满足条件；TP1 也天然满足。普通的
非-SP TP 则只拥有 shared MLP shard，不能直接传给完整 native shared MMA。

[模型初始化代码](vllm/models/deepseek_v4/nvidia/model.py)因此把能力判断写成：

```python
fuse_shared_experts = bool(
    self.shared_experts is not None
    and not envs.VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION
    and (self.use_sequence_parallel or self.tp_size == 1)
)

self.experts = DeepseekV4MegaMoEExperts(
    ...,
    num_shared_experts=(
        self.n_shared_experts if fuse_shared_experts else 0
    ),
)
```

这不是一个“能 launch 就 launch”的乐观判断。并行布局不满足时，模型从构建阶段就保留
原来的串行 shared MLP，避免运行时才发现权重不完整。

### 4.2 checkpoint 的 128x128 scale 如何变成 MMA 要的 1x32

shared 权重是 FP8 E4M3，checkpoint scale 是 UE8M0，通常每 128 行、128 个 K 元素
共享一个 scale。native shared UMMA 则要求每个权重行、每 32 个 K 元素一个 scale。

假设 checkpoint 中某个 128x128 block 的 scale 是 `s`。把这个 block 拆成
`128 * 4` 个 1x32 小块，并让每个小块仍使用 `s`，反量化结果仍然是：

```text
real_weight = fp8_weight * s
```

所以这只是 scale 的复制展开和内存布局转换，不是重新估计 scale，更不是重新量化权重。

代码先把 UE8M0 的 exponent byte 精确还原成 FP32 power-of-two：

```python
@staticmethod
def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    return (sf.to(torch.int32) << 23).view(torch.float32)
```

然后做二维展开：

```python
scale_fp32 = self._ue8m0_uint8_to_float(scale.view(torch.uint8))
scale_1x32 = (
    scale_fp32.repeat_interleave(block_m, dim=0)
    .repeat_interleave(block_k // 32, dim=1)[:mn, : k // 32]
    .contiguous()
)

return deep_gemm.transform_sf_into_required_layout(
    scale_1x32.unsqueeze(0),
    mn,
    k,
    (1, 32),
    1,
).squeeze(0)
```

这里先加 singleton group 维度，是为了复用 DeepGEMM grouped expert 的 MN-major、
TMA-aligned scale packer；native shared 接口最终需要二维 tensor，所以转换后再 squeeze。

实现还校验了：

- checkpoint scale shape 必须与 `weight_block_size` 一致；
- `block_k` 必须能被 32 整除；
- shared weights 必须是 `torch.float8_e4m3fn`；
- packed scales 必须是 DeepGEMM 预期的 `int32`；
- gate/up 和 down shape 必须精确匹配完整 shared intermediate size。

任何一项不满足都只关闭 native fusion，不会拿错误布局继续算。

### 4.3 gate/up interleave 不只是一次 `contiguous()`

SM100 L1 epilogue 会直接从 TMEM 读取 gate/up pair 并做 SwiGLU。为了让每组输出与
epilogue 的 atom 布局对齐，DeepGEMM 要求 L1 权重按 8 行粒度交错：

```text
checkpoint: [gate 0 ... gate N-1 | up 0 ... up N-1]

kernel:     [gate 0..7 | up 0..7 | gate 8..15 | up 8..15 | ...]
```

vendored DeepGEMM 的转换逻辑是：

```python
if isinstance(l1_weights, tuple):
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(
        _interleave_weights(l1_weights[1])
    )
    l1_transformed = (l1_w, l1_sf)
    l2_transformed = (
        l2_weights[0],
        _transpose_sf_for_utccp(l2_weights[1]),
    )
```

scale 的 UTCCP transpose 则把 `[group, MN, packed_K]` reshape 成
`[group, ..., 4, 32, packed_K]`，交换中间两维后再拷贝。weight 和 scale 必须一起
变换；只 interleave weight 会得到 shape 正确但数值错误的结果。

#### 一个实际的显存坑

L1 interleave 会分配完整新 tensor。如果把 loader 创建的原始 gate/up Parameter 和
interleaved tensor 都留着，DSV4 Flash 每卡会多占约 0.70 GiB。

native fusion 后普通 shared MLP 不再被 forward 调用，所以代码直接把 Parameter
storage 迁到 transformed tensor：

```python
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
    (gate_up_weight, gate_up_scale),
    (down_weight, down_scale),
)

gate_up.weight.data = transformed_l1[0]
self._transformed_shared_l1_weights = (
    gate_up.weight.data,
    transformed_l1[1],
)
self._transformed_shared_l2_weights = transformed_l2
```

这样 Parameter 仍持有 forward 要用的 storage，原始完整 L1 storage 又能被释放。最终
实测模型加载显存只从 20.73 GiB 增加到 20.76 GiB，而不是为一个离线布局转换永久保存
两份大权重。

### 4.4 routed 和 shared 共用 FP8 activation，但不能共用 scale 地址

输入 `hidden_states` 只需要量化一次。现有
[`prepare_megamoe_inputs`](vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py)
Triton kernel 已经做了三件事：

1. BF16 hidden 按 K=32 计算 `amax`；
2. 把 scale 向上取整到 UE8M0 power-of-two，并生成 FP8 E4M3 activation；
3. 写 routed 使用的 scale、top-k id 和 top-k weight staging buffer。

每个 program 处理一个 token 的 128 个 K 元素，内部有 4 个 K=32 group。四个
8-bit UE8M0 exponent 被打包进一个 `int32`：

```python
scale_exp = ((scale_bits >> 23) & 0xFF) + (
    (scale_bits & 0x7FFFFF) != 0
)
rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
fp8 = tl.reshape(scaled, [BLOCK_K]).to(tl.float8e4nv)

packed_scale = tl.sum(
    scale_exp << (tl.arange(0, 4) * 8), axis=0
).to(tl.int32)
```

问题是：shared L1 的 TMA scale load 要求 MN-major 行置换，不能直接拿 routed 的
row-major `x_sf` 指针。更麻烦的是，这个置换依赖 DeepGEMM 对当前 token 数动态选出的
`BLOCK_M`。

如果固定按 128 转置，某些 decode shape 可能碰巧能跑，换到 graph capture size 或
prefill shape 就会静默读错 scale。这类 bug 通常不会报 CUDA error，只会表现为模型输出
崩坏，所以必须从 runtime scheduler 取得真实值：

```python
shared_block_m = deep_gemm.get_block_m_for_mega_moe(
    get_ep_group().world_size,
    self.num_experts,
    symm_buffer.num_max_tokens_per_rank,
    num_tokens,
    self.top_k,
    "fp8xfp4",
)
```

然后在 packed scale 仍在 Triton 寄存器里时，顺手写第二个地址：

```python
m_block_id = token_id // SHARED_BLOCK_M
m_in_block = token_id % SHARED_BLOCK_M
aligned_block_m = triton.cdiv(SHARED_BLOCK_M, 128) * 128

transposed_m = (
    (m_in_block // 128) * 128
    + (m_in_block % 32) * 4
    + (m_in_block % 128) // 32
)
shared_row = m_block_id * aligned_block_m + transposed_m

tl.store(
    shared_x_sf
    + shared_row * shared_x_sf_stride_m
    + k_block_id * shared_x_sf_stride_k,
    packed_scale,
)
```

这一段是接入中最关键的代码：

- `aligned_block_m` 为每个 M block 留出 TMA 需要的 128-row 对齐空间；
- `(m % 32) * 4 + (m % 128) // 32` 把 128 行重排成 32×4 的访问顺序；
- `m // 128` 让 `BLOCK_M > 128` 时仍能覆盖后续 128-row 子块；
- 整个过程没有第二次 `amax`、没有第二个 quant kernel，也没有临时 activation tensor。

这比“复用量化结果”更精确：FP8 activation bytes 共用，scale 的数值也共用，但针对两个
TMA consumer 写成各自需要的地址布局。

### 4.5 symmetric buffer 也必须把 shared 算进去

native path 需要额外的 shared L1 scale、shared L2 activation/scale 和一个 combine
slot。DeepGEMM 的 buffer API 已经提供 `num_shared_experts`：

```python
symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group,
    self.num_experts,
    self.max_num_tokens,
    self.top_k,
    self.hidden_size,
    self.intermediate_size,
    num_shared_experts=self.num_shared_experts,
)
```

因此 vLLM 的 symmetric-buffer cache key 也加入了 `num_shared_experts`。否则同一进程先
建立无 shared buffer、再请求 fused buffer 时可能错误复用旧 allocation；这是 CUDA Graph
和多模型场景中很难排查的状态污染。

DeepGEMM 内部会把 combine buffer 从 `top_k` 个 slot 扩为 `top_k + 1`。shared L1
输入 activation 与原输入复用，额外持有的是 TMA scale 视图和 L1/L2 中间结果所需空间。

### 4.6 Python 最终只多传两组 weights

准备完成后，launch 侧的差异很小：

```python
deep_gemm.fp8_fp4_mega_moe(
    y,
    self._transformed_l1_weights,
    self._transformed_l2_weights,
    symm_buffer,
    shared_l1_weights=self._transformed_shared_l1_weights,
    shared_l2_weights=self._transformed_shared_l2_weights,
    activation_clamp=activation_clamp,
    fast_math=fast_math,
)
```

但这两个参数改变了 kernel 内部从调度到 combine 的完整路径，而不是简单地在一个 C++
wrapper 里连续调用两次 GEMM。

## 5. 深入 SM100 kernel：一次 launch 如何容纳两种 expert

这一节看 vendored DeepGEMM 的真实执行逻辑。

### 5.1 scheduler 把 shared 定义成两种新 phase

```cpp
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

CUTLASS_DEVICE bool is_shared() const {
    return kHasSharedExperts ? (block_phase > BlockPhase::Linear2) : false;
}
```

shared 不伪装成某个 routed expert id。`TaskInfo` 自己携带 phase、M block、N cluster、
有效 M 和真实 N/K shape，后面的 TMA/MMA/epilogue 都按 phase 选择描述符。

### 5.2 SharedLinear1 被安排在 dispatch wait 之前

scheduler 主循环的顺序很有意思：

```cpp
if constexpr (kHasShared) {
    // Shared expert L1 tasks do not depend on dispatch.
    shared_mainloop<BlockPhase::SharedLinear1, ...>(num_tokens, ...);
}

// Wait dispatch's results
fetch_expert_recv_count();

do {
    task_info = get_next_task();
    if (task_info.is_valid()) publish_task(task_info, lane_idx);
} while (task_info.is_valid());

if constexpr (kHasShared) {
    // Shared expert L2 tasks depend on SharedLinear1 completion.
    shared_mainloop<BlockPhase::SharedLinear2, ...>(num_tokens, ...);
}
```

routed L1 在收到跨 rank dispatch 结果前不知道每个 expert 有多少 token；shared L1 却只
依赖本地 token 数。scheduler 因此先把这批独立任务投喂给 persistent worker，用本来会
等待 dispatch 的时间做有效计算。随后执行 routed L1/L2，最后投放依赖 shared L1 完成的
SharedLinear2。

这就是 native scheduling 与外层双流的根本区别：它知道数据依赖和 tile shape，可以在
同一 persistent work queue 中填空，而不是把两条完整图交给 GPU 自己争抢资源。

### 5.3 同一个 MMA issue warp 动态选择 FP4 或 FP8 weight descriptor

routed 和 shared 共用 FP8 activation，但权重类型不同。kernel 预先构造两个 block-scaled
UMMA instruction descriptor：

```cpp
auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t, ...>();

auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    shared_b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t, ...>();

auto& instr_desc = task_info.is_shared()
    ? shared_instr_desc
    : routed_instr_desc;
```

在这个 FP8×FP4 kernel 中，`b_dtype_t` 是 FP4 E2M1，`shared_b_dtype_t` 是 FP8 E4M3。
同一个 MMA issue warp 根据 task phase 选 descriptor，再根据有效 M 动态更新 UMMA N。
TMA load warp 同样按 phase 选择 routed/shared activation、scale 和 weight tensor map。

所以 shared 保持 FP8 并不是在 FP4 kernel 外面嵌了一个普通 FP8 GEMM，而是 kernel 本身
具有 mixed instruction path。

### 5.4 L1 epilogue 共用 SwiGLU，但 shared 的 gate weight 恒为 1

routed L1 输出在做 SwiGLU 时还要乘 top-k weight。shared 不经过 router，因此代码把
cached weight 初始化为 `1.0f`，只在非 shared task 时才从 top-k weight buffer 更新：

```cpp
float stored_cached_weight = 1.0f;

if (not task_info.is_shared() && should_reload_weight) {
    stored_cached_weight = *buffer.l1_topk_weights_buffer...;
}
```

这保留了原始数学语义：routed contribution 乘 routing weight，shared contribution 不乘。
如果整个 MegaMoE 输出之后还有统一的 routed scaling，Python 侧会保证 shared 不被错误
缩放；本文模型的 top-k staging 已把 routed scaling 放在正确位置。

### 5.5 shared L2 写入专用 combine slot

routed L2 要把结果送回 token 来源 rank，并写入对应 top-k slot。shared 只处理本 rank
token，因此直接写本地 rank 的第 `kNumTopk` 个 slot：

```cpp
if (task_info.is_shared()) {
    dst_rank_idx = sym_buffer.rank_idx;
    dst_token_idx = pool_m_idx + m_idx_in_block;
    dst_topk_idx = kNumTopk;
} else {
    dst_rank_idx = src_metadata.rank_idx;
    dst_token_idx = src_metadata.token_idx;
    dst_topk_idx = src_metadata.topk_idx;
}
```

这里没有给 shared 创建 fake routed id，也没有让它经过 NVLink dispatch。它只复用
combine buffer 的 slot abstraction。

### 5.6 combine 在 FP32 register 中一次完成

combine warp 的 active mask 原来只包含 routed top-k，现在多包含 shared slot：

```cpp
const int stored_topk_slot_idx = lane_idx < kNumTopk
    ? input_topk_idx[token_idx * kNumTopk + lane_idx]
    : (kNumSharedExperts > 0 && lane_idx == kNumTopk
        ? static_cast<int>(kNumTopk)
        : -1);
```

每个 BF16 contribution 由 TMA 分阶段载入，然后在 `float2` register 中累加：

```cpp
float2 reduced[...] = {};
while (do_reduce) {
    ...
    ptx::accumulate(reduced[index], bf16_values[l]);
}

casted_bf16[l] = __float22bfloat162_rn(reduced[index]);
```

最后才 round 到 BF16 并 TMA store 到输出 `y`。各 expert 的 L2 contribution 仍会以
BF16 写进 combine slot，这是 MegaMoE 原有设计；本次消掉的是 combine 完成之后再把
shared dense output 读出来、执行一次全 tensor add、再写回 `y` 的额外 pass。

## 6. 如何保证没有重复算 shared

native launch 成功后，模型层必须跳过旧 shared MLP；fallback 时又必须保留它。实现用
真实 transformed-weight 状态，而不是只看配置开关：

```python
@property
def has_fused_shared_experts(self) -> bool:
    return self._transformed_shared_l1_weights is not None
```

forward 再判断：

```python
final_hidden_states = self.experts(...)

if (
    self.shared_experts is not None
    and not self.experts.has_fused_shared_experts
):
    shared_output = self.shared_experts(hidden_states)
    final_hidden_states += shared_output
```

这比设置一个 `enable_fusion=True` 更稳健：即使配置允许，但 wheel API、weight dtype 或
scale shape 在 post-load 阶段不兼容，`has_fused_shared_experts` 仍为 false，串行路径会
自动接管，不会漏算 shared。

## 7. 旧 DeepGEMM wheel 为什么不能只靠版本号判断

实际机器最初安装的是 0.27.0 wheel，其中 `_deep_gemm_C` 比当前 vLLM vendored source
旧。Python 文件可能已经有新函数，二进制 extension 却没有对应 ABI。只检查 package
version 或 `hasattr(deep_gemm, ...)` 都不够。

代码在任何 symmetric-memory rendezvous 之前检查真实 Python signature：

```python
buffer_params = signature(
    deep_gemm.get_symm_buffer_for_mega_moe
).parameters
kernel_params = signature(
    deep_gemm.fp8_fp4_mega_moe
).parameters

supported = (
    hasattr(deep_gemm, "get_block_m_for_mega_moe")
    and hasattr(deep_gemm, "transform_weights_for_mega_moe")
    and "num_shared_experts" in buffer_params
    and "shared_l1_weights" in kernel_params
    and "shared_l2_weights" in kernel_params
)
```

为什么一定要在 symmetric buffer 前做？因为多 rank symmetric memory 创建本身是一次
collective rendezvous。如果部分 rank 已经进入旧 API，另一些 rank 才异常退出，表现
可能不是干净的 Python exception，而是整组进程 hang。

检测失败时会打印 rebuild vendored `_deep_gemm_C` 的提示，并自动继续使用串行 shared
path。本次正式 benchmark 已重建与当前 vLLM pin 匹配的 extension，但没有为了本优化
修改 DeepGEMM pin。

生产紧急回滚不需要切换整个 MoE backend：

```bash
VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1
```

它只恢复 routed MegaMoE + serial shared MLP，便于线上做同 backend A/B 和快速止损。

## 8. 三种方案放在同一张表里

| 维度 | TokenSpeed 双流 | SGLang Waterfill | 本次 vLLM native fusion |
| --- | --- | --- | --- |
| shared 的身份 | 独立 dense MLP | 额外 routed expert slot | 独立 native shared phase |
| 调度层级 | CUDA streams + events | top-k/EP placement + MegaMoE | SM100 persistent scheduler |
| routed/shared mixed dtype | 两条 pipeline，各自处理 | 要求同构 expert 表示或先转换 | 同 kernel 动态选择 FP4/FP8 UMMA descriptor |
| 输入量化 | 两条路径可各自量化 | 进入 routed staging | 一次 FP8 quant，同时写两种 scale layout |
| shared EP 通信 | dense TP 时可能需要 | 大 batch 可远程 Waterfill | replicated 模式下无新增通信 |
| 最终相加 | `routed + shared` | 作为 top-k+1 combine | 原 MegaMoE combine 多一个本地 slot |
| 负载均衡能力 | 依赖两条 pipeline 自然重叠 | 强，可把 shared 发往较空 rank | 不迁移 shared，利用 dispatch wait 和 tile scheduler |
| tensor-sharded shared | 支持范围较宽 | 取决于 fused expert loader/backend | 不支持，自动 fallback |
| 硬件/后端范围 | 更通用 | DeepEP/MegaMoE 等融合路径 | DSV4 NVIDIA SM100 MegaMoE 专用 |

因此“我们的优势”需要加上适用条件：在 B200、DSV4 mixed-precision checkpoint、
replicated shared weights 这个明确目标上，native fusion 更贴近硬件真实能力：

- 不牺牲 shared FP8 权重精度；
- 不增加 top-k 元数据或 Waterfill 通信；
- 不依赖两条饱和 kernel pipeline 能否在不同 stream 上理想重叠；
- 消掉第二次输入量化和框架级最终 add；
- scheduler 能用 SharedLinear1 填充 dispatch wait。

但它不是所有拓扑上的通用替代：如果 shared 仍是 tensor shard，TokenSpeed 式 dense
communication 更完整；如果主要问题是 routed load skew，SGLang Waterfill 的 placement
能力更有价值。把优化边界讲清楚，比声称一个方案绝对优于另外两个更接近生产现实。

## 9. 测试：重点不是“kernel 能 launch”

新增单元测试覆盖了最容易静默出错的边界。

### 9.1 权重 finalization 与 storage ownership

测试构造 FP8 gate/up、down 和 checkpoint-style scale，验证：

- scale 输入按 128x128 -> 1x32 正确展开；
- `transform_weights_for_mega_moe` 收到二维 shared weights；
- transformed gate/up Parameter 的 storage pointer 已切换；
- `has_fused_shared_experts` 只在完整转换成功后为 true。

### 9.2 防止 double add

同一个模型级 forward 测试分别模拟 fused=false/true：fallback 必须调用一次
`shared_experts`，fused 必须完全跳过。这样能捕获“kernel 已经加了 shared，Python 又加
一次”这种数值看起来仍正常、但幅值系统性错误的回归。

### 9.3 动态 BLOCK_M 的 scale TMA layout

测试参数化覆盖：

```text
BLOCK_M = 8, 32, 96, 128, 192
```

每个 case 都检查 routed FP8 activation、routed packed scale 的原有结果不变，并验证
shared scale 出现在置换后的精确 row。token 数设为 `BLOCK_M + 7`，故意跨过 M block
边界，防止实现只对第一个 block 正确。

### 9.4 真实运行验证

最终验证包括：

```text
ruff format: passed
ruff check: passed
pytest -q tests/models/test_deepseek_v4_mega_moe.py
17 passed
```

真实 8 卡服务还成功 capture 了 1、2、4、8、16、32、64、128、256 的
FULL_AND_PIECEWISE CUDA Graph shape。这个检查很重要，因为 dynamic `BLOCK_M` 和
symmetric buffer cache 的错误经常只会在某个 capture size 上暴露。

## 10. Benchmark 方法和完整结果

服务命令的关键部分如下：

```bash
vllm serve /mnt/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name dsv4 \
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

baseline 只增加 rollback 环境变量，其他配置、模型、backend 和随机 seed 完全一致。
每个汇总是三个 paired seed 的算术平均。

### 10.1 Batch 1

`--max-concurrency 1`，每次 8 个 measured requests、2 个 warmup，temperature 0、
`ignore_eos`。

| 负载与指标 | Before | After | 变化 |
| --- | ---: | ---: | ---: |
| 128/256 output throughput | 130.02 tok/s | 149.53 tok/s | **+15.00%** |
| 128/256 mean TPOT | 7.653 ms | 6.644 ms | **-13.18%** |
| 128/256 median ITL | 7.577 ms | 6.567 ms | **-13.32%** |
| 128/256 mean TTFT | 17.32 ms | 17.70 ms | +2.18% |
| 1024/128 output throughput | 125.19 tok/s | 142.41 tok/s | **+13.75%** |
| 1024/128 mean TPOT | 7.158 ms | 6.294 ms | **-12.07%** |
| 1024/128 median ITL | 7.090 ms | 6.221 ms | **-12.25%** |
| 1024/128 mean TTFT | 113.13 ms | 99.36 ms | **-12.17%** |

decode-heavy 三个 seed 的吞吐收益是 15.01%、14.99%、15.01%；1024/128 是
13.82%、13.72%、13.72%。不是由单个幸运 seed 拉高。

fused benchmark client 的未编辑 stdout 已提交到：

- [128/256 原始日志](benchmarks/results/dsv4_sm100_shared_expert_fusion/fused-b1-decode.log)
- [1024/128 原始日志](benchmarks/results/dsv4_sm100_shared_expert_fusion/fused-b1-balanced.log)

日志包含完整 client arguments、warmup 状态、成功/失败请求数、实际 token 数和最终
延迟/吞吐输出，可以直接核对表格不是手写的虚构数字。

### 10.2 中高并发

| 负载与指标 | Before | After | 变化 |
| --- | ---: | ---: | ---: |
| 1024/128，concurrency 64：output throughput | 3,415.45 tok/s | 3,737.98 tok/s | **+9.44%** |
| 同负载：mean TTFT | 588.52 ms | 530.15 ms | **-9.92%** |
| 同负载：mean TPOT | 14.095 ms | 12.935 ms | **-8.23%** |
| 8192/32，concurrency 16：output throughput | 234.09 tok/s | 249.42 tok/s | **+6.55%** |
| 同负载：mean TTFT | 901.81 ms | 836.36 ms | **-7.26%** |
| 同负载：mean TPOT | 40.584 ms | 38.458 ms | **-5.24%** |

收益从 batch 1 的 13%~15% 降到较高并发的 6%~9% 是合理的：大 batch 下 routed
MegaMoE 占比上升，shared 尾部能被摊薄；但 native scheduler、一次量化和一次 combine
仍然有效，所以四个 workload 都是正收益。

### 10.3 显存

| 每卡指标 | Before | After | 变化 |
| --- | ---: | ---: | ---: |
| 模型加载 | 20.73 GiB | 20.76 GiB | +0.03 GiB |
| runtime consumed | 23.04 GiB | 23.17 GiB | +0.13 GiB |
| KV capacity | 386,897 tokens | 386,512 tokens | -385（-0.10%） |

这组数据也验证了前面的 storage re-home 确实生效：native layout 没有让模型永久保留
两份 shared L1 权重。

## 11. 精度为什么不能只比较 token-by-token

baseline generic FP8 shared linear 使用 per-128 activation quant，native shared MMA 使用
per-32 activation quant。权重 scale 的展开数学等价，但 activation quant granularity
改变会带来正常的 rounding 差异，因此不能要求两个版本生成 token 流逐字完全一致。

更合理的生产检查是固定样本、固定解码设置的 paired task accuracy：

| 路径 | 正确 | 准确率 | 无法解析 |
| --- | ---: | ---: | ---: |
| serial shared | 166 / 200 | 83.0% | 0 |
| native fused shared | 167 / 200 | 83.5% | 0 |

配对分解为：双方都对 156、仅 baseline 对 10、仅 fused 对 11、双方都错 23。exact
McNemar `p=1.0`。

另一次标准 `lm-eval 0.4.12` 完整 GSM8K fused run 的原始汇总是：

| Task | Version | Filter | n-shot | Metric | Value | Stderr |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| gsm8k | 3.0 | strict-match | 5 | exact_match | 0.9484457923 | 0.0060908880 |
| gsm8k | 3.0 | flexible-extract | 5 | exact_match | 0.9484457923 | 0.0060908880 |

样本数为 1319，服务端 model revision 是本分支 `9a09dfedf1`。由于没有跑同协议 serial
baseline，这张表只证明 fused 路径能完成整套标准评测且分数正常；真正的 before/after
精度结论来自上面的 paired 200 题。

## 12. 生产边界和下一步

当前 native fusion 有意限制在：

- NVIDIA SM100；
- DeepSeek V4 NVIDIA 专有模型路径；
- `deep_gemm_mega_moe`；
- FP4 routed + replicated block-FP8 shared；
- TP1，或 TP+EP sequence parallel 下的完整 shared weights。

以下场景自动保留旧路径：

- 非 sequence-parallel 的 tensor-sharded shared MLP；
- 旧 DeepGEMM wheel/extension；
- shared dtype、shape、block size 或 scale layout 不兼容；
- 其他 MoE backend 和其他模型架构。

下一步值得做的不是把 guard 强行放宽，而是分别解决两个真实问题：

1. **tensor-sharded shared native fusion**：需要把 shared dense TP 通信与 MegaMoE
   scheduler 协同起来，而不是直接把 shard 当完整权重；
2. **native shared 与 Waterfill 的组合**：让 replicated FP8 shared 默认走本地 native
   phase，只在 routed load skew 足以覆盖通信成本时选择远程 placement。这需要 cost model，
   不能简单地永远 local 或永远 least-loaded。

## 结语

这个优化最有价值的部分，不是最后那两个 `shared_l*_weights` 参数，而是把模型语义和
SM100 kernel 语义严丝合缝地接起来：

- checkpoint 的 coarse UE8M0 scale 被数学等价地展开；
- gate/up weight 与 scale 同步 interleave；
- 一次 activation quant 同时服务两种 TMA layout；
- scheduler 用 shared L1 填 dispatch wait；
- UMMA issue warp 在 FP4/FP8 weight descriptor 间按 task 切换；
- shared 最终只是 combine 中的一个本地 slot，在 FP32 中和 routed 一起归约。

TokenSpeed 告诉我们 shared 与 routed 可以并发；SGLang Waterfill 告诉我们 shared 也可以
成为负载调度的一部分；这次 vLLM 实现更进一步利用了 B200 的专有路径：不把 shared
降格成普通 routed expert，也不把它留在另一条独立 pipeline，而是保留 mixed-precision
语义，让同一个 SM100 persistent kernel 原生理解它。

相关实现与完整工程结果：

- [核心模型接入](vllm/models/deepseek_v4/nvidia/model.py)
- [Triton input staging](vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py)
- [DeepGEMM scheduler](vllm/third_party/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)
- [SM100 FP8×FP4 MegaMoE kernel](vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)
- [英文 PR 描述](DSV4_SM100_SHARED_EXPERT_FUSION_PR.md)
- [中文工程报告](DSV4_SM100_SHARED_EXPERT_FUSION_REPORT_ZH.md)
