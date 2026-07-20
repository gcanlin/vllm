# DeepSeek V3.2 推理优化背景知识

这篇文档解释我们正在调研的几个 vLLM 微优化所需要的全部背景:DeepSeek V3.2 的模型结构、vLLM 怎么跑它、GPU 执行的性能常识,以及每个候选优化到底在优化什么。假设读者不了解 DSV3.2,但写过 PyTorch。

---

## 1. DeepSeek V3.2 是什么样的模型

### 1.1 基本参数

DeepSeek V3.2 沿用 V3 的主体结构:671B 参数的 MoE 模型,每 token 激活约 37B,61 层,hidden size 7168,官方权重是 FP8。它在 V3 之上新增了 **DSA(DeepSeek Sparse Attention)**,这是 V3.2 的核心变化。

一个 decoder 层 = Attention(MLA)+ MoE FFN。我们关心的优化全部在 attention 和 speculative decoding 路径上,MoE 不涉及。

### 1.2 MLA:Multi-head Latent Attention

普通 MHA 的 KV cache 按 `[层数 × token 数 × head 数 × head_dim × 2]` 存,长上下文时显存和带宽都吃不消。MLA 的思路:**不存每个 head 的 K/V,只存一份压缩后的 latent 向量**,用的时候再投影回来。

关键维度(vLLM 里定义在 `vllm/model_executor/layers/attention/mla_attention.py`):

| 名称 | 值 | 含义 |
|---|---|---|
| `kv_lora_rank` | 512 | KV 压缩 latent 的维度,**KV cache 每 token 只存这 512 维** |
| `qk_rope_head_dim` | 64 | 带 RoPE(位置编码)的那部分 K,和 latent 一起缓存 |
| `q_lora_rank` | 1536 | Q 的低秩压缩维度 |
| `qk_nope_head_dim` | 128 | 不带 RoPE 的 Q/K 维度 |
| `v_head_dim` | 128 | V 维度 |
| head 数 | 128 | |

效果:每 token 的 KV cache 从 MHA 的几十 KB 降到 `512 + 64 = 576` 个数。代价是 attention kernel 更复杂:decode 时 128 个 Q head 共享同一份 latent KV 流,从 kernel 视角看**像 MQA**(多 Q 单 KV)。所以 MLA 的 attention metadata 主要是 `block_table`(token → 物理 KV 块的映射)和 `seq_lens`,没有 per-head 的东西。

### 1.3 DSA:DeepSeek Sparse Attention(V3.2 新增)

Dense attention 每个 query 要看全部 context,复杂度 O(L²)。DSA 的做法:**先用一个便宜的小模块给所有历史 token 打分,只挑 top-k(k=2048)个,再只对这 2048 个做完整的 MLA attention**,复杂度降到 O(L·k)。

打分模块叫 **lightning indexer**,结构上是个轻量 attention:

- 64 个 index head,head_dim 128,FP8 计算;
- 每层 attention 前先跑:`index_score = indexer(q, 历史所有 token 的 index_k)`;
- 取 top-2048 的 token 下标 → 通过 block_table 换算成 KV cache 里的物理位置 → sparse MLA 只读这些位置。

indexer 自己也有一份小的 K cache(FP8 + 每 128 元素一个 scale,即代码里的 `ue8m0` 格式)。

vLLM 里 indexer 的两部分:

- **模型侧**:`vllm/model_executor/models/deepseek_v2.py` 里的 `Indexer` 模块——算 index Q/K、量化、打分、top-k。这是**候选 4(K 路径融合)**的所在地。
- **调度侧**:`vllm/v1/attention/backends/mla/indexer.py`——为 indexer kernel 准备 metadata(每个请求看多长的历史、block table 怎么展开)。这是**候选 1 和候选 3**的所在地。

### 1.4 MTP:Multi-Token Prediction(投机解码)

V3/V3.2 权重里自带一个 **MTP 模块**:一个额外的轻量 decoder 层,输入是"当前 token 的 embedding + 主模型最后的 hidden state",输出对**下一个之后**那个 token 的预测。推理时把它当草稿模型做投机解码(speculative decoding):

1. 主模型(target)正常出一个 token;
2. MTP 模块连跑 `num_speculative_tokens` 步,猜出后面几个 token(draft);
3. 下一步主模型一次 forward 同时验证这几个猜测(verify),猜对的全收下,猜错的丢弃(reject)。

猜对时一步 decode 产出多个 token,TPOT 直接除以接受数。

MTP 模块的入口结构(`vllm/model_executor/models/deepseek_mtp.py:110-114`)正是**候选 2**:

```python
inputs_embeds = self.enorm(inputs_embeds)                # RMSNorm ①
previous_hidden_states = self.hnorm(previous_hidden_states)  # RMSNorm ②
hidden_states = self.eh_proj(
    torch.cat([inputs_embeds, previous_hidden_states], dim=-1))  # 拼接 ③ + GEMM
```

**重要副作用**:开了 MTP 后,一个"decode 请求"每步不再是 1 个 query token,而是 `1 + num_speculative_tokens` 个(本 token + 待验证的草稿)。而且被 reject 之后各请求的有效长度不一样。这就是后面 "decode expansion / uniform vs variable" 问题的根源。

---

## 2. vLLM 怎么跑这个模型

### 2.1 每一步 decode 的流水线

```
scheduler(CPU,Python)          model runner(CPU 发射 + GPU 执行)
┌─────────────────────┐        ┌──────────────────────────────┐
│ 决定这步跑哪些请求    │──────→ │ 构建 attention metadata       │
│ 维护 block table     │        │ (CPU 上算,拷到 GPU buffer)   │
│ 生成 CPU 侧长度数组   │        │ 发射/replay 模型 forward       │
└─────────────────────┘        │ 采样 → 结果异步拷回 CPU        │
                               └──────────────────────────────┘
```

关键数据结构 `CommonAttentionMetadata`(`vllm/v1/attention/backend.py`)同时持有 GPU tensor 和 **CPU mirror**:

- `query_start_loc` / `query_start_loc_cpu`:每个请求 query token 的累积偏移;
- `seq_lens`(GPU)/ `seq_lens_cpu_upper_bound`(CPU 上界);
- `block_table_tensor`、`slot_mapping`。

**CPU mirror 是 scheduler 自己写的,不是从 GPU 拷回来的**——scheduler 本来就在 CPU 上决定了每个请求多长,这些数字天然在 CPU。这个事实是候选 1 的结论的核心。

### 2.2 CUDA graph

decode 每步的 kernel 序列基本固定,Python 逐个发射 kernel 的开销(每个几微秒 × 几百个 kernel)反而成了大头。CUDA graph 把整段 kernel 序列**录制(capture)一次**,之后每步整体**重放(replay)**,一次 API 调用发射全部工作。

代价是 replay 时**形状和地址必须和录制时一致**:所以 vLLM 为一组固定的 batch size 各录一张图,输入写进预分配的静态 buffer,batch 不足就 padding 到最近的录制尺寸。vLLM 有 FULL(整个 forward 一张图)和 PIECEWISE(只录 attention 之间的部分)两种模式。

由此产生两类性能敏感点:

1. **replay 前的准备工作**(把这步的数据写进静态 buffer、算 metadata)是每步都要跑的纯 CPU + 小 kernel 工作,不在图里,快慢直接影响步间间隙;
2. 每张图占 GPU 内存,录几十张图内存可观——这是**候选 5(graph 去重)**的动机。

### 2.3 CPU-GPU 异步与"同步"的代价

CUDA 的执行模型:CPU 把 kernel 扔进 stream 队列就返回,GPU 在后面慢慢消化。理想状态下 CPU 永远跑在 GPU 前面几步,GPU 一刻不闲。

以下操作会**强制 CPU 停下来等 GPU**(同步),制造 GPU 空转气泡:

- 对**依赖 GPU 计算结果**的 tensor 调 `.item()` / `.cpu()` / `.tolist()`——CPU 必须等 GPU 算完才能拿到值;
- `torch.tensor(python_list, device="cuda")`——从可分页(pageable)host 内存发起拷贝,可能触发 `cudaStreamSynchronize`。

**但反过来不成立**:对一个本来就在 CPU 上的 tensor 调 `.item()`,只是普通内存读取,微秒级,和 GPU 无关。**判断一个 `.item()` 是否有害,唯一标准是这个 tensor 的数据从哪来。** SGLang 那两个被移植的 commit 修的是前者;vLLM 对应位置读的都是 scheduler 自产的 CPU 数据,属于后者——这就是候选 1 要用 profiling 最终确认的事。

### 2.4 Kernel launch 开销与融合

GPU kernel 不论多小,发射一次约 3-10µs 开销;小 kernel(比如对 8×7168 的 tensor 做 RMSNorm,本身 <5µs)的实际成本几乎全是发射和访存往返。**融合(fusion)**= 把相邻的几个小 kernel 合写成一个:省发射次数,更重要的是中间结果不落显存(省一读一写)。

decode batch 小的时候整个模型是 memory-bound,这类融合是主要的微优化手段。候选 2、3、4 全是这个类型。

---

## 3. 六个候选优化逐个解释

### 候选 1:确认 CPU 读取不引发同步(验证性质,预期"无需修改")

- **SGLang 原型**:`e671154ae1` 把 replay 时 page-table 拷贝宽度从 `seq_lens_cpu.max().item()` 换成静态 buffer 宽度,消掉了每步毫秒级的同步。
- **vLLM 对应点**:`indexer.py:377,407,572`、`flashinfer_sparse.py:480,505`、`dflash/speculator.py:249` 的各种 `.item()`。
- **代码分析结论**:这些 tensor 全是 scheduler 自产 CPU 数据(§2.3),`.item()` 应该无害。
- **profiling 要回答的**:decode 稳态每步之间有没有异常的 `cudaStreamSynchronize`。没有 → 原调研笔记正式关闭。

### 候选 2:MTP enorm/hnorm 融合(最小可行 PR)

- **SGLang 原型**:#29667 fused EH norm。
- **现状**:§1.4 那三行,每个 MTP draft step 跑 2 次 RMSNorm + 1 次 `cat`,3 个 kernel。
- **修法**:单 kernel 对两个输入各做 RMSNorm、直接写入拼接布局的输出,3 launch → 1 launch,且省掉 `cat` 的整读整写。
- **风险**:低,数值等价。
- **profiling 要回答的**:这 3 个 kernel 占 draft step 的时间比例,决定值不值得做。

### 候选 3:DSA decode expansion 的 variable 路径融合

背景:sparse MLA 的 decode kernel 每次只处理 1 个 query token,而 MTP 让每请求每步有 `1+k` 个 token(§1.4),所以 metadata 构建时要把"1 个请求 × n 个 token"**展开**成"n 个单 token 条目"(复制 seq_lens、block_table 各 n 份,这就是 decode expansion,在 `indexer.py`)。

- **uniform 路径**(没人被 reject,所有请求都是 n 个):已有单个 Triton kernel `_prepare_uniform_decode_kernel`,无需动。
- **variable 路径**(有 reject,各请求长度不一):现状是 `repeat_interleave` + 七八个零散 torch 小 op,每步多次 launch。
- **SGLang 原型**:#29499,620 行的融合 Triton kernel(`dsa_metadata.py`),可直接参考,还附了 474 行单测。
- **风险**:中——变长展开的边界逻辑(`actual_expanded` 是精确输出尺寸)必须写对。
- **profiling 要回答的**:实际负载下 variable 路径触发频率(取决于 reject 率)和这段的开销占比。

### 候选 4:Indexer K 路径融合

背景:indexer 每层要把当前 token 的 index Q/K 准备好(§1.3):归一化 → RoPE 加位置 → 量化成 FP8(ue8m0 scale 格式)写入 index cache。

- **现状**(`deepseek_v2.py` 的 `Indexer.forward`):Q 路径已有融合 kernel `fused_indexer_q_rope_quant`(norm+rope+quant 一把);**K 路径还是分离的** `k_norm → split → rope → quant`,每层 3-4 个小 kernel × 61 层。
- **SGLang 原型**:#27705 把 Q、K 都融了;vLLM 只差 K 这一半。
- **优势**:仓库里就有 Q 侧先例,kernel 结构、测试模式照抄即可,落地阻力最小。
- **风险**:低-中,FP8 量化输出要和现路径 bit 一致。

### 候选 5:CUDA graph executable 去重(独立项,收益面最大)

- **SGLang 原型**:#29625。多张 batch size 不同的图,kernel 拓扑其实一样,只是参数不同。`cudaGraphExecUpdate` 允许"用新录的图更新旧 executable 的参数"而不重新实例化,于是 N 张图可以共享 executable,省 GPU 内存和 capture 时间。
- **vLLM 现状**:全库无此机制,每个 (mode × size) 独立持有 executable。
- **状态**:与本次 profiling 无关;工程量中等(SGLang 约 400 行 mixin);动手前必须去 upstream 查重。

### 候选 6:FlashInfer prefill plan 免同步(暂缓)

FlashInfer 的 attention wrapper 每次形状变化要调 `plan()` 做 CPU 规划。vLLM 已有 `fast_decode_plan`,但 EAGLE draft-extend 用的 prefill wrapper 还是标准 `plan()`。只影响 EAGLE + FlashInfer 组合,和 DSV3.2 无关,单独验证。

---

## 4. Profiling 怎么回答这些问题

用 vLLM 内置 torch profiler(server 启动带 `--profiler-config`,压测中 `POST /start_profile` / `/stop_profile`,见 `profile_dsv32.sh`)。拿到 trace(Perfetto 打开)后:

| 问题 | 在 trace 里看什么 |
|---|---|
| 候选 1:有没有隐藏同步 | CPU 线程上 `cudaStreamSynchronize` / `aten::_local_scalar_dense` 的位置和耗时。µs 级=无害,ms 级=在等 GPU |
| 候选 2:MTP norm 占比 | draft step 内 2 个 RMSNorm + `cat` kernel 的耗时 ÷ draft step 总时长 |
| 候选 3:variable 路径频率与开销 | `repeat_interleave` 等 op 出现的步数比例、每步的 kernel 数和 CPU launch 时间 |
| 候选 4:K 路径占比 | `k_norm`/rope/quant 小 kernel 每层耗时 × 61 层 ÷ decode step 总时长 |
| 总体 | 每步 GPU 时间线的空隙(气泡)有多宽,气泡期间 CPU 在干什么 |

判断标准:某项占稳态 decode step 时间 **>2-3%** 才值得开 PR;否则记录数据、关闭该项。所有动手项开工前按 AGENTS.md 用 `gh pr list --search` 查重。

---

## 5. 附:读 trace 时会看到的双流并行(MoE 共享专家 overlap)

打开 trace 会看到 GPU 侧有**两条 stream**,同一个 `execute_context_..._generation_N(M)` graph-replay span 同时罩在两行上:主流跑 `fused_moe_kernel`(路由专家),辅助流同时跑 `deep_gemm::fp8_gemm_kernel_swapAB<7168u,...>`(shared expert 的稠密 GEMM,7168 = hidden size)。这是 vLLM 故意做的重叠,不是异常。实现分布在三个文件:

- `vllm/utils/torch_utils.py:745` — 辅助流创建
- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` — fork/join 逻辑
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` — MoE forward 编排

### 5.1 整体时序

```
主 stream:   [fork点] → gate/router → select_experts → fused_moe_kernel → [join点] → combine(相加)
                 ↘ (event)                                                   ↗ (event)
辅助 stream:      wait ──→ shared expert 的 GEMM (deep_gemm fp8) ──────────→
```

背景:DeepSeek 的 MoE 层每个 token 除了走 top-8 路由专家,还要过一个所有 token 共用的 shared expert。两者无数据依赖,可以并行。

### 5.2 辅助流是进程级单例

```python
# torch_utils.py:745
_aux_stream: torch.cuda.Stream | None = None
def aux_stream() -> torch.cuda.Stream | None:
    global _aux_stream
    if _aux_stream is None and current_platform.is_cuda_alike():
        _aux_stream = torch.cuda.Stream()
    return _aux_stream
```

全局只建一条辅助流,61 层 MoE 共用——所以 trace 里只有两行,不是 61 行。

### 5.3 何时启用(`shared_experts.py:99`)

```python
should_run_shared_in_aux_stream = (
    current_platform.is_cuda()
    and self._stream is not None
    and hidden_states.shape[0] <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
)
```

**token 数小于阈值才开**:大 batch(prefill)时路由专家已吃满 SM,并行只会抢资源;小 batch(decode)时 GPU 大量空闲,并行才有收益。另有两个兜底模式(`SharedExpertsOrder`):`NO_OVERLAP`(串行,EPLB 非默认后端等正确性受限场景)和 `MK_INTERNAL_OVERLAPPED`(EP 场景由 modular kernel 和 dispatch/combine 通信重叠)。

### 5.4 Fork:辅助流只等"输入就绪"这个点(`shared_experts.py:111`)

`_forward_impl`(`moe_runner.py:809`)在 gate 之前调用:

```python
def maybe_sync_shared_experts_stream(self, shared_experts_input):
    if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
        shared_experts_input.record_stream(self._stream)   # ① 显存生命周期保护
        self._stream.wait_stream(current_stream())          # ② fork 点
```

- ② 在主流当前位置记一个 event,辅助流插一条 wait。含义:辅助流之后提交的工作只等"主流到此为止"(hidden_states 已算好),不等主流后面的 router/专家计算。
- ① 是分配器保护:PyTorch 缓存分配器默认认为 tensor 只在创建流上使用,`record_stream` 防止主流侧提前释放这块显存导致辅助流读到被复写的数据。

### 5.5 Join 与关键理解:CPU 发射顺序 ≠ GPU 执行顺序

fork 之后主流依次发射 gate、`select_experts`、`forward_modular`(即 `fused_moe_kernel`),**最后**才轮到共享专家(`shared_experts.py:131`):

```python
with torch.cuda.stream(self._stream):      # 切换发射目标到辅助流
    output = self._layer(shared_experts_input)   # shared expert 的 GEMM 们
current_stream().wait_stream(self._stream)  # ③ join 点
```

Python 里共享专家在 `fused_moe_kernel` 之后才被调用,但那只是 CPU 塞队列的顺序。这些 kernel 进的是辅助流,其唯一依赖是 ② 的 fork event(gate 之前就记下了),所以 GPU 实际执行时与主流的 router/`fused_moe_kernel` 并行。③ 让主流等辅助流完成,因为紧接着 `_maybe_combine` 要把两路输出相加(也因此输出侧不需要再 `record_stream`,join 已建立顺序)。

### 5.6 与 CUDA graph / torch.compile 的配合

1. `moe_forward` / `moe_forward_shared` 注册为**不透明 custom op**(`moe_runner.py` 注释:"being opaque custom ops is a load-bearing assumption for the dual-stream path"),多流逻辑藏在 op 内部,防止 inductor 跨流重排破坏依赖。
2. `wait_stream` 底层就是 CUDA event record/wait,可被 graph capture。capture 时双流结构连同跨流依赖一起录进图,replay 原样重现——所以同一个 replay span 同时出现在两条 stream 行上。

### 5.7 读 trace 时可顺手验证的点

join(③)紧跟在共享专家发射之后:如果某 batch size 下辅助流的 GEMM 比主流 `fused_moe_kernel` 还慢,主流会在 join 处空等出气泡;健康状态是辅助流的 `deep_gemm` 完全被 `fused_moe_kernel` 时长罩住,overlap 等于免费。若发现辅助流 kernel 被排到主流 kernel 之后(退化成串行),值得单独报 issue。

## 6. 附:EP 场景的另一种 overlap——共享专家 ∥ dispatch/combine 通信(MK_INTERNAL_OVERLAPPED)

§5 讲的 `MULTI_STREAM_OVERLAPPED` 是**计算 ∥ 计算**,靠第二条 stream。EP(专家并行)场景下 vLLM 用的是另一条路:`MK_INTERNAL_OVERLAPPED`,**计算 ∥ 通信**,只用一条计算流。我们的 TP8 部署不走这条路(没有跨卡 all2all),但读 EP 部署的 trace 或看这段代码时需要知道它的存在。

### 6.1 背景:EP 下 MoE 多了两次 all2all

EP 把 256 个路由专家切到多张卡上,每张卡只放一部分。于是每层 MoE 多出两步跨卡通信:

```
router 选出 top-8 专家
  → dispatch:把每个 token 的 hidden state 发到"它的专家所在的卡"(all2all)
  → 本地专家 GEMM
  → combine:把各卡算出的结果发回 token 原来的卡并加权求和(all2all)
```

而 shared expert 是每张卡本地就有全量权重的稠密 MLP,**不需要通信**。它与 combine 无数据依赖——这就是 overlap 的空间。

### 6.2 选路逻辑:MK_INTERNAL 优先于双流

`shared_experts.py:96` 的 `_determine_shared_experts_order` 里,这个分支排在双流判断**之前**:

```python
if self._mk_can_overlap_shared_experts():
    return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
```

条件最终落到 `modular_kernel.py:1565`:

```python
@property
def can_overlap_shared_experts(self) -> bool:
    if isinstance(self.impl, FusedMoEKernelModularImpl):
        return self.impl.prepare_finalize.supports_async()
    return False
```

即:只要当前 all2all 后端实现了**异步版**的 prepare/finalize,就走这条路。实现了 `supports_async` 的正是各 EP 通信后端:`deepep_ht` / `deepep_ll` / `deepep_v2` / `mori` / `nixl_ep`(`fused_moe/prepare_finalize/` 目录)。没有 EP 时(如我们的 TP8)后端不支持 async,才轮到 §5 的双流判断。

### 6.3 核心机制:通信 kernel 拆成"发"和"等"两半

每个 async 后端把 dispatch/combine 拆成 `(hook, receiver)` 两段(`modular_kernel.py:302` / `:379` 的接口约定):

- `prepare_async()` / `finalize_async()`:发射"发送"部分后**立刻返回**,数据还没到;
- `hook()` + `receiver()`:之后再调,等数据到齐。两段合起来严格等价于同步版 `prepare()` / `finalize()`。

以 `deepep_ll.py:399` 的 combine 为例:

```python
_, _, recv_hook = self.buffer.low_latency_combine(
    fused_expert_output, ...,
    async_finish=False,
    return_recv_hook=True,   # 关键:只发,不等
    out=output,
)
return recv_hook, lambda: None
```

`return_recv_hook=True` 让 DeepEP 的 combine kernel 只负责把数据交给 NVLink/RDMA 发出去;"等对端数据到达"的那段 kernel 推迟到 `recv_hook()` 被调用时才发射。

### 6.4 共享专家插在"发"和"等"之间

`modular_kernel.py:1338`(`_finalize`)是 overlap 真正发生的地方:

```python
finalize_ret = self.prepare_finalize.finalize_async(...)   # ① combine 发送已发射
self._maybe_apply_shared_experts(shared_experts, ...)      # ② 共享专家 GEMM 入队
hook()                                                     # ③ 发射"等接收完成"的 kernel
receiver()                                                 # ④ output 此后才有效
```

②里 shared experts 以 `MK_INTERNAL_OVERLAPPED` order 被调用(`modular_kernel.py:1113`),走的是 `shared_experts.py:172` 的普通分支——**直接在当前流上执行**,没有 aux stream。

为什么单流也能重叠?因为①的发送 kernel 把数据交给 DMA 引擎/NIC 后很快就退出了,**数据传输不占 SM**。于是 GPU 时间线是:combine 发送 kernel(短)→ 共享专家 GEMM(此时字节还在 NVLink/RDMA 链路上飞)→ recv 等待 kernel。GEMM 的计算时间把通信延迟藏掉了。这也解释了它和 §5 的两点差异:

- **不需要 token 阈值**:通信不抢 SM,任何 batch size 下 overlap 都近乎免费(§5 的双流是两份计算抢 SM,才要限小 batch);
- **trace 里只有一条计算流**:识别特征不是第二条 stream,而是同一条流上 combine send 与 recv/wait 之间夹着 shared expert 的 GEMM。

注意这条路只把共享专家藏进 **combine**;dispatch 侧的 `prepare_async`(`modular_kernel.py:1178`)拆出的空档目前留给 DBO 用——开 DBO 时 `hook` 被注册到 ubatch 上下文,当前 micro-batch 等通信期间直接切去算另一个 micro-batch,是比塞一个 shared expert 更大粒度的通信隐藏。
