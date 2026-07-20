# vLLM MLA/DSA 背景与两个 SGLang 微优化

这份笔记用于判断两个 SGLang commit 是否能以类似 vLLM #47202 那种小粒度性能 PR 的方式移植到 vLLM。

两个 SGLang commit 是：

- `4f1854fc8a`: `[DSA] Remove host H2D sync in _apply_cuda_graph_metadata`
- `e671154ae1`: `[DSA] Use static page-table width in CUDA graph replay to drop seq_lens_cpu host read`

简短结论：

- 这类优化模式对 vLLM 是成立的：cuda graph replay 路径不应该从 Python list 创建 CUDA tensor，不应该对 GPU 派生值调用 `.item()`，也不应该在 captured graph 已经有固定 shape 时，从 CPU mirror 动态推导静态 buffer 宽度。
- vLLM 没有和 SGLang 完全一致的 `_apply_cuda_graph_metadata` 函数。对应位置主要是 vLLM 的 `CommonAttentionMetadata`、MLA metadata builder、DeepSeek V3.2 indexer metadata、DeepSeek V4 sparse MLA metadata，以及 speculative decode proposer 路径。
- `e671154ae1` 比 `4f1854fc8a` 更直接映射到 vLLM：vLLM 里已经有一些 `seq_lens_cpu_upper_bound.max().item()` / `query_lens_cpu.max().item()` 风格的位置。这些在普通 metadata 构建阶段可能没问题，但在 full cuda graph replay 或 spec decode 热路径里就值得怀疑。

## 1. vLLM 里的 MLA 是什么

MLA 是 Multi-head Latent Attention。DeepSeek 系列模型使用 MLA 来降低 KV cache 大小和带宽。

普通 MHA 中，每个 token 会为每个 KV head 存 K 和 V。MLA 中，模型存的是压缩后的 KV latent 表示，然后在 attention kernel 路径中重建或组合需要的 attention 分量。这会同时改变 tensor layout 和 attention backend 的接口约定。

vLLM 中关键维度定义在：

`vllm/model_executor/layers/attention/mla_attention.py`

- `q_lora_rank`: 可选的低秩 query projection rank。
- `kv_lora_rank`: 压缩 KV latent size。
- `qk_nope_head_dim`: 不使用 RoPE 的 query/key 维度。
- `qk_rope_head_dim`: 使用 RoPE 的 query/key 维度。
- `v_head_dim`: value 维度。

对于 DeepSeek V2/V3 风格配置，这些字段通常直接来自 HF config。对于 DeepSeek V4 风格配置，vLLM 会从统一的 `head_dim` 和 `qk_rope_head_dim` 推导这些维度。

重要的实现拆分是：

- `MLACommonMetadata`: MLA backend 的通用 per-layer metadata。
- `MLACommonMetadataBuilder`: 从 `CommonAttentionMetadata` 构建 decode 和 prefill metadata。
- FlashMLA、FlashInfer MLA、TRTLLM ragged、ROCm AITER、sparse MLA 等 backend-specific builder 会专门处理各自需要的 metadata。

在 decode 阶段，从 attention kernel 的视角看，MLA 通常更像 MQA：一个 KV stream 被多个 query head 共享。因此 metadata 经常包含的是 `block_table`、`seq_lens` 和 scheduler-specific metadata，而不是完整的 per-head KV table。

## 2. 这里的 DSA 是什么

这里的 DSA 指 DeepSeek Sparse Attention。它是 DeepSeek V3.2、DeepSeek V4、GLM-5.x 类 sparse MLA 变体以及相关 speculative decode 路径使用的 sparse MLA path。

Dense MLA 会 attend 整个 context。Sparse MLA 会先选择一部分 token 或压缩 block，再只对这个 sparse subset 做 attention。这通常会引入一个 indexer 路径：

1. 根据每个 request 当前 position 和 sequence length，计算相关 token 或 block 的 top-k indices。
2. 通过 `block_table` 把逻辑 token/block index 转成物理 KV cache slot。
3. 使用选出的 KV positions 启动 sparse attention。

vLLM 中 DSA 相关的主要文件：

- `vllm/v1/attention/backends/mla/indexer.py`: DeepSeek V3.2/V4 indexer metadata 和 decode expansion。
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`: 通用 FlashInfer sparse MLA backend。
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`: FlashMLA sparse path。
- `vllm/models/deepseek_v4/sparse_mla.py`: DeepSeek V4 sparse MLA metadata。
- `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`: DeepSeek V4 NVIDIA FlashInfer sparse MLA 执行路径。
- `vllm/v1/spec_decode/llm_base_proposer.py` 和 DFLASH speculator 文件：speculative decode 输入准备。

## 3. Metadata 术语

vLLM 的核心对象是 `vllm/v1/attention/backend.py` 里的 `CommonAttentionMetadata`。

重要字段：

- `query_start_loc`: GPU tensor，表示 query token 的 cumulative offsets。shape 是 `[num_reqs + 1]`。
- `query_start_loc_cpu`: 同一份 cumulative offsets 的 CPU mirror。
- `seq_lens`: GPU tensor，表示当前 sequence lengths。
- `seq_lens_cpu_upper_bound`: `seq_lens` 的 CPU upper bound。对 prefill row 和普通 decode row 是精确的，但对 async speculative decode row 可能是乐观上界。
- `max_query_len`: Python int，供 kernel 和 metadata builder 使用。
- `max_seq_len`: Python int，表示最大 context length 或一个 upper bound。
- `block_table_tensor`: 把 request/block index 映射到物理 KV cache block。
- `slot_mapping`: 把 token position 映射到 KV cache slot。

这两个细节对后面的优化很关键：

1. vLLM 有意保留 CPU mirror，用于调度和 metadata 构建。使用 CPU 本身并不自动代表有问题。
2. 当 CPU read 发生在 GPU work 已经排队之后的 replay critical path，或者会迫使 graph replay 等待 stream 时，它才会变成性能问题。

## 4. 为什么 cuda graph replay 对这些操作敏感

CUDA graph capture 会固定 kernel launch 结构和 tensor 地址。replay 时，vLLM 应该原地更新预分配 buffer，然后用稳定 shape replay 已 capture 的工作。

常见性能陷阱：

- 从 Python list 创建 CUDA tensor：

  ```python
  torch.tensor([num_draft_tokens] * bs, dtype=torch.int32, device=device)
  ```

  这会从 pageable host memory 出发，可能插入 blocking H2D 路径。

- 对依赖已排队 GPU work 的数据调用 `.item()`、`.tolist()`、`.numpy()` 或 `.cpu()`。这可能强制 D2H sync，并让 CPU 调度和 GPU stream 串行化。

- captured destination buffer 已经有静态宽度时，仍然从动态 CPU mirror 推导 copy buffer 的大小。

更安全的模式：

- 对常量 tensor 使用 device-side 创建：

  ```python
  torch.full((bs,), num_draft_tokens, dtype=torch.int32, device=device)
  ```

- 使用预分配 buffer，并通过 `copy_` 写入稳定地址。
- 当 kernel 已经用 per-row `seq_lens` 限制读取时，使用 captured buffer 的静态 shape，例如 `metadata.block_table.shape[1]`。
- 在 full cuda graph capture/replay 期间，使用保守的 Python upper bound，例如 `max_model_len` 或 captured buffer width。

## 5. SGLang commit `4f1854fc8a`

### 改了什么

SGLang 有两个 DSA replay 分支：

- target verify
- draft extend v2

这两个分支都会构建一个每个 request 都相同的 query/output length tensor，原来写法类似：

```python
torch.tensor([self.speculative_num_draft_tokens] * bs,
             dtype=torch.int32,
             device=self.device)
```

这个 commit 改成：

```python
torch.full((bs,),
           self.speculative_num_draft_tokens,
           dtype=torch.int32,
           device=self.device)
```

值完全一样。差异在于 tensor 从哪里创建：

- `torch.tensor(list, device="cuda")` 从 host data 开始。
- `torch.full(..., device="cuda")` 是 device-side fill operation。

SGLang commit message 里说，旧路径在 graph replay 期间触发 pageable H2D copy 和 `cudaStreamSynchronize`，在对应 GB200 workload 中每步大约损失 8.5 ms。

### 映射到 vLLM

这次扫描没有在 vLLM 主要 sparse MLA 路径中发现完全相同的模式。相关搜索命中大多是无害的一次性 scalar/dummy tensor 构造，例如 dummy input ids。

仍然需要重点检查的位置：

- `vllm/v1/spec_decode/llm_base_proposer.py`
- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/models/deepseek_v4/sparse_mla.py`
- `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`

搜索模式：

```bash
rg -n "torch\.tensor\(.*device=|torch\.as_tensor\(.*device=" \
  vllm/vllm/v1/attention/backends/mla \
  vllm/vllm/models/deepseek_v4 \
  vllm/vllm/v1/spec_decode \
  vllm/vllm/v1/worker/gpu/spec_decode
```

一个 vLLM PR 是否成立，取决于这些条件：

- tensor 创建发生在 decode/spec decode/cuda graph replay 热路径中。
- 输入是 Python list、NumPy array 或 CPU tensor，并且内容是常量，或可以从 scalar 参数生成。
- 这个 tensor 会立刻被 GPU kernel 使用，或 copy 进 persistent CUDA graph buffer。

对应修复应是下面之一：

- `torch.full(shape, scalar, dtype=..., device=...)`
- `torch.arange(..., device=...)`
- 对预分配 device buffer 调 `.fill_(scalar)`
- 复用已有 persistent buffer 并原地写入

如果找到具体位置，这会是一个很小的 PR。

## 6. SGLang commit `e671154ae1`

### 改了什么

SGLang DSA replay 需要把 page-table rows copy 到 captured metadata buffer。commit 之前，copy width 来自 CPU sequence lengths：

```python
max_len = int(seq_lens_cpu.max().item())
page_indices = self.req_to_token[req_pool_indices, :max_len]
metadata.page_table_1[:, :max_len].copy_(page_indices)
```

这会在每个 replay step 读取 CPU mirror。在受影响配置下，它会让 scheduler CPU work 和 queued GPU work 串行化。

commit 把 width 改成 captured buffer width：

```python
max_len = metadata.page_table_1.shape[1]
page_indices = self.req_to_token[req_pool_indices, :max_len]
metadata.page_table_1[:, :max_len].copy_(page_indices)
```

这个改法成立，是因为 attention/indexer kernel 不会盲目消费整行。它们使用 GPU `cache_seqlens` 限制每个 request 的有效读取范围。只要满足以下条件，把额外 padding columns copy 进预分配 page table 是安全的：

- destination buffer capture 时已经预留足够宽度；
- source table 有对应宽度；
- kernel 仍然尊重 per-row sequence lengths 或 valid lengths。

这个 commit 还让不再需要 `seq_lens_cpu` 的 replay 分支可以接受 `seq_lens_cpu=None`。

### 映射到 vLLM

vLLM 有几类相关模式。有些只是普通 metadata 构建，有些在 spec decode/cuda graph replay 中更可疑。

这次扫描最相关的位置：

- `vllm/v1/attention/backends/mla/indexer.py`
  - `decode_lens_cpu.min().item()`
  - `decode_lens_cpu.sum().item()`
  - `decode_lens_cpu.max().item()`
  - 这些控制 DeepSeek V3.2/V4 indexer decode expansion。
- `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
  - `int(decode_lens_cpu.max().item())` 作为 decode 的 `max_q_len` 传入。
  - `int(prefill_lens_cpu.max().item())` 作为 prefill 的 `max_q_len` 传入。
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
  - `input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()` 用来设置 draft max seq len。
- `vllm/v1/spec_decode/llm_base_proposer.py`
  - 构造新的 `CommonAttentionMetadata`，并从 CPU tensor 计算 `max_query_len` / `max_seq_len`。
- `vllm/model_executor/layers/attention/mla_attention.py`
  - generic MLA builder 大量使用 CPU values 处理 prefill/chunked prefill。这部分不那么直接对应，因为 prefill metadata 往往确实需要 CPU planning 和可变 allocation。

最有希望的 vLLM 类比点，不一定是 generic metadata construction 中的 `max_seq_len`。更准确地说，是任何 decode-only 或 spec-decode replay 路径：在那里，一个静态 captured shape 可能可以替代 CPU max。

## 7. 如何判断 vLLM 某个位置能不能安全修改

对每个候选 `.max().item()` 或 CPU-derived width，回答这些问题：

1. 它是否位于 full cuda graph replay、piecewise graph replay，或者 per-step speculative decode 热路径中？
2. 这个值是否只用作 buffer copy width、launch upper bound 或 metadata max，而不是精确的逻辑 token count？
3. 下游 kernel 是否有 per-row valid lengths，例如 `seq_lens`、`decode_lens`、`topk_lens` 或 `query_start_loc`，用来限制真实读取？
4. 是否已经存在 captured/preallocated destination buffer，并且它的 shape 是安全上界？
5. 把 width 增大到静态 captured width，是否只是 copy padding 或传入保守 max，不改变输出？

如果五个问题答案都是 yes，那么 SGLang `e671154ae1` 的模式大概率适用。

如果这个值控制 allocation size、Python loop count、output slicing，或者 kernel 不检查 valid lengths，就不要在没有深入分析 kernel 的情况下修改。

## 8. 具体 vLLM PR 候选

### Candidate A: DeepSeek V4 FlashInfer sparse MLA `max_q_len`

文件：

- `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`

当前 decode 路径计算：

```python
decode_lens_cpu = decode_cu_cpu[1:] - decode_cu_cpu[:-1]
max_q_len = int(decode_lens_cpu.max().item())
```

潜在替换：

- 如果 decode 分支在 graph replay 中只用于 uniform query length，可以使用 metadata 中已知的静态 query width 或 `self.reorder_batch_threshold`。
- 对 DFLASH/spec decode，如果 graph 是用 padded query shape capture 的，可以使用 `input_batch`/metadata 的静态 query width。

风险：

- `max_q_len` 会传入 FlashInfer/TRTLLM launcher。必须确认 launcher 接受保守 upper bound，并使用 `cum_seq_lens_q` 表示每个 request 的精确 query length。
- 如果 launcher 需要精确 maximum 来做 workspace 或 scheduling，使用更大的静态 max 可能仍然正确，但可能带来性能回退。

这是一个很好的调查目标，但不适合盲改。

### Candidate B: DeepSeek V3.2 indexer decode expansion

文件：

- `vllm/v1/attention/backends/mla/indexer.py`

当前代码用 CPU decode lengths 计算：

- `min_decode_len`
- `max_decode_len`
- `actual_expanded`

潜在替换：

- 对 padded spec decode，如果 `max_decode_len` 已知为 `1 + num_speculative_tokens`，可以在 uniform path 避免 CPU max。
- 对 graph replay，优先使用固定 `next_n` 和预分配 buffer width，让 rejected/short rows 通过 padding 和 valid lengths 表示。

风险：

- variable-length path 使用 `actual_expanded = decode_lens_cpu.sum().item()` 作为 `repeat_interleave` 的精确 output size。除非下游 buffer 和 mask 已经能容忍 padded entries，否则这不等价于使用静态 width。
- 这里更可能需要一个更窄的 PR，只针对 uniform/padded path。

### Candidate C: DFLASH draft max sequence length

文件：

- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`

当前代码计算：

```python
max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
self.draft_max_seq_len = min(max_seq_len + self.num_query_per_req,
                             self.max_model_len)
```

潜在替换：

- 在 cuda graph replay 期间，使用 captured/static maximum，或者 scheduler 已经提供的 Python int upper bound。
- 在 graph replay 之外，当前 CPU upper bound 可能可以接受。

风险：

- 这个值可能控制的是模型输入限制，不只是 page-table width。保守 upper bound 大概率正确，但可能增加工作量或内存占用。

### Candidate D: spec proposer padded path metadata

文件：

- `vllm/v1/spec_decode/llm_base_proposer.py`

padded path 明确写着不应引入 blocking CPU operations。它仍然从 `query_start_loc_cpu` 计算 `max_query_len`，但这个 CPU tensor 是 scheduler-owned，不是 GPU readback。

潜在替换：

- 对 padded speculative decode，`max_query_len` 通常可以从 `num_speculative_tokens + 1` 或 graph key 得到。可尽量使用静态值，而不是从 CPU offsets 重新计算。

风险：

- 只有 profiling 显示 CPU 计算本身有影响时，这才有价值。除非这个 CPU tensor 本身来自 GPU data，否则这和 SGLang 的 D2H sync 不是同一类问题。

## 9. 推荐的第一个 PR 形态

最小但有价值的 PR 应该只做下面一件事：

1. 把一个具体的 `torch.tensor(list, device=cuda)` 热路径 allocation 替换成 `torch.full` 或 persistent device buffer。
2. 把 replay-only 的 `seq_lens_cpu.max().item()` page-table copy width 替换成静态 captured buffer width。
3. 在确认下游 kernel 使用精确 cu-seqlens 后，把 graph replay max query length 替换成已有 static graph key 或 padded query width。

不要把这类 PR 和 kernel 重写、backend selection 变化或新的 sparse attention 能力混在一起。

## 10. 验证计划

功能检查：

- 如果已有 DeepSeek V3.2/V4 sparse MLA 测试，运行这些测试。
- 用 deterministic sampling 在小 prompt set 上跑 spec decode correctness。
- 对比启用和禁用 cuda graph 的输出。

性能检查：

- 用 Nsight Systems profile replay 附近。
- 重点看：
  - `cudaStreamSynchronize`
  - pageable H2D copies
  - replay 前的 D2H memcpy 或 CPU wait
  - kernel launch count 是否增加
- 在同一套 setup 下对比修改前后的 TTFT 和 per-step decode latency。

有用的本地 audit 命令：

```bash
rg -n "torch\.tensor\(.*device=|torch\.as_tensor\(.*device=" \
  vllm/vllm/v1/attention/backends/mla \
  vllm/vllm/models/deepseek_v4 \
  vllm/vllm/v1/spec_decode \
  vllm/vllm/v1/worker/gpu/spec_decode

rg -n "\.max\(\)\.item\(\)|\.min\(\)\.item\(\)|\.sum\(\)\.item\(\)|tolist\(\)|\.numpy\(\)" \
  vllm/vllm/v1/attention/backends/mla \
  vllm/vllm/models/deepseek_v4 \
  vllm/vllm/v1/spec_decode \
  vllm/vllm/v1/worker/gpu/spec_decode
```

## 11. 实际 takeaway

对 vLLM，建议先做 `e671154ae1` 风格的工作：

- 找 decode/spec decode 路径中 CPU max 只用于 metadata copy sizing 或 launch upper bound 的位置。
- 用 graph-static width 或已知 padded width 替换它们。
- 证明 kernel 仍然用精确 per-row lengths 保证正确性。

然后机会性做 `4f1854fc8a` 风格的清理：

- 如果任何 replay 热路径从 Python list 构造 CUDA tensor，就改成 device-side fill 或预分配 buffer。

这样可以把工作保持在一个聚焦 TTFT/decode-latency PR 的粒度：一个 backend path、一个 synchronization source、一个 benchmark。
