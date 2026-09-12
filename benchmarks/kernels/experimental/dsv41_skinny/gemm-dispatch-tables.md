# DSV4.1 / K3 skinny GEMM dispatch 对照

2026-09-13。N 是输出宽度，K 是输入宽度；形状使用本卡逻辑权重 N×K，不取决于实际转置存储。M 是传入该 GEMM 的行数，不等于请求并发数。离散集合 1、2、4 不包含 3；“—”表示该后端没有替换条目。

## DSV4.1：本次 TP8 实测的 M≤4 候选

这是实验 worker 的分发表，主源码没有默认启用。按 bench-lowm/selected.json，8 类投影、20 个 (投影,M) 组合。其中三个 MXFP8 主投影使用本次编写的 Triton SIMT；BF16 路径复用 vLLM 已有 CuTe skinny、CuTe ll_bf16 或本次 Triton SIMT。

| 投影 | 本卡 N×K | 类型 | 替换后的实现 | 命中的 M |
| --- | ---: | --- | --- | --- |
| `attn.fused_wqa_wkv` | 1792×5120 | MXFP8 | Triton SIMT | 1 |
| `attn.wo_a` | 1024×4096 | MXFP8 | Triton SIMT；连同 inverse-RoPE/量化路径验证 | 1、2、4 |
| `attn.wo_b` | 5120×1024 | MXFP8 | Triton SIMT | 1、2 |
| `indexer.weights_proj` | 32×5120 | BF16 | CuTe skinny：M1/2；Triton SIMT：M4 | 1、2、4 |
| `indexer.wk` | 128×512 | BF16 | CuTe skinny | 1、2、4 |
| ratio-2 `compressor.fused_wkv_wgate` | 1024×5120 | BF16→FP32 | CuTe `ll_bf16` dotprod | 1、2、4 |
| ratio-1 `compressor.fused_wkv_wgate` | 512×5120 | BF16→FP32 | CuTe `ll_bf16` dotprod | 1、2、4 |
| `lm_head` | 16160×5120 | BF16 | CuTe skinny | 1、2 |

未替换的有效 dense GEMM：

| 投影 | 本卡 N×K | 保留原因 |
| --- | ---: | --- |
| 主注意力 `wq_b` | 4096×1280 | 当前候选未胜过已有预量化 MXFP8 路径 |
| `indexer.wq_b` | 4096×1280 | 当前候选未胜过已有预量化 MXFP8 路径 |
| `engram.wkv` | 25600×6144 | 当前候选无收益 |
| MoE router `ffn.gate` | 384×5120 | 已有 ll_bf16 低延迟路径，未找到达到纳入门限的改进 |

shared expert 已融入 MegaMoE，保留的独立 Linear 不执行，未作为可替换 GEMM 计数。mHC 的内部融合矩阵计算也不在此表中。

完整 28-cell 版本还替换了 M8/M16 的 indexer.weights_proj、indexer.wk（Triton SIMT）和两类 compressor（CuTe ll_bf16 split-K）。M≤4 对照移除了这些八个组合。尝试过的 CuTe MXFP8 改写没有胜出，因此没有进入任何 serving 候选。

源文件：

- [最终实测分发表](bench-lowm/selected.json)
- [完整试验分发表](selected.json)
- [后端分发](screen.py)
- [Triton 实现](kernels.py)
- [模型接入](dispatch.py)
- [完整 benchmark 报告](RESULTS.md)

## K3：PR #53534 的 B200 / SM100 基础表

从合入 commit bc2d63e 的 vllm/models/kimi_k3/nvidia/low_latency_gemm.py 提取；与当前工作区 KIMI_K3_PROJECTIONS_SM100 逐条一致，共 24 种 shape。CuTe skinny 与 CUDA C++ 的 dsv3_fused_a_gemm 是两个不同后端。

这些是未量化 BF16 Linear/LM head 的 shape 匹配条目，跨多个 TP 配置。实际模型只安装自身本卡形状匹配的条目，并须满足 dtype、布局、设备和后端可用性检查；不能把 24 种全部算作同一次 TP8 部署的实际调用。投影名是调试标签，真正按形状匹配。

| 投影标签 | 本卡 N×K | CuTe skinny 的 M | dsv3_fused_a 的 M |
| --- | ---: | --- | --- |
| `f_b_proj` | 1536×128 | — | 1、16 |
| `f_b_proj` | 3072×128 | — | 8 |
| `shared_gate_up_proj/mla_g_proj` | 1536×7168 | 1、2 | 4、8 |
| `shared_gate_up_proj` | 3072×7168 | 1、2 | — |
| `fused_qkv_a_proj` | 2112×7168 | 1、2 | 4、16 |
| `q_b_proj` | 2304×1536 | — | 1–16 |
| `q_b_proj` | 4608×1536 | — | 1、2、4 |
| `in_proj_qkvgfab` | 6288×7168 | 1 | — |
| `in_proj_qkvgfab` | 12448×7168 | 1、2、3、4 | — |
| `shared_down_proj` | 7168×768 | — | 1 |
| `o_proj` | 7168×1536 | 1 | — |
| `o_proj` | 7168×3072 | 1、2 | — |
| `routed_expert_up_proj` | 7168×3584 | 1、2 | — |
| `dense_down_proj` | 7168×8448 | 1、2、3 | — |
| `dense_gate_up_proj` | 8448×7168 | 1、2 | — |
| `dense_gate_up_proj` | 16896×7168 | 1 | — |
| `lm_head` | 20480×7168 | 1、2、3、4 | — |
| `lm_head` | 40960×7168 | 1、2、3、4 | — |
| `lm_head` | 10240×7168 | 1、2、3、4 | — |
| `routed_expert_down_proj` | 3584×7168 | 1、2、3 | — |
| `mla_g_proj/shared_gate_up_proj` | 768×7168 | 1、2、3、4 | — |
| `q_b_proj` | 1152×1536 | 1 | — |
| `in_proj_qkvgfab` | 3216×7168 | 1、2 | — |
| `dense_gate_up_proj` | 4224×7168 | 1、2、3 | — |

`routed_expert_down_proj/up_proj` 是 latent-MoE 的外围投影，并不表示整个专家 grouped GEMM 被替换。这是 #53534 的基础表；当前源码里另外存在的 KDA 投影拆分/并行路径不应归入该 PR 的原始收益。

参考：

- [PR #53534](https://github.com/vllm-project/vllm/pull/53534)
- [当前本地 SM100 表](../../../../vllm/models/kimi_k3/nvidia/low_latency_gemm.py#L328)
- [CuTe skinny](../../../../vllm/model_executor/kernels/linear/cute_dsl/_skinny_gemm.py)
- [CuTe ll_bf16](../../../../vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py)
- [CUDA dsv3_fused_a](../../../../csrc/libtorch_stable/dsv3_fused_a_gemm.cu)
