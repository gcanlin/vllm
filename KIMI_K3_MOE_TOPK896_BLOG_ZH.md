# 拆弹 MoE 路由：Kimi K3 的 896-expert topk 为什么慢了 2 倍，以及一个被流水线"吃掉"的优化

> 一句话总结：vLLM 的 fused topk gating kernel 只有一组编译期特化（1~128 的 2 次幂，
> 加 192/320/384/448/512/576），Kimi K3 的 896 个 routed expert 落在表外，每次路由
> 都要走"逐 expert 二 kernel"的动态慢路径。补上 `case 896` 这一个特化，路由算子本身
> 在 B200 上提速 1.9-4.6x（30.8us → 15.2us @M=64）。同时我们也把它放回生产形态做了
> E2E 验证，得到一个值得分享的结论：**在这个 2 节点 TP8+PP2 部署里，路由完全不在
> decode 关键路径上** —— 三层不同时延的路由实现（10.5us / 30.8us / 16us）E2E 完全打平。

## 起点：trace 里的反常 kernel

在优化 #1（fused KDA decode 支持 DS 布局）落地后，我们对 patched 栈做了一次 kernel
时间线拆解，想找下一个"显然该拿"的收益。MoE 路由这边有两个可疑对象：

1. Kimi K3 默认走 flashinfer 的 `TrtLlmMxfp4ExpertsMonolithic`，路由由 TRT-LLM 内置的
   `routingIndicesBlockKernel` 完成。这个 kernel 的 grid 是 `[1,1,1]` ——
   **单 CTA、896 个线程**干完整个 topk-16 路由，每次 10.5us，每步 decode 出现 46 次
   （每 PP rank 的 MoE 层数）。
2. 如果把路由挪回 vLLM 侧（`TrtLlmMxfp4ExpertsModular` + 外置路由），K3 的路由形状是
   896 experts / topk=16 / sigmoid+bias（noaux_tc 退化到 group=1），会落到
   `FusedTopKBiasRouter` → `fused_topk_bias` → `ops.topk_sigmoid`（`_moe_C` 扩展）。

拆开第二条路一看，问题来了。

## 慢在哪：896 不在特化表里

`csrc/libtorch_stable/moe/topk_softmax_kernels.cu` 的 `topkGatingKernelLauncher` 靠一个
switch 把 `num_experts` 映射到编译期模板实例：

```cpp
case 512:
    LAUNCH_TOPK(512, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
    break;
case 576:
    LAUNCH_TOPK(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
    break;
#endif
default: {
    STD_TORCH_CHECK(workspace != nullptr, ...);
    // 动态路径：moeSigmoidCompute + moeTopK 两个 kernel，
    // 每 thread block 只扫 expert 的一段，再全量扫一遍归并
    ...
}
```

模板实例存在的 expert 数：`1,2,4,...,128`（2 次幂）加上 `192,320,384,448,512,576`
（MULTIPLE_64 加载粒度的 64 倍数档）。**896 = 64 x 14 是 64 的倍数，却不在表里**，于是
896-expert 路由每次都付"两个 kernel + workspace 中转"的价钱。

实测这个价钱（B200，bf16，含 torch op dispatch）：

| M (tokens) | 动态 2-kernel 路径 |
|-----------:|------------------:|
|         64 |            30.8us |
|        128 |            30.8us |
|       1024 |            70.1us |
|       4096 |           235.8us |

每个 MoE 层一次、每步 92 层，光是路由就是每步 ~2.8ms（两个 rank 合计）的 GPU 时间。

## 改动：一行 case，约束先验算

加特化之前先算模板参数是否满足 kernel 的全部 `static_assert`（`TopkConstants`）：

- CUDA 路径 `WARP_SIZE_PARAM=32`，`BYTES_PER_LDG_MULTIPLE_64` 对 bf16/fp16 取 4 字节、
  fp32 取 8 字节，所以两种 dtype 下 `ELTS_PER_LDG = 2`；
- 整除性：`896 % (ELTS_PER_LDG * 32) = 896 % 64 = 0` OK；
- `VEC_PER_THREAD = 896/64 = 14`，`VPT = 14 * 2 = 28`，
  `THREADS_PER_ROW = 896/28 = 32`（2 的幂且 ≤ 32）OK；
- `ELTS_PER_WARP = 32 * 28 = 896 = ELTS_PER_ROW`，整除 OK；bf16 的
  `ELTS_PER_LDG` 为偶数 OK。

全部成立，于是改动就是三行：

```cpp
case 896:
    LAUNCH_TOPK(896, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
    break;
```

外加把 `tests/kernels/moe/test_fused_topk.py` 的 `num_experts` 参数表补上 896，
让 `test_fused_topk` / `test_fused_topk_bias` 覆盖这个形状（sigmoid/softmax × bf16/fp16
/fp32 × renormalize × bias）。

## 收益：算子 1.9-4.6x

同卡 A/B（B200，bf16，E=896，topk=16，带 bias，含 dispatch；ids 与 torch 参考逐行
集合相等）：

|    M | 旧（2-kernel） | 新（fused 896） | 加速比 |
|-----:|---------------:|----------------:|-------:|
|    4 |         31.0us |          14.7us | 2.12x |
|   16 |         30.8us |          14.9us | 2.07x |
|   64 |         30.8us |          15.2us | 2.02x |
|  128 |         30.8us |          16.4us | 1.88x |
|  512 |         45.5us |          16.6us | 2.74x |
| 1024 |         70.1us |          23.7us | 2.96x |
| 4096 |        235.8us |          51.0us | 4.62x |

小 M 段的 ~15us 是 torch op dispatch 的固定地板（新 kernel 单次执行的纯 GPU 部分只有
几 us），大 M 段动态路径的二次扫描成本被整体消掉，所以加速比随 M 继续放大到 4.6x。

## E2E：一个诚实的"零收益"

把新 `.so` 挂进生产形态（2 节点 x 8xB200，TP8+PP2，FP8 KV，DS 布局，
`FULL_DECODE_ONLY` cudagraphs，路由走 vLLM 侧 `TrtLlmMxfp4ExpertsModular` 叠加
本 PR 的 fast path），与挂旧 `.so`（慢路径）做 A/B：

| 负载 | 旧 topk | 新 topk |
|------|--------:|--------:|
| conc=64, 128x(128in/512out) | 27.73-27.82ms TPOT | 27.85ms TPOT |
| conc=128, 256x(128in/512out) | 39.60ms / 3027.6 tok/s | 39.60ms / 3028.1 tok/s |
| conc=8, 8x(128in/512out) | 14.3ms TPOT（噪声 ±3%） | 14.3ms TPOT |

**完全一样。** 连三层路由实现之间都是平的：flashinfer 内置的 10.5us 单 CTA 路由、
vLLM 慢路径 30.8us、本 PR 的 16us，E2E 打平。

解释：这个部署 decode 的每步时间被流水线结构锁死 —— PP2 逐级串行，各 rank 有 ~47%
的时间在 NCCL PP send/recv 上等待对方 stage；换算下来每个 stage 每步有数 ms 的空窗。
路由 kernel 坐在 MoE 层内 TP allreduce 之前的计算段上，这点 15us/层 x 46 层的节省
（~0.7ms/stage）被空窗整个吸收，不出现在 TPOT 里。优化 #1（KDA 融合）同类量级的节省
能显现，是因为它砍在一条恰好贴着调度点、不被吸收的串行子链上；
本次用三组对照实验（外加 bs=8/128 两个点）把"路由不可见"钉死了。

这个结论本身有价值：它告诉我们**这个部署形态下，任何 ≤1ms/步 的 MoE 层内小算子优化
都会被吃掉**，后续优化要么打在 PP/TP 通信结构上，要么打在真正暴露的串行长链上。

## 那这个 case 还值得加吗？值得

1. 走 vLLM 侧路由的 896-expert 部署是真实存在的：非 flashinfer-monolithic 的后端
   （纯 vLLM/EP、其他算子库、ROCm）、K3 家族的变体、任何自己 select
   `TrtLlmMxfp4ExpertsModular` 的配方 —— 这些场景路由就暴露在关键路径上，
   收益就是上面的 2-4.6x。
2. 即便在本部署里，它去掉的也是每步 92 次 launch + workspace 中转的纯 GPU 时间
   （功耗/频率压强），对 cudagraph 内 kernel 排布也更友好。
3. 是三行的零风险覆盖面补齐，与表里既有的 192/320/384/448/512/576 完全同构。

正确性：单元测试（torch.topk 参考逐位集合相等，所有 dtype/shape 组合）；E2E greedy
输出与基线一致；GSM8K 全量 1319 条在挂新快的完整服务栈上：
exact_match flexible-extract = **0.9704 ± 0.0047**、strict-match = **0.9712 ± 0.0046**，
与 #1 优化时同栈测得的 0.9689/0.9697 持平，无精度回归。

## 附：复现要点

- 路由调用链：`FusedTopKBiasRouter._compute_routing` → `fused_topk_bias`
  → `torch.ops._moe_C.topk_sigmoid`（sigmoid+bias）/ `topk_softmax`（softmax）。
- 只有构建 `_moe_C_stable_libtorch` 目标即可（该 kernel 不在 `_C_stable_libtorch` 里）。
- 微基准脚本即按 `torch.ops._moe_C.topk_sigmoid` 直调，对比 `torch.topk` 参考。
