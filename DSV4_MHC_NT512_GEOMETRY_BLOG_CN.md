# DeepSeek V4 Flash 在 8×B200 上的 mHC Kernel 优化实录：四次尝试、两个失败与一个 1% 的真实胜利

本文记录一次完整的 kernel 粒度性能优化过程：目标是 vLLM 上
DeepSeek-V4-Flash-0731 的小 batch decode 延迟。最终落盘的改动只有二十几行，
但沿途做了四次机制上互不相同的尝试，其中两次微基准胜利被 E2E 否定，
一次 trace 分析意外发现上一笔提交里混进来的 PDL 回退 bug。把失败和修正
一起写出来，是因为对这个方向（"像 PR #53040 fuse shared experts 那样持续
挖 kernel 粒度收益"）来说，**知道哪些路走不通、为什么走不通，和那一个
百分点的收益同样有价值**。

## 一、目标选择：nsys 分解 decode step

在 BS=1、TP8+EP8（sequence parallel）的 decode 路径上，用 nsys
（`--cuda-graph-trace=node`，能看到 CUDA graph 内部节点）对一个完整
step 做时间线分解。一个 step 约 6.56 ms、主流 588 个 kernel、21 层
MoE；**主流 busy 只有 47.5%**，成分如下（图见
[DSV4_MHC_NT512_STEP_BREAKDOWN.png](DSV4_MHC_NT512_STEP_BREAKDOWN.png)，
数据在 [DSV4_MHC_NT512_STEP_BREAKDOWN.json](DSV4_MHC_NT512_STEP_BREAKDOWN.json)，
10 个 step 平均）：

| 类别 | 每步耗时 | 占 step |
| --- | ---: | ---: |
| MoE（MegaMoE 融合专家，42.9 µs × 21） | 897 µs | 13.7% |
| 投影 GEMM（fp8/fp4 1d1d × 104 + nvjet splitK） | 754 µs | 11.5% |
| **mHC（A `mhc_fused` 8.02µs×41 + B `mhc_pre_big_fuse` 4.77µs×41）** | **521 µs** | **7.9%** |
| 注意力（splitkv MLA + combine + indexer + norm/rope/quant） | 499 µs | 7.6% |
| TP 通信（reduce_scatter / all_gather × 21） | 346 µs | 5.3% |
| GEMM 输入量化（per_token_group quant ×83） | 301 µs | 4.6% |
| Router/TopK/prepare | 126 µs | 1.9% |
| 其它（head/elementwise/norm） | 33 µs | 0.5% |
| **空闲 / 跨 rank 同步气泡** | **3443 µs** | **52.5%** |

按 kernel 时间线跨度（含两侧启动间隔）度量，mHC 边界对约占 step 的 14%
——空闲中有相当一部分被边界对前后的依赖等待吸收。A→B 在 PDL 下几乎
背靠背（gap ≈ -0.22 µs），是名副其实的依赖链关键路径。B 里有一段
warp0 独占的 20 次 sinkhorn 串行迭代；消融实验（把 B 的 sinkhorn 删掉）显示
B 从 4.22 µs 降到 2.89 µs——也就是说 B 的时长完全由这条串行链决定。
这给了我们三个候选方向：压 A、压 B、或者把 sinkhorn 挪出关键链。

另外值得记录：52.5% 的空闲并非均匀的 launch 间隙，而是约 20 个重复出现的
~130 µs 大气泡（合计 ~2.7 ms），每个气泡结束于一个 8 卡完全对齐启动的
side-stream splitK GEMM——气泡期间本 rank 各流全空、其它 rank 仍在工作，
是典型的跨 rank 同步等待（BS=1 下各 rank 每层本地工作量仅 ~100-150 µs，
任何抖动都会在每层一个的同步点上累积放大）。这是 kernel 粒度之外、
量级更大的下一个结构性目标。

## 二、四次尝试

### 尝试 1：A kernel 权重预取（PDL staging）——微基准赢，E2E 回退 ~3% ❌

思路：A 每次启动都要冷读 1.5 MiB 的 fp32 fn 权重（每个边界各一份；两次
访问之间 MoE 层的海量专家权重读取会把 L2 完全洗刷，因此每次都是冷读），
而 PDL 允许本 kernel 在前驱未结束时提前启动。于是把权重切片
`T.copy` 到 shared memory 的代码挪到 `pdl_sync` **之前**，让冷读与前驱的
尾巴重叠。数学不变，逐位一致。

隔离图回放里 A 从 5.8 µs 降到约 3 µs，很好看。但 E2E b1 TPOT 从 6.50
回到 6.84（两个 server 实例高度复现）。trace 显示模型内 A 反而涨到
8.96 µs：真实管线里提前发起的 shared memory 流量与前驱的收尾互相干扰，
孤立 benchmark 里没有这个对手。**教训一：孤立 kernel benchmark 赢 ≠ 在
PDL 链上赢。**

### 尝试 2：A kernel 线程几何 nt512 —— 有效，但埋了个 bug ⚠️

思路：小 batch 下 A 是延迟受限的。原几何 256 线程、`h_per_split=512`，
每个线程要在 hidden 维上串行循环 2 次（`h_iters=2`）。改成
`n_thr=512` 后每线程恰好负责一个元素，`T.serial` 循环消失，
post-map / 平方和 / FMA 全部变成一拍平推；跨 warp 规约树不变，
逐位一致。E2E：b1 TPOT 6.56→6.503（-0.9%），c8 吞吐 968.9→993
（+2.4%），两次实例复现。**这是真实收益。**

但它和尝试 1 同属一笔中间提交，后来才发现那笔提交在插入新 kernel 时把
`mhc_fused_tilelang` 末尾的 `T.pdl_trigger()` 缩进弄丢了（挪到了模块级
死代码）——也就是说这次测量里 A 一直没有 PDL 出口触发，A→B 的提前启动
重叠被悄悄关掉了。见尝试 4。

### 尝试 3：sinkhorn sidecar —— 逐位正确，E2E 中性偏负 ❌

思路：B 的时长被 warp0 的 sinkhorn 串行迭代决定，而 sinkhorn 的输出
（post_mix/comb_mix）要等 MegaMoE 之后才被消费。把 B 拆成两个 kernel：
apply 半核放主流（保留 96 线程、warp1/2 布局，规约逐位一致），sinkhorn
半核放 side stream，用 event 做 fork/join（vLLM attention 里
`maybe_execute_in_parallel` 同款模式，CUDA graph 下合法）。

正确性完美：post_mix / comb_mix / layer_input 三分量 `torch.equal`
逐位一致。冷 harness 里每边界省 0.3-0.5 µs。但 E2E：b1 6.511、
c8 7.259，比尝试 2 略差。

nsys 一看就明白了：**side stream 上的 sinkhorn 半核要在 A 结束约 50 µs 后
才被调度**，紧贴着 join 点执行（graph 调度把分支节点放在汇合点附近），
非但没重叠，其 join 边还把后面的 all-gather 卡住了；而且拆开后两个半核
各自重付"启动 + 冷读 gemm_out"的固定成本，S 4.37 µs、B' 4.73 µs，
**两个都不比合并版 B（4.5 µs）快**。**教训二：在这个粒度上，上游把它们
fuse 起来本来就是对的；拆分 = 固定成本 ×2。教训三：大型 CUDA graph 中段的
side-branch fork 不会被热切调度，不要用孤立 harness 推断 graph 内调度行为。**

### 尝试 4：补回 A 的 `pdl_trigger` —— 白捡的修复 ✅

写尝试 3 的 trace 分析时发现：A.end→B.start 的 gap 从基线的 **-0.22 µs**
（重叠）变成了 **+0.22~0.26 µs**（串行）。追到代码才发现尝试 1/2 的中间
提交把 `mhc_fused_tilelang` 末尾的 `T.pdl_trigger()` 意外降级成了模块级
语句——kernel 里根本没有触发指令了。没有 trigger 时后驱的 `pdl_sync`
只能等 kernel 完整结束，A→B 的 PDL 提前启动就这么没了。

删掉尝试 1 遗留的未使用 kernel、把 trigger 恢复到正确缩进后重测：

## 三、最终数据

环境：8×B200（SM100）、driver 590.48.01、CUDA 13.0、torch 2.13.0+cu130、
baseline 为含 SM100 fused shared expert / FlashMLA single-batch 优化的
`d66f5ee254`。bench 用不同随机种子排除 prefix cache 命中。

| 场景 | Baseline | 本 PR | 变化 |
| --- | ---: | ---: | ---: |
| b1 decode median TPOT（1024/256，两次实例取均） | 6.5623 ms | **6.4912 ms** | **-1.08%** |
| b1 output throughput | 143.35 tok/s | **144.70 tok/s** | +0.94% |
| c8 output throughput（256 prompts） | 968.9 tok/s | **991.3 tok/s** | **+2.31%** |
| c8 median TPOT | 7.2939 ms | **7.2280 ms** | -0.90% |

逐 run 中位数：baseline 6.5612 / 6.5633；本 PR 6.4906 / 6.4917——每个
优化 run 都优于每个 baseline run，超出同实例 ±0.01 ms 的噪声带。

隔离 harness（每段重放 86 个边界、各边界用互不相干的权重模拟 L2 冷读、
CUDA graph 回放）：每边界 8.43 → 7.56 µs（-10.3%）。E2E 收益小于隔离值，
因为边界在真实管线里被 PDL 重叠盖住了一部分。

精度（GSM8K 全量 1319 题、5-shot、temperature 0、max 256 tokens）：
baseline 93.63%，本 PR 94.31%——改动对同输入逐位一致，精度波动属于
并发服务下生成路径的正常噪声。

## 四、写给后来人的 checklist

1. 先分解全 step 再选目标；13.7% 的链条占比决定了天花板，eval 要超噪声
   就得对自己的 0.5% 心里有数。
2. PDL 链上的 kernel，孤立 benchmark 的结论不能外推（尝试 1、3 各撞一次）。
3. 小 batch 下"融合 vs 拆分"几乎总是融合赢：拆分把启动/冷读固定成本乘以二。
4. side-stream fork 在大 CUDA graph 中段不会按直觉被调度；graph 内重叠要
   trace 验证，不能假设。
5. 每次实验把 PDL trigger/sync 的存在性当成一等公民检查——它被意外弄丢
   时功能完全正常、单测全绿，只是悄悄变慢。
6. 保留负结果的完整数字：它们就是下一笔优化的 baseline 对照。
