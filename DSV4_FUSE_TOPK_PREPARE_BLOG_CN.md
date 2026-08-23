# DeepSeek V4 Flash 优化实录之二：把 router top-k 融进 MegaMoE 的装载 kernel，TPOT −0.63%

这是 DSV4 Flash kernel 粒度优化系列的第二篇。上一篇用 mHC kernel 的
线程几何拿到了 1.08% 的 TPOT 收益；这一篇的目标更小：把每个 MoE 层
router 后边那个独立的 top-k kernel **整个消掉**——不是做快，是让它
不存在。最终改动是一个 280 行的 Triton kernel（其中 95% 是从原
staging kernel 原样复用的），BS=1 decode TPOT −0.63%，output
throughput +0.65%，跨两次服务重启复现，GSM8K 精度在噪声内持平。

## 一、为什么盯上这个 kernel

上次的 nsys 分解已经给出结论：decode step 是一整张 CUDA graph 的
replay（dev0 每步 ~1540 个 kernel、6 条流、设备级 union busy
97.4%），**没有可收割的大气泡**，想改善 TPOT 只能让依赖链上的
kernel 变少或变快。

在这个判据下扫一遍每层的 kernel 序列，router 这一段非常扎眼：

```
gate GEMM  ->  _dsv4_topk_kernel (2.48 µs)  ->  _prepare_megamoe_inputs_kernel (1.77 µs)
```

后两个 kernel 都是两三微秒的小 kernel，各自在图里占一次
launch + PDL 握手 + drain 的接缝；而且 top-k 的输出
`(num_tokens, top_k)` 刚写出来，立刻被 staging kernel 读回去重新
打包成 MegaMoE 的布局。两个 kernel、一次中间张量往返，干的其实是一件事。

再看 staging kernel 的 grid：`(num_tokens, hidden/128)`。每个 token
有 hidden/128 个 program，其中 `k_block_id == 0` 的那个 program 在等
hidden 数据 load 的时候完全是闲的；而 top-k 需要的输入只有 256 个
logits + 6 个槽位，寄存器里随便放。**把 top-k 塞进这个 program
里顺手做掉**，kernel 内部的代价约等于零（实测融合后 kernel 时长只比
原 staging kernel 多 ~0.35 µs），而图里直接少了一个 kernel launch
和它前后的两条接缝——43 个 MoE 层，每层一次。

## 二、实现：复用，而不是重写

新 kernel `_dsv4_topk_prepare_megamoe_kernel` 的身体就是把
`_prepare_megamoe_inputs_kernel` 的 staging/量化部分**原样保留**
（FP8 + UE8M0 group scale 的打包、shared expert 的 scale-factor
swizzle、`is_padding` 哨兵处理逐行不动），只在前面加了一段
`k_block_id == 0` 的序曲，序曲逐行复刻 `dsv4_topk` 的语义：

- sqrt-softplus 打分（`sqrt(where(logits>20, logits, log1p(exp(logits))))`）
- 加 correction bias 后做 6 轮串行 argmax（同值取最小 expert id）
- 按 routed scaling factor 归一化
- 同时写出公共路由张量（`(num_tokens, top_k)`）和 MegaMoE 的
  repack 布局——中间张量彻底消失。

调度侧在 `DeepseekV4MoE.forward` 里加一条分支：满足
`can_use_dsv4_topk`、无 hash 路由、无 EPLB 时走融合 kernel，并把
`prestaged=True` 传给 experts，让 `MegaMoEExperts.forward` 跳过已经
被融合 kernel 做掉的 padding/EPLB/staging 三件套；否则走原来的
`fused_topk_bias` 路径，行为逐字节不变。env 开关
`VLLM_DSV4_FUSE_TOPK_PREPARE`（默认开）可强制回退。

正确性验证：独立 harness 对照 `dsv4_topk + prepare_megamoe_inputs`
与融合 kernel，batch {1,2,4,8,16} × 5 seed，包括 shared-expert
staging 布局，**全部输出 0 bitwise mismatch**——连浮点都没有动过，
因为每个 program 执行的指令序列和原来完全一一对应。

## 三、性能：数字很小，但它不是噪声

配置：8×B200、TP8+EP8、`deep_gemm_mega_moe` 后端、KV fp8、indexer
mxfp4，1024 进 / 512 出、64 prompts、并发 1。

| 变体 | TPOT 中位 (ms) | Output tok/s |
| --- | ---: | ---: |
| 融合关 | 6.3694 (6.3693/6.3694) | 152.03 |
| 融合开 | 6.3291 (6.3290×4) | 153.02 |
| 变化 | **−0.63%** | **+0.65%** |
| 融合开，第二次重启 | 6.3319 (6.3320/6.3318) | 152.92 |

这里必须交代一个测量学问题，也是这次工作里真正花时间的部分：
**这台机器上每个新启动的 server 实例会随机落进几个明显的性能
平台期之一**（比如 TPOT ~6.57 ms 的"慢档"和 ~6.37 ms 的"快档"，档内
方差 < 0.02%，档间差 ~3%）。来源详见 PHASE_TRANSITION.md 的取证
（同实例内还会发生一次性的 slow→fast 相变，kernel 时长不变、收益
弥散在图内重叠里，CPU/GPU 降频和核竞争均已排除，触发器疑似对
enqueue 时序敏感且在 nsys 观测下不复现）。因此所有 A/B 对比必须
**按平台期配对**——上表就是快档内配对，且融合开的结果在两个独立
实例上各自复现。跨实例复现 + 档内方差比效应小一个数量级，这是
"不是噪声"的证据链。

机制账目也对得上：融合删掉的是每层 2.48 µs 的 top-k kernel + 两条
图内接缝，43 层合计 ~107 µs 设备 busy 时间/步；实测 stride 缩短
40 µs/步，差额就是原本被图内跨流重叠盖住的部分——与上一篇 mHC
"独立基准赢 10% 而 E2E 只有 1%"是同一个道理，只是这次连独立基准
都懒得做了，因为收益本来就不是 kernel 内部吞吐，而是接缝。

GSM8K（1319 题、5-shot、t=0、并发 64）对照：flexible 78.54% vs
78.39%，strict 59.14% vs 60.42%，两个 delta 都小于一个 stderr。
位精确的改动本来就不该动精度，这一步是防 dispatch 接错。

## 四、经验总结

1. **在 graph 视角下，"删除一个 kernel"是一种独立的优化类别**。
   它不需要任何 kernel 内部技巧，赚得是 launch/drain/PDL 接缝和
   中间张量往返；前提是找到"上游 grid 里本来就有闲着的 program"
   这种结构性机会。
2. **融合的安全做法是身体复用**：新 kernel 95% 的代码逐行来自被
   融合的双方，语义杠杆只加在一个明确标注的序曲里，位精确验证才能
   一句话说清楚。
3. **先解决测量学，再谈优化**。没有平台期配对协议，0.6% 的效应会
   被 3% 的实例间方差彻底掩埋（我们已经亲眼见过 c8 数据被淹）。
   判档方法：看 server log 里每 10 s 一行的 `Avg generation
   throughput`，慢档 ~148–152、快档 ~153+。

下一步候选：图外段（每步 87 次发射、~220 µs busy 且带 80–105 µs
idle）的 indexer/topk 管线融合；mega_moe 主 kernel（~45 µs/层 ×
43，全图最大单 kernel）的逐相插桩分解。
