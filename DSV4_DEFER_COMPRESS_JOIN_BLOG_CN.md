# DeepSeek V4 Flash 优化实录之三：把 compressor 的 join 挪到 attention 之后，零 kernel 改动换 +0.6% 吞吐

这一篇的改动**完全不含 kernel 代码**：只动了两条 CUDA event 的等待
位置。DSV4 每层 decode 都会把 compressor（压缩 KV 写入）放到辅流上
跑，但主流在发射 attention 之前就会等它完成——而 compressor 写的
压缩 KV，本步内根本没人读。

## 一、症状：每 4 步一个"重步"

nsys 看 BS=1 decode 的 CUDA graph replay：轻步 5.59 ms，每第 4 步
5.85 ms，重步多 270 µs wall。重步多出来的是 21 个
`SparseAttnCompressNormRopeStore`（C4A 层的 4-token 压缩组闭合
kernel，~13-15 µs/个，每 C4A 层一个），每层一个、间距均匀——
它们各自压在该层 attention 发射之前：

```
计算流:   q_proj+kv_insert ─► wait(compress_done) ─► indexer_topk ─► MLA ─► o_proj
compr辅流: kv_score GEMM ─► save_partial ─► C4 close(13µs) ┘
          |<----------- compressor 链 ≈ 37µs ----------->|
          |<---- 主流到 attention ≈ 35µs ---->|
```

compressor 搭的是通用 `execute_in_parallel` 便车，这个 helper 在
default 分支之后立刻 join 所有辅流。于是闭合 kernel 的计算量全部
暴露到关键路径上。

## 二、为什么这个 join 可以挪

关键论证：**第 t 步闭合写入的压缩 KV 行，第 t+1 步起才可能被读。**

- attention/indexer 的读取范围由每步开始时的 seq_len 定界
  （FlashMLA 的 `flash_mla_with_kvcache` 只按 `topk_indices` 读已提交
  的压缩行；compressor 本步写的行不在索引范围内）；
- 没有任何其他层访问本层 compressor 的 state/KV cache；
- 跨步顺序天然由 graph 顺序 replay 保证（压缩 kernel 只要在本步
  graph 结束前 join 回主流即可）。

所以对齐点唯一的约束是**CUDA 图捕获合法性**（fork 必须在 capture
结束前回流），join 放在该层 attention 模块末尾（`_o_proj` 之后）
即可。主流 slack 从 ~35µs 变成 ~64µs，37µs 的 compressor 链完全
藏进去；轻步不受影响（轻步上闭合 kernel 8 个 state 提前退出，只有
~1.2µs）。

一个工程细节：vLLM 这里同时用两种捕获——decode 走 FULL capture
（整步一张图），prefill/mixed 走 breakable piecewise capture。
piecewise 下每个 segment 必须在段尾 join 干净，悬空的跨段 fork 会
直接 `cudaErrorStreamCaptureUnjoined`（第一版补丁就在这里炸的）。
所以新路径加了 `not BreakableCUDAGraphCapture.is_active()` 守卫：
piecewise 段内回退到原来的 pre-attention join（行为逐字节不变），
只有 FULL capture 的 decode 路径吃到收益。

## 三、数字

8×B200、TP8+EP8、mega_moe 后端、1024 进/512 出、64 prompts、并发 1
（慢平台期配对，双方实例都落在 TPOT≈6.56ms 档）：

| 变体 | TPOT 中位 | output tok/s |
| --- | ---: | ---: |
| 基线（attention 前 join） | 6.56 ms ×3 | 147.4–147.7 |
| 本 PR（模块尾 join） | 6.52 ms ×2 | 148.5 |
| 变化 | **−0.61%** | **+0.58%** |

同实例重复性 <0.1%，效应 ~6 倍于噪声，跨实例复现。中间还试过
"紧随 attention 之后 join" 的保守版本：只收回 55%（+0.46%），因为
从 fork 到 attention 起点的 slack（35µs）比 compressor 链（37µs）
短 7µs——这直接促成了把 join 推到模块尾的最终版。

GSM8K 对照（1319 题、5-shot、t=0）：flexible 79.61% vs 79.23%
（差 0.38pp，远小于 stderr 1.11pp）；strict 61.26% vs 59.21%。
逐样本 verdict 翻转 127 个、方向 66/61 对称——是批次组成导致的
生成非确定性，不是系统性劣化（若是漏 join，错误会是一边倒的）。
这个改动只是移动同步点，理论上连比特都不该动；跑 eval 是为了抓
"忘记 join"这类调度错误。

## 四、经验

1. **同步位置也是性能代码。** aux 流 fork 容易，join 位置才决定
   暴露多少。
2. **"本步写入、下步可读"是一座金矿。** KV/压缩缓存的步级可见性
   语义让 attention/pre-Norm 之后的所有位置都成为合法 join 点，
   只要证明读取范围被 step-start 快照定界。
3. **vLLM 有了两种图捕获之后，流手术必须双向合法。** FULL 里合法
   的悬空-后接法在 piecewise 段尾就是硬错误；守卫条件
   `BreakableCUDAGraphCapture.is_active()` 是现成的。
