# DSV4.1 skinny：原理、覆盖与收益解释

Skinny GEMM 为小 M 的矩阵乘法 `Y[M,N] = X[M,K] @ W[N,K].T` 调整访存、线程分工和归约。
M 是本次投影处理的 token 行数，跟请求并发和 SP 切分有关，并不是上下文长度。
单请求 decode 即使已有很长的上下文，也可以有 M=1。

M 很小时，权重跨 token 的复用有限，数据读取和固定执行开销可能比峰值矩阵算力更重要。
典型 skinny 让一个 block 负责少数输出列，线程沿 K 分摊点积，再归约；对少量输入行展开循环，
尽量复用已读的数据。M 增大后，Tensor Core GEMM 的复用优势增强，因此必须按形状选择赢家。

## MXFP8 与实现方法

当前基线主投影已经使用 MXFP8；不是为了 skinny 才引入量化。每 32 个 FP8 元素共享一个
E8M0 scale，省去一部分 BF16 权重存储和读取流量。CuTeDSL/Triton 则是实现 kernel 的工具，
与数据格式是两个维度。详见 [模型图](model-skinny-map.md)。

三个主投影 `fused_wqa_wkv`、`wo_a`、`wo_b` 的获胜候选为 Triton SIMT。部分 BF16
辅助投影和 `lm_head` 复用 vLLM 现有 CuTe skinny；compressor 复用 CuTe `ll_bf16`
dotprod，保留 FP32 输出。CuTe MXFP8 的探索版本未胜出，没有用于 serving。

当前 MXFP8 原型还有局限：原始 BF16 输入路径在多个输出 tile 内重复计算激活量化/scale，
并且扫描只覆盖 M=1/2/4/8/16。`wq_b` 没找到赢家，表示本次候选没有胜出，不能证明没有空间。
改用 CuTeDSL 本身也不保证更快，仍要验证数据布局、复用、寄存器占用和线程组织。

## 为什么当前仅约 4%

M1 下三个主投影的单层节省约为 1.75、3.51、1.82 微秒。按 40 层串行相加是 0.283 ms，
实际 C1 平均 TPOT 的三轮中位数从 7.454 ms 降到 7.177 ms，节省 0.277 ms。
这只说明量级相符，不能把孤立 kernel 时间之和当作关键路径归因。

Compressor/indexer 的部分工作与主投影在不同 stream 重叠；辅助分支缩短不一定提前整体完成。
`lm_head` 每步只执行一次。完整候选 C16 回退 0.75%，M≤4 对照提高 0.29%；每个配置仅有一次
服务启动，不能把约 1% 的配置间差异全归因于较大 M 的辅助投影。

最终 skinny 配置没有重新采集 nsys/NCU，因此尚未定量分离 kernel 实现不足、通信等待和
执行重叠的贡献。**4% 是这版实现的结果，不是 DSV4.1 的架构上限。**

## 与 K3 的比较

[K3 #53534](https://github.com/vllm-project/vllm/pull/53534) 修复 SM100 架构判断导致的
BF16 投影回退，并按形状接入 CuTe skinny / `dsv3_fused_a_gemm`。其 C1 结果为 +9.8%，
测试配置是 16×B200、TP8×PP2；本实验是 DSV4.1 Flash、8×B200、TP8，不能视为严格同条件对照。

K3 表里的 24 个 shape 跨多个 TP 配置，按调试标签粗略归并为 13 类；DSV4.1 的当前 TP8
库存为 12 类有效 dense 投影，其中 8 类找到候选。表的行数不是实际 GEMM 调用次数。
本地模型配置中，K3 有 93 层（69 KDA + 24 MLA）、hidden size 7168；DSV4.1 Flash 有
40 层、hidden size 5120。K3 latent-MoE 还带完整复制的 7168→3584、3584→7168 两个外围投影。

| 位置 | K3 #53534 | 本次 DSV4.1 |
| --- | --- | --- |
| 路由专家内部 grouped GEMM | 没替换 | 没替换 |
| latent-MoE 降维/升维 | 接入 skinny | 无这对投影 |
| shared expert 独立投影 | 接入 skinny | 已融入 MegaMoE，基线就开启 |
| Attention / lm_head | 按形状接入 | 按形状接入部分候选 |

未优化的 MoE 耗时可能稀释 dense skinny 的总收益，但还没有证据认定它是主要原因。
若要研究专家 skinny，需要先区分 MegaMoE 内的矩阵计算、填充与等待，按每专家实际 token
数分析；直接拆成每专家一个普通 skinny 调用可能损失现有融合的好处。

## 证据与限制

[完整报告](RESULTS.md)记录三组各三轮 `vllm bench serve` 结果、固定请求/长度检查、
全部 rank 的数值检查、已有优化与未选中候选。四个短 smoke prompt 有两个文本完全相同，
另外两个有差异；这不构成模型质量评估。GSM8K 未运行。

[分发表](gemm-dispatch-tables.md)和[模型图](model-skinny-map.md)给出对应代码位置。
