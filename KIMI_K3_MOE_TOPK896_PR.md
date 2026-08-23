# [Perf][MoE] Add 896-expert specialization to the fused topk gating kernel

## Summary

`topkGatingKernelLauncher` (`csrc/libtorch_stable/moe/topk_softmax_kernels.cu`)
compiles fast single-pass specializations of the fused topk kernel for
`num_experts` in {1..128 powers of 2} and {192, 320, 384, 448, 512, 576}. Kimi K3
uses 896 routed experts with sigmoid scoring and an e-score correction bias
(noaux_tc degenerated to a single group), so every router call fell back to the
generic two-kernel path (per-segment sigmoid/topk followed by a full-width
merge pass) with a workspace round-trip.

896 = 64 * 14 satisfies all compile-time constraints of the templated kernel
with the existing MULTIPLE_64 load granularity (`ELTS_PER_LDG=2`, `VPT=28`,
`THREADS_PER_ROW=32` for both bf16 and fp32; `THREADS_PER_ROW` stays a power of
two within warp size), so this PR adds the missing `case 896`. One launch
replaces two and the workspace round-trip disappears; the op goes from ~31us to
~15-16us at decode-relevant batch sizes on B200 (~1.9-2.1x, rising to 4.6x at
M=4096 where the fallback's second pass dominates).

## Motivation

- K3-shaped models (896 routed experts, topk=16) issue one router call per MoE
  layer per decode step (92 MoE layers total, 46 per pipeline rank in our
  TP8+PP2 deployment). On the two-kernel path that is ~31us x 46 per rank per
  step of router GPU time, with the extra workspace traffic and double launch.
- Any deployment that routes through the vLLM router
  (`FusedTopKBiasRouter` / `fused_topk`) with 896 experts benefits directly —
  e.g. non-flashinfer-monolithic configurations, EP/DP-attention recipes, ROCm,
  or future K3 variants. The 896 slot is the same class of coverage as the
  existing 192/320/384/448/512/576 rows.

## Changes

- `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`: add `case 896:
  LAUNCH_TOPK(896, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64)` (CUDA-only section
  of the switch; ROCm keeps falling through to the generic path, unchanged).
- `tests/kernels/moe/test_fused_topk.py`: add 896 to `num_experts` in
  `test_fused_topk` and `test_fused_topk_bias` so the shape is covered for
  sigmoid/softmax x {fp32, fp16, bf16} x renormalize x bias.

No behavior change for any other `num_experts`.

## Correctness

- Extended unit tests pass on B200 (`tests/kernels/moe/test_fused_topk.py`),
  including all new (896, sigmoid, bias) cells; selected expert ids match the
  `torch.topk` reference as sets for every seeded case.
- Op-level check on B200 (E=896, topk=16, bias, M in {4..4096}, bf16 and fp32):
  per-row expert-id sets identical to the reference; weights are the sigmoid of
  the un-biased logits at the selected ids (corrected-score selection semantics
  preserved).
- E2E accuracy, full serving stack (2 nodes x 8xB200, TP8+PP2, FP8 KV cache,
  DS conv-state layout, `FULL_DECODE_ONLY` cudagraphs, routing through
  `TrtLlmMxfp4ExpertsModular` with this fast path active): GSM8K full test
  split (1319 samples, lm_eval 0.4.12 `local-chat-completions`, 5-shot default
  chat template, temperature 0, max_tokens 1500):
  exact_match = **0.9704 ± 0.0047** (flexible-extract) / **0.9712 ± 0.0046**
  (strict-match) - matches the 0.9689 / 0.9697 measured on the same stack
  before this PR, i.e. no accuracy regression.

## Performance

### Op-level (B200, bf16, E=896, topk=16, with bias, incl. torch dispatch)

|    M | two-kernel fallback | fused 896 (this PR) | speedup |
|-----:|--------------------:|--------------------:|--------:|
|    4 |             31.0us |              14.7us |   2.12x |
|   16 |             30.8us |              14.9us |   2.07x |
|   64 |             30.8us |              15.2us |   2.02x |
|  128 |             30.8us |              16.4us |   1.88x |
|  512 |             45.5us |              16.6us |   2.74x |
| 1024 |             70.1us |              23.7us |   2.96x |
| 4096 |            235.8us |              51.0us |   4.62x |

(The ~15us floor at small M is torch op dispatch; the kernel body is ~3-4us.)

### E2E on the 2-node K3 decode stack (honest measurement)

Controlled A/B where the only difference is the `_moe_C` extension
(with/without `case 896`), same serving config as above,
`vllm bench serve --dataset-name random --ignore-eos` with random 128 in /
512 out:

| workload               | fallback router      | this PR              | delta |
|------------------------|---------------------:|---------------------:|------:|
| conc 64,  128 prompts  | 27.73-27.82 ms TPOT  | 27.85 ms TPOT        |  ~0   |
| conc 128, 256 prompts  | 39.60 ms / 3027.6 t/s| 39.60 ms / 3028.1 t/s|  ~0   |

In this TP8+PP2 deployment decode is pipeline-slack bound (~47% of per-rank
step time is spent waiting inside NCCL point-to-point for the other pipeline
stage), and the router kernels sit inside the MoE-layer compute segment ahead
of the last TP allreduce, so the ~0.7ms/stage of saved GPU time is absorbed by
the slack and does not surface in TPOT. We verified this from both ends:
flashinfer's internal single-CTA `routingIndicesBlockKernel` (~10.5us), the old
two-kernel vLLM path (~31us) and this fast path (~16us) produce identical E2E
latency at every concurrency tested (8/64/128). Wherever router latency is
exposed instead — the op-level table above is the expected gain, and this PR
also removes one kernel launch and the workspace transaction per call.

## Test plan

```
pytest tests/kernels/moe/test_fused_topk.py -v
```

Op-level reproducer and the full serving recipe are included in the repo-root
notes `KIMI_K3_MOE_TOPK896_BLOG_ZH.md` (Chinese write-up with the motivation
and trace analysis).
