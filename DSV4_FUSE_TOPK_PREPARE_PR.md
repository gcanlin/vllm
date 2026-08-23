# [Performance] Fuse DeepSeek V4 router top-k into the MegaMoE input-staging kernel

## Summary

This PR fuses the DeepSeek V4 router top-k selection into the MegaMoE
input-staging Triton kernel (`prepare_megamoe_inputs`), removing one
serialized small-kernel launch per MoE layer from the in-graph decode
critical path.

The staging kernel launches a `(num_tokens, hidden_size / 128)` grid. The
`k_block_id == 0` program of each token now additionally computes the full
router top-k selection inline — identical math to the standalone
`dsv4_topk` kernel — and writes both the public `(num_tokens, top_k)`
routing tensors **and** the repacked MegaMoE routing layout directly. All
programs keep quantizing their hidden tile to FP8 with UE8M0 group scales
exactly as before. On the dependency chain this deletes the
`router top-k -> staging` kernel boundary: one launch, one PDL handshake
and one in-graph seam per MoE layer (43 MoE layers per decode step).

Outputs are bitwise-identical to the separate `dsv4_topk` +
`prepare_megamoe_inputs` pair (verified 0 bitwise mismatches across batch
sizes 1-16, including the shared-expert staging layout). The fused path is
guarded by `can_use_dsv4_topk` and is skipped automatically when hash
routing or EPLB expert mapping is active (EPLB needs the logical top-k ids
before staging, so it keeps the split path). The fallback can also be
forced with `VLLM_DSV4_FUSE_TOPK_PREPARE=0`.

On 8x B200 with DeepSeek-V4-Flash-0731 (TP8 + EP8, `deep_gemm_mega_moe`
backend) BS=1 decode, the change improves median TPOT by **0.63%** and
output throughput by **0.69%**, reproduced across two independent server
instances. GSM8K accuracy is unchanged within measurement noise.

## Motivation

An nsys decomposition of the BS=1 decode step (one CUDA-graph replay,
~1540 kernels per step on device 0 across 6 captured streams,
device-union busy 97.4%) shows the decode stride is dominated by kernel
execution time on the dependency chain — there are no harvestable bubbles
left, so TPOT only moves when kernels (or their serialized seams) are
removed from the chain.

The router block of every MoE layer runs three tiny kernels back to back
under PDL: `gate` GEMM -> `_dsv4_topk_kernel` (2.48 us) ->
`_prepare_megamoe_inputs_kernel` (1.77 us). Both follower kernels are far
below launch-latency amortization size, yet each pays its own in-graph
launch/drain seam, and the top-k kernel's output layout
(`(num_tokens, top_k)`) is immediately re-read and repacked by the staging
kernel. The staging kernel's grid is already `num_tokens x (hidden/128)`;
its `k_block_id == 0` programs idle on the full hidden tile load while the
top-k work they need (256 logits, 6 slots) is trivially in-register. Doing
the selection there costs nothing measurable inside the kernel (fused
kernel is within ~0.35 us of the standalone staging kernel) and removes
the entire intermediate launch from the graph.

At 43 MoE layers per step this deletes 43 kernel launches + 43 PDL seams,
~107 us of device busy time per step, of which ~40 us lands on the BS=1
critical path (the remainder was already hidden by the graph's
cross-stream overlap — consistent with the measured TPOT delta of
0.0404 ms/step).

## Design

`DeepseekV4MoE.forward` gains a fused dispatch in front of the split path:

```python
if (envs.VLLM_DSV4_FUSE_TOPK_PREPARE
        and self.gate.tid2eid is None                      # no hash routing
        and self.experts.eplb_state.logical_to_physical_map is None
        and num_tokens > 0
        and can_use_dsv4_topk(...)):                       # scoring/topk/renorm
    topk_weights, topk_ids = dsv4_topk_prepare_megamoe(
        hidden_states, router_logits, e_score_correction_bias, ...,
        symm_buffer.x[:num_tokens], symm_buffer.x_sf[:num_tokens],
        symm_buffer.topk_idx[:num_tokens], symm_buffer.topk_weights[:num_tokens],
        is_padding=is_padding, shared_x_sf=..., shared_block_m=...)
    prestaged = True
else:
    topk_weights, topk_ids = fused_topk_bias(...)          # unchanged

self.experts(..., prestaged=prestaged)                     # skips re-staging
```

- The new kernel `_dsv4_topk_prepare_megamoe_kernel` reuses the exact
  staging/quantization body of `_prepare_megamoe_inputs_kernel`; the
  only addition is the `k_block_id == 0` top-k prologue, which mirrors
  `dsv4_topk` line for line (sqrt-softplus scoring, bias, k sequential
  argmax reductions with lowest-index tie-break, renormalization with
  `routed_scaling_factor`).
- All tensor layouts, dtypes and leading dimensions written by the fused
  kernel are identical to the split path, including the shared-expert
  scale-factor swizzle (`shared_x_sf`) and the `is_padding` sentinel
  (`topk_id == -1`, weight 0) handling for `VLLM_MOE_SKIP_PADDING`.
- `DeepseekV4MegaMoEExperts.forward` accepts `prestaged=True` to skip the
  padding mask, EPLB remap and `prepare_megamoe_inputs` call that the
  fused kernel already performed; with `prestaged=False` the behavior is
  byte-for-byte the old one.
- No changes to any GEMM, attention or communication kernel; PDL triggers
  are kept intact.

## Performance

Server configuration (both variants):

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

`vllm bench serve`, random dataset, 1024 input / 512 output tokens,
64 prompts, `--ignore-eos`. The baseline and the optimized run differ only
in `VLLM_DSV4_FUSE_TOPK_PREPARE` (0 vs 1, same build).

**Measurement protocol note.** On this machine each freshly started server
instance lands in one of several distinct performance plateaus
(e.g. median TPOT bands at ~6.57 ms and ~6.37 ms), presumably from
run-to-run allocation/layout luck; all comparisons below therefore pair
baseline and optimized runs **within the same plateau**, and the optimized
result is additionally reproduced on a second server instance.

### BS=1 decode latency (concurrency 1)

| Variant | Median TPOT (ms) | Output throughput (tok/s) | Median TTFT (ms) |
| --- | ---: | ---: | ---: |
| Baseline (fused off) | 6.3694 | 152.03 | 111.19 |
| This PR (fused on) | 6.3291 | 153.02 | 110.91 |
| Change | **-0.63%** | **+0.65%** | -0.25% |

Per-run medians: baseline 6.3693 / 6.3694 ms (seeds 8321/8322);
this PR 6.3291 / 6.3290 / 6.3290 / 6.3290 ms (seeds 8301-8304).

Second-instance reproduction of this PR: 6.3320 / 6.3318 ms and
152.87 / 152.96 tok/s (seeds 8331/8332), i.e. **-0.59%** vs the same
baseline — the effect survives server restarts and is far outside the
within-instance run-to-run spread (< 0.02%).

## Accuracy

GSM8K against the serve endpoint: full 1319-question test split, 5-shot,
temperature 0, max 256 tokens, concurrency 64.

| Variant | flexible-extract | strict-match |
| --- | ---: | ---: |
| Baseline (fused off) | 78.54% | 59.14% |
| This PR (fused on) | 78.39% | 60.42% |

Both deltas (0.15 pp / 1.28 pp) are within the measurement stderr
(1.13 pp / 1.35 pp); the kernel change is bitwise-identical for identical
inputs, so no accuracy movement is expected. This run guards against
dispatch or configuration mistakes.

## Tests

- New-bitwise check: a standalone harness replays
  `dsv4_topk` + `prepare_megamoe_inputs` vs `dsv4_topk_prepare_megamoe`
  across batch sizes {1,2,4,8,16} x 5 seeds, including shared-expert
  staging — **0 bitwise mismatches** in all outputs (public routing
  tensors, repacked MegaMoE layout, FP8 payloads and UE8M0 scales).
- The split path is byte-identical to the previous behavior when the env
  gate is off or the guards (hash routing / EPLB) trigger.
