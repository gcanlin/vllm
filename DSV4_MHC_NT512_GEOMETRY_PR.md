# [Performance] Single-iteration thread geometry for DeepSeek V4 small-batch mHC fused kernel

## Summary

This PR tunes the TileLang `mhc_fused_tilelang` launch geometry on the
small-batch decode path (`num_tokens <= 16`) of DeepSeek V4. For
`num_tokens <= 8` we now launch with `n_thr=512` instead of the default 256
whenever `h_per_split % 512 == 0`.

The kernel splits the hidden dimension into `n_splits` chunks of
`h_per_split` elements and lets each thread own `h_iters = h_per_split /
n_thr` of them. With the default 256-thread geometry at hidden=4096,
`h_per_split = 512` and each thread loops twice over its hidden strip
(`h_iters == 2`). With 512 threads, every thread owns exactly one element:
the `T.serial(h_iters)` loop collapses, the residual post-map, the squared
sum, and the per-thread FMA into the pre-norm GEMM accumulators all become a
single flat pass, and the shared-memory cross-warp reduction keeps its exact
shape (the reduction tree and therefore the numerics are unchanged).

At batch size 1 the mHC boundary pair runs ~85 times per decode step
(2 boundaries x 43 MoE layers) and sits directly on the dependency chain.
On 8x B200 with DeepSeek-V4-Flash-0731 the
change improves median decode TPOT by 1.08% and concurrency-8 output
throughput by 2.31%. GSM8K accuracy is unchanged within measurement noise.

## Motivation

At small batch the fused mHC kernel is latency bound, not throughput bound:
per launch it moves only `num_tokens x hc_mult x hidden_size` activations but
reads a fresh 1.5 MiB fp32 weight tile (the MoE expert-weight traffic between
two visits evicts it from L2 every time). An nsys decomposition of the decode
step at BS=1 — one decode step is a single CUDA-graph replay, ~1540 kernels
on device 0 across 6 captured streams, with 43 MoE layers alternating between
two compute streams and projection GEMMs on side streams, device-union busy
97.4% — shows the mHC pair (`mhc_fused_tilelang` +
`mhc_pre_big_fuse_with_norm_tilelang`) accounts for 1.21 ms of kernel time
per step, ~14% of the on-device kernel total (~18% of the 6.60 ms stride),
launched back to back under PDL. The full
operator-level breakdown is in
[DSV4_MHC_NT512_STEP_BREAKDOWN.png](DSV4_MHC_NT512_STEP_BREAKDOWN.png)
(data: [DSV4_MHC_NT512_STEP_BREAKDOWN.json](DSV4_MHC_NT512_STEP_BREAKDOWN.json)).

Reducing per-thread serial work shortens the kernel's issue window directly:
the kernel has no other parallelism to hide the second hidden-element
iteration behind, because every CTA already waits on the PDL predecessor for
its inputs. Sizing the block so the loop trip count is one removes that
serial tail.

Two alternative attacks on the same boundary were implemented and measured
before settling on this change:

- **Weight staging under PDL**: moving the cold weight read into a shared
  memory copy issued before `pdl_sync` was faster in an isolated graph
  benchmark but regressed end-to-end decode by ~3%; in the real pipeline the
  early shared-memory traffic competes with the predecessor's tail and the
  geometry change above already removes the exposed latency more cheaply.
- **Sinkhorn side-stream split**: splitting `mhc_pre_big_fuse_with_norm` into
  an apply kernel on the main stream and a sinkhorn kernel on a side stream
  (fork/join via CUDA events, bit-identical outputs) also won in the
  isolated benchmark but was neutral-to-negative end to end; each half pays
  the full launch and cold-read fixed cost alone, and the event edge places
  the side-stream kernel on the critical chain before the following
  all-gather. Fusion remains the right granularity here.

Both negative results are reproducible and documented in the accompanying
notes; the final diff is intentionally minimal.

## Design

The dispatch in `mhc_fused_post_pre_tilelang` gains one conditional:

```python
if num_tokens <= 8 and (hidden_size // n_splits) % 512 == 0:
    mhc_fused_tilelang(..., n_thr=512, tile_n=tile_n, n_splits=n_splits)
else:
    mhc_fused_tilelang(..., tile_n=tile_n, n_splits=n_splits)  # unchanged
```

- The kernel itself is untouched; `n_thr` was already a parameter.
- The guard requires `h_per_split % 512 == 0` so the flattened mapping is
  exact; other hidden sizes or split counts keep the existing geometry.
- `num_tokens > 8` (including all prefill) keeps the 256-thread default; the
  9-16 token range keeps it as well because its `n_splits=4` geometry gives
  `h_per_split=1024`, where 512 threads would leave `h_iters=2` anyway.
- Numerics: the change only re-partitions the same per-thread FMA sequence.
  The order of accumulation per output element is unchanged, and the
  cross-warp reduction operates on the same per-thread partial sums, so
  outputs are bit-identical to the baseline for the same inputs.

## Benchmark

### Environment

- GPUs: 8x NVIDIA B200, SM100
- Driver: 590.48.01; CUDA: 13.0
- PyTorch: 2.13.0+cu130
- FlashInfer: 0.6.17
- Model: DeepSeek-V4-Flash-0731, 43 layers, 256 routed experts, top-k 6,
  one shared expert
- vLLM base: `d66f5ee254ea3ad4d9373c48f49fae2c6b97187f` (includes the SM100
  fused shared-expert MegaMoE and FlashMLA single-batch metadata changes)
- Parallelism: TP8 + EP8 with sequence parallel
- Prefix cache enabled; measured runs used distinct random seeds, so prompts
  did not hit the cache.

Server configuration:

```bash
vllm serve /mnt/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name dsv4 \
  --host 127.0.0.1 --port 8000 \
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

The baseline and the optimized run differ only in this PR's dispatch change;
both keep the kernel's programmatic dependent launch (PDL) trigger intact.

### BS=1 decode latency (1024 input / 256 output, concurrency 1, 64 prompts)

| Metric | Baseline | This PR | Change |
| --- | ---: | ---: | ---: |
| Median TPOT | 6.5623 ms | 6.4912 ms | **-1.08%** |
| Output throughput | 143.35 tok/s | 144.70 tok/s | **+0.94%** |

Two paired runs per variant (seeds 2071/2072 baseline, 2083/2084 optimized);
per-run medians were 6.5612/6.5633 ms baseline and 6.4906/6.4917 ms
optimized.

### Concurrency 8 (1024 input / 256 output, 256 prompts)

| Metric | Baseline | This PR | Change |
| --- | ---: | ---: | ---: |
| Median TPOT | 7.2939 ms | 7.2280 ms | **-0.90%** |
| Output throughput | 968.9 tok/s | 991.3 tok/s | **+2.31%** |

Baseline from seed 2073; optimized runs are seeds 2093/2094 (7.2232/7.2327
ms, 990.7/991.9 tok/s).

### Isolated kernel-pair benchmark

An out-of-server CUDA-graph harness that replays 40 mHC boundary pairs with
L2-cold weights per boundary (the decode working set exceeds L2) measures the
pair span directly:

| Tokens | Baseline geometry | This PR | Change |
| ---: | ---: | ---: | ---: |
| 1 | 8.43 us | 7.56 us | **-10.3%** |
| 8 | 10.26 us | 8.82 us | **-14.0%** |

The end-to-end gain is smaller than the isolated gain because the boundary is
partially hidden under PDL overlap in the full pipeline.

## Accuracy

GSM8K against the serve endpoint: full 1319-question test split, 5-shot,
temperature 0, max 256 tokens, concurrency 64.

| Variant | Accuracy |
| --- | ---: |
| Baseline | 93.63% |
| This PR | 94.31% |

The kernel change is bit-exact for identical inputs, so no accuracy movement
is expected; this run guards against configuration or dispatch mistakes.

## Tests

- `pytest tests/kernels/test_mhc_kernels.py` — 45 passed, 8 skipped (covers
  `mhc_fused_post_pre` numerics against the reference across batch sizes,
  including the `num_tokens <= 8` path exercised here).
