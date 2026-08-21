# [Kernel][Kimi K3] Fuse the DSpark context-KV output tail

## Summary

Fuse the Kimi K3 DSpark context-KV RMSNorm, RoPE, and grouped BF16 KV-cache
write into one CUDA kernel. The fast path consumes the token-major output of
the existing cross-layer KV projection directly, avoiding two transposes,
their materializations, a repeated-position copy, and separate RMSNorm, RoPE,
and cache-write launches.

The existing path remains the fallback for FP8 KV cache, non-uniform cache
layouts, missing slot mappings, and per-layer requests with absent mappings.

## Motivation

Kimi K3's five-layer DSpark draft projects target context states to five
independent 576-element MLA cache entries. After the projection, the current
BF16 path performs the following small operations separately:

1. Materialize the layer-major 512-element latent tensor.
2. Apply per-layer RMSNorm.
3. Materialize the layer-major 64-element positional tensor.
4. Replicate positions for all five layers.
5. Apply RoPE.
6. Insert the results into five KV caches with the grouped cache op.

These launches are latency-bound at the small context sizes used during
speculative decoding.

## Implementation

- Add `rms_norm_rope_and_cache_mla_grouped` to the stable libtorch extension.
- Launch one block per `(token, draft_layer)` and use the same 64-thread,
  eight-element RMSNorm reduction shape as the existing 512-wide RMSNorm.
- Apply GPT-J or NeoX RoPE to the 64-element positional suffix in the same
  block and write both BF16 results directly to the target layer's paged cache.
- Reuse the cached per-layer cache-pointer tensor and centralize grouped slot
  mapping construction in the DSpark model.
- Preserve all unsupported layouts and cache dtypes on the existing path.

## Correctness and tests

Run on an NVIDIA B200 with the nightly CUDA 13 image:

```text
.venv/bin/python -m pytest \
  tests/kernels/core/test_rotary_embedding_mla_cache_fused.py \
  -k rms_norm_rope_and_cache_mla_grouped -v

4 passed, 144 deselected
```

The test covers one and seventeen context tokens and both GPT-J and NeoX RoPE.
It compares every layer's cache against the existing RMSNorm, RoPE, and cache
write sequence.

```text
.venv/bin/python -m pytest tests/models/test_dspark_mla.py \
  -k context_kv_uses_fused_tail -v

1 passed, 10 deselected
```

The stable libtorch extension, including the new CUDA translation unit, was
built and linked successfully. Python compilation and pre-commit checks on the
changed files also passed.

## B200 kernel benchmark

Command:

```text
.venv/bin/python \
  benchmarks/kernels/benchmark_kimi_k3_dspark_context_kv.py \
  --tokens 1 8 16 32 64 --warmup 20 --samples 100
```

Median CUDA-event timings:

| Context tokens | Existing tail | Fused tail | Speedup | Saved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 55.06 us | 11.90 us | 4.62x | 43.15 us |
| 8 | 53.41 us | 11.89 us | 4.49x | 41.52 us |
| 16 | 53.58 us | 11.98 us | 4.47x | 41.60 us |
| 32 | 53.18 us | 13.41 us | 3.97x | 39.78 us |
| 64 | 53.66 us | 13.44 us | 3.99x | 40.22 us |

## End-to-end status

A 16xB200 Kimi K3 + `Inferact/Kimi-K3-DSpark` external-launcher A/B was
started with the local 1.56 TB checkpoint. The run reached distributed model
initialization and weight loading, but was stopped at the requester's direction
before candidate and baseline measurements completed. No end-to-end gain is
claimed here.

## Duplicate-work check

The open vLLM issues and PRs were searched for Kimi K3 DSpark, context KV,
RMSNorm/RoPE/cache fusion, and the DSpark tracker. PR #50585 merged the five
KV projections into one projection, but it retains this output tail. No open PR
was found that fuses this DSpark-specific tail.

## AI assistance

AI assistance was used to implement, test, benchmark, and draft this change.
The human submitter must review every changed line, understand the CUDA and
fallback contracts, rerun the relevant tests, and update the end-to-end/model
evaluation section before submitting an upstream PR.
