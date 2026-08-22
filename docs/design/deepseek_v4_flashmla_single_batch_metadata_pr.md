# [Perf][DeepSeek V4] Speed up FlashMLA single-request decode metadata

## Summary

Add a specialized FlashMLA metadata scheduler for `batch_size == 1`. It computes the single-request partitioning analytically and writes partition metadata in parallel across one warp. The existing generic scheduler remains unchanged for larger batches.

This benefits DeepSeek V4 Flash latency because its hybrid attention path invokes the metadata kernel four times per decode step.

## Implementation

- Add the FlashMLA single-batch fast path for dense and sparse decode, including dynamic top-k and extra-cache lengths.
- Apply the change to vLLM's pinned FlashMLA source during CMake configuration.
- Extend the sparse FlashMLA decode smoke test to validate the generated scheduler metadata.

The kernel change belongs upstream in `vllm-project/FlashMLA`. This branch carries a downstream patch only to make the vLLM result independently buildable and reproducible. After the FlashMLA PR lands, the final vLLM change should replace the patch with a FlashMLA commit-pin update.

## Performance

Environment: 8× NVIDIA B200, DeepSeek V4 Flash, TP8 + EP, FP8 KV cache, FP4 indexer KV, input/output 1024/128 tokens, concurrency 1. Each server was warmed up with two requests; results are five runs of 20 requests.

| Metric | Baseline | This PR | Change |
| --- | ---: | ---: | ---: |
| FlashMLA metadata kernel | 28.378 µs | 2.054 µs | **13.82×** |
| Mean TPOT | 6.6786 ms | 6.5819 ms | **-1.47%** |
| Output throughput | 133.61 tok/s | 134.46 tok/s | **+0.64%** |

The mean TPOT reduction is 96.64 µs/token with a paired 95% confidence interval of [94.93, 98.35] µs.

## Correctness

- FlashMLA PyTorch reference: 5/5 single-batch cases passed, covering V4 SWA/C4A/C128A, head counts 64/128, dynamic top-k/extra lengths, and a V3.2 shape.
- GSM8K first 200 examples: baseline 168/200 (84.0%), optimized 173/200 (86.5%), with zero parse failures in both runs. This is treated as no observed regression rather than an accuracy improvement because cross-server dynamic batching is not bit deterministic.
- vLLM sparse FlashMLA decode smoke test passes and checks the exact partition metadata.

## Test plan

```text
pytest -q tests/kernels/attention/test_flashmla_sparse.py
git diff --check
```

AI-assisted development disclosure: the implementation and documentation were produced with OpenAI Codex. All code paths, benchmark results, and correctness results were reviewed and validated locally.
