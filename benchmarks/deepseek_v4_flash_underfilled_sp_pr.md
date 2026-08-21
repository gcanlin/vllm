# [Perf][DeepSeek V4] Use all-reduce for underfilled sequence parallelism

## Purpose

DeepSeek V4 sequence parallelism pads the attention output to the TP world size before reduce-scatter. During low-batch decode with TP8, B1 communicates one real row and seven zero rows on every decoder layer.

This change adds an opt-in underfilled path to `sp_reduce_scatter`. When `0 < num_tokens <= tp_size // 2`, it all-reduces only the real rows and locally selects the row assigned to each SP rank. Ranks beyond `num_tokens` return one zero row, which is algebraically equivalent to reducing and scattering the padded tensor. Larger batches retain the existing custom reduce-scatter path.

Only the NVIDIA DeepSeek V4 decoder opts in. Other callers keep the previous behavior by default.

## Test Plan

- Run the sequence-parallel unit tests, including all TP4 rank mappings and the half-occupancy fallback boundary.
- Run the DeepSeek V4 MegaMoE model tests.
- Benchmark the baseline and this branch on 8×B200 with the same TP8+EP server configuration, using three measured seeds per case.
- Run the full 1319-question GSM8K 5-shot evaluation with temperature 0 and seed 42 on both revisions.

## Test Result

Environment: 8×NVIDIA B200 183 GB, CUDA 13.0, PyTorch 2.13.0, TP8+EP, `deep_gemm_mega_moe`, FP8 KV cache, MXFP4 indexer KV. Baseline: `ba53da60bb1aeec200d05101936a5474ee46c4eb`.

Inputs/outputs are 1024/512 for B1-B4. Results are medians of three runs.

| Scenario | Baseline output tok/s | This PR output tok/s | Delta | Baseline mean TPOT | This PR mean TPOT | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1/C1 | 155.129 | 157.888 | **+1.779%** | 6.2848 ms | 6.1593 ms | **-1.996%** |
| B2/C2 | 287.138 | 289.778 | **+0.919%** | 6.7000 ms | 6.5676 ms | **-1.977%** |
| B4/C4 | 566.258 | 577.144 | **+1.922%** | 6.8579 ms | 6.7069 ms | **-2.202%** |
| 128 prompts/C64 | 3819.756 | 3821.749 | +0.052% (noise) | 12.6886 ms | 12.6814 ms | -0.057% (noise) |

The C64 case is primarily outside the optimization threshold and shows no measurable regression.

GSM8K 5-shot:

| Revision | Accuracy | Invalid rate |
| --- | ---: | ---: |
| Baseline | 1235/1319 (0.9363153904473086) | 0.0 |
| This PR | 1235/1319 (0.9363153904473086) | 0.0 |

Unit tests:

```text
tests/models/kimi_k3/test_sequence_parallel.py: 27 passed
tests/models/test_deepseek_v4_mega_moe.py: 17 passed
```

Duplicate-work searches covered sequence-parallel padding, underfilled reduce-scatter/all-reduce, and DeepSeek V4 low-batch communication. [PR #52079](https://github.com/vllm-project/vllm/pull/52079) targets BF16 GEMM-RS for Kimi K3 prefill-sized workloads and is distinct from this attention-output collective substitution.

## AI assistance disclosure

This change, benchmark analysis, tests, and draft description were prepared with OpenAI Codex assistance. A human contributor must review and take responsibility for the code and results before submitting a PR.

---

<details>
<summary>Essential Elements of an Effective PR Description Checklist</summary>

- [x] The purpose of the PR is described.
- [x] The test plan and exact coverage are described.
- [x] Before/after performance and accuracy results are included.
- [x] No user-facing documentation update is required; the optimization is internal.

</details>
