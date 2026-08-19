# [Performance] Fuse DeepSeek V4 shared experts into SM100 MegaMoE

## Summary

This PR fuses DeepSeek V4's replicated FP8 shared expert into DeepGEMM's
persistent SM100 MegaMoE kernel in the NVIDIA-specific model path.

Before this change, every MoE layer launched the routed FP4 MegaMoE kernel and
then ran the shared FP8 gate/up and down projections serially, followed by a
separate add. The new path lets the native SM100 kernel schedule shared L1,
routed dispatch/MMA, shared L2, and the final FP32 accumulation together. It
retains the checkpoint's FP8 shared weights and produces one BF16 output store.

On 8x B200 with DeepSeek-V4-Flash-0731, the change improves end-to-end output
throughput by 9.44% for a balanced 1K/128 workload and by 6.55% for an 8K/32
prefill-heavy workload. GSM8K accuracy is unchanged within measurement noise
(83.0% to 83.5%, 200 deterministic examples).

## Motivation and prior art

The NVIDIA DeepSeek V4 path already uses a persistent SM100 kernel for routed
experts, but its shared expert was outside that kernel. This left a substantial
serial tail in every one of the model's MoE layers.

Two other runtimes address that gap in different ways:

- TokenSpeed overlaps the separate routed and shared expert calls on CUDA
  streams ([commit 4e51bed](https://github.com/lightseekorg/tokenspeed/commit/4e51bed840911e483d8ddf5f24d61061cf60bc4d)).
- SGLang Waterfill represents the shared expert as an additional routed expert;
  its conversion path can load the FP8 shared checkpoint as FP4
  ([commit eb31b53](https://github.com/sgl-project/sglang/commit/eb31b5310c8bf076f5ac9624269697e299d0865f)).

This PR uses the native shared-expert support added to the DeepGEMM version
already pinned by vLLM. Compared with stream overlap, it avoids a second kernel
pipeline and a later add. Compared with Waterfill's FP8-to-FP4 conversion, it
keeps shared weights in FP8.

## Design

### 1. Native shared-expert weight preparation

The checkpoint stores shared-expert FP8 scales in 128x128 blocks, while the
native MegaMoE shared MMA expects a per-row, per-32-column scale. During the
model post-load hook, this PR expands each checkpoint scale to a mathematically
equivalent 1x32 view, packs it into DeepGEMM's TMA layout, and transforms the
shared gate/up and down weights with `transform_weights_for_mega_moe`.

The transformed gate/up weight is a full interleaved copy. The loader Parameter
is re-homed onto that allocation, allowing the original storage to be released.
This reduces the final model-load overhead from roughly 0.70 GiB to 0.03 GiB per
rank in this configuration.

### 2. Shared activation-scale staging without another launch

The routed FP4 path and shared FP8 path use different activation-scale layouts.
DeepGEMM selects the shared layout's `BLOCK_M` dynamically, so a static
conversion is not correct for all token counts.

`prepare_megamoe_inputs` now optionally writes the shared L1 scale view while
the packed UE8M0 scale is already resident in the existing Triton staging
kernel. The row mapping is the same dynamic MN-major/TMA permutation consumed
by DeepGEMM. This adds stores to an existing launch and does not allocate a
temporary tensor or launch a conversion kernel.

### 3. One persistent MoE pipeline

When native fusion is available, `fp8_fp4_mega_moe` receives both routed and
shared transformed weights. DeepGEMM executes the shared and routed work in the
persistent SM100 scheduler and combines the routed top-k weighted result with
the unweighted shared result before the BF16 output store. vLLM then skips the
old serial shared MLP and add.

### 4. Compatibility and rollback

- Enabled for DeepSeek V4's NVIDIA MegaMoE backend when shared weights are
  replicated: sequence parallel (including TP8+EP8) or TP1.
- Non-sequence-parallel TP configurations retain the serial path because their
  shared MLP is tensor-sharded.
- The DeepGEMM Python API is feature-detected before symmetric-memory setup.
  An older precompiled `_deep_gemm_C` extension safely falls back to the serial
  path and emits a rebuild warning.
- `VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1` provides an emergency
  production rollback without changing the backend selection.

## Benchmark

### Environment

- GPUs: 8x NVIDIA B200, SM100
- Driver: 590.48.01; CUDA: 13.2
- PyTorch: 2.13.0+cu130
- FlashInfer: 0.6.17
- Model: DeepSeek-V4-Flash-0731, 43 layers, 256 routed experts, top-k 6,
  one shared expert
- vLLM base: `583a00257d4c5d1a54063d956057df1df6822b06`
- DeepGEMM: `8b1392b978f5a03c828dd1711090d7fb50958b8a` (the existing vLLM pin)
- Parallelism: TP8 + EP8 with sequence parallel
- Prefix cache was kept enabled, but every measured run used a different random
  seed, so measured prompts did not hit the cache.
- Each table entry is the arithmetic mean of seeds 2027, 2028, and 2029. The
  before and after runs use matching seeds.

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

The baseline adds
`VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1`; the optimized run uses the
default.

### Balanced workload: 128 requests, 1024 input, 128 output, concurrency 64

| Metric | Baseline | Fused | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 3,415.45 tok/s | 3,737.98 tok/s | **+9.44%** |
| Total token throughput | 30,739.04 tok/s | 33,641.81 tok/s | **+9.44%** |
| Mean TTFT | 588.52 ms | 530.15 ms | **-9.92%** |
| Mean TPOT | 14.095 ms | 12.935 ms | **-8.23%** |
| Median ITL | 9.644 ms | 8.939 ms | **-7.31%** |

Per-seed output-throughput gains were +11.38%, +11.24%, and +5.80%; all three
paired runs improved.

### Prefill-heavy workload: 32 requests, 8192 input, 32 output, concurrency 16

| Metric | Baseline | Fused | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 234.09 tok/s | 249.42 tok/s | **+6.55%** |
| Total token throughput | 60,160.73 tok/s | 64,099.90 tok/s | **+6.55%** |
| Mean TTFT | 901.81 ms | 836.36 ms | **-7.26%** |
| Mean TPOT | 40.584 ms | 38.458 ms | **-5.24%** |
| Median ITL | 8.256 ms | 7.493 ms | **-9.24%** |

Per-seed output-throughput gains were +2.65%, +7.52%, and +9.52%; all three
paired runs improved.

### Device memory

| Per-rank metric | Baseline | Fused | Delta |
| --- | ---: | ---: | ---: |
| Model load | 20.73 GiB | 20.76 GiB | +0.03 GiB |
| Runtime consumed | 23.04 GiB | 23.17 GiB | +0.13 GiB |
| Reported KV capacity | 386,897 tokens | 386,512 tokens | -385 (-0.10%) |

## Accuracy

The accuracy check uses the first fixed 200 examples from the official
`openai/gsm8k` test split, the model's DeepSeek V4 prompt encoding, low thinking,
temperature 0, and a 512-token generation limit.

| Variant | Correct | Accuracy | Unparsed |
| --- | ---: | ---: | ---: |
| Baseline serial shared expert | 166 / 200 | 83.0% | 0 |
| Native fused shared expert | 167 / 200 | 83.5% | 0 |

The final paired result has 156 examples correct in both modes, 10 baseline-only
correct, 11 fused-only correct, and 23 wrong in both. An exact McNemar test gives
`p = 1.0`, so this sample shows no measurable accuracy regression.

Exact token streams are not expected to match: the native shared kernel uses
per-32 activation quantization, while the previous generic FP8 linear path uses
per-128 activation quantization. The checkpoint weight scaling remains
mathematically equivalent and the task-level check covers the resulting numeric
variation.

## Validation

```bash
ruff format vllm/envs.py \
  vllm/models/deepseek_v4/nvidia/model.py \
  vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py \
  tests/models/test_deepseek_v4_mega_moe.py

ruff check vllm/envs.py \
  vllm/models/deepseek_v4/nvidia/model.py \
  vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py \
  tests/models/test_deepseek_v4_mega_moe.py

pytest -q tests/models/test_deepseek_v4_mega_moe.py
# 17 passed
```

The tests cover native shared-weight finalization and storage ownership, the
serial/fused no-double-add behavior, bitwise routed input staging, and the
dynamic shared activation-scale TMA layout for `BLOCK_M` values 8, 32, 96, 128,
and 192, including a cross-block token range.

## Scope and limitations

This optimization is intentionally limited to the NVIDIA DeepSeek V4 MegaMoE
path on SM100 with FP4 routed experts and replicated FP8 shared experts. Other
MoE backends and tensor-sharded shared experts retain their existing behavior.
The implementation does not change the DeepGEMM source pin.
