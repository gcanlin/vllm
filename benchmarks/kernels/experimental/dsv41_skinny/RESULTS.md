# DSV4.1 complete dense-projection skinny screening

This is the historical experiment report. Host paths and commands below describe
the measured runs; use [README.md](README.md) for the published package's
reproduction commands. Model-derived weight tensors are excluded from this
archive and can be exported again by the baseline worker.

Status: kernel screening, full serving A/B and M<=4 control complete.
The best tested scope is M<=4: +3.99% output throughput at C1, +0.62% at C4,
and +0.29% at C16 (effectively unchanged). This supports a low-concurrency
candidate, not a 10% end-to-end performance claim. GSM8K remains untested.

Runtime: node30, tmux window 5, `canlin-vllm-nightly-20260911`, 8×B200,
TP8 + EP + sequence parallel, ordinary nightly native build
`e7edf17cea217e52701f913cd8491fcacf2d9490`, editable source
`/gpfs/mszn/workspace/vllm` at `30aa0ade39ab6625674865f578fae19a49cf3e43`.
The existing working-tree changes are preserved. Experiments use a custom worker
and process-local method overrides; production source is unchanged.

## Actual loaded-weight inventory

`inventory-rank*.json` records module class, tensor dtype, shape, strides and
quantization kernel on all eight ranks. `weights/group*.pt` contains one
representative set of runtime-loaded rank-0 tensors per module/layout group.

Local weight dimensions below use `(N, K)` notation, irrespective of storage.

| Group | Projection | N × K | Type | Count | Actual path |
| --- | --- | --- | --- | ---: | --- |
| 1 | attention fused_wqa_wkv | 1792 × 5120 | MXFP8 | 40 | Quantization + CuTeDSL GEMM |
| 2 | attention wq_b | 4096 × 1280 | MXFP8 | 40 | Already quantized Q from fused norm |
| 3 | attention wo_a | 1024 × 4096, one local group | MXFP8 | 40 | Inverse RoPE quantization + DeepGEMM BMM |
| 4 | attention wo_b | 5120 × 1024 | MXFP8 | 40 | Quantization + CuTeDSL GEMM, then RS |
| 5 | MoE router gate | 384 × 5120 | BF16 → FP32 | 40 | Existing ll_bf16 low-latency dispatch |
| 9 | Engram wkv | 25600 × 6144 | MXFP8 | 2 | Quantization + CuTeDSL GEMM; local SP rows |
| 10 | indexer wq_b | 4096 × 1280 | MXFP8 | 8 | Shares the already quantized Q |
| 11 | indexer weights_proj | 32 × 5120 | BF16 | 8 | F.linear, auxiliary stream |
| 12 | indexer wk | 128 × 512 | BF16 | 4 | F.linear on compressor latent |
| 13 | ratio-2 compressor fused_wkv_wgate | 1024 × 5120 | BF16 → FP32 | 3 | torch.mm with FP32 output, auxiliary stream |
| 15 | ratio-1 compressor fused_wkv_wgate | 512 × 5120 | BF16 → FP32 | 1 | torch.mm with FP32 output, auxiliary stream |
| 16 | lm_head | 16160 × 5120 | BF16 | 1 | F.linear |

Groups 6/7 are the shared-expert Linear weights retained after fusion; the
current model skips their standalone forward because shared experts are already
inside MegaMoE. Groups 0/8/14 are embedding lookups. Engram q/k weights are
elementwise normalization/gating parameters, not additional dense projections.
mHC matrix work is embedded in the existing fused mHC implementation and is not
split apart for this experiment. The previous experiment screened group 3's
backend/whole inverse-RoPE chain; additional skinny work must preserve that chain.

## Controls

- Small-M coverage targets M=1/2/4/8/16; dispatch uses measured winners only.
  M is the row count presented to each local GEMM, not request concurrency;
  sequence parallelism and compressor shapes can make these differ.
- MXFP8 candidates retain each 32-element activation scale and FP8 rounding.
  Prequantized Q does not receive another quantization step.
- FP32-output compressor/router candidates must retain FP32 outputs.
- Rotating, pointer-distinct weights exceed 2.5× physical L2 capacity. Final
  winners require separate confirmation beyond the tuning measurements.
- Formal serving uses `vllm bench serve`, based on `kimi-bench.sh`: fixed
  1024/256 C1, 1024/256 C4, and 8192/1024 C16; ignore EOS, temperature 0,
  seed 42, three client repetitions per fresh server.
- Local numerical checks and short smoke prompts do not establish GSM8K
  accuracy. Any quality evaluation is reported separately.

## Duplicate-work check

Open-PR searches for `skinny DeepSeek` and `MXFP8 GEMM` found no direct existing
DSV4.1 implementation for this complete projection dispatch. Related work:
[K3 SM100 skinny #53534](https://github.com/vllm-project/vllm/pull/53534),
[block-FP8 low-latency prototype #43214](https://github.com/vllm-project/vllm/pull/43214),
and [optional router GEMM #49312](https://github.com/vllm-project/vllm/pull/49312).
The current router already uses its own low-latency BF16 implementation; its
baseline must be that implementation, not plain cuBLAS.

## Confirmed kernel candidates

Local latency reductions, not end-to-end gains. `selected.json` contains all
28 measured cells, including three whole inverse-RoPE/wo_a chains. Each selected
cell passed three alternating baseline/candidate timing rounds and changing-input
graph replay checks. All candidates preserve the existing output dtype.

| Projection | Selected M | Representative baseline → candidate, µs |
| --- | --- | --- |
| fused_wqa_wkv | 1 | M1: 8.05 → 6.30 |
| inverse RoPE + wo_a | 1, 2, 4 | M1: 7.93 → 4.41; M4: 7.93 → 6.52 |
| wo_b | 1, 2 | M1: 4.87 → 3.04; M2: 4.86 → 4.14 |
| indexer weights_proj | 1, 2, 4, 8, 16 | M1: 4.00 → 1.74; M16: 4.17 → 2.54 |
| indexer wk | 1, 2, 4, 8, 16 | M1: 2.08 → 1.51; M16: 4.03 → 1.69 |
| ratio-2 compressor | 1, 2, 4, 8, 16 | M1: 6.50 → 2.93; M16: 6.56 → 4.47 |
| ratio-1 compressor | 1, 2, 4, 8, 16 | M1: 5.49 → 2.13; M16: 5.76 → 3.71 |
| lm_head | 1, 2 | M1: 29.70 → 24.38; M2: 28.47 → 24.92 |

The router, main/indexer wq_b, Engram wkv, and unselected token counts retain
their original implementation. Compressor/indexer work can overlap other work,
so summing these savings is not a critical-path estimate.

`screen-bf16.json` has 627 valid measurements; `screen-mx.json` has 390 valid
measurements, including fused-quant SIMT and dequantizing Tensor Core variants.
The first MX run encountered an FP8 masked-load literal compilation error;
`screen-mx-initial.*` retains that failed run and is excluded from timing claims.
The mask literal was corrected from integer zero to floating-point zero.

`screen-cute-mx.json` records an additional 200 configurations adapting
PR #43214 to 32-element scales using a 128-FP8-element K tile. 175 configurations
passed local numerical checks but were slower; 25 configurations with tile_n=8,
four DMA warps failed numerical checks and were excluded. This adaptation is
an unsuccessful exploratory fork, not an evaluation of the unmodified PR or
a general limit on native MXFP8 skinny GEMM. It is not selected by serving.

`screen-woa.json` includes the entire inverse-RoPE/quantization/projection chain.
`selected-woa.json` retains its separate alternating confirmation and changing
input/position graph tests. The selected M1/M2/M4 paths use SIMT GEMM instead of
the earlier experiment's FlashInfer backend swap.

## Reproduction

Inside the active venv in tmux window 5:

```bash
PYTHONPATH=/gpfs/mszn/workspace/vllm uv run --active --no-project python \
  /gpfs/mszn/workspace/dsv41-skinny-full-20260912/run_bench.py \
  new-baseline:baseline new-candidate:candidate

PYTHONPATH=/gpfs/mszn/workspace/vllm uv run --active --no-project python \
  /gpfs/mszn/workspace/dsv41-skinny-full-20260912/run_bench_lowm.py \
  new-lowm:candidate
```

The driver starts a fresh TP8 server for each arm, validates fixed request/token
counts, and stops only its own process group. It leaves the interactive shell
inside the existing nightly container at `/gpfs/mszn/workspace/vllm`.

## First serving comparison

Median of three client repetitions per fresh server, baseline followed by the
full measured dispatch. All 840 formal requests succeeded with exact input and
output lengths. These client repetitions are not independent server replications.

| Workload | Baseline output token/s | Full dispatch output token/s | Change | Mean TPOT change |
| --- | ---: | ---: | ---: | ---: |
| 1024 / 256, C1 | 126.754 | 131.102 | +3.43% | -3.52% |
| 1024 / 256, C4 | 458.037 | 459.449 | +0.31% | -0.88% |
| 8192 / 1024, C16 | 1442.062 | 1431.263 | -0.75% | +0.67% |

`serving-summary.json` retains all runs, TTFT, E2E and tail metrics. The
integrated candidate passed 5152 loaded-weight numerical checks across all
layers and TP ranks, maximum relative RMSE 0.0004260 (0.04260%). Every selected
cell was exercised during runtime warmup/capture on every rank. These are local
checks: four greedy smoke prompts yielded two exact text matches and two
different stopping points. GSM8K has not been run for this experiment.

Source HEAD/diff, native image build, serving parameters and benchmark client
match. The baseline ran before candidate development finished: experiment
scripts were formatted and candidate-only capture counters were added to the
worker afterward. Per-arm command/script manifests retain these differences.
The full selected table is preserved in `bench-b/selected.json`.

## M<=4 serving control and final assessment

Because C16 did not benefit, `lowm_worker.py` adds a separate control retaining
only M<=4 cells (20 cells, all eight projection types). `run_bench_lowm.py` uses
the same C1/C4/C16 protocol with `serve-lowm.sh`. The only serving argument change
is the worker class; the low-M worker filters the plan before loading the model.

| Workload | Baseline output token/s | M<=4 output token/s | Change | Mean TPOT change |
| --- | ---: | ---: | ---: | ---: |
| 1024 / 256, C1 | 126.754 | 131.815 | +3.99% | -3.71% |
| 1024 / 256, C4 | 458.037 | 460.872 | +0.62% | -0.65% |
| 8192 / 1024, C16 | 1442.062 | 1446.204 | +0.29% | -0.06% |

All 1260 formal requests across baseline, full dispatch and M<=4 succeeded,
with exact input/output lengths and no recorded request errors. The M<=4
candidate passed 4640 loaded-weight checks over all layers and TP ranks;
maximum relative RMSE was 0.000346118 (0.03461%). All 20 selected cells were
invoked during runtime warmup/capture on every rank. Four smoke prompts again
gave two exact text matches and two differences; these are not a model-quality
evaluation. GSM8K has not been run for either candidate in this experiment.

`serving-lowm-summary.json` contains all repetitions, latency metrics, runtime
coverage and smoke comparisons. `bench-lowm/selected.json` preserves the exact
filtered plan and is validated against the M<=4 subset of `selected.json`.
`serving-validation.log` and `serving-lowm-validation.log` both finish with their
validation-complete markers.

The low-M result retains the C1 gain and shows no C16 regression in this run.
It is the preferred scope for further development. The experiments ran in order:
one fresh baseline server, one full-dispatch server, then one M<=4 server, each
with three client repetitions. Server-order variation and changes in generated
tokens/routing have not been separated from dispatch effects. In particular,
the roughly 1% full-versus-low-M difference at C16 does not establish that
larger-M auxiliary GEMMs caused the original regression. Treat C4/C16 as small
or negligible end-to-end gains under this protocol.

The main additional candidates are the complete inverse-RoPE/wo_a chain,
BF16 indexer/compressor projections and lm_head. Their isolated savings cannot
be added: auxiliary streams overlap other work, and each workload presents
different M values. Main/indexer wq_b and Engram did not produce a winning
candidate in this screening, while the router was already using a low-latency
implementation.

## Final workspace state

`final-receipt.json` records successful cleanup at 2026-09-13 00:01:30 +08:00:
all eight GPUs had zero allocated memory, port 8032 was free, and the source
HEAD and original tracked-diff hash were preserved. Tmux window 5 remains in
the existing nightly container, at `/gpfs/mszn/workspace/vllm`, with
`/tmp/dsv41-index-reuse-venv` active. Trial code remains in this experiment
directory and is enabled only by the custom workers; no production-source
change or PR was made for this experiment.
