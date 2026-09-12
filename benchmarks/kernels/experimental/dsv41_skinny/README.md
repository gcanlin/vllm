# DSV4.1 Flash skinny GEMM experiments

This directory preserves the implementation, measurements and explanations for
the September 12–13, 2026 B200 experiment. It is an experimental custom worker;
checking out this branch does not enable skinny dispatch in normal `vllm serve`.

On node30, 8×B200, TP8+EP+SP, the best tested scope was the measured M≤4 dispatch:

| Input / output tokens | Concurrency | Baseline output token/s | Full candidate | M≤4 candidate | M≤4 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 / 256 | 1 | 126.754 | 131.102 | 131.815 | +3.99% |
| 1024 / 256 | 4 | 458.037 | 459.449 | 460.872 | +0.62% |
| 8192 / 1024 | 16 | 1442.062 | 1431.263 | 1446.204 | +0.29% |

These are medians of three `vllm bench serve` client repetitions per fresh
server, run in order: baseline, full candidate, M≤4 candidate. They are not
independent server replications. All 1260 formal requests succeeded with the
specified lengths. Treat C4/C16 as small or negligible gains under this protocol.

The full candidate passed 5152 loaded-weight numerical checks (maximum relative
RMSE 0.04260%); M≤4 passed 4640 (0.03461%). Every selected dispatch cell was
exercised during warmup/capture on every rank. GSM8K was not run for this
experiment, and local checks do not establish model-quality equivalence.

## Contents

- [Full measurement report](RESULTS.md): screening, per-kernel timings,
  serving protocol, failed candidates and limitations.
- [Implementation and gain explanation](EXPLANATION.md): MXFP8, skinny,
  comparison with K3, and what has not been established about MoE bottlenecks.
- [Model diagram](model-skinny-map.md), with editable
  [overall](model-overall.mmd) and [attention](model-attention.mmd) Mermaid sources.
- [DSV4.1 / K3 dispatch tables](gemm-dispatch-tables.md).
- [Triton kernels](kernels.py), [dispatch](dispatch.py),
  [worker](skinny_worker.py), [M≤4 worker](lowm_worker.py).
- [Screening](screen.py), [whole wo_a chain](screen_woa.py),
  [separate confirmation](confirm.py), and unsuccessful CuTe MXFP8 exploration.
- `bench-a/`, `bench-b/`, `bench-lowm/`: unchanged raw client JSON/logs,
  server logs, command manifests, rank-local checks and capture coverage.
- `serving-summary.json`, `serving-lowm-summary.json`: original derived reports.
- [Archive manifest](archive-manifest.json): SHA256 hashes of original evidence
  and source snapshots. `original-source.tar.gz` contains the unmodified final
  experiment scripts and documents, including the exact original benchmark client.

Model-derived `weights/*.pt` are intentionally omitted. The baseline worker
exports these tensors again when screening is reproduced. The code snapshot is
the final experiment state; per-arm `commands.json` records earlier script
hashes and differences during development. The unsuccessful CuTe MXFP8 variant
is retained for traceability and is not selected by either serving candidate.

## Validate the archived results without GPUs

From a Python environment with `uv` available:

```bash
uv run --active --no-project python \
  benchmarks/kernels/experimental/dsv41_skinny/verify_archive.py
```

This verifies the original file hashes and source archive, reruns both serving
validators in a temporary directory, and compares their reports with the
archived summaries. It does not launch a server or rewrite recorded evidence.

## Reproduce in the prepared nightly container

The recorded source was `30aa0ade39ab6625674865f578fae19a49cf3e43`, with local DSV4.1
baseline changes. The ordinary nightly native build was
`e7edf17cea217e52701f913cd8491fcacf2d9490`, Torch 2.13.0+cu130, FlashInfer
0.6.18.post1 and Triton 3.7.1. Use a compatible prepared editable vLLM environment.

`baseline-dsv41.patch` preserves the relevant source changes on top of that
base: the shared-expert padding fix and the disabled sparse-index reuse
experiment, with their tests. Shared-expert fusion is enabled in **all** arms;
sparse-index reuse is disabled in **all** arms. Unrelated dirty K3 benchmark
changes are excluded. The original full-worktree diff hash in the manifests
therefore differs from this focused patch hash. The published package itself
does not apply the patch or modify production model files.

Starting at the root of a clean checkout of this branch:

```bash
export DSV41_SOURCE_ROOT="$PWD"
export DSV41_MODEL=/gpfs/mszn/models/deepseek-ai/DeepSeek-V4.1-Flash
EXP="$DSV41_SOURCE_ROOT/benchmarks/kernels/experimental/dsv41_skinny"

# Apply only if these baseline changes are not already present.
git apply --check "$EXP/baseline-dsv41.patch"
git apply "$EXP/baseline-dsv41.patch"

# Fresh scratch directory keeps the archived evidence unchanged.
RUN="$(mktemp -d /tmp/dsv41-skinny.XXXXXX)"
cp "$EXP"/*.py "$EXP"/*.sh "$EXP/selected.json" "$RUN/"

PYTHONPATH="$DSV41_SOURCE_ROOT" uv run --active --no-project python \
  "$RUN/run_bench.py" bench-a:baseline bench-b:candidate
PYTHONPATH="$DSV41_SOURCE_ROOT" uv run --active --no-project python \
  "$RUN/run_bench_lowm.py" bench-lowm:candidate

uv run --active --no-project python "$RUN/summarize.py"
uv run --active --no-project python "$RUN/summarize_lowm.py"
```

The drivers use all eight GPUs and localhost port 8032, start each server in a
separate process group, and stop only that group. Server/client parameters are
in `serve.sh`, `serve-lowm.sh` and `bench.sh`. Launching a driver runs the entire
campaign; the archive verifier above performs no GPU work.

For a standalone foreground candidate server after setting these variables:

```bash
export PYTHONPATH="$DSV41_SOURCE_ROOT:$RUN"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DSV41_REUSE_SPARSE_INDICES=0
export VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=0
export DSV41_GEMM_MODE=candidate
export DSV41_GEMM_OUTPUT="$RUN/manual"
mkdir -p "$DSV41_GEMM_OUTPUT"
bash "$RUN/serve-lowm.sh"
```

The standalone command requires the same eight-GPU distributed networking
configuration as the driver; inspect its explicit GLOO/NCCL interface settings
when running on another host. `final_check.py` is a historical node30/source
receipt checker; it is not a portable reproduction check.

## Packaging changes

Arithmetic and dispatch selections are preserved. Published scripts add SPDX
headers and repository formatting, resolve the benchmark client within this
directory, support `DSV41_SOURCE_ROOT`/`DSV41_MODEL`, and snapshot the selected
plan automatically for each candidate run. Triton imports use the vLLM wrapper;
screening uses equivalent `torch.accelerator` device, synchronization and cache
APIs. The original scripts are retained separately. GPU measurements have not
been rerun merely for this packaging.

This experiment was developed and documented with Codex assistance. It does not
establish a 4% architecture ceiling, a 10% expected gain, or a deployment-quality
acceptance result. See the [validation receipt](VALIDATION.md) for publication
checks performed on this package.
