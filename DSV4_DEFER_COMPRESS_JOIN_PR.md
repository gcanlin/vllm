# [Performance] Defer DeepSeek V4 compressor stream join past attention

## Summary

DeepSeek V4 decoder layers with a compressor (the 21 C4A layers) run the
compressor chain on an aux stream but join it back into the compute stream
**before** the layer's sparse attention launches. Every 4th decode step,
each C4A layer closes a 4-token compression group: the close kernel
(`SparseAttnCompressNormRopeStore*C4`, ~13-15 us per layer) and its
aux-chain predecessors (~37 us total) gate the attention launch
one layer at a time, adding ~270 us of wall time to every 4th decode step
(+4.8% on those steps).

This PR moves the compressor join to the end of the layer's attention
module (after `_o_proj`). This is safe because the compressed-KV rows a
compressor writes during step `t` are only readable from step `t + 1`
onward: both the sparse indexer's and the MLA attention's read ranges are
bounded by per-request sequence lengths sampled at step start, and no
other layer touches a layer's compressor state or KV cache. The join
must still happen inside the same forward to keep CUDA-graph capture
legal (forks must rejoin the capture stream before capture ends) and to
keep cross-step ordering exact; it is auto-disabled under breakable
piecewise capture, where per-segment capture boundaries cannot carry a
dangling fork (`BreakableCUDAGraphCapture.is_active()`).

Gate: `VLLM_DSV4_DEFER_COMPRESS_JOIN` (default 1).

On 8x B200 with DeepSeek-V4-Flash-0731 (TP8 + EP8, `deep_gemm_mega_moe`
backend) BS=1 decode, the change improves median TPOT by **0.61%** and
output throughput by **0.6%**. GSM8K accuracy is unchanged within
measurement noise.

## Motivation

An nsys decomposition of the BS=1 decode CUDA graph (one replay =
~1399 kernels on device 0 across 5+ streams) shows a bimodal step-length
distribution: light steps run ~5.59 ms, while every 4th step ("close
step") runs ~5.85 ms (+270 us wall). The close steps carry 21 large
compressor kernels (~12-15 us each, one per C4A layer) on the compressor
aux stream, evenly spaced one per layer — each one immediately finished
before the same layer's attention chain starts on the compute stream.

The per-layer structure on a close step is:

```
compute stream:  q_proj+kv_insert ──► wait(compress_done) ──► indexer_topk ──► MLA ──► o_proj ...
compress aux:    kv_score GEMM ──► save_partial_states ──► C4 close (13 us) ┘
                 |<----------------- compressor chain ≈ 37 us ----------------->|
                 |<---- main path to attention ≈ 35 us ---->|
```

The join (`wait(compress_done)`) sits before attention because the
compressor rides the generic `execute_in_parallel` helper, which joins
all aux branches right after the default branch. The join is not a
correctness requirement for the attention itself (see above), so it can
move to the end of the attention module, after `_o_proj`. That gives the
compressor chain ~64 us of main-path slack instead of ~35 us, hiding the
close kernel completely on close steps (and changing nothing on light
steps, where the close kernel exits 8 states early and takes ~1.2 us).

A variant joining immediately after `_sparse_indexer_and_attn` was
measured first: it recovered only ~55% of the loss (+0.46% TPOT), because
the slack from the compressor fork to the attention start is shorter than
the compressor chain by ~7 us per layer. Moving the join to the end of
the attention module (this PR) recovers the remainder.

## Design

`DeepseekV4Attention._prepare_and_attn` (the indexer+compressor branch)
now, when multi-stream is available, capture is not breakable, and the
gate is on:

```python
self.ln_events[0].record()
with torch.cuda.stream(aux_streams[1]):
    self.ln_events[0].wait()
    compressor(kv_score, positions, self.rotary_emb)
    self.ln_events[2].record()
q, (indexer_inputs, _) = execute_in_parallel(
    project_query_and_cache_kv, [indexer_fn, None], ...
)                                   # indexer keeps its pre-attention join
```

`DeepseekV4Attention.forward` then joins the compressor at the end of the
module:

```python
ret = self._o_proj(o, positions)
if self._defer_compress_join:
    torch.cuda.current_stream().wait_event(self.ln_events[2])
```

- The indexer branch behavior shapes, events and streams are unchanged;
  the compressor's event/stream usage is the same pair it used before
  (`ln_events[2]` on aux stream 1), only the wait site moves.
- The single-stream / ROCm / breakable-capture paths are byte-identical
  to the previous code (sequential fallback when `aux_streams is None`,
  and the pre-attention join under breakable capture).
- Behavior at other batch sizes / prefill is unchanged in ordering: all
  compressor work of step `t` still completes before step `t` ends, and
  before any kernel that could read its output.

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
64 prompts, `--ignore-eos`. Variants differ only in
`VLLM_DSV4_DEFER_COMPRESS_JOIN` (0 vs 1, same build).

**Measurement protocol note.** Each freshly started server instance on
this machine lands in one of several distinct performance plateaus
(TPOT bands around 6.56 ms / 6.37 ms / 6.33 ms at BS=1); all comparisons
below pair runs **within the slow (6.56 ms) plateau** on both sides,
which is where every measurement here happened to land.

### BS=1 decode latency (concurrency 1)

| Variant | Median TPOT (ms) | Output throughput (tok/s) |
| --- | ---: | ---: |
| Baseline (join before attention) | 6.56 | 147.64 |
| This PR (join at attention end) | 6.52 | 148.50 |
| Change | **-0.61%** | **+0.58%** |

Per-run numbers: baseline 6.56/6.56/6.56 ms and 147.44/147.64/147.71
tok/s (seeds 8403/8407/8408, two server instances); this PR
6.52/6.52 ms and 148.49/148.51 tok/s (seeds 8405/8406).
Within-instance run spread is below 0.1% on every metric.

## Accuracy

GSM8K against the serve endpoint: full 1319-question test split, 5-shot,
temperature 0, max 256 tokens, concurrency 64.

| Variant | flexible-extract | strict-match |
| --- | ---: | ---: |
| Baseline | 79.61% | 61.26% |
| This PR | 79.23% | 59.21% |

Per-sample verdict diff: 127 flips, split 66-wrong / 61-right (symmetric
coin flips), consistent with run-to-run generation nondeterminism across
batch compositions rather than a systematic degradation. The change only
moves CUDA stream synchronization points; no kernel or
data movement is altered, so no accuracy movement is expected. This run
guards against scheduling mistakes (e.g. a missed join).

## Tests

- Capture validity: FULL CUDA-graph capture of decode sizes 1..256 and
  piecewise capture of mixed batches both succeed with the gate on
  (previously the first iteration of this change failed piecewise
  capture with `cudaErrorStreamCaptureUnjoined`, which motivated the
  breakable-capture guard).
- Serve + greedy smoke (`17*23` -> 391 with correct steps).
- GSM8K above.
