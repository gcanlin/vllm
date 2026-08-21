# Kimi-K3 routed-down MXFP8 fusion

## Serving configuration

The end-to-end experiment uses two 8-GPU B200 nodes without P/D
disaggregation or Dspark.

| Setting | Value |
| --- | --- |
| Parallelism | TP8, DP2, EP16, PP1 |
| All-to-all backend | DeepEP V2 |
| MoE backend | `FLASHINFER_TRTLLM_MXFP4_MXFP8` |
| Maximum model length | 32,768 |
| Maximum sequences | 512 |
| Maximum batched tokens | 1,024 |
| Maximum CUDA graph capture size | 512 |
| CUDA graph mode | `FULL_DECODE_ONLY` |
| KV cache | 25 GiB per GPU, FP8 |
| Prefix caching | Enabled |

The serving benchmark uses random prompts with a fixed length of 128 tokens and
requests 256 output tokens with EOS ignored. Requests are split evenly between
the two data-parallel API endpoints except at concurrency one.

## Baseline results

The baseline sets `VLLM_KIMI_K3_ROUTED_DOWN_MXFP8=0`. This is one complete
benchmark pass after a separate warmup.

| Global concurrency | Prompts | Request/s | Output token/s | Mean TTFT (ms) | Mean TPOT (ms) | Approx. token/user/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.123 | 31.5 | 287.5 | 30.71 | 32.6 |
| 16 | 32 | 1.791 | 458.4 | 461.1 | 33.22 | 30.1 |
| 64 | 128 | 4.851 | 1,242.0 | 930.9 | 47.89 | 20.9 |
| 128 | 256 | 8.173 | 2,092.4 | 1,115.4 | 56.67 | 17.6 |
| 256 | 512 | 13.063 | 3,344.2 | 1,779.9 | 69.03 | 14.5 |
| 512 | 1,024 | 18.605 | 4,762.9 | 3,268.1 | 92.89 | 10.8 |

Raw JSON and console logs are retained on both nodes under
`/home/zetyun/vllm-k3-routed-down-mxfp8/ab-results/` with the prefix
`baseline-r1`.

## Fused results

The fused run sets `VLLM_KIMI_K3_ROUTED_DOWN_MXFP8=1`. All other server and
client settings are unchanged from the baseline. This is one complete benchmark
pass after a separate warmup; all requests completed successfully.

| Global concurrency | Prompts | Request/s | Output token/s | Mean TTFT (ms) | Mean TPOT (ms) | Approx. token/user/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.123 | 31.5 | 284.1 | 30.74 | 32.5 |
| 16 | 32 | 1.782 | 456.3 | 581.3 | 32.91 | 30.4 |
| 64 | 128 | 4.871 | 1,247.1 | 912.8 | 47.75 | 20.9 |
| 128 | 256 | 8.186 | 2,095.7 | 1,230.8 | 56.12 | 17.8 |
| 256 | 512 | 12.874 | 3,295.8 | 1,876.2 | 69.78 | 14.3 |
| 512 | 1,024 | 18.513 | 4,739.3 | 3,305.1 | 93.29 | 10.7 |

Raw JSON and console logs are retained on both nodes under
`/home/zetyun/vllm-k3-routed-down-mxfp8/ab-results/` with the prefix
`fused-r1`. The startup logs use the prefix `fused-bt1024-r2`.

## A/B comparison

Positive output-throughput deltas are improvements. Negative TPOT deltas are
improvements.

| Global concurrency | Baseline output token/s | Fused output token/s | Output throughput delta | Baseline TPOT (ms) | Fused TPOT (ms) | TPOT delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31.5 | 31.5 | -0.06% | 30.71 | 30.74 | +0.11% |
| 16 | 458.4 | 456.3 | -0.46% | 33.22 | 32.91 | -0.95% |
| 64 | 1,242.0 | 1,247.1 | +0.41% | 47.89 | 47.75 | -0.30% |
| 128 | 2,092.4 | 2,095.7 | +0.16% | 56.67 | 56.12 | -0.98% |
| 256 | 3,344.2 | 3,295.8 | -1.45% | 69.03 | 69.78 | +1.08% |
| 512 | 4,762.9 | 4,739.3 | -0.50% | 92.89 | 93.29 | +0.43% |

The single-pass end-to-end result does not show a stable serving improvement.
The deltas are mixed and mostly within approximately 1%; the largest throughput
change is a 1.45% regression at concurrency 256. Repeated trials would be needed
to distinguish small effects from run-to-run variance, but this first result
does not justify an accuracy evaluation or a performance claim.
