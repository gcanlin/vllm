# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Kimi-K3 routed-down GEMM → non-swizzled MXFP8."""

import argparse
import csv
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.models.kimi_k3.nvidia.ops.cute_dsl.routed_down_mxfp8 import (
    K_DIM,
    N_DIM,
    routed_down_mxfp8,
)

_DEFAULT_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192)


def _baseline(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp8_quantize

    output = torch.mm(x, weight.T).to(torch.bfloat16)
    q, scale = mxfp8_quantize(
        output,
        is_sf_swizzled_layout=False,
        alignment=256,
        backend="cute-dsl",
    )
    return q, scale.view(x.shape[0], -1)


def _median_cuda_ms(function: Callable[[], object], warmup: int, samples: int) -> float:
    for _ in range(warmup):
        function()
    torch.accelerator.synchronize()

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def _benchmark_shape(
    num_tokens: int,
    weight: torch.Tensor,
    warmup: int,
    samples: int,
) -> dict[str, float | int | str]:
    generator = torch.Generator(device="cuda").manual_seed(2000 + num_tokens)
    x = torch.randn(
        (num_tokens, K_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    baseline = lambda: _baseline(x, weight)
    baseline_ms = _median_cuda_ms(baseline, warmup, samples)
    row: dict[str, float | int | str] = {
        "M": num_tokens,
        "baseline_ms": baseline_ms,
        "fused_ms": "",
        "speedup": "",
    }

    fused = lambda: routed_down_mxfp8(x, weight)
    expected_q, expected_scale = baseline()
    actual_q, actual_scale = fused()
    torch.accelerator.synchronize()
    if not torch.equal(actual_scale, expected_scale):
        raise AssertionError(f"Scale mismatch at M={num_tokens}.")
    mismatch_rate = (
        (actual_q.view(torch.uint8) != expected_q.view(torch.uint8))
        .float()
        .mean()
        .item()
    )
    max_abs_diff = (actual_q.float() - expected_q.float()).abs().max().item()
    fused_ms = _median_cuda_ms(fused, warmup, samples)
    row.update(
        fused_ms=fused_ms,
        speedup=baseline_ms / fused_ms,
        q_mismatch_rate=mismatch_rate,
        q_max_abs_diff=max_abs_diff,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=_DEFAULT_M)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("Kimi-K3 routed-down MXFP8 requires SM100.")

    torch.manual_seed(0)
    weight = torch.randn(
        (N_DIM, K_DIM), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    rows = [_benchmark_shape(m, weight, args.warmup, args.samples) for m in args.m]
    fieldnames = (
        "M",
        "baseline_ms",
        "fused_ms",
        "speedup",
        "q_mismatch_rate",
        "q_max_abs_diff",
    )
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fieldnames))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
