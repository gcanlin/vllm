# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import statistics

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_up_gate_mix
from vllm.models.qwen4_exp.nvidia.ops.hc_up_gate_mix_cutedsl import (
    HC_UP_GATE_MIX_CONFIGS,
)

HC_COUNT = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC_COUNT * HIDDEN_SIZE
LORA_DIM = 320


def _bench_us(fn, inputs: tuple[torch.Tensor, ...]) -> float:
    for _ in range(10):
        fn(*inputs)
    torch.accelerator.synchronize()
    samples = bench_gpu_time_with_cupti(
        fn,
        input_args=inputs,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    return statistics.median(samples) * 1e3


def _baseline(
    x: torch.Tensor,
    lora: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return hc_gate_mix(x, F.linear(lora, weight), HC_COUNT)


def _fused(
    x: torch.Tensor,
    lora: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return hc_up_gate_mix(x, lora, weight, HC_COUNT)


def _gemm(lora: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(lora, weight)


def _gate_mix(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return hc_gate_mix(x, gate, HC_COUNT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise RuntimeError(f"The fused kernel requires SM100, got {capability}")

    torch.manual_seed(0)
    device = torch.device("cuda")
    weight = torch.randn(
        HYPER_HIDDEN_SIZE,
        LORA_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("dtype: BF16, HC=4, hidden=2560, rank=320, cold L2, CUDA graph")
    print(" M  path      gemm_us  mix_us  baseline_us  op_us   speedup  TFLOPS")
    for num_tokens in args.num_tokens:
        x = torch.randn(
            num_tokens,
            HYPER_HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
        )
        lora = torch.randn(
            num_tokens,
            LORA_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        inputs = (x, lora, weight)

        expected = _baseline(*inputs)
        actual = _fused(*inputs)
        torch.testing.assert_close(actual, expected)

        gate = F.linear(lora, weight)
        gemm_us = _bench_us(_gemm, (lora, weight))
        mix_us = _bench_us(_gate_mix, (x, gate))
        baseline_us = _bench_us(_baseline, inputs)
        fused_us = _bench_us(_fused, inputs)
        flops = 2 * num_tokens * HYPER_HIDDEN_SIZE * LORA_DIM
        tflops = flops / (fused_us * 1e6)
        path = "fused" if num_tokens in HC_UP_GATE_MIX_CONFIGS else "fallback"
        print(
            f"{num_tokens:2d}  {path:8s}  {gemm_us:7.3f}  {mix_us:6.3f}  "
            f"{baseline_us:11.3f}  {fused_us:6.3f}  "
            f"{baseline_us / fused_us:7.3f}  {tflops:7.3f}"
        )


if __name__ == "__main__":
    main()
