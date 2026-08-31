# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen4Exp QSA output gating on NVIDIA GPUs."""

import argparse
import statistics
from functools import partial

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_output_gate


@torch.compile(fullgraph=True)
def _compiled_gate(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return output * torch.sigmoid(gate)


def _eager_gate(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return output * torch.sigmoid(gate)


def _bench_us(fn) -> float:
    for _ in range(25):
        fn()
    torch.accelerator.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn)) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 16, 64, 256, 2048, 8192, 32768],
    )
    parser.add_argument("--width", type=int, default=1536)
    args = parser.parse_args()

    if not torch.accelerator.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA is required for QSA output-gate timing")

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    width = args.width
    rows = []
    for num_tokens in args.num_tokens:
        output = torch.randn(num_tokens, width, dtype=torch.bfloat16)
        gate = torch.randn_like(output)
        reference = output * torch.sigmoid(gate)

        eager = partial(_eager_gate, output, gate)
        compiled = partial(_compiled_gate, output, gate)
        fused = partial(qsa_output_gate, output, gate)

        _compiled_gate(output, gate)
        triton_output = fused()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            triton_output,
            reference,
            rtol=0,
            atol=2**-8,
        )

        eager_us = _bench_us(eager)
        compiled_us = _bench_us(compiled)
        fused_us = _bench_us(fused)
        theoretical_bytes = 3 * num_tokens * width * output.element_size()
        rows.append(
            {
                "tokens": num_tokens,
                "elements": num_tokens * width,
                "eager_us": eager_us,
                "compiled_us": compiled_us,
                "triton_us": fused_us,
                "vs_eager": eager_us / fused_us,
                "vs_compiled": compiled_us / fused_us,
                "triton_gbps": theoretical_bytes / (fused_us * 1e3),
            }
        )

    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "width": width,
    }
    print(pd.Series(metadata, name="value").to_string())
    print(
        pd.DataFrame(rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


if __name__ == "__main__":
    main()
