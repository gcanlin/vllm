# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.models.common.ops import (
    fused_q_kv_rmsnorm,
    fused_q_kv_rmsnorm_fp8_quant,
)
from vllm.triton_utils import triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=[1, 8, 32, 128, 256]
    )
    parser.add_argument("--rep", type=int, default=1000)
    return parser.parse_args()


def benchmark(num_tokens: int, rep: int) -> tuple[float, float]:
    q_size, kv_size = 1024, 512
    qr = torch.randn(num_tokens, q_size, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(num_tokens, kv_size, dtype=torch.bfloat16, device="cuda")
    q_weight = torch.randn(q_size, dtype=torch.bfloat16, device="cuda")
    kv_weight = torch.randn(kv_size, dtype=torch.bfloat16, device="cuda")

    def unfused():
        qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps=1e-6)
        return (
            *per_token_group_quant_fp8_packed_for_deepgemm(
                qr_out, group_size=128, use_ue8m0=True
            ),
            kv_out,
        )

    def fused():
        return fused_q_kv_rmsnorm_fp8_quant(qr, kv, q_weight, kv_weight, eps=1e-6)

    q_ref, scale_ref, kv_ref = unfused()
    q, scale, kv_out = fused()
    torch.testing.assert_close(q.float(), q_ref.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
    torch.testing.assert_close(kv_out, kv_ref, rtol=0, atol=0)

    unfused_ms = triton.testing.do_bench_cudagraph(unfused, rep=rep)
    fused_ms = triton.testing.do_bench_cudagraph(fused, rep=rep)
    return unfused_ms * 1000, fused_ms * 1000


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    print("tokens  unfused_us  fused_us  speedup  saved_us")
    for tokens in args.num_tokens:
        unfused_us, fused_us = benchmark(tokens, args.rep)
        print(
            f"{tokens:>6}  {unfused_us:>10.3f}  {fused_us:>8.3f}  "
            f"{unfused_us / fused_us:>7.3f}  {unfused_us - fused_us:>8.3f}"
        )
