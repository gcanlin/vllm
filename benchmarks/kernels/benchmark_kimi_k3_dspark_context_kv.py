# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Kimi-K3 DSpark context-KV output tail."""

import argparse
import statistics
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    return parser.parse_args()


def benchmark(op: Callable[[], None], warmup: int, samples: int) -> float:
    for _ in range(warmup):
        op()
    torch.accelerator.synchronize()

    timings = []
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        op()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    return statistics.median(timings)


@torch.inference_mode()
def benchmark_tokens(num_tokens: int, warmup: int, samples: int) -> tuple[float, float]:
    device = torch.device("cuda")
    num_layers = 5
    kv_lora_rank = 512
    rope_dim = 64
    width = kv_lora_rank + rope_dim
    block_size = 16
    num_blocks = max(4, (num_tokens + block_size - 1) // block_size)
    epsilon = 1e-6

    rope = RotaryEmbedding(
        rope_dim,
        rope_dim,
        8192,
        10000,
        False,
        torch.float32,
    ).to(device=device)
    kv = torch.randn(num_tokens, num_layers, width, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(
        num_layers, kv_lora_rank, dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(0, 8192, (num_tokens,), device=device)
    repeated_positions = torch.empty(
        num_layers * num_tokens, dtype=torch.int64, device=device
    )
    slot_mapping = torch.arange(num_tokens, device=device).expand(num_layers, -1)
    caches = [
        torch.zeros(
            num_blocks,
            block_size,
            width,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(num_layers)
    ]
    cache_ptrs = torch.tensor(
        [cache.data_ptr() for cache in caches], dtype=torch.int64, device=device
    )

    def unfused() -> None:
        kv_c = kv[..., :kv_lora_rank].permute(1, 0, 2).contiguous()
        kv_c_normed = torch.empty_like(kv_c)
        ops.rms_norm(kv_c_normed, kv_c, norm_weight, epsilon)
        k_pe = kv[..., kv_lora_rank:].permute(1, 0, 2).contiguous()
        k_pe_flat = k_pe.view(num_layers * num_tokens, 1, rope_dim)
        repeated_positions.view(num_layers, num_tokens).copy_(positions)
        ops.rotary_embedding(
            repeated_positions,
            k_pe_flat,
            None,
            rope.head_size,
            rope.cos_sin_cache,
            rope.is_neox_style,
        )
        ops.concat_and_cache_mla_grouped(
            kv_c_normed,
            k_pe,
            cache_ptrs,
            slot_mapping,
            block_size,
            caches[0].stride(0),
            caches[0].stride(1),
        )

    def fused() -> None:
        ops.rms_norm_rope_and_cache_mla_grouped(
            kv,
            norm_weight,
            positions,
            rope.cos_sin_cache,
            rope.is_neox_style,
            cache_ptrs,
            slot_mapping,
            block_size,
            caches[0].stride(0),
            caches[0].stride(1),
            epsilon,
        )

    unfused()
    expected = [cache.clone() for cache in caches]
    for cache in caches:
        cache.zero_()
    fused()
    for result, reference in zip(caches, expected):
        torch.testing.assert_close(result, reference)

    return benchmark(unfused, warmup, samples), benchmark(fused, warmup, samples)


def main() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config

    args = parse_args()
    print(f"device={torch.cuda.get_device_name()}")
    print("tokens  unfused_us  fused_us  speedup  saved_us")
    with set_current_vllm_config(VllmConfig()):
        for num_tokens in args.tokens:
            unfused_us, fused_us = benchmark_tokens(
                num_tokens, args.warmup, args.samples
            )
            print(
                f"{num_tokens:>6}  {unfused_us:>10.2f}  {fused_us:>8.2f}  "
                f"{unfused_us / fused_us:>7.2f}x  {unfused_us - fused_us:>8.2f}"
            )


if __name__ == "__main__":
    main()
