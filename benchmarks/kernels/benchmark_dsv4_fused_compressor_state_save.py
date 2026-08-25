# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepSeek-V4 batch-1 inline compressor state persistence."""

import argparse
from collections.abc import Callable
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.common.ops import save_partial_states
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.triton_utils import triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=200)
    return parser.parse_args()


def make_ops(
    path: str, boundary: bool
) -> tuple[Callable[[], None], Callable[[], None]]:
    head_dim = 128 if path == "indexer" else 512
    compress_ratio = 128 if path == "c128" else 4
    overlap = compress_ratio == 4
    state_width = (1 + overlap) * head_dim
    state_block_size = 8 if path == "c128" else 4
    position_value = compress_ratio - (1 if boundary else 2)
    num_state_blocks = (compress_ratio + state_block_size - 1) // state_block_size

    position = torch.tensor([position_value], dtype=torch.int64, device="cuda")
    state_slot = position.clone()
    kv_slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    token_to_req = torch.zeros(1, dtype=torch.int32, device="cuda")
    block_table = torch.arange(
        num_state_blocks, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)
    base_state = torch.randn(
        num_state_blocks,
        state_block_size,
        2 * state_width,
        dtype=torch.float32,
        device="cuda",
    )
    old_state = base_state.clone()
    fused_state = base_state.clone()
    kv_score = torch.randn(1, 2 * state_width, dtype=torch.float32, device="cuda")
    kv, score = kv_score.split(state_width, dim=-1)
    ape = torch.randn(compress_ratio, state_width, dtype=torch.float32, device="cuda")
    rms_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    cos_sin_cache = torch.randn(256, 64, dtype=torch.float32, device="cuda")
    cache_width = 132 if path == "indexer" else 584
    old_cache = torch.zeros(1, 64, cache_width, dtype=torch.uint8, device="cuda")
    fused_cache = old_cache.clone()

    def store(state: torch.Tensor) -> None:
        save_partial_states(
            kv=kv,
            score=score,
            ape=ape,
            positions=position,
            state_cache=state,
            slot_mapping=state_slot,
            block_size=state_block_size,
            state_width=state_width,
            compress_ratio=compress_ratio,
        )

    if path == "indexer":

        def compress(
            state: torch.Tensor, cache: torch.Tensor, fuse_state_save: bool
        ) -> None:
            compress_norm_rope_store_triton(
                state_cache=state,
                num_actual=1,
                token_to_req_indices=token_to_req,
                positions=position,
                slot_mapping=state_slot,
                block_table=block_table,
                block_size=state_block_size,
                state_width=state_width,
                cos_sin_cache=cos_sin_cache,
                kv_cache=cache,
                k_cache_metadata=SimpleNamespace(slot_mapping=kv_slot),
                pdl_kwargs={},
                head_dim=head_dim,
                rope_head_dim=64,
                compress_ratio=compress_ratio,
                overlap=overlap,
                use_fp4_cache=False,
                rms_norm_weight=rms_weight,
                rms_norm_eps=1e-6,
                quant_block=128,
                token_stride=128,
                scale_dim=4,
                kv_score=kv_score,
                ape=ape,
                fuse_state_save=fuse_state_save,
            )

    else:
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            split_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
        )

        def compress(
            state: torch.Tensor, cache: torch.Tensor, fuse_state_save: bool
        ) -> None:
            common_args = {
                "head_size": head_dim,
                "state_width": state_width,
                "rope_head_dim": 64,
                "fp8_max": 448.0,
                "quant_block": 64,
                "token_stride": 576,
                "scale_dim": 8,
                "compress_ratio": compress_ratio,
                "overlap": overlap,
                "fuse_state_save": fuse_state_save,
            }
            compressed_kv = torch.empty(1, head_dim, dtype=torch.float32, device="cuda")
            split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
                state,
                kv_score,
                ape,
                token_to_req,
                position,
                state_slot,
                block_table,
                state_block_size,
                compressed_kv,
                rms_weight,
                1e-6,
                cos_sin_cache,
                cache,
                kv_slot,
                cache.shape[1],
                cache.stride(0),
                **common_args,
            )

    def old() -> None:
        store(old_state)
        compress(old_state, old_cache, False)

    def fused() -> None:
        compress(fused_state, fused_cache, True)

    old()
    fused()
    torch.accelerator.synchronize()
    torch.testing.assert_close(fused_state, old_state, rtol=0, atol=0)
    assert torch.equal(fused_cache, old_cache)
    return old, fused


def benchmark(path: str, boundary: bool, rep: int) -> tuple[float, float]:
    old, fused = make_ops(path, boundary)
    old_graph = torch.cuda.CUDAGraph()
    fused_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph):
        old()
    with torch.cuda.graph(fused_graph):
        fused()
    torch.accelerator.synchronize()
    old_ms = triton.testing.do_bench(old_graph.replay, warmup=25, rep=rep)
    fused_ms = triton.testing.do_bench(fused_graph.replay, warmup=25, rep=rep)
    return old_ms * 1000, fused_ms * 1000


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    print("path     phase         old_us  fused_us  speedup  saved_us")
    for kernel_path in ("indexer", "c128"):
        for is_boundary in (False, True):
            old_us, fused_us = benchmark(kernel_path, is_boundary, args.rep)
            phase = "boundary" if is_boundary else "nonboundary"
            print(
                f"{kernel_path:<8} {phase:<11} {old_us:8.3f} {fused_us:9.3f} "
                f"{old_us / fused_us:8.3f} {old_us - fused_us:9.3f}"
            )
