# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

_M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192)


def _reference(
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


@pytest.mark.parametrize("num_tokens", _M_VALUES)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 routed-down MXFP8 requires SM100",
)
def test_routed_down_mxfp8_matches_flashinfer(num_tokens: int) -> None:
    """Guard scale bytes and FP8 values across the dispatch token range."""
    from vllm.models.kimi_k3.nvidia.ops.cute_dsl.routed_down_mxfp8 import (
        K_DIM,
        N_DIM,
        routed_down_mxfp8,
    )

    generator = torch.Generator(device="cuda").manual_seed(1000 + num_tokens)
    x = torch.randn(
        (num_tokens, K_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        (N_DIM, K_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    expected_q, expected_scale = _reference(x, weight)
    actual_q, actual_scale = routed_down_mxfp8(x, weight)
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    mismatch_rate = (
        (actual_q.view(torch.uint8) != expected_q.view(torch.uint8))
        .float()
        .mean()
        .item()
    )
    assert mismatch_rate <= 1e-3
    torch.testing.assert_close(
        actual_q.float(),
        expected_q.float(),
        rtol=0,
        atol=32,
        equal_nan=True,
    )


@pytest.mark.parametrize("value", (0.0, 64.0, float("nan"), float("inf")))
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 routed-down MXFP8 requires SM100",
)
def test_routed_down_mxfp8_edge_values_are_bitwise(value: float) -> None:
    """Guard UE8M0 special-value handling in the fused epilogue."""
    from vllm.models.kimi_k3.nvidia.ops.cute_dsl.routed_down_mxfp8 import (
        K_DIM,
        N_DIM,
        routed_down_mxfp8,
    )

    x = torch.full((7, K_DIM), value, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((N_DIM, K_DIM), dtype=torch.bfloat16, device="cuda")
    expected_q, expected_scale = _reference(x, weight)
    actual_q, actual_scale = routed_down_mxfp8(x, weight)
    torch.accelerator.synchronize()

    assert torch.equal(actual_scale, expected_scale)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 routed-down MXFP8 requires SM100",
)
def test_routed_down_mxfp8_dispatch_range() -> None:
    """Use the fused path only over the measured profitable token range."""
    from vllm.models.kimi_k3.nvidia.ops.cute_dsl.routed_down_mxfp8 import (
        K_DIM,
        N_DIM,
        KimiK3RoutedDownMxfp8Op,
    )

    weight = torch.empty((N_DIM, K_DIM), dtype=torch.bfloat16, device="cuda")
    assert KimiK3RoutedDownMxfp8Op.can_run(
        torch.empty((512, K_DIM), dtype=torch.bfloat16, device="cuda"), weight
    )
    assert not KimiK3RoutedDownMxfp8Op.can_run(
        torch.empty((513, K_DIM), dtype=torch.bfloat16, device="cuda"), weight
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 routed-down MXFP8 requires SM100",
)
def test_routed_down_mxfp8_cuda_graph_replay() -> None:
    """Guard output allocation and persistent state across graph replays."""
    from vllm.models.kimi_k3.nvidia.ops.cute_dsl.routed_down_mxfp8 import (
        K_DIM,
        N_DIM,
        routed_down_mxfp8,
    )

    x = torch.randn((128, K_DIM), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((N_DIM, K_DIM), dtype=torch.bfloat16, device="cuda")
    q_out = torch.empty((128, N_DIM), dtype=torch.float8_e4m3fn, device="cuda")
    scale_out = torch.empty((128, N_DIM // 32), dtype=torch.uint8, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            q, scale = routed_down_mxfp8(x, weight)
            q_out.copy_(q)
            scale_out.copy_(scale)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        q, scale = routed_down_mxfp8(x, weight)
        q_out.copy_(q)
        scale_out.copy_(scale)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()

    expected_q, expected_scale = _reference(x, weight)
    assert torch.equal(scale_out, expected_scale)
    assert torch.equal(q_out.view(torch.uint8), expected_q.view(torch.uint8))
