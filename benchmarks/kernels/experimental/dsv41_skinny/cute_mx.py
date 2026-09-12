# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exploratory 32-element-scale adaptation of vLLM PR #43214's CuTe kernel.

Use a 128-FP8-element K tile: four consumer warps each process one 32-element
scale group, so the four scales still fit in one packed word per pipeline stage.
Weights must retain the checkpoint's shared scale across each 32 output rows.
"""

import torch
from kernels import sf_offset

from vllm.triton_utils import tl, triton


@triton.jit
def _pack(S, P, M: tl.constexpr, K: tl.constexpr, PS: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    si = tl.arange(0, BK // 32)
    scales = tl.load(S + sf_offset(row, si, K), si < K // 32, 0).to(tl.uint32)
    packed = tl.sum(
        tl.reshape(scales, (BK // 128, 4)) << (tl.arange(0, 4)[None, :] * 8), 1
    )
    ki = tl.arange(0, BK // 128)
    tl.store(P + ki * PS + row, packed, ki < K // 128)


@triton.jit
def _quant(
    A,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    SS: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    ki = tl.arange(0, BK)
    a = tl.load(A + row * AS + ki, ki < K, 0.0).to(tl.float32)
    grouped = tl.reshape(a, (BK // 32, 32))
    normalized = tl.max(tl.abs(grouped), 1) * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    inv = tl.where(sf == 0, 0, (254 - sf) << 23).to(tl.float32, bitcast=True)
    quant = tl.reshape(grouped * inv[:, None], (BK,)).to(tl.float8e4nv)
    tl.store(Q + row * K + ki, quant, ki < K)
    packed = tl.sum(tl.reshape(sf, (BK // 128, 4)) << (tl.arange(0, 4)[None, :] * 8), 1)
    pi = tl.arange(0, BK // 128)
    tl.store(S + pi * SS + row, packed, pi < K // 128)


def pack_scale(scale, rows, k):
    assert k % 128 == 0
    out = torch.empty(
        (k // 128, triton.cdiv(rows, 4) * 4), device=scale.device, dtype=torch.int32
    ).T[:rows]
    _pack[(rows,)](
        scale, out, rows, k, out.stride(1), triton.next_power_of_2(k), num_warps=4
    )
    return out


def quantize(x):
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(
        (k // 128, triton.cdiv(m, 4) * 4), device=x.device, dtype=torch.int32
    ).T[:m]
    _quant[(m,)](
        x, q, s, m, k, x.stride(0), s.stride(1), triton.next_power_of_2(k), num_warps=4
    )
    return q, s


COMPILED = {}


def gemm(x, w, packed_sw, sx=None, tile_n=16, stages=2, dma=4):
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cute_mx_kernel import LLFp8BlockGemm
    from cutlass import BFloat16, Int32

    from vllm.utils.torch_utils import current_stream

    m, k = x.shape
    n = w.shape[0]
    assert 1 <= m <= 16 and k % 128 == 0 and n % 32 == 0
    assert packed_sw.shape == (n, k // 128) and packed_sw.dtype == torch.int32
    a, sa = (x, pack_scale(sx, m, k)) if sx is not None else quantize(x)
    key = (tile_n, stages, dma)
    if key not in COMPILED:
        ms = cute.sym_int()
        ns = cute.sym_int(divisibility=8)
        ks = cute.sym_int(divisibility=64)
        ss = cute.sym_int()
        stride = cute.sym_int64(divisibility=4)
        fa = cute.runtime.make_fake_tensor(
            BFloat16, (ms, ks), stride=(ks, 1), assumed_align=16
        )
        fw = cute.runtime.make_fake_tensor(
            BFloat16, (ns, ks), stride=(ks, 1), assumed_align=16
        )
        fy = cute.runtime.make_fake_tensor(
            BFloat16, (ms, ns), stride=(ns, 1), assumed_align=16
        )
        fsa = cute.runtime.make_fake_tensor(
            Int32, (ms, ss), stride=(1, stride), assumed_align=4
        )
        fsw = cute.runtime.make_fake_tensor(
            Int32, (ns, ss), stride=(1, ns), assumed_align=4
        )
        kernel = LLFp8BlockGemm(
            tile_n=tile_n,
            tile_k=64,
            num_stages=stages,
            num_dma_warps=dma,
            use_pdl=True,
            has_k_tail=False,
        )
        COMPILED[key] = cute.compile(
            kernel,
            fa,
            fw,
            fy,
            fsa,
            fsw,
            CUstream(current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    COMPILED[key](
        a.view(torch.bfloat16),
        w.view(torch.bfloat16),
        y,
        sa,
        packed_sw,
        CUstream(current_stream().cuda_stream),
    )
    return y
