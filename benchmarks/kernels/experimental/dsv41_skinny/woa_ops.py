# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP8 single-group wo_a experiment; exact existing inverse-RoPE quant math."""

import torch
from kernels import sf_offset as _sf_offset

from vllm.triton_utils import tl, triton


@triton.jit
def _invrope(O_PTR, P, C, A, S, M, OS0, OS1, CS0, PDL: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, 512)
    blocks = head * 16 + tl.arange(0, 16)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    # Fill only padded scales, amortized across the eight head CTAs for row 0.
    # Valid rows are written by their own CTAs, with disjoint masked stores.
    if row == 0:
        padrow = tl.arange(0, 128)
        tl.store(
            S + _sf_offset(padrow[:, None], blocks[None, :], 4096),
            0,
            padrow[:, None] >= M,
        )
    if row < M:
        base = O_PTR + row * OS0.to(tl.int64) + head * OS1.to(tl.int64)
        x = tl.load(base + offsets).to(tl.float32)
        pos = tl.load(P + row)
        rope = offsets >= 448
        local = offsets - 448
        partner = tl.load(base + (offsets ^ 1), rope, 0.0).to(tl.float32)
        idx = tl.maximum(local >> 1, 0)
        cos = tl.load(C + pos * CS0.to(tl.int64) + idx, rope, 1.0)
        sin = tl.load(C + pos * CS0.to(tl.int64) + 32 + idx, rope, 0.0)
        plus = x * cos + partner * sin
        minus = x * cos - partner * sin
        x = tl.where(rope, tl.where((local & 1) == 0, plus, minus), x)
        amax = tl.maximum(tl.max(tl.reshape(tl.abs(x), (16, 32)), axis=1), 1e-10)
        raw = amax * (1.0 / 448.0)
        scales = tl.exp2(tl.ceil(tl.log2(raw)))
        expanded = tl.reshape(tl.broadcast_to(scales[:, None], (16, 32)), (512,))
        quant = tl.clamp(x / expanded, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(A + row * 4096 + head * 512 + offsets, quant)
        sf = (scales.to(tl.int32, bitcast=True) >> 23) & 255
    else:
        sf = tl.full((16,), 0, tl.int32)
    tl.store(S + _sf_offset(row, blocks, 4096), sf)


def invrope(o, positions, cache):
    m = o.shape[0]
    a = torch.empty((m, 4096), device=o.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(
        (triton.cdiv(m, 128) * 128 * 128,), device=o.device, dtype=torch.uint8
    )
    assert 0 < m <= 128
    _invrope[(m, 8)](
        o,
        positions,
        cache,
        a,
        s,
        m,
        o.stride(0),
        o.stride(1),
        cache.stride(0),
        True,
        num_warps=1,
    )
    return a, s
