# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense tiny-M candidates; preserve MXFP8 activation rounding and output dtype."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def sf_offset(row, block_k, K: tl.constexpr):
    return (
        ((row // 128 * tl.cdiv(K, 128) + block_k // 4) * 32 + row % 32) * 16
        + row % 128 // 32 * 4
        + block_k % 4
    )


@triton.jit
def decode_scale(sf):
    # E8M0 exponents map directly to FP32 exponent bits except code zero,
    # which represents the subnormal value 2**-127.
    return tl.where(
        sf == 0,
        5.877471754111438e-39,
        (sf.to(tl.uint32) << 23).to(tl.float32, bitcast=True),
    )


@triton.jit
def round_mx(a, ROWS: tl.constexpr, BK: tl.constexpr):
    grouped = tl.reshape(a, (ROWS, BK // 32, 32))
    normalized = tl.max(tl.abs(grouped), 2) * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    inv = tl.where(sf == 0, 0, (254 - sf) << 23).to(tl.float32, bitcast=True)
    aq = (grouped * inv[:, :, None]).to(tl.float8e4nv).to(tl.float32)
    return tl.reshape(aq * decode_scale(sf)[:, :, None], (ROWS, BK))


@triton.jit
def _simt(
    A,
    W,
    SW,
    SA,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    MX: tl.constexpr,
    PRE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ni = tl.program_id(0) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    w = tl.load(
        W + ni[:, None] * K + ki[None, :], (ni[:, None] < N) & (ki[None, :] < K), 0.0
    ).to(tl.float32)
    if MX:
        sw = tl.load(
            SW + sf_offset(ni[:, None], ki[None, :] // 32, K),
            (ni[:, None] < N) & (ki[None, :] < K),
            127,
        ).to(tl.int32)
        w = w * decode_scale(sw)
    for r in tl.static_range(BM):
        mi = tl.program_id(1) * BM + r
        a = tl.load(A + mi * AS + ki, (mi < M) & (ki < K), 0.0).to(tl.float32)
        if MX:
            if PRE:
                sa = tl.load(
                    SA + sf_offset(mi, ki // 32, K), (mi < M) & (ki < K), 127
                ).to(tl.int32)
                a = a * decode_scale(sa)
            else:
                a = tl.reshape(round_mx(tl.reshape(a, (1, BK)), 1, BK), (BK,))
        y = tl.sum(w * a[None, :], 1)
        tl.store(Y + mi * N + ni, y, (mi < M) & (ni < N))


def simt(x, w, sw=None, sx=None, bn=4, bm=1, warps=4, out_fp32=False):
    m, k = x.shape
    n = w.shape[0]
    assert x.stride(1) == 1 and w.is_contiguous()
    y = torch.empty(
        (m, n), device=x.device, dtype=torch.float32 if out_fp32 else torch.bfloat16
    )
    _simt[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
        x,
        w,
        sw,
        sx,
        y,
        m,
        n,
        k,
        x.stride(0),
        sw is not None,
        sx is not None,
        bm,
        bn,
        triton.next_power_of_2(k),
        num_warps=warps,
    )
    return y


@triton.jit
def _tc(
    A,
    W,
    SW,
    SA,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    MX: tl.constexpr,
    PRE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    mi = tl.program_id(0) * 16 + tl.arange(0, 16)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    sk = tl.program_id(2)
    acc = tl.zeros((16, BN), tl.float32)
    for t in range(tl.cdiv(K, BK * SK)):
        ki = (t * SK + sk) * BK + tl.arange(0, BK)
        a = tl.load(
            A + mi[:, None] * AS + ki[None, :],
            (mi[:, None] < M) & (ki[None, :] < K),
            0.0,
        ).to(tl.float32)
        w = tl.load(
            W + ni[:, None] * K + ki[None, :],
            (ni[:, None] < N) & (ki[None, :] < K),
            0.0,
        ).to(tl.float32)
        if MX:
            sw = tl.load(
                SW + sf_offset(ni[:, None], ki[None, :] // 32, K),
                (ni[:, None] < N) & (ki[None, :] < K),
                127,
            ).to(tl.int32)
            w = w * decode_scale(sw)
            if PRE:
                sa = tl.load(
                    SA + sf_offset(mi[:, None], ki[None, :] // 32, K),
                    (mi[:, None] < M) & (ki[None, :] < K),
                    127,
                ).to(tl.int32)
                a = a * decode_scale(sa)
            else:
                a = round_mx(a, 16, BK)
        acc = tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), acc)
    tl.store(
        Y + sk * M * N + mi[:, None] * N + ni[None, :],
        acc,
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@triton.jit
def _reduce(P, Y, SIZE: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for s in range(SK):
        acc += tl.load(P + s * SIZE + i, i < SIZE, 0)
    tl.store(Y + i, acc, i < SIZE)


def tc(x, w, sw=None, sx=None, bn=32, bk=128, sk=1, warps=4, out_fp32=False):
    m, k = x.shape
    n = w.shape[0]
    assert x.stride(1) == 1 and w.is_contiguous()
    y = torch.empty(
        (m, n), device=x.device, dtype=torch.float32 if out_fp32 else torch.bfloat16
    )
    partial = (
        y if sk == 1 else torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    )
    _tc[(triton.cdiv(m, 16), triton.cdiv(n, bn), sk)](
        x,
        w,
        sw,
        sx,
        partial,
        m,
        n,
        k,
        x.stride(0),
        sw is not None,
        sx is not None,
        bn,
        bk,
        sk,
        num_warps=warps,
        num_stages=2,
    )
    if sk > 1:
        _reduce[(triton.cdiv(m * n, 256),)](partial, y, m * n, sk, 256)
    return y
