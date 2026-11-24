# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred RecoverSSM verify for KDA layers with an FP32 state (SM10x CuTe).

The RecoverSSM verify of a speculative step reads each request's checkpoint
without writing it, and the commit after sampling reads it again to fold the
accepted tokens in: two reads and one write of the state per step. Deferred, the
commit only records each request's accepted count (`pending`, per request
slot, whose verify records stay in the same slot); the next verify replays those
tokens into the checkpoint from the records, writes it back and verifies the new
window from it: one read and one write of the state per step.

One kernel runs a layer's verify step: the short convolution, the q / k
normalization, the F_B gate projection and decays, the replay, the chunked verify
and the gated RMSNorm (`fused_tc.py`: persistent, the dot products and the gate on
the tensor cores; where it does not fit shared memory, `fused_ws.py`, persistent
and warp specialized, for large steps, else `fused.py`, one item per CTA or
cluster). `fused_ws` and `fused` are copied from the Kimi-K3 specialized decode.
"""

from collections.abc import Callable
from functools import partial

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from . import fused_tc, fused_ws
from .fused import KdaFused
from .fused_tc import KdaFusedTC
from .fused_ws import KdaFusedWS
from .host import fake, layout

K = 128
# (request, head) items from which KdaFusedWS takes the step where KdaFusedTC does not
# fit (B200, 64 heads, q 6, 4 replayed, ms per step KdaFused / KdaFusedWS: 512 items
# 1.43 / 1.53, 768 items 2.19 / 1.98, 1024 items 2.92 / 2.28).
WS_MIN_ITEMS = 768
# CTAs per item of KdaFused by item count: clusters fill the GPU at small steps.
FUSED_SPLIT = ((48, 8), (96, 4), (192, 2), (1 << 30, 1))


def decode_conv_weight(weight: torch.Tensor) -> torch.Tensor:
    """The [3, width, H * K] (q, k, v) layout of a [3 * H * K, 1, width] conv
    weight that the kernels read."""
    width = weight.shape[-1]
    return weight.detach().view(3, -1, width).transpose(1, 2).contiguous()


def _aligned_nd(tensor: torch.Tensor, align: int) -> bool:
    return (
        tensor.data_ptr() % align == 0
        and all((s * tensor.element_size()) % align == 0 for s in tensor.stride()[:-1])
        and tensor.stride(-1) == 1
    )


def _kernel(
    heads, max_requests, requests, tokens, conv_state, qkv,
    f_a, gate, state, f_b_weight, correction, key_gate,
):  # fmt: skip
    """KdaFusedTC where its operands stream in (16-byte aligned, contiguous
    state heads/rows, a conv window of at least q + 2) and fit shared memory
    at `max_requests` requests (any batch up to that), else KdaFusedWS from
    WS_MIN_ITEMS items, else KdaFused (raising if even KdaFused's vector
    alignments are violated)."""
    if not _aligned_nd(state, 16):
        raise ValueError(
            "KdaDeferredVerify: state requires 16-byte aligned base, slot/head/row "
            f"strides, and stride(-1) == 1 (got stride={state.stride()})"
        )
    if not _aligned_nd(f_a, 8) or not _aligned_nd(f_b_weight, 8):
        raise ValueError(
            "KdaDeferredVerify: f_a and f_b_weight require 8-byte alignment and "
            "contiguous last dimension"
        )
    items = heads * requests
    shape, stride = conv_state.shape, conv_state.stride()
    window = shape[-1]
    dim_first = stride[-1] == 1
    size = conv_state.element_size()
    conv_aligned = (
        conv_state.data_ptr() % 16 == 0
        and (stride[0] * size) % 16 == 0
        and (
            stride[1:] == (window, 1)
            if dim_first
            else stride[1] == 1 and (stride[2] * size) % 16 == 0
        )
    )
    aligned_16 = (
        conv_aligned
        and state.stride()[2:] == (K, 1)
        and all(
            _aligned_nd(t, 16)
            for t in (qkv, f_a, gate, f_b_weight, correction, key_gate)
        )
    )
    streams = aligned_16 and window >= tokens + 2 and tokens >= 3
    sms = torch.cuda.get_device_properties(qkv.device).multi_processor_count
    tc_items = fused_tc.cta_items(heads, max_requests, sms)
    tc_smem = fused_tc.shared_bytes(tokens, tokens, dim_first, window, size, tc_items)
    if (
        streams
        and state.stride(1) == K * K
        and window == tokens + 2
        and 2 * tokens <= fused_tc.PROBES
        and tc_smem <= fused_tc.SMEM_LIMIT
    ):
        return KdaFusedTC(heads, tokens, 0, False, dim_first, window, sms, max_requests)
    ctas = -(-items // max(2, -(-items // sms)))
    if streams and items >= WS_MIN_ITEMS:
        smem = partial(fused_ws.shared_bytes, tokens, tokens, dim_first, window, size)
        stage_fb = smem(True) <= fused_ws.SMEM_LIMIT
        if stage_fb or smem() <= fused_ws.SMEM_LIMIT:
            return KdaFusedWS(
                heads, tokens, 0, False, dim_first, window, ctas, stage_fb
            )
    split = next(s for limit, s in FUSED_SPLIT if items < limit)
    return KdaFused(heads, tokens, 0, False, split)


# Operands whose leading dimension is total tokens (dynamic for all kernels).
_TOKEN_OPERANDS = frozenset((0, 4, 5, 10, 11))
# The operands whose leading dimension follows the batch (or the cache's slots).
_BATCH_OPERANDS = frozenset((0, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19))
# Operands whose leading row stride is read dynamically by KdaFusedTC.
_STRIDE_OPERANDS = frozenset((0, 3, 4, 5, 8, 9, 10, 11))
# Operands accessed only by scalar loads/stores (need only element alignment).
_SCALAR_OPERANDS = frozenset((1, 2, 5, 6, 7, 8, 10, 12, 14, 17, 18, 19))
# Compiled kernels by layer shape and operand layouts, shared by every layer.
_COMPILED: dict[tuple, Callable[..., None]] = {}


def _operand_align(kernel: object, idx: int, elem_size: int) -> int:
    if isinstance(kernel, (KdaFusedTC, KdaFusedWS)):
        return elem_size if idx in _SCALAR_OPERANDS else 16
    return 16 if idx == 9 else 8 if idx in (4, 13) else elem_size


class KdaDeferredVerify:
    """The deferred RecoverSSM verify of one KDA layer shape (heads, window) for
    steps of up to `max_requests` requests. KdaFusedTC compiles once per operand
    layout with the batch and leading row strides read at run time; the
    fallbacks compile per batch. The layers of one shape share the compiled
    kernels."""

    def __init__(self, num_heads: int, window: int, max_requests: int) -> None:
        self.num_heads, self.window = num_heads, window
        self.max_requests = max_requests

    def __call__(
        self,
        *,
        qkv: torch.Tensor,
        gate: torch.Tensor,
        f_a: torch.Tensor,
        beta: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        f_b_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        lower_bound: float,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        correction: torch.Tensor,
        key_gate: torch.Tensor,
        record_rows: torch.Tensor,
        pending: torch.Tensor,
        query_start_loc: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """One verify step of `B = state_indices.shape[0]` requests of up to `q`
        tokens each (`query_start_loc` [B + 1]). `qkv` [T, 3 H K] is the
        projection before the conv, `gate` [T, H K] the output gate, `f_a`
        [T, K], `beta` [T, H] (raw), `conv_weight` [3, width, H K]
        (`decode_conv_weight`), `conv_state` [slots, 3 H K, window] (a
        transposed view for the state-len-first layout), `f_b_weight` [H K, K],
        `norm_weight` [K] FP32, `state` [slots, H, K, K] FP32 and
        `state_indices` [B, n] (column 0: the checkpoint slot). `correction` /
        `key_gate` are the verify records by record slot, `record_rows` [B] each
        request's record slot and `pending` [record slots] the accepted tokens
        of its last verify not yet in its checkpoint. Writes `out` [T, H K], the
        replayed checkpoints, the rolled conv window and the new records."""
        requests = state_indices.shape[0]
        operands = (
            qkv, conv_weight, dt_bias, conv_state, f_a, beta, A_log, dt_bias,
            state_indices, state, out, gate, norm_weight,
            f_b_weight, num_accepted_tokens[:requests], correction, key_gate,
            record_rows[:requests], pending, query_start_loc[: requests + 1],
        )  # fmt: skip
        scalars = tuple(map(cutlass.Float32, (lower_bound, K**-0.5, norm_eps)))
        assert requests <= self.max_requests
        kernel = _kernel(
            self.num_heads, self.max_requests, requests, self.window, conv_state, qkv,
            f_a, gate, state, f_b_weight, correction, key_gate,
        )  # fmt: skip
        specs = list(map(layout, operands))
        if isinstance(kernel, KdaFusedTC):
            for i in _BATCH_OPERANDS:
                shape, stride, dtype = specs[i]
                if i in _STRIDE_OPERANDS:
                    stride = (None, *stride[1:])
                shape = (None, None) if i == 8 else (None, *shape[1:])
                specs[i] = (shape, stride, dtype)
        else:
            for i in _TOKEN_OPERANDS:
                shape, stride, dtype = specs[i]
                specs[i] = ((None, *shape[1:]), stride, dtype)
        # TC reads the batch at run time: q and max_requests are not in its layouts.
        key = (self.num_heads, self.window, self.max_requests, type(kernel), *specs)
        compiled = _COMPILED.get(key)
        if compiled is None:
            fakes = []
            for idx, (shape, stride, dtype) in enumerate(specs):
                elem_size = torch.empty((), dtype=dtype).element_size()
                align = _operand_align(kernel, idx, elem_size)
                shape = tuple(cute.sym_int() if s is None else s for s in shape)
                if stride[0] is None:
                    div = 1 if align == elem_size else align // elem_size
                    stride = (cute.sym_int(64, divisibility=div), *stride[1:])
                fakes.append(fake(shape, stride, dtype, align=align))
            with torch.accelerator.device_index(qkv.device.index):
                compiled = cute.compile(
                    kernel,
                    *fakes[:13],
                    *scalars,
                    *fakes[13:],
                    make_fake_stream(use_tvm_ffi_env_stream=True),
                    options="--enable-tvm-ffi",
                )
            _COMPILED[key] = compiled
        # TVM-FFI launch on the current stream, straight from the torch tensors.
        compiled(*operands[:13], lower_bound, K**-0.5, norm_eps, *operands[13:])
        return out
