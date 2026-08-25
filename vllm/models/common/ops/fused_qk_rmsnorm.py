# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_fp8_ptr,
    q_scale_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    q_fp8_stride,
    q_scale_stride_k,
    q_scale_num_elems,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    num_tokens,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUANTIZE_Q: tl.constexpr,
    Q_QUANT_GROUP_SIZE: tl.constexpr,
    Q_NUM_GROUPS: tl.constexpr,
    Q_PACKED_GROUPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    # DeepGEMM's packed scale buffer has a four-row TMA alignment.  The
    # logical tensor excludes those rows, but the backing storage must be
    # initialized because the consumer can issue an aligned TMA load.
    if token_idx >= num_tokens:
        if QUANTIZE_Q and pid_task == 0:
            pack_offsets = tl.arange(0, Q_PACKED_GROUPS)
            scale_offsets = token_idx + pack_offsets * q_scale_stride_k
            tl.store(
                q_scale_ptr + scale_offsets,
                tl.zeros((Q_PACKED_GROUPS,), dtype=tl.int32),
                mask=scale_offsets < q_scale_num_elems,
            )
        return

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    if QUANTIZE_Q and pid_task == 0:
        # Match the unfused path exactly: RMSNorm first rounds to BF16, then
        # the standalone activation kernel reloads BF16 and computes one
        # dynamic UE8M0 scale per 128 values.
        y_bf16 = y.to(tl.bfloat16)
        y_groups = tl.reshape(y_bf16.to(tl.float32), (Q_NUM_GROUPS, Q_QUANT_GROUP_SIZE))
        absmax = tl.max(tl.abs(y_groups), axis=1)
        scale_raw = tl.maximum(absmax * (1.0 / FP8_MAX), 1.0e-10)

        # ceil(log2(scale_raw)) via exponent/mantissa bit math.  This is the
        # same construction used by per_token_group_quant_8bit_packed and
        # avoids libdevice differences around exact powers of two.
        scale_bits = scale_raw.to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
            tl.uint32
        )
        rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

        scaled = y_groups * (1.0 / rounded_scale)[:, None]
        q = tl.reshape(
            tl.clamp(scaled, -FP8_MAX, FP8_MAX).to(tl.float8e4nv),
            (Q_SIZE,),
        )
        tl.store(q_fp8_ptr + token_idx * q_fp8_stride + block, q)

        scale_exp = tl.reshape(scale_exp, (Q_PACKED_GROUPS, 4))
        byte_offsets = tl.arange(0, 4)
        packed_scale = tl.sum(scale_exp << (byte_offsets[None, :] * 8), axis=1)
        pack_offsets = tl.arange(0, Q_PACKED_GROUPS)
        tl.store(
            q_scale_ptr + token_idx + pack_offsets * q_scale_stride_k,
            packed_scale.to(tl.int32),
        )
    else:
        tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
        qr,
        qr_out,
        qr_out,
        qr_out,
        q_weight,
        qr.stride(0),
        qr_out.stride(0),
        0,
        0,
        0,
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        num_tokens,
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        BLOCK_SIZE=block_size,
        QUANTIZE_Q=False,
        Q_QUANT_GROUP_SIZE=1,
        Q_NUM_GROUPS=1,
        Q_PACKED_GROUPS=1,
        FP8_MAX=1.0,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return qr_out, kv_out


def fused_q_kv_rmsnorm_fp8_quant(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    quant_group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q/KV RMSNorm with DeepGEMM-compatible Q activation quantization.

    Unlike :func:`fused_q_kv_rmsnorm`, this fast path does not materialize the
    normalized Q tensor.  It rounds Q to BF16 in registers, then emits FP8 Q
    and packed UE8M0 scales in the MN-major, four-row-aligned layout consumed
    by DeepGEMM.  KV is returned in the original dtype as before.

    Returns ``(q_fp8, q_scale_packed, kv_out)``.  ``q_scale_packed`` has
    logical shape ``[num_tokens, num_q_groups / 4]``, dtype INT32, and stride
    ``(1, align_up(num_tokens, 4))``.
    """
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert qr.shape[1] % quant_group_size == 0

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    block_size = triton.next_power_of_2(max(q_size, kv_size))
    # The fused program reshapes its Q register tile into 128-value groups;
    # keeping Q equal to the program width avoids padded values entering the
    # reduction.  DeepSeek-V4-Flash uses Q_SIZE=1024.
    assert q_size == block_size, (
        "fused Q quantization requires q_size to equal the RMSNorm block size, "
        f"got q_size={q_size}, block_size={block_size}"
    )
    num_groups = q_size // quant_group_size
    assert num_groups % 4 == 0, (
        f"packed UE8M0 requires a multiple of four Q groups, got {num_groups}"
    )

    num_tokens = qr.shape[0]
    packed_groups = num_groups // 4
    tma_aligned_tokens = ((num_tokens + 3) // 4) * 4
    q_fp8 = torch.empty_like(qr, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty_strided(
        (num_tokens, packed_groups),
        (1, tma_aligned_tokens),
        dtype=torch.int32,
        device=qr.device,
    )
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return q_fp8, q_scale, kv_out

    _fused_q_kv_rmsnorm_kernel[(tma_aligned_tokens, 2)](
        qr,
        kv_out,  # dead q_out pointer when QUANTIZE_Q=True
        q_fp8,
        q_scale,
        q_weight,
        qr.stride(0),
        0,
        q_fp8.stride(0),
        q_scale.stride(1),
        num_tokens + (packed_groups - 1) * tma_aligned_tokens,
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        num_tokens,
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        BLOCK_SIZE=block_size,
        QUANTIZE_Q=True,
        Q_QUANT_GROUP_SIZE=quant_group_size,
        Q_NUM_GROUPS=num_groups,
        Q_PACKED_GROUPS=packed_groups,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return q_fp8, q_scale, kv_out
