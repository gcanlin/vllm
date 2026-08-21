# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM100 Kimi-K3 routed-down GEMM with a fused MXFP8 epilogue."""

# The persistent GEMM mainloop is based on the Kimi-K3 GEMM-RS kernel in this
# directory. The MXFP8 conversion follows FlashInfer's Apache-licensed CuTeDSL
# quantization implementation.

from functools import cache, partial

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
)
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import get_smem_capacity_in_bytes

from vllm.cute_utils import _tcgen05, mbarrier, simple_tma_copy
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.platforms import current_platform

K_DIM = 7168
N_DIM = 3584
MX_BLOCK = 32
MAX_FUSED_TOKENS = 512
_FP8_MAX = 448.0


@dsl_user_op
def _float_to_ue8m0(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;
                .reg .u32 bits, exp_biased, mantissa, bump, result;

                setp.le.f32 p_zero, $1, 0f00000000;
                mov.b32 bits, $1;
                shr.b32 exp_biased, bits, 23;
                and.b32 exp_biased, exp_biased, 255;
                and.b32 mantissa, bits, 0x7FFFFF;
                setp.ne.u32 p_has_mant, mantissa, 0;
                selp.u32 bump, 1, 0, p_has_mant;
                setp.eq.u32 p_exp_zero, exp_biased, 0;
                setp.le.u32 p_tiny_sub, mantissa, 0x400000;
                and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;
                @p_tiny_sub mov.u32 bump, 0;
                add.u32 result, exp_biased, bump;
                setp.gt.u32 p_ovf, result, 254;
                selp.u32 result, 254, result, p_ovf;
                selp.u32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ue8m0_to_inv_scale(value: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .s32 new_exp;
                .reg .b32 float_bits;
                .reg .pred p_zero;

                setp.eq.u32 p_zero, $1, 0;
                sub.s32 new_exp, 254, $1;
                max.s32 new_exp, new_exp, 0;
                shl.b32 float_bits, new_exp, 23;
                mov.b32 $0, float_bits;
                @p_zero mov.b32 $0, 0;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_f32x4_to_e4m3x4(src: cute.Tensor, *, loc=None, ip=None) -> Int32:
    values = src.load()
    vec = values.ir_value(loc=loc, ip=ip)
    args = [
        Float32(vector.extract(vec, [], [idx], loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)
        for idx in range(4)
    ]
    return Int32(
        llvm.inline_asm(
            T.i32(),
            args,
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
                cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _quantize_group(rounded: cute.Tensor, quantized: cute.Tensor) -> Uint32:
    values = rounded.load()
    abs_values = type(values)(
        cutlass._mlir.dialects.math.absf(values.ir_value()),
        values.shape,
        values.dtype,
    )
    amax = abs_values.reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
    scale = _float_to_ue8m0(amax * Float32(1.0 / _FP8_MAX))
    inv_scale = _ue8m0_to_inv_scale(scale)

    q_i32 = cute.recast_tensor(quantized, Int32)
    for offset in cutlass.range_constexpr(0, MX_BLOCK, 4):
        packed_src = cute.make_rmem_tensor(4, Float32)
        for idx in cutlass.range_constexpr(4):
            packed_src[idx] = rounded[offset + idx] * inv_scale
        q_i32[offset // 4] = _cvt_f32x4_to_e4m3x4(packed_src)
    return scale


class Sm100RoutedDownMxfp8:
    def __init__(self, block_n: int) -> None:
        block_m, block_k = 128, 64
        if block_n not in (32, 64, 128):
            raise ValueError(f"Unsupported block_n={block_n}.")
        self.cta_tile = (block_m, block_n, block_k)
        stage_size = (block_m + block_n) * block_k * 2
        self.num_stages = get_smem_capacity_in_bytes("sm_100") // stage_size

    @cute.jit
    def prepare_tma(
        self,
        tensor: cute.Tensor,
        tile_m: cutlass.Constexpr,
        tile_k: cutlass.Constexpr,
    ) -> cpasync.TmaInfo:
        op = cpasync.CopyBulkTensorTileG2SOp(cta_group=tcgen05.CtaGroup.ONE)
        swizzle = cute.make_swizzle(3, 4, 3)
        layout = cute.make_layout(
            (tile_m, tile_k, self.num_stages),
            stride=(tile_k, 1, tile_m * tile_k),
        )
        layout = cute.make_composed_layout(swizzle, 0, layout)
        return cpasync.make_tiled_tma_atom(op, tensor, layout, (tile_m, tile_k))

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        q: cute.Tensor,
        scales: cute.Tensor,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        block_m, block_n, block_k = self.cta_tile
        a_tma = self.prepare_tma(a, block_m, block_k)
        b_tma = self.prepare_tma(b, block_n, block_k)
        self.kernel(a_tma, b_tma, q, scales).launch(
            grid=(grid_size, 1, 1),
            block=(10 * 32, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_tma: cpasync.TmaInfo,
        b_tma: cpasync.TmaInfo,
        q: cute.Tensor,
        scales: cute.Tensor,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        block_m, block_n, block_k = self.cta_tile
        num_tmem_stages = 512 // block_n

        smem = cutlass.utils.SmemAllocator()
        s_a = smem.allocate_tensor(
            BFloat16,
            a_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=a_tma.smem_layout.inner,
        )
        s_b = smem.allocate_tensor(
            BFloat16,
            b_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=b_tma.smem_layout.inner,
        )
        tma_full = smem.allocate_array(Int64, self.num_stages)
        tma_empty = smem.allocate_array(Int64, self.num_stages)
        tmem_full = smem.allocate_array(Int64, num_tmem_stages)
        tmem_empty = smem.allocate_array(Int64, num_tmem_stages)
        taddr = smem.allocate(Int32, 4)

        tmem_alloc_barrier = 1
        epilogue_barrier = 2

        m, k = a_tma.tma_tensor.shape
        n, _ = b_tma.tma_tensor.shape
        grid_m = cute.ceil_div(m, block_m)
        grid_n = cute.ceil_div(n, block_n)
        total_tiles = grid_m * grid_n

        if warp_id == 0:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.num_stages):
                    cute.arch.mbarrier_init(tma_full + stage, 1)
                    cute.arch.mbarrier_init(tma_empty + stage, 1)
                for stage in cutlass.range_constexpr(num_tmem_stages):
                    cute.arch.mbarrier_init(tmem_full + stage, 1)
                    cute.arch.mbarrier_init(tmem_empty + stage, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(a_tma.atom)
            cpasync.prefetch_descriptor(b_tma.atom)
        cute.arch.sync_threads()

        if warp_id == 9:
            tma_stage = 0
            parity = 1
            g_a_tiles = cute.zipped_divide(a_tma.tma_tensor, (block_m, block_k))
            g_b_tiles = cute.zipped_divide(b_tma.tma_tensor, (block_n, block_k))

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m = bid % grid_m
                bid_n = bid // grid_m
                for iter_k in cutlass.range(cute.ceil_div(k, block_k), unroll=1):
                    full_barrier = tma_full + tma_stage
                    cute.arch.mbarrier_wait(tma_empty + tma_stage, parity)
                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(
                            full_barrier,
                            (block_m + block_n) * block_k * 2,
                            "cluster",
                        )
                    simple_tma_copy(
                        a_tma.atom,
                        g_a_tiles[None, (bid_m, iter_k)],
                        s_a[None, None, tma_stage],
                        full_barrier,
                    )
                    simple_tma_copy(
                        b_tma.atom,
                        g_b_tiles[None, (bid_n, iter_k)],
                        s_b[None, None, tma_stage],
                        full_barrier,
                    )
                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        parity ^= 1

        elif warp_id == 8:
            cute.arch.barrier(barrier_id=tmem_alloc_barrier, number_of_threads=5 * 32)
            tma_stage = 0
            tma_parity = 0
            tmem_stage = 0
            tmem_parity = 1
            instruction_desc = _tcgen05.make_bf16_idesc(block_m, block_n)
            shared_desc = _tcgen05.make_sdesc_128B_swizzle(0)
            multicast_mask = Uint16(1)

            for _bid in range(raw_bid, total_tiles, num_bids):
                cute.arch.mbarrier_wait(tmem_empty + tmem_stage, tmem_parity)
                _tcgen05.fence_after_thread_sync()
                for iter_k in cutlass.range(cute.ceil_div(k, block_k), unroll=1):
                    d_tmem = block_n * tmem_stage
                    a_addr = s_a[None, None, tma_stage].iterator.toint()
                    b_addr = s_b[None, None, tma_stage].iterator.toint()
                    a_desc = shared_desc | (a_addr >> 4)
                    b_desc = shared_desc | (b_addr >> 4)
                    cute.arch.mbarrier_wait(tma_full + tma_stage, tma_parity)
                    _tcgen05.fence_after_thread_sync()

                    for mma_k in cutlass.range_constexpr(block_k // 16):
                        _tcgen05.mma_f16(
                            d_tmem,
                            a_desc,
                            b_desc,
                            instruction_desc,
                            iter_k > 0 or mma_k > 0,
                            1,
                        )
                        a_desc += 32 >> 4
                        b_desc += 32 >> 4
                    _tcgen05.commit(tma_empty + tma_stage, multicast_mask, 1)
                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

                _tcgen05.commit(tmem_full + tmem_stage, multicast_mask, 1)
                tmem_stage = (tmem_stage + 1) % num_tmem_stages
                if tmem_stage == 0:
                    tmem_parity ^= 1

        elif warp_id < 4:
            warp_in_group = warp_id % 4
            thread_in_group = tid % 128

            if warp_in_group == 0:
                _tcgen05.alloc(taddr, 1)
            cute.arch.barrier(barrier_id=tmem_alloc_barrier, number_of_threads=5 * 32)

            q_blocks = cute.zipped_divide(q, (1, MX_BLOCK))
            q_store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float8E4M3FN,
                num_bits_per_copy=256,
            )
            tmem_stage = 0
            parity = 0

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m = bid % grid_m
                bid_n = bid // grid_m
                if warp_in_group == 0:
                    cute.arch.mbarrier_wait(tmem_full + tmem_stage, parity)
                cute.arch.barrier(barrier_id=epilogue_barrier, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                for group in cutlass.range_constexpr(block_n // MX_BLOCK):
                    tmem_col = tmem_stage * block_n + group * MX_BLOCK
                    regs_lo = _tcgen05.ld(warp_in_group * 32, tmem_col, "32x32b", 16)
                    _tcgen05.wait_ld()
                    regs_hi = _tcgen05.ld(
                        warp_in_group * 32, tmem_col + 16, "32x32b", 16
                    )
                    _tcgen05.wait_ld()

                    if cutlass.const_expr(group == block_n // MX_BLOCK - 1):
                        _tcgen05.fence_before_thread_sync()
                        mbarrier.arrive(tmem_empty + tmem_stage, "cluster")

                    rounded = cute.make_rmem_tensor(MX_BLOCK, Float32)
                    lo_bf16 = cute.make_rmem_tensor(16, BFloat16)
                    hi_bf16 = cute.make_rmem_tensor(16, BFloat16)
                    lo_bf16.store(regs_lo.to(BFloat16))
                    hi_bf16.store(regs_hi.to(BFloat16))
                    lo_f32 = lo_bf16.load().to(Float32)
                    hi_f32 = hi_bf16.load().to(Float32)
                    for idx in cutlass.range_constexpr(16):
                        rounded[idx] = lo_f32[idx]
                        rounded[16 + idx] = hi_f32[idx]

                    q_regs = cute.make_rmem_tensor(MX_BLOCK, Float8E4M3FN)
                    scale = _quantize_group(rounded, q_regs)
                    global_row = bid_m * block_m + thread_in_group
                    scale_col = bid_n * (block_n // MX_BLOCK) + group
                    if global_row < m:
                        cute.copy(
                            q_store,
                            q_regs,
                            q_blocks[None, (global_row, scale_col)],
                        )
                        scales[global_row, scale_col] = scale.to(Uint8)

                cute.arch.barrier(barrier_id=epilogue_barrier, number_of_threads=128)
                tmem_stage = (tmem_stage + 1) % num_tmem_stages
                if tmem_stage == 0:
                    parity ^= 1

            cute.arch.barrier(barrier_id=epilogue_barrier, number_of_threads=128)
            if warp_in_group == 0:
                _tcgen05.dealloc(1)

    @staticmethod
    @cache
    def compile(block_n: int):
        m = cute.sym_int()
        a = make_fake_tensor(BFloat16, (m, K_DIM), (K_DIM, 1), assumed_align=16)
        b = make_fake_tensor(BFloat16, (N_DIM, K_DIM), (K_DIM, 1), assumed_align=16)
        q = make_fake_tensor(Float8E4M3FN, (m, N_DIM), (N_DIM, 1), assumed_align=32)
        scales = make_fake_tensor(
            Uint8,
            (m, N_DIM // MX_BLOCK),
            (N_DIM // MX_BLOCK, 1),
            assumed_align=16,
        )
        return cute.compile(
            Sm100RoutedDownMxfp8(block_n),
            a,
            b,
            q,
            scales,
            Int32(1),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def _select_block_n(num_tokens: int) -> int:
    if 16 <= num_tokens <= 128:
        return 32
    if 128 < num_tokens <= 256:
        return 64
    return 128


def routed_down_mxfp8(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-swizzled MXFP8 for Kimi-K3's routed down projection."""
    num_tokens = x.shape[0]
    q = torch.empty((num_tokens, N_DIM), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(
        (num_tokens, N_DIM // MX_BLOCK), dtype=torch.uint8, device=x.device
    )
    block_n = _select_block_n(num_tokens)
    total_tiles = (num_tokens + 127) // 128 * (N_DIM // block_n)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    compiled = Sm100RoutedDownMxfp8.compile(block_n)
    compiled(x, weight, q, scales, min(total_tiles, num_sms))
    return q, scales


class KimiK3RoutedDownMxfp8Op:
    """Validated, warmup-aware wrapper for the fixed Kimi-K3 projection."""

    def __init__(self) -> None:
        register_cutedsl_warmup_provider(self)

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        return tuple(
            CuTeDSLCompileUnit(
                name="K3 routed-down MXFP8",
                key=("k3-routed-down-mxfp8", block_n),
                compile=partial(Sm100RoutedDownMxfp8.compile, block_n),
            )
            for block_n in (32, 64, 128)
        )

    @staticmethod
    def can_run(x: torch.Tensor, weight: torch.Tensor) -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and x.ndim == 2
            and weight.ndim == 2
            and 0 < x.shape[0] <= MAX_FUSED_TOKENS
            and x.shape[1] == K_DIM
            and weight.shape == (N_DIM, K_DIM)
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.device == weight.device
            and x.is_contiguous()
            and weight.is_contiguous()
        )

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.can_run(x, weight):
            raise ValueError("Unsupported Kimi-K3 routed-down MXFP8 input.")
        return routed_down_mxfp8(x, weight)
