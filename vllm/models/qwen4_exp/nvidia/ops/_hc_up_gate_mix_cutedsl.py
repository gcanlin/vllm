# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr


class CuteHcUpGateMix:
    def __init__(
        self,
        *,
        element_type,
        num_rows: int,
        outputs_per_block: int,
        use_pdl: bool,
    ) -> None:
        self.element_type = element_type
        self.num_rows = num_rows
        self.outputs_per_block = outputs_per_block
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        gLora: cute.Tensor,
        gWeight: cute.Tensor,
        gX: cute.Tensor,
        gY: cute.Tensor,
        stream: CUstream,
    ) -> None:
        hidden_size = cute.size(gY, mode=[1])
        self.kernel(gLora, gWeight, gX, gY).launch(
            grid=[cute.ceil_div(hidden_size, self.outputs_per_block), 1, 1],
            block=[cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gLora: cute.Tensor,
        gWeight: cute.Tensor,
        gX: cute.Tensor,
        gY: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()

        num_rows: cutlass.Constexpr = self.num_rows
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        hc_count: cutlass.Constexpr = 4
        hidden_size: cutlass.Constexpr = 2560
        lora_dim: cutlass.Constexpr = 320
        vector_width: cutlass.Constexpr = 2
        block_size: cutlass.Constexpr = cute.arch.WARP_SIZE
        num_k_tiles: cutlass.Constexpr = lora_dim // (block_size * vector_width)

        acc_layout = cute.make_layout(
            (num_rows, hc_count, outputs_per_block),
            stride=(hc_count * outputs_per_block, outputs_per_block, 1),
        )
        acc = cute.make_rmem_tensor(acc_layout, cutlass.Float32)
        mixed_layout = cute.make_layout(
            (num_rows, outputs_per_block), stride=(outputs_per_block, 1)
        )
        mixed = cute.make_rmem_tensor(mixed_layout, cutlass.Float32)
        acc.fill(0.0)
        mixed.fill(0.0)

        copy_lora = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_weight = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        copy_weight_prefetch = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        lora_regs = cute.make_rmem_tensor(
            cute.make_layout((num_rows, vector_width), stride=(vector_width, 1)),
            self.element_type,
        )
        weight_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (hc_count, outputs_per_block, vector_width),
                stride=(outputs_per_block * vector_width, vector_width, 1),
            ),
            self.element_type,
        )

        gLora_vec = cute.logical_divide(gLora, (None, vector_width))
        gWeight_vec = cute.logical_divide(gWeight, (None, vector_width))
        tLora_all = cute.logical_divide(gLora_vec, (None, (None, block_size)))
        tWeight_all = cute.logical_divide(gWeight_vec, (None, (None, block_size)))
        tLora = tLora_all[None, (None, (tidx, None))]
        hidden_base = block_idx * outputs_per_block
        prefetched_weight = cute.make_rmem_tensor(
            cute.make_layout(
                (2, hc_count, outputs_per_block, vector_width),
                stride=(
                    hc_count * outputs_per_block * vector_width,
                    outputs_per_block * vector_width,
                    vector_width,
                    1,
                ),
            ),
            self.element_type,
        )
        for k_tile in cutlass.range_constexpr(2):
            for hc in cutlass.range_constexpr(hc_count):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    weight_row = hc * hidden_size + hidden_base + ni
                    tWeight = tWeight_all[weight_row, (None, (tidx, None))]
                    cute.copy(
                        copy_weight_prefetch,
                        tWeight[None, k_tile],
                        prefetched_weight[k_tile, hc, ni, None],
                    )

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        for k_tile in cutlass.range_constexpr(num_k_tiles):
            for mi in cutlass.range_constexpr(num_rows):
                cute.copy(
                    copy_lora,
                    tLora[mi, None, k_tile],
                    lora_regs[mi, None],
                )
            if const_expr(k_tile < 2):
                weight_f32 = (
                    prefetched_weight[k_tile, None, None, None]
                    .load()
                    .to(cutlass.Float32)
                )
            else:
                for hc in cutlass.range_constexpr(hc_count):
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        weight_row = hc * hidden_size + hidden_base + ni
                        tWeight = tWeight_all[weight_row, (None, (tidx, None))]
                        cute.copy(
                            copy_weight,
                            tWeight[None, k_tile],
                            weight_regs[hc, ni, None],
                        )
                weight_f32 = weight_regs.load().to(cutlass.Float32)
            for vi in cutlass.range_constexpr(vector_width):
                for mi in cutlass.range_constexpr(num_rows):
                    for hc in cutlass.range_constexpr(hc_count):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, hc, ni] += (
                                lora_regs[mi, vi].to(cutlass.Float32)
                                * weight_f32[hc, ni, vi]
                            )

        for mi in cutlass.range_constexpr(num_rows):
            for hc in cutlass.range_constexpr(hc_count):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    gate = cute.arch.warp_reduction_sum(acc[mi, hc, ni])
                    if tidx == 0:
                        gate = gate.to(self.element_type).to(cutlass.Float32)
                        gate = cute.arch.rcp_approx(
                            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
                        )
                        x_col = hc * hidden_size + hidden_base + ni
                        mixed[mi, ni] += gate * gX[mi, x_col].to(cutlass.Float32)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if tidx == 0:
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    gY[mi, hidden_base + ni] = (mixed[mi, ni] / hc_count).to(
                        self.element_type
                    )
