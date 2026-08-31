# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

_cutedsl_available: bool | None = None


@dataclass(frozen=True, slots=True)
class HcUpGateMixConfig:
    num_rows: int
    outputs_per_block: int


HC_UP_GATE_MIX_CONFIGS = {
    1: HcUpGateMixConfig(num_rows=1, outputs_per_block=2),
    2: HcUpGateMixConfig(num_rows=2, outputs_per_block=2),
}


class HcUpGateMixCuTeDSL:
    def __init__(self) -> None:
        self._compiled: dict[tuple[torch.dtype, HcUpGateMixConfig], Any] = {}
        self._warmup_configs: set[tuple[torch.dtype, HcUpGateMixConfig]] = set()
        self._warmup_registered = False

    @staticmethod
    def is_available() -> bool:
        global _cutedsl_available
        if _cutedsl_available is not None:
            return _cutedsl_available
        try:
            import cutlass  # noqa: F401
            import cutlass.cute  # noqa: F401

            _cutedsl_available = True
        except ImportError:
            _cutedsl_available = False
        return _cutedsl_available

    @staticmethod
    def _stream():
        from cuda.bindings.driver import CUstream

        from vllm.utils.torch_utils import current_stream

        return CUstream(current_stream().cuda_stream)

    @staticmethod
    def _use_pdl() -> bool:
        from vllm.platforms import current_platform

        return current_platform.is_arch_support_pdl()

    def _compile(self, dtype: torch.dtype, config: HcUpGateMixConfig) -> None:
        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        from ._hc_up_gate_mix_cutedsl import CuteHcUpGateMix

        if dtype != torch.bfloat16:
            raise ValueError("HC fused up projection requires BF16")
        element_type = BFloat16
        lora = make_fake_tensor(element_type, (config.num_rows, 320), divisibility=2)
        weight = make_fake_tensor(element_type, (10240, 320), divisibility=2)
        x = make_fake_tensor(
            element_type,
            (config.num_rows, 10240),
            divisibility=1,
        )
        y = make_fake_tensor(
            element_type,
            (config.num_rows, 2560),
            divisibility=1,
        )
        kernel = CuteHcUpGateMix(
            element_type=element_type,
            num_rows=config.num_rows,
            outputs_per_block=config.outputs_per_block,
            use_pdl=self._use_pdl(),
        )
        self._compiled[(dtype, config)] = cute.compile(
            kernel,
            lora,
            weight,
            x,
            y,
            self._stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )

    def request_warmup_configs(self, dtype: torch.dtype) -> None:
        self._warmup_configs.update(
            (dtype, config) for config in HC_UP_GATE_MIX_CONFIGS.values()
        )
        if self._warmup_registered:
            return
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )

        register_cutedsl_warmup_provider(self)
        self._warmup_registered = True

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        return tuple(
            CuTeDSLCompileUnit(
                name="Qwen4Exp fused HC up projection and gate mix",
                key=("qwen4-exp-hc-up-gate-mix", dtype, config),
                compile=partial(self._compile, dtype, config),
            )
            for dtype, config in sorted(
                self._warmup_configs,
                key=lambda item: (
                    str(item[0]),
                    item[1].num_rows,
                    item[1].outputs_per_block,
                ),
            )
        )

    def __call__(
        self,
        x: torch.Tensor,
        lora: torch.Tensor,
        weight: torch.Tensor,
        config: HcUpGateMixConfig,
    ) -> torch.Tensor:
        if x.shape != (config.num_rows, 10240):
            raise ValueError("x must have shape (M, 10240)")
        if lora.shape != (config.num_rows, 320):
            raise ValueError("lora must have shape (M, 320)")
        if weight.shape != (10240, 320):
            raise ValueError("weight must have shape (10240, 320)")
        if (
            x.dtype != torch.bfloat16
            or lora.dtype != x.dtype
            or weight.dtype != x.dtype
        ):
            raise ValueError("HC fused up projection requires BF16 inputs")
        if not x.is_cuda or not lora.is_cuda or not weight.is_cuda:
            raise ValueError("HC fused up projection requires CUDA inputs")
        if x.device != lora.device or x.device != weight.device:
            raise ValueError("HC fused up projection inputs must share a device")
        if (
            not x.is_contiguous()
            or not lora.is_contiguous()
            or not weight.is_contiguous()
        ):
            raise ValueError("HC fused up projection inputs must be contiguous")

        key = (x.dtype, config)
        if key not in self._compiled:
            self._compile(*key)
        output = x.new_empty((config.num_rows, 2560))
        self._compiled[key](lora, weight, x, output, self._stream())
        return output


hc_up_gate_mix_cutedsl = HcUpGateMixCuTeDSL()


__all__ = [
    "HC_UP_GATE_MIX_CONFIGS",
    "HcUpGateMixConfig",
    "hc_up_gate_mix_cutedsl",
]
