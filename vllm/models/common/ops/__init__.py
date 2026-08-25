# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ops shared across model implementations."""

from .fused_qk_rmsnorm import (
    fused_q_kv_rmsnorm,
    fused_q_kv_rmsnorm_fp8_quant,
)

__all__ = [
    "fused_q_kv_rmsnorm",
    "fused_q_kv_rmsnorm_fp8_quant",
]
