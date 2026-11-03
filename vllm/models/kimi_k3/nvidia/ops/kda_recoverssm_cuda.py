# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA RecoverSSM verify and state commit for an FP32 KDA state, built on first use."""

import os
from pathlib import Path

from torch.utils.cpp_extension import load

_ext = None


def ext():
    global _ext
    if _ext is None:
        build = os.environ.get("KDA_CUDA_BUILD", "/tmp/kda_cuda_build")
        os.makedirs(build, exist_ok=True)
        # The packed FP32x2 math is sm_100 only; the image's arch list spans older GPUs.
        archs = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
        try:
            _ext = load(
                "kda_recoverssm_cuda",
                [str(Path(__file__).with_name("kda_recoverssm.cu"))],
                extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
                build_directory=build,
                verbose=os.environ.get("KDA_CUDA_VERBOSE") == "1",
            )
        finally:
            if archs is None:
                del os.environ["TORCH_CUDA_ARCH_LIST"]
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = archs
    return _ext


def kda_recoverssm_verify(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    lower_bound,
    checkpoint_state,
    records,
    query_start_loc,
    state_indices,
    spec_query_len,
    out=None,
):
    if out is None:
        out = v.new_empty(v.shape)
    ext().verify(
        q[0],
        k[0],
        v[0],
        raw_g[0],
        raw_beta[0],
        A_log,
        dt_bias,
        checkpoint_state,
        records.correction,
        records.key_gate,
        out[0],
        query_start_loc,
        state_indices,
        lower_bound or 0.0,
        lower_bound is not None,
        0,
        spec_query_len,
    )
    return out


def recoverssm_commit(ctx, state_indices, batch, align):
    """Same contract as cute_recoverssm_commit: the commit plan in ctx is filled."""
    ext().commit(
        ctx.checkpoints[0],
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.correction_caches[0],
        ctx.correction_cache_base_addrs,
        ctx.correction_cache_block_strides,
        ctx.kg_caches[0],
        ctx.kg_cache_base_addrs,
        ctx.kg_cache_block_strides,
        ctx.A_log,
        ctx.dt_bias,
        state_indices,
        ctx.commit_lens,
        ctx.final_state_indices,
        ctx.boundary_state_indices,
        ctx.boundary_recovery_lens,
        ctx.lower_bound or 0.0,
        ctx.lower_bound is not None,
        0,
        batch,
        align,
    )
