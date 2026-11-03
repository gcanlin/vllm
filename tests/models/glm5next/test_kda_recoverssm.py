# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM on GLM-5.3-Flash KDA matches GLM's native speculative path.

The native path advances one recurrent state per draft position and keeps the slot
of the accepted length. RecoverSSM verifies the same window off one checkpoint and
rebuilds the accepted state after sampling. Both must give the same verify outputs
and committed state, with GLM's A_log layout and bounded gate, with the Triton and
the CUDA kernels.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.models.kimi_k3.nvidia.ops import recoverssm as recoverssm_ops
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
    can_use_cuda_recoverssm,
    kda_recoverssm_verify,
)

DEVICE = "cuda"


def _skip_without_cuda_kernels(window: int):
    supported, reason = can_use_cuda_recoverssm(
        128, window, torch.bfloat16, torch.float32
    )
    if not supported:
        pytest.skip(f"the CUDA RecoverSSM kernels cannot run: {reason}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("backend", ["triton", "cuda"])
@torch.inference_mode()
def test_glm_kda_recoverssm_matches_native_spec(
    backend: str, monkeypatch: pytest.MonkeyPatch
):
    num_heads, dim, lower_bound, num_spec = 64, 128, -5.0, 5
    query_len = num_spec + 1
    if backend == "cuda":
        _skip_without_cuda_kernels(query_len)
    monkeypatch.setattr(recoverssm_ops, "is_conv_state_dim_first", lambda: True)
    torch.manual_seed(0)
    accepted = [1, 3, query_len]
    num_seqs = len(accepted)
    total = num_seqs * query_len

    def rnd(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE)

    q, k, v, raw_g = (rnd(1, total, num_heads, dim) for _ in range(4))
    raw_beta = rnd(1, total, num_heads)
    A_log = (0.2 * torch.randn(num_heads, device=DEVICE)).view(1, 1, num_heads, 1)
    dt_bias = 0.1 * torch.randn(num_heads * dim, device=DEVICE)
    cu = torch.arange(0, total + 1, query_len, dtype=torch.int32, device=DEVICE)
    init = 0.05 * torch.randn(num_seqs, num_heads, dim, dim, device=DEVICE)

    # Native: slot 0 of each row holds the initial state; token t writes slot t.
    native_states = torch.zeros(1 + total, num_heads, dim, dim, device=DEVICE)
    slots = (1 + torch.arange(total, device=DEVICE, dtype=torch.int32)).view(
        num_seqs, query_len
    )
    native_states[slots[:, 0]] = init
    native_out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta,
        initial_state=native_states,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=slots,
        num_accepted_tokens=torch.ones(num_seqs, dtype=torch.int32, device=DEVICE),
        sigmoid_beta=True,
        a_log=A_log,
        g_bias=dt_bias,
        compute_gate=True,
        lower_bound=lower_bound,
    )
    expected_states = torch.stack(
        [native_states[slots[i, n - 1]] for i, n in enumerate(accepted)]
    )

    blocks = torch.arange(1, num_seqs + 1, dtype=torch.int32, device=DEVICE)
    checkpoint = torch.zeros(num_seqs + 1, num_heads, dim, dim, device=DEVICE)
    checkpoint[blocks] = init
    records = KDARecoverSSMRecords.allocate(
        num_seqs, num_heads, query_len, dim, torch.bfloat16, torch.device(DEVICE)
    )
    out = kda_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log.view(-1),
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        checkpoint_state=checkpoint,
        records=records,
        query_start_loc=cu,
        state_indices=blocks,
        spec_query_len=query_len,
        backend=backend,
    )
    torch.testing.assert_close(checkpoint[blocks], init)
    torch.testing.assert_close(out.float(), native_out.float(), atol=2e-2, rtol=2e-2)

    conv_state = torch.zeros(
        num_seqs + 1, 12, 3 + query_len - 1, dtype=torch.bfloat16, device=DEVICE
    )
    layer = SimpleNamespace(
        kv_cache=(conv_state, checkpoint),
        recoverssm_records=records,
        A_log=A_log,
        dt_bias=dt_bias,
        local_num_heads=num_heads,
        head_dim=dim,
        gate_lower_bound=lower_bound,
    )
    context = KDARecoverSSMCommitContext.create(
        [layer], spec_query_len=query_len, max_num_reqs=num_seqs, backend=backend
    )
    context.commit(torch.tensor(accepted, dtype=torch.int32, device=DEVICE), blocks, cu)
    torch.testing.assert_close(
        checkpoint[blocks], expected_states, atol=2e-3, rtol=2e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("window", [2, 6, 8])
@pytest.mark.parametrize("batch", [1, 13])
@torch.inference_mode()
def test_glm_kda_recoverssm_cuda_align_commit_matches_triton(
    window: int, batch: int, monkeypatch: pytest.MonkeyPatch
):
    """With prefix caching the CUDA commit stores the same final state, in the
    block of the window's last token, and the same state at a crossed block
    boundary as the Triton commit."""
    _skip_without_cuda_kernels(window)
    monkeypatch.setattr(recoverssm_ops, "is_conv_state_dim_first", lambda: True)
    torch.manual_seed(window * 31 + batch)
    num_heads, dim, block_size, width = 64, 128, 8, 4
    slots = 1 + batch * width

    def layer():
        conv_state = torch.zeros(
            slots, 12, 3 + window - 1, dtype=torch.bfloat16, device=DEVICE
        )
        state = 0.05 * torch.randn(slots, num_heads, dim, dim, device=DEVICE)
        records = KDARecoverSSMRecords(
            torch.randn(batch, num_heads, window, dim, device=DEVICE),
            torch.randn(batch, num_heads, window, 2 * dim, device=DEVICE).bfloat16(),
        )
        return SimpleNamespace(
            kv_cache=(conv_state, state),
            recoverssm_records=records,
            A_log=(0.2 * torch.randn(num_heads, device=DEVICE)).view(
                1, 1, num_heads, 1
            ),
            dt_bias=0.1 * torch.randn(num_heads * dim, device=DEVICE),
            local_num_heads=num_heads,
            head_dim=dim,
            gate_lower_bound=-5.0,
        )

    layers = [layer() for _ in range(2)]
    query_start = torch.arange(
        0, (batch + 1) * window, window, dtype=torch.int32, device=DEVICE
    )
    accepted = torch.randint(0, window + 1, (batch,), dtype=torch.int32, device=DEVICE)
    block_table = (
        1 + torch.arange(batch * width, dtype=torch.int32, device=DEVICE)
    ).view(batch, width)
    num_computed = torch.randint(
        1, 2 * block_size, (batch,), dtype=torch.int32, device=DEVICE
    )
    # Request 0 crosses a block boundary with every accepted token.
    num_computed[0] = block_size - 1
    accepted[0] = window
    state_indices = block_table.gather(
        1, ((num_computed - 1) // block_size).long().unsqueeze(1)
    ).squeeze(1)

    initial = [layer.kv_cache[1].clone() for layer in layers]
    results = {}
    for backend in ("triton", "cuda"):
        for layer, before in zip(layers, initial):
            layer.kv_cache[1].copy_(before)
        context = KDARecoverSSMCommitContext.create(
            layers, spec_query_len=window, max_num_reqs=batch, backend=backend
        )
        context.commit(
            accepted,
            state_indices,
            query_start,
            block_table=block_table,
            num_computed_tokens=num_computed,
            mamba_block_size=block_size,
        )
        results[backend] = [layer.kv_cache[1].clone() for layer in layers]
    for triton_state, cuda_state in zip(results["triton"], results["cuda"]):
        torch.testing.assert_close(cuda_state, triton_state, atol=1e-5, rtol=1e-5)
