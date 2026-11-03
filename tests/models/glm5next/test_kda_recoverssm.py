# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM on GLM-5.3-Flash KDA matches GLM's native speculative path.

The native path advances one recurrent state per draft position and keeps the slot
of the accepted length. RecoverSSM verifies the same window off one checkpoint and
rebuilds the accepted state after sampling. Both must give the same verify outputs
and committed state, with GLM's A_log layout and bounded gate.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.models.kimi_k3.nvidia.ops import recoverssm as recoverssm_ops
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
    kda_recoverssm_verify,
)

DEVICE = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_glm_kda_recoverssm_matches_native_spec(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(recoverssm_ops, "is_conv_state_dim_first", lambda: True)
    torch.manual_seed(0)
    num_heads, dim, lower_bound, num_spec = 64, 128, -5.0, 5
    query_len = num_spec + 1
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
        [layer], spec_query_len=query_len, max_num_reqs=num_seqs
    )
    context.commit(torch.tensor(accepted, dtype=torch.int32, device=DEVICE), blocks, cu)
    torch.testing.assert_close(
        checkpoint[blocks], expected_states, atol=2e-3, rtol=2e-3
    )
