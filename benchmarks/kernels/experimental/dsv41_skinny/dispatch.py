# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measured process-local dispatch. No checkpoint or production-source edits."""

import json
from collections import Counter
from pathlib import Path
from types import MethodType

import flashinfer
import torch
from screen import call, error
from woa_ops import invrope

import vllm.envs as envs
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    swizzle_mxfp8_scale,
)
from vllm.utils.multi_stream_utils import execute_in_parallel
from vllm.utils.torch_utils import direct_register_custom_op

ROOT = Path(__file__).parent
PLAN = json.loads((ROOT / "selected.json").read_text())
HITS = Counter()


def impl(
    x: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor | None,
    sx: torch.Tensor | None,
    group: int,
    fp32: bool,
) -> torch.Tensor:
    spec = PLAN[str(group)][str(x.shape[0])]["spec"]
    HITS[(group, x.shape[0])] += 1
    return call(spec, x, w, sw, sx, fp32)


def fake(
    x: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor | None,
    sx: torch.Tensor | None,
    group: int,
    fp32: bool,
) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0], w.shape[0]), dtype=torch.float32 if fp32 else torch.bfloat16
    )


direct_register_custom_op("dsv41_full_skinny_trial", impl, fake_impl=fake)


def inv_impl(
    o: torch.Tensor, p: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return invrope(o, p, c)


def inv_fake(
    o: torch.Tensor, p: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m = o.shape[0]
    return (
        o.new_empty((m, 4096), dtype=torch.float8_e4m3fn),
        o.new_empty((((m + 127) // 128) * 128 * 128,), dtype=torch.uint8),
    )


direct_register_custom_op("dsv41_full_skinny_inv_trial", inv_impl, fake_impl=inv_fake)


def o_proj(self, o, positions):
    if str(o.shape[0]) in PLAN.get("3", {}):
        a, s = torch.ops.vllm.dsv41_full_skinny_inv_trial(
            o, positions, self.rotary_emb.cos_sin_cache
        )
        z = torch.ops.vllm.dsv41_full_skinny_trial(
            a, self._skinny_woa_w, self._skinny_woa_s, s, 3, False
        )
        return self.wo_b(z)
    return self._skinny_original_o_proj(o, positions)


def supported(x, group):
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and str(x.shape[0]) in PLAN.get(str(group), {})
        and x.stride(1) == 1
        and x.is_contiguous()
    )


def apply_mx(self, layer, x, bias=None):
    group = self._skinny_group
    qa = as_quantized_activation(x, self.input_quant_key())
    data = qa.data if qa is not None else x
    # The plan was measured for this layer's actual input representation.
    wants_pre = group in (2, 10)
    if supported(data, group) and wants_pre == (qa is not None):
        entry = PLAN[str(group)][str(data.shape[0])]
        scale = (
            layer._skinny_packed_scale
            if entry["spec"]["kind"] == "cute_mx"
            else layer.weight_scale
        )
        y = torch.ops.vllm.dsv41_full_skinny_trial(
            data,
            layer.weight.T,
            scale,
            qa.scale if qa is not None else None,
            group,
            False,
        )
        return y if bias is None else y + bias
    return self._skinny_original(layer, x, bias)


def apply_bf16(self, layer, x, bias=None):
    group = self._skinny_group
    if supported(x, group) and x.dtype == torch.bfloat16:
        y = torch.ops.vllm.dsv41_full_skinny_trial(
            x, layer.weight, None, None, group, False
        )
        return y if bias is None else y + bias
    return self._skinny_original(layer, x, bias)


def gate_forward(self, x):
    if supported(x, 5) and x.dtype == torch.bfloat16:
        return torch.ops.vllm.dsv41_full_skinny_trial(
            x, self.weight, None, None, 5, True
        ), None
    return self._skinny_original(x)


def parallel_projections(self, hidden_states):
    aux_streams = self.aux_stream_list
    if aux_streams is not None:
        aux_streams = aux_streams[:2]
    aux_fns = [None, None]
    if self.compressor is not None:
        compressor = self.compressor
        group = 13 if compressor.has_gate else 15

        def compressor_score():
            w = compressor.fused_wkv_wgate.weight
            if supported(hidden_states, group):
                return torch.ops.vllm.dsv41_full_skinny_trial(
                    hidden_states, w, None, None, group, True
                )
            return torch.mm(hidden_states, w.T, out_dtype=torch.float32)

        aux_fns[0] = compressor_score
    if self.indexer is not None:

        def indexer_weights():
            weights, _ = self.indexer.weights_proj(hidden_states)
            return weights

        aux_fns[1] = indexer_weights
    qr_kv, (kv_score, indexer_weights_out) = execute_in_parallel(
        lambda: self._fused_wqa_wkv_gemm(hidden_states),
        aux_fns,
        self.ln_events[0],
        self.ln_events[1:3],
        aux_streams,
        enable=hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
    )
    return qr_kv, kv_score, indexer_weights_out


def group_for(name):
    if name == "lm_head":
        return 16
    for suffix, group in [
        (".attn.fused_wqa_wkv", 1),
        (".attn.wq_b", 2),
        (".attn.wo_b", 4),
        (".ffn.gate", 5),
        (".engram.wkv", 9),
        (".attn.indexer.wq_b", 10),
        (".attn.indexer.weights_proj", 11),
        (".attn.indexer.wk", 12),
    ]:
        if name.endswith(suffix):
            return group
    if name.endswith(".attn.compressor.fused_wkv_wgate"):
        return None  # Bound below using the loaded shape.
    return None


def install(model, rank):
    checked = []
    installed = Counter()
    for name, module in model.named_modules():
        group = group_for(name)
        if name.endswith(".attn.compressor.fused_wkv_wgate"):
            group = 13 if module.weight.shape[0] == 1024 else 15
        if group is None or not PLAN.get(str(group)):
            continue
        mx = module.weight.dtype == torch.float8_e4m3fn
        w = module.weight.T if mx else module.weight
        sw = module.weight_scale if mx else None
        packed_sw = None
        if mx and any(
            s["spec"]["kind"] == "cute_mx" for s in PLAN[str(group)].values()
        ):
            from cute_mx import pack_scale

            packed_sw = pack_scale(sw, w.shape[0], w.shape[1])
            grouped = packed_sw.reshape(w.shape[0] // 32, 32, w.shape[1] // 128)
            assert torch.equal(grouped, grouped[:, :1].expand_as(grouped))
            module.register_buffer("_skinny_packed_scale", packed_sw, persistent=False)
        pre = group in (2, 10)
        fp32 = group in (5, 13, 15)
        for ms, selected in PLAN[str(group)].items():
            m = int(ms)
            for seed in range(2):
                generator = torch.Generator(device=w.device).manual_seed(
                    731 + seed + rank
                )
                x = torch.randn(
                    m,
                    w.shape[1],
                    device=w.device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                if mx:
                    a, s = flashinfer.mxfp8_quantize(x, backend="cute-dsl")
                    qx = QuantizedActivation(
                        a,
                        s,
                        x.dtype,
                        x.shape,
                        module.quant_method.kernel.input_quant_key(),
                    )
                    ref = module.quant_method.kernel.apply_weights(
                        module, qx if pre else x
                    )
                elif group == 5:
                    ref, _ = module(x)
                elif fp32:
                    ref = torch.mm(x, w.T, out_dtype=torch.float32)
                else:
                    ref = module.quant_method.apply(module, x, None)
                y = call(
                    selected["spec"],
                    a if pre else x,
                    w,
                    packed_sw if selected["spec"]["kind"] == "cute_mx" else sw,
                    s if pre else None,
                    fp32,
                )
                err = error(y, ref)
                assert err["finite"] and err["rmse"] < (1e-5 if fp32 else 0.001), (
                    name,
                    m,
                    err,
                )
                checked.append(dict(name=name, group=group, m=m, seed=seed, **err))
        if mx:
            obj = module.quant_method.kernel
            obj._skinny_original = obj.apply_weights
            obj._skinny_group = group
            obj.apply_weights = MethodType(apply_mx, obj)
        elif group == 5:
            module._skinny_original = module.forward
            module.forward = MethodType(gate_forward, module)
        elif not fp32:
            obj = module.quant_method
            obj._skinny_original = obj.apply
            obj._skinny_group = group
            obj.apply = MethodType(apply_bf16, obj)
        installed[group] += 1
    for layer in model.model.layers:
        if layer.attn.compressor is not None:
            group = 13 if layer.attn.compressor.has_gate else 15
            if PLAN.get(str(group)):
                layer.attn._run_parallel_input_projections = MethodType(
                    parallel_projections, layer.attn
                )
        if PLAN.get("3"):
            from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
                fused_inv_rope_fp8_quant,
            )
            from vllm.utils.deep_gemm import fp8_einsum

            attn = layer.attn
            assert attn.n_local_groups == 1 and attn.n_local_heads == 8
            weight = attn.wo_a.weight.detach()[0]
            packed = attn.wo_a.weight_scale.detach()[0]
            assert weight.shape == (1024, 4096) and packed.shape == (1024, 32)
            linear = (
                (
                    (
                        packed.to(torch.int64)[:, :, None]
                        >> (torch.arange(4, device=weight.device) * 8)
                    )
                    & 255
                )
                .to(torch.uint8)
                .reshape(1024, 128)
            )
            sw = swizzle_mxfp8_scale(linear, 1024, 4096)
            for ms, chosen in PLAN["3"].items():
                m = int(ms)
                for seed in range(2):
                    generator = torch.Generator(device=weight.device).manual_seed(
                        573 + rank + seed
                    )
                    o = torch.randn(
                        m,
                        8,
                        512,
                        device=weight.device,
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    positions = torch.arange(m, device=weight.device)
                    cache = attn.rotary_emb.cos_sin_cache
                    a, s = fused_inv_rope_fp8_quant(
                        o,
                        positions,
                        cache,
                        1,
                        8,
                        quant_group_size=32,
                        tma_aligned_scales=True,
                    )
                    ref = torch.empty(
                        m, 1, 1024, device=weight.device, dtype=torch.bfloat16
                    )
                    fp8_einsum(
                        "bhr,hdr->bhd",
                        (a, s),
                        (attn.wo_a.weight, attn.wo_a.weight_scale),
                        ref,
                        recipe=(1, 1, 32),
                    )
                    aq, sq = invrope(o, positions, cache)
                    actual = call(chosen["spec"], aq, weight, sw, sq, False)
                    err = error(actual, ref[:, 0])
                    assert err["finite"] and err["rmse"] < 0.001, (attn.prefix, m, err)
                    checked.append(
                        dict(name=attn.prefix + ".wo_a", group=3, m=m, seed=seed, **err)
                    )
            attn.register_buffer("_skinny_woa_w", weight, persistent=False)
            attn.register_buffer("_skinny_woa_s", sw, persistent=False)
            attn._skinny_original_o_proj = attn._o_proj
            attn._o_proj = MethodType(o_proj, attn)
            installed[3] += 1
    import os

    output = Path(os.environ["DSV41_GEMM_OUTPUT"])
    (output / f"quality-rank{rank}.json").write_text(json.dumps(checked, indent=2))
    return dict(
        installed=dict(installed),
        checks=len(checked),
        max_rmse=max((r["rmse"] for r in checked), default=0.0),
    )
