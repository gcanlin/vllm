# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the complete inverse-RoPE/quantization/wo_a chain at TP8."""

import json
import math
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from screen import call, candidates, error, measure
from woa_ops import invrope

from vllm.model_executor.kernels.linear.mxfp8.deep_gemm import (
    DeepGemmMxfp8BmmLinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    swizzle_mxfp8_scale,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.utils.deep_gemm import fp8_einsum

ROOT = Path(__file__).parent
MODEL = Path(
    os.environ.get("DSV41_MODEL", "/gpfs/mszn/models/deepseek-ai/DeepSeek-V4.1-Flash")
)


def main():
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    loaded = []
    for suffix in ("weight", "scale"):
        key = "layers.0.attn.wo_a." + suffix
        with safe_open(MODEL / index[key], framework="pt", device="cpu") as f:
            loaded.append(f.get_tensor(key))
    w = loaded[0][:1024].contiguous().cuda()
    s = (
        loaded[1]
        .view(torch.uint8)
        .repeat_interleave(32, dim=0)[:1024]
        .contiguous()
        .cuda()
    )
    sw = swizzle_mxfp8_scale(s, 1024, 4096)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(w.clone(), requires_grad=False),
        weight_scale=torch.nn.Parameter(s.clone(), requires_grad=False),
    )
    kernel = DeepGemmMxfp8BmmLinearKernel(Mxfp8LinearLayerConfig(bmm_batch_size=1))
    kernel.process_weights_after_loading(layer)
    count = math.ceil(
        2.5
        * torch.cuda.get_device_properties(0).L2_cache_size
        / (w.numel() + sw.numel())
    )
    weights = [
        (
            w.clone(),
            sw.clone(),
            layer.weight.detach().clone(),
            layer.weight_scale.detach().clone(),
        )
        for _ in range(count)
    ]
    rows = []
    selected = {}
    torch.manual_seed(4343)
    angles = torch.randn(128, 32, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), 1)
    for m in (1, 2, 4, 8, 16):
        o = torch.randn(m, 8, 512, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(m, device="cuda")

        def baseline(w, sw, dw, ds, o=o, pos=pos, m=m):
            a, s = fused_inv_rope_fp8_quant(
                o, pos, cache, 1, 8, quant_group_size=32, tma_aligned_scales=True
            )
            out = torch.empty(m, 1, 1024, device="cuda", dtype=torch.bfloat16)
            fp8_einsum("bhr,hdr->bhd", (a, s), (dw, ds), out, recipe=(1, 1, 32))
            return out[:, 0]

        ref = baseline(*weights[0])
        base = measure([lambda v=v, fn=baseline: fn(*v) for v in weights])["us"]
        trials = candidates(m, 1024, 4096, True, False)
        best = None
        for spec in trials:

            def candidate(w, sw, dw, ds, spec=spec, o=o, pos=pos):
                a, s = invrope(o, pos, cache)
                return call(spec, a, w, sw, s, False)

            row = dict(group=3, m=m, spec=spec, baseline_us=base, copies=count)
            try:
                err = error(candidate(*weights[0]), ref)
                assert err["finite"] and err["rmse"] < 0.001, err
                row.update(err)
                row.update(measure([lambda v=v, fn=candidate: fn(*v) for v in weights]))
                row["gain"] = 1 - row["us"] / base
                if best is None or row["us"] < best["us"]:
                    best = row
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)[-1000:]}"
            rows.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "screen-woa.json").write_text(json.dumps(rows, indent=2))
        if best is not None and best["gain"] >= 0.05:
            spec = best["spec"]

            def chosen(w, sw, dw, ds, spec=spec, o=o, pos=pos):
                a, s = invrope(o, pos, cache)
                return call(spec, a, w, sw, s, False)

            times = {"baseline": [], "candidate": []}
            for repeat in range(3):
                for label in (
                    ("baseline", "candidate")
                    if repeat % 2 == 0
                    else ("candidate", "baseline")
                ):
                    fn = baseline if label == "baseline" else chosen
                    times[label].append(
                        measure([lambda v=v, fn=fn: fn(*v) for v in weights])["us"]
                    )
            b = statistics.median(times["baseline"])
            c = statistics.median(times["candidate"])
            if c < b * 0.95 and max(times["candidate"]) < min(times["baseline"]):
                chosen(*weights[0])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = chosen(*weights[0])
                errors = []
                for scale in (0.0, 1e-5, 0.1, 1.0, 10.0, 100.0):
                    o.normal_().mul_(scale)
                    pos.random_(0, 128)
                    graph.replay()
                    err = error(actual, baseline(*weights[0]))
                    assert err["finite"] and err["rmse"] < 0.001, (m, scale, err)
                    errors.append(dict(scale=scale, **err))
                selected[str(m)] = dict(
                    spec=spec,
                    baseline_us=b,
                    candidate_us=c,
                    gain=1 - c / b,
                    timings=times,
                    errors=errors,
                    scope="full inverse-RoPE chain",
                )
        (ROOT / "selected-woa.json").write_text(json.dumps(selected, indent=2))
    print("WOA_SCREEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
