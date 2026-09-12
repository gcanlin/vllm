# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Loaded-weight projection sweep with cold rotating-weight CUDA graphs."""

import argparse
import gc
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import flashinfer
import torch
from kernels import simt, tc

from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import LLBf16Gemm, ll_bf16_gemm
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

ROOT = Path(__file__).parent
LL_CACHE = {}


def ll_call(x, w, config):
    key = tuple(sorted(config.items()))
    if key not in LL_CACHE:
        obj = LLBf16Gemm(prefetch_pdl_weights=x.shape[0] == 1)
        ck = LLBf16Gemm.CompileKey(**config)
        obj.dispatch = lambda **kwargs: ck
        LL_CACHE[key] = obj
    return LL_CACHE[key](x, w)


def call(spec, x, w, sw=None, sx=None, fp32=False):
    kind, cfg = spec["kind"], spec["config"]
    if kind == "cute_mx":
        from cute_mx import gemm

        return gemm(x, w, sw, sx, **cfg)
    if kind == "cute":
        return shape_dynamic_skinny_gemm(x, w, SkinnyGemmConfig(**cfg))
    if kind == "ll":
        out = ll_call(x, w, cfg)
        return out if fp32 else out.to(torch.bfloat16)
    return {"simt": simt, "tc": tc}[kind](x, w, sw, sx, out_fp32=fp32, **cfg)


def measure(calls, samples=5):
    for fn in calls:
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in calls:
            fn()
    for _ in range(3):
        graph.replay()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        a.record()
        for _ in range(2):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / (2 * len(calls)))
    return dict(us=statistics.median(times), samples_us=times)


def error(y, ref):
    delta = y.float() - ref.float()
    return dict(
        rmse=(
            delta.square().mean().sqrt()
            / ref.float().square().mean().sqrt().clamp_min(1e-30)
        ).item(),
        max_abs=delta.abs().max().item(),
        finite=bool(torch.isfinite(y).all()),
    )


def candidates(m, n, k, mx, fp32):
    result = []
    if not mx:
        if not fp32:
            configs = [shape_dynamic_skinny_gemm._config(m, n, k)]
            for bs, outputs, ku in [
                (32, 4, 2),
                (64, 2, 2),
                (64, 4, 2),
                (64, 8, 2),
                (128, 2, 2),
                (128, 4, 2),
                (160, 2, 2),
                (160, 4, 2),
            ]:
                if k % (bs * 8) == 0 and n % outputs == 0:
                    configs.append(SkinnyGemmConfig(m, bs, outputs, ku))
            for bs, outputs in [(64, 4), (160, 4)]:
                if k % (bs * 8) == 0 and k >= bs * 16 and n % outputs == 0:
                    configs.append(SkinnyGemmConfig(m, bs, outputs, static_k=k))
            result.extend(
                dict(kind="cute", config=asdict(c)) for c in dict.fromkeys(configs)
            )
        if fp32 or n <= 1024:
            result.extend(
                dict(kind="ll", config=dict(backend="dotprod", M=m, K=k, bs=bs))
                for bs in (64, 128, 256)
            )
            if k >= 2048 and m > 1:
                result.extend(
                    dict(
                        kind="ll",
                        config=dict(backend="splitk", split_k=sk, num_stages=st),
                    )
                    for sk, st in [(4, 3), (6, 4), (8, 3)]
                )
    for bm in sorted(set((1, min(m, 2), min(m, 4)))):
        for bn, warps in [(1, 4), (2, 4), (4, 4), (8, 4)]:
            result.append(dict(kind="simt", config=dict(bm=bm, bn=bn, warps=warps)))
    if mx or fp32:
        for bn, bk, sk in [
            (32, 128, 1),
            (32, 256, 1),
            (64, 128, 1),
            (32, 256, 2),
            (64, 256, 2),
            (32, 128, 4),
        ]:
            result.append(dict(kind="tc", config=dict(bn=bn, bk=bk, sk=sk, warps=4)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--groups", type=int, nargs="+", default=[16, 13, 15, 11, 12, 5, 1, 2, 4, 10, 9]
    )
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--output", default="screen.json")
    args = p.parse_args()
    torch.manual_seed(2026)
    torch.accelerator.set_device_index(0)
    inventory = json.loads((ROOT / "inventory-rank0.json").read_text())
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    rows = []
    for group in args.groups:
        item = next(x for x in inventory if x["group"] == group)
        tensors = torch.load(ROOT / "weights" / f"group{group}.pt", weights_only=True)
        w = tensors["weight"].cuda()
        mx = w.dtype == torch.float8_e4m3fn
        if mx:
            w = w.T
        assert w.is_contiguous()
        sw = tensors["weight_scale"].cuda() if mx else None
        n, k = w.shape
        fp32 = group in (5, 13, 15)
        pre = group in (2, 10)
        weight_bytes = w.numel() * w.element_size() + (0 if sw is None else sw.numel())
        count = max(3, math.ceil(2.5 * l2 / weight_bytes))
        weights = [(w, sw)] + [
            (w.clone(), None if sw is None else sw.clone()) for _ in range(count - 1)
        ]
        for m in args.tokens:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            qa, sa = (
                flashinfer.mxfp8_quantize(x, backend="cute-dsl") if mx else (None, None)
            )

            def baseline(
                w, sw, x=x, qa=qa, sa=sa, mx=mx, pre=pre, group=group, fp32=fp32
            ):
                if mx:
                    a, s = (
                        (qa, sa)
                        if pre
                        else flashinfer.mxfp8_quantize(x, backend="cute-dsl")
                    )
                    return flashinfer.mm_mxfp8(a, w.T, s, sw, backend="cute-dsl")
                if group == 5:
                    return ll_bf16_gemm(x, w)
                return (
                    torch.mm(x, w.T, out_dtype=torch.float32)
                    if fp32
                    else torch.nn.functional.linear(x, w)
                )

            ref = baseline(w, sw).clone()
            bt = measure([lambda pair=pair, fn=baseline: fn(*pair) for pair in weights])
            common = dict(
                group=group,
                name=item["name"],
                m=m,
                n=n,
                k=k,
                mx=mx,
                pre=pre,
                fp32=fp32,
                copies=count,
                weight_bytes=weight_bytes,
                l2_bytes=l2,
                baseline_us=bt["us"],
            )
            print("SHAPE", json.dumps(common), flush=True)
            for spec in candidates(m, n, k, mx, fp32):
                row = dict(**common, spec=spec)
                try:

                    def run(w, sw, spec=spec, qa=qa, sa=sa, x=x, pre=pre, fp32=fp32):
                        return call(
                            spec, qa if pre else x, w, sw, sa if pre else None, fp32
                        )

                    y = run(w, sw)
                    err = error(y, ref)
                    row.update(err)
                    assert err["finite"] and err["rmse"] < (1e-5 if fp32 else 0.001), (
                        err
                    )
                    row.update(
                        measure(
                            [lambda pair=pair, fn=run: fn(*pair) for pair in weights]
                        )
                    )
                    row["gain"] = 1 - row["us"] / bt["us"]
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {str(exc)[-1400:]}"
                rows.append(row)
                print(json.dumps(row), flush=True)
                (ROOT / args.output).write_text(json.dumps(rows, indent=2))
            del ref, x, qa, sa
        del weights, w, sw, tensors
        gc.collect()
        torch.accelerator.empty_cache()
    print("SCREEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
