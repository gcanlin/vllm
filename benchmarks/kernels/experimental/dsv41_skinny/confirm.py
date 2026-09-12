# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Confirm tuning winners with alternating baseline/candidate graph timings."""

import gc
import json
import statistics
from pathlib import Path

import flashinfer
import torch
from screen import call, error, measure

from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

ROOT = Path(__file__).parent


def main():
    rows = []
    for filename in ("screen-bf16.json", "screen-mx.json"):
        rows.extend(json.loads((ROOT / filename).read_text()))
    if (ROOT / "screen-cute-mx.json").exists():
        rows.extend(json.loads((ROOT / "screen-cute-mx.json").read_text()))
    candidates = {}
    for row in rows:
        if "error" in row or row["gain"] < 0.05:
            continue
        key = row["group"], row["m"]
        if key not in candidates or row["us"] < candidates[key]["us"]:
            candidates[key] = row
    woa_path = ROOT / "selected-woa.json"
    woa = json.loads(woa_path.read_text()) if woa_path.exists() else {}
    selected = {"3": woa} if woa else {}
    evidence = []
    for (group, m), row in sorted(candidates.items()):
        tensors = torch.load(ROOT / "weights" / f"group{group}.pt", weights_only=True)
        w = tensors["weight"].cuda()
        if row["mx"]:
            w = w.T
        sw = tensors["weight_scale"].cuda() if row["mx"] else None
        cute_mx = row["spec"]["kind"] == "cute_mx"
        if cute_mx:
            from cute_mx import pack_scale

            psw = pack_scale(sw, w.shape[0], w.shape[1])
        else:
            psw = None
        n, k = w.shape
        torch.manual_seed(8781 + group * 100 + m)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        qa, sa = (
            flashinfer.mxfp8_quantize(x, backend="cute-dsl")
            if row["mx"]
            else (None, None)
        )

        def baseline(w, sw, psw, row=row, qa=qa, sa=sa, x=x, group=group):
            if row["mx"]:
                a, s = (
                    (qa, sa)
                    if row["pre"]
                    else flashinfer.mxfp8_quantize(x, backend="cute-dsl")
                )
                return flashinfer.mm_mxfp8(a, w.T, s, sw, backend="cute-dsl")
            if group == 5:
                return ll_bf16_gemm(x, w)
            return (
                torch.mm(x, w.T, out_dtype=torch.float32)
                if row["fp32"]
                else torch.nn.functional.linear(x, w)
            )

        def candidate(w, sw, psw, row=row, qa=qa, sa=sa, x=x, cute_mx=cute_mx):
            return call(
                row["spec"],
                qa if row["pre"] else x,
                w,
                psw if cute_mx else sw,
                sa if row["pre"] else None,
                row["fp32"],
            )

        count = row["copies"]
        weights = [(w, sw, psw)] + [
            (
                w.clone(),
                sw.clone() if sw is not None else None,
                psw.clone(memory_format=torch.preserve_format)
                if psw is not None
                else None,
            )
            for _ in range(count - 1)
        ]
        timing = {"baseline": [], "candidate": []}
        for repeat in range(3):
            for label in (
                ("baseline", "candidate")
                if repeat % 2 == 0
                else ("candidate", "baseline")
            ):
                fn = baseline if label == "baseline" else candidate
                timing[label].append(
                    measure([lambda pair=pair, fn=fn: fn(*pair) for pair in weights])[
                        "us"
                    ]
                )
        # Changing graph inputs catch stale-output or graph-replay errors.
        expected = baseline(w, sw, psw)
        graph = torch.cuda.CUDAGraph()
        candidate(w, sw, psw)
        with torch.cuda.graph(graph):
            output = candidate(w, sw, psw)
        errors = []
        for scale in (0.0, 1e-5, 0.1, 1.0, 10.0, 100.0):
            x.normal_().mul_(scale)
            if row["mx"]:
                aq, sq = flashinfer.mxfp8_quantize(x, backend="cute-dsl")
                qa.copy_(aq)
                sa.copy_(sq)
            graph.replay()
            expected = baseline(w, sw, psw)
            err = error(output, expected)
            assert err["finite"] and err["rmse"] < (1e-5 if row["fp32"] else 0.001), (
                group,
                m,
                scale,
                err,
            )
            errors.append(dict(scale=scale, **err))
        base = statistics.median(timing["baseline"])
        trial = statistics.median(timing["candidate"])
        gain = 1 - trial / base
        kept = gain >= 0.05 and max(timing["candidate"]) < min(timing["baseline"])
        record = dict(
            group=group,
            m=m,
            spec=row["spec"],
            baseline_us=base,
            candidate_us=trial,
            gain=gain,
            kept=kept,
            timings=timing,
            errors=errors,
        )
        evidence.append(record)
        if kept:
            selected.setdefault(str(group), {})[str(m)] = record
        print(json.dumps(record), flush=True)
        (ROOT / "confirmed.json").write_text(json.dumps(evidence, indent=2))
        (ROOT / "selected.json").write_text(json.dumps(selected, indent=2))
        del weights, w, sw, tensors, x, qa, sa, graph, output, expected
        gc.collect()
        torch.accelerator.empty_cache()
    print("CONFIRM_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
