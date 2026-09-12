# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full quantization/layout conversion costs included for native FP8 MMA."""

import json
import math
from pathlib import Path

import flashinfer
import torch
from cute_mx import gemm, pack_scale, quantize
from screen import error, measure

ROOT = Path(__file__).parent


def main():
    torch.manual_seed(9283)
    rows = []
    for group in (1, 2, 4, 10, 9):
        tensors = torch.load(ROOT / "weights" / f"group{group}.pt", weights_only=True)
        w = tensors["weight"].cuda().T
        sw = tensors["weight_scale"].cuda()
        n, k = w.shape
        psw = pack_scale(sw, n, k)
        grouped = psw.reshape(n // 32, 32, k // 128)
        assert torch.equal(grouped, grouped[:, :1].expand_as(grouped))
        pre = group in (2, 10)
        size = w.numel() + sw.numel()
        count = max(
            3, math.ceil(2.5 * torch.cuda.get_device_properties(0).L2_cache_size / size)
        )
        weights = [
            (w.clone(), sw.clone(), psw.clone(memory_format=torch.preserve_format))
            for _ in range(count)
        ]
        for m in (1, 2, 4, 8, 16):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            qa, sa = flashinfer.mxfp8_quantize(x, backend="cute-dsl")
            aq, sq = quantize(x)
            assert torch.equal(aq.view(torch.uint8), qa.view(torch.uint8))
            assert torch.equal(sq, pack_scale(sa, m, k))

            def baseline(w, sw, psw, x=x, qa=qa, sa=sa, pre=pre):
                a, s = (
                    (qa, sa)
                    if pre
                    else flashinfer.mxfp8_quantize(x, backend="cute-dsl")
                )
                return flashinfer.mm_mxfp8(a, w.T, s, sw, backend="cute-dsl")

            ref = baseline(*weights[0])
            base = measure([lambda v=v, fn=baseline: fn(*v) for v in weights])["us"]
            for tile_n, stages, dma in [
                (8, 2, 2),
                (16, 2, 2),
                (8, 3, 2),
                (16, 3, 2),
                (16, 4, 2),
                (8, 2, 4),
                (16, 2, 4),
                (16, 3, 4),
            ]:
                spec = dict(
                    kind="cute_mx", config=dict(tile_n=tile_n, stages=stages, dma=dma)
                )
                row = dict(
                    group=group,
                    m=m,
                    n=n,
                    k=k,
                    mx=True,
                    pre=pre,
                    fp32=False,
                    spec=spec,
                    baseline_us=base,
                    copies=count,
                )
                try:

                    def candidate(w, sw, psw, x=x, qa=qa, sa=sa, pre=pre, spec=spec):
                        return gemm(
                            qa if pre else x,
                            w,
                            psw,
                            sa if pre else None,
                            **spec["config"],
                        )

                    err = error(candidate(*weights[0]), ref)
                    assert err["finite"] and err["rmse"] < 0.001, err
                    row.update(err)
                    row.update(
                        measure([lambda v=v, fn=candidate: fn(*v) for v in weights])
                    )
                    row["gain"] = 1 - row["us"] / base
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {str(exc)[-1800:]}"
                rows.append(row)
                print(json.dumps(row), flush=True)
                (ROOT / "screen-cute-mx.json").write_text(json.dumps(rows, indent=2))
            # Do not keep all previous groups' clones alive.
        del weights, w, sw, psw, grouped, tensors
    print("CUTE_MX_SCREEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
