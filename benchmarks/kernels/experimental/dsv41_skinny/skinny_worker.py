# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Process-local dense-projection inventory and skinny experiment."""

import json
import os
from pathlib import Path

import torch

import vllm
from vllm.v1.worker.gpu_worker import Worker

ROOT = Path(__file__).parent


def inventory(model, rank):
    rows = []
    seen = {}
    dump = ROOT / "weights"
    if rank == 0:
        dump.mkdir(exist_ok=True)
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.ndim < 2:
            continue
        quant = getattr(module, "quant_method", None)
        kernel = getattr(quant, "kernel", None)
        row = dict(
            name=name,
            type=type(module).__name__,
            shape=list(w.shape),
            stride=list(w.stride()),
            dtype=str(w.dtype),
            quant=type(quant).__name__,
            kernel=type(kernel).__name__,
        )
        params = {k: p for k, p in module.named_parameters(recurse=False)}
        params.update(dict(module.named_buffers(recurse=False)))
        row["tensors"] = {
            k: dict(shape=list(p.shape), stride=list(p.stride()), dtype=str(p.dtype))
            for k, p in params.items()
        }
        key = json.dumps(
            [row[k] for k in ("type", "shape", "stride", "dtype", "quant", "kernel")]
        )
        if key not in seen:
            seen[key] = len(seen)
            # Representative loaded tensors only; expert banks and embeddings
            # are inventoried but are not ordinary dense GEMM candidates.
            if rank == 0 and w.ndim == 2 and "Embedding" not in type(module).__name__:
                tensors = {k: p.detach().cpu() for k, p in params.items()}
                torch.save(tensors, dump / f"group{seen[key]}.pt")
        row["group"] = seen[key]
        rows.append(row)
    (ROOT / f"inventory-rank{rank}.json").write_text(json.dumps(rows, indent=2))
    return rows


class SkinnyWorker(Worker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        if os.environ["DSV41_GEMM_MODE"] == "candidate":
            from dispatch import HITS

            output = Path(os.environ["DSV41_GEMM_OUTPUT"])
            report = [dict(group=g, m=m, calls=c) for (g, m), c in sorted(HITS.items())]
            (output / f"capture-hits-rank{self.rank}.json").write_text(
                json.dumps(report, indent=2)
            )
        return result

    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.model_runner.get_model().language_model
        mode = os.environ["DSV41_GEMM_MODE"]
        rows = inventory(model, self.rank) if mode == "baseline" else []
        details = {}
        if mode == "candidate":
            from dispatch import install

            details = install(model, self.rank)
        report = dict(
            rank=self.rank,
            mode=mode,
            source=vllm.__file__,
            sp=model.model.use_sequence_parallel,
            fused_shared_layers=sum(
                layer.ffn.experts.has_fused_shared_experts
                for layer in model.model.layers
            ),
            installed=mode == "candidate",
            inventory_count=len(rows),
            details=details,
        )
        output = Path(os.environ["DSV41_GEMM_OUTPUT"])
        (output / f"runtime-rank{self.rank}.json").write_text(
            json.dumps(report, indent=2)
        )
        print("SKINNY_RUNTIME", json.dumps(report), flush=True)
