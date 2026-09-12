# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Control: retain measured tiny-M winners; use original paths above M=4."""

import json
import os
from pathlib import Path

from skinny_worker import SkinnyWorker


class LowMWorker(SkinnyWorker):
    def load_model(self, *, load_dummy_weights=False):
        import dispatch

        dispatch.PLAN = {
            group: {m: entry for m, entry in cells.items() if int(m) <= 4}
            for group, cells in dispatch.PLAN.items()
        }
        super().load_model(load_dummy_weights=load_dummy_weights)
        output = Path(os.environ["DSV41_GEMM_OUTPUT"])
        path = output / f"runtime-rank{self.rank}.json"
        data = json.loads(path.read_text())
        data["variant"] = "max_m4"
        data["max_skinny_m"] = 4
        path.write_text(json.dumps(data, indent=2))
        if self.rank == 0:
            (output / "selected.json").write_text(json.dumps(dispatch.PLAN, indent=2))
