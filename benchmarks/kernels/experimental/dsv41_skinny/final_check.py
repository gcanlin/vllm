# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read-only runtime/source receipt after the experiment's own jobs stop."""

import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

import vllm

ROOT = Path(__file__).parent
REPO = ROOT.parent / "vllm"


def output(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    source_hash = hashlib.sha256(
        subprocess.check_output(["git", "diff", "--binary"], cwd=REPO)
    ).hexdigest()
    assert (
        source_hash
        == "38e9651feeb59ecb7884ccc579d3e576b4a350342185a2b1ec1d7bf045f0b420"
    )
    gpu = output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    assert len(gpu) == 8
    assert all(int(row.split(",")[1].strip()) == 0 for row in gpu), gpu
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 8032))
    report = dict(
        time_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        cwd=os.getcwd(),
        python=sys.executable,
        venv=os.environ.get("VIRTUAL_ENV"),
        vllm=vllm.__file__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        native_build=os.environ.get("VLLM_BUILD_COMMIT"),
        source_head=output(["git", "rev-parse", "HEAD"]),
        source_diff_sha256=source_hash,
        port_8032_free=True,
        gpus=gpu,
    )
    assert report["hostname"] == "xb01-gpu-200b-0030"
    assert report["cwd"] == str(REPO)
    assert report["venv"] == "/tmp/dsv41-index-reuse-venv"
    (ROOT / "final-receipt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("FINAL_CHECK_COMPLETE")


if __name__ == "__main__":
    main()
