# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import requests

ROOT = Path(__file__).parent.resolve()
REPO = Path(os.environ.get("DSV41_SOURCE_ROOT") or Path(__file__).resolve().parents[4])
BENCH = ROOT / "bench.sh"
URL = "http://127.0.0.1:8032"
WORKLOADS = [
    ("1k256-c1", 1024, 256, 1, 8),
    ("1k256-c4", 1024, 256, 4, 32),
    ("8k1k-c16", 8192, 1024, 16, 100),
]


def wait_free(timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", 8032))
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise
        time.sleep(1)


def run(label, mode):
    output = ROOT / label
    output.mkdir()
    if mode == "candidate":
        (output / "selected.json").write_bytes((ROOT / "selected.json").read_bytes())
    wait_free()
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        PYTHONPATH=str(REPO) + os.pathsep + str(ROOT),
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_ENGINE_READY_TIMEOUT_S="3600",
        GLOO_SOCKET_IFNAME="bond0",
        NCCL_SOCKET_IFNAME="bond0",
        NCCL_DEBUG="WARN",
        VLLM_DSV41_REUSE_SPARSE_INDICES="0",
        VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION="0",
        DSV41_GEMM_MODE=mode,
        DSV41_GEMM_OUTPUT=str(output),
        RESULT_DIR=str(output),
        BASE_URL=URL,
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    command = ["bash", str(ROOT / "serve-lowm.sh")]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = dict(
        label=label,
        mode=mode,
        command=command,
        started=time.time(),
        native_build=env.get("VLLM_BUILD_COMMIT"),
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        source_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"], cwd=REPO)
        ).hexdigest(),
        scripts={p.name: sha(p) for p in ROOT.glob("*.py")}
        | {p.name: sha(p) for p in ROOT.glob("*.sh")},
        bench_sha256=sha(BENCH),
        workloads=WORKLOADS,
        clients=[],
    )

    def save():
        (output / "commands.json").write_text(json.dumps(record, indent=2))

    save()
    session = requests.Session()
    session.trust_env = False

    def client(name, inp, out, conc, count):
        cmd = [
            "bash",
            str(BENCH),
            str(inp),
            str(out),
            str(conc),
            str(count),
            name + ".json",
        ]
        print("BENCH_START", label, name, flush=True)
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(
                cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        data = json.loads((output / (name + ".json")).read_text())
        assert data["completed"] == count and data["failed"] == 0
        assert data["total_output_tokens"] == count * out and not any(data["errors"])
        assert all(x == inp for x in data["input_lens"]) and all(
            x == out for x in data["output_lens"]
        )
        row = dict(
            name=name,
            command=cmd,
            throughput=data["output_throughput"],
            tpot=data["mean_tpot_ms"],
            ttft=data["mean_ttft_ms"],
        )
        record["clients"].append(row)
        save()
        print("BENCH_DONE", label, json.dumps(row), flush=True)

    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (output / "server.pid").write_text(str(server.pid))
        print("SERVER_START", label, server.pid, flush=True)
        try:
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}")
                try:
                    if session.get(URL + "/health", timeout=3).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("server readiness")
            states = [
                json.loads(p.read_text()) for p in output.glob("runtime-rank*.json")
            ]
            assert {r["rank"] for r in states} == set(range(8))
            assert all(
                r["sp"]
                and r["fused_shared_layers"] == 40
                and r["mode"] == mode
                and r["installed"] == (mode == "candidate")
                for r in states
            )
            print("SERVER_READY", label, flush=True)
            # Same short greedy prompts in both arms, only a smoke comparison.
            texts = []
            for prompt in [
                "The capital of France is",
                "Question: What is 12 times 13? Answer:",
                "请用一句话解释什么是光合作用。",
                "def fibonacci(n):\n",
            ]:
                response = session.post(
                    URL + "/v1/completions",
                    json=dict(
                        model="dsv41-bench",
                        prompt=prompt,
                        max_tokens=32,
                        temperature=0,
                        seed=42,
                    ),
                    timeout=120,
                )
                response.raise_for_status()
                texts.append(dict(prompt=prompt, response=response.json()))
            (output / "smoke-completions.json").write_text(
                json.dumps(texts, indent=2, ensure_ascii=False)
            )
            client("warm-c64", 1024, 32, 64, 128)
            client("warm-c16", 8192, 32, 16, 32)
            for repeat in range(1, 4):
                for name, inp, out, c, n in WORKLOADS:
                    client(f"{name}-r{repeat}", inp, out, c, n)
            record["completed"] = True
        finally:
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=15)
            wait_free()
            session.close()
            record["ended"] = time.time()
            record["server_exit_code"] = server.returncode
            save()
            print("SERVER_STOPPED", label, server.returncode, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    for item in a.runs:
        label, mode = item.split(":")
        assert mode in ("baseline", "candidate")
        run(label, mode)
    print("GEMM_CAMPAIGN_COMPLETE", flush=True)
