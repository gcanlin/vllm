# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate the completed serving comparison and retain per-run metrics."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = (
    "output_throughput",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "mean_e2el_ms",
    "p95_e2el_ms",
    "p95_itl_ms",
)


def read(path):
    return json.loads(path.read_text())


def main():
    commands = [read(ROOT / arm / "commands.json") for arm in ("bench-a", "bench-b")]
    assert all(item.get("completed") for item in commands)
    for key in (
        "source_head",
        "source_diff_sha256",
        "native_build",
        "bench_sha256",
        "workloads",
    ):
        assert commands[0][key] == commands[1][key], key
    # The baseline was run while candidates were being developed. Retain all
    # experiment-script differences; vLLM source, workload, and backend match.
    scripts_a, scripts_b = (item["scripts"] for item in commands)
    script_differences = {
        key: [scripts_a.get(key), scripts_b.get(key)]
        for key in scripts_a.keys() | scripts_b.keys()
        if scripts_a.get(key) != scripts_b.get(key)
    }
    assert scripts_a["serve.sh"] == scripts_b["serve.sh"]
    results = {}
    for workload, inp, out, concurrency, count in commands[0]["workloads"]:
        arms = {}
        for arm in ("bench-a", "bench-b"):
            rows = [read(ROOT / arm / f"{workload}-r{i}.json") for i in range(1, 4)]
            for row in rows:
                assert row["completed"] == count and row["failed"] == 0
                assert row["total_input_tokens"] == count * inp
                assert row["total_output_tokens"] == count * out
                assert row["max_concurrency"] == concurrency
                assert row["input_lens"] == [inp] * count
                assert row["output_lens"] == [out] * count
                assert len(row["errors"]) == count and not any(row["errors"])
            arms[arm] = {
                key: {
                    "runs": [row[key] for row in rows],
                    "median": statistics.median(row[key] for row in rows),
                    "min": min(row[key] for row in rows),
                    "max": max(row[key] for row in rows),
                }
                for key in METRICS
            }
        arms["candidate_change_pct"] = {
            key: (arms["bench-b"][key]["median"] / arms["bench-a"][key]["median"] - 1)
            * 100
            for key in METRICS
        }
        results[workload] = arms
    runtimes = {}
    for arm, mode in (("bench-a", "baseline"), ("bench-b", "candidate")):
        rows = [read(ROOT / arm / f"runtime-rank{i}.json") for i in range(8)]
        assert {row["rank"] for row in rows} == set(range(8))
        for row in rows:
            assert row["sp"] and row["fused_shared_layers"] == 40
            assert row["mode"] == mode
            assert row["installed"] == (mode == "candidate")
            if mode == "candidate":
                expected = {
                    "1": 40,
                    "3": 40,
                    "4": 40,
                    "11": 8,
                    "12": 4,
                    "13": 3,
                    "15": 1,
                    "16": 1,
                }
                assert row["details"]["installed"] == expected
                assert row["details"]["checks"] == 644
        runtimes[arm] = rows
    smokes = [
        read(ROOT / arm / "smoke-completions.json") for arm in ("bench-a", "bench-b")
    ]
    smoke_result = []
    for a, b in zip(*smokes, strict=True):
        assert a["prompt"] == b["prompt"]
        ca, cb = (r["response"]["choices"][0] for r in (a, b))
        smoke_result.append(
            {
                "prompt": a["prompt"],
                "text_equal": ca["text"] == cb["text"],
                "usage_equal": a["response"]["usage"] == b["response"]["usage"],
                "finish_reason_equal": ca["finish_reason"] == cb["finish_reason"],
            }
        )
    result = {
        "protocol": (
            "Median of three client runs per fresh server; baseline then candidate. "
            "Not independent server replications."
        ),
        "workloads": results,
        "runtime": runtimes,
        "smoke": smoke_result,
        "source_head": commands[0]["source_head"],
        "source_diff_sha256": commands[0]["source_diff_sha256"],
        "experiment_script_differences": script_differences,
    }
    checks = [
        item
        for rank in range(8)
        for item in read(ROOT / "bench-b" / f"quality-rank{rank}.json")
    ]
    assert len(checks) == 5152
    assert all(
        row["finite"] and row["rmse"] < (1e-5 if row["group"] in (13, 15) else 0.001)
        for row in checks
    )
    result["loaded_weight_checks"] = dict(
        count=len(checks), max_rmse=max(row["rmse"] for row in checks)
    )
    hits = {
        str(rank): read(ROOT / "bench-b" / f"capture-hits-rank{rank}.json")
        for rank in range(8)
    }
    result["capture_hits"] = hits
    result["selected"] = read(ROOT / "selected.json")
    assert result["selected"] == read(ROOT / "bench-b" / "selected.json")
    expected_cells = {
        (int(g), int(m)) for g, cells in result["selected"].items() for m in cells
    }
    assert len(expected_cells) == 28
    for rank_hits in hits.values():
        assert {
            (r["group"], r["m"]) for r in rank_hits if r["calls"] > 0
        } == expected_cells
    (ROOT / "serving-summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    print(
        json.dumps(
            {"workloads": results, "smoke": smoke_result}, indent=2, ensure_ascii=False
        )
    )
    print("SERVING_VALIDATION_COMPLETE")


if __name__ == "__main__":
    main()
