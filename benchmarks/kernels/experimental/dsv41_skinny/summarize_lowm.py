# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate the M<=4 serving control against the same baseline workloads."""

import json
import statistics
from pathlib import Path

from summarize import METRICS, read

ROOT = Path(__file__).parent


def main():
    baseline = read(ROOT / "bench-a" / "commands.json")
    trial = read(ROOT / "bench-lowm" / "commands.json")
    assert baseline["completed"] and trial["completed"]
    for key in (
        "source_head",
        "source_diff_sha256",
        "native_build",
        "bench_sha256",
        "workloads",
    ):
        assert baseline[key] == trial[key], key
    results = {}
    for workload, inp, out, concurrency, count in baseline["workloads"]:
        arms = {}
        for arm in ("bench-a", "bench-lowm"):
            rows = [read(ROOT / arm / f"{workload}-r{i}.json") for i in range(1, 4)]
            for row in rows:
                assert row["completed"] == count and row["failed"] == 0
                assert row["total_input_tokens"] == count * inp
                assert row["total_output_tokens"] == count * out
                assert row["max_concurrency"] == concurrency
                assert (
                    row["input_lens"] == [inp] * count
                    and row["output_lens"] == [out] * count
                )
                assert len(row["errors"]) == count and not any(row["errors"])
            arms[arm] = {
                key: dict(
                    runs=[r[key] for r in rows],
                    median=statistics.median(r[key] for r in rows),
                )
                for key in METRICS
            }
        arms["candidate_change_pct"] = {
            key: 100
            * (arms["bench-lowm"][key]["median"] / arms["bench-a"][key]["median"] - 1)
            for key in METRICS
        }
        results[workload] = arms
    selected = read(ROOT / "bench-lowm" / "selected.json")
    full = read(ROOT / "selected.json")
    expected = {
        g: {m: cell for m, cell in data.items() if int(m) <= 4}
        for g, data in full.items()
    }
    assert selected == expected
    cells = {(int(g), int(m)) for g, data in selected.items() for m in data}
    assert len(cells) == 20 and all(m <= 4 for g, m in cells)
    checks = []
    runtime = []
    for rank in range(8):
        row = read(ROOT / "bench-lowm" / f"runtime-rank{rank}.json")
        assert row["rank"] == rank and row["sp"] and row["fused_shared_layers"] == 40
        assert row["variant"] == "max_m4" and row["max_skinny_m"] == 4
        assert row["details"]["checks"] == 580
        assert row["details"]["installed"] == {
            "1": 40,
            "3": 40,
            "4": 40,
            "11": 8,
            "12": 4,
            "13": 3,
            "15": 1,
            "16": 1,
        }
        runtime.append(row)
        checks.extend(read(ROOT / "bench-lowm" / f"quality-rank{rank}.json"))
        hits = read(ROOT / "bench-lowm" / f"capture-hits-rank{rank}.json")
        assert {(r["group"], r["m"]) for r in hits if r["calls"] > 0} == cells
    assert len(checks) == 4640
    assert all(
        r["finite"] and r["rmse"] < (1e-5 if r["group"] in (13, 15) else 0.001)
        for r in checks
    )
    smokes = [
        read(ROOT / arm / "smoke-completions.json") for arm in ("bench-a", "bench-lowm")
    ]
    assert len(smokes[0]) == len(smokes[1]) == 4
    assert [s["prompt"] for s in smokes[0]] == [s["prompt"] for s in smokes[1]]
    smoke = [
        dict(
            prompt=a["prompt"],
            text_equal=a["response"]["choices"][0]["text"]
            == b["response"]["choices"][0]["text"],
        )
        for a, b in zip(*smokes, strict=True)
    ]
    report = dict(
        protocol=(
            "One fresh baseline, full candidate, then M<=4 candidate; "
            "three client repetitions per server. "
            "Not independent server replications."
        ),
        workloads=results,
        runtime=runtime,
        selected=selected,
        loaded_weight_checks=dict(
            count=len(checks), max_rmse=max(r["rmse"] for r in checks)
        ),
        smoke=smoke,
    )
    (ROOT / "serving-lowm-summary.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                workloads=results, quality=report["loaded_weight_checks"], smoke=smoke
            ),
            indent=2,
        )
    )
    print("LOWM_SERVING_VALIDATION_COMPLETE")


if __name__ == "__main__":
    main()
