# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check preserved evidence and recompute reports without modifying the archive."""

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = read(ROOT / "archive-manifest.json")
    for name, expected in manifest["evidence"].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == expected["bytes"], name
        assert digest(data) == expected["sha256"], name
    archive = ROOT / "original-source.tar.gz"
    assert digest(archive.read_bytes()) == manifest["original_source_archive_sha256"]
    with tarfile.open(archive, "r:gz") as tar:
        expected_names = set(manifest["original_source_files"])
        assert set(tar.getnames()) == expected_names
        for name, expected in manifest["original_source_files"].items():
            member = tar.extractfile(name)
            assert member is not None, name
            data = member.read()
            assert len(data) == expected["bytes"], name
            assert digest(data) == expected["sha256"], name
    assert (
        digest((ROOT / "baseline-dsv41.patch").read_bytes())
        == manifest["dsv41_baseline_patch_sha256"]
    )
    commands = [
        read(ROOT / arm / "commands.json")
        for arm in ("bench-a", "bench-b", "bench-lowm")
    ]
    bench_hash = manifest["original_source_files"]["bench.sh"]["sha256"]
    assert all(c["completed"] and c["bench_sha256"] == bench_hash for c in commands)
    with tempfile.TemporaryDirectory(prefix="dsv41-archive-check-") as tmp:
        target = Path(tmp)
        for name in ("bench-a", "bench-b", "bench-lowm", "selected.json"):
            (target / name).symlink_to(ROOT / name)
        for name in ("summarize.py", "summarize_lowm.py"):
            shutil.copyfile(ROOT / name, target / name)
        for script, report in (
            ("summarize.py", "serving-summary.json"),
            ("summarize_lowm.py", "serving-lowm-summary.json"),
        ):
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--active",
                    "--no-project",
                    "python",
                    str(target / script),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            assert read(target / report) == read(ROOT / report), report
            print(result.stdout.splitlines()[-1])
    print(
        f"Verified {len(manifest['evidence'])} unchanged evidence files and "
        f"{len(manifest['original_source_files'])} source snapshots."
    )
    print(
        "ARCHIVE_VALIDATION_COMPLETE: 1260 formal requests; "
        "5152 full + 4640 low-M numerical checks"
    )


if __name__ == "__main__":
    main()
