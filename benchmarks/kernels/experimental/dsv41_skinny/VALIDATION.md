# Publication validation

Validated on September 13, 2026 in node30's existing ordinary nightly container,
using the active `/tmp/dsv41-index-reuse-venv` environment and `uv`.

```bash
uv run --active --no-project python \
  benchmarks/kernels/experimental/dsv41_skinny/verify_archive.py
```

Result:

```text
SERVING_VALIDATION_COMPLETE
LOWM_SERVING_VALIDATION_COMPLETE
Verified 164 unchanged evidence files and 24 source snapshots.
ARCHIVE_VALIDATION_COMPLETE: 1260 formal requests; 5152 full + 4640 low-M numerical checks
```

The verifier reproduces both archived summaries from original client results and
rank-local receipts. It checks hashes for raw evidence, the original source
archive, and the baseline patch. `git apply --check` succeeds for that patch on
the published branch's base.

Additional publication checks passed:

- Both benchmark drivers import from a fresh scratch directory using
  `DSV41_SOURCE_ROOT`; all Python sources compile and shell launchers pass
  `bash -n`.
- Local Markdown links resolve, and embedded model diagrams match the separate
  Mermaid sources.
- Kernel and dispatch AST comparison against the archived source preserves
  arithmetic and dispatch; differences are import wrappers and equivalent
  accelerator API calls used by screening.
- `git diff --cached --check` passes. Raw logs and patch context retain their
  original whitespace, with scoped Git attributes to preserve exact bytes.
- `SKIP=actionlint,pip-compile uv run --active --no-project pre-commit run`
  passes applicable checks, including Ruff, typos, Markdown, shellcheck, SPDX,
  forbidden imports and accelerator APIs. Actionlint environment setup stalled
  and was interrupted; it and dependency compilation have no matching changed
  files in this package. The mypy hook excludes benchmark paths, so its pass is
  not a type-checking claim for this experiment.

Raw generated text/logs/patches are excluded from spelling correction in this
directory; source and explanatory documents are still checked. No model-derived
weight tensors, process ID files or Python caches are included.

Publication validation does not rerun the GPU campaign. The measured numerical
and serving results are preserved from the original experiment. No GSM8K result
is claimed for this candidate.
