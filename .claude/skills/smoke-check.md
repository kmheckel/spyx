---
name: smoke-check
description: Run the full Spyx local health check. Use when the user asks "is my install working", after updating dependencies, on a new machine, or before assuming a test failure is a real bug.
---

# Local health check

Spyx ships three pre-built check scripts. Run them in order and report results concisely.

## Step 1 — `check_install.py` (30 seconds)

```bash
uv run python scripts/check_install.py
```

Seven checks with coloured PASS/FAIL/SKIP:

1. Python / JAX version + visible devices.
2. Core Spyx imports.
3. Forward pass through a small Sequential SNN.
4. One `spyx.optimize.fit` training epoch on synthetic data.
5. NIR export + re-import roundtrip.
6. `smoke_notebook_apis.py` passthrough (all 6 tutorials).
7. Optional-extra availability (`tonic`, `qwix`).

Exit code is 0 on success. If anything fails, the stack trace is printed directly; decide whether to triage or invoke `setup-gpu` / `debug-training` based on the failure.

## Step 2 — full test suite (if step 1 passed)

```bash
uv run pytest -v
```

Expected: 50+ passed, 0–2 skipped (loader tests skip if `tonic` isn't installed via `uv sync --extra loaders`).

If pytest fails on a test module the user has locally but isn't on any remote branch (e.g. `tests/test_sy_nmnist.py`), ask whether they want to commit, remove, or `--ignore` it before re-running.

## Step 3 — lint (fast sanity check)

```bash
uv run ruff check
```

Expected: "All checks passed!"

## Interpretation

- **All three green** → environment is solid. Good to run tutorials / real training.
- **Step 1 fails at environment check with `platforms: ['cpu']`** → GPU not wired up. Invoke `setup-gpu`.
- **Step 1 fails at training-loop or forward-pass** → something broken in the user's install beyond extras. Check `uv run python -c "import spyx; print(spyx.__version__)"` and the flax / grain versions; ensure they ran `uv sync` not `pip install`.
- **Step 1 passes but step 2 fails** → something broke in a specific module. Report which test and ask whether to triage.

Keep the report short. One line per check result, one sentence of interpretation. Don't paste full stack traces unless the user asks.
