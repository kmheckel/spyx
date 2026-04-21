"""End-to-end smoke test for a local Spyx install.

Run this once after ``uv sync`` (or ``pip install -e .``) on your laptop to
confirm Spyx, its optional extras, and the accelerator backend are all wired
up correctly. It performs seven checks:

  1. Python / JAX versions and visible devices (flags GPU / TPU / CPU-only).
  2. Core Spyx imports.
  3. Build a small SNN and run a forward pass.
  4. One full ``spyx.optimize.fit`` training epoch on synthetic data.
  5. NIR export + re-import round-trip.
  6. Notebook smoke tests (scripts/smoke_notebook_apis.py).
  7. Optional-extra checks (tonic for loaders, qwix for quant).

Each check reports PASS / FAIL / SKIP with a short reason. Exit code is 0
if everything that *should* pass did; non-zero if a core check failed.

Typical usage:

.. code-block:: bash

    uv run python scripts/check_install.py

If you want the full matrix including qwix + tonic:

.. code-block:: bash

    uv sync --all-extras
    uv run python scripts/check_install.py

The whole script runs in well under a minute on CPU and a few seconds on GPU.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from typing import Callable


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _emit(status: str, name: str, detail: str = "") -> None:
    colour = {"PASS": GREEN, "FAIL": RED, "SKIP": YELLOW}.get(status, "")
    print(f"  {colour}{status:4s}{RESET}  {name}" + (f"  -- {detail}" if detail else ""))


def _run_check(name: str, fn: Callable[[], str | None]) -> str:
    """Run a check. Returns 'PASS' / 'FAIL' / 'SKIP'."""
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
    except _Skip as skip:
        _emit("SKIP", name, str(skip))
        return "SKIP"
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _emit("FAIL", name, f"{type(exc).__name__}: {exc}  ({elapsed:.2f}s)")
        traceback.print_exc(limit=5)
        return "FAIL"
    elapsed = time.perf_counter() - t0
    _emit("PASS", name, f"{detail}  ({elapsed:.2f}s)".strip())
    return "PASS"


class _Skip(Exception):
    """Raised inside a check to indicate SKIP rather than FAIL."""


# --- Checks --------------------------------------------------------------


def check_environment() -> str:
    py = ".".join(str(v) for v in sys.version_info[:3])
    import jax

    devices = jax.devices()
    device_kinds = sorted({d.platform for d in devices})
    return (
        f"Python {py}, JAX {jax.__version__}, devices: "
        f"{[str(d) for d in devices]}  (platforms: {device_kinds})"
    )


def check_core_imports() -> str:
    import spyx  # noqa: F401
    import spyx.axn  # noqa: F401
    import spyx.data  # noqa: F401
    import spyx.fn  # noqa: F401
    import spyx.nir  # noqa: F401
    import spyx.nn  # noqa: F401
    import spyx.optimize  # noqa: F401
    import spyx.quant  # noqa: F401

    return f"spyx {spyx.__version__}"


def check_forward_pass() -> str:
    import jax.numpy as jnp
    from flax import nnx

    import spyx
    import spyx.nn as snn

    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(16, 32, use_bias=False, rngs=rngs),
        snn.LIF((32,), activation=spyx.axn.triangular(), rngs=rngs),
        nnx.Linear(32, 5, use_bias=False, rngs=rngs),
        snn.LI((5,), rngs=rngs),
    )
    T, B = 8, 4
    x = jnp.ones((T, B, 16))
    traces, _ = snn.run(model, x)
    assert traces.shape == (T, B, 5), traces.shape
    return f"Sequential forward OK, shape={traces.shape}"


def check_training_loop() -> str:
    """One spyx.optimize.fit epoch on synthetic spike-train data."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    import spyx
    import spyx.nn as snn
    import spyx.optimize as opt

    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(8, 16, use_bias=False, rngs=rngs),
        snn.LIF((16,), activation=spyx.axn.triangular(), rngs=rngs),
        nnx.Linear(16, 3, use_bias=False, rngs=rngs),
        snn.LI((3,), rngs=rngs),
    )

    T, B = 16, 4
    Loss = spyx.fn.integral_crossentropy()

    def forward(m, x_BTC):
        x_TBC = jnp.transpose(x_BTC, (1, 0, 2))
        traces, _ = snn.run(m, x_TBC)
        return jnp.transpose(traces, (1, 0, 2))

    def loss_fn(m, x, y):
        return Loss(forward(m, x), y)

    rng = jax.random.PRNGKey(0)

    def train_iter():
        for _ in range(3):  # 3 batches / epoch
            k_x, k_y = jax.random.split(jax.random.fold_in(rng, _))
            yield (
                jax.random.normal(k_x, (B, T, 8)),
                jax.random.randint(k_y, (B,), 0, 3),
            )

    history = opt.fit(model, optax.adam(3e-3), loss_fn, train_iter, epochs=2)
    assert len(history) == 2, history
    return f"fit ran 2 epochs, final train_loss={history[-1]['train_loss']:.3f}"


def check_nir_roundtrip() -> str:
    import jax.numpy as jnp
    from flax import nnx

    import spyx.nir as snir
    import spyx.nn as snn

    rngs = nnx.Rngs(0)
    model = snn.Sequential(
        nnx.Linear(6, 10, use_bias=False, rngs=rngs),
        snn.LIF((10,), beta=0.8, rngs=rngs),
    )
    nir_graph = snir.to_nir(model, input_shape={"input": (6,)}, output_shape={"output": (10,)})
    restored = snir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(1))
    assert jnp.allclose(model.layers[0].kernel[...], restored.layers[0].kernel[...])
    return "LIF roundtrip equal"


def check_notebook_apis() -> str:
    """Delegate to scripts/smoke_notebook_apis.py (already tests all tutorials)."""
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parent.parent
    smoke = root / "scripts" / "smoke_notebook_apis.py"
    if not smoke.exists():
        raise _Skip("scripts/smoke_notebook_apis.py not found")
    result = subprocess.run(
        [sys.executable, str(smoke)],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=180,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"smoke_notebook_apis.py exited with {result.returncode}")
    last = [line for line in result.stdout.splitlines() if line.strip()][-1]
    return last.strip()


def check_extras() -> str:
    """Report whether optional extras are available; always PASS (no extras required)."""
    have_tonic = importlib.util.find_spec("tonic") is not None
    have_qwix = importlib.util.find_spec("qwix") is not None
    have_grain = importlib.util.find_spec("grain") is not None
    parts = [
        f"grain={'yes' if have_grain else 'NO'}",
        f"tonic={'yes' if have_tonic else 'no'}",
        f"qwix={'yes' if have_qwix else 'no'}",
    ]
    if not have_grain:
        raise RuntimeError("grain is a core dependency; install is broken")
    return ", ".join(parts)


# --- Main ----------------------------------------------------------------


def main() -> int:
    print("Spyx install check\n==================")
    checks: list[tuple[str, Callable[[], str]]] = [
        ("environment", check_environment),
        ("core imports", check_core_imports),
        ("forward pass", check_forward_pass),
        ("training loop", check_training_loop),
        ("NIR roundtrip", check_nir_roundtrip),
        ("notebook APIs", check_notebook_apis),
        ("optional extras", check_extras),
    ]
    results = {name: _run_check(name, fn) for name, fn in checks}

    print()
    passed = sum(v == "PASS" for v in results.values())
    failed = sum(v == "FAIL" for v in results.values())
    skipped = sum(v == "SKIP" for v in results.values())
    total = len(results)
    summary = f"{passed}/{total} passed"
    if skipped:
        summary += f", {skipped} skipped"
    if failed:
        summary += f", {RED}{failed} FAILED{RESET}"
    print(summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
