"""Reset-preserving parallel LIF: numerical equivalence + throughput stub.

Two things are measured for
:class:`spyx.experimental.parallel_reset.ParallelResetLIF`:

1. **Numerical equivalence** -- the FPT ``.parallel`` scan reproduces the
   sequential hard-reset spike train (exactly at ``K = T``; the max mismatch at
   ``K = 3`` is reported so the accuracy/K trade-off is visible).
2. **Throughput** -- a :mod:`spyx.bench` comparison of the sequential
   ``spyx.nn.run`` path vs. the FPT ``.parallel`` path vs. the reset-free
   :class:`spyx.nn.PSU_LIF` ``.parallel`` path.

Modes:

* ``SPYX_SMOKE=1`` (default; used by CI/agents): a tiny synthetic CPU run --
  small ``T`` / batch / hidden, a couple of bench iters. Seconds, no dataset,
  no GPU. The throughput numbers here are **not** meaningful (CPU, tiny shapes);
  the smoke run only checks the code path executes and the equivalence holds.
* Full run (**human-gated**): set ``SPYX_SMOKE=0`` and run on a GPU with large
  ``T`` to actually observe the ``O(K log T)`` vs ``O(T)`` speedup. Findings for
  the throughput comparison are **PENDING** that run.

Usage::

    SPYX_SMOKE=1 uv run python research/new/reset_preserving_parallel_lif/run_bench.py
    # full (human-gated, GPU):
    JAX_PLATFORMS=rocm SPYX_SMOKE=0 \
      uv run python research/new/reset_preserving_parallel_lif/run_bench.py
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from spyx import bench, nn
from spyx.experimental.parallel_reset import ParallelResetLIF

SMOKE = os.environ.get("SPYX_SMOKE", "1") != "0"

if SMOKE:
    T, BATCH, HIDDEN = 32, 8, 32
    SEQ_LENS = [16, 32]
    N_WARMUP, N_ITERS = 1, 3
else:
    # Human-gated GPU run: long sequences are where O(K log T) beats O(T).
    T, BATCH, HIDDEN = 512, 64, 256
    SEQ_LENS = [128, 256, 512]
    N_WARMUP, N_ITERS = 3, 20

BETA, THRESHOLD, K = 0.5, 1.0, 3


def _sequential_spikes(model, x):
    outputs, _ = nn.run(model, x)
    return outputs


def equivalence_check():
    """Max spike-train mismatch of the FPT scan vs. the sequential reference."""
    model = ParallelResetLIF(
        (HIDDEN,), beta=BETA, threshold=THRESHOLD, rngs=nnx.Rngs(0)
    )
    key = jax.random.PRNGKey(0)
    # Sparse-ish drive: the short-cascade regime where small K is near-exact.
    pulses = (jax.random.uniform(key, (T, BATCH, HIDDEN)) < 0.12).astype(jnp.float32)
    x = (
        pulses * 1.7
        + jax.random.normal(jax.random.PRNGKey(1), (T, BATCH, HIDDEN)) * 0.1
    )

    seq = _sequential_spikes(model, x)
    exact = model.parallel(x, K=T)  # K = T is exact in all regimes
    approx = model.parallel(x, K=K)

    return {
        "T": T,
        "K_exact": T,
        "K_approx": K,
        "max_mismatch_K_eq_T": float(jnp.max(jnp.abs(seq - exact))),
        "mean_mismatch_K_eq_T": float(jnp.mean(jnp.abs(seq - exact))),
        "max_mismatch_K3": float(jnp.max(jnp.abs(seq - approx))),
        "mean_mismatch_K3": float(jnp.mean(jnp.abs(seq - approx))),
    }


def throughput_check():
    """Bench the sequential vs. FPT-parallel vs. PSU_LIF-parallel paths."""

    def seq_reset():
        return ParallelResetLIF(
            (HIDDEN,), beta=BETA, threshold=THRESHOLD, rngs=nnx.Rngs(0)
        )

    def par_reset():
        return ParallelResetLIF(
            (HIDDEN,), beta=BETA, threshold=THRESHOLD, rngs=nnx.Rngs(0)
        )

    def psu():
        return nn.PSU_LIF((HIDDEN,), beta=BETA, threshold=THRESHOLD, rngs=nnx.Rngs(0))

    results = []
    for label, thunk, run_fn in [
        ("ParallelResetLIF/seq", seq_reset, lambda m, x: nn.run(m, x)[0]),
        ("ParallelResetLIF/fpt", par_reset, lambda m, x: m.parallel(x, K=K)),
        ("PSU_LIF/parallel", psu, lambda m, x: m.parallel(x)),
    ]:
        for seq_len in SEQ_LENS:
            results.append(
                bench.benchmark(
                    thunk,
                    (HIDDEN,),
                    seq_len=seq_len,
                    batch=BATCH,
                    n_warmup=N_WARMUP,
                    n_iters=N_ITERS,
                    run_fn=run_fn,
                    name=f"{label}[T={seq_len}]",
                )
            )
    return results


def main():
    equiv = equivalence_check()
    print("=== numerical equivalence (FPT vs sequential hard-reset LIF) ===")
    print(json.dumps(equiv, indent=2))

    results = throughput_check()
    print("\n=== throughput (spyx.bench) ===")
    if SMOKE:
        print("[SMOKE] CPU/tiny shapes -- latencies NOT meaningful; path-check only.")
    print(bench.format_table(results))

    out = {
        "smoke": SMOKE,
        "config": {"T": T, "batch": BATCH, "hidden": HIDDEN, "beta": BETA, "K": K},
        "equivalence": equiv,
        "throughput": [r.__dict__ for r in results],
    }
    dest = Path(__file__).with_name("bench_results.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
