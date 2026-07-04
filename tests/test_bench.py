"""Tests for the spyx.bench benchmarking harness.

Kept intentionally tiny (small seq_len / batch / n_iters, fixed PRNG) so the
whole module runs in a couple of seconds on CPU and downloads nothing.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from spyx import bench, nn


def _tiny_model():
    rngs = nnx.Rngs(0)
    return nn.Sequential(
        nnx.Linear(8, 4, rngs=rngs),
        nn.LIF((4,), rngs=rngs),
    )


def test_benchmark_populates_fields():
    result = bench.benchmark(
        _tiny_model(),
        input_shape=(8,),
        seq_len=5,
        batch=2,
        n_warmup=1,
        n_iters=3,
        key=jax.random.PRNGKey(0),
    )

    assert isinstance(result, bench.BenchResult)
    assert result.name == "Sequential"
    assert result.seq_len == 5
    assert result.batch == 2

    # Efficiency numbers must be real, positive latencies.
    assert result.param_count > 0
    assert result.fwd_latency_ms > 0
    assert result.fwd_bwd_latency_ms is not None
    assert result.fwd_bwd_latency_ms > 0
    assert result.throughput_elem_ts_per_s > 0

    # Spike rate is a fraction in [0, 1].
    assert 0.0 <= result.spike_rate <= 1.0

    # Best-effort fields are either populated or explicitly None.
    assert result.peak_mem_mb is None or result.peak_mem_mb > 0
    assert result.flops is None or result.flops > 0
    assert result.mfu is None or result.mfu > 0


def test_benchmark_run_fn_returning_tuple():
    """An explicit run_fn may return (outputs, state) — spyx.nn.run itself does.

    The harness must normalise that to outputs before computing spike_rate;
    a live PSU_LIF-vs-LIF comparison caught this (spikes was a (out, state)
    tuple, so `spikes != 0` was a Python bool).
    """
    result = bench.benchmark(
        nn.LIF((4,), rngs=nnx.Rngs(0)),
        input_shape=(4,),
        seq_len=5,
        batch=2,
        n_warmup=1,
        n_iters=3,
        run_fn=nn.run,  # returns (outputs, final_state)
        name="LIF-via-run",
        key=jax.random.PRNGKey(0),
    )
    assert result.name == "LIF-via-run"
    assert result.fwd_latency_ms > 0
    assert 0.0 <= result.spike_rate <= 1.0


def test_benchmark_thunk_and_no_backward():
    result = bench.benchmark(
        _tiny_model,  # thunk, not an instance
        input_shape=(8,),
        seq_len=4,
        batch=2,
        n_warmup=1,
        n_iters=2,
        backward=False,
        name="thunked",
    )
    assert result.name == "thunked"
    assert result.fwd_bwd_latency_ms is None
    assert result.fwd_latency_ms > 0


def test_flops_available_on_cpu():
    # XLA's cost model exposes flops on CPU; guard the plumbing works end to end.
    result = bench.benchmark(
        _tiny_model(),
        input_shape=(8,),
        seq_len=4,
        batch=2,
        n_warmup=1,
        n_iters=2,
    )
    assert result.flops is not None
    assert result.flops > 0


def test_custom_run_fn_is_used():
    called = {"hit": False}

    def my_run(module, x):
        called["hit"] = True
        outputs, _ = nn.run(module, x)
        return outputs

    result = bench.benchmark(
        _tiny_model(),
        input_shape=(8,),
        seq_len=3,
        batch=2,
        n_warmup=1,
        n_iters=2,
        run_fn=my_run,
    )
    assert called["hit"]
    assert result.fwd_latency_ms > 0


def test_compare_and_format_table():
    modules = {
        "linear_lif": _tiny_model,
        "just_lif": lambda: nn.Sequential(
            nnx.Linear(8, 4, rngs=nnx.Rngs(1)),
            nn.LIF((4,), rngs=nnx.Rngs(1)),
        ),
    }
    results = bench.compare(
        modules,
        input_shape=(8,),
        seq_lens=[3, 5],
        batch=2,
        n_warmup=1,
        n_iters=2,
    )

    # 2 seq_lens x 2 modules == 4 results.
    assert len(results) == 4
    assert all(isinstance(r, bench.BenchResult) for r in results)
    assert {r.seq_len for r in results} == {3, 5}
    assert {r.name for r in results} == {"linear_lif", "just_lif"}

    table = bench.format_table(results)
    assert isinstance(table, str)
    assert "name" in table
    assert "spike_rate" in table
    assert "linear_lif" in table
    # Header + separator + one line per result.
    assert len(table.splitlines()) == 2 + len(results)


def test_format_table_empty():
    assert bench.format_table([]) == "(no results)"


def test_result_as_dict_roundtrip():
    result = bench.benchmark(
        _tiny_model(),
        input_shape=(8,),
        seq_len=3,
        batch=2,
        n_warmup=1,
        n_iters=2,
    )
    d = result.as_dict()
    assert d["name"] == "Sequential"
    assert d["param_count"] == result.param_count
    assert set(d) >= {
        "name",
        "device",
        "seq_len",
        "batch",
        "param_count",
        "fwd_latency_ms",
        "fwd_bwd_latency_ms",
        "throughput_elem_ts_per_s",
        "spike_rate",
        "peak_mem_mb",
        "flops",
        "mfu",
    }


def test_deterministic_spike_rate():
    kw = dict(input_shape=(8,), seq_len=5, batch=2, n_warmup=1, n_iters=2)
    r1 = bench.benchmark(_tiny_model(), key=jax.random.PRNGKey(42), **kw)
    r2 = bench.benchmark(_tiny_model(), key=jax.random.PRNGKey(42), **kw)
    assert jnp.isclose(r1.spike_rate, r2.spike_rate)
