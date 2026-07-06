"""Benchmarking harness for Spyx neurons and models.

This module measures both **efficiency** (latency, throughput, peak memory,
FLOPs, model-FLOP-utilisation) and a spiking-specific **performance** proxy
(spike rate) for any Spyx / Flax NNX module that follows the ``spyx.nn``
stateful contract.

Timing methodology (this is the load-bearing part):

* Inputs are built **time-major** with shape ``(seq_len, batch, *input_shape)``
  and the module is driven over time with :func:`spyx.nn.run` (a
  ``jax.lax.scan``), exactly like training.
* The timed function is **JIT-compiled** and the *first* ``n_warmup`` iterations
  are discarded so we never time tracing/compilation.
* Because JAX dispatches asynchronously, every timed call is followed by
  :func:`jax.block_until_ready` on its outputs **before** the timer is stopped.
  Without this the numbers are meaningless (you would only be timing dispatch).
* We report the **median** over ``n_iters`` iterations, which is far more robust
  to OS jitter / GC pauses than the mean.

FLOPs come from XLA's own cost model:
``jax.jit(f).lower(...).compile().cost_analysis()['flops']`` when the backend
exposes it (``None`` otherwise). MFU is ``flops_per_second / device_peak_flops``
using a small hard-coded peak-FLOPs table; when the device is unknown the peak
is ``None`` and MFU is reported as ``None`` rather than guessed. The spike rate
is the mean fraction of non-zero output activations, i.e. the standard
event-driven energy proxy for SNNs.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from flax import nnx

from . import nn

__all__ = [
    "BenchResult",
    "benchmark",
    "compare",
    "format_table",
]

# A module, or a zero-arg thunk that builds one. Thunks let ``compare`` create a
# fresh model per sweep point (and avoid building every model up front).
ModuleOrThunk = Union[nnx.Module, Callable[[], nnx.Module]]

# Best-effort peak *dense fp32* FLOP/s by device kind. Keys are matched as
# case-insensitive substrings of ``jax.Device.device_kind``. This is only used
# to turn measured FLOP/s into a Model-FLOP-Utilisation ratio; when no key
# matches, the peak is ``None`` and MFU is reported as ``None`` (never guessed).
# Add entries as needed — being absent only costs you the MFU column.
#
# NOTE: this ceiling is the *dense fp32* peak. A quantized (int8/int4) model's
# MFU is therefore reported against the fp32 ceiling, not the (higher) integer
# tensor-core peak, so int8 vs fp32 MFU numbers are NOT directly comparable —
# a quantized model can beat fp32 on latency while showing a *lower* MFU here.
_PEAK_FLOPS: dict[str, float] = {
    # NVIDIA data-center GPUs (fp32, non-tensor-core, approximate).
    "a100": 19.5e12,
    "h100": 67.0e12,
    "v100": 15.7e12,
    # AMD ROCm parts (fp32, approximate).
    "mi250": 45.3e12,
    "mi300": 61.3e12,
    "radeon": 10.0e12,
    "gfx1151": 5.0e12,  # Strix Halo iGPU, rough placeholder.
}


@dataclass
class BenchResult:
    """Container for a single benchmark measurement.

    All latency fields are the **median** over the timed iterations. Fields that
    could not be determined on the current backend are ``None`` rather than a
    fabricated value.
    """

    name: str
    device: str
    seq_len: int
    batch: int
    param_count: int
    fwd_latency_ms: float
    fwd_bwd_latency_ms: Optional[float]
    # (sample, timestep) pairs processed per second, i.e. seq_len * batch / fwd
    # time. Includes the batch factor, so it is element-timesteps/s rather than
    # pure timesteps/s; comparable across configs that share a batch size.
    throughput_elem_ts_per_s: float
    spike_rate: float
    peak_mem_mb: Optional[float] = None
    flops: Optional[float] = None
    mfu: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` view (handy for logging / DataFrames)."""
        return asdict(self)


def _resolve_module(module: ModuleOrThunk) -> nnx.Module:
    """Return an ``nnx.Module``, calling ``module`` if it is a build thunk."""
    if isinstance(module, nnx.Module):
        return module
    if callable(module):
        built = module()
        if not isinstance(built, nnx.Module):
            raise TypeError(
                f"benchmark thunk must return an nnx.Module, got {type(built).__name__}"
            )
        return built
    raise TypeError(
        f"expected an nnx.Module or a zero-arg thunk, got {type(module).__name__}"
    )


def _resolve_run_fn(
    module: nnx.Module, run_fn: Optional[Callable[[nnx.Module, jax.Array], Any]]
) -> Callable[[nnx.Module, jax.Array], jax.Array]:
    """Pick how the module is driven over time.

    Priority: an explicit ``run_fn`` > a module's own ``parallel`` method
    (a time-parallel scan alternative) > the default :func:`spyx.nn.run`.
    Every choice is normalised to ``(module, x) -> outputs`` (outputs only).
    """
    if run_fn is not None:
        # Tolerate drivers that return (outputs, state) — notably spyx.nn.run
        # itself, the canonical time driver — not just outputs.
        def _explicit_run(m: nnx.Module, x: jax.Array) -> jax.Array:
            out = run_fn(m, x)
            return out[0] if isinstance(out, tuple) else out

        return _explicit_run

    parallel = getattr(module, "parallel", None)
    if callable(parallel):

        def _parallel_run(m: nnx.Module, x: jax.Array) -> jax.Array:
            out = getattr(m, "parallel")(x)  # noqa: B009 — dynamic, PSU-style neurons only
            return out[0] if isinstance(out, tuple) else out

        return _parallel_run

    def _default_run(m: nnx.Module, x: jax.Array) -> jax.Array:
        outputs, _ = nn.run(m, x)
        return outputs

    return _default_run


def _device_name(device: jax.Device) -> str:
    return f"{device.platform}:{device.device_kind}"


def _peak_flops_for(device: jax.Device) -> Optional[float]:
    kind = device.device_kind.lower()
    for key, peak in _PEAK_FLOPS.items():
        if key in kind:
            return peak
    return None


def _param_count(module: nnx.Module) -> int:
    params = nnx.state(module, nnx.Param)
    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


def _peak_mem_mb(device: jax.Device) -> Optional[float]:
    """Best-effort peak-memory reading in MB.

    Prefers the backend's own allocator stats (``device.memory_stats()``); on
    backends that don't expose them (e.g. CPU) it falls back to summing the
    bytes of all currently-live JAX arrays. Returns ``None`` if neither works.
    """
    try:
        stats = device.memory_stats()
    except Exception:
        stats = None
    if stats:
        for key in ("peak_bytes_in_use", "bytes_in_use", "peak_bytes"):
            if key in stats and stats[key] is not None:
                return float(stats[key]) / (1024.0 * 1024.0)
    try:
        live = jax.live_arrays()
        if live:
            return float(sum(a.nbytes for a in live)) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def _cost_flops(lowered_compiled: Any) -> Optional[float]:
    """Extract total FLOPs from a compiled executable's cost analysis."""
    try:
        cost = lowered_compiled.cost_analysis()
    except Exception:
        return None
    if isinstance(cost, (list, tuple)):
        cost = cost[0] if cost else None
    if isinstance(cost, dict) and cost.get("flops") is not None:
        flops = float(cost["flops"])
        return flops if flops > 0 else None
    return None


def _time_median(
    fn: Callable[..., Any], args: tuple, n_warmup: int, n_iters: int
) -> tuple[float, Any]:
    """Median wall-clock latency (ms) of ``fn(*args)`` with async guarded out.

    Discards ``n_warmup`` iterations (tracing/compilation/first-run effects) and
    blocks on the result of every timed call before stopping the clock.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")
    out = None
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    times_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times_ms), out


def benchmark(
    module: ModuleOrThunk,
    input_shape: tuple[int, ...],
    *,
    seq_len: int,
    batch: int,
    n_warmup: int = 3,
    n_iters: int = 20,
    backward: bool = True,
    run_fn: Optional[Callable[[nnx.Module, jax.Array], Any]] = None,
    name: Optional[str] = None,
    key: Optional[jax.Array] = None,
    dtype: Any = jnp.float32,
) -> BenchResult:
    """Benchmark a single Spyx module / neuron.

    :param module: an ``nnx.Module`` or a zero-arg thunk returning one.
    :param input_shape: per-timestep feature shape (everything after ``batch``).
    :param seq_len: number of timesteps ``T`` in the time-major input.
    :param batch: batch size ``B``.
    :param n_warmup: untimed warmup iterations (compilation + first-run).
    :param n_iters: timed iterations; the **median** is reported.
    :param backward: also time a ``value_and_grad`` of ``mean(outputs)``.
    :param run_fn: optional ``(module, x) -> outputs`` override; defaults to a
        module ``parallel`` method if present, else :func:`spyx.nn.run`.
    :param name: label for the result; defaults to the module class name.
    :param key: PRNG key for the random input (defaults to a fixed seed so
        results are deterministic).
    :param dtype: dtype of the generated input.
    :return: a populated :class:`BenchResult`.
    """
    module = _resolve_module(module)
    if key is None:
        key = jax.random.PRNGKey(0)
    if name is None:
        name = type(module).__name__

    device = jax.devices()[0]
    x = jax.random.normal(key, (seq_len, batch) + tuple(input_shape), dtype=dtype)

    driver = _resolve_run_fn(module, run_fn)
    # Split into differentiable params vs. everything else so backward can take a
    # gradient purely w.r.t. Params while the rest is threaded through untouched.
    graphdef, params, rest = nnx.split(module, nnx.Param, ...)

    def _fwd(params, x):
        m = nnx.merge(graphdef, params, rest)
        return driver(m, x)

    fwd_jit = jax.jit(_fwd)

    # FLOPs from XLA's cost model on the compiled forward executable.
    flops: Optional[float] = None
    try:
        compiled = fwd_jit.lower(params, x).compile()
        flops = _cost_flops(compiled)
    except Exception:
        flops = None

    fwd_ms, fwd_out = _time_median(fwd_jit, (params, x), n_warmup, n_iters)

    # Spike rate: mean fraction of non-zero output activations (energy proxy).
    spikes = jax.block_until_ready(fwd_out)
    spike_rate = float(jnp.mean((spikes != 0).astype(jnp.float32)))

    fwd_bwd_ms: Optional[float] = None
    if backward:

        def _loss(params, x):
            m = nnx.merge(graphdef, params, rest)
            outputs = driver(m, x)
            return jnp.mean(outputs.astype(jnp.float32))

        grad_jit = jax.jit(jax.value_and_grad(_loss))
        fwd_bwd_ms, _ = _time_median(grad_jit, (params, x), n_warmup, n_iters)

    peak_mem_mb = _peak_mem_mb(device)

    fwd_s = fwd_ms / 1000.0
    throughput = (seq_len * batch) / fwd_s if fwd_s > 0 else float("inf")

    peak_flops = _peak_flops_for(device)
    mfu: Optional[float] = None
    if flops is not None and peak_flops is not None and fwd_s > 0:
        mfu = (flops / fwd_s) / peak_flops

    return BenchResult(
        name=name,
        device=_device_name(device),
        seq_len=seq_len,
        batch=batch,
        param_count=_param_count(module),
        fwd_latency_ms=fwd_ms,
        fwd_bwd_latency_ms=fwd_bwd_ms,
        throughput_elem_ts_per_s=throughput,
        spike_rate=spike_rate,
        peak_mem_mb=peak_mem_mb,
        flops=flops,
        mfu=mfu,
    )


def compare(
    modules: dict[str, ModuleOrThunk],
    input_shape: tuple[int, ...],
    *,
    seq_lens: list[int],
    batch: int,
    n_warmup: int = 3,
    n_iters: int = 20,
    backward: bool = True,
    run_fn: Optional[Callable[[nnx.Module, jax.Array], Any]] = None,
    key: Optional[jax.Array] = None,
    dtype: Any = jnp.float32,
) -> list[BenchResult]:
    """Sweep ``seq_lens`` x ``modules`` and return one result per combination.

    Passing **thunks** (zero-arg builders) as the dict values is recommended so
    each sweep point gets a fresh module instance. The results are ordered
    seq_len-outer, module-inner.

    :param modules: mapping of label -> module or thunk.
    :param input_shape: per-timestep feature shape.
    :param seq_lens: list of sequence lengths to sweep.
    :param batch: batch size shared across the sweep.
    :return: flat list of :class:`BenchResult`.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    results: list[BenchResult] = []
    for seq_len in seq_lens:
        for label, module in modules.items():
            results.append(
                benchmark(
                    module,
                    input_shape,
                    seq_len=seq_len,
                    batch=batch,
                    n_warmup=n_warmup,
                    n_iters=n_iters,
                    backward=backward,
                    run_fn=run_fn,
                    name=label,
                    key=key,
                    dtype=dtype,
                )
            )
    return results


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value == 0.0:
            return "0"
        if abs(value) >= 1e6 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.3f}"
    return str(value)


def format_table(results: list[BenchResult]) -> str:
    """Pretty-print benchmark results as an aligned plain-text table."""
    columns: list[tuple[str, Callable[[BenchResult], Any]]] = [
        ("name", lambda r: r.name),
        ("device", lambda r: r.device),
        ("seq", lambda r: r.seq_len),
        ("batch", lambda r: r.batch),
        ("params", lambda r: r.param_count),
        ("fwd_ms", lambda r: r.fwd_latency_ms),
        ("fwd_bwd_ms", lambda r: r.fwd_bwd_latency_ms),
        ("elem_ts/s", lambda r: r.throughput_elem_ts_per_s),
        ("spike_rate", lambda r: r.spike_rate),
        ("mem_mb", lambda r: r.peak_mem_mb),
        ("flops", lambda r: r.flops),
        ("mfu", lambda r: r.mfu),
    ]

    if not results:
        return "(no results)"

    header = [name for name, _ in columns]
    rows = [[_fmt(getter(r)) for _, getter in columns] for r in results]

    widths = [
        max(len(header[i]), max(len(row[i]) for row in rows))
        for i in range(len(header))
    ]

    def _render(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [_render(header), _render(["-" * w for w in widths])]
    lines.extend(_render(row) for row in rows)
    return "\n".join(lines)
