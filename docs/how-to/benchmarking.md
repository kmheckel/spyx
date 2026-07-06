# How to benchmark neurons and models

Use [`spyx.bench`](../reference/bench.md) to measure the latency, throughput,
memory, and spiking activity of any module that follows the Spyx stepwise
contract (`(x_t, state) -> (out, state)` plus `initial_state`). It is the tool
behind the [parallel-spiking-neuron](../explanation/parallel-spiking-neurons.md)
crossover numbers, and the recommended way to fill in a
[research study](../explanation/research.md)'s results table so runs are comparable.

## Benchmark one module

`benchmark(module, input_shape, *, seq_len, batch, ...)` builds a random
time-major `(seq_len, batch, *input_shape)` input, drives the module over time,
and returns a `BenchResult`:

```python
import spyx
from flax import nnx

neuron = spyx.nn.LIF((256,), rngs=nnx.Rngs(0))
result = spyx.bench.benchmark(neuron, (256,), seq_len=512, batch=32)

print(result.fwd_latency_ms, result.throughput_elem_ts_per_s, result.spike_rate)
```

`input_shape` is the per-timestep feature shape (everything after the batch
axis). Pass a **zero-arg thunk** instead of a module when you want a fresh
instance built lazily (handy for sweeps):

```python
result = spyx.bench.benchmark(
    lambda: spyx.nn.LIF((256,), rngs=nnx.Rngs(0)),
    (256,), seq_len=512, batch=32,
)
```

By default the driver uses a module's own `parallel` method when it has one
(e.g. [`PSU_LIF`](../explanation/parallel-spiking-neurons.md)), otherwise it
falls back to `spyx.nn.run`. Override it explicitly with `run_fn=(module, x) ->
outputs` — for example, to force `PSU_LIF` down the **sequential** scan so you can
compare it against its own parallel path:

```python
seq_fn = lambda m, x: spyx.nn.run(m, x)[0]   # ignore the returned final state
result = spyx.bench.benchmark(
    lambda: spyx.nn.PSU_LIF((256,), rngs=nnx.Rngs(0)),
    (256,), seq_len=1024, batch=8, run_fn=seq_fn,
)
```

## What gets measured

| Field | Meaning |
|---|---|
| `fwd_latency_ms` | Median forward-pass latency (ms). |
| `fwd_bwd_latency_ms` | Median forward + backward (`value_and_grad` of `mean(outputs)`); `None` if `backward=False`. |
| `throughput_elem_ts_per_s` | `seq_len * batch / fwd_time` — element-timesteps per second. |
| `spike_rate` | Mean fraction of non-zero output activations — the SNN **energy proxy**. |
| `peak_mem_mb` | Peak device memory (`None` if the backend does not expose it). |
| `flops` | FLOPs from XLA's cost model (`None` if unavailable). |
| `mfu` | Model-FLOP-utilisation = achieved FLOP/s ÷ device peak (`None` when the device peak is unknown — never guessed). The peak is the **dense fp32** ceiling — see the int8 caveat below. |
| `param_count`, `device`, `seq_len`, `batch` | Run metadata. |

`spike_rate` is the standard event-driven energy proxy for spiking networks:
lower means sparser spiking, which on neuromorphic hardware means less energy.
It is why reset-free neurons — which fire more densely — cost something for their
parallelism (see the
[parallel-spiking-neuron explanation](../explanation/parallel-spiking-neurons.md)).

## The timing methodology (why the numbers are trustworthy)

`spyx.bench` is deliberate about *how* it times, because naïve JAX timing is
almost always wrong:

- **JIT first.** The timed function is `jax.jit`-compiled and the first
  `n_warmup` (default 3) iterations are discarded, so tracing and compilation are
  never counted.
- **Block before stopping the clock.** JAX dispatches asynchronously, so every
  timed call is followed by `jax.block_until_ready` on its outputs *before* the
  timer stops. Without this you would be timing dispatch, not computation.
- **Median, not mean.** The reported latency is the **median** over `n_iters`
  (default 20) iterations — far more robust to OS jitter and GC pauses than the
  mean.

## Sweep and compare

`compare(modules, input_shape, *, seq_lens, batch, ...)` sweeps a dict of
labelled modules across a list of sequence lengths and returns a flat list of
`BenchResult`. `format_table(results)` renders them as an aligned plain-text
table. This is the idiom for the parallel-vs-sequential crossover:

```python
import spyx
from flax import nnx

results = spyx.bench.compare(
    {
        "LIF (sequential)": lambda: spyx.nn.LIF((256,), rngs=nnx.Rngs(0)),
        "PSU_LIF (parallel)": lambda: spyx.nn.PSU_LIF((256,), rngs=nnx.Rngs(0)),
    },
    (256,),
    seq_lens=[128, 512, 2048],
    batch=8,
)
print(spyx.bench.format_table(results))
```

`LIF` has no `parallel` method so it is driven sequentially; `PSU_LIF` is driven
by its associative scan. Comparing the two rows at each `seq_len` shows the
crossover widening as the sequence grows and the sequential critical path starts
to dominate. Passing thunks (as above) gives every sweep point a fresh module.
`BenchResult.as_dict()` returns a plain dict if you would rather push results into
a DataFrame or log them.

!!! warning "int8 MFU is measured against the fp32 ceiling"
    The MFU peak-FLOPs table is the device's **dense fp32** peak. A quantized
    (int8/int4) model's MFU is therefore reported against the fp32 ceiling, not
    the (higher) integer tensor-core peak — so int8 and fp32 MFU numbers are
    **not directly comparable**. A quantized model can be faster (lower
    `fwd_latency_ms`) while showing a *lower* MFU here. Compare quantized models
    on latency / throughput, and treat MFU as an fp32-relative diagnostic. See
    [How to quantize a model](quantize.md) and the
    [deployment guide](deploy.md).

!!! note "Numbers are hardware-specific"
    Latency, throughput, and MFU depend entirely on the accelerator, driver, and
    JAX version. Always record them alongside your results — the reference
    machine for Spyx's parallel-neuron work is an **AMD Radeon 8060S (gfx1151)**
    on ROCm.
