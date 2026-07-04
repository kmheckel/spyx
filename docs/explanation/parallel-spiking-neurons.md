# Parallel spiking neurons

Spiking neurons are usually run **one timestep at a time**. A standard
[`spyx.nn.LIF`](../reference/nn.md) keeps a membrane voltage \(V\) and, at every
step, leaks it, integrates the input, fires, and **resets** by subtracting the
spike:

$$
S_t = H(V_t - \theta), \qquad V_{t+1} = \beta V_t + x_t - S_t\,\theta .
$$

That `- S_t θ` term is the whole story. Because \(S_t\) is a (nonlinear) function
of \(V_t\), the update from \(V_t\) to \(V_{t+1}\) is nonlinear and every step
depends on the spike emitted by the step before it. There is no way around
walking the sequence in order: the recurrence is intrinsically **sequential**,
and Spyx evaluates it with a `jax.lax.scan` inside [`spyx.nn.run`](../reference/nn.md).

This page explains a different family of neurons that trade the hard reset for
the ability to score a whole sequence **in parallel**, why that trade-off exists,
and what it buys you in practice.

## Sequential scan: `O(T)` depth

A `jax.lax.scan` over `T` timesteps is a chain of `T` dependent steps. On an
accelerator the wall-clock cost is dominated not by the arithmetic — each step is
tiny — but by the **length of the dependency chain**: `T` kernel launches that
cannot start until their predecessor finishes. The *critical path* is `O(T)`
deep. For short sequences on a busy GPU this is fine; for long sequences on a GPU
with spare compute it leaves the device mostly idle, waiting on latency rather
than doing work.

## The reset-free trick: a linear recurrence

Drop the reset. Without the `- S_t θ` term the membrane is a **pure linear leaky
integrator**:

$$
V_t = \beta \, V_{t-1} + x_t .
$$

This is a first-order *linear* recurrence, and linear recurrences are
**associative**: composing "multiply by \(\beta\), add \(x\)" maps can be
reordered freely. That is exactly the structure
[`jax.lax.associative_scan`](https://docs.jax.org/en/latest/_autosummary/jax.lax.associative_scan.html)
exploits — the same parallel-prefix-scan machinery behind modern diagonal
state-space models (S4D/S5, LRU, Mamba; see [`spyx.ssm`](../reference/ssm.md)).
The whole membrane trace \(V_0, \dots, V_{T-1}\) can be computed in
**\(O(\log T)\) parallel depth** instead of \(O(T)\). Spikes are then a pointwise
surrogate threshold applied to the whole trace, \(s_t = \sigma(V_t - \theta)\),
which is embarrassingly parallel.

The catch is the reset you gave up. A reset-free neuron never depresses after
firing, so it can spike on consecutive steps and its activity is **less sparse**.
On the reference machine below, [`spyx.nn.PSU_LIF`](../reference/nn.md) fires
roughly **3× more often** than a tuned `LIF` on the same input — you keep
activity bounded with the leak \(\beta\) and the threshold rather than with a
reset. Sparsity is an SNN's energy story (see the
[benchmarking how-to](../how-to/benchmarking.md) on the `spike_rate` proxy), so
this is a real cost, not a free lunch: you buy parallel depth with density.

## Two neurons that do this

Both neurons follow the standard Spyx contract — a stepwise
`__call__(x_t, state) -> (spikes, state)` usable in
[`spyx.nn.run`](../reference/nn.md), `Sequential`, and NIR — **and** additionally
expose a `parallel(x)` method that scores a time-major `(T, B, …)` sequence at
once via an associative scan. The two paths use the same parameters and the same
surrogate and are numerically identical: scanning the step reproduces the
parallel result exactly.

### `spyx.nn.PSU_LIF` — parallel real leaky integrator

[`spyx.nn.PSU_LIF`](../reference/nn.md) (Parallel Spiking Unit LIF) is the
reset-free real-valued neuron described above: \(V_t = \operatorname{clip}(\beta)
\, V_{t-1} + x_t\), with a per-unit learnable leak. Use it as a drop-in for `LIF`
when the reset is not essential and the sequence is long. It is the template for
reset-free parallel spiking neurons in Spyx.

### `spyx.phasor.ResonateFire` — parallel complex oscillator

[`spyx.phasor.ResonateFire`](../reference/phasor.md) is the complex/oscillatory
sibling of `PSU_LIF`. Its membrane is a **complex** number evolving as a damped
harmonic oscillator,

$$
z_t = a \, z_{t-1} + x_t , \qquad a = e^{\,\mathrm{dt}\,(-\lambda + i\,\omega)} ,
$$

with per-unit decay \(\lambda \ge 0\) and angular frequency \(\omega\). Real input
is injected into the real part; spikes come from a surrogate threshold on
\(\Re(z_t)\). Because it too is reset-free, the recurrence stays linear and the
same \(O(\log T)\) associative scan applies — only now over a *complex* pole `a`
instead of a real leak. Storing \(\lambda\) through a `softplus` keeps
\(|a| = e^{-\mathrm{dt}\,\lambda} \le 1\), so the oscillation never grows.
Following [`spyx.phasor.PhasorLinear`](../reference/phasor.md), the pole
parameters are stored as **real** `float32` params, so a stock `optax` +
`jax.grad` loop over a real loss trains them without the Wirtinger-conjugate
surprise. `ResonateFire` gives you tunable frequency selectivity that a
first-order leaky integrator cannot express.

## When does parallel actually win?

The `O(log T)` depth is an asymptotic statement about the *critical path*, not a
guarantee about wall-clock time on your GPU. Whether the parallel scan beats the
sequential one depends on whether the device has **slack**. The numbers below
were **measured** with [`spyx.bench`](../how-to/benchmarking.md) on an
**AMD Radeon 8060S iGPU (gfx1151)** on ROCm — treat them as an empirical
observation on that hardware, not a guarantee.

- **GPU saturated (large batch / large hidden size, throughput-bound).** When
  the device is already busy doing useful arithmetic every step, the sequential
  scan's latency is largely hidden, and the parallel scan wins only modestly —
  roughly **1–3×**. Here you are compute-bound and both schedules keep the GPU
  full.

- **GPU with slack (small batch, long sequences, latency-bound).** When each
  step is too small to fill the device, the sequential scan spends most of its
  time waiting on its own `O(T)` dependency chain while the GPU sits idle. The
  parallel scan collapses that chain to `O(log T)` and fills the device instead,
  and the measured speed-up climbs to **100×+** in the long-sequence,
  small-batch regime.

The crossover is exactly what the depth argument predicts: parallelism pays when
the sequential critical path — not the arithmetic — is the bottleneck. Use the
[benchmarking how-to](../how-to/benchmarking.md) to find the crossover on *your*
hardware and workload before committing to one path; `spyx.bench`'s default
driver automatically uses a neuron's `parallel` method when it has one, so
`compare()` measures the parallel path out of the box.

## Neuromorphic export

Because `PSU_LIF` and `ResonateFire` expose the standard stepwise contract, they
participate in the [NIR export/import flow](../how-to/nir.md) like any other
neuron. NIR describes stepwise dynamics, so it is the sequential step — not the
parallel scan — that is the exportable object; the parallel path is a
mathematically equivalent way to *evaluate* the same recurrence on a GPU. NIR
support for these reset-free neurons is being extended, so treat their interop as
high-level for now and consult the [NIR how-to](../how-to/nir.md) for the layers
with tested round-trip parity.

## Further reading

These neurons build directly on prior work on parallelizable spiking units — see
the **Stochastic Parallelizable Spiking Neurons** study and the
`parallel_spiking_neurons` work in the [Spyx research directory](research.md),
which benchmarks `PSU_LIF` and `ResonateFire` against a hard-reset `LIF`.
