# Glossary

Concise definitions of the terms that show up across the Spyx tutorials, how-to
guides, and API reference. For the fuller story behind spiking dynamics, read
the [SNN primer](snn-primer.md); for the training-method landscape, see
[Choosing an approach](choosing-an-approach.md).

### Membrane potential (V)
The internal state ("voltage") a spiking neuron accumulates over time. Input
currents push it up; leak pulls it down. When it crosses the threshold the
neuron emits a spike and the potential is reset. In Spyx it lives in the
neuron's carried `state`.

### Leak / β (beta)
The decay factor applied to the membrane potential each timestep, `V ← β·V + …`,
with `0 < β < 1`. Larger β means a longer memory (slower leak); β near 0 means
the neuron nearly forgets its past each step. Set per-layer, e.g.
`snn.LIF((64,), beta=0.8, …)`.

### Threshold
The membrane-potential value at which a neuron fires a spike. Crossing it
produces a `1`; otherwise `0`. The comparison is a step function, which is why
training needs a surrogate gradient.

### Subtractive reset
After firing, the neuron's potential is reduced by the threshold amount
(`V ← V − θ`) rather than being clamped to zero (a "hard reset"). Subtractive
reset preserves the overshoot above threshold and is the default for Spyx's LIF
family.

### Spike / spike train
A spike is a discrete `1` event at a single timestep; a spike train is the
time series of 0/1 events for a unit or channel. Spyx tensors are typically
`float32` arrays whose spiking entries are exactly `{0, 1}`.

### Time-major
Tensor layout `(T, B, C)` — time first, then batch, then channels/features.
`spyx.nn.run` scans over axis 0. Data loaders often hand you *batch-major*
`(B, T, C)`; transpose, or pass `time_axis=` to the `spyx.fn` losses/metrics.

### LIF (Leaky Integrate-and-Fire)
The workhorse spiking neuron: integrate input into the membrane potential, leak
it each step, fire and reset when it crosses threshold. `spyx.nn.LIF`.

### LI (Leaky Integrator)
A non-spiking neuron — it integrates and leaks but never fires. Used as the
readout layer, whose accumulated voltage becomes the class logits.
`spyx.nn.LI`.

### IF (Integrate-and-Fire)
LIF without the leak (β = 1): the potential only goes up until it fires.
`spyx.nn.IF`.

### ALIF (Adaptive LIF)
LIF with an adaptive threshold that rises after each spike and decays back down,
giving spike-frequency adaptation and longer effective memory.
`spyx.nn.ALIF`.

### CuBaLIF (Current-Based LIF)
LIF with a separate synaptic-current state that is itself low-pass filtered
before driving the membrane potential — a second time constant for smoother
temporal integration. `spyx.nn.CuBaLIF`.

### Recurrent neuron (RIF / RLIF / RCuBaLIF)
Variants that add a recurrent weight matrix so a layer's spikes feed back into
itself at the next timestep, not just forward to the next layer.

### Surrogate gradient
The spike's true derivative is zero almost everywhere and undefined at
threshold, so backprop is impossible through the raw step. A surrogate gradient
replaces that derivative with a smooth, bell-shaped function during the backward
pass only (the forward pass still emits hard spikes). Spyx implements these as
`jax.custom_gradient` factories in [`spyx.axn`](../reference/axn.md)
(`triangular`, `arctan`, `superspike`, `boxcar`, `tanh`).

### Surrogate width k
The sharpness/scale parameter of a surrogate gradient (e.g. `arctan(k=2)`).
Larger k makes the surrogate narrower and more spike-like; smaller k makes it
wider and smoother, passing gradient to units further from threshold.

### BPTT (Backpropagation Through Time)
Training a recurrent/temporal network by unrolling it over all timesteps and
backpropagating through the whole unrolled graph. In Spyx this is the gradient
that flows back through `snn.run`'s scan, with the surrogate standing in for the
spike at every step.

### Integral loss
A loss computed on the sum (integral) of the readout trace over time rather than
at a single timestep — the SNN's "answer" is the class whose voltage integrates
highest across the sequence. See `spyx.fn.integral_crossentropy` and
`integral_accuracy`.

### Rate coding
Encoding a value as a *firing rate* — a higher input drives more spikes per unit
time. `spyx.data.rate_code`.

### Latency coding
Encoding a value in *spike timing* — stronger inputs fire earlier. Sparse
(often one spike per channel) and energy-frugal. `spyx.data.latency_code`.

### Angle coding
Encoding values as phases/angles, natural for the complex-valued phasor
networks. `spyx.data.angle_code`, `spyx.phasor`.

### Spike-rate energy proxy
The mean number of spikes a network emits, used as a hardware-agnostic stand-in
for energy: on neuromorphic hardware, energy scales with spikes, so lower
spike-rate at equal accuracy means a more efficient model. Reported by
`spyx.bench`.

### MFU (Model FLOPs Utilization)
The fraction of a device's peak floating-point throughput a model actually
achieves — a measure of how well the workload uses the hardware. Reported by
`spyx.bench` from the XLA cost model.

### Associative scan / parallel scan
An algorithm that evaluates a sequential recurrence in `O(log T)` parallel depth
instead of `O(T)` sequential steps, when the update is associative. Spyx uses
`jax.lax.associative_scan` to parallelize state-space models
([`spyx.ssm`](../reference/ssm.md)), phasors, and the experimental
parallel spiking neurons over the time axis.

### State-space model (SSM)
A linear recurrence `h ← A·h + B·x`, `y = C·h`, with a diagonal `A` so it runs
as a fast associative scan. Spyx ships LRU, S5Diag, Mamba, and ChunkedSSM in
[`spyx.ssm`](../reference/ssm.md).

### Phasor network
A network whose activations are complex numbers (magnitude + phase), a
continuous relative of spike timing. `spyx.phasor` provides phasor linear/MLP
layers, a spiking-phasor neuron, and phase↔spike helpers.

### QAT (Quantization-Aware Training)
Training with simulated low-precision (e.g. int8/int4) weights and activations
so the model learns to tolerate the rounding, yielding a small, hardware-ready
network. `spyx.quant.quantize(..., mode="qat")`.

### PTQ (Post-Training Quantization)
Quantizing an already-trained model without further training — faster to apply
but usually a bit less accurate than QAT.
`spyx.quant.quantize(..., mode="ptq")`.

### NIR (Neuromorphic Intermediate Representation)
A hardware-agnostic graph format for spiking/neuromorphic models. Spyx converts
to and from it (`spyx.nir.to_nir` / `from_nir`) so models can move to
neuromorphic backends. See [neuroir.org](https://neuroir.org).

### ONNX
The Open Neural Network Exchange format. Spyx can export a spiking model — the
per-timestep step, or the whole `snn.run` temporal loop as an ONNX `Scan`/`Loop`
— via `spyx.experimental.onnx.to_onnx`.

### Evolution strategy (ES) / neuroevolution
Gradient-free optimization that perturbs the weights, keeps what scores better,
and never differentiates the network — useful when the objective is
non-differentiable or the surrogate misleads. Spyx wraps evosax behind the
`spyx[evo]` extra and the experimental hybrid trainer.

### Hybrid trainer
An experimental optimizer that combines the surrogate gradient with an
antithetic evolution-strategy estimate of the *true* (hard-spike) loss gradient,
projected orthogonal to the surrogate, to correct surrogate bias.
`spyx.experimental.hybrid`.

### Activity regularization
Extra loss terms that push a layer's spiking toward a target rate — penalizing
silent neurons (`spyx.fn.silence_reg`) or overly-active layers
(`spyx.fn.sparsity_reg`) — to keep the network healthy and sparse.

### `spyx.experimental`
The unstable-API namespace for research-stage building blocks (parallel spiking
neurons, resonate-and-fire, routing-slot memory, packed-bit activations,
stochastic neurons, the hybrid trainer, the recipe zoo, ONNX export). Its API
may change without a deprecation cycle; import from here so usage signals the
stability contract.
