# A primer on spiking neural networks

This page explains the four ideas you need to make sense of everything else in Spyx: what a spike is, how a leaky integrate-and-fire neuron works, why training needs *surrogate gradients*, and how information gets encoded into spikes in the first place. It is background reading — for hands-on material, start with the [first-SNN tutorial](../tutorials/first-snn.md).

## A spike is a binary event in time

Conventional artificial neurons exchange real-valued activations at every layer, every forward pass. Biological neurons don't: they communicate through *spikes* — brief, all-or-nothing electrical pulses. What matters is not the shape of the pulse but **whether** and **when** it happens. A spiking network is therefore best thought of as a dynamical system unrolled over discrete timesteps, where each neuron either fires (1) or stays silent (0) at each step.

This event-driven style is what makes SNNs attractive: activity is sparse (most neurons are silent most of the time), and on neuromorphic hardware, silence costs nothing. The price is that time becomes a first-class dimension of every computation.

Concretely, Spyx represents a batch of spike trains as a time-major `[T, B, C]` float32 tensor: `T` timesteps, `B` batch items, `C` channels, every value 0 or 1. Because such tensors are mostly zeros, the data loaders in `spyx.data` store them bit-packed along the time axis (`uint8`), and you recover the dense form with `jnp.unpackbits(obs, axis=1)[:, :T, :]`.

## The leaky integrate-and-fire neuron

The workhorse neuron model is the **leaky integrate-and-fire** (LIF) unit. Each neuron keeps one piece of state — a membrane voltage \(V\) — and updates it every timestep in three moves: *leak*, *integrate*, *fire*.

$$
S_t = H(V_t - \theta), \qquad
V_{t+1} = \beta V_t + x_t - S_t\,\theta
$$

where \(H\) is the Heaviside step function and \(\theta\) is the firing threshold. In words (this is exactly the recurrence in `spyx.nn.LIF`, `src/spyx/nn.py`):

- **Leak:** the voltage decays by a factor \(\beta \in [0, 1]\) each step. \(\beta = 0\) means no memory at all; \(\beta = 1\) means a perfect, lossless integrator. In Spyx, `beta` is a learnable `nnx.Param` by default, so each neuron can tune its own timescale.
- **Integrate:** the input \(x_t\) (typically the output of an `nnx.Linear` layer applied to the previous layer's spikes) is added to the voltage.
- **Fire and reset:** when the voltage crosses the threshold, the neuron emits a spike and the voltage is knocked back down by \(\theta\) ("reset by subtraction"), leaving any excess above threshold intact.

The output of the layer is the binary spike tensor \(S_t\); the voltage is carried forward as state. This is why every Spyx neuron has the signature `(x, state) -> (spikes, new_state)` and an `initial_state(batch_size)` method, and why `spyx.nn.run` exists to scan a network over time.

The other neurons in `spyx.nn` are variations on this theme:

- **`IF`** — the same with \(\beta = 1\): no leak, pure integration.
- **`CuBaLIF`** — *current-based* LIF: adds a separate synaptic-current state with its own decay \(\alpha\), so inputs are smoothed before they hit the membrane. Two coupled timescales instead of one.
- **`ALIF`** — *adaptive* LIF: the threshold itself becomes dynamic, rising after each spike and decaying back, giving the neuron spike-frequency adaptation (a cheap form of longer-term memory).
- **`RIF` / `RLIF` / `RCuBaLIF`** — recurrent variants that feed the layer's own spikes back through a learned square `recurrent_w` matrix.
- **`LI`** — the non-spiking readout: a leaky integrator with no threshold, whose voltage is used directly as logits.

## Surrogate gradients: making the spike differentiable

Here is the central obstacle of SNN training. The spike function is a step: \(S = H(V - \theta)\). Its derivative is zero everywhere except at the threshold, where it is undefined. Backpropagate through that and every gradient upstream of a spike is exactly zero — the network cannot learn.

The **surrogate gradient** trick resolves this with a deliberate inconsistency:

- **Forward pass:** keep the exact, hard step function. Spikes stay binary; the network you train is the network you deploy.
- **Backward pass:** *pretend* the step was a smooth function, and use that function's derivative instead — some bump-shaped curve centred on the threshold, such as SuperSpike's \(1 / (1 + k|x|)^2\) or a simple triangle \(\max(0, 1 - |kx|)\).

Intuitively, the surrogate says: "a neuron whose voltage was *near* the threshold should receive some gradient, because a small weight change could have flipped its decision." Neurons far from threshold get almost none. The result is a biased but remarkably effective descent direction — surrogate-gradient training routinely matches task-tuned alternatives.

In Spyx this is implemented with `jax.custom_gradient`, not hand-written backward passes. Each factory in `spyx.axn` (`superspike`, `arctan`, `tanh`, `boxcar`, `triangular`, and the build-your-own `custom`) returns a JIT-compiled function whose forward is the exact Heaviside and whose backward is the chosen surrogate:

```python
lif = spyx.nn.LIF((64,), activation=spyx.axn.triangular(), rngs=rngs)
```

The choice of surrogate (and its width parameter `k`) is a genuine hyperparameter — narrower surrogates give more precise credit assignment but risk dead gradients; wider ones learn faster but blur which neuron was responsible. The [surrogate comparison notebook](../examples/surrogate_gradient/shd_sg_surrogate_comparison.ipynb) benchmarks them head-to-head.

With the spike made differentiable, training is ordinary **backpropagation through time**: `spyx.nn.run` scans the network over the sequence with `jax.lax.scan`, and gradients flow backward through every timestep. The loss is usually computed on the *integral* of the readout voltages over time rather than any single step (`spyx.fn.integral_crossentropy`), which makes the objective robust to exactly *when* within the sequence the evidence arrives.

## Rate coding vs. latency coding

If your data is not already spikes (unlike event cameras or silicon cochleas, which produce spikes natively), you must choose how to encode real values into spike trains. The two classic schemes sit at opposite ends of a trade-off, and `spyx.data` implements both:

**Rate coding** (`spyx.data.rate_code`) represents a value by *how often* a neuron fires: each timestep is an independent Bernoulli draw with probability proportional to the input value. It is robust — losing or jittering a single spike barely changes the rate — and it is the assumption baked into integral-based losses. The cost is inefficiency: conveying one value precisely takes many spikes over many timesteps, and every spike costs energy on hardware.

**Latency coding** (`spyx.data.latency_code`) represents a value by *when* a neuron first fires: larger values fire earlier in the window, smaller values later, and values below a threshold never fire. One spike per neuron carries the whole message, giving far sparser activity and lower latency — decisions can be made as soon as the first spikes arrive. The cost is fragility: the information rides on precise timing, so noise and jitter hurt much more, and training signals are sparser too.

Rate coding is the safe default for classification benchmarks; latency coding matters when you care about the things SNNs are ultimately for — energy, sparsity, and reaction time on neuromorphic hardware. (A third option, `angle_code`, discretises a continuous value into a one-hot population — useful for encoding low-dimensional observations, e.g. in the [neuroevolution tutorial](../examples/neuroevolution/cartpole_evo.ipynb).)

## Where to go from here

- Do the [first-SNN tutorial](../tutorials/first-snn.md) to put all four ideas into practice.
- Read [Design and architecture](design.md) for why Spyx maps these concepts onto JAX and Flax NNX the way it does.
- Neftci, Mostafa & Zenke (2019), ["Surrogate Gradient Learning in Spiking Neural Networks"](https://arxiv.org/abs/1901.09948) — the standard review of the training method used throughout Spyx.
