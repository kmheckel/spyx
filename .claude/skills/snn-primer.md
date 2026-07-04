---
name: snn-primer
description: Explain spiking-neural-network concepts to someone new to the field using Spyx's API surface as the concrete hook. Use when the user asks what spikes / LIF / surrogate gradients are, or says they're new to SNNs and want a quick mental model before diving in.
---

# SNN primer for Spyx

When the user is new to spiking neural networks, walk them through the four concepts below *using Spyx code* as the anchor. Keep it concrete — point at `src/spyx/` files the user can open, not at abstract equations.

## 1. A spike is a binary event in time

Spyx represents a batch of spike trains as a `[T, B, C]` float32 tensor (time-major): T timesteps, B batch items, C channels. Each value is 0 or 1. Data loaders (`spyx.data.SHD_loader`, `NMNIST_loader`) emit packed `uint8` along the time axis for memory; you recover dense spikes with `jnp.unpackbits(obs, axis=1)[:, :T, :]`.

## 2. LIF is a leaky accumulator with a threshold

Read `src/spyx/nn.py`'s `LIF` class with the user. The recurrence is:

```python
V = beta * V + x - spikes * threshold
spikes = heaviside(V - threshold)
```

- `beta` is the membrane decay (0 = no memory, 1 = perfect integrator).
- Input `x` drives the voltage up; each spike resets it by `threshold`.
- `spikes` is 0/1; the output of the layer is that spike tensor.

`IF` is the same thing with `beta = 1` (no leak). `CuBaLIF` adds a separate synaptic-current variable with its own decay `alpha`. Recurrent variants (`RIF`/`RLIF`/`RCuBaLIF`) add a square `recurrent_w` matrix on top. `LI` is the non-spiking readout: a leaky integrator whose voltage is used directly as logits.

## 3. Surrogate gradients make Heaviside differentiable

Forward pass: spike = 1 if V > threshold else 0. Derivative of Heaviside is 0 almost everywhere, so backprop through it kills learning.

Trick: during backward, pretend the Heaviside is a smooth function like `1 / (1 + k|x|)^2` (SuperSpike). The factories in `src/spyx/axn.py` (`superspike`, `arctan`, `triangular`, `boxcar`, `tanh`, `custom`) all return JIT-compiled `jax.custom_gradient` functions that swap the backward pass while keeping the forward exact.

```python
activation = spyx.axn.triangular()   # pick any factory
lif = spyx.nn.LIF((64,), activation=activation, rngs=rngs)
```

## 4. Training is BPTT through `jax.lax.scan`

`spyx.nn.run(model, x)` scans a `Sequential` stack over the time axis. Gradients flow backward through every timestep. Loss is usually computed on the **integral** of the output-layer voltage traces rather than any single step — see `spyx.fn.integral_crossentropy` / `integral_accuracy`.

High-level recipe:

```python
from flax import nnx
import optax, spyx, spyx.nn as snn, spyx.optimize as opt

rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(in_dim, 64, use_bias=False, rngs=rngs),
    snn.LIF((64,), rngs=rngs),
    nnx.Linear(64, n_classes, use_bias=False, rngs=rngs),
    snn.LI((n_classes,), rngs=rngs),
)
Loss = spyx.fn.integral_crossentropy()
def loss_fn(m, x, y): return Loss(forward(m, x), y)

history = opt.fit(model, optax.lion(3e-4), loss_fn, train_iter, epochs=30)
```

## What to show next

- `docs/examples/surrogate_gradient/SurrogateGradientTutorial.ipynb` — the canonical end-to-end walkthrough on SHD.
- `docs/tutorials/first-snn.md` — guided install + first trained SNN.
- `docs/explanation/snn-primer.md` — the reader-facing version of this primer.
- For experiments: invoke the `new-experiment` skill to scaffold a training script.
