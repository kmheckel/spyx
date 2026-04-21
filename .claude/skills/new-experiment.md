---
name: new-experiment
description: Scaffold a new Spyx training experiment. Use when the user asks to "build a training script", "try Spyx on my dataset", "set up an experiment", or "how do I structure a new SNN project".
---

# Scaffold a new Spyx experiment

Create a single Python script under `research/<short-name>/train.py` (or a new notebook under `docs/examples/`, if the user wants interactive). The scaffold should cover five sections. Adapt each to the user's data and architecture.

## 1. Imports + config

```python
import jax, jax.numpy as jnp, optax
from flax import nnx
import spyx, spyx.nn as snn, spyx.optimize as opt

BATCH = 128
SAMPLE_T = 128
CHANNELS = ...          # depends on user's input
N_CLASSES = ...         # depends on user's task
HIDDEN = 64
EPOCHS = 30
LR = 3e-4
SEED = 0
```

## 2. Data iterators

- If the user has neuromorphic spikes (SHD/NMNIST/etc.): use `spyx.data.SHD_loader` or wire a custom `TonicSource` into `spyx.data.GrainLoader`.
- If the user has rate-coded continuous data (images, audio features): use `spyx.data.rate_code` (Bernoulli per timestep) or `spyx.data.latency_code` (single spike per cycle).
- If the user has already-formatted `[B, T, C]` float tensors: skip the encoders and yield them directly.

Required shape before the forward pass: `[B, T, CHANNELS]` dense floats. Wrap in a zero-arg generator-returning callable so `spyx.optimize.fit` can re-iterate each epoch:

```python
def train_iter():
    for batch in loader.train_epoch():
        events = _to_dense_BTC(batch.obs)       # user-specific
        yield events, jnp.asarray(batch.labels)
```

## 3. Model

Default template is three dense + one readout. The key choices are:

- **Neuron type**: `LIF` is the safe default; `ALIF` for tasks needing spike-rate adaptation; `RLIF` / `RCuBaLIF` for explicit recurrence; `CuBaLIF` when the problem benefits from separate synaptic-current dynamics.
- **Surrogate**: `spyx.axn.triangular()` is the safe default; `superspike()` is common; see `docs/examples/surrogate_gradient/shd_sg_surrogate_comparison.ipynb`.
- **Depth**: 2 hidden layers for SHD-scale, 3–4 for harder tasks.

```python
rngs = nnx.Rngs(SEED)
model = snn.Sequential(
    nnx.Linear(CHANNELS, HIDDEN, use_bias=False, rngs=rngs),
    snn.LIF((HIDDEN,), activation=spyx.axn.triangular(), rngs=rngs),
    nnx.Linear(HIDDEN, HIDDEN, use_bias=False, rngs=rngs),
    snn.LIF((HIDDEN,), activation=spyx.axn.triangular(), rngs=rngs),
    nnx.Linear(HIDDEN, N_CLASSES, use_bias=False, rngs=rngs),
    snn.LI((N_CLASSES,), rngs=rngs),
)
```

## 4. Loss, metric, forward

`nn.run` is time-major `[T, B, C]`; most loaders yield `[B, T, C]`. Transpose in both directions:

```python
Loss = spyx.fn.integral_crossentropy()
Acc  = spyx.fn.integral_accuracy()

def forward(m, x_BTC):
    x_TBC = jnp.transpose(x_BTC, (1, 0, 2))
    traces, _ = snn.run(m, x_TBC)
    return jnp.transpose(traces, (1, 0, 2))

def loss_fn(m, x, y):   return Loss(forward(m, x), y)
def eval_fn(m, x, y):
    traces = forward(m, x)
    acc, _ = Acc(traces, y)
    return acc, Loss(traces, y)
```

## 5. Training loop

```python
history = opt.fit(
    model,
    optax.lion(LR),
    loss_fn,
    train_iter,
    epochs=EPOCHS,
    eval_iter=eval_iter,
    eval_fn=eval_fn,
    on_epoch_end=lambda e, m: print(f"epoch {e}: {m}"),
)
```

## Extensions to offer, in order of usefulness

1. **Activity regularization**: wrap hidden layers in `spyx.nn.ActivityRegularization` or tap intermediate spike trains and add `spyx.fn.silence_reg` / `sparsity_reg` to the loss. See `shd_sg_template.ipynb`.
2. **Data augmentation**: `spyx.data.shift_augment` randomly rolls the channel axis per batch — cheap and effective on SHD.
3. **Quantization**: once fp32 works, drop in `spyx.quant.quantize(model, sample_x, sample_state)` for int8 QAT. Requires `[quant]` extra.
4. **NIR export**: after training, call `spyx.nir.to_nir(model, ...)` for neuromorphic deployment.

## Don'ts (common newcomer mistakes)

- Don't forget `use_bias=False` on the `nnx.Linear` layers unless you have a specific reason — SNNs typically don't use biases because the LIF threshold already provides an offset.
- Don't pass `complex64` inputs to a non-phasor layer; stick to real float32.
- Don't run `spyx.nn.run` on a stateless module (e.g. a bare `nnx.Linear`) — wrap in `Sequential` or `vmap` over time.
- Don't forget `wrt=nnx.Param` on `nnx.Optimizer` — flax 0.11+ requires it.

After scaffolding, run the script once on a tiny synthetic batch (`EPOCHS=1`, `BATCH=4`) to confirm it executes, then hand control back to the user to bump it up.
