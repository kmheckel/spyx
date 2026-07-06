# Quickstart: train an SNN in 40 lines (no downloads)

This is the fastest way to see Spyx actually train something. It builds a tiny
spiking network and trains it on **synthetic spike trains generated on the fly**,
so there is nothing to download and it runs in a few seconds on a laptop CPU.

If you have not installed Spyx yet, see [How to install Spyx](install.md) — for
this page a plain `pip install spyx` (or `uv add spyx`) is enough.

## The whole thing

Copy this into `quickstart.py` and run `python quickstart.py`:

```python
import jax, jax.numpy as jnp, optax
from flax import nnx
import spyx, spyx.nn as snn, spyx.optimize as opt

rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(8, 32, use_bias=False, rngs=rngs),
    snn.LIF((32,), activation=spyx.axn.triangular(), rngs=rngs),
    nnx.Linear(32, 3, use_bias=False, rngs=rngs),
    snn.LI((3,), rngs=rngs),  # non-spiking leaky readout -> class logits
)

T, B, C, n_cls = 16, 32, 8, 3  # time, batch, channels, classes

def make_batch(k):  # class c => channel c fires often (learnable structure)
    ky, ks = jax.random.split(k)
    y = jax.random.randint(ky, (B,), 0, n_cls)
    prob = jnp.full((B, C), 0.05).at[jnp.arange(B), y].set(0.5)
    x = (jax.random.uniform(ks, (T, B, C)) < prob).astype(jnp.float32)
    return x, y  # x is time-major (T, B, C)

Loss = spyx.fn.integral_crossentropy(time_axis=0)
Acc = spyx.fn.integral_accuracy(time_axis=0)

def loss_fn(m, x, y):
    return Loss(snn.run(m, x)[0], y)

def eval_fn(m, x, y):
    traces = snn.run(m, x)[0]
    return Acc(traces, y)[0], Loss(traces, y)

key = jax.random.PRNGKey(0)
train_iter = lambda: (make_batch(jax.random.fold_in(key, i)) for i in range(8))
eval_iter = lambda: iter([make_batch(jax.random.PRNGKey(999))])

opt.fit(
    model, optax.adam(2e-3), loss_fn, train_iter,
    epochs=15, eval_iter=eval_iter, eval_fn=eval_fn,
    on_epoch_end=lambda e, m: print(
        f"epoch {e:2d}  train_loss={m['train_loss']:.3f}  eval_acc={m['eval_acc']:.2%}"),
)
```

Expected output — a falling loss and a rising accuracy (exact numbers vary with
the JAX/hardware backend):

```text
epoch  0  train_loss=3.536  eval_acc=28.12%
epoch  3  train_loss=1.192  eval_acc=68.75%
epoch  7  train_loss=0.862  eval_acc=81.25%
epoch 14  train_loss=0.790  eval_acc=90.62%
```

## What just happened

- **The network** is a two-layer spiking MLP: `Linear -> LIF -> Linear -> LI`.
  `LIF` is the leaky integrate-and-fire spiking nonlinearity; `LI` is a
  non-spiking leaky integrator whose membrane voltage we read out as class
  logits. Anything following the `(x, state) -> (out, new_state)` contract drops
  into [`snn.Sequential`](../reference/nn.md).
- **`activation=spyx.axn.triangular()`** picks the *surrogate gradient* — the
  smooth stand-in used for the spike's (non-existent) derivative during
  backprop. See the [glossary](../explanation/glossary.md) for the vocabulary.
- **The data** is time-major `(T, B, C)`: for each sample of class `c`, input
  channel `c` fires with high probability and the rest fire rarely, so there is
  a real pattern for the network to learn — hence the honest rise in accuracy.
- **`snn.run(model, x)`** scans the network over the time axis with
  `jax.lax.scan`, returning `(traces, final_state)`. `traces[0]` is the readout
  voltage at every timestep.
- **`integral_crossentropy` / `integral_accuracy`** sum the readout voltages
  over time (the "integral" over the sequence) before applying cross-entropy /
  argmax. We pass `time_axis=0` because we kept the tensors time-major; the
  loaders in [Your first SNN](../tutorials/first-snn.md) hand you batch-major
  data, where the default `time_axis=1` applies.
- **`spyx.optimize.fit`** wraps the canonical `nnx.Optimizer` +
  `nnx.value_and_grad` loop. To write that loop yourself, see
  [How to train a model](train.md).

## Next steps

- **For real data**, continue to [Your first SNN](../tutorials/first-snn.md),
  which trains the same architecture on the Spiking Heidelberg Digits dataset.
- **To pick a training method** (surrogate gradients vs. evolution vs.
  quantization-aware vs. ANN→SNN conversion vs. the hybrid), read
  [Choosing an approach](../explanation/choosing-an-approach.md).
- **For the theory**, read the [SNN primer](../explanation/snn-primer.md).
