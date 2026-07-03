# How to train a model

To train a Spyx model you need three ingredients: a loss function from [`spyx.fn`](../reference/fn.md), an [Optax](https://optax.readthedocs.io/) optimizer, and batches of `(events, targets)`. You can then either use the high-level `spyx.optimize.fit` loop or write the NNX training step yourself.

Both recipes below assume this model and loss:

```python
import jax.numpy as jnp
import optax
from flax import nnx
import spyx
import spyx.nn as snn

rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(128, 64, use_bias=False, rngs=rngs),
    snn.LIF((64,), activation=spyx.axn.triangular(), rngs=rngs),
    nnx.Linear(64, 20, use_bias=False, rngs=rngs),
    snn.LI((20,), rngs=rngs),
)

Loss = spyx.fn.integral_crossentropy()   # (traces_BTC, targets_B) -> scalar

def forward(m, x_BTC):
    x_TBC = jnp.transpose(x_BTC, (1, 0, 2))
    traces, _ = snn.run(m, x_TBC)
    return jnp.transpose(traces, (1, 0, 2))
```

## Option 1: `spyx.optimize.fit`

To train without writing any loop boilerplate, pass a loss closure and a *zero-argument callable* that yields fresh batches each epoch (this matches the `loader.train_epoch()` convention of the [Spyx data loaders](load-data.md)):

```python
import spyx.optimize as opt

def loss_fn(m, events, targets):
    return Loss(forward(m, events), targets)

def train_iter():
    for batch in dl.train_epoch():          # dl = spyx.data.SHD_loader(...)
        yield unpack(batch.obs), jnp.asarray(batch.labels)

history = opt.fit(
    model,
    optax.lion(3e-4),
    loss_fn,
    train_iter,
    epochs=30,
)
```

`fit` mutates `model` in place and returns a list of per-epoch metric dicts (`train_loss`, plus `eval_acc` / `eval_loss` if you also pass `eval_iter=` and `eval_fn=`). Use `on_epoch_end=lambda epoch, metrics: ...` for progress printing or early logging.

To keep JIT compilation but customise the step, build the pieces `fit` uses internally:

```python
train_step = opt.make_train_step(loss_fn)    # (model, optimizer, *batch) -> loss
eval_step = opt.make_eval_step(eval_fn)      # (model, *batch) -> (acc, loss)
```

## Option 2: hand-rolled `nnx.Optimizer` loop

To control the training loop yourself — custom metrics, gradient clipping, multiple losses — write the canonical Flax NNX pattern:

```python
optimizer = nnx.Optimizer(model, optax.lion(3e-4), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, events, targets):
    def loss_fn(m):
        return Loss(forward(m, events), targets)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

for epoch in range(30):
    losses = []
    for events, targets in train_iter():
        losses.append(train_step(model, optimizer, events, targets))
    print(epoch, float(jnp.mean(jnp.stack(losses))))
```

Notes:

- Spyx requires `flax>=0.11`, where `nnx.Optimizer` takes `wrt=nnx.Param` and the update call is `optimizer.update(model, grads)`.
- `@nnx.jit` compiles the whole step, including the time scan inside `snn.run`; the model and optimizer are updated in place.
- To compose Optax transforms, just chain them: `optax.chain(optax.centralize(), optax.lion(3e-4))`.

## Add activity regularisation

To discourage silent or over-active neurons, tap the intermediate spike trains in a custom module and add `spyx.fn.silence_reg` / `spyx.fn.sparsity_reg` penalties:

```python
Sil = spyx.fn.silence_reg(2.0)     # penalise neurons spiking < 2 times on average
Spa = spyx.fn.sparsity_reg(8.0)    # penalise layers spiking > 8 times on average

def loss_fn(m):
    traces, spikes = m(events)     # a model returning (readout, [spike taps])
    return Loss(traces, targets) + 1e-3 * (Sil(spikes) + Spa(spikes))
```

See `scripts/smoke_notebook_apis.py` (`smoke_shd_template`) in the repository for a complete model that exposes its per-layer spike taps via `jax.lax.scan`.
