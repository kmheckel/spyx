# Migrate from Haiku Spyx (≤0.1.x) to Flax NNX

Spyx `1.0` replaces the DeepMind **Haiku** backend with **Flax NNX**. This is a
breaking change: models are now stateful `nnx.Module` objects instead of
`hk.transform`-ed functions, so existing training scripts need edits. This guide
maps every old pattern to its new form.

If you only want the headline: **modules now carry their own parameters, take an
`rngs=` argument at construction, and you train them with `nnx.Optimizer`
instead of threading a `params` pytree through `SNN.apply`.**

## At a glance

| Concern | Old (Haiku, ≤0.1.x) | New (Flax NNX, 1.0+) |
| --- | --- | --- |
| NN library | `import haiku as hk` | `from flax import nnx` |
| Dense layer | `hk.Linear(64, with_bias=False)` | `nnx.Linear(in, 64, use_bias=False, rngs=rngs)` |
| Neuron | `snn.LIF((64,), activation=...)` | `snn.LIF((64,), activation=..., rngs=rngs)` |
| Stack layers | `hk.DeepRNN([...])` | `snn.Sequential(...)` |
| Unroll over time | `hk.dynamic_unroll(core, x, s0, time_major=False)` | `snn.run(core, x_TBC)` |
| Materialise params | `SNN = hk.transform(f); params = SNN.init(key, x)` | `model = MyModel(rngs=nnx.Rngs(0))` |
| Forward | `SNN.apply(params, x)` | `model(x)` |
| Grad | `jax.value_and_grad(loss)(params, ...)` | `nnx.value_and_grad(loss)(model)` |
| Optimiser | `opt.init(params)` + manual `opt.update` | `nnx.Optimizer(model, tx, wrt=nnx.Param)` |
| JIT | `@jax.jit` on the pure `apply` | `@nnx.jit` on a `model`-taking fn |
| Data loaders | `spyx.loaders` | `spyx.data` |
| Mixed precision | `jmp` + `hk.mixed_precision.set_policy` | pass `param_dtype=`/`dtype=` to layers |
| Python | `>=3.10` | `>=3.11, <3.15` |

New dependencies: `flax>=0.11`, `grain`. Removed: `dm-haiku`, `jmp`. New
optional modules: [`spyx.ssm`](../reference/ssm.md),
[`spyx.phasor`](../reference/phasor.md), [`spyx.quant`](../reference/quant.md).

## 1. Building a model

The Haiku version defined a **function** and transformed it; parameters lived in
an external pytree.

```python
# OLD — Haiku
import haiku as hk
import spyx, spyx.nn as snn

def shd_snn(x):                                        # x: (B, T, C)
    x = hk.BatchApply(hk.Linear(64, with_bias=False))(x)
    core = hk.DeepRNN([
        snn.LIF((64,), activation=spyx.axn.triangular()),
        hk.Linear(64, with_bias=False),
        snn.LIF((64,), activation=spyx.axn.triangular()),
        hk.Linear(20, with_bias=False),
        snn.LI((20,)),
    ])
    spikes, V = hk.dynamic_unroll(
        core, x, core.initial_state(x.shape[0]), time_major=False, unroll=32
    )
    return spikes, V

key = jax.random.PRNGKey(0)
SNN = hk.without_apply_rng(hk.transform(shd_snn))
params = SNN.init(rng=key, x=x[0])                     # params live here
```

The NNX version defines a **class**; parameters live inside the instance. Note
`nnx.Linear` needs the input dimension and every submodule needs `rngs=`.

```python
# NEW — Flax NNX
from flax import nnx
import spyx, spyx.nn as snn

class SHDSNN(nnx.Module):
    def __init__(self, in_dim, hidden, n_classes, *, rngs):
        self.core = snn.Sequential(
            nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs),
            snn.LIF((hidden,), activation=spyx.axn.triangular(), rngs=rngs),
            nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs),
            snn.LIF((hidden,), activation=spyx.axn.triangular(), rngs=rngs),
            nnx.Linear(hidden, n_classes, use_bias=False, rngs=rngs),
            snn.LI((n_classes,), rngs=rngs),
        )

    def __call__(self, x_BTC):                         # x: (B, T, C)
        x_TBC = jnp.transpose(x_BTC, (1, 0, 2))        # snn.run is time-major
        traces, _ = snn.run(self.core, x_TBC)          # replaces dynamic_unroll
        return jnp.transpose(traces, (1, 0, 2))        # back to (B, T, classes)

model = SHDSNN(in_dim=128, hidden=64, n_classes=20, rngs=nnx.Rngs(0))
```

Key points:

- **`snn.run` is time-major** `(T, B, C)`. Haiku's `dynamic_unroll(..., time_major=False)` accepted `(B, T, C)`; transpose in/out as above, or keep your tensors time-major throughout.
- **State is implicit.** `core.initial_state(batch)` is handled inside `snn.run`; you no longer build and pass it. If you drive a single neuron manually, call `layer.initial_state(batch)` and `spikes, V = layer(x, V)`.
- **`hk.BatchApply` is gone** — `nnx.Linear` already applies over leading batch/time axes.

## 2. The training loop

```python
# OLD — Haiku: params + opt_state threaded by hand
opt = optax.lion(3e-4)
opt_state = opt.init(params)

@jax.jit
def net_eval(weights, events, targets):
    traces, V = SNN.apply(weights, events)
    return Loss(traces, targets)

grads = jax.grad(net_eval)(params, events, targets)
updates, opt_state = opt.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

```python
# NEW — Flax NNX: optimizer owns model + opt state
optimizer = nnx.Optimizer(model, optax.lion(3e-4), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, events, targets):
    def loss_fn(m):
        return Loss(m(events), targets)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)                     # in-place; mutates model
    return loss
```

`optimizer.update(model, grads)` is the canonical flax≥0.11 signature (older
NNX used `optimizer.update(grads)`). `nnx.Optimizer(..., wrt=nnx.Param)` is
required — the `wrt=` keyword is not optional in flax 0.11+.

Prefer the batteries-included loop when you don't need a custom step.
`fit` passes each batch to your `loss_fn`/`eval_fn` as `(model, *batch)`, and
`spyx.data` loaders yield `State(obs, labels)` namedtuples with **bit-packed**
`obs` — so unpack inside the closure:

```python
from spyx.optimize import fit

def unpack(o):
    return jnp.unpackbits(o, axis=1)[:, :128].astype(jnp.float32)

history = fit(
    model, optax.lion(3e-4),
    loss_fn=lambda m, o, y: Loss(m(unpack(o)), y),
    train_iter=shd_dl.train_epoch, epochs=50,
    eval_iter=shd_dl.test_epoch,
    eval_fn=lambda m, o, y: Acc(m(unpack(o)), y),
)
```

## 3. Data loaders

`spyx.loaders` was renamed to **`spyx.data`** and now uses **Google Grain**.
Loaders are Python **iterables** of batches instead of returning one big array,
constructor args are keyword-only, and observations are **bit-packed** along
time (unpack before use).

```python
# OLD
shd_dl = spyx.loaders.SHD_loader(256, 128, 128)        # positional
x, y = shd_dl.train_epoch(key)                         # full arrays, takes a key
```

```python
# NEW
shd_dl = spyx.data.SHD_loader(batch_size=256, sample_T=128, channels=128)
for batch in shd_dl.train_epoch():                     # iterate; no key
    obs = jnp.unpackbits(batch.obs, axis=1)[:, :128].astype(jnp.float32)
    labels = batch.labels
```

Set `worker_count=` to parallelise the (CPU-bound) tonic event→frame conversion;
it defaults to a conservative `min(cores//2, 4)`.

## 4. Mixed precision

The `jmp` policy hooks are gone. Set dtypes directly on the layers instead:

```python
# OLD
import jmp
hk.mixed_precision.set_policy(snn.LIF, jmp.get_policy("half"))

# NEW — pass dtypes to the flax layers
nnx.Linear(in_dim, hidden, use_bias=False, dtype=jnp.bfloat16,
           param_dtype=jnp.float32, rngs=rngs)
```

## 5. NIR import/export

`spyx.nir` now walks NNX modules. `to_nir`/`from_nir` take and return
`nnx.Module` instances rather than Haiku `params` dicts; see
[the NIR how-to](nir.md).

## Checklist

- [ ] `import haiku as hk` → `from flax import nnx`; drop `jmp`.
- [ ] Every layer construction takes `rngs=`; `nnx.Linear` takes `in_features`.
- [ ] `hk.transform`/`.init`/`.apply` removed; instantiate the module directly.
- [ ] `hk.DeepRNN` → `snn.Sequential`; `hk.dynamic_unroll` → `snn.run` (time-major).
- [ ] `jax.grad` on `apply` → `nnx.value_and_grad` on a `model`-taking closure.
- [ ] Optimiser → `nnx.Optimizer(model, tx, wrt=nnx.Param)` + `optimizer.update(model, grads)`.
- [ ] `@jax.jit` → `@nnx.jit` for functions that take a `model`.
- [ ] `spyx.loaders` → `spyx.data`; iterate loaders and `jnp.unpackbits` the obs.
- [ ] Python 3.11–3.14.
