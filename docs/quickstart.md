# Quickstart

This page walks through a minimal end-to-end Spyx workflow: install, define an SNN, train it, and evaluate. It covers the pieces a newcomer will hit in their first 15 minutes with the library.

## Installation

Spyx uses [uv](https://github.com/astral-sh/uv) for dependency management. To add Spyx to an existing project:

```bash
uv add spyx
```

Or to install from source (with the development tooling):

```bash
git clone https://github.com/kmheckel/spyx
cd spyx
uv sync
```

As with anything built on JAX, install the right JAX wheel for your accelerator — see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). The default `jax` package runs on CPU; GPU / TPU builds are separate wheels.

### Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `spyx[loaders]` | `tonic`, `numba` | `SHD_loader`, `NMNIST_loader` |
| `spyx[quant]` | `qwix` (from GitHub) | `spyx.quant.quantize` (int8 / int4 / BitNet) |
| `spyx[docs]` | `mkdocs`, `mkdocs-material`, `mkdocstrings` | building these docs locally |

## Your first SNN

A feed-forward spiking network for the [Spiking Heidelberg Digits](https://compneuro.net/posts/2019-spiking-heidelberg-digits/) (SHD) dataset. It's three layers: an input `Linear`, a hidden LIF layer, a `Linear` + `LI` readout that integrates voltages across time.

```python
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
```

`nnx.Linear` and `nnx.Conv` layers come from Flax NNX; the spiking layers come from `spyx.nn`. Anything that follows the `(x, state) -> (out, new_state)` contract drops into `snn.Sequential`.

## Driving the SNN over time

Spyx SNNs are stateful: each forward call takes an input plus a per-layer state and returns an output plus the updated state. For a sequence of length `T`, `spyx.nn.run` does the scan for you:

```python
import jax.numpy as jnp

# Dense (T, B, C) spike train
T, B = 128, 32
x = jnp.zeros((T, B, 128))  # replace with your loader's output
traces, _final_state = snn.run(model, x)
# traces.shape == (T, B, 20)
```

`run` is time-major; if your loader yields `(B, T, C)` (as `spyx.data.SHD_loader` does after `jnp.unpackbits`), transpose before and after.

## Training with `spyx.optimize.fit`

The `spyx.optimize.fit` helper wraps the canonical `nnx.Optimizer` + `nnx.value_and_grad` loop, so you rarely need to hand-write it.

```python
import jax
import jax.numpy as jnp
import optax
import spyx
import spyx.nn as snn
import spyx.optimize as opt
from flax import nnx

# --- data ---
dl = spyx.data.SHD_loader(batch_size=256, sample_T=128, channels=128)
SAMPLE_T = 128

def unpack(batch_obs):
    # Loaders return packed uint8 (B, ceil(T/8), C). Recover dense spikes.
    obs = jnp.asarray(batch_obs)
    return jnp.unpackbits(obs, axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)

def train_iter():
    for batch in dl.train_epoch():
        yield unpack(batch.obs), jnp.asarray(batch.labels)

def eval_iter():
    for batch in dl.test_epoch():
        yield unpack(batch.obs), jnp.asarray(batch.labels)

# --- model ---
rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(128, 64, use_bias=False, rngs=rngs),
    snn.LIF((64,), activation=spyx.axn.triangular(), rngs=rngs),
    nnx.Linear(64, 20, use_bias=False, rngs=rngs),
    snn.LI((20,), rngs=rngs),
)

# --- loss + metric ---
Loss = spyx.fn.integral_crossentropy()
Acc = spyx.fn.integral_accuracy()

def forward(m, x_BTC):
    return jnp.transpose(snn.run(m, jnp.transpose(x_BTC, (1, 0, 2)))[0], (1, 0, 2))

def loss_fn(m, events, targets):
    return Loss(forward(m, events), targets)

def eval_fn(m, events, targets):
    traces = forward(m, events)
    acc, _ = Acc(traces, targets)
    return acc, Loss(traces, targets)

# --- train ---
history = opt.fit(
    model,
    optax.lion(3e-4),
    loss_fn,
    train_iter,
    epochs=30,
    eval_iter=eval_iter,
    eval_fn=eval_fn,
    on_epoch_end=lambda epoch, metrics: print(epoch, metrics),
)
```

## Quantization-aware training

Once the fp32 model converges, you can drop it into a quantized training loop with one line via `spyx.quant`:

```python
# Provide sample inputs so qwix can trace the module graph.
sample_x = next(iter(train_iter()))[0][:1]                # (1, T, C)
sample_state = model.core.initial_state(1) if hasattr(model, "core") else None

qmodel = spyx.quant.quantize(model, sample_x, sample_state)  # int8 W+A by default
```

By default only `nnx.Linear` / `nnx.Conv` layers are quantized; spiking dynamics stay fp32 to preserve the delicate reset arithmetic. See the [Quantization-Aware Training tutorial](examples/quantization/qat_intro.ipynb) for rule overrides including BitNet-style ternary weights.

## Exporting to NIR

For deployment on neuromorphic hardware:

```python
import spyx.nir as snir

nir_graph = snir.to_nir(
    model,
    input_shape={"input": (128,)},
    output_shape={"output": (20,)},
    dt=1.0,
)
# nir_graph is a standard nir.NIRGraph you can write to HDF5 or feed into
# a target-specific compiler.
```

The round-trip with `snir.from_nir` covers LIF, CuBaLIF, RLIF, and RCuBaLIF; see the [NIR Conversion tutorial](examples/nir/conversion.ipynb) for a worked example.

## Next steps

* [Surrogate Gradient Tutorial](examples/surrogate_gradient/SurrogateGradientTutorial.ipynb) — the extended SHD walkthrough.
* [Cartpole Evolution](examples/neuroevolution/cartpole_evo.ipynb) — neuroevolution on gymnax with `vmap` parallelism.
* [Quantization-Aware Training](examples/quantization/qat_intro.ipynb) — int8 + BitNet-style QAT on a Spyx SNN.
* [NIR Conversion](examples/nir/conversion.ipynb) — export to NIR and back.
* [API Reference](api.md) — exhaustive list of every public function and class.
