# How to use SSM and phasor layers

To model long-range temporal structure beyond what LIF dynamics capture, combine your spiking stack with the sequence layers in [`spyx.ssm`](../reference/ssm.md) (LRU, S5Diag, Mamba, MambaBlock, ChunkedSSM) or the complex-valued layers in [`spyx.phasor`](../reference/phasor.md).

The key contract difference: Spyx neurons are **stepwise** (`(x_t, state) -> (out, state)`, driven by `spyx.nn.run`), while SSM layers are **whole-sequence** (`(T, B, d_model) -> (T, B, d_model)`, internally parallelised with `jax.lax.associative_scan`). You compose them by running the spiking front-end first, then feeding its spike train to the SSM.

## Use a single SSM layer

```python
import jax.numpy as jnp
from flax import nnx
from spyx import ssm

rngs = nnx.Rngs(0)
layer = ssm.LRU(d_model=8, d_state=64, rngs=rngs)   # Linear Recurrent Unit

u = jnp.ones((128, 32, 8))     # time-major (T, B, d_model)
y = layer(u)                   # same shape, real-valued
```

The available layers (all time-major, all trainable with a stock `optax.adam` + `nnx.Optimizer` loop):

| Layer | What it is |
|---|---|
| `ssm.LRU(d_model, d_state, ...)` | Linear Recurrent Unit (Orvieto et al., 2023); stability enforced by construction. |
| `ssm.S5Diag(d_model, d_state, ...)` | Diagonal S4D/S5 with HiPPO-LegS init and learnable log-step — best for long-range tasks. |
| `ssm.Mamba(d_inner, d_state, ...)` | Selective SSM core (input-dependent Δ, B, C). |
| `ssm.MambaBlock(d_model, d_state, d_conv, expand, ...)` | Full Mamba block: in-proj → depthwise conv → SSM → gate → out-proj. |
| `ssm.ChunkedSSM(inner, outer, chunk_size=..., pool=...)` | H-Net-style hierarchy: an inner SSM per timestep plus an outer SSM over chunk summaries. |

`ChunkedSSM` wraps any two `(T, B, D) -> (T, B, D)` modules, so the inner/outer pair can mix layer types:

```python
inner = ssm.MambaBlock(d_model=8, d_state=4, rngs=rngs)
outer = ssm.LRU(d_model=8, d_state=8, rngs=rngs)
hnet = ssm.ChunkedSSM(inner, outer, chunk_size=4, pool="mean")  # T must divide by chunk_size
```

## Build a hybrid SNN → SSM stack

To add an SSM on top of a spiking front-end (the Linear → LIF → LRU pattern exercised in `tests/test_ssm.py`), run the spiking layers with `spyx.nn.run`, then apply the SSM to the resulting spike train:

```python
import spyx
import spyx.nn as snn

rngs = nnx.Rngs(0)
snn_front = snn.Sequential(
    nnx.Linear(4, 8, use_bias=False, rngs=rngs),
    snn.LIF((8,), activation=spyx.axn.triangular(), rngs=rngs),
)
ssm_layer = ssm.LRU(d_model=8, d_state=16, rngs=rngs)
readout = nnx.Linear(8, 3, use_bias=False, rngs=rngs)

T, B = 128, 32
u = jnp.ones((T, B, 4))                      # time-major input
spikes, _ = snn.run(snn_front, u)            # (T, B, 8) binary spikes
h = ssm_layer(spikes)                        # (T, B, 8) real features
logits = readout(h.sum(axis=0))              # (B, 3)
```

Gradients flow through the whole pipeline, so the stack trains end-to-end with the usual `nnx.value_and_grad` step — see [How to train a model](train.md). Wrap the three stages in one `nnx.Module` if you want a single trainable object.

To quantize the `nnx.Linear` layers around an SSM (the SSM's own B/C projections are raw params and stay fp32), use [`spyx.quant`](quantize.md); `scripts/ssm_demo.py` demonstrates both int8 and BitNet-ternary variants.

## Use phasor layers

Phasor networks (Bybee, Frady & Sommer, 2022) represent activations as unit-magnitude complex numbers and convert to single-spike-per-cycle trains at inference. Train in the continuous complex domain:

```python
from spyx import phasor

model = phasor.PhasorMLP(
    in_features=8, hidden_features=16, out_features=4, depth=2, rngs=rngs,
)
x = jnp.ones((32, 8))          # real inputs in [0, 1]
logits = model(x)              # real (32, 4) — trains with stock optax.adam
```

`PhasorLinear` stores its complex kernel as paired `kernel_re` / `kernel_im` float32 params, so gradients stay real and a plain `optax.adam` + `nnx.Optimizer` loop converges without Wirtinger-gradient surprises.

To run a trained phasor layer in the **spike domain**, wrap it in `SpikingPhasor`, which decodes phases from an incoming spike train, applies the layer, and re-emits spikes:

```python
linear = phasor.PhasorLinear(16, 16, rngs=rngs)
spiking = phasor.SpikingPhasor(linear, period_T=32)

theta = jnp.zeros((32, 16))                       # (B, features) phases
spikes_in = phasor.phase_to_spikes(theta, T=32)   # (T, B, features)
spikes_out = spiking(spikes_in)                   # (T, B, features)
```

The codec helpers `phase_to_spikes` / `spikes_to_phase` convert between phases in `(-π, π]` and one-spike-per-cycle trains; round-trip error is bounded by the bin size `2π / T`. See `scripts/phasor_demo.py` and the [Phasor Networks notebook](../examples/phasor/phasor_intro.ipynb) for end-to-end examples, and the [State-Space Models notebook](../examples/ssm/ssm_intro.ipynb) for SSM training runs.
