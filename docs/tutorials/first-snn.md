# Your first SNN

In this tutorial you will build, train, and evaluate a spiking neural network (SNN) from scratch with Spyx. By the end you will have:

- a working Spyx installation,
- a three-layer spiking network for the [Spiking Heidelberg Digits](https://compneuro.net/posts/2019-spiking-heidelberg-digits/) (SHD) audio dataset,
- a trained model, and an understanding of how state, time, and gradients flow through it.

No prior SNN experience is needed, but you should be comfortable with Python and have seen JAX or Flax before. If you want the conceptual background first, read the [SNN primer](../explanation/snn-primer.md) — this tutorial focuses on doing.

## 1. Set up your environment

Create a fresh project and install Spyx with the data-loading extra (we need it for the SHD dataset):

```bash
uv init spyx-tutorial && cd spyx-tutorial
uv add "spyx[loaders]"
```

The default `jax` dependency runs on CPU, which is fine for this tutorial. If you have a GPU, install the matching JAX wheel — see [How to install Spyx](../how-to/install.md).

Check that everything imports:

```python
import spyx
print(spyx.__version__)
```

## 2. Build the network

An SNN in Spyx is an ordinary Flax NNX module. We'll stack:

1. a `Linear` layer projecting the 128 input channels to 64 hidden units,
2. a `LIF` (leaky integrate-and-fire) layer — the spiking nonlinearity,
3. a `Linear` + `LI` readout, a non-spiking leaky integrator whose voltage we use as class logits.

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

Two things to notice:

- `nnx.Linear` comes straight from Flax; the spiking layers come from `spyx.nn`. Anything following the `(x, state) -> (out, new_state)` contract drops into `snn.Sequential`.
- `activation=spyx.axn.triangular()` selects the *surrogate gradient* — the smooth stand-in used for the spike's derivative during backprop. Spyx ships several ([`spyx.axn`](../reference/axn.md)); triangular is a solid default for SHD.

## 3. Run it over time

Spiking networks are stateful: each neuron carries a membrane voltage between timesteps. A single forward call advances the network by **one** timestep:

```python
import jax.numpy as jnp

state = model.initial_state(batch_size=4)
x_t = jnp.zeros((4, 128))               # one timestep of input
out, state = model(x_t, state)
```

Real inputs are spike *trains* — sequences of 0/1 tensors. `spyx.nn.run` scans the model over a time-major `(T, B, C)` array with `jax.lax.scan`:

```python
T, B = 128, 4
x = jnp.zeros((T, B, 128))              # placeholder spike train
traces, final_state = snn.run(model, x)
print(traces.shape)                      # (128, 4, 20)
```

`traces` holds the readout layer's voltage at every timestep. The network's "answer" is the class whose voltage integrates highest over the whole sequence.

## 4. Load the data

The SHD dataset contains spoken digits (0–9 in English and German, 20 classes) recorded through an artificial cochlea, so each sample is already a spike train. Spyx wraps it in a [Grain](https://github.com/google/grain)-based loader:

```python
dl = spyx.data.SHD_loader(batch_size=256, sample_T=128, channels=128)
```

The first call downloads the dataset (~500 MB) to `./data`. Batches arrive with spikes *bit-packed* along the time axis to save memory, so we unpack them into dense `(B, T, C)` float tensors:

```python
SAMPLE_T = 128

def unpack(batch_obs):
    obs = jnp.asarray(batch_obs)
    return jnp.unpackbits(obs, axis=1)[:, :SAMPLE_T, :].astype(jnp.float32)

def train_iter():
    for batch in dl.train_epoch():
        yield unpack(batch.obs), jnp.asarray(batch.labels)

def eval_iter():
    for batch in dl.test_epoch():
        yield unpack(batch.obs), jnp.asarray(batch.labels)
```

Peek at one batch to make sure it looks right:

```python
events, labels = next(train_iter())
print(events.shape, labels.shape)        # (256, 128, 128) (256,)
print(float(events.mean()))              # a small number — spikes are sparse!
```

## 5. Define the loss and the forward pass

We can't take a max over a single timestep — the signal is spread across time. Instead we sum the readout voltages over the time axis and apply cross-entropy to that integral. `spyx.fn` provides this as a factory:

```python
Loss = spyx.fn.integral_crossentropy()   # (traces, targets) -> scalar
Acc = spyx.fn.integral_accuracy()        # (traces, targets) -> (accuracy, preds)
```

Our loader yields batch-major `(B, T, C)` data while `snn.run` is time-major, so the forward function transposes on the way in and out:

```python
def forward(m, x_BTC):
    x_TBC = jnp.transpose(x_BTC, (1, 0, 2))
    traces, _ = snn.run(m, x_TBC)
    return jnp.transpose(traces, (1, 0, 2))

def loss_fn(m, events, targets):
    return Loss(forward(m, events), targets)

def eval_fn(m, events, targets):
    traces = forward(m, events)
    acc, _ = Acc(traces, targets)
    return acc, Loss(traces, targets)
```

## 6. Train

`spyx.optimize.fit` wraps the canonical `nnx.Optimizer` + `nnx.value_and_grad` loop (you'll write that loop by hand in [How to train a model](../how-to/train.md) — for now, let the helper drive):

```python
import optax
import spyx.optimize as opt

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

You should see the training loss fall steadily and `eval_acc` climb well above the 5% chance level within the first few epochs, ending somewhere around 70–80% after 30 epochs (exact numbers vary with the seed):

```text
0 {'train_loss': 2.97..., 'eval_acc': 0.31..., 'eval_loss': 2.63...}
1 {'train_loss': 2.51..., 'eval_acc': 0.46..., 'eval_loss': 2.35...}
...
```

Gradients flow *backwards through every timestep* of the scan (backpropagation through time), with the surrogate gradient standing in for the non-differentiable spike at each step.

## 7. Inspect the trained model

Confirm the network actually spikes and classifies:

```python
events, labels = next(eval_iter())
traces = forward(model, events)
acc, preds = Acc(traces, labels)
print(f"batch accuracy: {float(acc):.2%}")
print("first 10 predictions:", preds[:10])
print("first 10 labels:     ", labels[:10])
```

That's it — you have trained a spiking neural network with surrogate gradient descent, entirely JIT-compiled by JAX.

## Where to go next

- [Surrogate Gradient Tutorial](../examples/surrogate_gradient/SurrogateGradientTutorial.ipynb) — the extended notebook version of this walkthrough.
- [How to quantize a model](../how-to/quantize.md) — compress the network you just trained to int8 or ternary weights.
- [How to export to NIR](../how-to/nir.md) — take it to neuromorphic hardware.
- [SNN primer](../explanation/snn-primer.md) — the theory behind what you just did.
