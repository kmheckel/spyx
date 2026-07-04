---
name: add-neuron-model
description: Implement a new spiking-neuron layer in spyx.nn following the library's state contract. Use when the user asks to "add a neuron model", "implement <X> neuron in Spyx", "port a neuron from snnTorch/Norse", or wants a custom LIF variant that plugs into Sequential and run.
---

# Add a neuron model to spyx.nn

Every spiking layer in Spyx is a `flax.nnx.Module` that obeys one contract, so
it drops into `spyx.nn.Sequential` and the time-major `spyx.nn.run` scan.
Match the existing neurons (`LIF`, `ALIF`, `CuBaLIF`, `RLIF`) exactly.

## The contract

A neuron layer MUST implement:

1. `__init__(self, hidden_shape: tuple, ..., *, rngs: nnx.Rngs)` — store shape,
   threshold, the surrogate `activation` (default to `_DEFAULT_ACTIVATION`),
   and register learnable dynamics as `nnx.Param`.
2. `__call__(self, x, state) -> (spikes, new_state)` — one timestep. `x` is the
   input from the previous layer; `state` is this layer's carry.
3. `initial_state(self, batch_size) -> state` — the zero/initial carry, shaped
   `(batch_size,) + self.hidden_shape` (a tuple/pytree if the neuron carries
   more than one variable, e.g. `CuBaLIF` carries `(V, I)`).

`run` threads the state for you across time; you only write the single step.

## Template (leaky integrate-and-fire, as the reference)

```python
class MyNeuron(nnx.Module):
    """One-line description (cite the paper if it's from one)."""

    def __init__(self, hidden_shape: tuple, beta=None, threshold=1.,
                 activation=None, *, rngs: nnx.Rngs):
        self.hidden_shape = hidden_shape
        self.threshold = threshold
        self.spike = activation if activation is not None else _DEFAULT_ACTIVATION
        if beta is None:
            self.beta = nnx.Param(
                nnx.initializers.truncated_normal(stddev=0.5)(rngs.params(), hidden_shape) + 0.25
            )
        else:
            self.beta = nnx.Param(jnp.full((), beta))

    def __call__(self, x, V):
        beta = jnp.clip(self.beta[...], 0, 1)     # keep decay in [0,1]
        spikes = self.spike(V - self.threshold)   # surrogate-grad Heaviside
        V = beta * V + x - spikes * self.threshold  # integrate, then soft reset
        return spikes, V

    def initial_state(self, batch_size):
        return jnp.zeros((batch_size,) + self.hidden_shape)
```

Key details, all load-bearing:

- **Read a Param with `self.beta[...]`**, not `self.beta`. `nnx.Param` is a
  wrapper; `[...]` unwraps the array.
- **Fire before reset.** Compute `spikes` from the *pre-update* voltage, then
  subtract `spikes * threshold` — this is the standard soft-reset. A recurrent
  neuron (`RLIF`) adds a recurrent `nnx.Linear(V_prev)` term to `x`.
- **Multi-variable state** (synaptic current + voltage, adaptation, etc.):
  return and accept a tuple; `initial_state` returns the matching tuple. See
  `CuBaLIF` (`VI` carry) and `ALIF` (`VT` carry).
- **The surrogate gradient does the backward pass.** Never hand-write a
  `custom_gradient` in the neuron — take an `activation` from `spyx.axn`.

## Where to put it and wire it up

1. Add the class to `src/spyx/nn.py` next to the related neurons.
2. If it should round-trip through NIR, add import/export in `src/spyx/nir.py`
   (follow the `LIF` / `RLIF` handling; recurrent neurons export as an inner
   `NIRGraph` subgraph).
3. **Docstrings render into the reference** — `docs/reference/nn.md` uses
   `::: spyx.nn`, so a well-formed docstring is the whole reference entry.

## Tests (required)

Add cases to `tests/test_nn_nnx.py` mirroring the existing neurons:

- shape/dtype of `__call__` output and `initial_state`,
- runs through `spyx.nn.run` inside a `Sequential`,
- a tiny gradient step (loss decreases or gradients are finite),
- NIR round-trip if you added export (`tests/test_nir.py`).

Then:

```bash
uv run pytest tests/test_nn_nnx.py -q
uv run ruff check
uv run python scripts/smoke_notebook_apis.py   # if you touched any tutorial API
```

## Don'ts

- Don't break the `(x, state) -> (out, new_state)` shape — `run` will fail
  cryptically. If the layer is stateless, it's not a neuron; use `nnx.Linear`.
- Don't leave `beta`/decay unclipped — an unstable leak diverges under BPTT.
- Don't add biases to the surrounding `Linear` layers by default; SNNs lean on
  the threshold for offset (`use_bias=False`).
