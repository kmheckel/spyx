# How to export and import models via NIR

To move a trained Spyx model onto neuromorphic hardware (or into another SNN framework), convert it to a [Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/) graph with [`spyx.nir`](../reference/nir.md). The conversion is bidirectional: `to_nir` exports, `from_nir` imports.

Supported layers (round-trip with numerical parity, covered by tests):
`nnx.Linear`, `nnx.Conv`, `spyx.nn.Flatten`, `IF`, `LIF`, `CuBaLIF`, and the
recurrent variants `RIF`, `RLIF`, `RCuBaLIF` (exported as NIR RNN subgraphs).
The model must be a `spyx.nn.Sequential` (or a single layer).

Convolutional models — including spiking convolutions (a neuron directly after
a conv, over the spatial feature map) — round-trip. NIR is channels-first
`(C, H, W)` while Spyx is channels-last `(B, H, W, C)`; `spyx.nir` bridges the
two, so a conv-following neuron in Spyx uses a channels-last `(H, W, C)` state.

## Export a feed-forward model

```python
from flax import nnx
import spyx.nn as snn
import spyx.nir as snir

rngs = nnx.Rngs(0)
model = snn.Sequential(
    nnx.Linear(128, 64, use_bias=False, rngs=rngs),
    snn.LIF((64,), beta=0.8, rngs=rngs),
    nnx.Linear(64, 20, use_bias=False, rngs=rngs),
    snn.LI((20,), rngs=rngs),
)

nir_graph = snir.to_nir(
    model,
    input_shape={"input": (128,)},
    output_shape={"output": (20,)},
    dt=1.0,
)
```

`nir_graph` is a standard `nir.NIRGraph`. Save it to HDF5 with the `nir` package (a core Spyx dependency) and hand the file to any NIR-compatible toolchain:

```python
import nir
nir.write("model.nir", nir_graph)
```

Two conventions to be aware of:

- `dt` is the simulation timestep used to convert Spyx's discrete decay factors into NIR's continuous time constants: `tau = dt / (1 - beta)`. Use the same `dt` on export and import.
- Layers `to_nir` doesn't recognise are **skipped with a printed warning**, not an error — check the output when exporting custom stacks.

## Export a recurrent model

Recurrent layers need no special handling — `RIF` / `RLIF` / `RCuBaLIF` are exported automatically as 4-node NIR RNN subgraphs (input → W_rec ⇄ LIF → output), the topology other NIR frameworks expect:

```python
model = snn.Sequential(
    nnx.Linear(8, 12, use_bias=False, rngs=rngs),
    snn.RLIF((12,), beta=0.85, rngs=rngs),
)
nir_graph = snir.to_nir(model, {"input": (8,)}, {"output": (12,)}, dt=1)
```

The recurrent weight matrix and decay parameters round-trip exactly (see `tests/test_nir.py`).

## Import a NIR graph

To build a Spyx model from a NIR graph (e.g. one trained in snnTorch or Norse):

```python
import nir
from flax import nnx
import spyx.nir as snir

nir_graph = nir.read("model.nir")

# from_nir builds the model *and runs it* on time-major input (T, B, ...),
# returning (model, outputs). outputs has shape (T, B, ...).
model, outputs = snir.from_nir(nir_graph, input_data, dt=1, rngs=nnx.Rngs(0))
```

`from_nir` reconstructs a `spyx.nn.Sequential` with weights and time constants loaded from the graph (including RNN subgraphs, imported as `RIF` / `RLIF` / `RCuBaLIF`), scans it over the leading time axis of `input_data`, and returns `(model, outputs)`. Pass `return_all_states=True` to also get the per-layer final states:

```python
model, (outputs, states) = snir.from_nir(
    nir_graph, input_data, dt=1, return_all_states=True
)
```

Reuse the returned `model` directly for further inference with `spyx.nn.run`.

!!! note "Graph topology"
    The importer assumes a linear input → ... → output chain of edges. Arbitrary branching NIR graphs are not supported.

## Fix parameter ordering after external optimizers

Some optimization libraries permute the keys of a parameter PyTree. If you trained with such a tool and export produces mismatched weights, restore the original key order first:

```python
trained = snir.reorder_layers(init_params, trained_params)
```

## Verify the round-trip

Before deploying, confirm that export → import preserves behaviour:

```python
import spyx.nn as snn

x = jnp.ones((10, 5, 128))  # (T, B, in)

ref_out, _ = snn.run(original_model, x)
_, imported_out = snir.from_nir(nir_graph, x, dt=1, rngs=nnx.Rngs(1))
assert jnp.allclose(ref_out, imported_out, atol=1e-5)
```

For a full worked example, see the [NIR Conversion notebook](../examples/nir/conversion.ipynb) and the NIR [Braille RSNN](../examples/nir/rsnn/braille_spyx.ipynb) / [N-MNIST SCNN](../examples/nir/scnn/nmnist_spyx.ipynb) notebooks.
