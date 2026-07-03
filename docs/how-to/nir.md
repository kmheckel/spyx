# How to export and import models via NIR

To move a trained Spyx model onto neuromorphic hardware (or into another SNN framework), convert it to a [Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/) graph with [`spyx.nir`](../reference/nir.md). The conversion is bidirectional: `to_nir` exports, `from_nir` imports.

Supported layers: `nnx.Linear`, `nnx.Conv`, `nnx.Flatten`, `spyx.nn.SumPool`, `IF`, `LIF`, `CuBaLIF`, and the recurrent variants `RIF`, `RLIF`, `RCuBaLIF` (exported as NIR RNN subgraphs). The model must be a `spyx.nn.Sequential` (or a single layer).

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
model = snir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(0))
```

`from_nir` returns a `spyx.nn.Sequential` with weights and time constants loaded from the graph, including RNN subgraphs (imported as `RIF` / `RLIF` / `RCuBaLIF`). Drive it with `spyx.nn.run` as usual:

```python
state = model.initial_state(batch_size)
out, state = model(x_t, state)
```

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
imported = snir.from_nir(nir_graph, dt=1, rngs=nnx.Rngs(1))

x = jnp.ones((5, 128))
out_a, _ = model(x, model.initial_state(5))
out_b, _ = imported(x, imported.initial_state(5))
assert jnp.allclose(out_a, out_b)
```

For a full worked example, see the [NIR Conversion notebook](../examples/nir/conversion.ipynb) and the NIR [Braille RSNN](../examples/nir/rsnn/braille_spyx.ipynb) / [N-MNIST SCNN](../examples/nir/scnn/nmnist_spyx.ipynb) notebooks.
