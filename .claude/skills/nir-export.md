---
name: nir-export
description: Export a trained Spyx model to NIR (Neuromorphic Intermediate Representation) or import a NIR graph into Spyx. Use when the user asks to "deploy to neuromorphic hardware", "export to NIR", "convert my model for Loihi/SpiNNaker", "load a NIR graph", or wants interop with other SNN toolchains.
---

# Export / import via NIR

`spyx.nir` bridges Spyx and the [Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/),
the common exchange format for SNN toolchains (Norse, snnTorch, Lava/Loihi,
SpiNNaker, etc.). Two entry points: `to_nir` and `from_nir`.

## Export: Spyx → NIR

```python
import spyx.nir as sxnir

graph = sxnir.to_nir(
    model,                       # an nnx.Sequential (or single layer)
    input_shape=(channels,),
    output_shape=(n_classes,),
    dt=1,                        # discretization step used for beta <-> tau
)
# graph is a nir.NIRGraph; serialize with nir.write / the nir library.
```

Supported layers: `nnx.Linear` (→ `Affine`/`Linear`), `nnx.Conv` (→ `Conv2d`),
and the neuron models `IF`, `LIF`, `CuBaLIF`, plus the recurrent `RIF` / `RLIF`
/ `RCuBaLIF`. Recurrent neurons export as an inner `NIRGraph` subgraph — that's
how NIR represents the feedback loop.

## Import: NIR → Spyx

```python
from flax import nnx
model = sxnir.from_nir(nir_graph, dt=1.0, rngs=nnx.Rngs(0))
```

`from_nir` reconstructs the `nnx.Sequential`, mapping NIR nodes back to Spyx
neurons and lifting RNN subgraphs into `RLIF` / `RCuBaLIF`.

## The `dt` ↔ `beta`/`tau` convention (read this)

Spyx's LIF recurrence is `V = β·V + x - spikes·threshold` with **no `(1-β)`
scaling on the input term**. `to_nir` emits `β = 1 - dt/τ` and `from_nir`
reconstructs it symmetrically, so **every Spyx → NIR → Spyx round-trip is
exact** (this is what `tests/test_nir.py` checks).

Caveat for **foreign** NIR graphs (authored by another toolchain with a
different discretization convention): the imported Spyx model will have
time-scaled leak but unscaled input drive. This is a known limitation, not a
bug — do NOT "fix" it by scaling imported weights by `(1-β)`, because that
would diverge from Spyx's own forward semantics and break the round-trip. If a
user hits mismatched dynamics importing an external graph, explain the
convention rather than patching the import path. (See the discussion on PR #40,
`src/spyx/nir.py:189`.)

## Layer ordering after training

`reorder_layers(init_params, trained_params)` realigns parameter ordering when
needed — use it if an exported graph's layers come out permuted relative to the
freshly-initialized model.

## Verify the round-trip

Always confirm before trusting a deployment export:

```python
graph = sxnir.to_nir(model, (C,), (N,), dt=1)
model2 = sxnir.from_nir(graph, dt=1.0, rngs=nnx.Rngs(0))
# run both on the same input; outputs should match to fp tolerance.
```

The how-to guide `docs/how-to/nir.md` and `tests/test_nir.py` are the
reference. `docs/examples/nir/conversion.ipynb` is a full worked notebook.
