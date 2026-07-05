# spyx.experimental

Research-stage building blocks that are **not** part of the stable Spyx surface.
Everything here is tested and usable, but the contract is different from the rest
of the library.

!!! warning "Stability contract"
    The APIs in `spyx.experimental` — and in some cases their *numerical
    behaviour* — may change **without a deprecation cycle** as the underlying
    research matures. Anything you depend on for production or a long-lived
    experiment should come from the stable top-level modules
    ([`spyx.nn`](nn.md), [`spyx.ssm`](ssm.md), [`spyx.phasor`](phasor.md),
    [`spyx.nir`](nir.md), [`spyx.bench`](bench.md), [`spyx.quant`](quant.md),
    [`spyx.data`](data.md), [`spyx.optimize`](optimize.md)).

    The rule of thumb: **import experimental things from `spyx.experimental`** so
    the dependency is explicit; rely on the top-level modules for stable work.
    See [Research with Spyx](../explanation/research.md) for how things graduate
    from here into the core.

## What's here

| Symbol | Kind | Notes |
|---|---|---|
| `spyx.experimental.PSU_LIF` | Neuron | Reset-free parallel LIF. *Physically defined in [`spyx.nn`](nn.md)*, surfaced here as its supported experimental entry point. |
| `spyx.experimental.ResonateFire` | Neuron | Complex resonate-and-fire oscillatory neuron. *Physically defined in [`spyx.phasor`](phasor.md)*. |
| [`spyx.experimental.raven`](#spyxexperimentalraven) | Module | Routing-slot memory (`RavenRSM`), spiking sibling (`SpikingSlotMemory`), `SlotRouter`, and the `make_recall_batch` MQAR generator. |
| [`spyx.experimental.compress`](#spyxexperimentalcompress) | Module | Bit-packed activation storage for memory-efficient BPTT. |
| [`spyx.experimental.stochastic`](#spyxexperimentalstochastic) | Module | Stochastic (Bernoulli-spiking) and parallelizable prototypes: `SPSN`, `StochasticAssociative{LIF,CuBaLIF}`, and the `sigmoid_bernoulli` activations. |
| [`spyx.experimental.hybrid`](#spyxexperimentalhybrid) | Module | The 0+1 hybrid trainer: surrogate gradient + antithetic-NES correction projected orthogonal to the surrogate (`hybrid_gradient`, `make_hybrid_train_step`, `es_gradient`, `hybrid_diagnostics`). |
| [`spyx.experimental.zoo`](#spyxexperimentalzoo) | Package | Runnable reference recipes keyed by application (control / classification / language) and tagged by training method × architecture (`REGISTRY`, `list_recipes`, `get`). |
| [`spyx.experimental.onnx`](#spyxexperimentalonnx) | Module | Export a spiking model to ONNX — per-timestep step, or the whole `spyx.nn.run` loop as a native ONNX `Scan`/`Loop`. Conversion deps imported lazily. |

Related research studies live under
[`research/new/`](https://github.com/kmheckel/spyx/tree/main/research/new) in the
repository.

## Re-exported neurons

These two are physically defined in stable modules and re-exported here so the
experimental surface is discoverable in one place.

::: spyx.experimental.PSU_LIF

::: spyx.experimental.ResonateFire

## spyx.experimental.raven

::: spyx.experimental.raven

## spyx.experimental.compress

::: spyx.experimental.compress

## spyx.experimental.stochastic

::: spyx.experimental.stochastic
    options:
      show_if_no_docstring: true

## spyx.experimental.hybrid

Surrogate-gradient descent corrected by an orthogonalised evolutionary term. See
[Surrogate gradients & Gaussian smoothing](../explanation/surrogate-gradients-and-gaussian-smoothing.md)
for the theory and [Training methods](../explanation/training-methods.md) for
where it fits.

::: spyx.experimental.hybrid

## spyx.experimental.zoo

Runnable recipes tagged by application × training method × architecture. Each
`Recipe` exposes `build` / `synthetic_batch` / `demo` on synthetic data; browse
with `list_recipes(application=..., method=...)`.

::: spyx.experimental.zoo

## spyx.experimental.onnx

::: spyx.experimental.onnx
