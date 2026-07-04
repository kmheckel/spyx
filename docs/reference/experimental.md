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
