# Research with Spyx

Spyx ships a [`research/`](https://github.com/kmheckel/spyx/tree/main/research)
directory in the repository: a home for research done *with* Spyx, so a paper, an
experiment, or a new idea can live next to runnable code, be reproduced by someone
else, and be extended without archaeology.

Every study is a self-contained folder that copies a shared template (Title,
Paper, Claim under test, Method, Spyx modules used, How to run, Results table,
Findings, Reproducibility) and declares which of three kinds it is:

- **Reproductions** — faithfully re-implement a published result and check whether
  it holds in Spyx (and how it compares to other frameworks).
- **Extensions** — take a published method further: a new dataset, an ablation, a
  scaling sweep, a different optimizer or neuron model.
- **New research** — novel ideas that do not yet have a paper, such as new
  parallelizable spiking neurons.

## Studies relevant to these docs

The [parallel spiking neurons](parallel-spiking-neurons.md)
([`spyx.nn.PSU_LIF`](../reference/nn.md),
[`spyx.phasor.ResonateFire`](../reference/phasor.md)) build on the
**Stochastic Parallelizable Spiking Neurons** prior art in
[`research/SPSN/`](https://github.com/kmheckel/spyx/tree/main/research/SPSN) and
are evaluated in
[`research/new/parallel_spiking_neurons/`](https://github.com/kmheckel/spyx/tree/main/research/new).
Those studies use [`spyx.bench`](../reference/bench.md) for their latency,
throughput, and memory numbers — see the
[benchmarking how-to](../how-to/benchmarking.md).

The reference machine for parallel-neuron work is an **AMD Radeon 8060S
(gfx1151)** on ROCm. Timing numbers are meaningless without recording the
accelerator, driver/runtime, and JAX version.

Browse the full index — reproductions, extensions, scaling sweeps, and
cross-framework comparisons — in the
[research README on GitHub](https://github.com/kmheckel/spyx/tree/main/research).
