# Research with Spyx

Spyx is built for research, not just deployment. This page explains what makes it
a good research platform, how the [`research/`](https://github.com/kmheckel/spyx/tree/main/research)
corpus is organised, how to add a study, and how an idea graduates from
`spyx.experimental` into the stable core.

## Why Spyx is built for research

Four properties make Spyx a fast place to try new ideas:

- **Shared associative-scan machinery.** Spiking neurons, state-space models, and
  phasor networks in Spyx are all expressed as linear recurrences over time, so
  they share the same `jax.lax.associative_scan` / chunked-scan backbone. A
  reset-free spiking neuron ([`spyx.nn.PSU_LIF`](../reference/nn.md)), a diagonal
  SSM ([`spyx.ssm.S5Diag`](../reference/ssm.md)), and a resonate-and-fire phasor
  ([`spyx.phasor.ResonateFire`](../reference/phasor.md)) are variations on one
  parallel-prefix theme. That makes cross-pollination — porting an SSM trick to a
  spiking neuron, say — a small change rather than a rewrite. See
  [Parallel spiking neurons](parallel-spiking-neurons.md) for the worked example.
- **Measurement is first-class.** [`spyx.bench`](../reference/bench.md) gives you
  comparable numbers — median forward and forward+backward latency, throughput,
  peak memory, XLA-cost FLOPs / MFU, and a spike-rate energy proxy — so a claim
  about "faster" or "cheaper" is backed by a table, not a vibe. See the
  [benchmarking how-to](../how-to/benchmarking.md).
- **A staging area for unstable ideas.** [`spyx.experimental`](../reference/experimental.md)
  lets a new neuron or memory block ship, be tested, and be depended on
  *explicitly* — without freezing its API or destabilising the core.
- **Efficiency you can quantify.** [`spyx.quant`](../reference/quant.md) applies
  int8 / int4 / ternary quantization-aware training; because spike activations are
  binary, weight quantization is lossless on `{0, 1}` inputs, so you can study the
  accuracy/efficiency frontier honestly (recurrent/einsum paths stay fp32).

Underneath, the [design principles](design.md) — JAX-first, NNX modules,
surrogate gradients as `jax.custom_gradient`, a functional core — mean your idea
composes with `jit`, `vmap` (population sweeps, per-seed runs), and Optax without
special-casing.

## The `research/` corpus

The repository ships a [`research/`](https://github.com/kmheckel/spyx/tree/main/research)
directory: a home for research done *with* Spyx, so a paper, an experiment, or a
new idea lives next to runnable code, can be reproduced by someone else, and can
be extended without archaeology.

Every study is a self-contained folder that copies the shared
[`_template/README.md`](https://github.com/kmheckel/spyx/blob/main/research/_template/README.md)
and declares which of three **buckets** it belongs to:

| Bucket | Directory | What belongs here |
|---|---|---|
| **Reproductions** | `reproductions/` | Faithfully re-implement a published result and check whether it holds in Spyx (and how it compares to other frameworks). One study = one claim reproduced or refuted. |
| **Extensions** | `extensions/` | Take a published method further: a new dataset, an ablation, a scaling sweep, a different optimizer or neuron model. |
| **New research** | `new/` | Novel ideas with no paper yet — e.g. new parallelizable spiking neurons. Becomes an Extension/Reproduction target once published. |

Existing studies to read for orientation:

- [`new/parallel_spiking_neurons/`](https://github.com/kmheckel/spyx/tree/main/research/new/parallel_spiking_neurons)
  — reset-free parallel neurons (`PSU_LIF`, `ResonateFire`) vs. `LIF`, building on
  the **Stochastic Parallelizable Spiking Neurons** prior art in
  [`SPSN/`](https://github.com/kmheckel/spyx/tree/main/research/SPSN).
- [`new/ssm_to_spiking_transfer/`](https://github.com/kmheckel/spyx/tree/main/research/new/ssm_to_spiking_transfer)
  — transferring state-space dynamics into spiking models.
- [`new/pretrain_finetune_curriculum/`](https://github.com/kmheckel/spyx/tree/main/research/new/pretrain_finetune_curriculum)
  — pretrain/finetune curricula for SNNs.
- [`new/raven_sparse_memory_recall/`](https://github.com/kmheckel/spyx/tree/main/research/new/raven_sparse_memory_recall)
  — high-recall sequence modeling with the routing-slot memory in
  [`spyx.experimental.raven`](../reference/experimental.md).
- [`reproductions/qs5/`](https://github.com/kmheckel/spyx/tree/main/research/reproductions/qs5)
  — a quantized-S5 reproduction.

The reference machine for parallel-neuron work is an **AMD Radeon 8060S
(gfx1151)** on ROCm. Timing numbers are meaningless without recording the
accelerator, driver/runtime, and JAX version.

## Adding a study

1. Pick a bucket (`reproductions/`, `extensions/`, or `new/`) and create a folder
   with a short, descriptive name.
2. Copy [`_template/README.md`](https://github.com/kmheckel/spyx/blob/main/research/_template/README.md)
   into it and fill in **every** section — Title, Paper, Claim under test, Method,
   Spyx modules used, How to run, Results, Findings, Reproducibility. Keep the
   headers; write `N/A` if one genuinely doesn't apply. The template is the
   contract: a study tests **one** claim well rather than many poorly.
3. Fill the Results table with [`spyx.bench`](../reference/bench.md) numbers
   (latency, throughput, peak memory, spike-rate energy proxy) so results are
   comparable across studies. See the [benchmarking how-to](../how-to/benchmarking.md).
4. **Record reproducibility**: every RNG seed (`jax.random.PRNGKey`, data-shuffle
   seed, NumPy seed), the accelerator + driver/runtime + JAX version, and the Spyx
   commit hash. A study that can't be re-run with the same seed isn't reproducible.
5. Report honestly — include the failing runs, not only the good ones.

## Promoting experimental work into the core

[`spyx.experimental`](../reference/experimental.md) is the staging area: it lets a
prototype ship and be depended on while its API is still moving. A piece graduates
into a stable module (`spyx.nn`, `spyx.ssm`, `spyx.phasor`, ...) when:

- **It has a study behind it.** A `research/` folder demonstrating the claim, with
  a [`spyx.bench`](../reference/bench.md) results table — not just code that runs.
- **The API has settled.** Signatures and numerical behaviour are stable enough to
  commit to backwards compatibility.
- **It's tested and documented.** Unit tests plus reference/how-to coverage.
- **It composes with the core.** Works under `jax.jit` / `vmap`, plugs into
  [`spyx.nn.Sequential`](../reference/nn.md) and `run`, and — where relevant —
  round-trips through [`spyx.nir`](../reference/nir.md).

`PSU_LIF` (a reset-free parallel LIF, physically in `spyx.nn`) and `ResonateFire`
(physically in `spyx.phasor`) are the model of this: promoted into stable modules
for their supported implementations, yet still surfaced under `spyx.experimental`
as their research entry points. When you promote something, move the
implementation into the stable module, keep (or add) the experimental re-export if
it's still research-facing, and update the [module map](../index.md) and reference
pages.
