# Spyx Research

This directory is the home for **research done with [Spyx](../README.md)** — the
JAX / Flax NNX spiking-neural-network library. It exists so that a paper, an
experiment, or a new idea can live next to runnable code, be reproduced by
someone else, and be extended without archaeology.

## Purpose

Three kinds of work live here, and every study should declare which kind it is:

1. **Reproductions** — faithfully re-implement a published result and check
   whether it holds in Spyx (and how it compares to other frameworks).
2. **Extensions** — take a published method and push it somewhere new: a new
   dataset, a scaling sweep, an ablation, a different optimizer or neuron model.
3. **New research** — novel ideas that do not yet have a paper, e.g. new
   parallelizable spiking neurons.

## Taxonomy

| Bucket | Directory | What belongs here |
| --- | --- | --- |
| Reproductions | [`reproductions/`](reproductions/README.md) | Faithful re-implementation of a paper's headline result. One study = one claim reproduced (or refuted). |
| Extensions | [`extensions/`](extensions/README.md) | A published method taken further: new data, ablations, scaling sweeps, optimizer/neuron swaps. |
| New research | [`new/`](new/README.md) | Novel ideas with no paper yet. Becomes an Extension or Reproduction target once published. |
| Template | [`_template/`](_template/README.md) | The study template every new study copies. |

Every study is a self-contained folder that copies [`_template/README.md`](_template/README.md)
and fills it in. That template is the contract: Title, Paper, Claim under test,
Method, Spyx modules used, How to run, Results table, Findings, Reproducibility.

## Process (agentic research + promotion gate)

Research here runs on a loop with a human gate. The standing agenda — pillars,
flagship studies, and honest field constraints — lives in **[PROGRAM.md](PROGRAM.md)**.

- **[`/research-scout`](../.claude/skills/research-scout.md)** — surveys current
  papers and proposes backlog **candidates**, each classified `replication` /
  `extension` / `novelty` against the taxonomy above. You triage candidates to `ready`.
- **[BACKLOG.md](BACKLOG.md)** — the queue of falsifiable claims to study. Current
  focus track: **quantization & efficient architectures.**
- **[`/research-study`](../.claude/skills/research-study.md)** — the runner: pulls the
  top backlog item, builds and adversarially verifies one study, opens a PR + ledger
  row, and **stops at the gate**. Two modes — **scheduled web** (smoke/CPU only) and
  **local loop** on the AMD GPU (bounded small-scale *real* runs). Never edits core,
  never promotes, never merges.
- **[FINDINGS.md](FINDINGS.md)** — the ledger: every study's honest verdict and its
  promotion status. Your review surface.
- **[PROMOTION.md](PROMOTION.md)** — the gate: criteria for `research →
  spyx.experimental → core`. Every rung up is a human decision.
- **[`/promote-finding`](../.claude/skills/promote-finding.md)** — you invoke this when
  you decide to graduate a finding; it runs the checklist and stages the promotion PR.

The loop: **scout → triage → study (web breadth / local-GPU depth) → review → promote.**

Honest negatives and nulls are first-class: they stay in `research/`, indexed in the
ledger, and are never deleted or reshaped.

## Index of existing work

The material already in this directory maps onto the taxonomy as follows. Nothing
below has been moved — the links point at the original locations.

| Directory | Bucket | Description |
| --- | --- | --- |
| [`paper/`](paper/) | Reproductions | Cross-framework reproductions of the Spyx paper's SHD & N-MNIST benchmarks: the same task implemented in `torch`, `norse`, `spikingjelly`, `mlgenn`, and `jax` (Spyx) so training speed and accuracy can be compared apples-to-apples. |
| [`scaling_experiments/`](scaling_experiments/) | Extensions | `shd_sg_<channels>c-<T>t.ipynb` — a sweep of surrogate-gradient SHD training over input channels (72–700) and sequence length T (128–1024). Extends the reproduction into a scaling study. |
| [`misc/`](misc/) | Extensions | Ablations and comparisons on SHD / N-MNIST: surrogate-function choice (`shd_sg_surrogate_comparison`), optimizer choice (`shd_sg_optimizer_choice`), neuron-model comparison (`shd_sg_neuron_model_comparison`), plus evolutionary-training variants (`*_evo_*`). |
| [`SPSN/`](SPSN/) | Prior art (Reproductions) | **Stochastic Parallelizable Spiking Neurons** — `SHD_jax_SPSN.ipynb`. Directly related prior art for the parallel spiking neurons in Spyx: [`spyx.nn.PSU_LIF`](../src/spyx/nn.py) and [`spyx.phasor.ResonateFire`](../src/spyx/phasor.py). Read this before working in [`new/parallel_spiking_neurons/`](new/parallel_spiking_neurons/README.md). |
| [`pallas/`](pallas/) | Kernels | `aot.ipynb` — ahead-of-time / custom-kernel investigation using JAX Pallas, aimed at faster spiking primitives. |
| [`figures/`](figures/) | Assets | Rendered figures (confusion matrices, surrogate/neuron/optimizer curves, rasters) referenced by the studies above. |

## Spyx modules these studies lean on

- [`spyx.nn.LIF`](../src/spyx/nn.py) — standard hard-reset leaky-integrate-and-fire
  neuron; the accuracy/speed baseline.
- [`spyx.nn.PSU_LIF`](../src/spyx/nn.py) — reset-free LIF whose membrane is a pure
  linear recurrence `V_t = clip(beta)·V_{t-1} + x_t`. Exposes both a sequential
  step and a `.parallel(x)` path via `jax.lax.associative_scan` (O(log T) depth).
  The template for reset-free parallel spiking neurons.
- [`spyx.phasor.ResonateFire`](../src/spyx/phasor.py) — complex-valued
  resonate-and-fire neuron; another parallelizable spiking primitive, evaluated
  against `PSU_LIF` and `LIF` in [`new/parallel_spiking_neurons/`](new/parallel_spiking_neurons/README.md).
- [`spyx.bench`](../src/spyx/bench.py) — benchmarking harness: `benchmark(module,
  input_shape, *, seq_len, batch, ...)` returns a `BenchResult` (median fwd and
  fwd+bwd latency, throughput, peak memory, XLA-cost FLOPs/MFU, spike-rate energy
  proxy); `compare()` and `format_table()` aggregate runs. Use it to fill every
  study's Results table.

## Contributing a study

1. Pick a bucket (`reproductions/`, `extensions/`, `new/`) and create a folder
   with a short, descriptive name.
2. Copy [`_template/README.md`](_template/README.md) into it and fill in every
   section. Do not delete section headers — write `N/A` if one does not apply.
3. Report results honestly: include the failing runs, not only the good ones.
4. **Seeds** — set and record every RNG seed (`jax.random.PRNGKey`, data-shuffle
   seed, NumPy seed). A study that cannot be re-run with the same seed is not
   reproducible.
5. **Hardware** — record the accelerator, driver/runtime, and JAX version. Timing
   numbers are meaningless without them. The reference machine for parallel-neuron
   work is a **Radeon 8060S / gfx1151** on ROCm.
6. **Reporting** — put numbers in the Results table with units; use
   [`spyx.bench`](../src/spyx/bench.py) for latency/throughput/memory so results
   are comparable across studies. Commit rendered figures under
   [`figures/`](figures/) and reference them relatively.
7. Record the **commit hash** the study was run at in the Reproducibility section.
