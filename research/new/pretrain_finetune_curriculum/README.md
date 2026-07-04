# Pretrain-then-finetune curriculum: cheap PSU_LIF warm-start for a hard-reset LIF

> **STATUS: METHOD + SMOKE-VALIDATED (2026-07-03).** The pipeline runs end-to-end
> on CPU in `SMOKE=1` mode (synthetic spikes); the full SHD / GPU numbers are
> **TODO** and left for the maintainer's run. This is a controlled method study,
> not a leaderboard number.

## Title

Pretrain-then-finetune curriculum: use the reset-free, parallel-scannable
`spyx.nn.PSU_LIF` as a cheap pretrainer, then transfer into and finetune a
hard-reset `spyx.nn.LIF`.

## Paper & arXiv/DOI

- **Title:** novel — no paper yet.
- **Bucket:** new
- **Related prior art:**
  - Sibling study [`../parallel_spiking_neurons/`](../parallel_spiking_neurons/):
    the fair neuron-to-neuron LIF vs PSU_LIF vs ResonateFire comparison that
    motivates this curriculum. It found PSU_LIF to be the *fastest* neuron (8x
    fwd/bwd via its `O(log T)` parallel scan) but **not stably trainable alone**
    (accuracy peaks ~11% then collapses on SHD).
  - **Q-S5** (Abreu, Pedersen, Heckel, Pierro; arXiv:2406.09477, 2024): fully
    quantized S5 drops <1% on sMNIST/most-LRA, but the *recurrent* weights need
    >=8-bit precision while other components compress much further, and QAT beats
    PTQ. The shared lesson — spend your expensive, high-fidelity training budget
    on the recurrent dynamics, and cheapen everything else — is exactly the
    trade this curriculum makes across *time* rather than across *precision*.

## Claim under test

A hard-reset `spyx.nn.LIF` and the reset-free `spyx.nn.PSU_LIF` are the **same
neuron minus the reset term** and therefore share an identical trainable-parameter
structure (the `nnx.Linear` kernels and each neuron's `beta`). Because PSU_LIF's
membrane is a first-order *linear* recurrence, it can be scored with an
`O(log T)` associative scan (`PSU_LIF.parallel`) instead of the `O(T)` sequential
scan a hard-reset neuron requires — it drives forward much faster. We claim that
**pretraining `K` epochs cheaply reset-free, transferring the shared weights into
an identical-shape hard-reset LIF, then finetuning the remaining `N-K` epochs
sequentially** can match (or beat) LIF-from-scratch accuracy at **lower total
training wall-clock**, sidestepping PSU_LIF's standalone instability by only using
it as a warm-start.

## Method

- **Dataset:** Spiking Heidelberg Digits (SHD), same bit-packed `.npz` cache
  mechanism (`SHD_CACHE`) as the sibling study.
- **Architecture (identical across all runs):** `Linear(128->64) -> neuron ->
  Linear(64->64) -> neuron -> Linear(64->20) -> LI` readout, triangular
  surrogate, `optax.lion(3e-4)`, seed 0, T=128, 128 input channels. The only
  thing that changes is the hidden `neuron` (`PSU_LIF` during pretrain, `LIF`
  during finetune) — and the neurons share their parameters exactly.
- **`baseline()`:** `LIF` from scratch, `N` epochs, sequential `spyx.nn.run`
  BPTT. Reports `(test_acc, total_wallclock)`.
- **`curriculum(K)`:**
  1. Build the `PSU_LIF` net; pretrain `K` epochs driving the forward through
     the **parallel associative-scan path** (`SHDSNN.parallel`, which uses
     `PSU_LIF.parallel` for the hidden neurons and a matching leaky parallel scan
     for the `LI` readout — verified numerically identical to the sequential
     path to float32 precision, ~3e-8).
  2. **Transfer** every trainable parameter into a fresh identical-shape `LIF`
     net via a filtered `nnx.update(dst, nnx.state(src, nnx.Param))`, then
     **assert** every `Linear` kernel and neuron/LI `beta` matches post-transfer.
  3. Finetune the remaining `N-K` epochs with the sequential `spyx.nn.run`.
  Reports `(test_acc, total_wallclock = pretrain + finetune)`.
- **`main()`:** runs `baseline` plus `curriculum(K)` over a small sweep
  (`K in {5, 10}` for SHD; `{1, 2}` in smoke), prints an accuracy-vs-total-time
  table, and writes `results.json`.
- **Fairness of timing:** every timed loop excludes its one-off warm-compile
  step, identical to the sibling study's convention, so the reported wall-clock
  is steady-state training time. The curriculum's total explicitly *includes*
  the pretrain cost — a warm-start is only a win if the cheap epochs plus the
  fewer expensive epochs beat the all-expensive baseline.

## Spyx modules used

- [`spyx.nn.PSU_LIF`](../../../src/spyx/nn.py) — reset-free neuron; sequential
  `__call__` and `O(log T)` `.parallel`.
- [`spyx.nn.LIF`](../../../src/spyx/nn.py) — hard-reset target neuron.
- [`spyx.nn.LI`](../../../src/spyx/nn.py) / [`spyx.nn.Sequential`](../../../src/spyx/nn.py)
  / [`spyx.nn.run`](../../../src/spyx/nn.py) — readout, container, sequential scan.
- [`spyx.axn.triangular`](../../../src/spyx/axn.py) — surrogate gradient.
- [`spyx.fn.integral_crossentropy`](../../../src/spyx/fn.py) /
  [`spyx.fn.integral_accuracy`](../../../src/spyx/fn.py) — loss / metric.
- `flax.nnx` state tooling (`nnx.state` / `nnx.update` with a `nnx.Param` filter)
  for the robust, asserted parameter transfer.

## How to run

```bash
# Fast self-check (no dataset; synthetic {0,1} spikes, tiny epochs/hidden):
SMOKE=1 python research/new/pretrain_finetune_curriculum/run.py

# Full SHD run (prebuilt bit-packed cache via SHD_CACHE, else prestaged once):
JAX_PLATFORMS=rocm SHD_CACHE=/path/to/shd_cache.npz EPOCHS=40 \
  PYTHONPATH=src python research/new/pretrain_finetune_curriculum/run.py
```

Writes `results.json` (per-method rows: `K`, pretrain/finetune/total seconds,
`test_acc`, and the intermediate PSU_LIF pretrain accuracy).

## Results

**Smoke self-check (CPU, synthetic random spikes)** — validates the pipeline,
the parallel pretrain path, the asserted transfer, and the table; the accuracies
are meaningless (random data → chance ties):

```
method             K  epochs   pre(s)  fine(s)  total(s)   acc(%)  speedup
--------------------------------------------------------------------------
baseline           -       4     0.00     0.28      0.28    31.25    1.00x
curriculum         1       4     0.35     0.00      0.36    31.25    0.79x
curriculum         2       4     0.55     0.00      0.55    31.25    0.51x
```

**Full SHD (40 epochs, hidden=64, GPU):** _TODO — maintainer's run._ Fill from
`results.json`:

| Method | K | Pretrain (s) | Finetune (s) | Total (s) | Test acc | Speedup vs baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` (LIF from scratch) | – | – | TODO | TODO | TODO | 1.00x |
| `curriculum` | 5 | TODO | TODO | TODO | TODO | TODO |
| `curriculum` | 10 | TODO | TODO | TODO | TODO | TODO |

**What to look for.** The curriculum wins if a `curriculum(K)` row reaches
`baseline` accuracy (or within noise) at `total_s < baseline.total_s`. Because
each pretrain epoch is ~8x cheaper than a sequential epoch (sibling-study
neuron-primitive number), replacing `K` expensive epochs with `K` cheap ones
should cut total time by roughly `K/N * (1 - 1/8)` if the warm-start does not
hurt final accuracy. The open risk is the opposite of a free lunch: PSU_LIF's
reset-free dynamics may drift the weights somewhere the hard-reset finetune has
to spend epochs undoing — the sweep over `K` is there to find where the
warm-start stops helping and starts hurting.

## Findings

- _TODO after the SHD/GPU run._ Expected narrative to confirm or refute: a small
  `K` warm-start is a net wall-clock win at equal accuracy; a large `K`
  eventually degrades final accuracy as the reset-free pretraining pulls the
  shared weights toward a regime the hard-reset finetune cannot fully recover
  (mirroring the sibling study's PSU_LIF-alone collapse — the failure mode this
  curriculum is designed to *bound* rather than avoid entirely).

## Reproducibility

- **Seeds:** `nnx.Rngs(0)` (override with `SEED`); a fresh optimizer is built
  after transfer (pretrain momentum is intentionally *not* carried over).
- **Config:** `EPOCHS` (default 40), `LR` (default 3e-4), `SHD_CACHE`,
  `SMOKE=1`/`--smoke`. Smoke mode: `SAMPLE_T=32, CHANNELS=16, N_CLASSES=4,
  HIDDEN=8, EPOCHS=4, K in {1,2}`.
- **Determinism note:** parameter transfer is verified in-run by asserting Linear
  kernels and betas match after `nnx.update`; the parallel pretrain path is
  numerically identical to the sequential scan (float32, ~3e-8).
- **JAX / hardware:** smoke validated on CPU (`jax` CPU backend). Full run:
  Radeon 8060S / gfx1151, ROCm, per the sibling study's environment.
- **Date:** 2026-07-03.
