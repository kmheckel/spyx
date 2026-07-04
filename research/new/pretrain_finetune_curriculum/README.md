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

**Full SHD (40 epochs, hidden=64, Radeon 8060S / gfx1151), commit `78ef9db`:**

| Method | K | Pretrain (s) | Finetune (s) | Total (s) | Test acc | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` (LIF from scratch) | – | 0.00 | 9.36 | 9.36 | **34.1%** | 1.00× |
| `curriculum` | 5 | 2.13 | 7.29 | 9.42 | 29.7% | 0.99× |
| `curriculum` | 10 | 2.23 | 6.24 | 8.46 | 28.4% | 1.11× |

**Result: the curriculum did not win at this configuration** — it was slightly
*slower*-to-no-faster and *lost* 4–6 points of accuracy.

## Findings

This is an honest negative result, and it pins down *why* — a boundary condition
worth recording:

1. **No speed lever in a Linear-dominated net.** The parallel-scan advantage is on
   the *neuron primitive* (~8× fwd/bwd in the sibling study). But a real SNN layer
   is `Linear → neuron → Linear`, and the `Linear` matmuls dominate the FLOPs; the
   neuron recurrence is a small fraction. So parallelising only the neuron barely
   moves end-to-end time — pretrain ran at ~0.22 s/epoch vs baseline's ~0.23 s/epoch
   (essentially equal). The 8× neuron speedup is *diluted* to ~1×.
2. **Saturated regime.** At batch 256 / hidden 64 / T 128 the GPU is throughput-
   bound (the sibling study's "saturated" end), exactly where the parallel scan's
   O(log T) depth advantage is smallest. The curriculum's premise (cheap parallel
   pretrain) only has teeth in the *device-slack* regime — small batch, long T,
   neuron-dominated compute — which standard SHD-at-batch-256 is not.
3. **Weak warm-start hurts slightly.** PSU_LIF only reached 7–11% before transfer
   (barely above chance), so the transferred init was a *worse* starting point than
   fresh random for the hard-reset finetune, costing a few points.

**Takeaway.** Parallelising the neuron is not, by itself, an end-to-end training
speedup for standard SNN architectures — the Linear layers must also be made
time-parallel (apply them batched over T, run only the neuron via `.parallel`),
and the pretrain must be long enough to give a genuinely useful init. Both are
concrete follow-ups; until then, the curriculum's premise doesn't hold here. The
speedup and the warm-start are *separable* problems, and this config delivered
neither.

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
