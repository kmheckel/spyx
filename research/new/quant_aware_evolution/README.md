# Quantization-aware evolution: unbiased ES vs. STE bias at extreme precision

## Title

Gradient-free evolution optimizes an extreme-low-precision spiking net without
straight-through-estimator bias — does it beat STE-QAT where STE hurts most?

## Paper & arXiv/DOI

- **Title:** novel — no paper yet (Spyx research note). Directly relevant prior
  art:
  - **Straight-Through Estimator** (Bengio, Léonard, Courville 2013,
    arXiv:1308.3432) — the biased gradient QAT relies on.
  - **BitNet b1.58** (Ma et al. 2024, arXiv:2402.17764) — the ternary
    ({-1, 0, +1}) weight recipe used for the ternary arm.
  - **CR-FM-NES** (Nomura & Ono, CEC 2022) — the O(d) rank-1 + diagonal adaptive
    ES used as the gradient-free optimizer (same strategy as the Spyx thesis and
    `research/misc/nmnist_evo_crfmnes.ipynb`).
  - **NVFP4 / MXFP4** micro-scaled 4-bit floating point (via `qwix`).
- **Authors / venue / year:** N/A (Spyx research note).
- **Bucket:** new

## Claim under test

Quantization-aware training with STE is **cheap but biased**: the forward pass
uses hard-quantized weights, but the backward pass substitutes the identity for
the quantizer's (zero-almost-everywhere) gradient. As the grid coarsens — int8 →
nvfp4 → ternary — the quantized forward and the surrogate backward diverge, so
the STE-descended direction is a progressively worse proxy for the true
quantized-loss gradient.

Evolution strategies **never differentiate the quantizer**: CR-FM-NES only reads
loss *values*, evaluated on the true quantized forward. It is therefore
**unbiased on the exact objective STE only approximates**, at the cost of high
variance and many forward evaluations.

**Hypothesis.** At precisions where STE bias is worst, ES reaches a lower TRUE
quantized-forward loss than STE-QAT at matched precision, and the ES advantage
**grows as precision drops** (ternary > nvfp4). The headline per precision is the
STE-bias gap = (STE-QAT true loss) − (ES true loss); positive means ES wins.

## Honest expected outcome

This is a boundary-result-prone comparison and we do not assume the hypothesis
holds. ES is unbiased but high-variance and compute-heavy (POP × GENERATIONS
forward evaluations vs. STE's one forward+backward per step); STE is cheap but
biased only at *extreme* precision. Plausible outcomes, all of which we would
report faithfully:

1. **ES advantage grows as precision drops** — the hypothesis: ES ≈ STE (or
   loses) at nvfp4, ES wins at ternary where STE bias is largest.
2. **ES only matches STE at nvfp4 and wins only at ternary.**
3. **ES loses everywhere once compute is fixed** — the STE bias is real but small
   relative to the ES estimator variance at a feasible population/generation
   budget, so the cheap biased gradient still wins per unit compute.

The `SPYX_SMOKE=1` self-check runs a deliberately tiny ES budget (POP=16, 20
generations) purely to prove the multi-arm plumbing runs end-to-end in seconds;
its numbers are **not** evidence for or against the claim (ES is heavily
under-converged there). A full run (larger population, more generations, multiple
seeds) is required before the Findings below can be filled in.

## Method

- **Task:** the same synthetic spiking-classification problem as
  `methods_0_1_hybrid.py` — each class owns a band of input channels that spike at
  an elevated rate; a `Linear → LIF → Linear → LI` classifier reads the summed
  membrane trace via `spyx.fn.integral_*`. Generated in-process, **no dataset
  download**. Dimensions are chosen multiples of 16 so NVFP4's tile-16
  contraction-axis quantization is exact.
- **Shared true-quant forward.** A single function `Q(w) = dequant(quant(w))`
  defines the quantized weight value for every arm:
  - **nvfp4** — `qwix.quantize(w, "nvfp4", tiled_axes={0: 16})` then
    `qwix.dequantize` (the verified qwix path, constructed standalone with an
    explicit tile size so this study does not depend on any in-flight
    `spyx.quant` builder change).
  - **ternary** — BitNet b1.58: per-tensor absmean scale, weights rounded to
    {−1, 0, +1}.
  Only the rank-2 dense **kernels** are quantized; rank-1 neuron state
  (`beta`, thresholds) stays fp32, matching `spyx.quant`'s linear-only default.
- **Three arms**, identical init (same `nnx.Rngs(seed)` → same flat θ₀), identical
  synthetic task, and **the identical `Q`-forward for the reported number**:
  1. **fp32** — surrogate SGD (`optax.adam`), no quantization. Reference ceiling
     (reported on the fp32 forward).
  2. **STE-QAT @ prec** — surrogate SGD where the forward uses
     `w + stop_gradient(Q(w) − w)` (forward = `Q(w)`, gradient = identity — the
     textbook STE). This is the biased path.
  3. **ES @ prec** — CR-FM-NES over the fp32 weights (distribution stays fp32);
     each candidate is quantized by `Q` in the true forward before its fitness
     (mean `integral_crossentropy` over the fitness batches) is measured. **No STE
     anywhere.** JIT-scanned ask/tell loop, mirroring `methods_shd_crfmnes.py`.
- **Precisions:** `nvfp4` and `ternary` (int8 is essentially lossless here and
  would show no gap — the claim is about *extreme* precision). mxfp4 (tile 32) is
  available via the same `qwix` path but omitted so the smoke dims stay small.
- **Measured:** final **true quantized-forward** cross-entropy and accuracy per
  arm (the reported evaluation is identical across arms, isolating the gradient
  path), plus per-arm wall time so the compute asymmetry is explicit. Writes
  `quant_aware_evolution_results.json`.

`SPYX_SMOKE=1` shrinks every dimension (channels, hidden, time, samples, epochs,
population, generations, seeds) so the whole fp32 + 2×(STE, ES) comparison runs
on CPU in seconds; the same code path runs the fuller config without the flag.

## Spyx modules used

- [`spyx.nn.LIF`](../../../src/spyx/nn.py) / `LI` / `Sequential` / `run`
- [`spyx.axn.superspike`](../../../src/spyx/axn.py)
- [`spyx.fn.integral_crossentropy`](../../../src/spyx/fn.py) / `integral_accuracy`
- [`spyx.quant`](../../../src/spyx/quant.py) — conceptual reference for the
  linear-only / weight-only quantization recipe; this study calls `qwix` directly
  (`qwix.quantize` / `qwix.dequantize`) so it is self-contained and does not
  depend on an in-flight `tile_size` addition to the `spyx.quant` builders.
- External: `qwix` (NVFP4 weight quantization), `evosax` (`CR_FM_NES`).

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/quant_aware_evolution/quant_aware_evolution.py   # multi-arm plumbing check, seconds
uv run python research/new/quant_aware_evolution/quant_aware_evolution.py                # full config (larger ES budget)
```

No dataset download — the task is synthetic and generated in-process. The script
writes `quant_aware_evolution_results.json` with per-arm true-quant loss/acc/wall
time and the per-precision STE-bias gap.

## Findings

**Full run on the Radeon 8060S / gfx1151 GPU** (C=32, H=64, 4 classes, T=16, 15
epochs, CR-FM-NES POP=64 × 150 generations, seeds 0/1/2). All arms **converged** —
accuracy 99–100 % everywhere, well above the 25 % chance floor — so this is a fair
comparison, not an under-budgeted one.

| Precision | fp32 ref | STE-QAT true loss | ES true loss | STE-bias gap (STE − ES) | seed spread |
| --- | --- | --- | --- | --- | --- |
| nvfp4 | 0.6719 (99.5 %) | 0.6786 (99.2 %) | 0.6735 (99.7 %) | **+0.0051** (ES lower) | ±0.006 / ±0.006 |
| ternary | 0.6719 (99.5 %) | 0.6445 (99.7 %) | 0.6486 (100 %) | **−0.0041** (STE lower) | ±0.012 / ±0.005 |

**Honest verdict: a null result — no measurable STE-bias gap on this task.** Both
gaps (±0.005) are **smaller than the per-arm seed spread** (±0.005–0.012), so they
are statistically indistinguishable from zero, and they do **not** grow from nvfp4
to ternary — the core hypothesis is *not* supported here. The reason is visible in
the accuracies: the synthetic task is easy enough that **even ternary reaches
100 %**, so the quantized model has capacity to spare and there is simply *no STE
bias for ES to remove*. STE's biased gradient only misleads when the quantization
actually costs accuracy; here it doesn't.

**And ES is not free.** At nvfp4 it spent ~2× the STE-QAT wall time (2.4 s vs 1.1 s)
for a loss difference in the noise — so on an easy, well-conditioned task **STE-QAT
dominates once compute is normalized.** This mirrors the honest-negative pattern of
the sibling hybrid study: the gradient-free method's edge needs a regime with
genuinely large bias, and an easy task doesn't provide one.

**Where the real test lives (next step).** To actually stress the STE bias, the
quantized model must be **capacity-constrained on a hard task** so that nvfp4 /
ternary *degrades* accuracy — e.g. real SHD (20 classes, 128 channels, long T) with
a tight hidden width, where the fp32→ternary drop is large. Only there can ES's
unbiased optimization of the true quantized objective plausibly pull ahead. On an
easy task where every precision saturates, it cannot, and this run says so plainly.

**Smoke self-check (plumbing only, NOT a result).** `SPYX_SMOKE=1` completes the
sweep in seconds with a deliberately tiny ES budget (POP=16, 20 generations) that
does not converge; it confirms the comparison *runs* and says nothing about the
claim.

## Reproducibility

- **Seeds:** `nnx.Rngs(seed)` for model init (shared flat θ₀ across arms per
  seed); `jax.random.PRNGKey(seed)` / `PRNGKey(seed+1)` for CR-FM-NES
  init/evolution; NumPy `default_rng(seed)` for the synthetic dataset. `SEEDS =
  [0, 1]` (smoke) / `[0, 1, 2]` (full).
- **Quantization:** `qwix.quantize(w, "nvfp4", tiled_axes={0: 16})` +
  `qwix.dequantize`; ternary is BitNet b1.58 absmean, weights in {−1, 0, +1}.
  Only rank-2 kernels are quantized.
- **JAX / hardware:** JAX 0.10.2, Flax 0.12.7, CPU. Timing is reported per arm
  (the compute asymmetry is part of the claim), but correctness of the matched
  true-quant comparison is the point.
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-06 (script + smoke self-check; Findings pending full run).
