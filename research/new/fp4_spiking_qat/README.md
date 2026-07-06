# FP4 block-microscaling for SNNs: weight-QAT + membrane-state quantization

## Title

Sub-4-bit spiking: NVFP4 / MXFP4 weight-QAT and membrane-state quantization on a
spiking classifier — the first FP4-microscaling × SNN datapoints, reported
honestly against a bit-matched quantized-ANN baseline.

## Paper & arXiv/DOI

- **Title:** novel — no paper yet (Spyx research note; PROGRAM.md flagship **F3**,
  the one CONFIRMED unclaimed gap). Directly relevant prior art:
  - **SQUAT** (Chowdhury et al. 2024, arXiv:2404.19668) — the membrane-state
    quantization idea: quantize the neuron membrane potential `V` each step, not
    just the weights. The state-quant arms here are FP4/int8 instances of it.
  - **Energy-matched ANN/SNN accounting** (arXiv:2409.08290) — a T-step binary
    spike accumulator carries `T+1` levels, so the honest bit-matched ANN baseline
    uses `ceil(log2(T+1))` bits. Used verbatim for the `QANN` arm.
  - **NVFP4 / MXFP4** micro-scaled 4-bit floating point — shared block exponent per
    tile (16 for NVFP4, 32 for MXFP4), via `qwix`.
  - **BitNet b1.58** (Ma et al. 2024, arXiv:2402.17764) — the ternary
    ({−1, 0, +1}) weight recipe used for the ternary arm.
  - **AQ4SViT** (arXiv:2606.15523) — the *nearest boundary* the survey found; it is
    **integer** search quantization, **not** block-scaled FP4, which is exactly why
    the FP4 × SNN gap is unclaimed.
- **Authors / venue / year:** N/A (Spyx research note).
- **Bucket:** new (novelty).

## Claim under test

**FP4 microscaling is a viable weight format for SNNs** — test accuracy
competitive with int8 and better than ternary at matched-ish bits — **and
membrane-state quantization is feasible** (quantizing the LIF membrane potential
`V` each step at NVFP4 / int8 does not collapse the network). These would be the
first FP4-microscaling × SNN datapoints. Because SNN efficiency claims are only
credible against a bit-matched dense baseline (arXiv:2409.08290), every SNN arm is
reported next to a `ceil(log2(T+1))`-bit rate-coded ANN of the **same
architecture**, so the numbers are not graded on a curve.

## Method

- **Task:** synthetic spiking classification (no dataset download). Each class
  owns a contiguous band of input channels that spike at an elevated rate; a
  `Linear → LIF → Linear → LI` classifier reads the summed membrane trace via
  `spyx.fn.integral_*`. Dimensions are multiples of **32** so both NVFP4 (tile 16)
  and MXFP4 (tile 32) tile the contraction axis exactly.
- **Shared true-quant forward.** A single `Q(x) = dequant(quant(x))` defines the
  quantized value for every arm, applied to rank-2 dense **kernels** (weights,
  contraction axis 0) and — for the SQUAT arms — to the LIF membrane `V` (feature
  axis) each step. Neuron `beta` / thresholds (rank-1) stay fp32, matching
  `spyx.quant`'s linear-only default.
  - **int8 / int-b** — symmetric absmax integer grid.
  - **nvfp4** — `qwix.quantize(x, "nvfp4", tiled_axes={axis: 16})` + `dequantize`.
  - **mxfp4** — `qwix.quantize(x, "mxfp4", tiled_axes={axis: 32})` + `dequantize`.
  - **ternary** — BitNet b1.58: per-tensor absmean scale, weights in {−1, 0, +1}.
- **STE-QAT.** Training uses `w + stop_gradient(Q(w) − w)` (forward = `Q(w)`,
  gradient = identity — the textbook straight-through estimator), for both weights
  and, in the SQUAT arms, membrane state. **Evaluation uses the true `Q`-forward
  with no STE**, identical across arms, so the reported number isolates the format.
- **Arms** (matched `nnx.Rngs(seed)` init, matched synthetic task):
  1. **fp32** — no quantization (reference ceiling).
  2. **int8 / nvfp4 / mxfp4 / ternary** — weight-only QAT.
  3. **nvfp4 +Vstate / int8 +Vstate** — weights **and** LIF membrane `V`
     quantized at that precision each step (the SQUAT gap).
  4. **QANN(int-b)** — rate-coded (`sum` over time) matched-arch ANN,
     `Linear → ReLU → Linear`, weights quantized to `b = ceil(log2(T+1))` bits
     (= **int5** at T=16). The honest energy-matched baseline.
- **Reported per arm:** test accuracy (mean ± std over seeds), weight
  bit-footprint (effective bits/element **including block-scale overhead** —
  NVFP4 4.5 b, MXFP4 4.25 b, ternary log₂3 ≈ 1.58 b — and KiB for the whole
  weight set), and, for SQUAT arms, the membrane-state bit-width. Writes
  `fp4_spiking_qat_results.json`.

`SPYX_SMOKE=1` shrinks every dimension so the full 8-arm sweep runs on CPU in
seconds; the same code path runs the larger synthetic config without the flag.

## Honest expected outcome

We do **not** assume the claim holds. The synthetic task is deliberately
easy/linearly-separable so the *plumbing* is cheap to exercise; on it every
precision — including ternary — is likely to saturate near ceiling, which means
the smoke (and even the larger synthetic) run is a **feasibility/plumbing check,
not evidence of a format advantage**. Plausible outcomes we would report
faithfully:

1. **FP4 ≈ int8 ≫ ternary** — the hypothesis: microscaling's shared block
   exponent buys back most of the int8 accuracy at ~4.3–4.5 effective bits, while
   flat ternary loses more.
2. **All SNN arms saturate; QANN(int-b) matches or beats them** — the honest
   negative the bit-matched baseline exists to catch: if a `log2(T+1)`-bit ANN
   ties the FP4 SNN, the spiking format bought nothing here.
3. **Membrane-state quant degrades or collapses** — the SQUAT arm is the riskiest;
   quantizing the recurrent state can round away the `V = βV + x` cancellations
   (this is exactly why `spyx.quant` leaves neuron state fp32 by default).

Only a **capacity-constrained, hard task** (real SSC — 20 classes, 700 channels,
long T, tight hidden width) can separate the formats; that run is **human-gated**
(needs a dataset download + GPU) and is intentionally not wired here.

## Spyx modules used

- [`spyx.nn.LIF`](../../../src/spyx/nn.py) (dynamics mirrored by the local
  `QStateLIF`) / `LI` / `Sequential` / `run`
- [`spyx.axn.superspike`](../../../src/spyx/axn.py)
- [`spyx.fn.integral_crossentropy`](../../../src/spyx/fn.py) / `integral_accuracy`
- [`spyx.quant`](../../../src/spyx/quant.py) — conceptual reference for the
  weight-only / microscaled recipe (its `weights_only_rules` now auto-fills the
  NVFP4/MXFP4 tile size); this study calls `qwix` directly so it is self-contained.
- External: `qwix` (NVFP4 / MXFP4 weight & state quantization).

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/fp4_spiking_qat/fp4_spiking_qat.py   # 8-arm plumbing check, seconds
uv run python research/new/fp4_spiking_qat/fp4_spiking_qat.py                # larger synthetic config
```

No dataset download — the task is synthetic and generated in-process. The script
writes `fp4_spiking_qat_results.json` with per-arm accuracy, weight KiB, effective
weight bits, and membrane-state bits. **The real-SSC run is human-gated and not
wired here.**

## Findings

**Full run on the Radeon 8060S / gfx1151 GPU** (C=64, H=128, 8 classes, T=16, 25
epochs, seeds 0/1/2). **These are, to our knowledge, the first FP4-microscaling ×
SNN datapoints** — the gap the field survey confirmed was unclaimed.

| Arm | test acc | weight bits (eff) | weight KiB | Vstate bits |
| --- | --- | --- | --- | --- |
| fp32 | 99.7 % ± 0.2 | 32.00 | 36.00 | — |
| int8 | 100.0 % ± 0.0 | 8.00 | 9.00 | — |
| **nvfp4** | 99.7 % ± 0.2 | 4.50 | 5.06 | — |
| **mxfp4** | 99.7 % ± 0.2 | 4.25 | 4.78 | — |
| ternary | 100.0 % ± 0.0 | 1.58 | 1.78 | — |
| **nvfp4 +Vstate** | 99.9 % ± 0.1 | 4.50 | 5.06 | 4.50 |
| int8 +Vstate | 99.8 % ± 0.2 | 8.00 | 9.00 | 8.00 |
| QANN(int5) | 100.0 % ± 0.0 | 5.00 | 5.62 | — |

**1. Feasibility confirmed (the point of the study).** NVFP4 and MXFP4 weight-QAT run
end-to-end on a spiking classifier and are **lossless** here (99.7 % vs fp32 99.7 %) at
~4.25–4.5 effective bits/weight. **Membrane-state quantization** (the SQUAT gap) also
works — quantizing the neuron potential V to nvfp4 costs nothing (99.9 %). So FP4
microscaling *is* a viable weight- and state-format for SNNs — the first such datapoints.

**2. But this task does not *discriminate* the formats — honest caveat.** Every arm,
including ternary (1.58 b) and the int5 quantized-ANN baseline, saturates at 99.7–100 %.
No format is separated beyond seed noise, and the bit-matched QANN ties the FP4 SNN arms
— the degenerate regime the honest baseline exists to flag. So this establishes
**feasibility, not superiority**: it does not yet show FP4 *beats* ternary/int at matched
bits, because the task is too easy to make precision cost accuracy (the same lesson as
`quant_aware_evolution_hard`).

**3. The discriminating experiment (human-gated).** To rank the formats you need a task
where fp32 is near its capacity limit and precision genuinely costs accuracy — real SSC
(the survey's recommended non-saturated benchmark) with a tight hidden width, so the
fp32→low-precision drop is large. There, matched-footprint NVFP4 vs ternary vs int should
separate. That run needs the dataset + a longer budget and is flagged for a human.

**Verdict:** ✅ feasibility (first FP4×SNN result) · ➖ no format ranking on this easy task.

**Smoke** (`SPYX_SMOKE=1`, C=32/H=32, CPU, ~4 s) runs the same 8-arm sweep and shows
the same qualitative picture (all arms 96–100 %) — plumbing confirmation only.

## Reproducibility

- **Seeds:** `nnx.Rngs(seed)` for SNN init (shared flat θ₀ across weight/state
  arms per seed); NumPy `default_rng(seed)` for the synthetic dataset and
  `default_rng(1000+seed)` for ANN init. `SEEDS = [0, 1]` (smoke) / `[0, 1, 2]`
  (full).
- **Quantization:** `qwix.quantize(x, "nvfp4", tiled_axes={axis: 16})` /
  `"mxfp4"` tile 32 + `qwix.dequantize`; int8/int-b symmetric absmax; ternary
  BitNet b1.58 absmean. Weights tile the contraction axis (0); membrane state
  tiles the feature axis (−1). STE forward at train, true `Q`-forward at eval.
- **Bit accounting:** effective bits include the shared block scale — NVFP4 = 4 +
  8/16 = 4.50, MXFP4 = 4 + 8/32 = 4.25, ternary = log₂3 ≈ 1.58, int-b = b. ANN
  precision `b = ceil(log2(T+1))` = 5 at T=16.
- **JAX / hardware:** JAX 0.10.2, Flax 0.12.7, CPU for the smoke check.
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-06 (smoke self-check only; Findings pending full run).
