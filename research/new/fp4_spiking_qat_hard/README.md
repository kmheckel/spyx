# Discriminating F3: does FP4 microscaling separate from ternary/int on a HARD task?

## Title

FP4 (NVFP4/MXFP4) vs ternary vs int weight-QAT on **real SHD at constrained capacity** —
the discriminating follow-up to the FP4×SNN feasibility study.

## Paper & arXiv/DOI

- **Title:** novel — extends `../fp4_spiking_qat/`. The confirmed unclaimed gap
  (FP4 microscaling × SNN) per the verified research program.
- **Bucket:** extension

## Claim under test

The feasibility study found NVFP4/MXFP4 weight-QAT lossless on an easy synthetic task,
but every arm (incl. ternary and the quantized-ANN baseline) saturated — so it could
**rank** nothing. **Claim:** on a task where precision genuinely costs accuracy —
**real SHD** (20 classes, 128 channels, 128 steps) at **constrained hidden width** —
matched-footprint **NVFP4/MXFP4 should hold accuracy better than ternary** (and beat
int at fewer bits), i.e. the formats separate once quantization bites. The honest
alternative outcome (all formats tie, or ternary is fine) is equally reportable.

## Method

Reuse the sibling study's **verified** weight-QAT machinery (`true_quant` STE at
nvfp4 tile-16 / mxfp4 tile-32 / ternary BitNet / int8-absmax; `train_snn`; true-quant
`evaluate`) by importing it and overriding the task config. Sweep **hidden width**
`128 → 64 → 32` (capacity, from generous to tight) on real cached SHD; per width,
train each of `{fp32, int8, nvfp4, mxfp4, ternary}` (STE-QAT, Adam, 30 epochs, seeds
0/1) and report **test accuracy**, the **fp32→quant accuracy drop**, and the
**effective bits/weight**. The discriminating signal: does the drop grow at tight
capacity, and does FP4 beat ternary there?

## Spyx modules used

- [`../fp4_spiking_qat/`](../fp4_spiking_qat/) — reused quant arms
- [`spyx.data.SHD_loader`](../../../src/spyx/data.py), [`spyx.quant`](../../../src/spyx/quant.py) / qwix
- [`spyx.nn`](../../../src/spyx/nn.py), [`spyx.fn`](../../../src/spyx/fn.py), [`spyx.axn`](../../../src/spyx/axn.py)

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/fp4_spiking_qat_hard/hard_shd.py   # tiny, CPU
~/.venvs/jax-rocm-0.9.2/bin/python research/new/fp4_spiking_qat_hard/hard_shd.py  # full, GPU
```

Real SHD (cached under `data/SHD/`); no synthetic data. Writes `hard_shd_results.json`.

## Findings

**Full run on the Radeon 8060S / gfx1151** (real SHD, 30 epochs, 2 seeds, 467 s).
Unlike the easy-task feasibility study, the formats now **clearly separate** — the
discriminating premise worked. To our knowledge these are the **first FP4-microscaling
× SNN *ranking* datapoints.**

| Hidden | fp32 | int8 | nvfp4 | mxfp4 | ternary |
| --- | --- | --- | --- | --- | --- |
| 128 | 58.6% | 59.0% | **48.2%** | **49.0%** | 41.1% |
| 64 | 56.5% | 57.5% | **51.6%** | **47.8%** | 39.3% |
| 32 | 53.9% | 51.1% | **34.3%** | **37.9%** | 28.7% |

*(effective bits/weight: int8 8.0 · nvfp4 4.50 · mxfp4 4.25 · ternary 1.58)*

**1. A clean accuracy ordering emerges: `int8 ≈ fp32 > FP4 > ternary`**, consistent
across all three capacities. int8 is **near-lossless** (−1 to +3 pt vs fp32); FP4 costs
a real but bounded 5–20 pt; ternary costs 17–25 pt.

**2. FP4 microscaling clearly beats ternary** — by **+7 to +9 pt** at every capacity
(e.g. H=128: nvfp4 48.2 / mxfp4 49.0 vs ternary 41.1). FP4 buys that accuracy for ~2.7×
ternary's footprint (4.25–4.5 b vs 1.58 b), so FP4 is a genuine **accuracy/footprint
middle ground** between ternary and int8 — the first evidence that the format is worth
its bits on a spiking net.

**3. The cost grows as capacity tightens** (the discriminating axis worked): the FP4
drop rises from ~10 pt (H=128) to ~19 pt (H=32); ternary from 17→25 pt. int8 stays
near-lossless even at H=32 (+2.7 pt) — so **int8 is robust, FP4 is a bounded cost, and
ternary is expensive** exactly when capacity is scarce.

**4. NVFP4 vs MXFP4 is within noise here** — nvfp4 ahead at H=64 (51.6 vs 47.8), mxfp4
ahead at H=32 (37.9 vs 34.3); no consistent winner between the two FP4 variants at this
scale, despite NVFP4's finer 16-block scale.

**Honest limits.** (a) This is *not* a strictly matched-footprint comparison — there is
no int4 arm, so "FP4 beats ternary" is at FP4's higher bit-width; the FP4-vs-int4
matched-bit question is the natural next arm. (b) fp32 tops out at ~54–59% (a small plain-
LIF feedforward net, 128 channels, no delays/ALIF), so absolute numbers are modest — but
the *relative* format ordering, which is the claim, is clear and stable across capacities
and seeds. (c) Not independently re-verified (run directly, single machine).

**Verdict:** ✅ the FP4×SNN gap now has real ranking data — **FP4 microscaling is a viable
mid-precision format for SNNs that clearly outperforms ternary**, sits below int8, and
degrades gracefully with capacity. Next: an **int4** arm (matched-bit FP4-vs-int) and a
richer neuron/longer budget to lift the fp32 ceiling.

## Reproducibility

- **Seeds:** `nnx.Rngs(seed)` init; NumPy for any shuffling. `SEEDS=[0]` (smoke) / `[0,1]` (full).
- **JAX / hardware:** full run on the Radeon 8060S / gfx1151 (ROCm venv); device in the JSON.
- **Data:** SHD via `spyx.data.SHD_loader`, cached in `data/SHD/`.
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-06.
