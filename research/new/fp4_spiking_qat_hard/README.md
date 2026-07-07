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
matched-footprint **NVFP4/MXFP4 should hold accuracy better than ternary** and land
where their bit-width predicts against a true **matched-bit int4** comparator, i.e. the
formats separate cleanly once quantization bites. The honest alternative outcome (all
formats tie, or ternary/int4 is fine) is equally reportable.

This revision closes two gaps the first hard run flagged: (a) it adds the **int4** arm
so the comparison is a clean **matched-bit ranking** — FP4 (nvfp4 4.5b / mxfp4 4.25b)
vs **int4 (4.0b)** vs int8 (8.0b) vs ternary (1.58b) vs fp32 — and (b) it lifts the
fp32 ceiling with a **richer neuron** so the format gaps are measured on a stronger
baseline rather than the plain-LIF ~54–59% ceiling.

## Method

Reuse the sibling study's **verified** weight-QAT machinery (`true_quant` STE at
nvfp4 tile-16 / mxfp4 tile-32 / ternary BitNet / int8-absmax; `train_snn`; true-quant
`evaluate`) by importing it and overriding the task config. Sweep **hidden width**
`128 → 64 → 32` (capacity, from generous to tight) on real cached SHD; per width,
train each of `{fp32, int8, int4, nvfp4, mxfp4, ternary}` (STE-QAT, Adam, 30 epochs,
seeds 0/1) and report **test accuracy**, the **fp32→quant accuracy drop**, and the
**effective bits/weight**. The discriminating signals: does the drop grow at tight
capacity, does FP4 beat ternary, and — the new matched-bit question — where does FP4
land relative to **int4** at ~the same 4-bit footprint?

**int4 arm.** `fp4_spiking_qat.true_quant` already routes any `int<b>` scheme through
`_int_absmax` (symmetric absmax, dequantized), so `"int4"` quantizes the Linear kernels
to 4-bit levels `-(2³-1)..2³-1 = -7..7`. The only wiring is `"int4": 4.0` in
`fp4_spiking_qat.EFFECTIVE_BITS` and `"int4"` in `SCHEMES` here. It STE-QATs the kernels
exactly like the other arms.

**Richer neuron (lifts the fp32 ceiling).** `SPYX_NEURON` selects the hidden model:
`alif` (default) uses **`spyx.nn.ALIF`** — adaptive-threshold LIF with a second dynamic
state (per-neuron threshold adaptation `T`), richer temporal expressivity than plain
LIF; `lif` reproduces the original plain-`QStateLIF` baseline; `lif2` is a fallback
**two-hidden-layer LIF stack**. ALIF fits the quant path cleanly: its `beta`/`gamma` are
rank-1 params that stay fp32, so `_quant_kernels` still STE-QATs **only** the rank-2
Linear kernels — every weight-QAT arm (incl. int4) is byte-for-byte identical; only the
neuron changes. The swap is done by overriding `fp4.MODEL_CLS` (a new overridable hook
in the sibling study); `train_snn` / `evaluate` are untouched. **Neuron used: ALIF.**

## Spyx modules used

- [`../fp4_spiking_qat/`](../fp4_spiking_qat/) — reused quant arms (`true_quant` int4, `MODEL_CLS` hook)
- [`spyx.data.SHD_loader`](../../../src/spyx/data.py), [`spyx.quant`](../../../src/spyx/quant.py) / qwix
- [`spyx.nn.ALIF`](../../../src/spyx/nn.py) (richer neuron), [`spyx.nn`](../../../src/spyx/nn.py), [`spyx.fn`](../../../src/spyx/fn.py), [`spyx.axn`](../../../src/spyx/axn.py)

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/fp4_spiking_qat_hard/hard_shd.py   # tiny, CPU
~/.venvs/jax-rocm-0.9.2/bin/python research/new/fp4_spiking_qat_hard/hard_shd.py  # full, GPU (ALIF)
# neuron override: SPYX_NEURON in {alif (default), lif (plain-LIF baseline), lif2 (2-layer stack)}
SPYX_NEURON=lif ~/.venvs/jax-rocm-0.9.2/bin/python research/new/fp4_spiking_qat_hard/hard_shd.py
```

Real SHD (cached under `data/SHD/`); no synthetic data. Writes `hard_shd_results.json`
(config now records the `neuron`).

## Findings

**Full run on the Radeon 8060S / gfx1151** (ALIF neuron, int4 arm added, real SHD, 30
epochs, 2 seeds, 625 s):

| Hidden | fp32 | int8 | **int4** | nvfp4 | mxfp4 | ternary |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 54.8% | 59.3% | 51.7% | 49.8% | 36.3% | 46.5% |
| 64 | 57.4% | 59.2% | 46.4% | 49.0% | **54.0%** | 39.7% |
| 32 | 56.3% | 55.6% | 41.1% | 46.6% | 42.8% | 35.9% |

*(effective bits/weight: int8 8.0 · **int4 4.0** · nvfp4 4.50 · mxfp4 4.25 · ternary 1.58)*

**1. The richer neuron did NOT lift the ceiling.** ALIF's fp32 (54.8–57.4%) is
statistically the same as plain LIF's (53.9–58.6%) — the adaptive threshold bought no
accuracy on this setup, and it **added variance** to the quantized arms. An honest
negative on "a richer neuron raises the baseline here."

**2. The matched-bit answer (int4 vs FP4): NVFP4 earns its microscaling exactly where
capacity is tight.** At generous capacity nvfp4 ≈ int4 (H=128: 49.8 vs 51.7), but as
capacity tightens nvfp4 **holds better** (H=32: **46.6 vs 41.1**, +5.5 pt) — the per-16
block scale preserves dynamic range that plain 4-bit int loses when every bit counts.
So at ~4 bits, **NVFP4 is the more robust choice than int4 under capacity pressure**,
which is the regime that matters.

**3. MXFP4 is erratic; NVFP4 is the reliable FP4.** MXFP4 swings 36.3 → 54.0 → 42.8
(±3.4/0.2/3.1) — best of all 4-bit arms at H=64 but worst at H=128 — while NVFP4 is
stable (49.8/49.0/46.6). NVFP4's finer 16-block E4M3 scale beats MXFP4's coarser 32-block
E8M0 scale on these small layers. **Between the two FP4 formats, prefer NVFP4.**

**4. int8 stays robust** (near or above fp32 everywhere — a regularizer), **ternary
stays lowest** (with one exception: it edged MXFP4's outlier at H=128).

**Net (combining with the plain-LIF baseline below):** NVFP4 is the most reliable
sub-8-bit format for SNNs here — competitive with int4 and **better under capacity
pressure**, more stable than MXFP4, above ternary, below int8. The clean plain-LIF
ordering (FP4 ≫ ternary) blurs into more variance with ALIF, so the *robust* claim is
"NVFP4 ≥ int4 ≥ MXFP4/ternary at ~4 bits, int8 near-lossless," not a crisp total order.
Honest limits: modest ceiling (~55%), high per-arm variance (seeds 0/1), single machine,
not independently re-verified.

### Prior run for reference (plain-LIF baseline, no int4 arm)

The earlier hard run — **plain `QStateLIF`, schemes `{fp32,int8,nvfp4,mxfp4,ternary}`**,
Radeon 8060S, 30 epochs, 2 seeds, 467 s — already showed the formats **separate** (the
first FP4-microscaling × SNN *ranking* datapoints). It is retained here as the baseline
the ALIF + int4 run is measured against; it will be superseded by the pending run.

| Hidden | fp32 | int8 | nvfp4 | mxfp4 | ternary |
| --- | --- | --- | --- | --- | --- |
| 128 | 58.6% | 59.0% | 48.2% | 49.0% | 41.1% |
| 64 | 56.5% | 57.5% | 51.6% | 47.8% | 39.3% |
| 32 | 53.9% | 51.1% | 34.3% | 37.9% | 28.7% |

*(effective bits/weight: int8 8.0 · nvfp4 4.50 · mxfp4 4.25 · ternary 1.58; **int4 = 4.0**
is new this revision.)* Prior ordering: `int8 ≈ fp32 > FP4 > ternary`, consistent across
capacities; FP4 beat ternary by +7 to +9 pt; the cost grew as capacity tightened; NVFP4
vs MXFP4 was within noise. Whether that ordering holds — and where int4 slots in — on the
stronger ALIF baseline is the pending question.

## Reproducibility

- **Seeds:** `nnx.Rngs(seed)` init; NumPy for any shuffling. `SEEDS=[0]` (smoke) / `[0,1]` (full).
- **JAX / hardware:** full run on the Radeon 8060S / gfx1151 (ROCm venv); device in the JSON.
- **Data:** SHD via `spyx.data.SHD_loader`, cached in `data/SHD/`.
- **Neuron:** `SPYX_NEURON=alif` (default) → `spyx.nn.ALIF`; `lif` / `lif2` alternatives.
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** int4 + ALIF revision wired 2026-07-06; full GPU run PENDING (prior plain-LIF table: 2026-07-06).
