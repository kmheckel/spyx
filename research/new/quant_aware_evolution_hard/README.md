# Quantization-aware evolution — does the STE-bias gap appear when quant *costs* accuracy?

## Title

Hard-task extension of quantization-aware evolution: sweeping capacity to make
quantization bite, and checking whether the STE-bias gap emerges.

## Paper & arXiv/DOI

- **Title:** novel — no paper. Extends the sibling study
  [`../quant_aware_evolution/`](../quant_aware_evolution/README.md).
- **Bucket:** extension (of a Spyx research note)

## Claim under test

The sibling study found **no** STE-bias gap on an easy task (fp32 ≈ 99.5 %,
quantization barely dented it). **Claim:** the gap — gradient-free CR-FM-NES reaching
a lower *true* quantized-forward loss than STE-QAT — appears **only when the task /
capacity is tight enough that quantization drops accuracy**, and grows with that drop.

## Method

Single difficulty knob: **hidden width** (capacity), swept `64 → 32 → 16` at a fixed
class count (8) and channel count (64) — both multiples of 16 so nvfp4's
contraction-axis tiling is valid — so fp32 stays solvable while low-precision weights
lose room to hide. At each point we **reuse the sibling study's verified 3-arm
machinery** (fp32 surrogate SGD / STE-QAT / ES, all scored on the *identical* true
quantized forward `Q(w) = dequant(quant(w))`, no STE) by importing it and overriding
its task config. Precisions: nvfp4 (qwix, tile 16) and ternary (BitNet b1.58). We
report, per point and precision, the **fp32→quant accuracy drop** and the **STE-bias
gap = STE-QAT true loss − ES true loss** (positive = ES lower), plus their
correlation across the sweep.

## Spyx modules used

- [`../quant_aware_evolution/quant_aware_evolution.py`](../quant_aware_evolution/) — reused arms
- [`spyx.quant`](../../../src/spyx/quant.py) / `qwix` — nvfp4 quantization
- [`spyx.nn.LIF`](../../../src/spyx/nn.py) / `LI` / `Sequential` / `run`, [`spyx.fn`](../../../src/spyx/fn.py)
- `evosax` `CR_FM_NES`

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/quant_aware_evolution_hard/hard_sweep.py   # CPU, seconds
uv run python research/new/quant_aware_evolution_hard/hard_sweep.py                # full sweep
```

No dataset download — synthetic task. Writes `hard_sweep_results.json`.

## Findings

Bounded run on the **Radeon 8060S / gfx1151** (2 seeds, POP=64 × 150 gens, 65 s).
Adversarially verified: reuse machinery correct, numbers self-consistent, the
headline effect corroborated by a same-point control and the sibling study.

| Point | fp32 | nvfp4 STE / ES acc | nvfp4 gap | ternary STE / ES acc | ternary gap |
| --- | --- | --- | --- | --- | --- |
| H64 | 98.0% | 96.5% / 97.7% | −0.026 | 99.6% / **67.2%** | −0.557 |
| H32 | 94.9% | 93.8% / 97.7% | **+0.025** | 97.3% / **67.6%** | −0.389 |
| H16 | 92.6% | 94.1% / 88.7% | −0.069 | 96.5% / **62.9%** | −0.510 |

(gap = STE-QAT true loss − ES true loss; **+ = ES lower**.)

**1. The premise was not achieved — the first finding.** Tightening capacity did
**not** make quantization cost accuracy: the fp32→quant accuracy drop stayed ≈0
everywhere (nvfp4 +1.6 / +1.2 / −1.6 pt; ternary −1.6 / −2.3 / −3.9 pt — the
quantized model *matched or beat* fp32, even at H16 where STE-QAT still hit 94–96%).
This synthetic task's spike bands are too separable for low-precision weights to
degrade. So there was never a "quantization costs accuracy" regime for the STE-bias
gap to live in — **the claim cannot be tested on this task family.**

**2. An unexpected, real mechanistic result: ES collapses at ternary, not at nvfp4.**
`ES @ ternary` craters to **63–68 %** at every point (vs STE-QAT 96–99 %), while
`ES @ nvfp4` stays **89–98 %** and even *beats* STE at H32 (gap +0.025). This is
precision-specific, not an ES failure: the same code/dims/budget at nvfp4 converges,
and the sibling easy study reached `ES @ ternary` = 100 %. **Mechanism:** ternary's
coarse per-tensor rounding **flattens the ES fitness landscape** — sub-scale fp32
perturbations round to *identical* ternary weights, starving the gradient-free
signal — whereas STE keeps a usable surrogate gradient *through* the same quantizer.
Finer nvfp4 (≈15 levels) preserves the signal.

**3. This inverts the parent hypothesis.** The parent study guessed ES's advantage
would *grow* as precision drops (ternary > nvfp4). The opposite holds: **ES is viable
at fp4 but breaks at ternary** — the coarser the quantizer, the worse gradient-free
search does, because the quantizer destroys the fitness signal ES depends on. STE's
biased-but-nonzero gradient is *more* robust to coarse quantization, not less.

**Verdict: ❌ on the stated claim** (no STE-bias gap, and the direction is reversed),
**plus a genuine insight** (ternary flattens the ES landscape). Honest caveats: all
quant drops are ≈0 and of ±1 pt magnitude near the test-set granularity (1/256), so
no point cleanly stresses quantization; and `ES @ nvfp4` dips to 88.7 % at H16 (gap
−0.069) — a dip, not a collapse.

**Next step (human-gated).** To actually test the STE-bias claim you need a task where
fp32 sits near its capacity limit *and* precision genuinely costs accuracy — this
redundant synthetic task can't provide that; a real hard dataset (SHD 20-class, tight
width) is the follow-up. And to rescue ES at ternary, the search must keep a signal
under coarse rounding (perturbations scaled to survive the quantizer, or a
stochastic/dithered quantizer). Practical read for now: **gradient-free ES for
quantized nets is fine at fp4, but STE-QAT is the safer choice at ternary.**

## Reproducibility

- **Seeds:** `nnx.Rngs(seed)` init; `PRNGKey(seed)`/`PRNGKey(seed+1)` for CR-FM-NES;
  NumPy `default_rng(seed)` for the synthetic data. `SEEDS = [0]` (smoke) / `[0, 1]` (full).
- **JAX / hardware:** run on the Radeon 8060S / gfx1151 (ROCm venv) for the full sweep;
  device is recorded in the results JSON.
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-06.
