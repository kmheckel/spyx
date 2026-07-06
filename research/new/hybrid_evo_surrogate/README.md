# Hybrid surrogate + orthogonalised evolutionary correction

## Title

Correcting surrogate-gradient bias in spiking nets with an orthogonalised
evolutionary error term.

## Paper & arXiv/DOI

- **Title:** novel — no paper yet. Closest prior art: **Guided Evolutionary
  Strategies** (Maheswaranathan, Metz, Tucker, Choi, Sohl-Dickstein 2019,
  arXiv:1806.10230), which restricts the ES search to the *surrogate's* subspace.
  This study does the complement: restrict ES to the surrogate's *orthogonal
  complement* and use it as an additive correction.
- **Authors / venue / year:** N/A (Spyx research note).
- **Bucket:** new

## Claim under test

Surrogate-gradient descent through hard-spiking neurons is cheap but *biased*:
the true forward objective uses a Heaviside spike whose gradient is zero almost
everywhere, so the backward pass substitutes a smooth surrogate and descends a
*related* landscape. **Hypothesis:** an antithetic-NES estimate of the true
(hard-spike) loss gradient, projected onto the subspace the surrogate does *not*
already cover (`g_orth = g_es − ⟨g_es, ĝ_s⟩ ĝ_s`) and added back as
`g = g_s + λ·g_orth`, corrects that bias and reaches a lower **true** loss than
either pure surrogate descent or pure ES under a matched step budget.

**Honest expected outcome.** This is a boundary-result-prone idea. Pure ES is
unbiased but high-variance; on a Gaussian-smoothed Heaviside loss its signal is
weak in the parameter dimension, so with a small sample budget `K` the ES
estimate is noisy. Crucially, that noise is *not* small in magnitude — an
antithetic NES estimate `g_es = coeff·ε` has norm that grows with the parameter
dimension and the loss curvature, so `‖g_orth‖` can easily *exceed* `‖g_s‖`.
That makes `g = g_s + λ·g_orth` fragile: unless `λ` is small the orthogonal ES
term dominates and drags the update toward ES noise. We expect the interesting
regime to be a well-matched-but-large-bias surrogate with large `K` and a small,
tuned `λ`; in the cheap smoke regime we expect hybrid to be *pulled below* the
surrogate rather than to beat it, and we report exactly that — then test the
obvious fix (a self-normalising `λ`) and report where it lands (see Findings).

## Method

- **Task:** a tiny synthetic spiking classification problem — each class owns a
  band of input channels that spike at an elevated rate; a `Linear → LIF →
  Linear → LI` classifier reads the summed membrane trace (`spyx.fn.integral_*`).
- **Four arms**, identical init, identical optimizer (`optax.adam`), identical
  step budget:
  1. **surrogate** — pure `∇ loss_surrogate` (`spyx.axn` superspike backward).
  2. **es** — pure antithetic NES on the true (hard-spike forward) loss
     (`spyx.experimental.hybrid.es_gradient`), no surrogate at all.
  3. **hybrid-raw** — `make_hybrid_train_step(..., normalize=False)`
     (`g_s + λ·g_orth`, raw `λ`; the original PR #49 setting).
  4. **hybrid-norm** — `make_hybrid_train_step(..., normalize=True)`, where `λ` is
     a dimensionless *fraction of the surrogate step*: the correction is rescaled
     to `‖λ·‖g_s‖‖` via `λ_eff = λ·‖g_s‖/‖g_orth‖`, so the ES term can never swamp
     the bulk direction regardless of its variance.
- **True vs. surrogate loss:** both are `integral_crossentropy`. They share the
  *same* forward (every `spyx.axn` activation forwards through the identical
  Heaviside step), so the "true" loss is the hard-spike forward loss; they differ
  only in the backward pass — the surrogate arm backprops through the smooth
  surrogate, the ES arm never differentiates at all. This is exactly the
  `loss_true = hard-spike forward eval` case in the algorithm.
- **Measured:** final **true** cross-entropy and accuracy per arm, plus the
  hybrid diagnostics (`cosine` between `g_es` and `g_s`, `‖g_orth‖`) so the
  correction magnitude is visible.

`SMOKE=1` shrinks every dimension (channels, hidden, time, samples, epochs) so
the whole three-arm comparison runs on CPU in a few seconds; the same code path
runs the fuller config without the flag.

## Spyx modules used

- [`spyx.experimental.hybrid`](../../../src/spyx/experimental/hybrid.py) —
  `hybrid_gradient`, `es_gradient`, `make_hybrid_train_step`, `hybrid_diagnostics`
- [`spyx.nn.LIF`](../../../src/spyx/nn.py) / `LI` / `Sequential` / `run`
- [`spyx.axn.superspike`](../../../src/spyx/axn.py)
- [`spyx.fn.integral_crossentropy`](../../../src/spyx/fn.py) / `integral_accuracy`

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/hybrid_evo_surrogate/run.py   # 4-arm demo, seconds
uv run python research/new/hybrid_evo_surrogate/run.py                # fuller config
uv run python research/new/hybrid_evo_surrogate/sweep.py              # multi-seed sweep
```

No dataset download — the task is synthetic and generated in-process. `sweep.py`
sweeps seeds × ES sample count `K` × correction fraction `λ` over a harder regime
and reports the mean Δ(true_loss) = hybrid-norm − surrogate with its seed spread,
so a claimed "win" has to survive noise (writes `sweep_results.json`).

## Results

Filled in by `run.py` (writes `study_results.json`). Representative `SPYX_SMOKE=1`
run on CPU (seed 0); numbers are tiny-regime and illustrative, not a benchmark:

| Arm | Final true loss | Accuracy | Notes |
| --- | --- | --- | --- |
| surrogate | see JSON | see JSON | pure `∇` surrogate |
| es | see JSON | see JSON | pure antithetic NES |
| hybrid-raw | see JSON | see JSON | `g_s + λ·g_orth`, raw `λ` |
| hybrid-norm | see JSON | see JSON | self-normalised `λ` (fraction of step) |

Hybrid diagnostics (mean over steps): `cosine(g_es, g_s)`, `‖g_orth‖`,
`correction_fraction` — see JSON. See the Findings section for the multi-seed
`sweep.py` verdict.

## Findings

**1. The raw-`λ` failure mode is real** (`SPYX_SMOKE=1`, `study_results.json`).
Pure surrogate reaches ~1.47 true loss; pure ES trails at ~2.68; **hybrid-raw is
*dragged below* the surrogate at ~2.53**. Diagnostics explain why: `‖g_orth‖ ≈ 41`
vs `‖g_s‖ ≈ 8.6` (correction ~5× the bulk direction) with `cosine(g_es, g_s) ≈
0.28`, so at `λ = 0.5` the noisy ES term dominates the update. "Orthogonalise +
add" is *not* free — the correction magnitude has to be controlled.

**2. Self-normalising `λ` fixes the failure mode and is safe.** Reinterpreting `λ`
as a fraction of the surrogate step (`normalize=True`,
`λ_eff = λ·‖g_s‖/‖g_orth‖`) removes the blow-up: in the same smoke regime
hybrid-norm recovers to ~1.93 (from raw's 2.53). The multi-seed `sweep.py` on a
harder regime (24ch/32h/4-class, 20 epochs, surrogate floor ≈ 0.684) confirms it
holds across seeds — at `K = 32`:

| λ (fraction) | surrogate | hybrid-norm | hybrid-raw | Δ(norm−sur) | wins |
| --- | --- | --- | --- | --- | --- |
| 0.15 | 0.684 | 0.688 | 0.776 | +0.004 ± 0.011 | 1/3 |
| 0.30 | 0.684 | 0.697 | 0.776 | +0.014 ± 0.009 | 0/3 |
| 0.50 | 0.684 | 0.707 | 0.776 | +0.024 ± 0.019 | 1/3 |

hybrid-raw sits at 0.776 across the board (the same drag); hybrid-norm tracks the
surrogate closely and the gap *grows with `λ`*, exactly as the fraction
interpretation predicts. (The `K = 96` cells were not completed in this run.)

**3. But it does not *beat* the surrogate on these easy tasks.** At the smallest
fraction (`λ = 0.15`) hybrid-norm reaches parity within seed noise (Δ = +0.004,
one seed of three actually wins), and it never clears the surrogate on average.
The reason is a task ceiling, not an implementation gap: the surrogate is already
a good descent direction here and sits near the achievable floor, so there is
little bias left for an orthogonal correction to remove, and the residual ES
variance costs slightly more than the correction is worth.

**Net.** The self-normalising `λ` turns a *fragile* method (hybrid-raw, which can
end up much worse than plain surrogate) into a *safe* one (hybrid-norm ≈ surrogate,
never catastrophically worse) — a real, verified improvement over the PR #49
version. The stronger claim — that the orthogonal-ES correction *beats* surrogate
descent — remains **unproven**; it needs a regime with genuinely large surrogate
bias (mismatched surrogate, deeper/recurrent nets, or a hard-spike objective that
truly diverges from the smooth one) and larger `K`. We report parity-plus-safety,
not a win, and do not overclaim. The mechanics (global orthogonalisation, exact
`λ·‖g_s‖` scaling, drop-in grad pytree) are unit-tested and correct.

## Companion: SHD training-method comparison

The `methods_shd_*.py` scripts in this directory are a sibling study line: instead
of the synthetic task above they pit **0th-order evolution, 1st-order surrogate,
and 0+1 hybrids/SGES against each other on real SHD** (128 channels, 20 classes,
the `Linear→LIF→Linear→LI` classifier, ~9,556 SNN weights). Each script isolates
one variable and writes its own `shd_*_results.json`; all use `integral_crossentropy`
fitness and JIT-scanned ask/tell loops. This directly instruments the thesis
finding that neuroevolution's competitiveness is **dimensionality-dependent**.

| Script | Search space | Result (test acc) | Takeaway |
| --- | --- | --- | --- |
| `methods_shd.py` | full 9,556-d | surrogate **29.2%** · ES 5.4% · SGES 7.4% | isotropic ES *collapses* in full space |
| `methods_shd_hyper.py` / `_population.py` | full vs 256-d hyper | full-ES ~8% vs hyper roughly at par | compression helps ES more than surrogate |
| `methods_shd_jit.py` | 128-d hyper latent, K=512 | **SGES 43.2% > surrogate 34.3% > ES 32.6%** | the honest headline: in a *compressed* space SGES wins |
| `methods_shd_solvers.py` | 512-d latent | surrogate 32.4% ≈ ES 32.8% ≫ CMA-ES 7.1% | naive CMA-ES needs the `std_init` fix |
| `methods_shd_cmaes.py` | 128-d hyper latent | CMA-ES 22.2% · primed 31.5% | adaptive full-cov ES, `std_init` matched to param scale |
| **`methods_shd_crfmnes.py`** | **full 9,556-d (no hypernet)** | **CR-FM-NES 29.6% · primed 23.4%** | **full-space adaptive ES — reproduces the thesis's *hard* config** |

**The CR-FM-NES full-space arm (new).** CR-FM-NES (Nomura & Ono, CEC 2022) is the
rank-1 + diagonal, **O(d)** adaptive ES the thesis actually used to scale
neuroevolution past a million parameters — the one strategy that *can* run directly
on all 9,556 weights where CMA-ES (O(d²)) and isotropic ES cannot. Run here it
reaches **29.6% test / 33.5% train** (loss 84.3 → 2.56 over 500 generations): it
optimises cleanly (a monotone fitness curve, and it beats isotropic ES's 5–8%
full-space collapse by ~4×), but it under-converges relative to both the compressed
SGES arm (43.2%) and the thesis's own SHD number (73.3% test).

This is a **faithful reproduction of the thesis's explicitly-flagged hard
configuration, not a contradiction of it.** The thesis reached 73.3% only after
three changes this smoke config deliberately omits: **Adaptive-LIF** neurons (not
plain LIF), a **spike-rate MSE** objective (not integral cross-entropy), a **LIF**
output layer (not LI), and a **full-dataset, 2,500-epoch** budget (not a fixed
8-batch subset over 500 generations). Train accuracy tops out at 33.5% → the mean
is *under-converged*, not overfit. The takeaway matches the thesis's conclusion:
adaptive ES scales to the full parameter space, but closing the last gap to
surrogate-competitive accuracy needs the richer neuron/objective/budget — which is
why the thesis (and this repo's `spyx.experimental.hybrid`/`sges`) argues for
**hybridising** surrogate and evolutionary methods rather than picking one.

```bash
uv run python research/new/hybrid_evo_surrogate/methods_shd_crfmnes.py   # writes shd_crfmnes_results.json
```

## Reproducibility

- **Seeds:** `jax.random.PRNGKey(0)` for perturbations/data; `nnx.Rngs(0)` for
  model init; NumPy `default_rng(0)` for the synthetic dataset.
- **JAX / hardware:** JAX 0.10.2, Flax 0.12.7, CPU (study forces no accelerator
  requirement). Timing is not the point of this study; correctness of the
  three-arm comparison is.
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-04 (initial three-arm study; self-normalising `λ` +
  multi-seed sweep added same day).
