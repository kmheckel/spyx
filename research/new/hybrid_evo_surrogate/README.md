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
surrogate rather than to beat it, and we report exactly that.

## Method

- **Task:** a tiny synthetic spiking classification problem — each class owns a
  band of input channels that spike at an elevated rate; a `Linear → LIF →
  Linear → LI` classifier reads the summed membrane trace (`spyx.fn.integral_*`).
- **Three arms**, identical init, identical optimizer (`optax.adam`), identical
  step budget:
  1. **surrogate** — pure `∇ loss_surrogate` (`spyx.axn` superspike backward).
  2. **es** — pure antithetic NES on the true (hard-spike forward) loss
     (`spyx.experimental.hybrid.es_gradient`), no surrogate at all.
  3. **hybrid** — `spyx.experimental.hybrid.make_hybrid_train_step`
     (`g_s + λ·g_orth`).
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
SPYX_SMOKE=1 uv run python research/new/hybrid_evo_surrogate/run.py   # seconds, CPU
uv run python research/new/hybrid_evo_surrogate/run.py                # fuller config
```

No dataset download — the task is synthetic and generated in-process.

## Results

Filled in by `run.py` (writes `study_results.json`). Representative `SPYX_SMOKE=1`
run on CPU (seed 0); numbers are tiny-regime and illustrative, not a benchmark:

| Arm | Final true loss | Accuracy | Notes |
| --- | --- | --- | --- |
| surrogate | see JSON | see JSON | pure `∇` surrogate |
| es | see JSON | see JSON | pure antithetic NES |
| hybrid | see JSON | see JSON | `g_s + λ·g_orth` |

Hybrid diagnostics (mean over steps): `cosine(g_es, g_s)`, `‖g_orth‖` — see JSON.

## Findings

Honest reading of the `SPYX_SMOKE=1` run (seed 0; exact numbers in
`study_results.json`):

- **Surrogate wins in this regime.** Pure surrogate descent reaches the lowest
  true loss (~1.47, ~33% acc on a 3-class toy). It is also the cheapest arm by
  far — one `jax.grad` per step vs. `2K` forward evals for the ES/hybrid arms.
- **Pure ES is the weakest arm** (true loss ~2.68). Antithetic NES on a
  Gaussian-smoothed Heaviside loss is high-variance with a small sample budget,
  so it learns slowly.
- **Hybrid is *dragged below* the surrogate, not lifted above it** (true loss
  ~2.53). The diagnostics explain why: measured `‖g_orth‖ ≈ 41` while
  `‖g_s‖ ≈ 8.6`, i.e. the orthogonal ES correction is ~5× larger in magnitude
  than the surrogate bulk direction, and `cosine(g_es, g_s) ≈ 0.28` (weak
  alignment, so most of `g_es` survives the projection). With `λ = 0.5` the
  update is dominated by noisy ES, and hybrid ends up tracking ES rather than
  the surrogate. This refutes the naive hope that "orthogonalise + add" is
  free — **`λ` must be scaled to `‖g_s‖ / ‖g_orth‖`, not set to O(1)**.
- **Net:** a negative/boundary result in the cheap regime, reported as such per
  the research guidance. The method is only plausible when (a) the surrogate is
  genuinely biased, (b) `K` is large enough to shrink ES variance, and (c) `λ`
  is small / adaptively normalised so the correction never overwhelms the bulk
  direction. The mechanics (global orthogonalisation, drop-in grad pytree) are
  verified and correct; the *win* is not demonstrated here and we do not claim
  it. A natural follow-up is a self-normalising `λ_eff = λ · ‖g_s‖ / (‖g_orth‖ +
  eps)`.

## Reproducibility

- **Seeds:** `jax.random.PRNGKey(0)` for perturbations/data; `nnx.Rngs(0)` for
  model init; NumPy `default_rng(0)` for the synthetic dataset.
- **JAX / hardware:** JAX 0.10.2, Flax 0.12.7, CPU (study forces no accelerator
  requirement). Timing is not the point of this study; correctness of the
  three-arm comparison is.
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-04 (initial).
