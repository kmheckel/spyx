# SSM → spiking transfer: pretrain S5Diag, transplant eigenvalues into ResonateFire

> **STATUS: METHOD + TRANSFER VERIFIED (2026-07-03).** The eigenvalue transfer is
> exact and asserted (`max|a − λ| = 0`); the end-to-end pipeline runs on CPU via
> `SMOKE=1`. The accuracy/wall-clock *comparison* on real SHD has not been run
> here — the SMOKE numbers are synthetic-data noise and are **not** a result.

## Title

SSM → spiking transfer: warm-starting a `ResonateFire` spiking neuron from a
pretrained `S5Diag` state-space backbone via exact diagonal-eigenvalue transfer.

## Paper & arXiv/DOI

- **Title:** novel — no paper yet.
- **Bucket:** new
- **Related prior art:**
  - **Q-S5** (Abreu, Pedersen, Heckel, Pierro, *Q-S5: Towards Quantized State
    Space Models*, [arXiv:2406.09477](https://arxiv.org/abs/2406.09477), 2024).
    Q-S5 shows a fully-quantized S5 drops <1% on sMNIST / most of LRA, with the
    *recurrent* eigenvalues needing ≥8-bit while feedforward components compress
    much further (~6× memory at heterogeneous precision). That makes an S5-style
    diagonal SSM a cheap, compressible, well-behaved *backbone* — exactly the
    thing we want to pretrain and then donate to a spiking neuron.
  - **ResonateFire study**
    ([`../parallel_spiking_neurons/`](../parallel_spiking_neurons/)), where
    `spyx.phasor.ResonateFire` was the *best* neuron on SHD (both accuracy and
    parallel-scan speed) but is a spiking neuron with a surrogate threshold and is
    harder to optimise from scratch than a linear SSM.

## Claim under test

`spyx.ssm.S5Diag` and `spyx.phasor.ResonateFire` are the **same diagonal complex
linear recurrence** under the hood — an `associative_scan` over a per-unit pole —
differing only in that ResonateFire adds a surrogate spiking threshold on
`Re(z)`. Their poles have *identical functional form*:

| | continuous diagonal | realised discrete pole |
| --- | --- | --- |
| `S5Diag` | `A = A_re + i·A_im`, step `dt_s5 = exp(log_dt)` | `λ = exp(A · dt_s5)` |
| `ResonateFire` | `−λ_decay + i·ω`, step `dt_rf` | `a = exp(dt_rf·(−λ_decay + i·ω))` |

**Claim:** a pretrained S5Diag backbone (fast, parallel, stable — the regime
Q-S5 shows compresses well) can be transplanted *eigenvalue-for-eigenvalue* into
ResonateFire as a warm start, and finetuning from that warm start reaches useful
accuracy **faster / better** than training ResonateFire from scratch on the same
budget.

## Method

### The eigenvalue mapping (the scientifically load-bearing part)

Read straight off the two source modules
([`src/spyx/ssm.py`](../../../src/spyx/ssm.py) `S5Diag._complex_matrices`,
[`src/spyx/phasor.py`](../../../src/spyx/phasor.py) `ResonateFire.a`). Fix
ResonateFire's `dt_rf = 1` (its default) and **fold S5's per-state `dt_s5` into
the transferred continuous diagonal**. Then matching `a == λ` gives a *closed
form*:

```
λ_decay = −Re(A) · dt_s5          # ≥ 0 while the S5 pole is stable (|λ| < 1)
ω       =  Im(A) · dt_s5
raw_λ   = inverse_softplus(λ_decay)   # invert ResonateFire's softplus storage
```

so `a = exp(−λ_decay + i·ω) = exp(A·dt_s5) = λ` **exactly**. `run.py` asserts
`max|a − λ| < 1e-3` after assignment (observed: `0.0`).

**Assumption / limitation (documented, not hidden):** ResonateFire enforces
`λ_decay = softplus(raw_λ) ≥ 0`, i.e. `|a| ≤ 1`, *by construction*. Any S5 state
that drifted to `Re(A) ≥ 0` during pretraining is an unstable, magnitude-growing
pole (`|λ| ≥ 1`) that ResonateFire **cannot represent**; those are clamped to the
stability boundary and *counted* in the transfer diagnostics. With the HiPPO-LegS
init (`Re(A) = −½`) and short training this set is empty, but we report it rather
than assert a fake mapping.

**Linear weights.** Both classifiers wrap the recurrent core in the *same* real
`nnx.Linear` `encoder` (in→hidden) and `readout` (hidden→classes), so those
kernels/biases transfer verbatim. S5Diag's *internal* complex `B`/`C`/`D` are SSM
mechanics with no ResonateFire counterpart and are **not** transferred — a real,
acknowledged gap: the warm start hands over the *dynamics* (poles) and the *outer*
projections, but the spiking neuron must relearn how input current maps through a
thresholded `Re(z)` (which is why the `@init` readout accuracy is low before
finetuning).

### Pipeline (`run.py`)

1. **Pretrain** `S5Classifier` = `Linear → S5Diag → Linear`, time-major, output a
   `(B, T, classes)` logit trace scored by `spyx.fn.integral_crossentropy` /
   `integral_accuracy` (sum over time), `optax.adam`, seed 0.
2. **Transfer**: build `RFClassifier` = `Linear → ResonateFire.parallel → Linear`,
   run `transfer_eigenvalues` (closed form above + verbatim encoder/readout copy),
   **assert the pole match**, and record the pre-finetune `@init` accuracy.
3. **Finetune** the transferred ResonateFire on the same data/budget.
4. **Baseline**: an identical `RFClassifier` trained **from scratch** (same seed,
   same budget). Report accuracy + wall-clock for transferred vs scratch, plus the
   S5 pretraining cost.

- **Data:** SHD, bit-packed `.npz` via `SHD_CACHE` (same mechanism as
  [`../parallel_spiking_neurons/run_study.py`](../parallel_spiking_neurons/run_study.py)).
- **`SMOKE=1`:** a tiny synthetic class-separable spike dataset (T=16, C=16,
  hidden=16, 4 classes) that exercises the *entire* transfer path on CPU in <1s.
  It validates the mechanism, not the science.

## Spyx modules used

- [`spyx.ssm.S5Diag`](../../../src/spyx/ssm.py) — diagonal complex SSM backbone
  (HiPPO-LegS init, learnable `log_dt`, `associative_scan`).
- [`spyx.phasor.ResonateFire`](../../../src/spyx/phasor.py) — complex
  resonate-and-fire spiking neuron; `.parallel` associative-scan path.
- [`spyx.fn`](../../../src/spyx/fn.py) — `integral_crossentropy`,
  `integral_accuracy`.
- [`spyx.axn`](../../../src/spyx/axn.py) — `triangular` surrogate.
- Backbone could itself be a **quantized** S5 (Q-S5): the recurrent poles kept at
  ≥8-bit (which is precisely what the transfer reads out) and the feedforward path
  compressed via [`spyx.quant`](../../../src/spyx/quant.py).

## How to run

```bash
# Fast path check (synthetic, CPU, exercises the transfer end-to-end):
SMOKE=1 uv run python research/new/ssm_to_spiking_transfer/run.py

# Real SHD run (prebuilt bit-packed cache via SHD_CACHE, else prestages once):
SHD_CACHE=/path/to/shd_cache.npz EPOCHS=40 \
  uv run python research/new/ssm_to_spiking_transfer/run.py
```

Writes `study_results.json` (config, transfer diagnostics — pole error, clamped
state count, `|λ|` range — and accuracy/wall-clock for all four stages).

## Results

**Transfer correctness (the verified part).** On both the SMOKE task and at
HIPPO-LegS init the eigenvalue transfer is **exact**:

| Quantity | Value |
| --- | --- |
| `max\|a − λ\|` post-transfer | **0.0** (< `1e-3` tol) |
| S5 states clamped for instability | 0 / 16 |
| S5 discrete `\|λ\|` range | `[0.978, 0.999]` (stable) |

**Full SHD (40 epochs, hidden=64, Radeon 8060S / gfx1151), commit `78ef9db`:**

| Stage | Test acc | Train time | Note |
| --- | --- | --- | --- |
| S5Diag backbone (non-spiking) | **61.5%** | 4.54 s | trained, stable |
| ResonateFire — transfer @init | 5.6% | – | before any finetune (≈ chance) |
| ResonateFire — transfer + finetune | 27.6% | 1.21 s (5.75 s total) | **worse than scratch** |
| ResonateFire — from scratch | **48.7%** | 1.21 s | baseline |

At transfer time: pole match `max|a−λ| = 8.4e-8` (exact), but **39 / 64 S5 states
were unstable** (`|λ| ∈ [0.954, 1.022]`, i.e. at/above the unit circle) and had to
be clamped.

**Result: the transfer hurt** — warm-started ResonateFire (27.6%) trained *worse*
than ResonateFire from scratch (48.7%).

## Findings

- **The two modules are provably the same recurrence** — pole match is exact and
  closed-form, so ResonateFire *is* a spiking-thresholded S5Diag state. This
  mechanism is the reusable, verified result regardless of downstream accuracy.
- **But naive eigenvalue transfer does not help — it hurts.** Two reasons, both
  real: (1) **most of S5's useful modes sit at the stability boundary** (39/64 had
  `|λ| ≥ 1`), which a softplus-stable resonate-and-fire neuron cannot represent, so
  clamping discards exactly the dynamics S5 relied on; (2) **only the poles
  transfer, not the `B`/`C` input/output coupling** where much of S5's learning
  lives — so you warm-start a fraction of the model into a constrained regime and
  the spiking finetune spends its budget fighting the clamp rather than benefiting.
- **The loudest signal is orthogonal to the hypothesis:** the *non-spiking* S5Diag
  backbone (61.5%) beat every spiking neuron measured here — LIF 34%, RF-scratch
  49%, PSU_LIF 8%. On SHD accuracy, the continuous SSM is simply the stronger
  model; the interesting efficiency question is therefore **not** "SSM → spiking"
  but "how cheap can the SSM itself be" — which is exactly Q-S5's quantized-S5
  program (see [`../../reproductions/qs5/`](../../reproductions/qs5/)).

**Next:** rather than transfer S5 → spiking, pursue efficient S5 directly
(quantized per Q-S5, or a hybrid SSM-mixer + spiking readout); if transfer is
revisited, it needs a principled `B`/`C` → input-current map and a way to honour
boundary poles instead of clamping them.

## Reproducibility

- **Seeds:** `nnx.Rngs(0)` (S5 backbone), `nnx.Rngs(1)` (both ResonateFire
  classifiers, so transfer vs scratch share init). Synthetic data seeded with
  `np.random.default_rng(0)`.
- **Transfer tolerance:** `POLE_TOL = 1e-3`, `DECAY_FLOOR = 1e-4`.
- **Validation:** `SMOKE=1 uv run python research/new/ssm_to_spiking_transfer/run.py`
  completes on CPU (`backend=cpu`) in <1s.
- **Date:** 2026-07-03.
