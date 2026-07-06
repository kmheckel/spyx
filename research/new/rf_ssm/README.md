# S5-RF: Resonate-and-Fire as a scaled spiking SSM

> **STATUS: SCAFFOLD (2026-07-06).** The neuron
> ([`spyx.experimental.rf_ssm.RFSSM`](../../../src/spyx/experimental/rf_ssm.py))
> and its scan-exactness / init / gradient tests
> ([`tests/test_rf_ssm.py`](../../../tests/test_rf_ssm.py)) are implemented and
> green. The accuracy comparison below is a **SPYX_SMOKE synthetic CPU smoke**
> only. **Findings = PENDING full run** (SSC / psMNIST, human-gated).

## Title

S5-RF: Resonate-and-Fire as a scaled spiking SSM (S5/HiPPO pole init + PRF
decoupled reset).

## Paper & arXiv/DOI

- **Title:** novel Spyx synthesis of two papers.
- **Related prior art:**
  - S5-RF — "Scaling Up Resonate-and-Fire Networks" (**arXiv:2504.00719**):
    initialise resonate-and-fire poles from the S5/HiPPO-LegS spectrum.
  - PRF — "Parallel Resonate and Fire" / decoupled reset
    (**arXiv:2410.03530**): fold the reset onto the imaginary axis so the RF
    recurrence stays linear and parallelisable.
  - [`parallel_spiking_neurons`](../parallel_spiking_neurons/) — the sibling
    study benchmarking `PSU_LIF` / `ResonateFire` vs `LIF`.
- **Bucket:** new

## Claim under test

Giving the reset-free resonate-and-fire neuron
([`spyx.phasor.ResonateFire`](../../../src/spyx/phasor.py)) an **S5/HiPPO-LegS
pole initialisation** and a **PRF-style decoupled reset** (RFSSM) improves
long-range integration over the plainly-initialised, reset-free `ResonateFire`
and the real-leak [`PSU_LIF`](../../../src/spyx/nn.py), **without** losing the
O(log T) associative-scan parallelism — because the decoupled reset lives on the
imaginary axis (orthogonal to the `Re(z)` spike readout) and is independent of
the state, so the complex recurrence `z_t = a z_{t-1} + (x_t + i b)` stays linear
and the parallel scan remains *exactly* equal to the sequential reference.

## Method

- **Neuron:** [`RFSSM`](../../../src/spyx/experimental/rf_ssm.py). Poles
  initialised from HiPPO-LegS eigenvalues `λ_n = -1/2 + i π n` (reusing
  [`spyx.ssm._hippo_legs_diagonal`](../../../src/spyx/ssm.py)) with a per-unit
  learnable log-step, plus a learnable imaginary-axis reset `b`. An LRU-sampled
  init (reusing [`spyx.ssm._init_lru_eigenvalues`](../../../src/spyx/ssm.py)) is
  also available.
- **Baselines:** reset-free `ResonateFire` (no HiPPO init, no reset) and
  real-leak `PSU_LIF`.
- **Scan-exactness** (the load-bearing correctness property) is pinned in
  [`tests/test_rf_ssm.py`](../../../tests/test_rf_ssm.py): sequential `__call__`
  scanned over time equals `.parallel(x)` to tight tolerance, **including with
  the decoupled reset engaged** and for both pole-init modes.
- **Task (this study):** synthetic **delayed cumulative-sign classification** — a
  cue of random sign planted early in a length-`T` noise sequence; the label is
  the cue sign, so the readout must integrate across the whole sequence. This is
  the regime HiPPO init is meant to help; it is a *controlled probe*, not SSC.
- **Architecture:** `Linear(1→H) → neuron → Linear(H→H) → neuron → Linear(H→2)
  → LI` readout, identical across the three neurons; triangular surrogate,
  `optax.lion(3e-4)`, seed 0.
- **Speed:** neuron-primitive fwd / fwd+bwd latency via
  [`spyx.bench`](../../../src/spyx/bench.py), sequential `spyx.nn.run` path vs
  the `.parallel` associative-scan path.

## Spyx modules used

- [`spyx.experimental.rf_ssm.RFSSM`](../../../src/spyx/experimental/rf_ssm.py)
- [`spyx.phasor.ResonateFire`](../../../src/spyx/phasor.py)
- [`spyx.experimental.PSU_LIF`](../../../src/spyx/nn.py)
- [`spyx.ssm`](../../../src/spyx/ssm.py) — `_hippo_legs_diagonal`,
  `_init_lru_eigenvalues` (pole init)
- [`spyx.bench`](../../../src/spyx/bench.py), [`spyx.fn`](../../../src/spyx/fn.py)

## How to run

```bash
# Fast synthetic CPU smoke (default; what CI/agents run). Proves the pipeline
# runs end-to-end and emits finite numbers — NOT a scientific result.
SPYX_SMOKE=1 uv run python research/new/rf_ssm/run_study.py

# Full long-range run (HUMAN-GATED, GPU recommended).
SPYX_SMOKE=0 T=784 N_TRAIN=4096 HIDDEN=128 EPOCHS=40 \
    uv run python research/new/rf_ssm/run_study.py
```

The full **SSC / psMNIST** accuracy runs are separate and **human-gated** — wire
the real loaders ([`spyx.data`](../../../src/spyx/data.py)) and launch on the GPU
box; do not run them inside the agent workflow.

## Results

| Metric | Value | Notes |
| --- | --- | --- |
| Scan-exactness (seq vs parallel) | spikes `max|Δ| = 0.0`; real-trace `max|Δ| ≈ 2.4e-6` | `tests/test_rf_ssm.py`, incl. reset on + LRU init |
| Accuracy (RFSSM / ResonateFire / PSU_LIF) | PENDING | full run only |
| Fwd / fwd+bwd latency | PENDING | `spyx.bench`, full run |
| Spike rate (energy proxy) | PENDING | `spyx.bench` |

## Findings

**PENDING full run.** The neuron and its correctness guarantees are in place and
tested (the decoupled reset provably preserves the exact parallel scan), but the
long-range accuracy comparison vs `ResonateFire` / `PSU_LIF`, and the SSC /
psMNIST numbers, require the human-gated GPU run. Record confirmed / partial /
refuted here once those land — including honest negatives.

## Reproducibility

- **Seeds:** `jax.random.PRNGKey(0)` (data + model); `nnx.Rngs(0)`.
- **JAX / hardware:** smoke on CPU (`conftest`-pinned in tests); full run
  intended for Radeon 8060S / gfx1151.
- **Spyx commit:** `19c997a` (branch `feat/method-app-arch-org`).
- **Date run:** 2026-07-06 (scaffold + smoke only).
