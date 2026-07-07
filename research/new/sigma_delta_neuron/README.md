# Sigma-delta / graded-spike neuron: fewer events at matched accuracy

## Title

`spyx.experimental.SigmaDelta` — a graded neuron that transmits only the quantized
*change* in its membrane — vs a rate-coded `spyx.nn.LIF` on temporally-redundant input.

## Paper & arXiv/DOI

- **Title:** novel Spyx neuron; the change-transmission mechanism is prior art.
- **Prior art:** O'Connor & Welling, *Sigma-Delta Quantized Networks*, ICLR 2017
  ([arXiv:1611.02024](https://arxiv.org/abs/1611.02024)); Shrestha et al., Loihi-2
  resonate-and-fire + sigma-delta, ICASSP 2024 ([arXiv:2310.03251](https://arxiv.org/abs/2310.03251)).
  Reference impl is Intel `lava-dl` (`slayer.neuron.sigma_delta`, PyTorch); **no clean
  JAX/Flax graded neuron existed** — this is the first, with an `associative_scan`
  `.parallel` path.
- **Bucket:** new

## Claim under test

On **temporally-redundant** input (a signal that holds a value), a sigma-delta graded
neuron reaches the **same accuracy as a rate-coded LIF while transmitting far fewer
events** — because it emits only the quantized change of its membrane, which is ~0 when
the input is stable. Event rate (fraction of non-zero events per step) is the
device-agnostic energy proxy: each transmitted event is one synaptic op / packet,
graded or binary.

## Method

`Linear → {LIF | SigmaDelta} → Linear → LI`, identical architecture and budget, on a
synthetic **step-and-hold** task: each class's channel band jumps to a constant level
early and holds it to the end (genuine temporal redundancy). Adam, `integral_crossentropy`.
Report per neuron: test accuracy and mean non-zero-event rate (the hidden neuron's
output scanned over the pre-activation). Also the `SigmaDelta.parallel` associative-scan
throughput via `spyx.bench`. Self-contained; no dataset.

## Spyx modules used

- [`spyx.experimental.SigmaDelta`](../../../src/spyx/experimental/sigma_delta.py) / `graded_quant`
- [`spyx.nn.LIF`](../../../src/spyx/nn.py) / `Sequential` / `run`, [`spyx.bench`](../../../src/spyx/bench.py), [`spyx.fn`](../../../src/spyx/fn.py)

## How to run

```bash
SPYX_SMOKE=1 uv run python research/new/sigma_delta_neuron/sigma_delta_bench.py   # tiny, CPU
uv run python research/new/sigma_delta_neuron/sigma_delta_bench.py                # full
```

Writes `sigma_delta_results.json`.

## Findings

**Full run on the Radeon 8060S** (C=32, H=64, 5-class step-and-hold, T=80, 2 seeds):

| Neuron | Test acc | Event rate | Note |
| --- | --- | --- | --- |
| LIF (rate-coded) | 99.4% | 11.0% | fires through the whole hold |
| **SigmaDelta** | **99.8%** | **2.9%** | fires at the step, then silent |

**Sigma-delta transmits 3.84× fewer events at matched (slightly better) accuracy** on
genuinely redundant input — the efficiency mechanism works, and the graded neuron drops
into `spyx.nn.Sequential`/`run` and parallelises with an `associative_scan` like the
binary neurons.

**Honest caveats.** (1) The win is **redundancy-dependent** — on a *continuously-varying*
input (a half-sine, tried first) sigma-delta does **not** win, because the membrane keeps
changing so it keeps emitting. The benefit requires genuinely stable stretches. (2) A
graded event carries more bits than a binary spike, so "3.84× fewer events" is the
event-count story, not a full energy accounting (see `../honest_energy_accounting/`).
(3) On a dense GPU neither sparsity is exploited — the win is realised on event-driven /
neuromorphic hardware. (4) This is the *feedforward* delta form, which drifts by
`O(sqrt(T)·step)` over long sequences (a closed-loop drift-free variant is sequential).

## Reproducibility

- **Seeds:** `nnx.Rngs(seed)` init; NumPy `default_rng(seed)` for the synthetic task. `SEEDS=[0]` (smoke) / `[0,1]` (full).
- **JAX / hardware:** full run on Radeon 8060S / gfx1151 (ROCm); device in the JSON. Runs on CPU too (small).
- **Correctness:** `tests/test_sigma_delta.py` (parallel==sequential, sparsity, telescoping, STE gradient).
- **Spyx commit:** record `git rev-parse HEAD` at run time.
- **Date run:** 2026-07-06.
