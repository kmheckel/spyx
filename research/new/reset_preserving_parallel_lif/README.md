# Reset-preserving parallel LIF (FPT scan)

> **STATUS: FINDINGS PENDING full GPU run.** The numerical-equivalence result is
> established and tested; the throughput speedup claim needs the human-gated
> large-`T` GPU run (`SPYX_SMOKE=0`). The smoke run in this folder is a
> CPU/tiny-shape path-check only — its latencies are not meaningful.

## Title

Reset-preserving parallel LIF: keeping the exact hard reset while parallelising
the time loop via a fixed-point-threshold (FPT) associative scan.

## Paper & arXiv/DOI

- **Title:** novel implementation — no paper of our own yet.
- **Bucket:** new
- **Method / prior art:**
  - **FPT** — *Fixed-Point RNN*, Zhang et al., [arXiv:2506.12087](https://arxiv.org/abs/2506.12087):
    parallelise a nonlinear recurrence by a fixed-point iteration whose inner
    solves are linear associative scans.
  - **Bullet Trains** — [arXiv:2603.13283](https://arxiv.org/abs/2603.13283):
    reset-preserving parallelisation of spiking neurons (segment-scan
    formulation) — the direct prior art for a *reset-preserving* parallel LIF.

## Claim under test

A hard-reset LIF — which normally forces a sequential `O(T)` scan because each
step's reset depends on the previous step's (nonlinear) spike — can be evaluated
in `O(K log T)` parallel depth while **preserving the exact subtractive reset**,
using an FPT fixed-point iteration whose inner solves are the same
`jax.lax.associative_scan` machinery that
[`spyx.nn.PSU_LIF`](../../../src/spyx/nn.py) uses reset-free. Two sub-claims:

1. **Exactness.** With `K >= T` the FPT `.parallel` path reproduces the
   sequential [`spyx.nn.LIF`](../../../src/spyx/nn.py) spike train *exactly*.
2. **Speed.** For long sequences on an accelerator, the FPT path (`O(K log T)`)
   is faster than the sequential `spyx.nn.run` path (`O(T)`) — **PENDING** the
   full GPU run.

This neuron is the **reset-PRESERVING complement** to the reset-FREE `PSU_LIF`:
`PSU_LIF` drops the reset to get one associative scan; `ParallelResetLIF` keeps
the reset and pays `K` scans to reconstruct it.

## Method

- **Neuron:**
  [`spyx.experimental.parallel_reset.ParallelResetLIF`](../../../src/spyx/experimental/parallel_reset.py).
  Its `__call__` is byte-for-byte [`spyx.nn.LIF`](../../../src/spyx/nn.py)
  (`V = beta*V + x - spikes*threshold`) — the sequential reference. Its
  `.parallel(x, K)` implements the FPT solve:
  - Split the pre-input membrane as `V_t = U_t - R_t`, with reset-free part
    `U_t = Σ_{j<t} beta^{t-1-j} x_j` and accumulated reset
    `R_t = threshold · Σ_{j<t} beta^{t-1-j} s_j`. Both are one-step-shifted
    associative scans over the shared leak (`spyx.nn._leaky_associative_op`).
  - Seed `s = H(U - threshold)` (reset `R = 0`), then iterate `K` times:
    recompute `R` from the current spikes and re-threshold
    `s = H(U - R - threshold)`.
  - Iteration `k` makes the first `k+1` timesteps exact (a correctness
    *wavefront* advancing one step per iteration), so `K = T` is exact and small
    `K` is near-exact where reset **cascades** are short (sparse / low-`beta`
    activity — the regime trained SNNs occupy).
- **Equivalence measurement:** max and mean spike-train mismatch of
  `.parallel(x, K)` vs. the sequential `spyx.nn.run` reference, at `K = T`
  (exact) and `K = 3` (fast approx).
- **Speed measurement:** [`spyx.bench`](../../../src/spyx/bench.py) comparison of
  three paths — `ParallelResetLIF` sequential (`spyx.nn.run`), `ParallelResetLIF`
  FPT (`.parallel`, `K=3`), and `PSU_LIF` reset-free `.parallel` — swept over
  `seq_len`. As with the other parallel-neuron studies, the speedup grows with
  device slack (longer `T`, smaller batch/hidden).

## Spyx modules used

- [`spyx.experimental.parallel_reset.ParallelResetLIF`](../../../src/spyx/experimental/parallel_reset.py)
- [`spyx.nn.LIF`](../../../src/spyx/nn.py) — the sequential reference
- [`spyx.nn.PSU_LIF`](../../../src/spyx/nn.py) — reset-free parallel baseline
- [`spyx.nn._leaky_associative_op`](../../../src/spyx/nn.py) — the shared scan op
- [`spyx.bench`](../../../src/spyx/bench.py) — `benchmark()`, `format_table()`

## How to run

```bash
# Smoke (CPU, seconds): equivalence + path-check, latencies NOT meaningful.
SPYX_SMOKE=1 uv run python research/new/reset_preserving_parallel_lif/run_bench.py

# Full (HUMAN-GATED, GPU): large T, real throughput numbers.
JAX_PLATFORMS=rocm SPYX_SMOKE=0 \
  uv run python research/new/reset_preserving_parallel_lif/run_bench.py
```

Writes `bench_results.json`. No dataset download.

The tests that pin the equivalence and K-convergence live in
[`tests/test_parallel_reset.py`](../../../tests/test_parallel_reset.py) and are
not network-gated (`uv run pytest tests/test_parallel_reset.py -q`).

## Results

**Numerical equivalence (established, tested).** Across random
inputs / `beta ∈ [0.1, 0.99]` / `threshold ∈ [0.3, 1.5]`:

| Path | `K` | Max spike mismatch | Mean spike mismatch |
| --- | --- | --- | --- |
| FPT `.parallel` vs sequential | `K = T` | **0** (exact) | **0** |
| FPT `.parallel` vs sequential | `K = 3`, sparse/`beta≤0.4` | 0 (regime-exact) | 0 |
| FPT `.parallel` vs sequential | `K = 3`, `beta = 0.5` | 1 (rare single flip) | ~7e-5 |
| FPT `.parallel` vs sequential | `K = 1`, `beta = 0.5` | 1 | ~4e-3 |

`K = 3` reduces the mean mismatch ~60× over `K = 1` in the moderate regime and is
exact in the sparse short-cascade regime; error is monotone-decreasing in `K` and
reaches machine-exact by `K = T`.

**Throughput (`spyx.bench`).** PENDING the human-gated full GPU run. The smoke
run confirms all three paths execute and the equivalence holds; its CPU/tiny-shape
latencies are deliberately not reported as findings.

| Path | Fwd latency | Notes |
| --- | --- | --- |
| `ParallelResetLIF` sequential (`O(T)`) | PENDING | baseline |
| `ParallelResetLIF` FPT (`O(K log T)`) | PENDING | reset-preserving |
| `PSU_LIF` parallel (`O(log T)`) | PENDING | reset-free reference |

## Seeds / hardware / commit

- Equivalence tests: seeds 0–5 (exactness) + 0–19 (convergence), CPU (pinned by
  `tests/conftest.py`).
- Full throughput run: record GPU + commit here when run.

## Interpretation & honest caveats

- **The reset is exact, not approximated.** The neuron the network trains and
  deploys is a true hard-reset LIF; FPT only changes *how* the identical spike
  train is computed.
- **`K` is an accuracy/latency knob.** `K = 3` is the fast default and is
  near-exact only where reset cascades are short (sparse, low `beta`). Dense
  high-`beta` near-threshold activity has long cascades and needs larger `K`
  (up to `T`) for exactness — the correctness wavefront advances one step per
  iteration. Report the `K` used alongside any accuracy number.
- **Speed is unproven until the GPU run.** On CPU / short `T` the `K` extra scans
  can make FPT slower than the sequential scan; the win is an accelerator +
  long-`T` claim. Findings PENDING.
