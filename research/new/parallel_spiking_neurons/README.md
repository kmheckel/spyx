# PSU_LIF & ResonateFire vs LIF: accuracy vs speed on SHD

> **STATUS: FIRST RESULTS (2026-07-03).** Run on the Radeon 8060S (gfx1151) at
> commit `23b271e`. A controlled small-scale comparison (hidden=64, ~3k-sample
> SHD subset, 40 epochs) — not a state-of-the-art run; the point is a fair
> neuron-to-neuron comparison, not a leaderboard number.

## Title

PSU_LIF & ResonateFire vs LIF: accuracy vs speed on SHD.

## Paper & arXiv/DOI

- **Title:** novel — no paper yet.
- **Bucket:** new
- **Related prior art:** [Stochastic Parallelizable Spiking Neurons
  (`../../SPSN/`)](../../SPSN/), which motivates the parallelizable neurons under
  test here.

## Claim under test

Reset-free parallelizable spiking neurons —
[`spyx.nn.PSU_LIF`](../../../src/spyx/nn.py) and
[`spyx.phasor.ResonateFire`](../../../src/spyx/phasor.py) — can train to
**accuracy competitive with a standard hard-reset**
[`spyx.nn.LIF`](../../../src/spyx/nn.py) on SHD while being **substantially faster
to train**, because their linear membrane recurrence admits an O(log T)-depth
parallel scan instead of the O(T) sequential scan a hard-reset neuron requires.

## Method

- **Dataset:** Spiking Heidelberg Digits (SHD), same preprocessing as the
  reproductions in [`../../paper/SHD_jax.ipynb`](../../paper/SHD_jax.ipynb) and
  the sweeps in [`../../scaling_experiments/`](../../scaling_experiments/).
- **Architecture:** `Linear(128→64) → neuron → Linear(64→64) → neuron →
  Linear(64→20) → LI` readout, with the hidden `neuron` swapped between `LIF`,
  `PSU_LIF`, and `ResonateFire`. Triangular surrogate, `optax.lion(3e-4)`, seed 0,
  40 epochs, T=128, 128 input channels — identical across the three.
- **Accuracy** is trained under the *same* sequential `spyx.nn.run` BPTT loop for
  all three neurons, so any accuracy difference is attributable to the neuron
  model alone (not the execution path).
- **Speed** is measured separately at the neuron-primitive level with
  [`spyx.bench`](../../../src/spyx/bench.py) (fwd / fwd+bwd latency, throughput,
  peak mem, spike-rate energy proxy): each neuron's sequential `spyx.nn.run` path,
  plus the `.parallel(x)` associative-scan path that `PSU_LIF` / `ResonateFire`
  support and `LIF` cannot. The parallel speedup grows as the device gains slack
  (smaller batch/hidden, longer T) — see the sweep in
  [`../../../research/README.md`](../../README.md) and the
  [parallel-neurons explanation](../../../docs/explanation/parallel-spiking-neurons.md).
- **Reproduce:** [`run_study.py`](run_study.py).

## Spyx modules used

- [`spyx.nn.LIF`](../../../src/spyx/nn.py)
- [`spyx.nn.PSU_LIF`](../../../src/spyx/nn.py)
- [`spyx.phasor.ResonateFire`](../../../src/spyx/phasor.py)
- [`spyx.bench`](../../../src/spyx/bench.py) — `benchmark()`, `compare()`,
  `format_table()`

## How to run

```bash
# Prebuilt SHD cache (bit-packed .npz) via SHD_CACHE, else it prestages once:
JAX_PLATFORMS=rocm SHD_CACHE=/path/to/shd_cache.npz EPOCHS=40 \
  PYTHONPATH=src python research/new/parallel_spiking_neurons/run_study.py
```

Writes `study_results.json` (accuracy rows + neuron-primitive bench rows).

## Results

**Accuracy** (SHD test, 40 epochs, hidden=64, identical training loop):

| Neuron | Test accuracy | Training wall-clock | Notes |
| --- | --- | --- | --- |
| `LIF` (baseline) | **31.7%** | 13.9 s | monotonic, stable |
| `PSU_LIF` | 8.3% | 13.8 s | peaks ~11% @ ep10 then **destabilises** |
| `ResonateFire` | **38.5%** | 14.8 s | best; monotonic, stable |

**Neuron-primitive speed** (`spyx.bench`, T=256, batch=64, hidden=256, 8060S):

| Neuron / path | Fwd (ms) | Fwd+bwd (ms) | Speedup vs `LIF` seq | Spike rate |
| --- | --- | --- | --- | --- |
| `LIF` (sequential) | 1.86 | 6.32 | 1.0× | 0.12 |
| `PSU_LIF` (sequential) | 1.86 | 6.32 | 1.0× | 0.21 |
| `PSU_LIF` (**parallel**) | 0.23 | 0.84 | **8.2× / 7.5×** | 0.21 |
| `ResonateFire` (sequential) | 1.94 | 6.62 | 0.96× | 0.17 |
| `ResonateFire` (**parallel**) | 0.34 | 1.29 | **5.5× / 4.9×** | 0.17 |

(The parallel speedup here is in the device-slack regime; it shrinks toward ~1–3×
when the GPU is saturated and grows to 100×+ at longer T / smaller batch — see the
sweep in [`../../README.md`](../../README.md).)

## Findings

- **ResonateFire wins on *both* axes** for this task: higher accuracy than `LIF`
  (38.5% vs 31.7%) *and* ~5× faster fwd/bwd via its parallel complex scan. The
  oscillatory (resonate-and-fire) dynamics appear well matched to SHD's temporal /
  spectral structure — a promising, publishable direction.
- **`PSU_LIF` is fast but does not train stably here.** It is the fastest neuron
  (8× fwd/bwd) but accuracy peaks early (~11%) and then collapses. Removing the
  reset makes the membrane an *unbounded* linear accumulator (note its ~1.7×
  higher spike rate), and nothing regularises the state, so BPTT destabilises at
  this configuration. The speed is real; the neuron needs a bounding mechanism
  (membrane normalisation, a saturating readout, a small leak floor, or gradient
  clipping) before the parallel speed is usable. **Efficiency ≠ trainability.**
- **Speed and accuracy are separable wins.** The parallel scan is a pure
  execution-path speedup (identical math to the sequential scan); the accuracy
  differences come entirely from the neuron model. ResonateFire happens to be good
  on both; PSU_LIF trades trainability for its extra speed.

**Next:** stabilise `PSU_LIF` (normalised / bounded membrane) and re-measure;
scale ResonateFire to a full-size SHD run (hidden 256–512, full train set,
delays) to see if the accuracy edge holds; add NIR export for ResonateFire once a
neuromorphic resonate-and-fire primitive is available.

## Reproducibility

- **Seeds:** `nnx.Rngs(0)`; SHD prestage from the shared cache (bit-packed,
  `sample_T=128`, `channels=128`, batch 256; 12 train / 8 test batches).
- **JAX / hardware:** **Radeon 8060S / gfx1151**, ROCm nightly (gfx1151,
  2026-06-08 build), `jax==0.8.2` (`jaxlib`/`jax_rocm7_plugin` 0.8.2+rocm).
- **Spyx commit:** `23b271e`.
- **Date run:** 2026-07-03.
