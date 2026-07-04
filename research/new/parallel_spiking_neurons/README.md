# PSU_LIF & ResonateFire vs LIF: accuracy vs speed on SHD

> **STATUS: STUB — reserved for an upcoming study.** The non-result sections below
> are filled in now. Results and Findings are marked `TODO` for the maintainer.

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
- **Models:** one architecture, three neuron models swapped in the recurrent
  layer:
  1. `LIF` — hard-reset baseline, run sequentially with `spyx.nn.run`.
  2. `PSU_LIF` — reset-free linear recurrence, run via its `.parallel(x)`
     associative-scan path.
  3. `ResonateFire` — complex-valued resonate-and-fire, parallel path.
- **Training:** surrogate-gradient BPTT, matched optimizer / epochs / seeds
  across the three neuron models so only the neuron differs.
- **Measured:** test accuracy (quality) and, via
  [`spyx.bench`](../../../src/spyx/bench.py), fwd and fwd+bwd latency, throughput,
  peak memory, and spike-rate energy proxy (speed / cost). Speedup of the
  parallel scan is expected to grow with sequence length T, so the comparison is
  swept over T.

## Spyx modules used

- [`spyx.nn.LIF`](../../../src/spyx/nn.py)
- [`spyx.nn.PSU_LIF`](../../../src/spyx/nn.py)
- [`spyx.phasor.ResonateFire`](../../../src/spyx/phasor.py)
- [`spyx.bench`](../../../src/spyx/bench.py) — `benchmark()`, `compare()`,
  `format_table()`

## How to run

```bash
# TODO: add the study notebook / script, then:
uv run jupyter nbconvert --to notebook --execute parallel_spiking_neurons.ipynb
```

## Results

**TODO — placeholder, to be filled by the maintainer.** Run each neuron model and
report accuracy alongside `spyx.bench` timings; sweep sequence length T.

| Neuron | Accuracy | Fwd latency | Fwd+bwd latency | Throughput | Peak mem | Spike rate |
| --- | --- | --- | --- | --- | --- | --- |
| `LIF` (baseline) | TODO | TODO | TODO | TODO | TODO | TODO |
| `PSU_LIF` | TODO | TODO | TODO | TODO | TODO | TODO |
| `ResonateFire` | TODO | TODO | TODO | TODO | TODO | TODO |

## Findings

**TODO — placeholder, to be filled by the maintainer.** State whether the parallel
neurons match `LIF` accuracy and how much wall-clock they save, and how the gap
scales with T.

## Reproducibility

- **Seeds:** TODO — record `jax.random.PRNGKey`, data-shuffle, and NumPy seeds.
- **JAX / hardware:** reference machine **Radeon 8060S / gfx1151** (ROCm); record
  the JAX version and ROCm runtime at run time.
- **Spyx commit:** TODO — record the commit hash the study is run at.
- **Date run:** TODO
