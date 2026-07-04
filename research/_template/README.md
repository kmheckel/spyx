# <Study title>

> Copy this file into a new folder under `reproductions/`, `extensions/`, or
> `new/` and fill in every section. Keep the headers; write `N/A` if a section
> genuinely does not apply. Do not delete headers.

## Title

A one-line name for the study.

## Paper & arXiv/DOI

- **Title:** <paper title, or "novel — no paper yet">
- **Authors / venue / year:**
- **Link:** arXiv / DOI / URL
- **Bucket:** reproductions | extensions | new

## Claim under test

State, in one or two sentences, the specific claim this study checks (a number, a
trend, a qualitative behaviour). A study tests one claim well rather than many
claims poorly.

## Method

How the experiment is set up: dataset, model architecture, neuron model, training
regime (loss, optimizer, epochs), and what is being measured. Note any deviation
from the original paper and why.

## Spyx modules used

List the Spyx pieces the study depends on, e.g.:

- [`spyx.nn.LIF`](../../src/spyx/nn.py)
- [`spyx.nn.PSU_LIF`](../../src/spyx/nn.py)
- [`spyx.phasor.ResonateFire`](../../src/spyx/phasor.py)
- [`spyx.bench`](../../src/spyx/bench.py)

## How to run

Exact commands to reproduce, e.g.:

```bash
uv run jupyter nbconvert --to notebook --execute study.ipynb
# or
uv run python run.py --seed 0
```

Note any dataset download / preprocessing step.

## Results

| Metric | Value | Notes |
| --- | --- | --- |
| Accuracy | | |
| Fwd latency (median) | | from `spyx.bench` |
| Fwd+bwd latency (median) | | from `spyx.bench` |
| Throughput | | |
| Peak memory | | |
| Spike rate (energy proxy) | | |

## Findings

What the numbers say about the claim: confirmed, partially confirmed, or refuted,
and any surprises. Be specific and honest — record failures too.

## Reproducibility

- **Seeds:** `jax.random.PRNGKey(...)`, data-shuffle seed, NumPy seed.
- **JAX / hardware:** JAX version, accelerator (e.g. Radeon 8060S / gfx1151),
  driver / ROCm or CUDA runtime.
- **Spyx commit:** `<git hash>`
- **Date run:**
