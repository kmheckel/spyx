# API Reference

Auto-generated from docstrings via [mkdocstrings](https://mkdocstrings.github.io/). Every public function and class is listed; if something doesn't appear here it's internal and subject to change.

## Neuron models

Spiking-neuron layers and the time-major scan helper.

::: spyx.nn

## Surrogate gradients

Factories that return JIT-compiled `jax.custom_gradient` functions. Pass the returned callable as the `activation` argument to any spiking neuron.

::: spyx.axn

## Losses and metrics

All factories return JIT-compiled callables of signature `(traces, targets) -> ...`. Shape checks raise `ValueError` at trace time if `traces` and `targets` disagree.

::: spyx.fn

## Data loading

Grain-based data pipeline. The functional encoders (`rate_code`, `angle_code`, `latency_code`, `shift_augment`) return JIT-compiled callables; the `RateCode` / `AngleCode` / `LatencyCode` / `ShiftAugment` classes are their `grain.MapTransform` counterparts for use inside dataset pipelines.

::: spyx.data

## Training helpers

High-level training loop that wraps `nnx.Optimizer` + `nnx.value_and_grad`. Use `fit(...)` for the common case or `make_train_step` / `make_eval_step` to roll your own loop.

::: spyx.optimize

## NIR interoperability

Import / export [Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/) graphs. Supports feed-forward networks with `Linear`, `Conv`, `LIF`, `CuBaLIF`, and recurrent subgraphs for `RIF`, `RLIF`, `RCuBaLIF`.

::: spyx.nir

## Quantization (optional)

Thin SNN-aware wrapper around Google's [qwix](https://github.com/google/qwix) library. Install with `pip install "spyx[quant]"` (or `uv pip install "spyx[quant]"`).

::: spyx.quant

## State-space models

Diagonal complex-valued SSMs driven by `jax.lax.associative_scan`. Includes LRU, S4D/S5-style HiPPO-LegS init, the full Mamba block, and a minimal H-Net skeleton.

*Available on branch [`claude/spyx-ssm`](https://github.com/kmheckel/spyx/tree/claude/spyx-ssm) — not yet merged to main. Docstrings render only once this module lands in `src/spyx/ssm.py` on the default branch; until then see the branch's source for the full signatures.*

Preview of the exported surface:

- `spyx.ssm.LRU` — Linear Recurrent Unit (Orvieto et al., 2023).
- `spyx.ssm.S5Diag` — diagonal S4D / S5 with HiPPO-LegS initialisation.
- `spyx.ssm.Mamba` — selective SSM core.
- `spyx.ssm.MambaBlock` — full Mamba block (in-proj / depthwise conv / gate / out-proj).
- `spyx.ssm.ChunkedSSM` — minimal H-Net skeleton (chunk-level + token-level SSM).

## Phasor networks

Complex-valued phasor layers with spike-time conversion helpers. Weights are stored as paired `kernel_re` / `kernel_im` float32 parameters so a stock `optax.adam` loop converges (the Wirtinger-gradient caveat that bit the original prototype).

*Available on branch [`claude/spyx-phasor`](https://github.com/kmheckel/spyx/tree/claude/spyx-phasor) — not yet merged to main.*

Preview of the exported surface:

- `spyx.phasor.PhasorLinear` — complex-valued dense layer.
- `spyx.phasor.PhasorActivation` — unit-circle projection (TPAM threshold).
- `spyx.phasor.PhasorReadout` — real logits from complex activations.
- `spyx.phasor.PhasorMLP` — convenience stack.
- `spyx.phasor.phase_to_spikes` / `spikes_to_phase` — codec helpers.
- `spyx.phasor.SpikingPhasor` — spiking-inference wrapper for a `PhasorLinear`.

## Experimental

Research-grade neurons. The contract here is not frozen — APIs may change without deprecation.

::: spyx.experimental
