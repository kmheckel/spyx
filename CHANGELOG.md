# Changelog

All notable changes to Spyx are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] — unreleased

The **modernization release** and Spyx's first stable major: Spyx moves from
DeepMind Haiku to **Flax NNX** and gains state-space, phasor, and quantization
modules. This is a **breaking release** — see the
[Haiku→NNX migration guide](docs/how-to/migrate-haiku-to-nnx.md).

### Changed (breaking)

- **Neural-network backend is now Flax NNX** instead of Haiku. Models are
  stateful `nnx.Module` objects; parameters live in the module, not an external
  pytree. `hk.transform` / `.init` / `.apply` are gone. Every layer takes an
  `rngs=` argument at construction.
- **Neuron models** (`LIF`, `ALIF`, `CuBaLIF`, `LI`, `IF`, recurrent variants)
  reimplemented as `nnx.Module`. `__call__(x, V) -> (spikes, V)`;
  `initial_state(batch)` replaces Haiku's `initial_state`.
- **`spyx.nn.Sequential` + `spyx.nn.run`** replace `hk.DeepRNN` +
  `hk.dynamic_unroll` for stacking and time-unrolling. `run` is time-major
  `(T, B, C)`.
- **`spyx.loaders` removed**; use **`spyx.data`**, now built on **Google Grain**.
  Loaders are iterables of `State(obs, labels)` batches (was: return full
  arrays); constructor args are keyword-only; `obs` is bit-packed along time
  (`jnp.unpackbits` before use); `train_epoch()` no longer takes a PRNG key.
- **`spyx.nir`** rewritten to walk NNX modules. `from_nir` is now run-and-return:
  `from_nir(graph, input_data, dt=1, return_all_states=False) -> (model, outputs)`
  (was `from_nir(graph, dt, rngs) -> model`). It reconstructs the model and runs
  it over the time axis of `input_data`; `return_all_states=True` also returns
  per-timestep neuron states (membrane traces).
- **Mixed precision** via `jmp` removed; pass `dtype=` / `param_dtype=` to layers.
- **Python support** is now `>=3.11, <3.13` (was `>=3.10`).
- Dependencies: added `flax>=0.11`, `grain`; removed `dm-haiku`, `jmp`.

### Added

- **`spyx.ssm`** — state-space layers: `LRU`, `S5Diag` (HiPPO-init diagonal
  SSM), `Mamba` / `MambaBlock` (selective SSM), and `ChunkedSSM`.
- **`spyx.phasor`** — complex-valued phasor networks: `PhasorLinear`,
  `PhasorActivation`, `PhasorReadout`, `PhasorMLP`, and `SpikingPhasor` with
  phase↔spike conversion helpers.
- **`spyx.quant`** — int8/int4 and BitNet-ternary quantization via a
  [qwix](https://github.com/google/qwix) wrapper (`spyx[quant]` extra).
- **`spyx.optimize`** — high-level training loop: `fit`, `make_train_step`,
  `make_eval_step`.
- **`spyx.data`** — latency (time-to-first-spike) encoding; on-device SHD
  prestaging; configurable Grain `worker_count`.
- Latency encoding, expanded `spyx.fn` losses/metrics, and shape-checked traces.
- `spyx.nn.Flatten` — a stateless flatten layer for use in `Sequential` and NIR
  graphs (`flax.nnx` has no built-in flatten).
- Diátaxis-structured docs (tutorials / how-to / reference / explanation), a
  Haiku→NNX migration guide, new SSM & phasor tutorial notebooks.
- Tooling: `uv` + `ruff` CI, `ty` static type checking (`uv run ty check`), and
  `scripts/check_install.py` end-to-end smoke check.

### Fixed

- `spyx.quant` rules matched nothing on NNX models — qwix's `module_path` regex
  is checked against the NNX attribute path, so `.*Linear.*` never matched and
  `quantize()` was a silent no-op. Rules now select dense/conv work by op
  (`dot_general` / `conv_general_dilated`).
- NIR export was broken for several layer types (untested before): `IF`
  (`r=1` int vs. array), `Flatten` (referenced the non-existent `nnx.Flatten`;
  malformed node), and `Conv2d` (missing `input_shape`, `padding="SAME"`).
  All fixed, and `nnx.Conv` / `spyx.nn.SumPool` now round-trip — including
  **spiking convolutions** (a neuron over the spatial feature map) and full
  SCNNs — via a channels-first↔channels-last neuron-shape bridge. Covered by
  numerical round-trip tests.
- `spyx.nn.sum_pool` ignored `channel_axis` when given a tuple window (it always
  pooled the trailing axes), so channels-last `SumPool` pooled the channel
  dimension. Now respects `channel_axis` for both int and sequence windows.
- SHD prestaging: "Too many open files" and empty-frame issues; a tonic `HSD`
  monkey-patch drops non-finite event timestamps.
- Phasor weights stored as a real/imag pair so Optax converges.

### Migration

Existing Haiku-based training scripts require changes. Start with the
[migration guide](docs/how-to/migrate-haiku-to-nnx.md); the short version is:
instantiate modules directly with `rngs=`, train with
`nnx.Optimizer(model, tx, wrt=nnx.Param)` + `optimizer.update(model, grads)`,
and swap `spyx.loaders` for iterated `spyx.data` loaders.

## [0.1.20] and earlier

Haiku-based releases. See the git history prior to the modernization merge.
