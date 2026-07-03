# Changelog

All notable changes to Spyx are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] — unreleased

The **modernization release**: Spyx moves from DeepMind Haiku to **Flax NNX**
and gains state-space, phasor, and quantization modules. This is a **breaking
release** — see the
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
- **`spyx.nir`** rewritten to walk NNX modules; `to_nir` / `from_nir` take and
  return `nnx.Module` instances.
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
- Diátaxis-structured docs (tutorials / how-to / reference / explanation),
  new SSM & phasor tutorial notebooks, and a `uv` + `ruff` CI workflow.
- `scripts/check_install.py` end-to-end smoke check.

### Fixed

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
